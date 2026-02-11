#!/usr/bin/env python3
"""
Standalone data fetching script for EODHD API.

Downloads four datasets per ticker and saves to Parquet:
  1. 1-minute intraday bars  -> data/raw/1min/{TICKER}.parquet
  2. Daily EOD bars           -> data/raw/eod/{TICKER}.parquet
  3. Stock splits history     -> data/raw/splits/{TICKER}.parquet
  4. Dividends history        -> data/raw/dividends/{TICKER}.parquet

EODHD plan assumed: EOD+Intraday All World Extended.

Usage:
    uv run python scripts/fetch_data.py              # fetch everything
    uv run python scripts/fetch_data.py --eod-only   # daily + splits + divs only
    uv run python scripts/fetch_data.py --intraday-only
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Configuration – edit EODHD_API_KEY before running
# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "EODHD_API_KEY": "your_key_here",  # <-- replace with your key
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "exchange": "US",
    "intraday_chunk_days": 120,   # EODHD max per intraday request
    "request_delay": 0.35,        # seconds between API calls
    "base_url": "https://eodhd.com/api",
}


# ──────────────────────────────────────────────────────────────────────
# Logging helper
# ──────────────────────────────────────────────────────────────────────

def _log(msg: str, log_path: Path | None = None) -> None:
    """Print to console and optionally append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────────────────────────────

def _date_chunks(
    start: str, end: str, chunk_days: int,
) -> list[tuple[str, str]]:
    """Split a date range into consecutive chunks of at most *chunk_days*."""
    fmt = "%Y-%m-%d"
    s = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    chunks: list[tuple[str, str]] = []
    while s <= e:
        chunk_end = min(s + timedelta(days=chunk_days - 1), e)
        chunks.append((s.strftime(fmt), chunk_end.strftime(fmt)))
        s = chunk_end + timedelta(days=1)
    return chunks


# ──────────────────────────────────────────────────────────────────────
# Generic request wrapper with retry
# ──────────────────────────────────────────────────────────────────────

def _api_get(url: str, timeout: int = 30, retries: int = 3) -> requests.Response | None:
    """GET with exponential-backoff retries on transient failures."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None


# ──────────────────────────────────────────────────────────────────────
# 1. Intraday 1-min bars
# ──────────────────────────────────────────────────────────────────────

def fetch_1min_chunk(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    exchange: str = "US",
) -> pl.DataFrame | None:
    """
    Fetch a single <=120-day chunk of 1-minute bars.

    Returns Polars DataFrame [timestamp, open, high, low, close, volume]
    or None on failure / empty response.
    """
    base = CONFIG["base_url"]
    from_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    to_ts = int((datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).timestamp())
    url = (
        f"{base}/intraday/{ticker}.{exchange}"
        f"?api_token={api_key}&interval=1m"
        f"&from={from_ts}&to={to_ts}&fmt=json"
    )

    resp = _api_get(url)
    if resp is None:
        return None

    data = resp.json()
    if not data or not isinstance(data, list):
        return None

    df = pl.DataFrame(data)

    # Normalise timestamp column name
    if "datetime" in df.columns:
        df = df.rename({"datetime": "timestamp"})

    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.to_datetime().alias("timestamp"))
    elif df["timestamp"].dtype in (pl.Int64, pl.UInt64, pl.Float64):
        df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp"))

    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.select([c for c in keep if c in df.columns])
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))

    return df if len(df) > 0 else None


def fetch_intraday_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    exchange: str = "US",
    chunk_days: int = 120,
    delay: float = 0.35,
    log_path: Path | None = None,
) -> pl.DataFrame | None:
    """Fetch all 1-min bars for *ticker*, concatenating 120-day chunks."""
    chunks = _date_chunks(start_date, end_date, chunk_days)
    frames: list[pl.DataFrame] = []
    for i, (cs, ce) in enumerate(chunks):
        _log(f"  {ticker} 1min: chunk {i+1}/{len(chunks)}  {cs} -> {ce}", log_path)
        df = fetch_1min_chunk(ticker, cs, ce, api_key, exchange)
        if df is not None:
            frames.append(df)
        if i < len(chunks) - 1:
            time.sleep(delay)
    if not frames:
        return None
    combined = pl.concat(frames).unique(subset=["timestamp"]).sort("timestamp")
    return combined


# ──────────────────────────────────────────────────────────────────────
# 2. Daily EOD bars
# ──────────────────────────────────────────────────────────────────────

def fetch_eod_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    exchange: str = "US",
) -> pl.DataFrame | None:
    """
    Fetch daily EOD bars for a single ticker.

    Returns [date, open, high, low, close, adjusted_close, volume].
    """
    base = CONFIG["base_url"]
    url = (
        f"{base}/eod/{ticker}.{exchange}"
        f"?api_token={api_key}&fmt=json"
        f"&from={start_date}&to={end_date}"
    )

    resp = _api_get(url)
    if resp is None:
        return None

    data = resp.json()
    if not data or not isinstance(data, list):
        return None

    df = pl.DataFrame(data)

    if "date" in df.columns:
        df = df.with_columns(pl.col("date").str.to_date().alias("date"))

    keep = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df.select([c for c in keep if c in df.columns])

    for c in ["open", "high", "low", "close", "adjusted_close"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))

    return df if len(df) > 0 else None


# ──────────────────────────────────────────────────────────────────────
# 3. Splits
# ──────────────────────────────────────────────────────────────────────

def fetch_splits_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    exchange: str = "US",
) -> pl.DataFrame | None:
    """
    Fetch stock split history for a single ticker.

    Returns [date, split] where split is a string like '4/1'.
    """
    base = CONFIG["base_url"]
    url = (
        f"{base}/splits/{ticker}.{exchange}"
        f"?api_token={api_key}&fmt=json"
        f"&from={start_date}&to={end_date}"
    )
    resp = _api_get(url)
    if resp is None:
        return None

    data = resp.json()
    if not data or not isinstance(data, list):
        return None

    df = pl.DataFrame(data)
    if len(df) == 0:
        return None

    if "date" in df.columns:
        df = df.with_columns(pl.col("date").str.to_date().alias("date"))

    return df


# ──────────────────────────────────────────────────────────────────────
# 4. Dividends
# ──────────────────────────────────────────────────────────────────────

def fetch_dividends_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    exchange: str = "US",
) -> pl.DataFrame | None:
    """
    Fetch dividend history for a single ticker.

    Returns at minimum [date, value] (dividend amount per share).
    """
    base = CONFIG["base_url"]
    url = (
        f"{base}/div/{ticker}.{exchange}"
        f"?api_token={api_key}&fmt=json"
        f"&from={start_date}&to={end_date}"
    )
    resp = _api_get(url)
    if resp is None:
        return None

    data = resp.json()
    if not data or not isinstance(data, list):
        return None

    df = pl.DataFrame(data)
    if len(df) == 0:
        return None

    if "date" in df.columns:
        df = df.with_columns(pl.col("date").str.to_date().alias("date"))

    return df


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def _save_if_present(
    df: pl.DataFrame | None,
    out_path: Path,
    ticker: str,
    label: str,
    log_path: Path | None,
) -> None:
    if df is None:
        _log(f"  {ticker} {label}: no data", log_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    _log(f"  {ticker} {label}: {len(df)} rows -> {out_path}", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EODHD data")
    parser.add_argument("--eod-only", action="store_true",
                        help="Skip intraday, fetch only EOD + splits + dividends")
    parser.add_argument("--intraday-only", action="store_true",
                        help="Skip EOD/splits/dividends, fetch only 1-min bars")
    args = parser.parse_args()

    api_key = CONFIG["EODHD_API_KEY"]
    if api_key == "your_key_here":
        print("ERROR: Set your EODHD_API_KEY in CONFIG before running.")
        sys.exit(1)

    # Import universe from central config
    try:
        from utils.config import get_all_tickers
        universe = get_all_tickers()
    except ImportError:
        print("ERROR: Could not import universe from utils.config.")
        print("Make sure you run from the project root directory.")
        sys.exit(1)

    start_date = CONFIG["start_date"]
    end_date = CONFIG["end_date"]
    exchange = CONFIG["exchange"]
    chunk_days = CONFIG["intraday_chunk_days"]
    delay = CONFIG["request_delay"]

    # Directories
    data_root = Path("data")
    intraday_dir = data_root / "raw" / "1min"
    eod_dir = data_root / "raw" / "eod"
    splits_dir = data_root / "raw" / "splits"
    div_dir = data_root / "raw" / "dividends"

    for d in [intraday_dir, eod_dir, splits_dir, div_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = data_root / "fetch_log.txt"

    do_intraday = not args.eod_only
    do_eod = not args.intraday_only

    _log(
        f"Starting fetch: {len(universe)} tickers, {start_date} to {end_date} "
        f"[intraday={'Y' if do_intraday else 'N'}, eod={'Y' if do_eod else 'N'}]",
        log_path,
    )

    for ticker in tqdm(universe, desc="Fetching"):
        # ── EOD daily ──────────────────────────────────────
        if do_eod:
            eod_file = eod_dir / f"{ticker}.parquet"
            if eod_file.exists():
                _log(f"  {ticker} eod: SKIP (exists)", log_path)
            else:
                df = fetch_eod_ticker(ticker, start_date, end_date, api_key, exchange)
                _save_if_present(df, eod_file, ticker, "eod", log_path)
                time.sleep(delay)

            # ── Splits ─────────────────────────────────────
            split_file = splits_dir / f"{ticker}.parquet"
            if split_file.exists():
                _log(f"  {ticker} splits: SKIP (exists)", log_path)
            else:
                df = fetch_splits_ticker(ticker, start_date, end_date, api_key, exchange)
                _save_if_present(df, split_file, ticker, "splits", log_path)
                time.sleep(delay)

            # ── Dividends ──────────────────────────────────
            div_file = div_dir / f"{ticker}.parquet"
            if div_file.exists():
                _log(f"  {ticker} divs: SKIP (exists)", log_path)
            else:
                df = fetch_dividends_ticker(ticker, start_date, end_date, api_key, exchange)
                _save_if_present(df, div_file, ticker, "divs", log_path)
                time.sleep(delay)

        # ── Intraday 1-min ─────────────────────────────────
        if do_intraday:
            intra_file = intraday_dir / f"{ticker}.parquet"
            if intra_file.exists():
                _log(f"  {ticker} 1min: SKIP (exists)", log_path)
            else:
                df = fetch_intraday_ticker(
                    ticker, start_date, end_date, api_key,
                    exchange, chunk_days, delay, log_path,
                )
                _save_if_present(df, intra_file, ticker, "1min", log_path)
                time.sleep(delay)

    _log("Fetch complete.", log_path)


if __name__ == "__main__":
    main()
