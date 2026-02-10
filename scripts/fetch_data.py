#!/usr/bin/env python3
"""
Standalone data fetching script for 1-minute bar data from EODHD API.

Downloads intraday data in 120-day chunks (EODHD API limit),
concatenates all chunks per ticker, and saves to Parquet.

Usage:
    python scripts/fetch_data.py
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration – edit before running
# ──────────────────────────────────────────────
CONFIG = {
    "EODHD_API_KEY": "your_key_here",  # <-- replace with your key
    "universe": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM",
        "ADBE", "ORCL",
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
        "JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "LLY",
        "AMZN", "WMT", "HD", "COST", "NKE", "MCD", "SBUX",
        "XOM", "CVX", "COP", "SLB", "EOG",
    ],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "output_dir": "data/raw/1min/",
    "exchange": "US",           # EODHD exchange suffix
    "interval": "1m",           # 1-minute bars
    "chunk_days": 120,          # EODHD max per request
    "request_delay": 0.35,      # seconds between API calls
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _log(msg: str, log_path: Path | None = None) -> None:
    """Print to console and optionally append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def _date_chunks(
    start: str, end: str, chunk_days: int
) -> list[tuple[str, str]]:
    """
    Split a date range into consecutive chunks of at most *chunk_days* days.

    Returns list of (chunk_start, chunk_end) as ISO-format strings.
    """
    fmt = "%Y-%m-%d"
    s = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    chunks: list[tuple[str, str]] = []
    while s <= e:
        chunk_end = min(s + timedelta(days=chunk_days - 1), e)
        chunks.append((s.strftime(fmt), chunk_end.strftime(fmt)))
        s = chunk_end + timedelta(days=1)
    return chunks


def fetch_1min_chunk(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    exchange: str = "US",
) -> pl.DataFrame | None:
    """
    Fetch a single chunk of 1-minute bars from EODHD.

    Args:
        ticker:   Stock ticker symbol (e.g. 'AAPL').
        start:    Start date 'YYYY-MM-DD'.
        end:      End date 'YYYY-MM-DD'.
        api_key:  EODHD API key.
        exchange: Exchange code (default 'US').

    Returns:
        Polars DataFrame with columns [timestamp, open, high, low, close, volume],
        or None if the request failed or returned no data.
    """
    url = (
        f"https://eodhd.com/api/intraday/{ticker}.{exchange}"
        f"?api_token={api_key}"
        f"&interval=1m"
        f"&from={int(datetime.strptime(start, '%Y-%m-%d').timestamp())}"
        f"&to={int((datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).timestamp())}"
        f"&fmt=json"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return None

    data = resp.json()
    if not data or not isinstance(data, list):
        return None

    df = pl.DataFrame(data)

    # EODHD returns 'datetime' (or 'timestamp') as string/epoch
    if "datetime" in df.columns:
        df = df.rename({"datetime": "timestamp"})

    # Ensure timestamp is parsed correctly
    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime().alias("timestamp")
        )
    elif df["timestamp"].dtype in (pl.Int64, pl.UInt64, pl.Float64):
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp")
        )

    # Keep only the columns we need and ensure correct types
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.select([c for c in keep_cols if c in df.columns])

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))

    return df if len(df) > 0 else None


def fetch_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    exchange: str = "US",
    chunk_days: int = 120,
    delay: float = 0.35,
    log_path: Path | None = None,
) -> pl.DataFrame | None:
    """
    Fetch all 1-minute bars for a ticker, handling 120-day chunking.

    Args:
        ticker:     Stock ticker.
        start_date: Overall start date.
        end_date:   Overall end date.
        api_key:    EODHD API key.
        exchange:   Exchange code.
        chunk_days: Max days per request.
        delay:      Seconds to wait between requests.
        log_path:   Optional log file path.

    Returns:
        Concatenated Polars DataFrame, or None if no data retrieved.
    """
    chunks = _date_chunks(start_date, end_date, chunk_days)
    frames: list[pl.DataFrame] = []

    for i, (cs, ce) in enumerate(chunks):
        _log(
            f"  {ticker}: chunk {i + 1}/{len(chunks)}  {cs} -> {ce}",
            log_path,
        )
        df = fetch_1min_chunk(ticker, cs, ce, api_key, exchange)
        if df is not None:
            frames.append(df)
        if i < len(chunks) - 1:
            time.sleep(delay)

    if not frames:
        return None

    combined = pl.concat(frames)
    combined = combined.unique(subset=["timestamp"]).sort("timestamp")
    return combined


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    api_key = CONFIG["EODHD_API_KEY"]
    if api_key == "your_key_here":
        print("ERROR: Set your EODHD_API_KEY in CONFIG before running.")
        sys.exit(1)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(CONFIG["output_dir"]).parent.parent  # data/
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fetch_log.txt"

    universe = CONFIG["universe"]
    start_date = CONFIG["start_date"]
    end_date = CONFIG["end_date"]
    exchange = CONFIG["exchange"]
    chunk_days = CONFIG["chunk_days"]
    delay = CONFIG["request_delay"]

    _log(
        f"Starting fetch: {len(universe)} tickers, "
        f"{start_date} to {end_date}",
        log_path,
    )

    for ticker in tqdm(universe, desc="Fetching tickers"):
        out_file = output_dir / f"{ticker}.parquet"

        # Resume capability: skip if already fetched
        if out_file.exists():
            _log(f"  {ticker}: SKIP (file exists)", log_path)
            continue

        df = fetch_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            exchange=exchange,
            chunk_days=chunk_days,
            delay=delay,
            log_path=log_path,
        )

        if df is None:
            _log(f"  {ticker}: NO DATA returned", log_path)
            continue

        df.write_parquet(out_file)
        _log(
            f"  {ticker}: saved {len(df)} bars -> {out_file}",
            log_path,
        )
        time.sleep(delay)

    _log("Fetch complete.", log_path)


if __name__ == "__main__":
    main()
