#!/usr/bin/env python3
"""
Standalone multi-threaded data fetching script for EODHD API.

Downloads four datasets per ticker and saves to Parquet:
  1. 1-minute intraday bars  -> data/raw/1min/{TICKER}.parquet
  2. Daily EOD bars           -> data/raw/eod/{TICKER}.parquet
  3. Stock splits history     -> data/raw/splits/{TICKER}.parquet
  4. Dividends history        -> data/raw/dividends/{TICKER}.parquet

For intraday data, automatically applies:
  - Timezone conversion to US/Eastern
  - Market hours filtering (excludes pre-market and after-hours trading)
  - Half-day holiday detection and filtering (e.g., day before Thanksgiving)

EODHD plan assumed: EOD+Intraday All World Extended.

Usage:
    uv run python scripts/fetch_data.py                    # fetch everything (5 workers)
    uv run python scripts/fetch_data.py --force            # re-fetch and overwrite all
    uv run python scripts/fetch_data.py --workers 10       # use 10 concurrent threads
    uv run python scripts/fetch_data.py --no-filter        # disable market hours filtering
"""

import argparse
import datetime as dt
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pandas_market_calendars as pcal
import polars as pl
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Configuration – edit EODHD_API_KEY before running
# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "EODHD_API_KEY": os.environ.get("EODHD_KEY", "your_key_here"),
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "exchange": "US",
    "exchange_calendar": "NYSE",  # pandas_market_calendars exchange name
    "intraday_chunk_days": 120,  # EODHD max per intraday request
    "request_delay": 0.1,  # seconds between API calls
    "base_url": "https://eodhd.com/api",
    "max_workers": 5,  # number of concurrent threads for fetching
    "filter_market_hours": True,  # filter out pre-market and after-hours trading
}

# Thread-safe logging lock
_log_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────
# Logging helper
# ──────────────────────────────────────────────────────────────────────


def _log(msg: str, log_path: Path | None = None) -> None:
    """Print to console and optionally append to log file (thread-safe)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────────────────────────────


def _date_chunks(
    start: str,
    end: str,
    chunk_days: int,
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
# Data processing helpers
# ──────────────────────────────────────────────────────────────────────


def _get_time_diff(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Calculate time differences between consecutive rows to detect gaps.

    Returns:
        Tuple of (time_change_df, time_diff_counts_df)
    """
    time_change = df.with_columns(
        pl.col("time").diff().alias("time_diff"),
        pl.col("date").shift(1).alias("prev_date"),
    ).with_columns(
        pl.when(pl.col("date") == pl.col("prev_date"))
        .then(pl.col("time_diff"))
        .otherwise(pl.lit(None))
        .alias("same_day_time_diff")
    )

    time_diff_counts = (
        time_change.filter(pl.col("same_day_time_diff").is_not_null())
        .group_by("same_day_time_diff")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    time_diff_counts = time_diff_counts.with_columns(
        (pl.col("count") / time_change.shape[0] * 100).round(2).alias("percentage")
    )
    return time_change, time_diff_counts


def _filter_market_hours(
    df: pl.DataFrame,
    start_date: str,
    end_date: str,
    exchange_calendar: str = "NYSE",
) -> pl.DataFrame:
    """
    Filter intraday data to only include regular trading hours.

    Uses pandas_market_calendars to determine valid trading times,
    excluding after-hours and pre-market trading.

    Args:
        df: DataFrame with 'timestamp' column
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        exchange_calendar: Exchange calendar name (default: NYSE)

    Returns:
        Filtered DataFrame with only regular market hours
    """
    # Get exchange calendar and schedule
    calendar = pcal.get_calendar(exchange_calendar)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)

    # Generate minute-by-minute index of valid trading times
    valid_times = pcal.date_range(schedule, frequency="1min", closed="both")

    # Convert to Polars Series and match timezone
    valid_times_series = pl.Series(valid_times).dt.convert_time_zone("US/Eastern")
    valid_times_df = valid_times_series.dt.cast_time_unit("us").to_frame("timestamp")
    valid_times_df = valid_times_df.with_columns(pl.lit(True).alias("is_market_hours"))

    # Join with original data to mark valid times
    df = df.join(valid_times_df, on="timestamp", how="left")
    df = df.with_columns(pl.col("is_market_hours").fill_null(False))

    # Filter to only market hours
    return df.filter(pl.col("is_market_hours")).drop("is_market_hours")


def _filter_half_day_holidays(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect and handle half-day trading holidays (e.g., day before Thanksgiving).

    Half-day holidays are detected by finding days with a ~2h20m gap in trading,
    then filtering those days to only include data up to 1pm.

    Args:
        df: DataFrame with 'date' and 'time' columns

    Returns:
        DataFrame with half-day holidays filtered to 1pm close
    """
    # Add time and date columns if not present
    if "time" not in df.columns:
        df = df.with_columns(pl.col("timestamp").dt.time().alias("time"))
    if "date" not in df.columns:
        df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

    # Detect time gaps to find half-day holidays
    time_change, _ = _get_time_diff(df)

    # Half-day holidays typically show a 2h20m gap (early close at 1pm instead of 4pm)
    holidays = (
        time_change.filter(
            pl.col("same_day_time_diff") == pd.Timedelta(hours=2, minutes=20)
        )
        .select("date")
        .to_series()
        .to_list()
    )

    # For detected half-day holidays, only keep data up to 1pm
    df = df.filter(
        ~pl.col("date").is_in(holidays)
        | (pl.col("date").is_in(holidays) & (pl.col("time") <= dt.time(13, 0)))
    )

    return df


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
        df = df.drop(["timestamp"]).rename({"datetime": "timestamp"})

    # Parse timestamp and convert to US/Eastern timezone
    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("timestamp")
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S", time_zone="UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("timestamp")
        )
    elif df["timestamp"].dtype in (pl.Int64, pl.UInt64, pl.Float64):
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="s")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("US/Eastern")
            .alias("timestamp")
        )

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
    filter_hours: bool = True,
    exchange_calendar: str = "NYSE",
    log_path: Path | None = None,
) -> pl.DataFrame | None:
    """
    Fetch all 1-min bars for *ticker*, concatenating 120-day chunks.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: EODHD API key
        exchange: Exchange code (default: US)
        chunk_days: Days per API request chunk (default: 120)
        delay: Delay between API requests in seconds
        filter_hours: Whether to filter to regular market hours (default: True)
        exchange_calendar: Exchange calendar for market hours (default: NYSE)
        log_path: Optional path for logging

    Returns:
        DataFrame with filtered intraday data or None if no data
    """
    chunks = _date_chunks(start_date, end_date, chunk_days)
    frames: list[pl.DataFrame] = []
    for i, (cs, ce) in enumerate(chunks):
        _log(f"  {ticker} 1min: chunk {i + 1}/{len(chunks)}  {cs} -> {ce}", log_path)
        df = fetch_1min_chunk(ticker, cs, ce, api_key, exchange)
        if df is not None:
            frames.append(df)
        if i < len(chunks) - 1:
            time.sleep(delay)
    if not frames:
        return None

    # Concatenate all chunks
    combined = pl.concat(frames).unique(subset=["timestamp"]).sort("timestamp")

    # Apply data processing filters if requested
    if filter_hours:
        rows_before = len(combined)

        # Filter to regular market hours (exclude pre-market and after-hours)
        combined = _filter_market_hours(
            combined, start_date, end_date, exchange_calendar
        )
        _log(
            f"  {ticker} 1min: market hours filter: {rows_before} -> {len(combined)} rows",
            log_path,
        )

        # Add time and date columns for holiday detection
        combined = combined.with_columns(
            pl.col("timestamp").dt.time().alias("time"),
            pl.col("timestamp").dt.date().alias("date"),
        )

        # Handle half-day holidays
        rows_before = len(combined)
        combined = _filter_half_day_holidays(combined)
        if len(combined) < rows_before:
            _log(
                f"  {ticker} 1min: half-day holiday filter: {rows_before} -> {len(combined)} rows",
                log_path,
            )

        # Remove temporary time and date columns
        combined = combined.drop("time", "date")

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


def process_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    exchange: str,
    exchange_calendar: str,
    chunk_days: int,
    delay: float,
    force: bool,
    filter_hours: bool,
    intraday_dir: Path,
    eod_dir: Path,
    splits_dir: Path,
    div_dir: Path,
    log_path: Path,
) -> str:
    """Process all data types for a single ticker. Returns ticker name when done."""
    # ── EOD daily ──────────────────────────────────────
    eod_file = eod_dir / f"{ticker}.parquet"
    if eod_file.exists() and not force:
        _log(f"  {ticker} eod: SKIP (exists)", log_path)
    else:
        df = fetch_eod_ticker(ticker, start_date, end_date, api_key, exchange)
        _save_if_present(df, eod_file, ticker, "eod", log_path)
        time.sleep(delay)

    # ── Splits ─────────────────────────────────────
    split_file = splits_dir / f"{ticker}.parquet"
    if split_file.exists() and not force:
        _log(f"  {ticker} splits: SKIP (exists)", log_path)
    else:
        df = fetch_splits_ticker(ticker, start_date, end_date, api_key, exchange)
        _save_if_present(df, split_file, ticker, "splits", log_path)
        time.sleep(delay)

    # ── Dividends ──────────────────────────────────
    div_file = div_dir / f"{ticker}.parquet"
    if div_file.exists() and not force:
        _log(f"  {ticker} divs: SKIP (exists)", log_path)
    else:
        df = fetch_dividends_ticker(ticker, start_date, end_date, api_key, exchange)
        _save_if_present(df, div_file, ticker, "divs", log_path)
        time.sleep(delay)

    # ── Intraday 1-min ─────────────────────────────────
    intra_file = intraday_dir / f"{ticker}.parquet"
    if intra_file.exists() and not force:
        _log(f"  {ticker} 1min: SKIP (exists)", log_path)
    else:
        df = fetch_intraday_ticker(
            ticker,
            start_date,
            end_date,
            api_key,
            exchange,
            chunk_days,
            delay,
            filter_hours,
            exchange_calendar,
            log_path,
        )
        _save_if_present(df, intra_file, ticker, "1min", log_path)
        time.sleep(delay)

    return ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EODHD data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and overwrite existing data files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["max_workers"],
        help=f"Number of concurrent worker threads (default: {CONFIG['max_workers']})",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable market hours and holiday filtering for intraday data",
    )
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
    exchange_calendar = CONFIG["exchange_calendar"]
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

    force = args.force
    max_workers = args.workers
    filter_hours = CONFIG["filter_market_hours"] and not args.no_filter

    _log(
        f"Starting fetch: {len(universe)} tickers, {start_date} to {end_date} "
        f"[force={'Y' if force else 'N'}, filter={'Y' if filter_hours else 'N'}, "
        f"workers={max_workers}]",
        log_path,
    )

    # Use ThreadPoolExecutor for concurrent fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all ticker processing tasks
        futures = {
            executor.submit(
                process_ticker,
                ticker,
                start_date,
                end_date,
                api_key,
                exchange,
                exchange_calendar,
                chunk_days,
                delay,
                force,
                filter_hours,
                intraday_dir,
                eod_dir,
                splits_dir,
                div_dir,
                log_path,
            ): ticker
            for ticker in universe
        }

        # Track progress with tqdm as tasks complete
        with tqdm(total=len(universe), desc="Fetching") as pbar:
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                    pbar.update(1)
                except Exception as exc:
                    _log(f"  {ticker} ERROR: {exc}", log_path)
                    pbar.update(1)

    _log("Fetch complete.", log_path)


if __name__ == "__main__":
    main()
