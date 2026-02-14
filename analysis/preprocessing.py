"""
Data preprocessing module.

Loads raw 1-minute Parquet files, cleans data, and resamples to multiple
timeframes. All functions assume raw data exists in data/raw/1min/.

Also loads EOD daily data with adjusted_close from data/raw/eod/ and
applies split adjustments to intraday bars when split history is available.
"""

from pathlib import Path

import polars as pl
from tqdm import tqdm

from utils.config import CONFIG

TIMEFRAMES = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "1hour": "1h",
    "daily": "1d",
}


# ──────────────────────────────────────────────────────────────────────
# Raw data loaders
# ──────────────────────────────────────────────────────────────────────

def load_raw(ticker: str, raw_dir: str | None = None) -> pl.DataFrame:
    """
    Load raw 1-minute data for a single ticker.

    Args:
        ticker:  Stock ticker symbol.
        raw_dir: Directory containing raw 1-min Parquet files.
                 Defaults to CONFIG['raw_dir'].

    Returns:
        Polars DataFrame with columns [timestamp, open, high, low, close, volume].

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
    """
    raw_dir = raw_dir or CONFIG["raw_dir"]
    path = Path(raw_dir) / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data for {ticker}. Expected: {path}\n"
            "Run scripts/fetch_data.py first."
        )
    return pl.read_parquet(path)


def load_raw_eod(ticker: str, eod_dir: str | None = None) -> pl.DataFrame:
    """
    Load raw daily EOD data for a single ticker.

    This data comes directly from the EODHD EOD endpoint and includes
    an ``adjusted_close`` column that accounts for splits and dividends.

    Args:
        ticker:  Stock ticker symbol.
        eod_dir: Directory containing raw EOD Parquet files.
                 Defaults to CONFIG['raw_eod_dir'].

    Returns:
        Polars DataFrame with columns
        [date, open, high, low, close, adjusted_close, volume].

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
    """
    eod_dir = eod_dir or CONFIG["raw_eod_dir"]
    path = Path(eod_dir) / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No EOD data for {ticker}. Expected: {path}\n"
            "Run scripts/fetch_data.py (or --eod-only) first."
        )
    return pl.read_parquet(path)


def load_splits(ticker: str, splits_dir: str | None = None) -> pl.DataFrame | None:
    """
    Load split history for a ticker, or None if no splits file exists.

    Returns DataFrame with at least [date, split] columns.
    """
    splits_dir = splits_dir or CONFIG["splits_dir"]
    path = Path(splits_dir) / f"{ticker}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def filter_market_hours(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    exchange_calendar: str = "NYSE",
) -> pl.DataFrame:
    """
    Filter intraday data to only include regular trading hours.

    Uses pandas_market_calendars to determine valid trading times,
    excluding after-hours, pre-market trading, and non-trading days.

    Args:
        df: DataFrame with 'timestamp' column (timezone-aware US/Eastern)
        start_date: Start date (YYYY-MM-DD). If None, uses min(df.timestamp)
        end_date: End date (YYYY-MM-DD). If None, uses max(df.timestamp)
        exchange_calendar: Exchange calendar name (default: NYSE)

    Returns:
        Filtered DataFrame with only regular market hours
    """
    import pandas_market_calendars as pcal

    # Determine date range
    if start_date is None:
        start_date = df.select(pl.col("timestamp").min()).item().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = df.select(pl.col("timestamp").max()).item().strftime("%Y-%m-%d")

    # Get exchange calendar and schedule
    calendar = pcal.get_calendar(exchange_calendar)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)

    # Generate minute-by-minute valid trading times
    valid_times = pcal.date_range(schedule, frequency="1min", closed="both")

    # Convert to Polars and match timezone
    valid_times_series = pl.Series(valid_times).dt.convert_time_zone("US/Eastern")
    valid_times_df = valid_times_series.dt.cast_time_unit("us").to_frame("timestamp")
    valid_times_df = valid_times_df.with_columns(pl.lit(True).alias("is_market_hours"))

    # Join and filter
    df = df.join(valid_times_df, on="timestamp", how="left")
    df = df.with_columns(pl.col("is_market_hours").fill_null(False))

    return df.filter(pl.col("is_market_hours")).drop("is_market_hours")


# ──────────────────────────────────────────────────────────────────────
# Split adjustment
# ──────────────────────────────────────────────────────────────────────

def _parse_split_ratio(split_str: str) -> float:
    """
    Parse an EODHD split string like '4/1' into a float multiplier.

    '4/1' means each old share becomes 4 new shares, so prices before
    the split should be divided by 4 (multiplied by 1/4) to be
    comparable to post-split prices.

    Returns the forward adjustment factor (new_shares / old_shares).
    """
    parts = split_str.split("/")
    if len(parts) != 2:
        return 1.0
    try:
        return float(parts[0]) / float(parts[1])
    except (ValueError, ZeroDivisionError):
        return 1.0


def adjust_intraday_for_splits(
    df: pl.DataFrame,
    splits_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Adjust intraday OHLCV data for stock splits.

    For each split, all bars *before* the split date have their prices
    divided and volume multiplied by the split ratio.

    Args:
        df:        Intraday DataFrame with [timestamp, open, high, low, close, volume].
        splits_df: Splits DataFrame with [date, split].

    Returns:
        Split-adjusted intraday DataFrame.
    """
    price_cols = ["open", "high", "low", "close"]

    for row in splits_df.iter_rows(named=True):
        split_date = row["date"]
        ratio = _parse_split_ratio(row["split"])
        if ratio == 1.0:
            continue

        # Bars strictly before the split date
        mask = pl.col("timestamp").dt.date() < split_date
        df = df.with_columns(
            [
                pl.when(mask).then(pl.col(c) / ratio).otherwise(pl.col(c)).alias(c)
                for c in price_cols
            ]
            + [
                pl.when(mask)
                .then((pl.col("volume").cast(pl.Float64) * ratio).cast(pl.Int64))
                .otherwise(pl.col("volume"))
                .alias("volume")
            ]
        )

    return df


# ──────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────

def clean(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean a 1-minute DataFrame.

    Steps:
        1. Remove duplicate timestamps.
        2. Sort by timestamp.
        3. Filter to regular market hours (09:30-16:00 ET).
        4. Drop rows with null OHLC values.
        5. Drop rows where close <= 0.

    Args:
        df: Raw Polars DataFrame.

    Returns:
        Cleaned Polars DataFrame.
    """
    df = df.unique(subset=["timestamp"]).sort("timestamp")

    # Filter to regular market hours using exchange calendar
    df = filter_market_hours(df)

    # Drop nulls and invalid prices
    df = df.drop_nulls(subset=["open", "high", "low", "close"])
    df = df.filter(pl.col("close") > 0)

    return df


# ──────────────────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────────────────

def resample(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Resample cleaned 1-minute data to a higher timeframe.

    Args:
        df:        Cleaned 1-minute Polars DataFrame.
        timeframe: One of '1min', '5min', '15min', '1hour', 'daily'.

    Returns:
        Resampled Polars DataFrame with OHLCV columns.
    """
    if timeframe == "1min":
        return df

    # For daily, group by date
    if timeframe == "daily":
        resampled = (
            df.with_columns(pl.col("timestamp").dt.date().alias("date"))
            .group_by("date")
            .agg(
                pl.col("timestamp").first().alias("timestamp"),
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            )
            .sort("date")
            .drop("date")
        )
        return resampled

    interval = TIMEFRAMES[timeframe]

    # For intraday resampling, use group_by_dynamic
    resampled = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=interval)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
    )

    # Drop bars where aggregation produced nulls (incomplete bars)
    resampled = resampled.drop_nulls(subset=["open", "high", "low", "close"])

    return resampled


# ──────────────────────────────────────────────────────────────────────
# EOD daily processing (uses EODHD adjusted_close directly)
# ──────────────────────────────────────────────────────────────────────

def process_eod_daily(
    ticker: str,
    eod_dir: str | None = None,
    processed_dir: str | None = None,
) -> int:
    """
    Process EOD daily data: use EODHD adjusted_close as the canonical
    daily close price (already accounts for splits + dividends).

    Saves to data/processed/daily/{TICKER}.parquet with the standard
    schema [timestamp, open, high, low, close, volume].

    Args:
        ticker:        Stock ticker symbol.
        eod_dir:       Raw EOD data directory.
        processed_dir: Processed data base directory.

    Returns:
        Number of daily bars saved.
    """
    eod_dir = eod_dir or CONFIG["raw_eod_dir"]
    processed_dir = processed_dir or CONFIG["processed_dir"]

    df = load_raw_eod(ticker, eod_dir)

    # Compute adjustment ratio from adjusted_close / close
    if "adjusted_close" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("close") != 0)
            .then(pl.col("adjusted_close") / pl.col("close"))
            .otherwise(1.0)
            .alias("_adj_factor")
        )
        for c in ["open", "high", "low", "close"]:
            df = df.with_columns((pl.col(c) * pl.col("_adj_factor")).alias(c))
        df = df.drop(["adjusted_close", "_adj_factor"])

    # Rename date -> timestamp for consistency
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp")
        ).drop("date")

    df = df.drop_nulls(subset=["open", "high", "low", "close"])
    df = df.filter(pl.col("close") > 0).sort("timestamp")

    out_dir = Path(processed_dir) / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / f"{ticker}.parquet")

    return len(df)


# ──────────────────────────────────────────────────────────────────────
# Top-level preprocessing functions
# ──────────────────────────────────────────────────────────────────────

def preprocess_ticker(
    ticker: str,
    raw_dir: str | None = None,
    processed_dir: str | None = None,
    splits_dir: str | None = None,
) -> dict[str, int]:
    """
    Load, clean, optionally split-adjust, and resample intraday data for
    a single ticker.  Also processes EOD daily data when available.

    Saves resampled data to data/processed/{timeframe}/{TICKER}.parquet.

    Args:
        ticker:        Stock ticker symbol.
        raw_dir:       Raw 1-min data directory.
        processed_dir: Processed data base directory.
        splits_dir:    Directory with split history Parquets.

    Returns:
        Dict mapping timeframe -> number of bars saved.
    """
    raw_dir = raw_dir or CONFIG["raw_dir"]
    processed_dir = processed_dir or CONFIG["processed_dir"]
    splits_dir = splits_dir or CONFIG["splits_dir"]

    counts: dict[str, int] = {}

    # ── EOD daily (from EODHD adjusted_close) ──────────────
    eod_dir = CONFIG.get("raw_eod_dir")
    if eod_dir and (Path(eod_dir) / f"{ticker}.parquet").exists():
        counts["daily"] = process_eod_daily(ticker, eod_dir, processed_dir)

    # ── Intraday timeframes ────────────────────────────────
    raw_path = Path(raw_dir) / f"{ticker}.parquet"
    if raw_path.exists():
        df_raw = load_raw(ticker, raw_dir)

        # Apply split adjustment if split history available
        splits_df = load_splits(ticker, splits_dir)
        if splits_df is not None and len(splits_df) > 0:
            df_raw = adjust_intraday_for_splits(df_raw, splits_df)

        df_clean = clean(df_raw)

        for tf in TIMEFRAMES:
            if tf == "daily" and "daily" in counts:
                # Already handled by EOD data above
                continue
            df_resampled = resample(df_clean, tf)
            out_dir = Path(processed_dir) / tf
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{ticker}.parquet"
            df_resampled.write_parquet(out_path)
            counts[tf] = len(df_resampled)

    return counts


def preprocess_all_tickers(
    tickers: list[str] | None = None,
    raw_dir: str | None = None,
    processed_dir: str | None = None,
) -> pl.DataFrame:
    """
    Preprocess all tickers in the universe.

    Args:
        tickers:       List of tickers. Defaults to all tickers with raw data.
        raw_dir:       Raw 1-min data directory.
        processed_dir: Processed data base directory.

    Returns:
        Polars DataFrame summarizing bars per timeframe per ticker.
    """
    raw_dir = raw_dir or CONFIG["raw_dir"]
    processed_dir = processed_dir or CONFIG["processed_dir"]
    eod_dir = CONFIG.get("raw_eod_dir", "")

    if tickers is None:
        # Collect tickers that have *either* intraday or EOD data
        found: set[str] = set()
        raw_path = Path(raw_dir)
        if raw_path.exists():
            found.update(p.stem for p in raw_path.glob("*.parquet"))
        eod_path = Path(eod_dir)
        if eod_path.exists():
            found.update(p.stem for p in eod_path.glob("*.parquet"))
        tickers = sorted(found)

    if not tickers:
        print("No tickers found to preprocess.")
        return pl.DataFrame()

    rows: list[dict] = []
    for ticker in tqdm(tickers, desc="Preprocessing"):
        try:
            counts = preprocess_ticker(ticker, raw_dir, processed_dir)
            row = {"ticker": ticker, **counts}
            rows.append(row)
        except Exception as exc:
            print(f"  {ticker}: ERROR - {exc}")

    return pl.DataFrame(rows)


def load_processed(
    ticker: str,
    timeframe: str = "daily",
    processed_dir: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Load preprocessed data for a ticker at a given timeframe.

    Args:
        ticker:        Stock ticker symbol.
        timeframe:     One of '1min', '5min', '15min', '1hour', 'daily'.
        processed_dir: Processed data base directory.
        start_date:    Optional start date filter (YYYY-MM-DD or datetime).
                       If None, loads from beginning.
        end_date:      Optional end date filter (YYYY-MM-DD or datetime).
                       If None, loads to end.

    Returns:
        Polars DataFrame with OHLCV data and technical indicators.

    Raises:
        FileNotFoundError: If the processed file does not exist.
    """
    processed_dir = processed_dir or CONFIG["processed_dir"]
    path = Path(processed_dir) / timeframe / f"{ticker}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"No processed data for {ticker} at {timeframe}. "
            f"Expected: {path}\nRun preprocess_all_tickers() first."
        )

    # Use scan_parquet for lazy loading with predicate pushdown
    lazy_df = pl.scan_parquet(path)

    # Apply date filters if specified (pushed down to Parquet reader)
    if start_date is not None:
        lazy_df = lazy_df.filter(
            pl.col("timestamp") >= pl.lit(start_date).str.to_datetime()
        )
    if end_date is not None:
        lazy_df = lazy_df.filter(
            pl.col("timestamp") <= pl.lit(end_date).str.to_datetime()
        )

    # Collect the result
    return lazy_df.collect()
