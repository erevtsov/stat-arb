"""
Data preprocessing module.

Loads raw 1-minute Parquet files, cleans data, and resamples to multiple
timeframes. All functions assume raw data exists in data/raw/1min/.
"""

from pathlib import Path

import polars as pl
from tqdm import tqdm

from utils.config import CONFIG

# US equity regular market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

TIMEFRAMES = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "1hour": "1h",
    "daily": "1d",
}


def load_raw(ticker: str, raw_dir: str | None = None) -> pl.DataFrame:
    """
    Load raw 1-minute data for a single ticker.

    Args:
        ticker:  Stock ticker symbol.
        raw_dir: Directory containing raw Parquet files.
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


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean a 1-minute DataFrame.

    Steps:
        1. Remove duplicate timestamps.
        2. Sort by timestamp.
        3. Filter to regular market hours (09:30–16:00 ET).
        4. Drop rows with null OHLC values.
        5. Drop rows where close <= 0.

    Args:
        df: Raw Polars DataFrame.

    Returns:
        Cleaned Polars DataFrame.
    """
    df = df.unique(subset=["timestamp"]).sort("timestamp")

    # Filter market hours: 09:30 <= time < 16:00
    df = df.filter(
        (
            (pl.col("timestamp").dt.hour() > MARKET_OPEN_HOUR)
            | (
                (pl.col("timestamp").dt.hour() == MARKET_OPEN_HOUR)
                & (pl.col("timestamp").dt.minute() >= MARKET_OPEN_MINUTE)
            )
        )
        & (
            (pl.col("timestamp").dt.hour() < MARKET_CLOSE_HOUR)
            | (
                (pl.col("timestamp").dt.hour() == MARKET_CLOSE_HOUR)
                & (pl.col("timestamp").dt.minute() == 0)
            )
        )
    )

    # Drop nulls and invalid prices
    df = df.drop_nulls(subset=["open", "high", "low", "close"])
    df = df.filter(pl.col("close") > 0)

    return df


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

    interval = TIMEFRAMES[timeframe]

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


def preprocess_ticker(
    ticker: str,
    raw_dir: str | None = None,
    processed_dir: str | None = None,
) -> dict[str, int]:
    """
    Load, clean, and resample data for a single ticker.

    Saves resampled data to data/processed/{timeframe}/{TICKER}.parquet.

    Args:
        ticker:        Stock ticker symbol.
        raw_dir:       Raw data directory.
        processed_dir: Processed data base directory.

    Returns:
        Dict mapping timeframe -> number of bars saved.
    """
    raw_dir = raw_dir or CONFIG["raw_dir"]
    processed_dir = processed_dir or CONFIG["processed_dir"]

    df_raw = load_raw(ticker, raw_dir)
    df_clean = clean(df_raw)

    counts: dict[str, int] = {}
    for tf in TIMEFRAMES:
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
        tickers:       List of tickers. Defaults to all tickers in raw_dir.
        raw_dir:       Raw data directory.
        processed_dir: Processed data base directory.

    Returns:
        Polars DataFrame summarizing bars per timeframe per ticker.
    """
    raw_dir = raw_dir or CONFIG["raw_dir"]
    processed_dir = processed_dir or CONFIG["processed_dir"]

    if tickers is None:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {raw_path}\n"
                "Run scripts/fetch_data.py first."
            )
        tickers = sorted(
            p.stem for p in raw_path.glob("*.parquet")
        )

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
) -> pl.DataFrame:
    """
    Load preprocessed data for a ticker at a given timeframe.

    Args:
        ticker:        Stock ticker symbol.
        timeframe:     One of '1min', '5min', '15min', '1hour', 'daily'.
        processed_dir: Processed data base directory.

    Returns:
        Polars DataFrame.

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
    return pl.read_parquet(path)
