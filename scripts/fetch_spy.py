#!/usr/bin/env python3
"""
Fetch daily adjusted SPY prices from EODHD and save to data/benchmark/spy_daily.parquet.

Kept separate from the universe fetch (fetch_data.py) — SPY is used only as a
benchmark / beta reference, not as a tradeable asset in the strategy.

Usage:
    uv run python scripts/fetch_spy.py
    uv run python scripts/fetch_spy.py --start 2017-01-01 --end 2026-02-28
    uv run python scripts/fetch_spy.py --force   # overwrite existing file
"""

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl
import requests

from utils.config import CONFIG

OUT_DIR = Path("data/benchmark")
OUT_FILE = OUT_DIR / "spy_daily.parquet"

START_DEFAULT = "2017-01-01"
END_DEFAULT   = "2026-02-28"


def _fetch_spy_eod(start: str, end: str, api_key: str) -> pl.DataFrame:
    url = (
        f"{CONFIG.api.base_url}/eod/SPY.US"
        f"?api_token={api_key}&fmt=json"
        f"&from={start}&to={end}"
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch SPY after 3 attempts: {e}") from e
            time.sleep(2 ** attempt)

    if not data or not isinstance(data, list):
        raise RuntimeError(f"Empty or unexpected response: {data!r}")

    df = pl.DataFrame(data)
    df = df.with_columns(pl.col("date").str.to_date())

    keep = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df.select([c for c in keep if c in df.columns])

    for col in ["open", "high", "low", "close", "adjusted_close"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))

    return df.sort("date")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily SPY prices from EODHD")
    parser.add_argument("--start", default=START_DEFAULT)
    parser.add_argument("--end",   default=END_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    api_key = os.environ.get("EODHD_KEY", CONFIG.api.api_key)
    if api_key == "your_key_here":
        print("ERROR: Set EODHD_KEY environment variable before running.")
        sys.exit(1)

    if OUT_FILE.exists() and not args.force:
        existing = pl.read_parquet(OUT_FILE)
        print(f"Found existing {OUT_FILE} ({len(existing):,} rows, "
              f"{existing['date'].min()} → {existing['date'].max()}). "
              f"Use --force to overwrite.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching SPY daily EOD: {args.start} → {args.end} ...")
    df = _fetch_spy_eod(args.start, args.end, api_key)
    df.write_parquet(OUT_FILE)
    print(f"Saved {len(df):,} rows to {OUT_FILE}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()
