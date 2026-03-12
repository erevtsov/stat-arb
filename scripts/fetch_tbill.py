#!/usr/bin/env python3
"""
Fetch 3-month T-bill daily yield data from Yahoo Finance (^IRX) and save to
data/benchmark/tbill_3m_daily.parquet.

^IRX is the CBOE 13-Week Treasury Bill index — annualised yield in percent
(e.g. 5.25 means 5.25% per year). No API key required.

Kept separate from the universe fetch — T-bill data is used only as the
risk-free benchmark for Sharpe calculation and market-neutrality reporting.

Usage:
    uv run python scripts/fetch_tbill.py
    uv run python scripts/fetch_tbill.py --start 2017-01-01 --end 2026-02-28
    uv run python scripts/fetch_tbill.py --force   # overwrite existing file
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import yfinance as yf

OUT_DIR  = Path("data/benchmark")
OUT_FILE = OUT_DIR / "tbill_3m_daily.parquet"

START_DEFAULT = "2017-01-01"
END_DEFAULT   = "2026-02-28"

TICKER = "^IRX"  # CBOE 13-Week Treasury Bill index, annualised yield in %


def _fetch(start: str, end: str) -> pl.DataFrame:
    raw = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKER}")

    # Flatten multi-level columns, then normalize to lowercase after reset_index
    raw.columns = [col[0].lower() if isinstance(col, tuple) else col.lower()
                   for col in raw.columns]
    raw = raw.reset_index()
    raw.columns = [str(c).lower() for c in raw.columns]  # 'Date' → 'date'

    df = pl.from_pandas(raw[["date", "close"]].rename(columns={"close": "yield_pct"}))
    df = df.with_columns(pl.col("date").cast(pl.Date)).sort("date")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 3-month T-bill yield from Yahoo Finance")
    parser.add_argument("--start", default=START_DEFAULT)
    parser.add_argument("--end",   default=END_DEFAULT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.force:
        existing = pl.read_parquet(OUT_FILE)
        print(f"Found existing {OUT_FILE} ({len(existing):,} rows, "
              f"{existing['date'].min()} → {existing['date'].max()}). "
              f"Use --force to overwrite.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {TICKER} (3-month T-bill yield): {args.start} → {args.end} ...")
    df = _fetch(args.start, args.end)
    df.write_parquet(OUT_FILE)

    max_yield = df["yield_pct"].max()
    print(f"Saved {len(df):,} rows to {OUT_FILE}")
    print(f"Date range:  {df['date'].min()} → {df['date'].max()}")
    print(f"Yield range: {df['yield_pct'].min():.3f}% – {max_yield:.3f}%")
    print(f"Max yield (conservative Sharpe hurdle): {max_yield:.3f}% p.a.")


if __name__ == "__main__":
    main()
