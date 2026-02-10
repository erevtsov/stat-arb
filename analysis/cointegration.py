"""
Cointegration analysis module.

Finds cointegrated pairs using the Engle-Granger two-step method with
ADF test on residuals. Tests all pairwise combinations within the same sector.
"""

import itertools
import math
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools import add_constant
from tqdm import tqdm

from analysis.preprocessing import load_processed
from utils.config import CONFIG, get_all_sectors, get_tickers_by_sector


def _compute_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute hedge ratio via OLS: y = alpha + beta * x + eps.

    Args:
        y: Dependent variable (price series A).
        x: Independent variable (price series B).

    Returns:
        Hedge ratio (beta coefficient).
    """
    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    return float(model.params[1])


def _compute_half_life(spread: np.ndarray) -> float:
    """
    Compute half-life of mean reversion from an AR(1) model.

    half_life = -log(2) / log(phi)

    where phi is the AR(1) coefficient from:
        spread_t - spread_{t-1} = alpha + phi * spread_{t-1} + eps

    Args:
        spread: Spread time series as numpy array.

    Returns:
        Half-life in periods, or inf if not mean-reverting.
    """
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    spread_lag_const = add_constant(spread_lag)
    model = OLS(spread_diff, spread_lag_const).fit()
    phi = model.params[1]

    if phi >= 0:
        return float("inf")  # not mean-reverting

    half_life = -math.log(2) / math.log(1 + phi)
    return max(half_life, 0.0)


def test_cointegration(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
) -> dict:
    """
    Run Engle-Granger cointegration test on two price series.

    Args:
        prices_a: Close price series for stock A.
        prices_b: Close price series for stock B.

    Returns:
        Dict with keys: hedge_ratio, adf_stat, p_value, half_life.
    """
    hedge_ratio = _compute_hedge_ratio(prices_a, prices_b)
    spread = prices_a - hedge_ratio * prices_b

    adf_result = adfuller(spread, maxlag=1, regression="c", autolag=None)
    adf_stat = float(adf_result[0])
    p_value = float(adf_result[1])

    half_life = _compute_half_life(spread)

    return {
        "hedge_ratio": hedge_ratio,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "half_life": half_life,
    }


def find_cointegrated_pairs(
    tickers: list[str] | None = None,
    processed_dir: str | None = None,
    p_value_threshold: float | None = None,
    min_half_life: float | None = None,
    max_half_life: float | None = None,
) -> pl.DataFrame:
    """
    Find all cointegrated pairs within each sector.

    Tests all pairwise combinations of tickers within the same sector
    using daily close prices. Only pairs with complete overlapping data
    are tested.

    Args:
        tickers:            List of tickers to consider. Defaults to all in config.
        processed_dir:      Directory with processed daily data.
        p_value_threshold:  Max p-value to include pair. Defaults to config value.
        min_half_life:      Min half-life filter. Defaults to config value.
        max_half_life:      Max half-life filter. Defaults to config value.

    Returns:
        Polars DataFrame with columns:
        [ticker_a, ticker_b, hedge_ratio, adf_stat, p_value, half_life, sector]
        sorted by p_value ascending.
    """
    processed_dir = processed_dir or CONFIG["processed_dir"]
    p_value_threshold = p_value_threshold or CONFIG["coint_p_value_threshold"]
    min_half_life = min_half_life if min_half_life is not None else CONFIG["min_half_life"]
    max_half_life = max_half_life if max_half_life is not None else CONFIG["max_half_life"]

    # Load daily close prices for all available tickers
    daily_dir = Path(processed_dir) / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError(
            f"Daily processed data not found: {daily_dir}\n"
            "Run preprocess_all_tickers() first."
        )

    available_files = {p.stem for p in daily_dir.glob("*.parquet")}
    sector_mapping = CONFIG["sector_mapping"]

    if tickers is None:
        tickers = sorted(available_files & set(sector_mapping.keys()))
    else:
        tickers = sorted(set(tickers) & available_files)

    # Load all daily close prices
    price_data: dict[str, pl.DataFrame] = {}
    for ticker in tickers:
        try:
            df = load_processed(ticker, "daily", processed_dir)
            df = df.select(["timestamp", "close"]).rename({"close": ticker})
            price_data[ticker] = df
        except FileNotFoundError:
            continue

    if len(price_data) < 2:
        print("Not enough tickers with data for cointegration analysis.")
        return pl.DataFrame(
            schema={
                "ticker_a": pl.Utf8,
                "ticker_b": pl.Utf8,
                "hedge_ratio": pl.Float64,
                "adf_stat": pl.Float64,
                "p_value": pl.Float64,
                "half_life": pl.Float64,
                "sector": pl.Utf8,
            }
        )

    # Build sector-based pair combinations
    pairs_to_test: list[tuple[str, str, str]] = []  # (ticker_a, ticker_b, sector)
    for sector in get_all_sectors():
        sector_tickers = [t for t in get_tickers_by_sector(sector) if t in price_data]
        if len(sector_tickers) < 2:
            continue
        for a, b in itertools.combinations(sector_tickers, 2):
            pairs_to_test.append((a, b, sector))

    if not pairs_to_test:
        print("No valid pairs to test.")
        return pl.DataFrame(
            schema={
                "ticker_a": pl.Utf8,
                "ticker_b": pl.Utf8,
                "hedge_ratio": pl.Float64,
                "adf_stat": pl.Float64,
                "p_value": pl.Float64,
                "half_life": pl.Float64,
                "sector": pl.Utf8,
            }
        )

    results: list[dict] = []

    for ticker_a, ticker_b, sector in tqdm(
        pairs_to_test, desc="Testing cointegration"
    ):
        df_a = price_data[ticker_a]
        df_b = price_data[ticker_b]

        # Inner join on timestamp to get overlapping dates
        merged = df_a.join(df_b, on="timestamp", how="inner")

        if len(merged) < 60:  # need minimum observations
            continue

        prices_a = merged[ticker_a].to_numpy().astype(np.float64)
        prices_b = merged[ticker_b].to_numpy().astype(np.float64)

        try:
            result = test_cointegration(prices_a, prices_b)
        except Exception:
            continue

        # Apply filters
        if result["p_value"] > p_value_threshold:
            continue
        if result["half_life"] < min_half_life or result["half_life"] > max_half_life:
            continue

        results.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "hedge_ratio": result["hedge_ratio"],
                "adf_stat": result["adf_stat"],
                "p_value": result["p_value"],
                "half_life": result["half_life"],
                "sector": sector,
            }
        )

    if not results:
        print("No cointegrated pairs found with current thresholds.")
        return pl.DataFrame(
            schema={
                "ticker_a": pl.Utf8,
                "ticker_b": pl.Utf8,
                "hedge_ratio": pl.Float64,
                "adf_stat": pl.Float64,
                "p_value": pl.Float64,
                "half_life": pl.Float64,
                "sector": pl.Utf8,
            }
        )

    df_results = pl.DataFrame(results).sort("p_value")

    return df_results


def save_cointegration_results(
    df: pl.DataFrame,
    results_dir: str | None = None,
) -> Path:
    """
    Save cointegration results to Parquet.

    Args:
        df:          Cointegration results DataFrame.
        results_dir: Output directory. Defaults to CONFIG['results_dir'].

    Returns:
        Path to the saved file.
    """
    results_dir = results_dir or CONFIG["results_dir"]
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cointegration_pairs.parquet"
    df.write_parquet(out_path)
    print(f"Saved {len(df)} pairs to {out_path}")
    return out_path
