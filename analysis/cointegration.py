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
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller
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
    timeframe: str = "daily",
    processed_dir: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    p_value_threshold: float | None = None,
    min_half_life: float | None = None,
    max_half_life: float | None = None,
) -> pl.DataFrame:
    """
    Find all cointegrated pairs within each sector.

    Tests all pairwise combinations of tickers within the same sector
    using close prices at the specified timeframe. Only pairs with complete
    overlapping data are tested.

    Args:
        tickers:            List of tickers to consider. Defaults to all in config.
        timeframe:          Timeframe to use ('daily', '1hour', '15min', etc).
        processed_dir:      Directory with processed data.
        start_date:         Start date for formation period (YYYY-MM-DD or datetime).
                            If None, uses all available data from beginning.
        end_date:           End date for formation period (YYYY-MM-DD or datetime).
                            If None, uses all available data to end.
        p_value_threshold:  Max p-value to include pair. Defaults to config value.
        min_half_life:      Min half-life filter (in bars). Defaults to config value.
        max_half_life:      Max half-life filter (in bars). Defaults to config value.

    Returns:
        Polars DataFrame with columns:
        [ticker_a, ticker_b, hedge_ratio, adf_stat, p_value, half_life, sector,
         timeframe, formation_start, formation_end, n_observations]
        sorted by p_value ascending.

    Examples:
        # Use all available data (backward compatible)
        pairs = find_cointegrated_pairs(timeframe="daily")

        # Formation period only (proper walk-forward)
        pairs = find_cointegrated_pairs(
            timeframe="15min",
            start_date="2023-01-01",
            end_date="2023-06-30"
        )
    """
    processed_dir = processed_dir or CONFIG.paths.processed_dir
    p_value_threshold = p_value_threshold or CONFIG.cointegration.p_value_threshold
    min_half_life = (
        min_half_life if min_half_life is not None else CONFIG.cointegration.min_half_life
    )
    max_half_life = (
        max_half_life if max_half_life is not None else CONFIG.cointegration.max_half_life
    )

    # Load close prices for all available tickers at specified timeframe
    timeframe_dir = Path(processed_dir) / timeframe
    if not timeframe_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found for timeframe '{timeframe}': {timeframe_dir}\n"
            "Run preprocess_all_tickers() first."
        )

    available_files = {p.stem for p in timeframe_dir.glob("*.parquet")}
    sector_mapping = CONFIG.universe.sector_mapping

    if tickers is None:
        tickers = sorted(available_files & set(sector_mapping.keys()))
    else:
        tickers = sorted(set(tickers) & available_files)

    # Load all close prices at specified timeframe
    price_data: dict[str, pl.DataFrame] = {}
    for ticker in tickers:
        try:
            # Pass date filters to load_processed() for efficient Parquet filtering
            df = load_processed(ticker, timeframe, processed_dir, start_date, end_date)

            # Skip if no data remains after filtering
            if len(df) == 0:
                continue

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
                "timeframe": pl.Utf8,
                "formation_start": pl.Utf8,
                "formation_end": pl.Utf8,
                "n_observations": pl.Int64,
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
                "timeframe": pl.Utf8,
                "formation_start": pl.Utf8,
                "formation_end": pl.Utf8,
                "n_observations": pl.Int64,
            }
        )

    # Validate formation period length
    if start_date is not None and end_date is not None:
        import warnings
        from datetime import datetime

        # Parse dates if strings
        if isinstance(start_date, str):
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = start_date

        if isinstance(end_date, str):
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = end_date

        duration_days = (end_dt - start_dt).days

        # Calculate expected bars for this timeframe
        bars_per_day = {"1min": 390, "5min": 78, "15min": 26, "1hour": 6.5, "daily": 1}
        expected_bars = duration_days * bars_per_day.get(timeframe, 1)

        if expected_bars < 60:
            warnings.warn(
                f"Formation period very short: {duration_days} days ({expected_bars:.0f} bars). "
                f"Minimum 60 observations recommended for reliable cointegration testing."
            )
        elif expected_bars < 250:
            warnings.warn(
                f"Formation period marginal: {duration_days} days ({expected_bars:.0f} bars). "
                f"250+ observations recommended for robust results."
            )

    results: list[dict] = []

    for ticker_a, ticker_b, sector in tqdm(pairs_to_test, desc="Testing cointegration"):
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
                "timeframe": timeframe,
                "formation_start": start_date,
                "formation_end": end_date,
                "n_observations": len(merged),
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
                "timeframe": pl.Utf8,
                "formation_start": pl.Utf8,
                "formation_end": pl.Utf8,
                "n_observations": pl.Int64,
            }
        )

    df_results = pl.DataFrame(results).sort("p_value")

    return df_results


def find_cointegrated_pairs_rolling(
    window_days: int,
    step_days: int,
    start_date: str,
    end_date: str,
    **kwargs,
) -> dict[str, pl.DataFrame]:
    """
    Find cointegrated pairs on rolling windows for walk-forward analysis.

    Generates overlapping or non-overlapping windows and runs cointegration
    testing on each window independently.

    Args:
        window_days: Size of formation window in calendar days (e.g. 84 ≈ 60 trading days).
        step_days:   Step size between windows in calendar days (e.g. 1 = daily recomputation).
                     If step_days == window_days, windows are non-overlapping.
        start_date:  First window start date (YYYY-MM-DD).
        end_date:    Last window end date (YYYY-MM-DD).
        **kwargs:    Additional arguments passed to find_cointegrated_pairs()
                     (tickers, timeframe, thresholds, etc.).

    Returns:
        Dict mapping window identifiers to cointegration results DataFrames.
        Keys are formatted as "YYYY-MM-DD_YYYY-MM-DD" (start_end).

    Example:
        # 84-day rolling window (≈60 trading days), 1-day step (daily recomputation)
        results = find_cointegrated_pairs_rolling(
            window_days=84,
            step_days=1,
            start_date="2023-01-01",
            end_date="2024-12-31",
            timeframe="15min",
            p_value_threshold=0.01
        )

        # Access specific window
        window1_pairs = results["2023-01-01_2023-03-26"]

        # Analyze all windows
        for window_id, pairs_df in results.items():
            print(f"{window_id}: {len(pairs_df)} pairs")
    """
    from datetime import datetime, timedelta

    # Parse dates
    current_start = datetime.fromisoformat(start_date)
    final_end = datetime.fromisoformat(end_date)

    results = {}
    window_count = 0

    while True:
        # Calculate window end
        window_end = current_start + timedelta(days=window_days)

        # Stop if window extends beyond final_end
        if window_end > final_end:
            break

        # Format dates for this window
        window_start_str = current_start.strftime("%Y-%m-%d")
        window_end_str = window_end.strftime("%Y-%m-%d")
        window_id = f"{window_start_str}_{window_end_str}"

        # Find cointegrated pairs for this window
        print(f"Processing window {window_count + 1}: {window_id}")
        pairs_df = find_cointegrated_pairs(
            start_date=window_start_str, end_date=window_end_str, **kwargs
        )

        results[window_id] = pairs_df
        window_count += 1

        # Move to next window
        current_start = current_start + timedelta(days=step_days)

    print(f"Completed {window_count} windows")
    return results


def track_pair_stability(
    ticker_a: str,
    ticker_b: str,
    windows: list[tuple[str, str]],
    timeframe: str = "daily",
    **kwargs,
) -> pl.DataFrame:
    """
    Track cointegration metrics for a specific pair across multiple time windows.

    Useful for validating coefficient stability and monitoring
    pair health in production. Tests the same pair on different formation periods
    to see how hedge ratio, ADF statistic, and half-life evolve.

    Args:
        ticker_a:  First ticker symbol.
        ticker_b:  Second ticker symbol.
        windows:   List of (start_date, end_date) tuples defining formation periods.
        timeframe: Timeframe to use for testing (default: "daily").
        **kwargs:  Additional arguments passed to find_cointegrated_pairs().

    Returns:
        Polars DataFrame with columns:
        [window_start, window_end, window_id, cointegrated, hedge_ratio,
         adf_stat, p_value, half_life, n_observations]

        If pair not cointegrated in a window, cointegrated=False and metrics are null.

    Example:
        # Define quarterly windows for 2023
        windows = [
            ("2023-01-01", "2023-03-31"),  # Q1
            ("2023-04-01", "2023-06-30"),  # Q2
            ("2023-07-01", "2023-09-30"),  # Q3
            ("2023-10-01", "2023-12-31"),  # Q4
        ]

        # Track KO/PEP stability
        stability = track_pair_stability(
            "KO", "PEP",
            windows=windows,
            timeframe="daily"
        )

        # Analyze coefficient of variation
        stable_windows = stability.filter(pl.col("cointegrated") == True)
        hedge_ratio_cv = (
            stable_windows["hedge_ratio"].std() /
            stable_windows["hedge_ratio"].mean()
        )
        print(f"Hedge ratio CV: {hedge_ratio_cv:.3f}")
    """
    results = []

    for window_start, window_end in windows:
        window_id = f"{window_start}_{window_end}"

        # Find cointegrated pairs for this window
        pairs_df = find_cointegrated_pairs(
            tickers=[ticker_a, ticker_b],
            timeframe=timeframe,
            start_date=window_start,
            end_date=window_end,
            **kwargs,
        )

        # Check if this specific pair is cointegrated
        pair_result = pairs_df.filter(
            ((pl.col("ticker_a") == ticker_a) & (pl.col("ticker_b") == ticker_b))
            | ((pl.col("ticker_a") == ticker_b) & (pl.col("ticker_b") == ticker_a))
        )

        if len(pair_result) > 0:
            # Pair is cointegrated in this window
            row = pair_result.row(0, named=True)
            results.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_id": window_id,
                    "cointegrated": True,
                    "hedge_ratio": row["hedge_ratio"],
                    "adf_stat": row["adf_stat"],
                    "p_value": row["p_value"],
                    "half_life": row["half_life"],
                    "n_observations": row["n_observations"],
                }
            )
        else:
            # Pair NOT cointegrated in this window
            results.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_id": window_id,
                    "cointegrated": False,
                    "hedge_ratio": None,
                    "adf_stat": None,
                    "p_value": None,
                    "half_life": None,
                    "n_observations": None,
                }
            )

    return pl.DataFrame(results)
