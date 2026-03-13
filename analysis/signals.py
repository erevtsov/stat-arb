"""
Signal generation module.

Computes spreads, z-scores, and entry/exit signals for cointegrated pairs
using rolling-window normalization.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

from utils.config import CONFIG


def compute_spread(
    prices_a: pl.Series,
    prices_b: pl.Series,
    hedge_ratio: float,
) -> pl.Series:
    """
    Compute the log-price spread between two price series.

    log_spread_t = log(prices_a_t) - hedge_ratio * log(prices_b_t)

    hedge_ratio is the log-elasticity beta from OLS of log(A) on log(B).
    The spread is in log-return space: a unit change in log_spread represents
    approximately a 1% divergence between A and B.

    Args:
        prices_a:    Close prices for stock A (must be positive).
        prices_b:    Close prices for stock B (must be positive).
        hedge_ratio: Log-elasticity beta from log-price OLS regression.

    Returns:
        Polars Series of log-spread values.
    """
    return prices_a.log(math.e) - hedge_ratio * prices_b.log(math.e)


def compute_rolling_stats(
    spread: pl.Series,
    window: int,
) -> tuple[pl.Series, pl.Series]:
    """
    Compute rolling mean and standard deviation of the spread.

    Args:
        spread: Spread time series.
        window: Rolling window in bars.

    Returns:
        Tuple of (rolling_mean, rolling_std) Polars Series.
    """
    rolling_mean = spread.rolling_mean(window_size=window)
    rolling_std = spread.rolling_std(window_size=window)
    return rolling_mean, rolling_std


def compute_zscore(
    spread: pl.Series,
    window: int | None = None,
) -> pl.Series:
    """
    Compute rolling z-score of the spread.

    z_t = (spread_t - rolling_mean) / rolling_std

    Args:
        spread: Spread time series.
        window: Rolling window in bars. Required — use zscore_window_bars(timeframe)
                from utils.config to compute the right value for your timeframe.

    Returns:
        Polars Series of z-score values (first `window-1` values are null).
    """
    if window is None:
        raise ValueError(
            "window is required. Use zscore_window_bars(timeframe) from utils.config."
        )
    rolling_mean, rolling_std = compute_rolling_stats(spread, window)
    return (spread - rolling_mean) / rolling_std


def generate_signals(
    zscore: pl.Series,
    spread: pl.Series | None = None,
    rolling_mean: pl.Series | None = None,
    rolling_std: pl.Series | None = None,
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    max_holding_bars: int | None = None,
    fixed_exit_norm: bool = False,
) -> pl.Series:
    """
    Generate position signals from a z-score series.

    Rules:
      - Enter long spread  when z < -z_entry  (signal = +1)
      - Enter short spread when z > +z_entry  (signal = -1)
      - Exit when |z| <= z_exit (mean reversion)
      - Exit when |z| > z_stop (stop loss)
      - Exit after max_holding_bars (time stop)

    Signals are stateful: once entered, position is held until an exit
    condition fires. Only one position at a time.

    Args:
        zscore:           Z-score series.
        spread:           Raw log-spread series. Required when fixed_exit_norm=True.
        rolling_mean:     Rolling mean series. Required when fixed_exit_norm=True.
        rolling_std:      Rolling std series. Required when fixed_exit_norm=True.
        z_entry:          Entry threshold. Defaults to CONFIG value.
        z_exit:           Exit threshold. Defaults to CONFIG value.
        z_stop:           Stop-loss threshold. None = no stop-loss (rely on z_exit
                          and time stop only). Defaults to CONFIG value.
        max_holding_bars: Max bars before forced exit. None = no time stop.
        fixed_exit_norm:  If True, the mean-reversion exit z-score is computed using
                          the rolling mean/std snapshotted at entry rather than the
                          current rolling values. This prevents vol expansion from
                          triggering spurious exits when sigma grows after entry.

    Returns:
        Polars Series of signals: +1 (long spread), -1 (short spread), 0 (flat).
    """
    z_entry = z_entry if z_entry is not None else CONFIG.signal.z_entry
    z_exit = z_exit if z_exit is not None else CONFIG.signal.z_exit
    # z_stop=None means no stop-loss (rely on z_exit and time stop only)

    # Use numpy for faster array access; nulls become NaN via to_numpy()
    zvals = zscore.to_numpy()
    n = len(zvals)
    signals = np.zeros(n, dtype=np.int32)
    position = 0  # current position: +1, -1, or 0
    bars_held = 0

    # Cache locals to avoid repeated attribute/global lookups in the hot loop
    _z_entry = z_entry
    _z_exit = z_exit
    _z_stop = z_stop
    _max_holding = max_holding_bars
    _fixed = fixed_exit_norm and spread is not None and rolling_mean is not None and rolling_std is not None

    spread_vals = spread.to_numpy() if _fixed else None
    mean_vals = rolling_mean.to_numpy() if _fixed else None
    std_vals = rolling_std.to_numpy() if _fixed else None
    mu_entry = 0.0
    sigma_entry = 1.0

    for i in range(n):
        z = zvals[i]
        if z != z:  # NaN check (faster than np.isnan for scalars)
            signals[i] = 0
            position = 0
            bars_held = 0
            continue

        if position == 0:
            # Check entry
            if z < -_z_entry:
                position = 1  # long spread
                if _fixed:
                    mu_entry = mean_vals[i]
                    sigma_entry = std_vals[i]
            elif z > _z_entry:
                position = -1  # short spread
                if _fixed:
                    mu_entry = mean_vals[i]
                    sigma_entry = std_vals[i]
        else:
            bars_held += 1
            # Check exit conditions
            exit_trade = False

            # Mean reversion exit — use fixed entry-time normalization if requested
            if _fixed and sigma_entry != 0.0:
                z_exit_val = (spread_vals[i] - mu_entry) / sigma_entry
            else:
                z_exit_val = z

            if position == 1 and z_exit_val >= -_z_exit:
                exit_trade = True
            elif position == -1 and z_exit_val <= _z_exit:
                exit_trade = True

            # Stop loss exit uses rolling z-score (vol expansion into stop is a real signal)
            if not exit_trade and _z_stop is not None:
                if z > _z_stop or z < -_z_stop:
                    exit_trade = True

            # Time stop exit
            if not exit_trade and _max_holding is not None and bars_held >= _max_holding:
                exit_trade = True

            if exit_trade:
                position = 0
                bars_held = 0

        signals[i] = position

    return pl.Series("signal", signals)


# ---------------------------------------------------------------------------
# P-value weighting
# ---------------------------------------------------------------------------


def compute_pvalue_weights(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rank-inverse signal weights to a cointegrated pairs DataFrame.

    Pairs are assumed to be sorted by p_value ascending (most significant first),
    as returned by find_cointegrated_pairs().  Rank 1 = most significant.

    weight_i = (1 / rank_i) / sum(1 / rank_j for all j)

    Args:
        pairs_df: DataFrame with at least a ``p_value`` column, sorted ascending.

    Returns:
        pairs_df with an additional ``signal_weight: Float64`` column.
    """
    n = len(pairs_df)
    if n == 0:
        return pairs_df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("signal_weight")
        )

    # Ranks are 1-based; pairs_df is already sorted by p_value ascending
    inv_ranks = [1.0 / (i + 1) for i in range(n)]
    total = sum(inv_ranks)
    weights = [w / total for w in inv_ranks]
    return pairs_df.with_columns(pl.Series("signal_weight", weights, dtype=pl.Float64))


# ---------------------------------------------------------------------------
# Day-level signal generation
# ---------------------------------------------------------------------------


def generate_pair_signals_for_day(
    pairs_df: pl.DataFrame,
    date: dt.date | str,
    intraday_prices: dict[str, pl.DataFrame],
    zscore_window: int,
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    max_holding_bars: int | None = None,
    fixed_exit_norm: bool = False,
) -> pl.DataFrame:
    """
    Generate intraday signals for all cointegrated pairs on a single trading day.

    For each pair in ``pairs_df``:
      1. Extract close prices from ``intraday_prices``.
      2. Compute the spread and rolling z-score.
      3. Generate stateful binary signals (+1/-1/0).
      4. Multiply by the pair's ``signal_weight`` to produce a weighted signal.

    Args:
        pairs_df:        DataFrame from find_cointegrated_pairs() +
                         compute_pvalue_weights().  Must have columns:
                         ticker_a, ticker_b, hedge_ratio, p_value, signal_weight.
        date:            Trading date (used to tag rows).
        intraday_prices: Dict mapping ticker → 15-min OHLCV DataFrame with
                         columns [timestamp, open, high, low, close, volume].
        zscore_window:   Rolling window in bars.  Compute with
                         zscore_window_bars(timeframe) from utils.config.
        z_entry:         Entry z-score threshold (default: CONFIG.signal.z_entry).
        z_exit:          Exit z-score threshold (default: CONFIG.signal.z_exit).
        z_stop:          Stop-loss z-score threshold (default: CONFIG.signal.z_stop).
        max_holding_bars: Max bars before forced exit (default:
                          CONFIG.signal.max_holding_minutes derived at call site).
        fixed_exit_norm:  If True, use entry-time rolling mean/std for exit z-score
                          to prevent vol expansion from triggering spurious exits.

    Returns:
        Polars DataFrame with schema:
        [date, timestamp, ticker_a, ticker_b, hedge_ratio, p_value,
         signal_weight, zscore, signal_binary, signal_weighted]

        Pairs for which one or both tickers are missing from intraday_prices,
        or have fewer than 2 overlapping bars, are silently skipped.
    """
    date_val = str(date)
    _schema = {
        "date": pl.Utf8,
        "timestamp": pl.Datetime,
        "ticker_a": pl.Utf8,
        "ticker_b": pl.Utf8,
        "hedge_ratio": pl.Float64,
        "p_value": pl.Float64,
        "signal_weight": pl.Float64,
        "zscore": pl.Float64,
        "signal_binary": pl.Int32,
        "signal_weighted": pl.Float64,
    }
    pair_dfs: list[pl.DataFrame] = []

    for row in pairs_df.iter_rows(named=True):
        ticker_a = row["ticker_a"]
        ticker_b = row["ticker_b"]
        hedge_ratio = row["hedge_ratio"]
        p_value = row["p_value"]
        weight = row["signal_weight"]

        if ticker_a not in intraday_prices or ticker_b not in intraday_prices:
            continue

        df_a = intraday_prices[ticker_a]
        df_b = intraday_prices[ticker_b]

        # Align on common timestamps
        merged = df_a.select(["timestamp", "close"]).join(
            df_b.select(["timestamp", "close"]),
            on="timestamp",
            how="inner",
            suffix="_b",
        )
        if len(merged) < 2:
            continue

        prices_a = merged["close"]
        prices_b = merged["close_b"]
        timestamps = merged["timestamp"]

        spread = compute_spread(prices_a, prices_b, hedge_ratio)
        rolling_mean, rolling_std = compute_rolling_stats(spread, window=zscore_window)
        zscore = (spread - rolling_mean) / rolling_std
        signal_bin = generate_signals(
            zscore,
            spread=spread,
            rolling_mean=rolling_mean,
            rolling_std=rolling_std,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            max_holding_bars=max_holding_bars,
            fixed_exit_norm=fixed_exit_norm,
        )

        n = len(timestamps)
        pair_dfs.append(
            pl.DataFrame(
                {
                    "date": pl.Series([date_val] * n, dtype=pl.Utf8),
                    "timestamp": timestamps,
                    "ticker_a": pl.Series([ticker_a] * n, dtype=pl.Utf8),
                    "ticker_b": pl.Series([ticker_b] * n, dtype=pl.Utf8),
                    "hedge_ratio": pl.Series([hedge_ratio] * n, dtype=pl.Float64),
                    "p_value": pl.Series([p_value] * n, dtype=pl.Float64),
                    "signal_weight": pl.Series([weight] * n, dtype=pl.Float64),
                    "zscore": zscore.cast(pl.Float64),
                    "signal_binary": signal_bin.cast(pl.Int32),
                    "signal_weighted": signal_bin.cast(pl.Float64) * weight,
                }
            )
        )

    if not pair_dfs:
        return pl.DataFrame(schema=_schema)

    return pl.concat(pair_dfs)
