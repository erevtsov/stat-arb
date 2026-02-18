"""
Signal generation module.

Computes spreads, z-scores, and entry/exit signals for cointegrated pairs
using rolling-window normalization.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from utils.config import CONFIG


def compute_spread(
    prices_a: pl.Series,
    prices_b: pl.Series,
    hedge_ratio: float,
) -> pl.Series:
    """
    Compute the spread between two price series.

    spread_t = prices_a_t - hedge_ratio * prices_b_t

    Args:
        prices_a:    Close prices for stock A.
        prices_b:    Close prices for stock B.
        hedge_ratio: OLS hedge ratio (beta).

    Returns:
        Polars Series of spread values.
    """
    return prices_a - hedge_ratio * prices_b


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
    rolling_mean = spread.rolling_mean(window_size=window)
    rolling_std = spread.rolling_std(window_size=window)
    return (spread - rolling_mean) / rolling_std


def generate_signals(
    zscore: pl.Series,
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    max_holding_bars: int | None = None,
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
        z_entry:          Entry threshold. Defaults to CONFIG value.
        z_exit:           Exit threshold. Defaults to CONFIG value.
        z_stop:           Stop-loss threshold. Defaults to CONFIG value.
        max_holding_bars: Max bars before forced exit. None = no time stop.

    Returns:
        Polars Series of signals: +1 (long spread), -1 (short spread), 0 (flat).
    """
    z_entry = z_entry if z_entry is not None else CONFIG.signal.z_entry
    z_exit = z_exit if z_exit is not None else CONFIG.signal.z_exit
    z_stop = z_stop if z_stop is not None else CONFIG.signal.z_stop

    zvals = zscore.to_list()
    n = len(zvals)
    signals = [0] * n
    position = 0  # current position: +1, -1, or 0
    bars_held = 0

    for i in range(n):
        z = zvals[i]
        if z is None:
            signals[i] = 0
            position = 0
            bars_held = 0
            continue

        if position == 0:
            # Check entry
            if z < -z_entry:
                position = 1  # long spread
            elif z > z_entry:
                position = -1  # short spread
        else:
            bars_held += 1
            # Check exit conditions
            exit_trade = False

            # Mean reversion exit
            if position == 1 and z >= -z_exit:
                exit_trade = True
            elif position == -1 and z <= z_exit:
                exit_trade = True

            # Stop loss exit
            if abs(z) > z_stop:
                exit_trade = True

            # Time stop exit
            if max_holding_bars is not None and bars_held >= max_holding_bars:
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
        return pairs_df.with_columns(pl.lit(None).cast(pl.Float64).alias("signal_weight"))

    # Ranks are 1-based; pairs_df is already sorted by p_value ascending
    inv_ranks = [1.0 / (i + 1) for i in range(n)]
    total = sum(inv_ranks)
    weights = [w / total for w in inv_ranks]
    return pairs_df.with_columns(
        pl.Series("signal_weight", weights, dtype=pl.Float64)
    )


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

    Returns:
        Polars DataFrame with schema:
        [date, timestamp, ticker_a, ticker_b, hedge_ratio, p_value,
         signal_weight, zscore, signal_binary, signal_weighted]

        Pairs for which one or both tickers are missing from intraday_prices,
        or have fewer than 2 overlapping bars, are silently skipped.
    """
    date_val = str(date)
    rows: list[dict] = []

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
        zscore = compute_zscore(spread, window=zscore_window)
        signal_bin = generate_signals(
            zscore,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            max_holding_bars=max_holding_bars,
        )

        zscore_list = zscore.to_list()
        signal_list = signal_bin.to_list()

        for ts, z, sig in zip(timestamps.to_list(), zscore_list, signal_list):
            rows.append(
                {
                    "date": date_val,
                    "timestamp": ts,
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "hedge_ratio": hedge_ratio,
                    "p_value": p_value,
                    "signal_weight": weight,
                    "zscore": z,
                    "signal_binary": sig,
                    "signal_weighted": float(sig) * weight,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
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
        )

    return pl.DataFrame(rows).with_columns(
        pl.col("signal_binary").cast(pl.Int32)
    )
