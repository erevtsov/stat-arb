"""
Signal generation module.

Computes spreads, z-scores, and entry/exit signals for cointegrated pairs
using rolling-window normalization.
"""

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
        window: Rolling window in bars. Defaults to CONFIG['zscore_window'].

    Returns:
        Polars Series of z-score values (first `window-1` values are null).
    """
    window = window or CONFIG["zscore_window"]
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
    z_entry = z_entry if z_entry is not None else CONFIG["z_score_entry"]
    z_exit = z_exit if z_exit is not None else CONFIG["z_score_exit"]
    z_stop = z_stop if z_stop is not None else CONFIG["z_score_stop"]

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
