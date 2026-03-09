"""
Grid search worker module.

Defines _init_worker and _eval_combo as top-level importable functions so that
multiprocessing with the 'spawn' start method can pickle them by reference.
Functions defined inside Jupyter notebook cells cannot be pickled by spawn
(they live in __main__ which the spawned process can't import).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when imported by a spawned worker process
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import datetime as dt

import polars as pl

# ---------------------------------------------------------------------------
# Module-level globals — set once per worker by _init_worker
# ---------------------------------------------------------------------------

_worker_prices: dict[str, pl.DataFrame] = {}
_worker_trading_days: list[dt.date] = []
_worker_pairs_cache: dict[tuple[int, int, int, dt.date], pl.DataFrame | None] = {}


def _init_worker(
    prices_data: dict[str, pl.DataFrame],
    trading_days_data: list[dt.date],
    pairs_cache_data: dict[tuple[int, int, int, dt.date], pl.DataFrame | None],
) -> None:
    """Pool initializer — runs once per worker at startup."""
    global _worker_prices, _worker_trading_days, _worker_pairs_cache
    _worker_prices = prices_data
    _worker_trading_days = trading_days_data
    _worker_pairs_cache = pairs_cache_data


def _eval_combo(args: tuple) -> dict | None:
    """Evaluate one parameter combo. Returns a result dict or None."""
    (params, timeframe, bars_per_day_val, minutes_per_bar_val,
     holding_horizons, cost_bps) = args

    from analysis.signals import generate_pair_signals_for_day
    from analysis.evaluation import evaluate_all

    prices       = _worker_prices
    trading_days = _worker_trading_days
    pairs_cache  = _worker_pairs_cache

    zscore_window_bars_val = params["zscore_window_days"] * bars_per_day_val
    max_holding_bars_val   = params["max_holding_minutes"] // minutes_per_bar_val
    formation_days         = params["formation_window_days"]
    min_half_life          = params["min_half_life"]
    max_half_life          = params["max_half_life"]

    all_day_signals: list[pl.DataFrame] = []

    for day_idx, day in enumerate(trading_days):
        pairs_df = pairs_cache.get((formation_days, min_half_life, max_half_life, day))
        if pairs_df is None:
            continue

        lookback_days = params["zscore_window_days"] + 1
        lookback_start = trading_days[max(0, day_idx - lookback_days)]

        day_prices = {
            t: df.filter(
                (pl.col("timestamp").dt.date() >= lookback_start) &
                (pl.col("timestamp").dt.date() <= day)
            )
            for t, df in prices.items()
        }

        signals_day = generate_pair_signals_for_day(
            pairs_df=pairs_df,
            date=day,
            intraday_prices=day_prices,
            zscore_window=zscore_window_bars_val,
            z_entry=params["z_entry"],
            z_exit=params["z_exit"],
            z_stop=params["z_stop"],
            max_holding_bars=max_holding_bars_val,
        )

        if len(signals_day) > 0:
            signals_day = signals_day.filter(
                pl.col("timestamp").dt.date() == day
            )

        if len(signals_day) > 0:
            all_day_signals.append(signals_day)

    if not all_day_signals:
        return None

    signals_all = pl.concat(all_day_signals)

    eval_df = evaluate_all(
        signals_df=signals_all,
        prices=prices,
        holding_bars_list=holding_horizons,
        cost_bps=cost_bps,
    )

    if len(eval_df) == 0:
        return None

    agg = eval_df.select([
        pl.col("ic_gross_weighted").fill_nan(None).mean(),
        pl.col("mean_net_return").mean(),
        pl.col("hit_rate_binary").mean(),
        pl.col("n_observations").sum(),
    ]).to_dicts()[0]

    if agg["n_observations"] == 0 or agg["mean_net_return"] is None:
        return None

    mean_net = agg["mean_net_return"]
    # breakeven_bps: cost per leg at which net return = 0.
    # mean_net_return = mean_gross_return - 4 * cost_bps / 10_000
    # => breakeven_bps = cost_bps + mean_net_return * 2_500
    if mean_net is not None:
        mean_gross = mean_net + 4.0 * cost_bps / 10_000.0
        breakeven = cost_bps + mean_net * 2_500.0
    else:
        mean_gross = None
        breakeven = None

    return {
        **params,
        "ic_gross":         agg["ic_gross_weighted"],
        "mean_gross_return": mean_gross,
        "mean_net_return":  mean_net,
        "breakeven_bps":    breakeven,
        "hit_rate":         agg["hit_rate_binary"],
        "n_obs":            agg["n_observations"],
    }
