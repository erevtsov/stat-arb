"""
Grid search v2 for intraday pairs trading strategy parameters.

Two-stage design:
  Stage 1: Build cointegration pairs cache (expensive, one-time per formation combo)
  Stage 2: Sweep signal parameters in parallel (cheap per combo)

Key separation:
  - Formation params (rolling_window_days, min/max_half_life, p_value_threshold)
    drive ADF tests — expensive. Cached once per combo, reused across all signal combos.
  - Signal params (zscore_window, z_entry/exit/stop, max_holding) are cheap per day.

Metrics reported per (formation_combo × signal_combo × holding_horizon):
  - mean_ic_gross:     Mean cross-sectional Spearman IC (no cost) — directional signal quality
  - ic_t_stat:         IC / (std_IC / sqrt(n_days)) — statistical significance
  - mean_net_return:   Mean per-signal net return after 4-leg round-trip costs
  - mean_gross_return: Mean per-signal gross return (net + costs added back)
  - breakeven_bps:     Cost per leg at which net return = 0
  - mean_hit_rate:     Fraction of signals with correct direction
  - total_n_obs:       Total signal count across all days

Usage:
    # Phase 1 — directional correctness (fixed formation params, ~144 signal combos)
    python -m scripts.grid_search_v2 --start 2022-07-01 --end 2022-09-30

    # Phase 2 — full grid (all formation + signal combos)
    python -m scripts.grid_search_v2 --phase 2 --start 2022-07-01 --end 2022-09-30
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TextIO

import polars as pl
from dotenv import load_dotenv

from analysis.cointegration import find_cointegrated_pairs
from analysis.evaluation import evaluate_all
from analysis.preprocessing import load_processed
from analysis.signals import compute_pvalue_weights, generate_pair_signals_for_day
from utils.config import (
    BARS_PER_DAY,
    CONFIG,
    MINUTES_PER_BAR,
    get_all_tickers,
    holding_horizons_bars,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Logging (timestamped, tee'd to console + log file)
# ---------------------------------------------------------------------------

_log_fh: TextIO | None = None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

SIGNAL_GRID: dict[str, list] = {
    "zscore_window_days": [2, 5, 10],
    "z_entry": [2.5, 3.0, 3.5],
    "z_exit": [1.5, 2, 2.4],
    "z_stop": [4.5, 5.0],
    "max_holding_minutes": [120, 240, 390],
    "fixed_exit_norm": [True, False],
}

# Phase 2 formation grid
# FORMATION_GRID: dict[str, list] = {
#     "rolling_window_days": [21, 42, 84],
#     "min_half_life": [4],
#     "max_half_life": [12, 24, 48],
#     "p_value_threshold": [0.01, 0.05],
# }
FORMATION_GRID: dict[str, list] = {
    "rolling_window_days": [42, 84],
    "min_half_life": [2],
    "max_half_life": [12, 24],
    "p_value_threshold": [0.05],
}

# Phase 1 defaults (best-guess starting point)
FORMATION_DEFAULTS: dict = {
    "rolling_window_days": 42,
    "min_half_life": 4,
    "max_half_life": 18,
    "p_value_threshold": 0.05,
}

# ---------------------------------------------------------------------------
# Worker globals (populated by _init_worker in each spawned process)
# ---------------------------------------------------------------------------

_pairs_cache: dict | None = None  # date_str → pl.DataFrame of cointegrated pairs
_full_prices: dict | None = (
    None  # ticker → pl.DataFrame (full range: lookback + eval + lookahead)
)
_holding_bars: list | None = None  # list[int] — forward horizons in bars
_cost_bps: float | None = None  # one-way cost per leg in basis points
_timeframe: str | None = None


def _init_worker(
    pairs_cache: dict,
    full_prices: dict,
    holding_bars: list,
    cost_bps: float,
    timeframe: str,
) -> None:
    """Initialize per-worker globals. Called once per worker process at pool startup."""
    global _pairs_cache, _full_prices, _holding_bars, _cost_bps, _timeframe
    # Limit Polars' rayon thread pool to 1 per worker process.
    # Without this, each of N workers spawns ~8 rayon threads → N×8 threads competing
    # on the same cores, causing massive context-switch overhead.
    os.environ["POLARS_MAX_THREADS"] = "1"
    _pairs_cache = pairs_cache
    _full_prices = full_prices
    _holding_bars = holding_bars
    _cost_bps = cost_bps
    _timeframe = timeframe


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _eval_combo(combo: dict) -> list[dict] | None:
    """
    Evaluate one signal parameter combination across all cached days.

    Returns a list of result dicts (one per holding horizon), or None if no signals.
    """
    zscore_window_days: int = combo["zscore_window_days"]
    z_entry: float = combo["z_entry"]
    z_exit: float = combo["z_exit"]
    z_stop: float | None = combo["z_stop"]
    max_holding_minutes: int = combo["max_holding_minutes"]
    fixed_exit_norm: bool = combo.get("fixed_exit_norm", False)

    bars_per_day = BARS_PER_DAY[_timeframe]
    minutes_per_bar = MINUTES_PER_BAR[_timeframe]
    zscore_window = zscore_window_days * bars_per_day
    max_holding_bars_val = max_holding_minutes // minutes_per_bar

    # Extra calendar days to include in lookback so rolling zscore has enough bars.
    # Add buffer for weekends/holidays (~1.4x calendar days per trading day).
    lookback_calendar_days = int(zscore_window_days * 2) + 7

    all_signals: list[pl.DataFrame] = []

    for date_str, pairs_df in sorted(_pairs_cache.items()):
        if len(pairs_df) == 0:
            continue

        pairs_with_weights = compute_pvalue_weights(pairs_df)

        # Slice prices: lookback for z-score history + current day
        current_date = date.fromisoformat(date_str)
        lookback_start = current_date - timedelta(days=lookback_calendar_days)

        # Use pl.lit() to keep date comparisons native (avoids row-by-row Python
        # datetime conversion). Filter once and reuse the result (eliminates
        # the previous double-filter anti-pattern).
        day_prices: dict[str, pl.DataFrame] = {}
        for ticker, df in _full_prices.items():
            filtered = df.filter(
                (pl.col("timestamp").dt.date() >= pl.lit(lookback_start))
                & (pl.col("timestamp").dt.date() <= pl.lit(current_date))
            )
            if len(filtered) > 0:
                day_prices[ticker] = filtered

        day_signals = generate_pair_signals_for_day(
            pairs_df=pairs_with_weights,
            date=date_str,
            intraday_prices=day_prices,
            zscore_window=zscore_window,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            max_holding_bars=max_holding_bars_val,
            fixed_exit_norm=fixed_exit_norm,
        )

        if len(day_signals) > 0:
            all_signals.append(day_signals)

    if not all_signals:
        return None

    signals_df = pl.concat(all_signals)

    # Only evaluate at horizons the strategy can actually achieve (≤ max_holding_bars).
    # Longer horizons would measure counterfactual returns the state machine never captures.
    all_horizons = holding_horizons_bars(_timeframe)
    valid_horizons = [h for h in all_horizons if h <= max_holding_bars_val]
    if not valid_horizons:
        valid_horizons = [min(all_horizons)]  # always evaluate at the shortest horizon

    # evaluate_all uses full_prices for forward-return lookups (includes lookahead bars)
    eval_df = evaluate_all(
        signals_df=signals_df,
        prices=_full_prices,
        holding_bars_list=valid_horizons,
        cost_bps=_cost_bps,
    )

    if len(eval_df) == 0:
        return None

    # Aggregate daily metrics across all eval dates per holding horizon.
    # weighted_mean_ic_gross weights each day's IC by sqrt(n_obs) to de-weight
    # low-observation days whose IC estimates are unreliable.
    agg = (
        eval_df.group_by("n_bars")
        .agg(
            [
                pl.col("ic_gross_weighted").mean().alias("mean_ic_gross"),
                pl.col("ic_gross_weighted").std().alias("std_ic_gross"),
                (
                    (
                        pl.col("ic_gross_weighted")
                        * pl.col("n_observations").cast(pl.Float64).sqrt()
                    ).sum()
                    / pl.col("n_observations").cast(pl.Float64).sqrt().sum()
                ).alias("weighted_mean_ic_gross"),
                pl.col("pooled_ic_gross").first().alias("pooled_ic_gross"),
                pl.col("mean_net_return").mean().alias("mean_net_return"),
                pl.col("hit_rate_binary").mean().alias("mean_hit_rate"),
                pl.col("n_observations").sum().alias("total_n_obs"),
                pl.col("n_observations").mean().alias("mean_n_obs_per_day"),
                pl.len().alias("n_days"),
            ]
        )
        .with_columns(
            (
                pl.col("mean_ic_gross")
                / (pl.col("std_ic_gross") / pl.col("n_days").cast(pl.Float64).sqrt())
            ).alias("ic_t_stat"),
        )
        .sort("n_bars")
    )

    rows: list[dict] = []
    round_trip_cost = 4.0 * (_cost_bps / 10_000.0)

    for row in agg.iter_rows(named=True):
        mean_net = row["mean_net_return"]
        mean_gross = (mean_net + round_trip_cost) if mean_net is not None else None
        breakeven_bps = (
            (mean_gross * 10_000.0 / 4.0)
            if (mean_gross is not None and mean_gross > 0)
            else None
        )
        rows.append(
            {
                **combo,
                "n_bars": row["n_bars"],
                "mean_ic_gross": row["mean_ic_gross"],
                "weighted_mean_ic_gross": row["weighted_mean_ic_gross"],
                "pooled_ic_gross": row["pooled_ic_gross"],
                "std_ic_gross": row["std_ic_gross"],
                "ic_t_stat": row["ic_t_stat"],
                "mean_gross_return": mean_gross,
                "mean_net_return": mean_net,
                "breakeven_bps": breakeven_bps,
                "mean_hit_rate": row["mean_hit_rate"],
                "total_n_obs": row["total_n_obs"],
                "mean_n_obs_per_day": row["mean_n_obs_per_day"],
                "n_days": row["n_days"],
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Pairs cache construction
# ---------------------------------------------------------------------------


def build_pairs_cache(
    eval_dates: list[date],
    rolling_window_days: int,
    min_half_life: float,
    max_half_life: float,
    p_value_threshold: float,
    timeframe: str,
    price_cache: dict,
) -> dict[str, pl.DataFrame]:
    """
    Build a daily cointegration pairs cache.

    For each eval date d, computes cointegrated pairs using the formation window
    [d - rolling_window_days, d - 1] calendar days. Uses price_cache to avoid
    repeated Parquet I/O.

    Returns:
        dict mapping date.isoformat() → cointegrated pairs DataFrame (possibly empty).
    """
    cache: dict[str, pl.DataFrame] = {}
    for d in eval_dates:
        formation_start = (d - timedelta(days=rolling_window_days)).isoformat()
        formation_end = (d - timedelta(days=1)).isoformat()

        pairs_df = find_cointegrated_pairs(
            timeframe=timeframe,
            start_date=formation_start,
            end_date=formation_end,
            p_value_threshold=p_value_threshold,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            price_cache=price_cache,
        )
        cache[d.isoformat()] = pairs_df

    return cache


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def load_all_prices(
    timeframe: str,
    start_date: str,
    end_date: str,
) -> dict[str, pl.DataFrame]:
    """Load processed close prices for all universe tickers in the date range."""
    prices: dict[str, pl.DataFrame] = {}
    for ticker in get_all_tickers():
        try:
            df = load_processed(
                ticker, timeframe, start_date=start_date, end_date=end_date
            )
            if len(df) > 0:
                prices[ticker] = df
        except FileNotFoundError:
            pass
    return prices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search for intraday pairs trading parameters"
    )
    parser.add_argument(
        "--timeframe", default="15min", help="Bar timeframe (default: 15min)"
    )
    parser.add_argument(
        "--start", default="2019-01-01", help="Eval period start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2019-09-30", help="Eval period end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=float(CONFIG.portfolio.transaction_cost_bps),
        help="One-way transaction cost per leg in basis points",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 4)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(CONFIG.paths.results_dir) / "grid_search_v2.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=fixed formation params (~144 signal combos); 2=full grid",
    )
    args = parser.parse_args()

    timeframe = args.timeframe
    eval_start = date.fromisoformat(args.start)
    eval_end = date.fromisoformat(args.end)
    holding_bars = holding_horizons_bars(timeframe)

    # --- Formation combos ---
    if args.phase == 1:
        formation_combos = [FORMATION_DEFAULTS]
    else:
        f_keys = list(FORMATION_GRID.keys())
        formation_combos = [
            dict(zip(f_keys, vals))
            for vals in itertools.product(*FORMATION_GRID.values())
        ]

    # --- Signal combos ---
    s_keys = list(SIGNAL_GRID.keys())
    signal_combos = [
        dict(zip(s_keys, vals)) for vals in itertools.product(*SIGNAL_GRID.values())
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global _log_fh
    log_path = output_path.with_suffix(".log")
    _log_fh = open(log_path, "a")  # noqa: SIM115  (plain open fine here)

    try:
        _run(args, timeframe, eval_start, eval_end, holding_bars, formation_combos, signal_combos, output_path)
    finally:
        _log_fh.close()
        _log_fh = None


def _run(
    args: argparse.Namespace,
    timeframe: str,
    eval_start: date,
    eval_end: date,
    holding_bars: list[int],
    formation_combos: list[dict],
    signal_combos: list[dict],
    output_path: Path,
) -> None:
    _log(
        f"Phase {args.phase} | {len(formation_combos)} formation × "
        f"{len(signal_combos)} signal = {len(formation_combos) * len(signal_combos)} total combos"
    )
    _log(f"Eval period: {eval_start} → {eval_end} | timeframe: {timeframe}")
    _log(f"Cost: {args.cost_bps} bps/leg ({4 * args.cost_bps} bps round-trip)")
    _log(f"Log: {output_path.with_suffix('.log')}")

    # --- Date range for data loading ---
    max_formation_days = max(c["rolling_window_days"] for c in formation_combos)
    max_zscore_days = max(SIGNAL_GRID["zscore_window_days"])
    max_horizon_days = max(holding_bars) // BARS_PER_DAY[timeframe] + 2

    data_start = (
        eval_start - timedelta(days=max_formation_days + max_zscore_days * 2 + 14)
    ).isoformat()
    data_end = (eval_end + timedelta(days=max_horizon_days + 5)).isoformat()

    _log(f"Loading prices [{data_start} → {data_end}]...")
    price_cache = load_all_prices(timeframe, data_start, data_end)
    _log(f"Loaded {len(price_cache)} tickers")

    # --- Eval dates: all calendar days in eval period ---
    eval_dates: list[date] = []
    d = eval_start
    while d <= eval_end:
        eval_dates.append(d)
        d += timedelta(days=1)

    # --- Checkpoint directory ---
    checkpoint_dir = output_path.parent / (output_path.stem + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_EVERY = 25  # overwrite checkpoint file after every N completed combos

    # --- Run formation combos sequentially, signal sweep in parallel ---
    all_results: list[dict] = []

    run_start = time.monotonic()

    for f_idx, formation_params in enumerate(formation_combos):
        _log(f"[{f_idx + 1}/{len(formation_combos)}] Formation params: {formation_params}")

        _log("  Building pairs cache (ADF tests)...")
        pairs_cache = build_pairs_cache(
            eval_dates=eval_dates,
            timeframe=timeframe,
            price_cache=price_cache,
            **formation_params,
        )

        n_days_with_pairs = sum(1 for v in pairs_cache.values() if len(v) > 0)
        mean_pairs = sum(len(v) for v in pairs_cache.values()) / max(
            n_days_with_pairs, 1
        )
        _log(
            f"  Pairs cache: {n_days_with_pairs}/{len(eval_dates)} trading days | "
            f"avg {mean_pairs:.1f} pairs/day"
        )

        fp = formation_params
        checkpoint_path = checkpoint_dir / (
            f"f{fp['rolling_window_days']}d"
            f"_hl{fp['min_half_life']}-{fp['max_half_life']}"
            f"_p{fp['p_value_threshold']}"
            f"_idx{f_idx:02d}.parquet"
        )
        formation_results: list[dict] = []

        n_workers = args.workers or max(1, (os.cpu_count() or 2) - 4)
        _log(f"  Sweeping {len(signal_combos)} signal combos with {n_workers} workers...")

        n_valid = 0
        n_total = len(signal_combos)
        sweep_start = time.monotonic()

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(pairs_cache, price_cache, holding_bars, args.cost_bps, timeframe),
        ) as pool:
            future_to_combo = {
                pool.submit(_eval_combo, combo): combo for combo in signal_combos
            }
            for future in as_completed(future_to_combo):
                rows = future.result()
                if rows is None:
                    continue
                n_valid += 1
                for row in rows:
                    formation_results.append({**formation_params, **row})
                if n_valid % CHECKPOINT_EVERY == 0:
                    pl.DataFrame(formation_results).write_parquet(str(checkpoint_path))
                    elapsed = time.monotonic() - sweep_start
                    rate = n_valid / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_valid) / rate if rate > 0 else float("inf")
                    eta_str = _fmt_duration(eta) if eta != float("inf") else "?"
                    _log(
                        f"  [checkpoint] {n_valid}/{n_total} combos"
                        f" | elapsed {_fmt_duration(elapsed)} | ETA ~{eta_str}"
                        f" → {checkpoint_path.name}"
                    )

        # Final checkpoint for this formation combo
        if formation_results:
            pl.DataFrame(formation_results).write_parquet(str(checkpoint_path))
            elapsed = time.monotonic() - sweep_start
            _log(
                f"  [done] formation {f_idx + 1} | {n_valid}/{n_total} combos"
                f" | sweep took {_fmt_duration(elapsed)}"
                f" → {checkpoint_path.name}"
            )
        all_results.extend(formation_results)

    total_elapsed = time.monotonic() - run_start

    if not all_results:
        _log("No results produced. Check date range, data availability, and params.")
        return

    results_df = pl.DataFrame(all_results)

    results_df.write_parquet(str(output_path))
    _log(f"Saved {len(results_df)} result rows → {output_path} (total {_fmt_duration(total_elapsed)})")

    # --- Quick summary at shortest horizon ---
    shortest_h = min(holding_bars)
    summary = (
        results_df.filter(pl.col("n_bars") == shortest_h)
        .sort("mean_ic_gross", descending=True)
        .head(10)
        .select(
            [
                "zscore_window_days",
                "z_entry",
                "z_exit",
                "z_stop",
                "max_holding_minutes",
                "mean_ic_gross",
                "ic_t_stat",
                "mean_net_return",
                "mean_hit_rate",
                "total_n_obs",
            ]
        )
    )
    _log(f"Top 10 signal combos by mean_ic_gross at {shortest_h}-bar horizon:")
    _log(str(summary))

    n_positive_ic = (
        results_df.filter(pl.col("n_bars") == shortest_h)
        .filter(pl.col("mean_ic_gross") > 0)
        .height
    )
    n_positive_net = (
        results_df.filter(pl.col("n_bars") == shortest_h)
        .filter(
            pl.col("mean_net_return").is_not_null() & (pl.col("mean_net_return") > 0)
        )
        .height
    )
    total = results_df.filter(pl.col("n_bars") == shortest_h).height
    _log(
        f"Directional check (at {shortest_h}-bar horizon): "
        f"{n_positive_ic}/{total} combos have mean_ic_gross > 0 | "
        f"{n_positive_net}/{total} have mean_net_return > 0"
    )

    _send_completion_imessage(
        results_df, args, shortest_h, n_positive_ic, total, output_path
    )


def _send_completion_imessage(
    results_df: pl.DataFrame,
    args: argparse.Namespace,
    shortest_h: int,
    n_positive_ic: int,
    total: int,
    output_path: Path,
) -> None:
    """Send an iMessage via the macOS Messages app (osascript). No credentials needed."""
    notify_to = os.environ.get("NOTIFY_IMESSAGE_TO")
    if not notify_to:
        _log("Warning: NOTIFY_IMESSAGE_TO not set in .env — skipping iMessage notification")
        return

    top5 = (
        results_df.filter(pl.col("n_bars") == shortest_h)
        .sort("mean_ic_gross", descending=True)
        .head(5)
        .select(
            [
                "zscore_window_days",
                "z_entry",
                "z_exit",
                "mean_ic_gross",
                "mean_net_return",
            ]
        )
        .to_pandas()
        .to_string(index=False)
    )

    text = textwrap.dedent(f"""\
        [stat-arb] grid_search_v2 done (phase {args.phase})
        Period: {args.start} → {args.end} | tf: {args.timeframe}
        {n_positive_ic}/{total} combos with positive IC at {shortest_h}-bar horizon
        Output: {output_path}

        Top 5 by IC:
        {top5}
    """).strip()

    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'tell application "Messages" to send "{escaped}" '
        f'to buddy "{notify_to}" of '
        f"(first service whose service type is iMessage)"
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, timeout=15
        )
        if result.returncode == 0:
            _log(f"Notification iMessage sent → {notify_to}")
        else:
            _log(
                f"Warning: osascript exited {result.returncode}: {result.stderr.decode().strip()}"
            )
    except Exception as exc:
        _log(f"Warning: could not send iMessage ({exc})")


if __name__ == "__main__":
    main()
