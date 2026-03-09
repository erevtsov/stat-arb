"""
Formation parameter search — Stage A quality sweep.

Evaluates each (rolling_window_days, min_half_life, max_half_life, p_value_threshold)
combination on pair quality metrics WITHOUT running any signal backtest:

  mean_n_pairs          — avg pairs/day passing all filters
  split_consistency_rate — fraction of pairs where split_consistent=True
  mean_n_sectors        — avg distinct sectors contributing at least 1 pair
  mean_hedge_ratio_cv   — avg hedge ratio CV (lower = more stable relationship)
  mean_half_life        — avg half-life in bars
  pct_days_with_pairs   — fraction of eval days that have ≥ 1 pair

After running this script, pick combos that satisfy minimum quality thresholds
(see QUALITY_THRESHOLDS below) and pass them to grid_search_v2.py Phase 2 for
the full signal sweep.

Usage:
    python -m scripts.formation_search \\
        --start 2022-07-01 --end 2022-09-30 --timeframe 15min

Output:
    results/formation_search.parquet   (one row per formation combo)
    Ranked summary table printed to stdout
"""

from __future__ import annotations

import argparse
import itertools
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from analysis.cointegration import find_cointegrated_pairs
from analysis.preprocessing import load_processed
from utils.config import CONFIG, get_all_tickers

# ---------------------------------------------------------------------------
# Formation parameter grid
# ---------------------------------------------------------------------------

FORMATION_GRID: dict[str, list] = {
    "rolling_window_days": [21, 42, 84],
    "min_half_life": [2, 4, 8],
    "max_half_life": [12, 24, 48],
    "p_value_threshold": [0.005, 0.01, 0.05],
}
# 3 × 3 × 3 × 2 = 54 combos

# ---------------------------------------------------------------------------
# Minimum quality thresholds for advancing to Stage B (signal sweep)
# ---------------------------------------------------------------------------

QUALITY_THRESHOLDS: dict[str, float] = {
    "mean_n_pairs": 5.0,
    "pct_days_with_pairs": 0.50,
    "split_consistency_rate": 0.40,
}


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def load_all_prices(
    timeframe: str, start_date: str, end_date: str
) -> dict[str, pl.DataFrame]:
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
# Per-combo evaluation
# ---------------------------------------------------------------------------


def _eval_formation_combo(
    combo: dict,
    eval_dates: list[date],
    price_cache: dict,
    timeframe: str,
) -> dict:
    """
    Evaluate one formation parameter combination across all eval dates.

    For each date, forms the rolling window [d - rolling_window_days, d-1],
    finds cointegrated pairs (with stability metrics), then aggregates quality
    metrics across all dates.
    """
    rolling_window_days: int = combo["rolling_window_days"]
    min_hl: float = combo["min_half_life"]
    max_hl: float = combo["max_half_life"]
    p_thresh: float = combo["p_value_threshold"]

    day_n_pairs: list[int] = []
    day_split_rates: list[float] = []
    day_n_sectors: list[int] = []
    day_hr_cvs: list[float] = []
    day_half_lives: list[float] = []

    for d in eval_dates:
        formation_start = (d - timedelta(days=rolling_window_days)).isoformat()
        formation_end = (d - timedelta(days=1)).isoformat()

        pairs_df = find_cointegrated_pairs(
            timeframe=timeframe,
            start_date=formation_start,
            end_date=formation_end,
            p_value_threshold=p_thresh,
            min_half_life=min_hl,
            max_half_life=max_hl,
            price_cache=price_cache,
        )

        n_pairs = len(pairs_df)
        day_n_pairs.append(n_pairs)

        if n_pairs == 0:
            day_split_rates.append(0.0)
            day_n_sectors.append(0)
            day_hr_cvs.append(float("nan"))
            day_half_lives.append(float("nan"))
            continue

        # Split-window consistency rate
        if "split_consistent" in pairs_df.columns:
            n_consistent = pairs_df["split_consistent"].sum()
            day_split_rates.append(float(n_consistent) / n_pairs)
        else:
            day_split_rates.append(float("nan"))

        # Number of distinct sectors with at least 1 pair
        if "sector" in pairs_df.columns:
            n_sectors = pairs_df["sector"].n_unique()
        else:
            n_sectors = 0
        day_n_sectors.append(n_sectors)

        # Mean hedge ratio CV (skip NaN)
        if "hedge_ratio_cv" in pairs_df.columns:
            cv_vals = pairs_df["hedge_ratio_cv"].drop_nulls().to_numpy()
            finite_cvs = cv_vals[np.isfinite(cv_vals)]
            day_hr_cvs.append(
                float(np.mean(finite_cvs)) if len(finite_cvs) > 0 else float("nan")
            )
        else:
            day_hr_cvs.append(float("nan"))

        # Mean half-life
        if "half_life" in pairs_df.columns:
            hl_vals = pairs_df["half_life"].drop_nulls().to_numpy()
            finite_hls = hl_vals[np.isfinite(hl_vals)]
            day_half_lives.append(
                float(np.mean(finite_hls)) if len(finite_hls) > 0 else float("nan")
            )
        else:
            day_half_lives.append(float("nan"))

    # Aggregate across dates
    arr_n_pairs = np.array(day_n_pairs, dtype=float)
    arr_split = np.array(day_split_rates, dtype=float)
    arr_sectors = np.array(day_n_sectors, dtype=float)
    arr_cvs = np.array(day_hr_cvs, dtype=float)
    arr_hls = np.array(day_half_lives, dtype=float)

    n_days = len(eval_dates)
    pct_days_with_pairs = float(np.mean(arr_n_pairs > 0))

    def _nanmean(a: np.ndarray) -> float:
        finite = a[np.isfinite(a)]
        return float(np.mean(finite)) if len(finite) > 0 else float("nan")

    mean_n_pairs = float(np.mean(arr_n_pairs))
    split_consistency_rate = (
        _nanmean(arr_split[arr_n_pairs > 0]) if (arr_n_pairs > 0).any() else 0.0
    )
    mean_n_sectors = (
        _nanmean(arr_sectors[arr_n_pairs > 0]) if (arr_n_pairs > 0).any() else 0.0
    )
    mean_hedge_ratio_cv = _nanmean(arr_cvs)
    mean_half_life = _nanmean(arr_hls)

    # Composite quality score: rewards stability × capacity × sector breadth
    # log(max(n_pairs,1)) prevents log(0); all terms normalised to comparable scale
    composite_score = (
        (split_consistency_rate * np.log(max(mean_n_pairs, 1.0)) * mean_n_sectors)
        if not any(
            np.isnan(x) for x in [split_consistency_rate, mean_n_pairs, mean_n_sectors]
        )
        else 0.0
    )

    return {
        **combo,
        "n_days_evaluated": n_days,
        "mean_n_pairs": round(mean_n_pairs, 2),
        "pct_days_with_pairs": round(pct_days_with_pairs, 3),
        "split_consistency_rate": round(split_consistency_rate, 3),
        "mean_n_sectors": round(mean_n_sectors, 2),
        "mean_hedge_ratio_cv": round(mean_hedge_ratio_cv, 4)
        if np.isfinite(mean_hedge_ratio_cv)
        else None,
        "mean_half_life": round(mean_half_life, 2)
        if np.isfinite(mean_half_life)
        else None,
        "composite_score": round(float(composite_score), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Formation parameter quality sweep")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument(
        "--start", default="2022-07-01", help="Eval start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2022-09-30", help="Eval end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        default=str(Path(CONFIG.paths.results_dir) / "formation_search.parquet"),
    )
    args = parser.parse_args()

    eval_start = date.fromisoformat(args.start)
    eval_end = date.fromisoformat(args.end)

    # Build formation combos
    f_keys = list(FORMATION_GRID.keys())
    formation_combos = [
        dict(zip(f_keys, vals)) for vals in itertools.product(*FORMATION_GRID.values())
    ]
    print(f"Formation combos: {len(formation_combos)}")
    print(f"Eval period: {eval_start} → {eval_end} | timeframe: {args.timeframe}")

    # Eval dates: all calendar days in range
    eval_dates: list[date] = []
    d = eval_start
    while d <= eval_end:
        eval_dates.append(d)
        d += timedelta(days=1)

    # Load prices: cover max formation window + eval range
    max_window = max(c["rolling_window_days"] for c in formation_combos)
    data_start = (eval_start - timedelta(days=max_window + 5)).isoformat()
    data_end = eval_end.isoformat()

    print(f"Loading prices [{data_start} → {data_end}]...")
    price_cache = load_all_prices(args.timeframe, data_start, data_end)
    print(f"Loaded {len(price_cache)} tickers\n")

    # Evaluate each combo
    all_results: list[dict] = []
    for i, combo in enumerate(formation_combos, 1):
        print(f"[{i:2d}/{len(formation_combos)}] {combo}")
        result = _eval_formation_combo(combo, eval_dates, price_cache, args.timeframe)
        all_results.append(result)
        print(
            f"        n_pairs={result['mean_n_pairs']:.1f}  "
            f"split_rate={result['split_consistency_rate']:.2%}  "
            f"sectors={result['mean_n_sectors']:.1f}  "
            f"hr_cv={result['mean_hedge_ratio_cv']}  "
            f"score={result['composite_score']:.4f}"
        )

    results_df = pl.DataFrame(all_results).sort("composite_score", descending=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(str(output_path))
    print(f"\nSaved {len(results_df)} rows → {output_path}")

    # --- Summary ---
    print("\n" + "=" * 75)
    print("RANKED FORMATION COMBOS (by composite_score)")
    print("=" * 75)
    print(results_df.to_pandas().to_string(index=False))

    # --- Survivors ---
    survivors = results_df
    for col, thresh in QUALITY_THRESHOLDS.items():
        survivors = survivors.filter(pl.col(col) >= thresh)
    print(f"\n{'=' * 75}")
    print(f"COMBOS PASSING QUALITY THRESHOLDS ({len(survivors)}/{len(results_df)})")
    print(f"  min mean_n_pairs          ≥ {QUALITY_THRESHOLDS['mean_n_pairs']}")
    print(
        f"  min pct_days_with_pairs   ≥ {QUALITY_THRESHOLDS['pct_days_with_pairs']:.0%}"
    )
    print(
        f"  min split_consistency_rate ≥ {QUALITY_THRESHOLDS['split_consistency_rate']:.0%}"
    )
    print("=" * 75)
    if len(survivors) > 0:
        print(survivors.to_pandas().to_string(index=False))
        print(
            "\n→ Pass these formation combos to grid_search_v2.py Phase 2 (update FORMATION_GRID)."
        )
    else:
        print("No combos passed all thresholds. Consider relaxing QUALITY_THRESHOLDS.")


if __name__ == "__main__":
    main()
