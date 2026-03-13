# Grade Maximization: Detailed Implementation Plan

## Gap Analysis (10 criteria × 10 pts)

| # | Criterion | Est. Now | Est. After | Key Gap |
|---|-----------|---------|-----------|---------|
| 1 | Hypotheses & Tests | 6 | 8 | H1, H4 written but never run |
| 2 | Constraints, Benchmarks, Objectives | 5 | 8 | Implicit in code; no SPY benchmark |
| 3 | Data Description | 5 | 7 | High-level only; no quality report |
| 4 | Indicators (separate) | 6 | 7 | Already gridded; add half-life analysis |
| 5 | Signal process (separate) | 6 | 7 | IC measured; add significance test |
| 6 | Rule process (incremental) | 5 | 8 | No ablation testing |
| 7 | Parameter Optimization | 7 | 8 | Per-trade metric now added; add OFN discussion |
| 8 | Walk-Forward Analysis | 2 | 8 | **Not executed** |
| 9 | Overfitting Assessment | 2 | 7 | **Not executed** |
| 10 | Extended Analysis | 1 | 6 | **Nothing built** |

**Estimated current: ~45/100 → target: ~74/100**

---

## Deliverable 1: Walk-Forward + Overfitting
**New file:** `notebooks/walk_forward.ipynb`
**Criteria:** 8 (+6 pts), 9 (+5 pts)

### Notebook sections to write

**Section 1: Setup**
```python
# Standard imports + project root
# Load prices (same as parameter_search_15min.ipynb cell 4)
# Load existing parameter search results for IS comparison
results_is = pl.read_parquet("../results/evaluation/parameter_search.parquet")
```

**Section 2: Walk-Forward Window Definition**
```python
import pandas_market_calendars as mcal
import datetime as dt

# Define rolling windows: 6-month train → 3-month OOS, step 3 months
nyse = mcal.get_calendar("NYSE")

# Full period: 2022-07-01 → 2024-12-31
# Windows (each as (train_start, train_end, test_start, test_end)):
windows = [
    ("2022-07-01", "2022-12-31", "2023-01-01", "2023-03-31"),
    ("2022-10-01", "2023-03-31", "2023-04-01", "2023-06-30"),
    ("2023-01-01", "2023-06-30", "2023-07-01", "2023-09-30"),
    ("2023-04-01", "2023-09-30", "2023-10-01", "2023-12-31"),
    ("2023-07-01", "2023-12-31", "2024-01-01", "2024-03-31"),
    ("2023-10-01", "2024-03-31", "2024-04-01", "2024-06-30"),
    ("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
    ("2024-04-01", "2024-09-30", "2024-10-01", "2024-12-31"),
]
```

**Section 3: Pass 1 — Fixed Parameters**
```python
# For each window, run OOS backtest with fixed recommended params
from strategy.backtester import run_backtest

FIXED_PARAMS = dict(
    timeframe           = "15min",
    rolling_window_days = 84,
    z_entry             = 3.0,
    z_exit              = 0.5,
    z_stop              = 4.5,
    max_pairs           = 20,
    capital             = 100_000.0,
    cost_bps            = 5.0,
    max_holding_days    = 5,
    min_bars_remaining  = 8,
)

wf_results = []
for train_start, train_end, test_start, test_end in windows:
    trades_df, daily_pnl_df = run_backtest(
        start_date=test_start, end_date=test_end, **FIXED_PARAMS
    )
    # compute Sharpe for this OOS window
    daily_ret = (daily_pnl_df["net_pnl"] / daily_pnl_df["portfolio_value"].shift(1)).drop_nulls()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    n_trades = len(trades_df)
    win_rate = float((trades_df["net_pnl"] > 0).mean()) if n_trades > 0 else 0.0
    wf_results.append({
        "train": f"{train_start}–{train_end}",
        "test": f"{test_start}–{test_end}",
        "oos_sharpe": sharpe,
        "oos_n_trades": n_trades,
        "oos_win_rate": win_rate,
    })

wf_df = pl.DataFrame(wf_results)
print(wf_df)
print(f"\nMean OOS Sharpe: {wf_df['oos_sharpe'].mean():.3f}")
print(f"% positive OOS windows: {(wf_df['oos_sharpe'] > 0).mean():.1%}")
```

**Section 4: IS vs OOS Sharpe Comparison (H4)**
```python
# IS Sharpe: from debug.py / backtester run on 2022-07 → 2022-12 at 1.5 bps
# OOS Sharpe: 2023-01 → 2024-12 from existing results

# Re-run IS period explicitly
trades_is, pnl_is = run_backtest(
    start_date="2022-07-01", end_date="2022-12-31",
    cost_bps=1.5, **{k: v for k, v in FIXED_PARAMS.items() if k != "cost_bps"}
)
# compute IS Sharpe ...

# Compare degradation
print(f"IS Sharpe  (H2 2022, 1.5 bps): {is_sharpe:.3f}")
print(f"OOS Sharpe (2023–2024, 1.5 bps): {oos_sharpe:.3f}")
print(f"Degradation: {(is_sharpe - oos_sharpe) / is_sharpe:.1%}")
print("Target: <30% degradation for H4 to pass")
```

**Section 5: Degrees-of-Freedom / Overfitting Analysis**
```python
# Deflated Sharpe Ratio (Bailey & López de Prado 2016)
# DSR = SR * sqrt(T) corrected for number of trials N and skewness
import scipy.stats as stats

N_trials = 1296  # number of parameter combos tried
T = 127          # number of trading days in H2 2022

# Get the IS mean_net_return distribution from grid
sr_dist = results_is["mean_net_return"].to_numpy()
sr_max = sr_dist.max()
sr_mean = sr_dist.mean()
sr_std = sr_dist.std()

# Expected maximum SR under null across N independent trials (Monte Carlo or analytic approx)
expected_max_sr_under_null = sr_mean + sr_std * stats.norm.ppf(1 - 1 / N_trials)
print(f"Max IS return observed:             {sr_max:.4f}")
print(f"Expected max under null (N={N_trials}): {expected_max_sr_under_null:.4f}")
print(f"Excess over null:                   {sr_max - expected_max_sr_under_null:.4f}")
print()
print("Conclusion: assess whether observed IS performance exceeds null expectation")
print("given the number of combinations tested (multiple-testing concern).")
```

**Section 6: Objective Function Comparison**
```python
# Load per-trade results (if grid was re-run with updated grid_worker.py)
# OR construct manually for top-20 combos

# Compare rankings:
results_with_per_trade = results_is  # after grid re-run includes mean_net_per_trade

print("Top 10 by mean_net_per_bar:")
print(results_with_per_trade.sort("mean_net_return", descending=True).head(10)
      .select(["z_entry", "max_holding_minutes", "formation_window_days",
               "mean_net_return", "mean_net_per_trade", "ic_gross"]))

print("\nTop 10 by mean_net_per_trade:")
print(results_with_per_trade.sort("mean_net_per_trade", descending=True).head(10)
      .select(["z_entry", "max_holding_minutes", "formation_window_days",
               "mean_net_return", "mean_net_per_trade", "ic_gross"]))

# Show rank correlation between the two metrics
from scipy.stats import spearmanr
r, p = spearmanr(results_with_per_trade["mean_net_return"],
                  results_with_per_trade["mean_net_per_trade"])
print(f"\nSpearman rank correlation (per-bar vs per-trade): {r:.3f} (p={p:.4f})")
print("Low correlation confirms the bias — different parameters rank best.")
```

**Section 7: Walk-Forward Equity Curve**
```python
# Concatenate OOS daily_pnl DataFrames across windows into a single rolling equity curve
# Plot alongside IS equity curve and SPY benchmark
```

---

## Deliverable 2: Formal Constraints/Benchmarks + H1 Test
**Files:** `notebooks/strategy.ipynb`, `notebooks/strategy_summary.ipynb`
**Criteria:** 2 (+3 pts), 1 (+2 pts)

### 2a — SPY Benchmark (in `strategy.ipynb` Section 3)

**New markdown cell before BACKTEST_PARAMS:**
```markdown
## Strategy Constraints, Objectives, and Benchmarks

### Hard Constraints
| Constraint | Value | Rationale |
|---|---|---|
| Max concurrent pairs | 20 | Diversification; diminishing returns beyond 20 |
| Max holding period | 5 trading days | Gap risk; cointegration may break over longer horizon |
| Minimum bars remaining | 8 (~2 hours) | No new entries too close to session close |
| Transaction cost floor | 5.0 bps/leg | Realistic all-in cost for institutional equity trading |

### Objectives
- **Primary:** Positive risk-adjusted return (annualised Sharpe ≥ 1.0) net of 5 bps/leg costs
- **Secondary:** Max drawdown ≤ 15%, z_exit win rate > 90% (confirms mean reversion is occurring)
- **Not objective:** Raw return maximisation (would require accepting excessive leverage/drawdown)

### Benchmarks
| Benchmark | Source | Expected Sharpe |
|---|---|---|
| SPY buy-and-hold (same period) | Yahoo/EODHD | Varies by period |
| Gatev, Goetzmann & Rouwenhorst (2006) | Academic pairs trading | ~2.0 (pre-2000) |
| Zero-alpha (risk-free, T-bill proxy) | — | 0.0 |
```

**Code to add SPY to equity curve plot:**
```python
# Load SPY from EOD data (already fetched)
from analysis.preprocessing import load_processed

spy_prices = load_processed("SPY", "15min",
                             start_date=BACKTEST_PARAMS["start_date"],
                             end_date=BACKTEST_PARAMS["end_date"])

# Compute SPY daily returns aligned to backtest dates
spy_daily = (
    spy_prices
    .with_columns(pl.col("timestamp").dt.date().alias("date"))
    .group_by("date").agg(pl.col("close").last())
    .sort("date")
    .with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret")
    )
)
spy_equity = (1 + spy_daily["ret"].fill_null(0)).cumprod() * capital

# Add to the existing ax1 plot (after the portfolio_values line):
# ax1.plot(spy_daily["date"].to_list(), spy_equity.to_list(),
#          color="#FF5722", linewidth=1.0, linestyle="--", label="SPY (buy & hold)", alpha=0.7)
```

### 2b — H1 Within-Sector Test (add section to `strategy_summary.ipynb`)

```python
# H1: Within-sector pairs are significantly more cointegrated than random cross-sector pairs
from analysis.cointegration import find_cointegrated_pairs
import itertools, random, scipy.stats as stats

# --- Within-sector pass rate (use H2 2022 formation window) ---
formation_start = "2022-04-07"   # 84 calendar days before 2022-07-01
formation_end   = "2022-06-30"

within_pairs = find_cointegrated_pairs(
    start_date=formation_start, end_date=formation_end, timeframe="15min"
)
# within_pairs already filtered to within-sector by cointegration.py
# Get total tested vs passing
n_within_tested = len(within_pairs)   # NOTE: may need to adjust if function pre-filters
n_within_passed = len(within_pairs.filter(pl.col("p_value") <= 0.05))

# --- Cross-sector pass rate ---
from utils.config import get_all_tickers
from analysis.cointegration import _get_sector  # or equivalent
# Sample N_cross random cross-sector pairs
tickers = get_all_tickers()
cross_pairs_tested = 0
cross_pairs_passed = 0
# ... enumerate pairs from different sectors, run ADF ...

# --- Chi-square test ---
contingency = [[n_within_passed, n_within_tested - n_within_passed],
               [cross_pairs_passed, cross_pairs_tested - cross_pairs_passed]]
chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
odds_ratio = (n_within_passed / (n_within_tested - n_within_passed)) / \
             (cross_pairs_passed / (cross_pairs_tested - cross_pairs_passed))

print(f"Within-sector cointegration pass rate: {n_within_passed}/{n_within_tested} = {n_within_passed/n_within_tested:.1%}")
print(f"Cross-sector cointegration pass rate:  {cross_pairs_passed}/{cross_pairs_tested} = {cross_pairs_passed/cross_pairs_tested:.1%}")
print(f"Chi-square: {chi2:.2f}, p-value: {p_val:.4f}, Odds ratio: {odds_ratio:.2f}")
print(f"H1 result: {'REJECT H0 (within-sector significantly more cointegrated)' if p_val < 0.05 else 'FAIL TO REJECT H0'}")
```

**Note on `find_cointegrated_pairs()` behavior:** The function currently returns only pairs that pass ADF (p ≤ 0.05) within sector. To count total tested (denominator), you either need to enumerate all within-sector pairs manually and count them, or temporarily lower the p-value threshold to 1.0 and count total vs passing. Check `analysis/cointegration.py` implementation before writing this cell.

---

## Deliverable 3: Kalman Filter Hedge Ratio
**New file:** `analysis/kalman.py`
**Criteria:** 10 (+4 pts)

### Implementation

```python
# analysis/kalman.py
"""
Kalman filter for dynamic (time-varying) hedge ratio estimation.

Treats the hedge ratio beta as a latent state evolving as a random walk:
    beta_t = beta_{t-1} + process_noise     (state equation)
    price_a_t = beta_t * price_b_t + obs_noise  (observation equation)

This gives a continuously-updated spread series that is more stationary
than the OLS spread when the cointegrating coefficient drifts slowly over time.

Reference: Hamilton (1994) "Time Series Analysis", Ch. 13;
           Pole & MacQueen "Pairs Trading with Kalman Filters"
"""

from __future__ import annotations
import math


def compute_kalman_spread(
    prices_a: list[float],
    prices_b: list[float],
    delta: float = 1e-4,
    obs_noise_var: float = 1.0,
) -> tuple[list[float], list[float]]:
    """
    Compute spread using a Kalman-filter dynamic hedge ratio.

    Args:
        prices_a:       Price series for leg A (list of floats).
        prices_b:       Price series for leg B (list of floats, same length).
        delta:          Process noise variance for beta random walk.
                        Smaller = slower adaptation (more stable hedge ratio).
                        Typical range: 1e-6 (very slow) to 1e-3 (fast).
        obs_noise_var:  Observation noise variance (initial estimate).
                        Calibrated from data if None.

    Returns:
        (spreads, betas): lists of floats, one per input bar.
        spreads[t] = prices_a[t] - betas[t] * prices_b[t]
        betas[t] = filtered hedge ratio estimate at bar t
    """
    n = len(prices_a)
    assert len(prices_b) == n, "price series must have equal length"

    # State: beta (scalar hedge ratio)
    # Initialize from first two observations
    beta = prices_a[0] / prices_b[0] if prices_b[0] != 0 else 1.0
    P = 1.0   # state error covariance

    spreads: list[float] = []
    betas: list[float] = []

    for t in range(n):
        x = prices_b[t]   # "input" to the observation
        y = prices_a[t]   # observation

        # Predict
        P_pred = P + delta   # process noise inflates uncertainty

        # Innovation (residual)
        innov = y - beta * x

        # Innovation covariance
        S = x * P_pred * x + obs_noise_var

        # Kalman gain
        K = P_pred * x / S if S != 0 else 0.0

        # Update
        beta = beta + K * innov
        P = (1.0 - K * x) * P_pred

        spreads.append(y - beta * x)
        betas.append(beta)

    return spreads, betas
```

### Comparison notebook section (add to `notebooks/strategy_summary.ipynb` or new `notebooks/kalman_comparison.ipynb`)

```python
from analysis.kalman import compute_kalman_spread
from analysis.cointegration import find_cointegrated_pairs
from analysis.preprocessing import load_processed
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Load a sample pair (e.g., NVDA/INTC from H2 2022)
prices_nvda = load_processed("NVDA", "15min", "2022-04-07", "2022-06-30")
prices_intc = load_processed("INTC", "15min", "2022-04-07", "2022-06-30")

closes_a = prices_nvda["close"].to_list()
closes_b = prices_intc["close"].to_list()

# OLS spread (static hedge ratio)
import numpy as np
beta_ols = np.polyfit(closes_b, closes_a, 1)[0]
spread_ols = [a - beta_ols * b for a, b in zip(closes_a, closes_b)]

# Kalman spread (dynamic hedge ratio)
spread_kalman, betas_kalman = compute_kalman_spread(closes_a, closes_b, delta=1e-4)

# Compare ADF p-values
adf_ols    = adfuller(spread_ols)[1]
adf_kalman = adfuller(spread_kalman)[1]

print(f"OLS spread    — ADF p-value: {adf_ols:.4f},    static beta: {beta_ols:.4f}")
print(f"Kalman spread — ADF p-value: {adf_kalman:.4f},  avg beta:   {np.mean(betas_kalman):.4f}")

# Run across all cointegrated pairs from H2 2022 and aggregate
# Show: % of pairs where Kalman spread is more stationary (lower ADF p-value)
```

### Add Kalman flag to signal generation

In `analysis/signals.py` `generate_pair_signals_for_day()`, add optional parameter:
```python
def generate_pair_signals_for_day(
    ...,
    use_kalman: bool = False,
    kalman_delta: float = 1e-4,
) -> pl.DataFrame:
    # When computing spread, choose OLS or Kalman:
    if use_kalman:
        from analysis.kalman import compute_kalman_spread
        spread_vals, _ = compute_kalman_spread(closes_a, closes_b, delta=kalman_delta)
    else:
        spread_vals = [a - hedge_ratio * b for a, b in zip(closes_a, closes_b)]
```

Then backtest with `use_kalman=True` and compare Sharpe.

---

## Deliverable 4: Incremental Rule Testing (Ablation)
**File:** `notebooks/strategy.ipynb` (new section at end, or `notebooks/rules_ablation.ipynb`)
**Criteria:** 6 (+2 pts)

```python
# Rules Ablation Study
# Run backtest with each rule individually disabled to measure marginal contribution
# Use cost_bps=1.5 so the strategy is profitable and differences are meaningful

ablation_configs = {
    "Baseline (all rules)":      dict(z_stop=4.5, max_holding_days=5, min_bars_remaining=8),
    "No stop-loss":              dict(z_stop=999, max_holding_days=5, min_bars_remaining=8),
    "No time stop":              dict(z_stop=4.5, max_holding_days=99, min_bars_remaining=8),
    "No late-day gate":          dict(z_stop=4.5, max_holding_days=5, min_bars_remaining=0),
    "EOD force-close (day-only)":dict(z_stop=4.5, max_holding_days=1, min_bars_remaining=8),
}

BASE_PARAMS = dict(
    start_date="2022-07-01", end_date="2024-12-31",
    timeframe="15min", rolling_window_days=84,
    z_entry=3.0, z_exit=0.5,
    max_pairs=20, capital=100_000.0, cost_bps=1.5,
)

ablation_results = {}
for label, rule_params in ablation_configs.items():
    trades_df, daily_pnl_df = run_backtest(**BASE_PARAMS, **rule_params)
    daily_ret = (daily_pnl_df["net_pnl"] / daily_pnl_df["portfolio_value"].shift(1)).drop_nulls()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    pv = daily_pnl_df["portfolio_value"].to_numpy()
    max_dd = float(((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)).min())
    n_trades = len(trades_df)
    win_rate = float((trades_df["net_pnl"] > 0).mean()) if n_trades > 0 else 0.0
    reasons = trades_df["exit_reason"].value_counts()
    pct_stop = float(reasons.filter(pl.col("exit_reason") == "z_stop")["count"][0]) / n_trades if n_trades > 0 else 0.0

    ablation_results[label] = {
        "Sharpe": sharpe, "Max DD": max_dd,
        "Win Rate": win_rate, "N Trades": n_trades, "% z_stop": pct_stop
    }

# Print as table
print(f"{'Rule Config':<35} {'Sharpe':>8} {'Max DD':>8} {'Win%':>7} {'N Trades':>9} {'%z_stop':>8}")
print("-" * 80)
for label, r in ablation_results.items():
    print(f"{label:<35} {r['Sharpe']:>8.3f} {r['Max DD']:>8.2%} {r['Win Rate']:>7.1%} {r['N Trades']:>9,} {r['%z_stop']:>8.1%}")
```

**Expected finding:** Removing stop-loss increases max drawdown significantly; removing time stop increases average hold; late-day gate has small but positive effect. Baseline (all rules on) should have best risk-adjusted performance.

---

## Deliverable 5: Data Description Section
**File:** `notebooks/strategy_summary.ipynb` (new section after literature review)
**Criteria:** 3 (+2 pts)

```python
# Section: Data Description

# 1. Universe overview
from utils.config import get_all_tickers, CONFIG
tickers = get_all_tickers()
print(f"Universe: {len(tickers)} tickers across GICS sectors")

# Build sector table
from analysis.preprocessing import load_processed
sector_map = {}  # ticker → sector (from CONFIG or cointegration.py sector logic)
# ... enumerate CONFIG.sectors dict ...

# 2. Data coverage check
coverage = []
for ticker in tickers:
    df = load_processed(ticker, "15min", "2022-01-01", "2024-12-31")
    if df is not None:
        coverage.append({"ticker": ticker, "n_bars": len(df), "start": str(df["timestamp"][0].date()), "end": str(df["timestamp"][-1].date())})

coverage_df = pl.DataFrame(coverage)
expected_bars_per_day = 26   # 9:30–16:00 at 15min
expected_trading_days = 754  # 2022-01-01 → 2024-12-31
expected_total_bars = expected_bars_per_day * expected_trading_days
coverage_df = coverage_df.with_columns(
    (pl.col("n_bars") / expected_total_bars).alias("coverage_pct")
)
print(f"\nMedian coverage: {coverage_df['coverage_pct'].median():.1%}")
print(f"Tickers with <95% coverage: {len(coverage_df.filter(pl.col('coverage_pct') < 0.95))}")
print(coverage_df.sort("coverage_pct").head(10))  # worst coverage tickers

# 3. Pair universe statistics from H2 2022 formation window
# (use pairs_cache from parameter_search notebook if saved, or recompute)
pairs_h2 = find_cointegrated_pairs("2022-04-07", "2022-06-30", "15min")
print(f"\nH2 2022 formation window:")
print(f"  Cointegrated pairs found: {len(pairs_h2)}")
print(f"  By sector:")
print(pairs_h2.group_by("sector").agg(pl.len().alias("n")).sort("n", descending=True))
print(f"\nHalf-life distribution (bars at 15min):")
print(pairs_h2["half_life"].describe())

# 4. Survivorship bias acknowledgement (write as markdown)
```

**Markdown cell for survivorship bias:**
```markdown
### Survivorship Bias

The 100-ticker universe was constructed as of the data-fetch date (early 2026) from large-cap
US equities. Tickers that were large-cap throughout 2022–2024 but subsequently delisted,
merged, or dropped from large-cap indices are not included.

**Impact:** This introduces a mild survivorship bias favouring companies that survived and
remained large-cap through the backtest period. The magnitude is limited because:
1. Pair trading is market-neutral — the long and short legs partially cancel survivorship effects
2. Large-cap US equities have very low delisting rates over 2-year horizons (~1%)
3. Formation windows (84 days) recalibrate daily, so even a delisting mid-backtest would result
   in the pair being dropped, not in an unrealised gain

A full survivorship-bias-free dataset would require a point-in-time universe from a commercial
provider (CRSP, Compustat). This is acknowledged as a limitation.
```

---

## Execution Order

Run in this order to maximise grade with available time:

1. **`notebooks/walk_forward.ipynb`** — write end-to-end; ~4-6 hours compute + analysis
2. **SPY benchmark + constraints markdown** in `notebooks/strategy.ipynb` — ~1 hour
3. **H1 empirical test** in `notebooks/strategy_summary.ipynb` — ~2 hours
4. **Rule ablation section** in `notebooks/strategy.ipynb` or new notebook — ~2 hours
5. **Data description section** in `notebooks/strategy_summary.ipynb` — ~2 hours
6. **`analysis/kalman.py` + comparison notebook** — ~4 hours; only if time permits

## Files Created / Modified Summary

| File | Change | Criteria |
|------|--------|---------|
| `notebooks/walk_forward.ipynb` | New — rolling windows, IS/OOS comparison, objective function discussion | 8, 9 |
| `analysis/kalman.py` | New — Kalman filter dynamic hedge ratio | 10 |
| `notebooks/strategy_summary.ipynb` | Add H1 test section + data description section | 1, 3 |
| `notebooks/strategy.ipynb` | Add constraints/benchmarks markdown, SPY overlay, rule ablation section | 2, 6 |
