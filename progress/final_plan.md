# Final Report Plan — Maximize Rubric Score

## Context

The project has a working intraday pairs-trading strategy with backtesting infrastructure, parameter search results, and multiple portfolio runs. The goal is to compile everything into a single `notebooks/final_report.ipynb` that maximizes the score across 10 rubric criteria (10 pts each). The plan covers: (A) new computation/code needed, and (B) the notebook structure.

---

## Rubric Criteria → Status

| # | Criterion | Status | Source / Gap |
|---|-----------|--------|-------------|
| 1 | Hypotheses & tests summary | Partial | theory in strategy_summary.ipynb; need compiled results |
| 2 | Constraints, Benchmarks, Objectives | Gap | no formal benchmark comparison |
| 3 | Data Description | Gap | no dedicated section |
| 4 | Indicators tested separately | Partial | formation_search.parquet exists |
| 5 | Signal process tested separately | Partial | params/params_15min.parquet IC/hit data exists |
| 6 | Rule process tested incrementally | Gap | need ablation runs |
| 7 | Parameter optimization | Partial | both parquet files + visualization needed |
| 8 | Walk-forward analysis | Gap | implement IS/OOS split |
| 9 | Overfitting assessment | Gap | IS vs OOS comparison needed |
| 10 | Extension | Partial | p-value weighting + Mahalanobis distance + Kalman filter |

---

## Part A: New Code & Computations

### A1. Incremental Rule Ablation (Criterion 6)
Run 4 backtest configs with rules added one at a time. Each takes ~10 min.

| Config | Rules active | Key param change |
|--------|-------------|-----------------|
| Base | z_entry/z_exit only | z_stop=None, max_hold=9999, min_bars=0 |
| +StopLoss | + z_stop | z_stop=4.5 |
| +MaxHold | + max_holding_minutes | max_hold=300 |
| Full | + min_bars_remaining | min_bars=8 (final strategy) |

Use period 2017-2019, 15min, 42-day formation (IS period — cleanest apples-to-apples with signal search). Save portfolio parquets with descriptive names to `results/portfolio/ablation_*.parquet`.

### A2. IS/OOS Split Validation (Criteria 8 & 9)
- **In-sample:** 2017-01-01 → 2019-12-31 (3 years — formation params fit on 2017-2018, signal params fit on 2019)
- **Out-of-sample:** 2020-01-01 → 2024-12-31 (5 years, fully held out)
  - 2020 included deliberately; call it out explicitly in analysis as a COVID stress year
- **Extended OOS:** 2025-01-01 → 2026-02-28 (recent history, ~14 months)
- Run best params on IS period to confirm selection
  - **formation:** `rolling_window_days=42,min_half_life=2,max_half_life=24,p_value_threshold=0.05`
  - **signals:** `zscore_window_days=5,z_entry=3,z_exit=2,z_stop=6,max_holding_minutes=390`
    - might need to justify `max_holding_minutes` since the grid search suggests 120. Can explain it as the grid search doesn't account for trading costs of trades that don't reverse.
- Run those SAME params unchanged on OOS and extended OOS periods
- Save to `results/portfolio/is_oos_*.parquet`
- Record IS Sharpe, OOS Sharpe (full + ex-2020), extended OOS Sharpe, IS max DD, OOS max DD

### A3. P-Value Weighted Position Sizing (Criterion 10, extension A)
Modify `strategy/backtester.py` to accept a `use_pvalue_weights: bool = False` field in `PortfolioConfig`.
When enabled: `pair_weight = (1 / p_value) / sum(1 / p_values_today)`, then `notional_pair = capital * pair_weight` instead of equal split.
Run backtest with same params as main config; compare Sharpe and drawdown.

### A4. Mahalanobis Distance Pair Selection (Criterion 10, extension B)
Add `find_distance_pairs()` to `analysis/cointegration.py` (or new `analysis/distance.py`).
- For each day, use rolling 42-day log-return matrix
- Compute pairwise Mahalanobis distance in normalized return space (Gatev et al. 2006 style)
- Select pairs with distance < threshold (analogous to ADF p-value filter)
- Apply same half-life filter as cointegration method
Run backtest with distance-based pairs; compare to cointegration-based results.

### A5. Kalman Filter Hedge Ratio (Criterion 10, extension C)
Add `compute_kalman_hedge_ratio()` to `analysis/cointegration.py`.
- Replace static OLS β with a Kalman filter tracking time-varying β
- State model: β_t = β_{t-1} + noise (random walk)
- Observation: log_spread_t = β_t × log(price_b_t) - log(price_a_t)
- Implementation: use `filterpy` library or manual scalar Kalman equations
- Compare vs static OLS: hedge ratio CV, spread stationarity, strategy Sharpe

### A6. Benchmark Construction (Criterion 2)
Two benchmark components, both computed directly in the notebook (no pre-run needed):

1. **Risk-free rate**: use the max 3-month T-bill rate observed over the full backtest period (IS + OOS) as a single constant hurdle. Source: FRED series `DTB3` or hard-code from known values. Apply uniformly across time — conservative choice (uses peak rate rather than the lower rates prevalent in 2017–2021).

2. **SPY beta**: download SPY daily returns for the same period, regress daily strategy net P&L (as return on capital) against SPY returns. Report beta, alpha (annualised), and correlation. Expected result: beta ≈ 0, validating market-neutrality and confirming risk-free rate is the correct hurdle rather than any equity benchmark.

---

## Part B: Notebook Structure — `notebooks/final_report.ipynb`

**Goal:** Single self-contained report. Each section maps to rubric criteria. All heavy computation is pre-run; notebook loads results and visualizes.

---

### Section 0: Setup
- Imports, path constants, helper display functions
- Load pre-computed results: portfolio parquets, formation_search.parquet, params_15min.parquet, trades

---

### Section 1: Strategy Summary & Hypotheses (Criterion 1)
- 1–2 paragraph narrative: what the strategy does and why it should work
- Formal hypotheses table (H1–H7):
  - H1: Within-sector cointegrated pairs exist and are identifiable
  - H2: Z-score normalization generates statistically significant entry signals
  - H3: Mean-reversion occurs within intraday horizon (≤ 6.5h)
  - H4: Stop-loss improves risk-adjusted returns
  - H5: Transaction costs <5 bps/leg remain profitable
  - H6: Parameters generalize to out-of-sample data
  - H7: P-value weighting improves risk-adjusted returns vs equal weighting
- Cross-reference table: hypothesis → section where it is tested

---

### Section 2: Constraints, Benchmarks, Objectives (Criterion 2)
- **Constraints**: sector-neutral, intraday only, no overnight positions, max N simultaneous pairs, realistic transaction costs
- **Objective function**: maximize OOS Sharpe ratio; discuss IC as alternative
- **Benchmark rationale**: long-only benchmarks (SPY, equal-weight) are inappropriate for a dollar-neutral strategy — the relevant alternative is cash. State this explicitly.
- **Primary benchmark**: risk-free rate — max 3-month T-bill rate over the full period (from A6). Conservative choice: uses peak rate rather than the lower rates in 2017–2021, so if the strategy beats it, the result is credible.
- **Market neutrality verification**: SPY beta regression (from A6) — report beta, alpha, correlation. If beta ≈ 0, this validates the benchmark choice. Show SPY alongside strategy equity curve for regime context only (COVID 2020, 2022 drawdown, 2023 rally) — not as a performance comparison.
- Benchmark comparison table: strategy Sharpe vs risk-free hurdle; strategy beta to SPY

---

### Section 3: Data Description (Criterion 3)
- Universe: ~100 large-cap US equities, 11 GICS sectors (bar chart of sector counts)
- Source: 1-min OHLCV bars resampled to 15-min / 5-min via `analysis/preprocessing.py`
- Period covered: 2017–2025
- Coverage stats: bars per ticker per year heatmap, missing bar rate
- Descriptive stats table: avg daily range, avg spread, cross-sectional correlation

---

### Section 4: Indicators — Cointegration (Criterion 4)
Test the **indicator** (Engle-Granger ADF cointegration score) independently of the strategy:
- From `formation_search.parquet` (81 combos):
  - Mean pairs/day vs rolling_window_days (line chart)
  - Sector diversity per day (avg sectors contributing ≥1 pair)
  - Half-life distribution histogram
  - Composite score heatmap: rolling_window_days × max_half_life
- Key finding: 84-day window → 30+ pairs/day; 42-day → ~17/day
- Conclusion: indicator produces stable, economically meaningful signal counts

---

### Section 5: Signal Process (Criterion 5)
Test the **z-score signal** independently of the full strategy:
- From `params/params_15min.parquet` (660 combos):
  - IC distribution histogram across all combos
  - IC t-statistic vs z_entry (scatter + regression line)
  - Hit rate vs z_entry threshold
  - Mean gross return per signal vs z_entry × z_exit heatmap
  - Breakeven cost (bps) distribution — median + range
- Key finding: statistically significant IC (t-stat > 2) for most configs
- Conclusion: predictive power exists before any position sizing or risk rules

---

### Section 6: Rule Process — Incremental Ablation (Criterion 6)
Results from A1:
- Equity curves overlaid (4 configs on same axes)
- Summary table: Sharpe, max DD, total return, win rate, avg P&L/trade
- Discussion: which rule contributes the most (expected: stop-loss is critical)
- Confirms H4: stop-loss improves risk-adjusted returns

---

### Section 7: Parameter Optimization (Criterion 7)

**7a. Formation Parameters** (from `formation_search.parquet`):
- Composite score heatmap: rolling_window_days × max_half_life
- Show why 84-day window was selected
- Trade-off: higher pair count vs slower adaptation to regime changes

**7b. Signal Parameters** (from `params/params_15min.parquet`):
- IC heatmap: z_entry × z_exit for best zscore_window
- Sensitivity of Sharpe-proxy to zscore_window_days
- Key observation: flat plateau around z_entry=3.0–3.5 → parameter stability

---

### Section 8: Walk-Forward Analysis (Criterion 8)
Results from A2:
- IS period (2017–2019): parameter selection methodology + selected params
- OOS period (2020–2024): apply IS-selected params without modification
  - Explicitly flag 2020 as COVID stress year; show sub-period stats (2020 alone vs 2021–2024)
- Extended OOS (2025–Feb 2026): apply same params to recent history
- Side-by-side table: IS vs OOS vs extended OOS — Sharpe, return, max DD, trade count
- Equity curve: IS + OOS + extended on single chart with vertical dividing lines at 2020-01 and 2025-01
- Discussion: Sharpe used as selection objective; why IC is a valid alternative (less sensitive to position sizing assumptions); impact of objective choice on selected params

---

### Section 9: Overfitting Assessment (Criterion 9)
- IS vs OOS vs extended OOS Sharpe bar chart (three bars)
- Sharpe degradation metric: (IS − OOS) / IS; note 2020 contribution separately
- From params_15min.parquet: show top-20% IS IC combos retain significantly higher OOS IC vs bottom 80% → parameter stability evidence
- Reference parameter sensitivity plateau (Section 7) as structural protection against overfitting
- Probability of Backtest Overfitting (PBO) conceptual discussion
- Look-ahead bias mitigations in implementation

---

### Section 10: Full Strategy Backtest — Main Results
- Best configuration params table
- Equity curve (2021–2024): 3 panels — portfolio value, daily net P&L, drawdown
- Trade statistics: total trades, win rate, avg P&L, avg hold time, exit reason breakdown
- Monthly returns heatmap (2021–2024)
- Benchmark comparison: strategy vs equal-weight long-only

---

### Section 11: Extensions (Criterion 10)

**11a. P-Value Weighted Position Sizing** (A3):
- Method: allocate pair notional proportional to 1/p_value (stronger cointegration → larger size)
- Compare vs equal-weight: Sharpe, max DD, trade count, win rate
- Tests H7

**11b. Mahalanobis Distance Pair Selection** (A4):
- Method: minimum distance in normalized log-return space (Gatev et al. 2006)
- Compare pairs found vs cointegration: overlap %, quality metrics
- Compare strategy Sharpe and IC
- Discussion: cointegration captures long-run equilibrium; distance captures short-run co-movement — conceptually complementary

**11c. Kalman Filter Hedge Ratio** (A5):
- Method: time-varying β via scalar Kalman filter (random-walk state model)
- Compare vs static OLS: hedge ratio CV (stability), spread stationarity, strategy Sharpe
- Discussion: adapts to structural breaks in real time; more robust to regime changes

---

### Section 12: Conclusions
- Hypothesis results table (H1–H7: Supported / Partially / Not Supported + evidence)
- Key findings (3–5 bullets)
- Limitations: data snooping risk, execution assumptions (mid-price fills), universe stability
- Future work: 5-min timeframe analysis, sector-conditional activation, multi-asset extension

---

## Critical Files

| File | Role |
|------|------|
| `notebooks/final_report.ipynb` | Output — new notebook |
| `results/formation_search.parquet` | Section 4, 7a |
| `results/params/params_15min.parquet` | Section 5, 7b, 9 |
| `results/portfolio/*.parquet` | Section 6, 8, 10 |
| `results/trades/trades_15min.parquet` | Section 10 |
| `strategy/backtester.py` | Modify for A3 (p-value weights) |
| `analysis/cointegration.py` | Add A4 (distance) and A5 (Kalman) |
| `utils/config.py` | Add `use_pvalue_weights` field |

---

## Execution Order (Checklist)

- [ ] 0. Write this plan to `progress/final_plan.md` ✓
- [ ] 1. Write `README.md`
- [ ] 2. A3: Implement p-value weighting (`PortfolioConfig.use_pvalue_weights`, backtester logic)
- [ ] 3. A4: Implement `find_distance_pairs()` in `analysis/`
- [ ] 4. A5: Implement `compute_kalman_hedge_ratio()` in `analysis/cointegration.py`
- [ ] 5. A1: Run 4 ablation backtests → `results/portfolio/ablation_*.parquet`
- [ ] 6. A2: Run IS backtest + OOS backtest → `results/portfolio/is_oos_*.parquet`
- [ ] 7. A3 run: Run p-value-weighted backtest for comparison
- [ ] 8. A4 run: Run distance-pairs backtest for comparison
- [ ] 9. A5 run: Run Kalman-hedge backtest for comparison
- [ ] 10. Write `notebooks/final_report.ipynb` sections 0–12

---

## README.md Sections

1. **Project Overview** — one paragraph on strategy
2. **Repository Structure** — annotated directory tree
3. **Setup** — `pip install -r requirements.txt`, `.env` config (`DATA_DIR`, `NOTIFY_IMESSAGE_TO`)
4. **Data Pipeline** — raw 1-min bars → resampling via `analysis/preprocessing.py`
5. **Formation Search** — `python -m scripts.formation_search --start 2017-01-01 --end 2018-12-31`
6. **Grid Search** — `python -m scripts.grid_search_v2 --phase 1 --start 2019-01-01 --end 2019-09-30`; phases 1 vs 2
7. **Running a Backtest** — `notebooks/strategy.ipynb` → configure `Config()` → `run_backtest(config=cfg)`
8. **Results** — what each subdirectory in `results/` contains
9. **Final Report** — open `notebooks/final_report.ipynb`

---

## Time Estimate

| Task | Est. Hours |
|------|-----------|
| README.md | 0.5 |
| A3 p-value weighting (code + run) | 1.5 |
| A4 Mahalanobis distance (code + run) | 3.0 |
| A5 Kalman filter (code + run) | 2.0 |
| Ablation + IS/OOS backtests (6 × 10 min) | 1.5 |
| Notebook writing (12 sections) | 5–6 |
| **Total** | **~14–15** |

If time is tight: A4 (Mahalanobis) can be described conceptually without a full backtest comparison; A5 (Kalman) same. P-value weighting (A3) is the quickest extension to fully implement.
