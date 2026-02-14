# Statistical Power and Data Requirements Analysis

## Executive Summary

To achieve statistically significant results across all 10 priority hypotheses, we need:

- **Minimum Time Period**: 18-24 months of intraday data
- **Current Data**: 2 years (2023-2024) ✓ **SUFFICIENT**
- **Minimum Observations per Pair**: 240,000 bars (6 months of 1-min data)
- **Minimum Number of Trades**: 200-300 per hypothesis test
- **Universe Size**: 100 stocks (current) ✓ **SUFFICIENT**
- **Walk-Forward Windows**: 6-8 windows minimum

---

## 1. Cointegration Testing (H1, H2, H3)

### H1: Within-Sector vs. Cross-Sector Cointegration

**Required Data**:
- **Time period for cointegration test**: 6-12 months (formation period)
- **Minimum observations**: 240,000 1-minute bars per pair (≈6 months)
  - Daily data: 120-250 observations minimum (MacKinnon 1991)
  - Intraday 1-min: 240,000 observations (6 months × 252 days × 6.5 hours × 60 min)
  - **Current status**: ✓ We have 2 years (2023-2024)

**Statistical Power**:
```
Number of within-sector pairs: C(100,2) filtered by sector ≈ 400-500 pairs
Number of cross-sector random pairs: 1,000 sample
Chi-square test power: n_pairs ≥ 100 for 80% power at α=0.05
Current sample: 400-500 within + 1,000 cross = ✓ SUFFICIENT
```

**Critical Constraint**: ADF test requires minimum 60-120 observations for reliable critical values
- Daily data: 3-6 months minimum
- 1-min data: effectively ∞ observations (use 5-min or 15-min aggregation for stability)

### H2: Half-Life Stability

**Required Data**:
- **Number of rolling windows**: ≥ 6 non-overlapping windows
  - Window size: 6 months each
  - Total period required: 36 months for CV calculation
  - **Current status**: ⚠️ 24 months available → **4 windows** (marginal)

**Statistical Power**:
```
Coefficient of Variation (CV) = σ/μ
Reliable CV estimation: n ≥ 30 (rule of thumb)
Per-pair windows: 4 (current) → CV estimate unreliable
Solution: Use overlapping windows (monthly roll) → 18 windows ✓ ACCEPTABLE
```

**Recommendation**: Use 3-month rolling windows with 1-month step
- Windows from 24 months: 24 - 3 + 1 = 22 windows ✓ SUFFICIENT

### H3: P-Value as Quality Signal

**Required Data**:
- **Pairs per quintile**: ≥ 20 pairs (for 5 quintiles = 100 total pairs)
  - Current universe: 100 stocks → ~400-500 pairs → 80-100 pairs/quintile ✓ SUFFICIENT
- **Trades per quintile**: ≥ 30 trades for reliable Sharpe ratio
  - Expected trades/pair/year: 20-50 (intraday)
  - Trades per quintile: 80 pairs × 30 trades = 2,400 trades ✓ SUFFICIENT

---

## 2. Parameter Optimization (H5, H9)

### H5: Z-Score Entry Threshold

**Required Data**:
- **Grid search points**: 5 thresholds {1.5, 2.0, 2.5, 3.0, 3.5}
- **Minimum trades per threshold**: 200 trades (for Sharpe SE < 0.15)
  - Sharpe ratio standard error: SE = √((1 + 0.5×Sharpe²) / n)
  - For Sharpe = 2.0, SE(n=200) = 0.11 ✓ ACCEPTABLE
  - For Sharpe = 2.0, SE(n=100) = 0.16 (marginal)

**Statistical Power for Bootstrap**:
```
Bootstrap iterations: 1,000
Minimum observations: 200 trades per threshold
Current expected: 300-500 trades/year × 2 years = 600-1,000 trades
Per threshold distribution:
  z=1.5: ~40% of trades = 240-400 trades ✓
  z=2.0: ~30% of trades = 180-300 trades ✓
  z=2.5: ~20% of trades = 120-200 trades ⚠️ (marginal)
  z=3.0: ~10% of trades = 60-100 trades ✗ (insufficient for bootstrap)
  z=3.5: ~5% of trades = 30-50 trades ✗ (insufficient)
```

**Current Status**: ⚠️ Higher thresholds (3.0, 3.5) may have insufficient trades
**Solution**: Pool 2 years of data or extend to 3 years for rare thresholds

### H9: Z-Score Window Size

**Required Data**:
- **Window sizes to test**: {30, 60, 90, 120} bars
- **Minimum observations**: Must have ≥ 3× max_window before first signal
  - Max window = 120 bars → need 360 bars warmup (6 hours)
  - Per day usable: 390 - 360 = 30 minutes (not acceptable for 1-min)
  - **Solution**: Use daily warmup, test intraday

**Statistical Power**:
- Same trade count requirements as H5
- Expected sufficient data ✓

---

## 3. Transaction Costs (H12, H13)

### H12: Cost Sensitivity

**Required Data**:
- **Cost levels**: {5, 10, 15, 20, 25, 30} bps
- **Trades per cost level**: Same backtest, different accounting
- **Minimum trades**: 200-300 for reliable Sharpe
  - **Current status**: ✓ SUFFICIENT (same 600-1,000 trades)

**Statistical Power**:
```
Linear regression: Sharpe ~ Cost
R² determination: n_points = 6 (cost levels)
Coefficient significance: t-test with df = 4
Power: Moderate (small sample but strong effect expected)
Recommendation: Plot full curve, don't rely solely on regression
```

---

## 4. Robustness Testing (H16, H17, H18)

### H16: In-Sample vs. Out-of-Sample

**Required Data**:
- **Split**: 60% IS (14.4 months) / 40% OOS (9.6 months)
  - Current: 24 months × 0.6 = 14.4 months IS ✓
  - Current: 24 months × 0.4 = 9.6 months OOS ✓

**Statistical Power**:
```
Minimum trades in OOS: 200 trades
Expected OOS trades: 9.6 months × 30 trades/month = 288 trades ✓ SUFFICIENT

Sharpe ratio comparison (paired t-test):
  Null: Sharpe_IS = Sharpe_OOS
  Alternative: Sharpe_IS > Sharpe_OOS
  Power for d=0.5 (medium effect), n=20 (windows): 70%
  Power for d=0.8 (large effect), n=20 (windows): 95% ✓
```

### H17: Walk-Forward Robustness

**Required Data**:
- **Window structure**: 6-month IS, 3-month OOS, 1-month step
- **Number of windows**: (24 - 9) / 1 + 1 = 16 windows ✓ GOOD

**Statistical Power**:
```
Minimum windows for consistency test: 10 windows
Current: 16 windows ✓ SUFFICIENT

Mean OOS Sharpe estimation:
  SE(mean) = SD / √n
  For SD = 0.5, n = 16: SE = 0.125
  95% CI width: ±0.25 (acceptable precision)

Percentage of positive periods:
  Binomial test: n = 16, p = 0.6 (expected)
  Power to detect p > 0.5: 60% (moderate)
  Recommendation: Report with confidence intervals
```

**Critical Issue**: Each OOS window has only 3 months
```
Trades per OOS window: 3 months × 30 trades/month = 90 trades
Sharpe SE with 90 trades: √((1 + 0.5×2²) / 90) = 0.17 (marginal)
```

**Solution**: Use overlapping windows or extend OOS to 6 months (fewer windows but more reliable per-window estimates)

### H18: Parameter Stability

**Required Data**:
- **Rolling windows**: 6-month windows, 3-month step
- **Number of windows**: (24 - 6) / 3 + 1 = 7 windows (marginal)
  - Preferred: ≥ 10 windows
  - **Solution**: Use 1-month step → 19 windows ✓ SUFFICIENT

**Statistical Power**:
```
Variance of optimal z_entry across windows:
  Need ≥ 10 observations for variance estimation
  Current with 1-month step: 19 windows ✓ SUFFICIENT

Rank correlation test (Spearman):
  H0: ρ = 0 (no correlation)
  Power with n = 19, true ρ = 0.5: 80% ✓
  Power with n = 7, true ρ = 0.5: 40% ✗
```

---

## 5. Diversification & Risk (H7, H14)

### H7: Stop Loss Effectiveness

**Required Data**:
- **Configurations**: 3 (with stop, without stop, alternative stops)
- **Minimum stopped trades**: 20-30 for distribution analysis
  - Expected stop rate: 5-10% of trades
  - Total trades needed: 200-300 → stopped trades = 10-30 ✓ MARGINAL
  - **Current status**: ⚠️ May need to pool multiple pairs or extend period

**Statistical Power**:
```
Tail risk metrics (95th percentile loss):
  Minimum observations for percentile: n ≥ 100
  Current expected: 600-1,000 trades ✓ SUFFICIENT

Max drawdown comparison:
  Bootstrap confidence intervals: 1,000 iterations
  Requires full trade sequence (not just count)
  Current: 2 years of daily returns ✓ SUFFICIENT
```

### H14: Portfolio Diversification

**Required Data**:
- **Portfolio sizes**: {3, 5, 10, 15, 20} pairs
- **Correlation matrix**: Need ≥ 60 observations (daily returns)
  - 2 years = 504 trading days ✓ SUFFICIENT for daily correlation
  - Intraday: 504 days × 390 min = 196,560 bars ✓ EXCESSIVE (use daily)

**Statistical Power**:
```
Correlation estimation (pairwise):
  SE(r) ≈ (1 - r²) / √(n - 3)
  For r = 0.1, n = 504: SE = 0.044 ✓ GOOD precision

Portfolio volatility vs. number of pairs:
  Need ≥ 3 pairs minimum, test up to 20
  Each configuration is full backtest (same data)
  Statistical power: ✓ SUFFICIENT (deterministic for given data)
```

---

## 6. Timeframe Comparison (H10)

### H10: Intraday vs. Daily

**Required Data per Timeframe**:

| Timeframe | Bars/Day | Bars/2yr | Trades/Pair/Yr | Min Pairs | Status |
|-----------|----------|----------|----------------|-----------|--------|
| 1-minute  | 390      | 196,560  | 50-100         | 5         | ✓      |
| 5-minute  | 78       | 39,312   | 30-60          | 5         | ✓      |
| 15-minute | 26       | 13,104   | 15-30          | 10        | ✓      |
| 1-hour    | 6.5      | 3,276    | 10-20          | 15        | ⚠️     |
| Daily     | 1        | 504      | 5-15           | 20        | ⚠️     |

**Critical Constraint**: Daily timeframe
```
Minimum observations for cointegration: 120 days (4 months)
IS period: 12 months (252 days) ✓ SUFFICIENT
OOS period: 6 months (126 days) ✓ SUFFICIENT

Expected trades on daily: 10 pairs × 10 trades/yr × 2 yr = 200 trades ✓ MARGINAL
```

**Statistical Power**:
```
Sharpe comparison across timeframes:
  Test: ANOVA or Kruskal-Wallis
  Groups: 5 timeframes
  Observations per group: 200-1,000 trades (varies)
  Power: High for 1-min/5-min, Moderate for daily ⚠️
```

---

## Summary Table: Data Sufficiency

| Hypothesis | Min Period | Min Trades | Min Pairs | Current Status | Recommendation |
|------------|------------|------------|-----------|----------------|----------------|
| H1: Sector | 6 months   | N/A        | 100       | ✓ 24 months    | SUFFICIENT     |
| H2: Half-life | 18 months | N/A      | 50        | ⚠️ 24 months   | Use overlapping windows |
| H3: P-value | 12 months  | 30/quintile| 100      | ✓ 24 months    | SUFFICIENT     |
| H5: Threshold | 12 months | 200/level | 10       | ⚠️ 24 months   | Pool z≥3.0     |
| H7: Stop loss | 12 months | 200 total  | 10       | ✓ 24 months    | SUFFICIENT     |
| H9: Window | 12 months   | 200/level  | 10       | ✓ 24 months    | SUFFICIENT     |
| H12: Costs | 12 months   | 200        | 10       | ✓ 24 months    | SUFFICIENT     |
| H14: Diversify | 12 months | 200      | 20       | ✓ 24 months    | SUFFICIENT     |
| H16: IS/OOS | 18 months   | 200 OOS   | 10       | ✓ 24 months    | SUFFICIENT     |
| H17: Walk-fwd | 24 months | 90/window | 10       | ⚠️ 24 months   | Extend OOS to 6mo |
| H18: Stability | 24 months | 200/window| 10      | ⚠️ 24 months   | 1-month step   |

**Overall Assessment**: ✓ **24 months (2 years) is SUFFICIENT for most tests**

**Marginal Cases** (⚠️):
1. **H2** (Half-life stability): Use overlapping windows to increase sample
2. **H5** (High thresholds z≥3.0): Pool data or report with wider CIs
3. **H17** (Walk-forward): Extend OOS windows to 6 months (fewer windows, better per-window estimates)
4. **H18** (Parameter stability): Use 1-month step instead of 3-month

---

## Minimum Data Requirements by Analysis Type

### 1. Cointegration Testing
```
Rule of Thumb (MacKinnon 1991):
  Daily data: 120-250 observations (6-12 months)
  Intraday 1-min: Use aggregated bars (5-min or 15-min)
  Effective observations: 252 days × 78 (5-min bars) = 19,656 ✓ EXCESSIVE

Recommended:
  Use daily or 5-min aggregation for cointegration
  Use 1-min for signal generation and backtesting
```

### 2. Sharpe Ratio Estimation
```
Standard Error Formula:
  SE(Sharpe) = √[(1 + 0.5 × Sharpe²) / n]

For Sharpe = 2.0:
  n = 50:   SE = 0.22 (poor)
  n = 100:  SE = 0.16 (marginal)
  n = 200:  SE = 0.11 (good)
  n = 500:  SE = 0.07 (excellent)

Minimum for publication: n ≥ 200 trades or 2 years monthly returns
```

### 3. Parameter Optimization
```
Degrees of Freedom:
  Grid search: k parameters, p points each → p^k configurations
  Example: 3 parameters, 5 points → 125 configurations

Data split:
  Train (60%): 14.4 months → optimize
  Validation (20%): 4.8 months → select
  Test (20%): 4.8 months → final evaluation

Minimum: 18 months for 3-way split with 200 trades in test set
Current: 24 months ✓ SUFFICIENT
```

### 4. Walk-Forward Analysis
```
Minimum Structure:
  IS window: 6 months (sufficient for cointegration + optimization)
  OOS window: 3-6 months (trade-off: more windows vs. more trades/window)
  Step size: 1-3 months (smaller = more windows, more overlapping data)

Optimal for 24 months:
  IS: 6 months, OOS: 6 months, Step: 1 month
  Windows: (24 - 12) / 1 + 1 = 13 windows ✓ GOOD

Alternative (more windows):
  IS: 6 months, OOS: 3 months, Step: 1 month
  Windows: (24 - 9) / 1 + 1 = 16 windows ✓ BETTER
```

---

## Statistical Power Analysis

### Power Calculations

#### Hypothesis Tests (Two-Sample Comparisons)

**Example: H16 (In-Sample vs. Out-of-Sample Sharpe)**

```
Assumptions:
  True Sharpe_IS = 2.0
  True Sharpe_OOS = 1.5 (25% degradation)
  Standard deviation of Sharpe estimates ≈ 0.3

Effect size (Cohen's d):
  d = (2.0 - 1.5) / 0.3 = 1.67 (large effect)

Power with rolling windows (n = 16):
  Two-sample t-test, α = 0.05, two-tailed
  Power ≈ 99% (very high) ✓

Power with single split (n = 2):
  Power ≈ 30% (very low) ✗
  Conclusion: Walk-forward essential for statistical power
```

#### Correlation Tests

**Example: H18 (Parameter Stability via Rank Correlation)**

```
True rank correlation ρ = 0.6 (moderate stability)
Windows: n = 19 (with 1-month step)
α = 0.05, two-tailed

Power to detect ρ ≠ 0:
  Power ≈ 85% ✓ GOOD

Windows: n = 7 (with 3-month step)
  Power ≈ 45% ✗ POOR

Conclusion: Use 1-month step for adequate power
```

#### Chi-Square Test

**Example: H1 (Within-Sector vs. Cross-Sector Cointegration)**

```
Expected cointegration rates:
  Within-sector: 30% (150/500 pairs)
  Cross-sector: 10% (100/1000 pairs)

Chi-square test:
  df = 1
  α = 0.05

Power calculation:
  Effect size w = √[(0.3-0.1)² / (0.5×0.2)] = 0.63 (medium-large)
  N = 1,500 (total pairs tested)
  Power > 99% ✓ EXCELLENT
```

---

## Recommendations

### Immediate (Current 2-Year Data)

1. **Proceed with all 10 hypotheses** using current 24-month dataset
2. **Use overlapping windows** for H2, H17, H18 to maximize sample size
3. **Pool high-threshold data** for H5 (z ≥ 3.0) or report with wider confidence intervals
4. **Extend OOS windows** to 6 months in H17 for more reliable per-window Sharpe estimates

### Optimal (If Extending Data)

1. **Extend to 36 months (3 years)** for:
   - Better half-life stability estimation (H2): 6 non-overlapping windows
   - Higher threshold testing (H5): More rare z ≥ 3.0 signals
   - More walk-forward windows (H17): 20+ windows with 6-month OOS

2. **Priority**: Data quality > quantity
   - Ensure after-hours filtering is correct
   - Validate half-day holiday handling
   - Check for corporate actions (splits, dividends)

### Long-Term (Production Monitoring)

1. **Rolling 2-year window** for continuous validation
2. **Monthly re-testing** of top 20 pairs for cointegration breakdown
3. **Quarterly parameter review** based on most recent 6 months
4. **Annual walk-forward** extending by 12 months

---

## Sample Size Formulas Reference

### Sharpe Ratio Standard Error
```
SE(Sharpe) = √[(1 + 0.5 × Sharpe²) / n]

where:
  n = number of independent observations (trades or return periods)
  Sharpe = estimated Sharpe ratio
```

### Minimum Observations for Correlation
```
n ≥ (Z_α/2 + Z_β)² / (0.5 × ln[(1+ρ)/(1-ρ)])²

For ρ = 0.3, α = 0.05, β = 0.20 (80% power):
  n ≥ 85 observations

For ρ = 0.5:
  n ≥ 29 observations
```

### ADF Test Power
```
MacKinnon (1991) minimum observations:
  5% significance: n ≥ 100
  1% significance: n ≥ 200

Critical values adjust for sample size
Use at least 120 observations for reliable ADF test
```

### Chi-Square Power
```
N = (Z_α/2 + Z_β)² / w²

where:
  w = effect size = √[Σ(p_i - p_0i)² / p_0i]
  p_i = proportions in categories

For medium effect (w = 0.3), 80% power:
  N ≥ 88 observations
```

---

## Conclusion

**Current Status: ✓ SUFFICIENT**

With **24 months (2023-2024)** of 1-minute intraday data for **100 stocks**:

✅ **Sufficient** for 8/10 priority hypotheses
⚠️ **Marginal** for 2/10 hypotheses (H2, H17)
🔧 **Solutions exist** using overlapping windows and adjusted window sizes

**No additional data collection required to proceed with statistically significant analysis.**

**Confidence Level**: We can achieve **80-95% statistical power** for most hypothesis tests with current data using appropriate methodological adjustments (overlapping windows, pooled estimates for rare events).
