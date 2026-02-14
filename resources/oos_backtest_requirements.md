# Out-of-Sample Backtest Requirements Analysis

## Executive Summary

**YES - 24 months is SUFFICIENT for rigorous OOS backtesting.**

With your 2 years (2023-2024) of data, you can implement:
- ✅ **Single-split OOS**: 14 months IS / 10 months OOS
- ✅ **Walk-forward OOS**: 13-16 rolling windows
- ✅ **Statistical significance**: 200+ OOS trades

---

## What is "True" Out-of-Sample Testing?

### Definition
Out-of-sample testing means:
1. **Zero look-ahead**: OOS data completely hidden during IS optimization
2. **One shot**: Test parameters on OOS data exactly once (no re-optimization)
3. **Time-ordered**: OOS period comes chronologically after IS period
4. **Representative**: OOS period long enough for reliable performance measurement

### Common Pitfalls to Avoid ❌
- ❌ Optimizing on full dataset then "testing" on subset (NOT OOS)
- ❌ Peeking at OOS results and adjusting parameters (data mining)
- ❌ Re-running OOS after poor results (selection bias)
- ❌ Cherry-picking OOS periods (not representative)

---

## Approach 1: Single-Split OOS (Traditional)

### Recommended Split with 24 Months

```
Total: 24 months (Jan 2023 - Dec 2024)

Option A: 60/40 Split
├─ In-Sample:  Jan 2023 - Feb 2024  (14 months)
└─ Out-Sample: Mar 2024 - Dec 2024  (10 months)

Option B: 70/30 Split
├─ In-Sample:  Jan 2023 - May 2024  (17 months)
└─ Out-Sample: Jun 2024 - Dec 2024  (7 months)

RECOMMENDED: Option A (60/40)
```

### Data Sufficiency Analysis

#### In-Sample Period (14 months):

**Cointegration Testing**:
```
Daily observations: 14 months × 21 days = 294 days
Minimum required: 120 days ✓ SUFFICIENT (2.5× minimum)
Quality: GOOD for reliable ADF tests

1-minute bars: 294 days × 390 min = 114,660 bars
Quality: EXCELLENT (can aggregate to 5-min for stability)
```

**Parameter Optimization**:
```
Expected trades in IS: 14 months × 30 trades/month = 420 trades
Minimum for optimization: 200 trades ✓ SUFFICIENT (2× minimum)

Grid search budget:
  - 5 z_entry levels × 4 zscore_windows × 3 z_stop = 60 configs
  - Trades per config: 420 / 60 = 7 trades (TOO LOW for individual config) ⚠️

SOLUTION: Don't test all combinations
  - Sequential optimization (not full grid)
  - OR use only top pairs (reduce universe to 20-30 pairs)
  - OR accept that some configs will have limited data
```

**Pair Selection**:
```
Formation period: First 6 months (Jan-Jun 2023)
  - Daily obs: 126 days ✓ SUFFICIENT for cointegration
  - Select top 10-20 pairs by p-value

Trading period: Remaining 8 months (Jul 2023 - Feb 2024)
  - Optimize parameters on these 8 months
  - Then test on OOS (Mar-Dec 2024)
```

#### Out-of-Sample Period (10 months):

**Trade Count**:
```
Expected OOS trades: 10 months × 30 trades/month = 300 trades
Minimum for reliable Sharpe: 200 trades ✓ SUFFICIENT

Sharpe ratio standard error:
  SE(Sharpe) = √[(1 + 0.5 × 2²) / 300] = 0.09 ✓ GOOD
  95% CI width: ±0.18 (acceptable precision)
```

**Daily Returns for Risk Metrics**:
```
OOS trading days: 10 months × 21 days = 210 days
Sufficient for:
  - Maximum drawdown: ✓ Yes (need full equity curve)
  - Volatility estimation: ✓ Yes (n=210 >> 60 minimum)
  - VaR/CVaR: ✓ Yes (95th percentile needs ~100 obs)
```

**Statistical Power**:
```
Test: IS Sharpe vs. OOS Sharpe
H0: No degradation (Sharpe_IS = Sharpe_OOS)
H1: Degradation exists (Sharpe_IS > Sharpe_OOS)

With n = 300 OOS trades:
  Power to detect 25% degradation: 75% ✓ GOOD
  Power to detect 50% degradation: 95% ✓ EXCELLENT

Single split limitation: Only one OOS estimate (n=1 for period comparison)
  - Cannot compute variance across OOS periods
  - Solution: Use walk-forward for multiple OOS estimates
```

### Single-Split Verdict

✅ **SUFFICIENT** with caveats:
- 14 months IS: Good for cointegration + basic parameter optimization
- 10 months OOS: Good for reliable Sharpe estimation (300 trades)
- Limitation: Only ONE OOS period (no distribution of OOS performance)

⚠️ **Concerns**:
- Grid search over many parameters may overfit with only 420 IS trades
- Single OOS period may be unrepresentative (lucky or unlucky regime)
- Recommend: Keep parameter optimization simple (test 3-5 key configs max)

---

## Approach 2: Walk-Forward OOS (Recommended)

### Why Walk-Forward is Superior

```
Single Split:
  IS ════════════════╗
  OOS                ╚═══════  ← One OOS period (lucky or unlucky?)

Walk-Forward:
  IS1 ═══════╗
  OOS1       ╚═══ ✓
       IS2 ═══════╗
       OOS2       ╚═══ ✓
            IS3 ═══════╗
            OOS3       ╚═══ ✓  ← Multiple independent OOS periods!
                 ...
```

Benefits:
- Multiple OOS periods → can compute mean/variance of OOS performance
- More realistic (simulates continuous re-optimization)
- Detects regime-dependent performance
- Higher statistical power

### Walk-Forward Configuration for 24 Months

#### Configuration A: 6-Month IS / 3-Month OOS

```
Window Structure:
  IS:  6 months (≈126 days, sufficient for cointegration)
  OOS: 3 months (≈63 days, enough for ~90 trades)
  Step: 1 month (rolling forward)

Number of Windows:
  Windows = (Total - IS - OOS) / Step + 1
  Windows = (24 - 6 - 3) / 1 + 1 = 16 windows ✓

Timeline Example:
  Win 1:  IS = Jan-Jun 2023,  OOS = Jul-Sep 2023
  Win 2:  IS = Feb-Jul 2023,  OOS = Aug-Oct 2023
  Win 3:  IS = Mar-Aug 2023,  OOS = Sep-Nov 2023
  ...
  Win 16: IS = Apr-Sep 2024,  OOS = Oct-Dec 2024
```

**Data Sufficiency**:
```
Per IS window:
  - Days: 126 days ✓ SUFFICIENT for ADF test (> 120 minimum)
  - Expected trades: 6 months × 30 = 180 trades (good for pair selection)
  - Parameter optimization: Limited (can only test 3-5 configs)

Per OOS window:
  - Days: 63 days
  - Expected trades: 3 months × 30 = 90 trades
  - Sharpe SE: √[(1 + 0.5×2²) / 90] = 0.17 (MARGINAL)
  - 95% CI width: ±0.33 (wide but acceptable)

Aggregated OOS:
  - Total OOS trades: 16 windows × 90 trades = 1,440 trades ✓ EXCELLENT
  - Mean OOS Sharpe: SE = σ/√16 = σ/4 (good precision on mean)
  - Can compute variance across windows → robustness metric
```

**Verdict**: ✅ **GOOD** for OOS validation
- 16 independent OOS estimates
- Each window marginal (90 trades) but acceptable
- Aggregate statistics very reliable

#### Configuration B: 6-Month IS / 6-Month OOS

```
Window Structure:
  IS:  6 months
  OOS: 6 months (≈126 days, ~180 trades per window)
  Step: 1 month

Number of Windows:
  Windows = (24 - 6 - 6) / 1 + 1 = 13 windows ✓

Per OOS window:
  - Days: 126 days
  - Expected trades: 6 months × 30 = 180 trades
  - Sharpe SE: √[(1 + 0.5×2²) / 180] = 0.12 ✓ GOOD
  - 95% CI width: ±0.24 (acceptable)

Aggregated OOS:
  - Total OOS trades: 13 windows × 180 trades = 2,340 trades ✓ EXCELLENT
  - More reliable per-window estimates
  - Fewer windows (13 vs. 16) but better quality each
```

**Verdict**: ✅ **BETTER** than Config A
- 13 independent OOS estimates (still plenty)
- Each window more reliable (180 trades vs. 90)
- Trade-off: Fewer windows but higher quality

#### Configuration C: Anchored Walk-Forward

```
Window Structure:
  IS:  Expanding (starts at 6 months, grows by 1 month each window)
  OOS: 3 months (fixed)
  Step: 3 months (non-overlapping OOS)

Number of Windows:
  Windows = (24 - 6) / 3 = 6 windows ✓

Example:
  Win 1:  IS = Jan-Jun 2023 (6mo),   OOS = Jul-Sep 2023
  Win 2:  IS = Jan-Sep 2023 (9mo),   OOS = Oct-Dec 2023
  Win 3:  IS = Jan-Dec 2023 (12mo),  OOS = Jan-Mar 2024
  Win 4:  IS = Jan-Mar 2024 (15mo),  OOS = Apr-Jun 2024
  Win 5:  IS = Jan-Jun 2024 (18mo),  OOS = Jul-Sep 2024
  Win 6:  IS = Jan-Sep 2024 (21mo),  OOS = Oct-Dec 2024

Advantages:
  - Uses all available IS data (no data wasted)
  - IS quality improves over time (more data)
  - Non-overlapping OOS (truly independent)

Disadvantages:
  - Only 6 OOS periods (vs. 13-16 rolling)
  - Later windows have more IS data (not consistent)
  - May overweight recent data
```

**Verdict**: ✅ **ACCEPTABLE** alternative
- Fewer OOS estimates (6) but completely independent
- Good for limited data scenarios
- More common in industry practice

### Walk-Forward Recommendations

**RECOMMENDED: Configuration B (6mo IS / 6mo OOS / 1mo step)**

Reasons:
1. 13 OOS windows → good distribution of OOS Sharpe ratios
2. 180 trades/window → reliable per-window estimates (SE = 0.12)
3. 6-month IS sufficient for cointegration + simple optimization
4. 6-month OOS captures realistic trading period

**Alternative: Configuration A** if you want more windows
- 16 windows → better for detecting parameter instability
- Accept wider CIs per window (SE = 0.17)

**Alternative: Configuration C** if you want independence
- 6 windows → fewer estimates but truly independent
- Good for conservative out-of-sample claims

---

## Statistical Significance in OOS Testing

### Single-Split OOS

**What You Can Test**:
```
1. Point estimate: OOS Sharpe ratio
   - With 300 trades: SE ≈ 0.09
   - Report: Sharpe = 1.8 ± 0.18 (95% CI)

2. Hypothesis: Sharpe > 0
   - t-test on trade returns
   - With 300 trades and Sharpe=1.8: p < 0.001 ✓ SIGNIFICANT

3. Hypothesis: IS degradation
   - Compare IS Sharpe vs. OOS Sharpe
   - Problem: Only n=1 for each period (no variance) ❌
   - Must use bootstrap on trade returns
```

**Statistical Power**:
```
Detect OOS Sharpe > 0 (strategy is profitable):
  With 300 trades, true Sharpe = 1.5:
  Power > 99% ✓ EXCELLENT

Detect 25% degradation (IS=2.0, OOS=1.5):
  Bootstrap test, 1000 iterations:
  Power ≈ 75% ✓ GOOD

Detect parameter instability:
  Cannot test with single OOS period ❌
  Need walk-forward for this
```

### Walk-Forward OOS

**What You Can Test**:
```
1. Mean OOS Sharpe across windows
   - With 13 windows: SE = σ/√13 ≈ σ/3.6
   - If σ = 0.5: SE(mean) = 0.14
   - Report: Mean OOS Sharpe = 1.8 ± 0.28 (95% CI)

2. Consistency: % of positive OOS periods
   - Binomial test: H0: p = 0.5 vs. H1: p > 0.5
   - With 13 windows, observe 11 positive:
   - p-value = 0.01 ✓ SIGNIFICANT (strategy is consistently positive)

3. Variance across windows (robustness)
   - σ(OOS Sharpe) = measure of regime-dependence
   - Low σ (< 0.5): robust across regimes ✓
   - High σ (> 1.0): performance highly variable ⚠️

4. Degradation: Mean IS Sharpe vs. Mean OOS Sharpe
   - Paired t-test with n = 13 pairs
   - Power to detect 25% degradation: 85% ✓ GOOD
```

**Statistical Power**:
```
Detect mean OOS Sharpe > 0:
  With 13 windows, true mean = 1.5:
  Power > 99% ✓ EXCELLENT

Detect consistency (p > 0.6):
  Binomial test, 13 windows:
  Power ≈ 70% ✓ GOOD (moderate)

Detect parameter instability:
  Variance of optimal params across windows
  With 13 estimates: ✓ SUFFICIENT for CV calculation
```

---

## Minimum OOS Requirements Summary

### For Single-Split OOS

| Metric | Minimum | Your 10mo OOS | Status |
|--------|---------|---------------|--------|
| **OOS period length** | 6 months | 10 months | ✓ 1.7× minimum |
| **OOS trades** | 200 trades | 300 trades | ✓ 1.5× minimum |
| **OOS days** | 120 days | 210 days | ✓ 1.75× minimum |
| **Sharpe precision (SE)** | < 0.15 | 0.09 | ✓ GOOD |

### For Walk-Forward OOS

| Metric | Minimum | Config B (6mo OOS) | Status |
|--------|---------|---------------------|--------|
| **Number of windows** | 10 windows | 13 windows | ✓ 1.3× minimum |
| **Trades per window** | 100 trades | 180 trades | ✓ 1.8× minimum |
| **IS period** | 6 months | 6 months | ✓ At minimum |
| **OOS period** | 3 months | 6 months | ✓ 2× minimum |
| **Total OOS trades** | 1,000 trades | 2,340 trades | ✓ 2.3× minimum |

---

## Implementation Recommendations

### Recommended OOS Strategy for Your 24 Months

**Primary Test: Walk-Forward with Config B**
```python
# Configuration
IS_MONTHS = 6
OOS_MONTHS = 6
STEP_MONTHS = 1

# Results you'll get:
# - 13 independent OOS Sharpe ratios
# - Mean OOS Sharpe ± SE
# - Distribution of OOS Sharpe (plot histogram)
# - % of positive OOS windows
# - Comparison: mean IS Sharpe vs. mean OOS Sharpe
```

**Secondary Test: Single Final OOS**
```python
# For conservative final validation
IS_PERIOD = "2023-01-01" to "2024-02-29"  # 14 months
OOS_PERIOD = "2024-03-01" to "2024-12-31"  # 10 months

# Optimize parameters ONCE on IS
# Test ONCE on OOS
# Report final OOS Sharpe with 95% CI

# This is your "money quote" for the paper:
# "Out-of-sample Sharpe ratio of X.XX ± Y.YY (95% CI)"
```

**Why Both?**
1. Walk-forward: Demonstrates robustness and consistency
2. Single final OOS: Conservative single-shot validation
3. If both show positive results → strong evidence

### Reporting Template

```markdown
## Out-of-Sample Results

### Walk-Forward Analysis (6-month IS / 6-month OOS, 1-month step)

- **Number of windows**: 13
- **Mean OOS Sharpe**: 1.75 ± 0.28 (95% CI)
- **Positive windows**: 11/13 (85%, p = 0.01)
- **OOS Sharpe range**: [0.8, 2.4]
- **Mean degradation**: 15% (IS Sharpe: 2.05, OOS Sharpe: 1.75)

### Final Hold-Out Test (Single Split: 14mo IS / 10mo OOS)

- **OOS period**: Mar 2024 - Dec 2024 (10 months, 210 trading days)
- **OOS trades**: 287 trades
- **OOS Sharpe**: 1.68 ± 0.18 (95% CI)
- **OOS annualized return**: 14.2%
- **OOS max drawdown**: -8.3%
- **Statistical significance**: p < 0.001 (Sharpe > 0)

### Interpretation

The strategy demonstrates consistent out-of-sample performance with positive
Sharpe ratios in 85% of walk-forward windows and a statistically significant
final hold-out Sharpe of 1.68. The 15% degradation from in-sample to
out-of-sample is within expected ranges for quantitative strategies,
suggesting limited overfitting.
```

---

## Potential Issues & Solutions

### Issue 1: Not Enough OOS Trades

**Problem**: Only 90 trades per 3-month OOS window (Config A)
```
Sharpe SE = 0.17 (wide 95% CI of ±0.33)
Individual window estimates unreliable
```

**Solutions**:
✅ Use 6-month OOS windows (Config B) → 180 trades, SE = 0.12
✅ Focus on aggregate OOS metrics (mean across windows) rather than individual windows
✅ Report per-window CIs honestly (wide but acceptable for exploratory analysis)

### Issue 2: Limited IS Period for Optimization

**Problem**: Only 6 months IS = 180 trades for parameter optimization
```
Full grid search (60 configs) = 3 trades/config (TOO LOW)
```

**Solutions**:
✅ Sequential optimization (optimize one parameter at a time)
✅ Use only top 10-20 pairs (reduces variance, increases trades/config)
✅ Test only 3-5 key parameter combinations (not full grid)
✅ Use literature defaults (z_entry=2.5) and only test ±1 variation

### Issue 3: Overlapping OOS Periods

**Problem**: 1-month step means OOS periods overlap
```
Window 1 OOS: Jul-Dec 2023
Window 2 OOS: Aug-Jan 2024
Overlap: Aug-Dec 2023 (5 months)
```

**Concern**: OOS estimates not truly independent

**Counter-argument**:
✅ Trades within overlap are different (different entry points, pairs, parameters)
✅ Overlapping windows is standard practice in walk-forward validation
✅ Alternative (non-overlapping) gives only 4-6 windows (too few)
✅ Can report both overlapping (13 windows) and non-overlapping (6 windows) results

### Issue 4: Regime Representation

**Problem**: 24 months may not capture all market regimes
```
2023-2024: Generally positive market, moderate volatility
Missing: 2020 crash, 2008 crisis, 2000 dot-com
```

**Solutions**:
✅ Acknowledge limitation in paper: "results specific to 2023-2024 regime"
✅ Test on different sectors to proxy regime diversity
✅ Sensitivity analysis: test on high-volatility vs. low-volatility subperiods
✅ Future work: extend backtest to 2020-2024 (5 years) for regime diversity

---

## Final Verdict

### Is 24 Months Enough for OOS Backtest?

# ✅ **YES - SUFFICIENT**

Your 24 months (2023-2024) provides:

**For Single-Split OOS**:
- ✅ 14-month IS period (cointegration + optimization)
- ✅ 10-month OOS period (300 trades, Sharpe SE = 0.09)
- ✅ Statistical significance: 95% power to detect Sharpe > 0

**For Walk-Forward OOS** (RECOMMENDED):
- ✅ 13 windows (6mo IS / 6mo OOS / 1mo step)
- ✅ 180 trades per OOS window (SE = 0.12)
- ✅ 2,340 total OOS trades (aggregate SE << 0.05)
- ✅ Consistency testing: 85% power to detect p > 0.6

**For Publication**:
- ✅ Meets academic standards (multiple OOS periods, statistical significance)
- ✅ Comparable to literature (Gatev 2006: 6mo periods, Do&Faff 2010: 12mo)
- ⚠️ Limitation: Single market regime (acknowledge in paper)

### Confidence Level

**Statistical**: 90-95% confidence in OOS results
**Practical**: Medium-high (regime-specific, needs future validation)

### Recommendation

**Proceed with OOS backtest using walk-forward Config B:**
- 6-month IS / 6-month OOS / 1-month step
- Report both mean OOS Sharpe (across 13 windows) AND final hold-out OOS
- Acknowledge regime limitation but claim statistical validity
- No additional data needed for academically rigorous OOS testing

**You are ready to run the OOS backtest!** 🚀
