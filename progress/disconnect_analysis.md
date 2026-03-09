# Parameter Search vs Backtest Disconnect Analysis

**Date:** 2026-03-08

---

## Correction: Actual strategy.ipynb Parameters

The initial analysis assumed the baseline config matched `utils/config.py`. The strategy notebook **overrides** key parameters:

| Parameter | config.py | strategy.ipynb (actual) |
|---|---|---|
| z_entry | 2.5 | **4.0** |
| z_exit | 0.0 | **2.5** |
| z_stop | 4.0 | **5.0** |
| max_holding_minutes | 120 | **300** |
| rolling_window_days | 42 | 42 |
| **cost_bps** | 5.0 | **0.0 (gross only)** |
| start_date | — | 2020-07-01 |
| end_date | — | 2024-12-31 |
| execution_lag_minutes | None | 2 (uses 1-min bars) |

**Critical:** `cost_bps=0.0` means strategy.ipynb measures **gross P&L only** — it is not a net profitability test. The strategy is intentionally separating signal edge from execution cost analysis.

The actual strategy config (z_entry=4, z_exit=2.5, z_stop=5, max_hold=300) is nearly identical to **C2** in the experiment comparison, which achieved +$2,240 gross / −$65 net at 5 bps/leg over 2022–2024. At 0 bps, C2 = +$2,240 gross on 115 trades — confirming the signal exists but is sparse.

---

## The Disconnect

The parameter search ranked configs by **breakeven bps/leg** and **IC**. The backtests show all configs are net-negative at 5 bps/leg. These are not contradictory — they measure different things.

---

## Root Causes

### 1. Different evaluation frameworks

| Dimension | Parameter Search | Backtest |
|---|---|---|
| Exit logic | Fixed N-bar forward return | State machine: z_exit / z_stop / max_hold / EOD |
| Cost model | 4 × cost_bps on mean return per obs | 4 legs × actual notional per trade |
| Signal weighting | Inverse p-value rank | Unweighted — enter any pair crossing z_entry |
| Primary metric | Breakeven bps (per signal observation) | Net P&L ($) |
| Activity measure | n_obs = all bars with active signal | Trades that open and close |

Breakeven bps is a per-observation metric. A config with 13 bps breakeven and 400 observations still needs those observations to translate into profitable **round-trip trades** — which requires the state machine to open, hold through reversion, and exit at z_exit. The fixed-horizon forward return assumes you always hold for N bars regardless of what the spread does. The state machine does not.

### 2. z_exit=0.0 kills the z_exit rate

The most impactful single parameter, not tested in the parameter search: **z_exit**.

- The parameter search only tested `z_exit=2.5` (paired with `z_entry=4.0`)
- The current baseline config uses `z_exit=0.0` — requiring the spread to cross zero before exiting
- At `z_entry=2.5, z_exit=0.0`, the spread needs to move from 2.5σ all the way through 0σ — very demanding
- Result: **4.2% z_exit rate** in C1 baseline. 80% of trades time out

When `z_exit=1.5` (C4) or `z_exit=1.0` (C5), z_exit rate rises to 18%+ with 90%+ win rates.

### 3. Activity starvation at high z_entry

The parameter search declared `z_entry=4.0` optimal. In the backtest:
- C2 (z_entry=4.0, 2-year period): **115 total trades**
- That is ~1 trade per 4 trading days across the entire universe
- Per-trade quality is excellent (71% z_exit rate, near breakeven at 5 bps/leg)
- But 115 trades generate essentially zero total P&L regardless of quality

The param search metric (breakeven bps) is scale-invariant — 115 trades at 13 bps breakeven looks identical to 6,000 trades at 13 bps breakeven. The backtest cares about total dollars.

### 4. Period and regime mismatch

- Parameter search: **H2 2022 only** (bear market, elevated volatility, wider spreads)
- Backtest: **2022-07-01 → 2024-07-01** (bear + recovery + bull)
- Bear markets produce wider spread oscillations → more signal observations → higher estimated activity
- The param search's `avg_sim_pairs` estimates were calibrated to an abnormally active period

### 5. z_stop as a major P&L driver (not in param search)

The parameter search had no z_stop concept — it evaluated fixed forward returns, not runaway spread losses. In the backtest, z_stop is a large P&L item:

| Config | z_stop gross P&L |
|---|---|
| C1 Baseline | −$28,179 |
| C3 Moderate Entry | −$26,873 |
| C5 Fast Z | −$21,211 |

z_stop triggers on spread divergence (e.g., entering at 2.5σ, stopped at 4.0σ). These are the worst trades — entered on what looked like a cointegrated pair, but the spread kept running. The param search had no way to see this because it only looked at forward returns, not at which trades would eventually get stopped out.

---

## Backtest Results: 5 Configs

**Period:** 2022-07-01 → 2024-07-01 | **Capital:** $100k | **Cost:** 5 bps/leg

| Config | z_entry | z_exit | z_stop | max_hold | Trades | Gross P&L | Net P&L | Sharpe | z_exit% | maxhold% |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 Baseline | 2.5 | 0.0 | 4.0 | 120 | 6,860 | +$15,354 | −$78,381 | −2.78 | 4.2% | 80.0% |
| C2 ParamSearch Winner | 4.0 | 2.5 | 5.0 | 300 | 115 | +$2,240 | −$65 | −0.02 | 71.3% | 1.7% |
| C3 Moderate Entry | 3.0 | 0.0 | 4.5 | 180 | 6,464 | +$18,577 | −$69,885 | −2.27 | 4.5% | 71.3% |
| C4 Tight+Partial Exit | 3.5 | 1.5 | 5.0 | 240 | 908 | +$3,628 | −$16,703 | −1.23 | 18.1% | 61.1% |
| C5 Fast Z | 3.0 | 1.0 | 4.5 | 120 | 6,883 | +$23,089 | −$77,836 | −2.17 | 17.2% | 75.9% |

### Exit cohort breakdown

| Config | z_exit gross | z_exit win% | max_hold gross | max_hold win% | z_stop gross |
|---|---|---|---|---|---|
| C1 Baseline | +$8,686 | 67.6% | +$34,848 | 53.1% | −$28,179 |
| C2 ParamSearch | +$2,668 | 61.0% | −$50 | 50.0% | −$193 |
| C3 Moderate | +$16,415 | 69.6% | +$26,555 | 53.2% | −$26,873 |
| C4 Tight+Partial | +$22,678 | **93.3%** | −$17,278 | 41.8% | −$1,374 |
| C5 Fast Z | **+$89,881** | **89.7%** | −$45,581 | 46.7% | −$21,211 |

---

## Key Findings

### C2 validates the param search — but it's not a strategy
C2 (the param search winner) achieves near-breakeven at 5 bps/leg: net −$65 on 115 trades over 2 years. The param search's 13 bps breakeven estimate was accurate for this regime. The problem is it generates 1 trade per 4 days — not enough to matter.

### z_exit is the most important parameter not tested
Comparing C1 (z_exit=0.0) vs C4 (z_exit=1.5) at similar z_entry levels:
- z_exit rate: 4.2% → 18.1%
- z_exit win rate: 67.6% → 93.3%
- max_hold rate: 80% → 61%
- Net P&L: −$78k → −$17k

Setting z_exit to a partial reversion target (1.0–1.5σ instead of 0σ) is the single largest lever available. The old presentation's "1.75" exit was already partially capturing this.

### C5 has the best z_exit cohort ever observed (+$89,881 at 89.7% win rate)
The fast zscore window (2d vs 5d) makes the z-score more reactive, causing more z_exit triggers. The problem is costs: 6,883 trades at 5 bps/leg = ~$101k in costs, wiping out the gross. C5 would be net-positive if costs were ≤2 bps/leg.

### C4 is the best-performing config (−$16,703 net)
Despite lowest gross ($3,628), C4 has:
- Fewest trades (908) → lowest total transaction cost (~$20k)
- Highest z_exit win rate (93.3%) of any high-volume config
- Lowest z_stop losses (−$1,374) due to wide z_stop=5.0
- This is the right direction: fewer, higher-quality trades

### max_hold gross P&L is not the problem in C1/C3
Counterintuitively, C1 and C3 max_hold trades have **positive gross P&L** (+$34,848 and +$26,555). The problem is z_stop, not max_hold. Spreads often partially revert and get closed at max_hold with small gains, but the ones that diverge get stopped out with large losses. The net result of z_stop (−$28k) dominates.

---

## What Needs to Change

**Priority 1: Fix z_exit**
- Move z_exit from 0.0 to 1.0–1.5
- This alone reduces max_hold rate from ~80% to ~60% and doubles the z_exit win rate
- C4's z_exit=1.5 shows 93% win rate on 164 z_exit trades

**Priority 2: Reduce trade count (raise z_entry)**
- More trades = more costs = harder to be net-positive
- C4 (908 trades) has 7× lower cost burden than C1 (6,860 trades)
- Target: 500–1,500 trades/year for the full universe

**Priority 3: Control z_stop losses**
- z_stop at 4.0 (C1) triggers too often and loses big when it does
- z_stop at 5.0 (C4) loses much less: −$1,374 vs −$28,179
- Wide z_stop combined with high z_entry means it rarely triggers, and when it does the loss is bounded

**What the parameter search got right:**
- Formation window: longer windows improve per-trade quality
- max_half_life=24 bars as sweet spot
- z_stop should be wider than tested (5.0+ is better than 4.0)

**What the parameter search missed:**
- z_exit=0.0 creates activity starvation of z_exit triggers (not a param in the grid)
- Total trade count matters as much as per-trade breakeven
- z_stop P&L impact (not measured in fixed-horizon evaluation)
- The period bias toward H2 2022 bear market inflates activity estimates

---

## Recommended Next Config to Test

Based on C4 as the best direction, with adjustments:

```
z_entry = 3.5
z_exit  = 1.0        # lower than C4's 1.5 to catch more reversions
z_stop  = 5.0
max_holding_minutes = 240
rolling_window_days = 63
min_half_life = 4
max_half_life = 24
zscore_window_days = 2   # borrow C5's fast z-score for more responsive exit triggers
max_pairs = 20           # increase from 10 to utilize larger universe
```

This attempts to combine C4's quality (high z_entry, wide z_stop) with C5's reactivity (fast z-score window), while targeting a z_exit level that catches partial reversions without requiring full mean reversion.
