# Intraday Statistical Arbitrage on US Large-Cap Equities

**Graduate Seminar — March 2026**

> **Thesis:** Cointegrated large-cap equity pairs exhibit predictable intraday spread reversion.
> A simple z-score signal captures this alpha, but converting gross signal edge into net profitability
> requires controlling hold-period risk — filtering pairs whose spreads don't revert within the trading day
> is more important than minimizing transaction costs.

---

## Slide 1 — Motivation & Thesis

**What is statistical arbitrage?**

- Exploit temporary price dislocations between *related* assets
- No directional bet on the market — profit from *relative* mispricing
- Classic example: if stock A and B historically move together, trade when they diverge

**The exploitable inefficiency**

- Cointegrated pairs share a long-run equilibrium: `log(A) ≈ β · log(B) + constant`
- When the spread diverges beyond 3–4 standard deviations, informed traders correcting it create a predictable return
- Importantly: the spread's stationarity is *testable* and *measurable* — not a belief

**Why intraday / large-cap?**

- Large-cap equities: deep liquidity, minimal market impact at small size
- Intraday: avoid overnight gap risk and earnings surprises
- 15-min bars: enough signal frequency, manageable transaction cost per trade

**The central question:** Can we construct a cointegration-based signal that earns more than it costs to trade?

---

## Slide 2 — Data & Universe

**Universe: ~190 US large-cap equities, 8 consolidated sectors**

| Sector | Tickers | Notes |
|--------|---------|-------|
| Technology | 38 | Software, IT services, semiconductors merged — same GICS IT parent |
| Consumer | 38 | Discretionary + Staples merged — shared demand driver |
| Real Assets | 28 | Utilities + REITs merged — both rate-sensitive, asset-heavy |
| Financials | 20 | Banks, insurers, asset managers |
| Healthcare | 20 | Pharma, biotech, managed care, devices |
| Industrials | 20 | Aerospace, defense, machinery, transportation |
| Energy | 18 | Integrated, E&P, oilfield services, midstream |
| Communication Services | 18 | Telecom, media, entertainment |

- Only **within-sector pairs** tested: cross-sector pairs share fewer common factors, weaker cointegration
- **~2,660 candidate pairs** across all sectors (C(n,2) per sector, summed)
- Source: EODHD API — 1-minute OHLCV data, resampled to **15-min bars**
- Market hours: 09:30–16:00 ET (26 bars/day), split-adjusted; pre/post-market filtered

**Backtest period:** 2022-07-01 → 2024-07-01 (504 trading days, 2 years)

**Why 15-min bars?**
At 1-min granularity, execution latency dominates and noise overwhelms signal.
At daily bars, per-trade costs require much larger spread movements to overcome.
15-min is the sweet spot: ~26 bars/day, manageable signal frequency, achievable execution.

---

## Slide 3 — Pair Selection: Cointegration

**Recalibrated every trading day using a rolling 42-calendar-day window (~30 trading days)**

### Step 1 — Estimate the hedge ratio (OLS)

```
Regress: log(price_A) = β · log(price_B) + ε
Spread:  s_t = log(A_t) − β · log(B_t)
```

- Hedge ratio β makes the pair dollar-neutral in log-space
- OLS minimizes squared residuals → β is the cointegrating coefficient

### Step 2 — Test for stationarity (ADF test)

- **H₀:** spread has a unit root (non-stationary → no mean reversion → don't trade)
- **Accept pair if:** ADF p-value ≤ 0.05
- **Half-life filter:** AR(1) fit on spread gives the e-folding mean reversion time
  - Require: **4–18 bars** (~1–4.5 hours at 15-min frequency)
  - Pairs with half-life > 18 bars are unlikely to revert within a single trading day

### Step 3 — Rank and select

- Sort accepted pairs by p-value ascending (lowest p = strongest cointegration)
- Weight by inverse rank: rank 1 gets the most weight
- Dollar-neutral sizing: `shares_A = notional / price_A`, `shares_B = β · shares_A`

**Why daily recalibration?**
Cointegration relationships drift. A pair that was strongly cointegrated 42 days ago may have structurally broken.
Daily recalibration using a rolling window catches these breaks before they cause large losses.

---

## Slide 4 — Signal Generation: Z-Score State Machine

**Intraday z-score** for each pair at each 15-min bar:

```
z_t = (s_t − μ_window) / σ_window
```

- Rolling window: short enough to stay responsive to recent spread behavior, long enough to estimate variance reliably
- Uses only past bars — no look-ahead in normalization

**Stateful position machine** — one position per pair at a time:

| Condition | Action | Rationale |
|-----------|--------|-----------|
| `z > +z_entry` | Short spread (sell A, buy B) | Spread too wide, expect reversion down |
| `z < −z_entry` | Long spread (buy A, sell B) | Spread too narrow, expect reversion up |
| `\|z\| ≤ z_exit` | Close position | Reversion target reached |
| `\|z\| ≥ z_stop` | Close position (stop loss) | Spread running away |
| Hold > `max_holding` | Close position (time stop) | Spread unlikely to revert in time |
| Last bar of day | Close all | Avoid overnight exposure |

**No look-ahead bias:** signal fires at bar T close → execution at bar T+1 midpoint `(high + low) / 2`.

*Specific parameter values (z_entry, z_exit, z_stop, zscore window, max_holding) are determined by grid search — see slide 6.*

---

## Slide 5 — Implementation: Backtest & Execution

**Two design decisions worth explaining:**

### 1. Execution model

```
Bar T close:  z-score computed → signal fires
Bar T+1:      trade executed at midpoint (high+low)/2
```

- 1-bar lag consistently applied to entries and exits
- EOD close at bar T close price (market-on-close)
- Midpoint is an approximation — an optimistic upper bound on fill quality
  (actual fills depend on order type: market orders pay the spread, limits may not fill)

### 2. Cost model: four legs per round-trip

| Leg | Description |
|-----|-------------|
| Entry A | Buy (or sell) stock A |
| Entry B | Sell (or buy) stock B |
| Exit A | Reverse position in A |
| Exit B | Reverse position in B |

- Cost per leg: `cost_bps × notional_per_leg`
- Calibrated at **5 bps per leg** (representative of large-cap bid-ask spread + market impact)
- 5 bps per leg = 20 bps round-trip. For a $100 stock, this covers 1–2 cents of bid-ask plus typical
  timing slippage at 15-min execution. Conservative but defensible for large-cap.

---

## Slide 6 — Parameter Selection

**The strategy has 7 tunable parameters. How do we choose them?**

### Parameter space

| Group | Parameter | Role |
|-------|-----------|------|
| Formation | `rolling_window_days` | How much history to estimate cointegration |
| Formation | `min_half_life` | Fastest acceptable mean reversion speed |
| Formation | `max_half_life` | Slowest acceptable (must revert intraday) |
| Formation | `p_value_threshold` | Strictness of cointegration acceptance |
| Signal | `zscore_window_days` | Rolling window for z-score normalization |
| Signal | `z_entry` | How extreme does the spread need to be to enter |
| Signal | `z_exit` | How far does the spread need to revert to exit |
| Signal | `z_stop` | Stop-loss: exit if spread moves further against us |
| Signal | `max_holding_minutes` | Time stop: forced exit |

### Grid search design

**Key insight:** formation params (ADF tests across ~2,660 pairs) are expensive to compute.
Signal params (rolling z-score, state machine) are cheap per bar.

- **Stage 1:** Build cointegration pairs cache for each calendar day (rolling window).
  Cache is keyed by `(formation_params, date)`. Computed once, reused across all signal combos.
- **Stage 2:** Sweep signal parameters in parallel. Each worker iterates all cached days,
  generates signals, evaluates IC and returns.

### Evaluation metrics per combo

| Metric | Purpose |
|--------|---------|
| `mean_ic_gross` | Spearman(signal, fwd_return) — is the signal directionally correct? |
| `ic_t_stat` | Statistical significance of IC |
| `mean_gross_return` | Per-signal P&L before costs |
| `mean_net_return` | After 4-leg round-trip costs |
| `breakeven_bps` | Cost per leg at which net return = 0 |
| `mean_hit_rate` | % of signals where direction was correct |

**Phase 1 (current):** Fix formation params at defaults, sweep signal params (144 combos).
Primary question: is IC > 0 at any horizon? If yes, cointegration is directionally valid.

**Phase 2:** Full grid including formation params (~7,700 combos).

*Findings from this search are discussed in slide 7.*

---

## Slide 7 — Results & Issues

### In-sample results: 2022-07-01 → 2024-07-01 | $100k capital | 5 bps/leg

**Exit reason breakdown (7,820 total trades)**

| Exit reason | Trades | Share | Gross P&L | Win rate | Interpretation |
|---|---|---|---|---|---|
| **z_exit** (reversion reached) | 1,523 | 19.5% | +$64,976 | **86.7%** | Signal works |
| **max_hold** (time stop) | 6,172 | 78.9% | −$59,170 | 34.8% | Spread didn't revert in 120 min |
| **eod** (forced close) | 125 | 1.6% | −$795 | 28.0% | Late entries |
| **Total** | **7,820** | | **+$5,012 gross / −$42,966 net** | | |

### The core finding

**The good:** when a trade exits via z_exit, win rate is 86.7%.
This is direct evidence the cointegration signal is *valid* — when `|z| ≤ z_exit` fires,
the spread has by definition reverted, and the position is almost always profitable.

**The problem:** 79% of trades *never reach the exit target* within 120 minutes.
Of those time-stopped trades, only 35% are profitable — most are still wide or wider.
The signal identifies that *some* extreme spreads revert. It doesn't yet identify *which ones* will.

### Cost sensitivity

The aggregate break-even is ~0.5 bps/leg — but this is misleading.

| Trade population | Count | Gross P&L | Break-even |
|---|---|---|---|
| z_exit exits (reversion completed) | 1,523 | +$64,976 | ~7 bps/leg (viable) |
| max_hold + eod (time-stopped) | 6,297 | −$59,964 | Never profitable |

The 7 bps/leg break-even on z_exit trades is comfortably above realistic costs.
The time-stopped trades are structurally unprofitable regardless of cost reduction.
**This is an engineering problem, not a cost problem.**

### Parameter findings (from grid search)

- **z_entry is the dominant parameter:** threshold must be high enough to select only extreme dislocations. Low z_entry (1.5–2.0) generates many trades with negative mean return after costs.
- **Short max_holding outperforms:** 60–120 min beats 240–390 min. Positions that don't revert quickly tend not to revert at all — holding longer accumulates losses.
- **Formation window ~42 days > longer:** shorter windows are more responsive to structural changes in pair relationships.
- **Activity vs. selectivity trade-off:** narrow half-life filter + high z_entry → very few signals per day. Enough trades needed for reliable statistics.

### Issues and next steps

**Issue 1 — In-sample only.** All parameters were selected on the same data used to evaluate them.
Walk-forward OOS validation (42-day rolling train → 21-day test) is the critical next step.
The 87% z_exit win rate should hold OOS if the signal is genuine — that's the key test.

**Issue 2 — Max_hold rate too high.** 79% of positions fail to revert in 120 min.
Approaches: (a) stricter half-life filter at pair selection; (b) early exit signal when
spread shows no reversion in first N bars; (c) z_stop to cut runaway positions earlier.

**Issue 3 — Parameter sensitivity.** z_entry=4.0 generates ~1,500 trades over 2 years — thin
statistical base. z_entry=2.5 generates many more but at negative expected return. The right
threshold trades off signal quality vs statistical reliability.

**Issue 4 — Execution model is optimistic.** Midpoint execution (high+low)/2 is not
achievable in practice — market orders pay the spread, limit orders risk non-fill.
A more realistic model would apply an additional 1–3 bps adverse fill assumption.
