# Strategy Improvement Approaches

Each approach includes the theoretical basis, what it targets in the current strategy, and a concrete testing method. Ordered roughly from highest to lowest expected impact.

---

## 1. Reduce z_stop Drag with a Time-Decaying Stop

**What it targets:** z_stop exits account for ~18% of trades but only ~25% win rate — the largest single source of losses. At z_stop=4.5 the strategy waits for extreme spread widening before exiting, by which point the loss is already severe. The current stop is a blunt instrument: fixed regardless of how long the position has been held.

**Theory:** In a mean-reverting system, the longer a spread has been at extreme levels without reverting, the less likely it is to be a temporary fluctuation and the more likely it is to represent a structural break. A time-decaying stop encodes this: the tolerated z-score distance shrinks as the position ages. Formally, if the half-life is τ bars, a position that has survived 2τ bars without reverting has roughly a 25% prior probability of being a genuine regime shift — tighter stops are warranted.

**Test:**
- Introduce `z_stop_initial` (e.g. 4.5) and `z_stop_final` (e.g. 3.0) that linearly decay over `max_holding_days`
- At day d: `z_stop_d = z_stop_initial - (z_stop_initial - z_stop_final) * (d / max_holding_days)`
- Grid: `z_stop_initial` ∈ [4.0, 4.5], `z_stop_final` ∈ [2.5, 3.5]
- Measure: change in z_stop win rate, total z_stop drag, and overall Sharpe

---

## 2. Cointegration Stability Filter (Require Both Half-Windows to Pass ADF)

**What it targets:** Pair selection currently runs ADF on the full 84-day formation window. A pair can pass on the aggregate window while having wildly different behaviour in the first and second halves — meaning the relationship is intermittent or recently broken. These pairs have high cointegration instability and are the primary source of stop-loss losses.

**Theory:** A cointegrating relationship that is stable over subperiods is more likely to hold during the trading window. Requiring both the first 42 days and the second 42 days of the formation window to independently pass ADF (at p≤0.05) selects for structurally persistent relationships, not statistical flukes. This is analogous to cross-validation applied to cointegration.

**Test:**
- Modify `analysis/cointegration.py` `find_cointegrated_pairs()` to run ADF on `[t-84d, t-42d]` and `[t-42d, t-0d]` separately
- Require both p-values ≤ 0.05 (strict) or ≤ 0.10 (relaxed)
- Compare: n_pairs per day, z_stop rate, overall Sharpe vs baseline
- Expect: fewer pairs per day (tighter filter), but better win rate on z_stop exits

---

## 3. Adaptive Hedge Ratio via Kalman Filter

**What it targets:** The OLS hedge ratio is estimated once on the 84-day formation window and then held fixed for the entire trading day. For pairs where the cointegrating relationship drifts slowly (common in sectors with evolving relative fundamentals), a stale hedge ratio makes the spread appear to diverge when it is actually tracking a shifted equilibrium. This produces spurious z-score extremes that trigger entries into positions that will not revert.

**Theory:** The Kalman filter treats the hedge ratio as a latent state that evolves according to a random walk (`β_t = β_{t-1} + ε_t`). The observation equation is `price_A = β_t * price_B + spread_t`. This continuously updates the hedge ratio with each new bar, ensuring the spread is always measured relative to the current cointegrating vector. The State Space / DLM literature (Hamilton 1994, Pairs Trading with Kalman Filters — Pole & MacQueen) shows that Kalman-filtered spreads have lower autocorrelation in the residuals and better stationarity properties than rolling-OLS spreads.

**Test:**
- Implement a `compute_kalman_spread(prices_a, prices_b, delta=1e-5)` function in `analysis/signals.py` or a new `analysis/kalman.py`
- `delta` controls how fast the hedge ratio adapts (smaller = slower)
- Replace `compute_spread()` call with `compute_kalman_spread()` in `generate_pair_signals_for_day()`
- Compare: half-life stability across pairs, z_stop rate, Sharpe
- Grid: `delta` ∈ [1e-6, 1e-5, 1e-4] to find the adaptation speed that matches the pairs' actual drift rate

---

## 4. Partial Exits (Scale Out at Multiple z-Score Thresholds)

**What it targets:** Currently all shares are exited at once when `|z| ≤ z_exit`. This is suboptimal in two ways: (a) exits at z=0.5 often occur when the spread is still moving toward zero and leaves money on the table, and (b) once a position is fully closed, a re-entry is needed if the spread overshoots and re-extends.

**Theory:** In a mean-reverting spread, the posterior probability of continued reversion increases as the spread approaches zero. Partial exits (e.g. 50% at z=1.5, remaining 50% at z=0.5) harvest the early reversion while maintaining exposure to the full reversion. This is analogous to delta hedging in options: scaling out as the underlying moves toward the target reduces path dependency. The tradeoff is additional transaction costs on the partial exit.

**Test:**
- Add `z_exit_partial` parameter and `partial_exit_fraction` (e.g. 0.5) to `PositionState` and the backtester
- When `|z| ≤ z_exit_partial`, close `partial_exit_fraction` of the position; when `|z| ≤ z_exit`, close the remainder
- Cost is incurred twice (two exit events)
- Grid: `z_exit_partial` ∈ [1.5, 2.0] with `partial_exit_fraction` ∈ [0.3, 0.5]
- Measure: avg gross P&L per trade, total cost, Sharpe vs single-exit baseline

---

## 5. Regime Filter: Suspend Trading During High-Volatility Regimes

**What it targets:** Cointegration is a long-run equilibrium concept derived from historical price relationships. During market stress events (VIX > 30, index drawdowns > 5% in a week), spread volatility spikes, half-lives lengthen unpredictably, and pairs that were cointegrated stop mean-reverting on their normal timescale. The backtester has no mechanism to suspend trading during these periods.

**Theory:** There is strong empirical evidence (Mitchell & Pulvino 2001 on merger arbitrage; Gatev, Goetzmann & Rouwenhorst 2006 on pairs trading) that convergence-arbitrage strategies underperform in market dislocations because the very events that create wider spreads also impair the arbitrageur's ability to hold positions and reduce the probability of reversion within any reasonable holding window. A simple VIX-level filter avoids the worst drawdown periods at the cost of reduced trade frequency.

**Test:**
- Fetch VIX daily from EODHD (or download from Yahoo Finance as a CSV) and store in `data/raw/`
- In `run_backtest()`, at the SOD step, check VIX level for the current trading day
- If `vix_close > vix_threshold`, skip new entries for that day (allow existing positions to continue)
- Grid: `vix_threshold` ∈ [25, 30, 35] vs no filter
- Measure: trades skipped per threshold, drawdown change, Sharpe change
- Note: the 2022 bear market had sustained VIX 25–35; a high threshold could eliminate a large portion of the sample

---

## 6. Beta-Neutral Position Sizing

**What it targets:** Current position sizing is dollar-neutral (equal notional in each leg). Within-sector pairs share a common market beta (e.g. two semiconductor stocks are both highly correlated to the Nasdaq). If SPY sells off 2%, both legs move — but not equally if their betas differ. The spread picks up a systematic component that is not mean-reverting and that inflates the z-score, creating false entries and inflating losses on z_stop exits.

**Theory:** True pairs trading isolates the idiosyncratic spread between two stocks. Dollar-neutral hedging removes the price-level effect but not the market-factor exposure. Beta-neutral sizing (`shares_b = (beta_A / beta_B) * shares_a * hedge_ratio_ols`) removes the first-order market factor. The remaining spread is the idiosyncratic component, which should be more stationary. This is the approach in Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market."

**Test:**
- Compute 60-day rolling beta of each stock to SPY from the EOD data (already fetched)
- In the backtester, compute `beta_ratio = beta_A / beta_B` at the start of each day
- Adjust `shares_b = hedge_ratio * shares_a * beta_ratio`
- Compare: spread stationarity (ADF p-value distribution before/after), z_stop rate, Sharpe
- Simpler variant: use sector ETF as the factor (XLK for Technology, XLF for Financials, etc.)

---

## 7. Entry Timing Filter: Avoid First 15 Minutes

**What it targets:** 52% of current entries occur at 9:45 (the second 15-min bar, first bar after open). The first 15 minutes of trading are known to have elevated bid-ask spreads, higher-than-normal volatility, and order imbalances from overnight order flow. Z-scores computed on opening prices are noisier and more likely to be driven by transient microstructure effects rather than genuine cointegration deviations.

**Theory:** Amihud & Mendelson (1987) and subsequent market microstructure literature document that bid-ask spreads are widest and price discovery is most chaotic in the first 15–30 minutes of the trading session. A z-score computed using these noisy opening prices overstates the true spread deviation. Filtering out entries in the first 30 minutes (2 bars at 15-min) reduces false entries at the cost of missing genuine opening-gap reversions. The morning entry effect could also be investigated: are 9:45 entries genuinely profitable (overnight gaps create real mispricings) or are they noise-driven losses?

**Test:**
- Segment entry analysis: compare `avg_net_pnl` and `win_rate` for entries at 9:45 vs 10:00+ vs 10:30+
- If 9:45 entries underperform, add a `min_entry_bar` parameter to `run_backtest()` that blocks entries before a certain time
- Grid: skip first [0, 1, 2] bars of the trading day
- This is a one-line filter in the entry section of the backtester: `if bar_ts.time() < dt.time(10, 0): continue`

---

## 8. Pair Universe Refresh: Require N Consecutive Cointegration Days

**What it targets:** A pair that passes ADF on one day's formation window may fail the next day and pass again the following day. Trading such intermittently cointegrated pairs increases exposure to false signals. Requiring that a pair has been continuously cointegrated for at least N trading days before it is traded would filter out borderline relationships.

**Theory:** Persistence of cointegration is a stronger signal than a single point-in-time ADF test. If a relationship has been stable for 10+ consecutive days, it is more likely to represent a genuine economic tie (shared revenue exposure, common input costs, regulatory linkage) rather than a statistical coincidence on one specific 84-day window.

**Test:**
- Maintain a `pair_consecutive_days` counter in the backtester (or as a pre-computed Parquet cache)
- Only allow entries for pairs that have been cointegrated for ≥ N consecutive trading days
- Grid: N ∈ [0, 3, 5, 10] (0 = current behaviour)
- Measure: n_pairs per day, z_stop rate, overall Sharpe

---

## 9. Transaction Cost Reduction: Limit Order Entry

**What it targets:** The current model assumes market-order execution at the midpoint of the next bar. Market orders incur the full bid-ask spread as additional slippage. At the notional sizes involved (~$5K per leg), the effective all-in cost for large-cap equities with tight spreads could be 1–2 bps lower with limit orders that passively provide liquidity rather than taking it.

**Theory:** Passive limit order execution earns the bid-ask spread rather than paying it. For a liquid large-cap stock (AAPL, MSFT) with a 1-cent spread and a $180 price, a limit order saves ~0.55 bps per leg vs a market order. Across 4 legs per trade, that is ~2 bps per trade — which moves the break-even from ~3.5 bps to ~5.5 bps, making the strategy viable at realistic broker costs. The risk is fill uncertainty: limit orders may not fill if the price moves away, causing missed entries.

**Test:**
- In the backtester, add a `use_limit_orders` flag
- Limit order execution model: entry fills at `bar_open` (no slippage) rather than `(high + low) / 2`
- This is an optimistic approximation; realistic model would use `low` for longs and `high` for shorts on the next bar (fills if price touches limit level)
- Compare: avg execution price vs midpoint, effective cost reduction, number of missed entries (bars where high < limit_price)
- Expected benefit: 1–2 bps per leg reduction in effective cost

---

## 10. Multi-Factor Pair Scoring

**What it targets:** Currently pairs are ranked by ADF p-value only, and the top pairs by p-value fill the `max_pairs` slots. P-value alone does not capture half-life (speed of reversion), spread volatility (magnitude of opportunity), or recent signal quality. A composite score could allocate more capital to pairs that are currently offering better risk-adjusted opportunities.

**Theory:** The expected return of a pairs trade scales with the z-score at entry (higher entry = larger expected reversion) and inversely with the half-life (shorter half-life = faster reversion = less overnight risk). A score of `z_entry / sqrt(half_life)` captures both dimensions and is proportional to the Sharpe of an idealized mean-reversion trade. This is related to the Ornstein-Uhlenbeck process Sharpe derivation in Bertram (2010) "Analytic Solutions for Optimal Statistical Arbitrage Trading."

**Test:**
- Compute per-pair score at SOD: `score = abs(current_z) / sqrt(half_life_bars)` (or variants)
- Rank pairs by score rather than p-value for entry priority and position sizing
- Position size proportional to score (rather than equal notional): `notional_i = capital * score_i / sum(scores)`
- Compare: avg gross P&L per trade, concentration risk (max single-pair notional), Sharpe
- Simpler variant: just add half-life as a second sort key after p-value

---

## Summary Table

| Approach | Targets | Complexity | Expected Impact |
|---|---|---|---|
| 1. Time-decaying stop | z_stop drag | Low | High |
| 2. Cointegration stability filter | Pair quality | Low | High |
| 3. Kalman filter hedge ratio | Stale hedge ratio | High | High |
| 4. Partial exits | P&L capture | Medium | Medium |
| 5. Regime filter (VIX) | Drawdown | Low | Medium |
| 6. Beta-neutral sizing | Systematic noise | Medium | Medium |
| 7. Entry timing filter | Microstructure noise | Low | Low–Medium |
| 8. Consecutive cointegration days | Pair quality | Low | Medium |
| 9. Limit order execution | Transaction costs | Medium | High (if viable) |
| 10. Multi-factor pair scoring | Capital allocation | Medium | Medium |

**Recommended starting order**: 1 → 2 → 7 → 5 (lowest complexity, highest expected impact). Approach 3 (Kalman filter) has the highest ceiling but requires the most implementation work and should be validated on a subset of pairs first.
