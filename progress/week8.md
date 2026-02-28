## Progress Since Week 7

Week 7 ended with a working signal pipeline and grid-search parameter selection, but only bar-level forward-return evaluation — a known biased proxy for actual trading P&L. The central challenge was implementing a genuine trade-level backtest. That work is now complete, along with a systematic validity investigation of the results.

---

### What Has Been Built

**Portfolio Backtester** (`strategy/backtester.py`)

The main deliverable is a full portfolio simulation engine. It runs day-by-day over an arbitrary date range, recalibrating cointegrated pairs each morning over a trailing 84-calendar-day formation window, then executing a bar-by-bar signal loop intraday. The key design decisions:

- **No-lookahead execution**: entries and exits execute at the midpoint of the *next* bar after a signal fires, not the bar where it is observed. EOD and max-hold forced closes use the current bar's close price.
- **Multi-day position holding**: positions are *not* force-closed at the end of each trading day. Because the estimated half-lives of cointegrated pairs (14–28 bars at 15-minute resolution, or ~3.5–7 trading hours) exceed a single session, allowing overnight carry gives the spread time to mean-revert. The final trading day and any position exceeding `max_holding_days=5` are still force-closed.
- **Late-day entry gate** (`min_bars_remaining=8`): new entries are blocked with fewer than 8 bars left in the session (~2 hours), preventing positions that open with insufficient time for mean reversion and that would inevitably carry overnight by necessity rather than by choice.
- **Dollar-neutral position sizing**: `notional_per_pair = portfolio_value / max_pairs`. Shares in leg A are `notional / price_a`; shares in leg B are `hedge_ratio × shares_a`. This scales positions with the growing portfolio and maintains approximate dollar neutrality within each pair.
- **Four-leg transaction costs**: each trade incurs costs on entry and exit for both legs (`cost_bps` per leg). At `max_pairs=20` and a $100K starting portfolio, this amounts to roughly $24 per trade at 1.5 bps.
- **Exit reason tracking**: every closed trade is tagged `z_exit` (mean reversion), `z_stop` (stop-loss), `max_hold` (time-stop), or `eod` (final trading day).

**Three bugs found and fixed during implementation**

1. *Last-bar fallback*: when z_exit fired at the final 15-min bar of a non-final trading day, `next_bar_ts` was None, so the code fell back to the current bar's midpoint — effectively executing an intraday exit instead of carrying overnight. Fixed by computing `force_close` before the signal-transition check and guarding signal checks with `not is_last_bar`.

2. *Direction-flip overnight gap*: if a position's z-score crossed z_exit and immediately re-entered z_entry in the opposite direction between sessions (rare overnight gap), the backtester would miss the exit. Fixed by adding a direction-reversal check alongside the `sig == 0` check.

3. *Stale carry-over signal on bar_idx=0*: the z-score lookback window generates signals for prior days that can leak a non-zero `prev_sig` into the first bar of the trading day. Added a hard z-score threshold check at `bar_idx == 0` to ensure entries only occur on genuine fresh crossings.

**Systematic validity investigation** (`scripts/debug.py`)

After observing a Sharpe ratio of ~3.5 at 1.5 bps, a structured Phase 1 investigation was conducted to rule out inflated results:

- *Exit-reason breakdown*: z_exit trades produce 98.3% win rate on same-day exits — mathematically correct by construction (when z_exit fires, the spread has by definition reverted toward zero, so gross PnL > 0 unless costs dominate). Not a bug.
- *Last-bar fallback impact*: only 12 of ~1,700 z_exit trades hit the 15:45 bar; those 12 have *lower* average PnL ($144) than normal exits ($186), confirming the fallback was a minor drag, not inflation.
- *Cost sensitivity*: gross PnL was $63K over the full period; transaction costs at 1.5 bps = $22K; break-even is approximately 3.5 bps. At 5 bps the strategy loses ~11%.
- *No lookahead bias* confirmed: entry signal on bar i → execution at bar i+1 throughout.

**Strategy notebook** (`notebooks/strategy.ipynb`)

The analysis notebook was updated to match the backtester's current interface. `BACKTEST_PARAMS` now explicitly includes `max_holding_days` and `min_bars_remaining`. The summary section was expanded from a single EOD-exit percentage to a full exit-reason breakdown (z_exit / z_stop / max_hold / eod). The parameter table was updated with a rationale column documenting the reasoning behind each value.

**Test suite**

26 tests pass across all modules: backtester, cointegration, config, evaluation, and signals.

---

### Current Results (2022-07-01 → 2024-12-31, 1.5 bps)

| Metric | Value |
|---|---|
| Total return | ~+41% |
| Annualised Sharpe | ~3.5 |
| Max drawdown | ~−8.6% |
| Total trades | ~2,300 |
| Win rate | ~64% |
| Avg hold | ~1,280 min (~21 h, multi-day) |
| Break-even cost | ~3.5 bps per leg |

At the notebook's default `cost_bps=5.0` the strategy is unprofitable. The signal has alpha, but it is cost-sensitive.

---

### Future Challenges

**1. Cost realism is the defining constraint**

The profitable regime requires costs below ~3.5 bps per leg. Institutional all-in costs (including spread, borrow, and market impact) are realistically 2–5 bps for large-cap equities at the notional sizes here (~$5K per leg). Retail costs are higher. The strategy needs either a higher z_entry threshold to require larger mispricings that can absorb more friction, or a switch to a lower-frequency bar size (e.g. daily) where the edge-per-trade is larger relative to fixed costs.

**2. Walk-forward out-of-sample validation**

All parameter selection was done on H2 2022 (the grid-search sample period), and the 2023–2024 portion of the backtest is loosely out-of-sample — but parameters were still known when the backtest was constructed. A proper walk-forward is needed: fix parameters on a rolling training window, test on the immediately following window, and repeat. This is the minimum bar for claiming the strategy generalises.

**3. Overnight and gap risk**

Multi-day holding exposes positions to overnight gaps, weekend risk, and event risk (earnings, macro surprises). The current model has no gap protection: a position entered on a Monday can hold through a Friday close and face a gap on Monday open. A production system would need gap-risk limits, earnings-date blackouts, and possibly a tighter `max_holding_days` combined with earlier entries.

**4. Cointegration instability**

Pairs are recalibrated daily over a trailing window, but there is no mechanism to detect when cointegration has broken down structurally (e.g. a sector demerger, major acquisition, or prolonged regime shift). Pairs that pass the ADF test on the formation window can fail to revert during the trading day if the underlying relationship has changed. A robustness check — requiring cointegration to hold on both halves of the formation window, or using a rolling ADF p-value trend — could reduce exposure to decaying pairs.

**5. Computational cost of daily recalibration**

Running `find_cointegrated_pairs` on every trading day over 629 days is the dominant runtime cost. For a production or research loop with many parameter combinations, this becomes prohibitive. The grid search cached cointegration results by (formation window, day), but the full backtest does not. Pre-computing and caching the pair universe for each day would make iterative experimentation tractable.

**6. Correlation between pairs**

The current selection logic chooses the top pairs by p-value within sectors, but pairs within the same sector often share a common factor (e.g. three semiconductor pairs all move with NVDA). Concurrent positions in correlated pairs compound drawdowns without proportionally increasing expected return. A portfolio-level correlation constraint, or explicit orthogonalisation of the pair universe, would improve risk-adjusted returns.
