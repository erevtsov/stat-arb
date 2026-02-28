# Experiments & Progress Log

Chronological record of work done, decisions made, and results observed across the project.

---

## 2026-02-10 — Project Scaffolding

**Commits:** `d044b4c` (initial system), `79ed59fd` (switch to uv)

Built the full skeleton of the project in one pass. The system was designed end-to-end from day one:

- **Data layer** (`scripts/fetch_data.py`): EODHD API client fetching 1-minute intraday OHLCV bars with 120-day chunking per request. Raw Parquet files stored per-ticker.
- **Preprocessing** (`analysis/preprocessing.py`): market-hours filtering, multi-timeframe resampling (1min → 5min, 15min, 1hour, daily) using Polars `group_by_dynamic`.
- **Cointegration** (`analysis/cointegration.py`): Engle-Granger two-step — OLS hedge ratio, ADF stationarity test on residual spread, p-value and half-life filtering.
- **Spread + signals** (`analysis/spread.py`, `strategy/signals.py`): rolling z-score, stateful entry/exit/stop state machine.
- **Backtester** (`strategy/backtest.py`): vectorized engine (not yet trade-level simulation).
- **Portfolio** (`strategy/portfolio.py`): multi-pair capital allocation.
- **Metrics** (`utils/metrics.py`): Sharpe, drawdown, win rate, profit factor.
- **Config** (`utils/config.py`): typed dataclasses for all parameters.
- **Universe**: 37 tickers across initial sectors.

Tooling: switched from requirements.txt to `uv` for dependency management.

---

## 2026-02-11 — Universe Expansion + Data Improvements

**Commits:** `add1695` (universe + EOD), `ab85999` (date range), `a0d8d89` (env var), `76e69b2` (build system), `fbe5747` (multi-threaded fetch)

- Universe expanded from 37 → **100 tickers across 11 GICS sectors** (Technology, Semiconductors, Financials, Healthcare, Consumer Discretionary, Consumer Staples, Energy, Industrials, Communication Services, Utilities, REITs). 7–10 liquid large-cap names per sector for sufficient within-sector pair coverage.
- Added **EODHD EOD endpoint**: daily adjusted_close for split/dividend adjustment, splits history, dividends history.
- **Split-adjustment** for intraday data using EODHD split history — critical for pairs whose prices have adjusted between the formation window and trading day.
- Fetch script made **multi-threaded** (5 workers).
- Date range extended to cover 2023–2024 for proper out-of-sample evaluation.

---

## 2026-02-14 — Data Quality Fixes + Architecture Reset

**Commits:** `76ad3b1`, `571a24c`, `adced40`, `a15cbac`, `891d326`, `14d7c7a`, `de51d64`, `b4db985`

This day had several significant iterations:

### After-hours filtering
Market data from EODHD includes pre-market and after-hours bars. These must be stripped to NYSE trading hours (09:30–16:00 ET). Added filtering in `fetch_data.py`, later refactored into `preprocessing.py` as the correct architectural home (raw data stays raw; filtering is a preprocessing concern).

### Cointegration lookahead bias fix (major)
Identified that the initial cointegration implementation had a lookahead bias: the formation window used to compute hedge ratios and ADF statistics was not strictly prior to the trading day. Fixed by enforcing that `formation_end = trading_day - 1`. This is the correct Engle-Granger setup for a walk-forward simulation. `pandas_market_calendars` added for NYSE holiday awareness.

### Architecture reset
Old strategy modules (`strategy/backtest.py`, `strategy/portfolio.py`, `strategy/signals.py`, `utils/metrics.py`, `analysis/spread.py`) were deleted in a clean reset. The vectorized backtester was insufficient for the intended simulation fidelity. A plan for rebuilding with a trade-level simulation was established.

---

## 2026-02-15 — Hypothesis Framework + New Strategy Architecture

**Commits:** `d8972451` (strategy package + plan), `e610a6b` (hypothesis tests notebook)

Established a formal hypothesis-driven research framework. Nine hypotheses were defined covering signal quality, cost sensitivity, half-life ranges, sector specificity, and out-of-sample performance. A `notebooks/hypothesis_tests.ipynb` was created to test each one systematically.

New strategy package files were scaffolded (later superseded by the Feb 17–24 implementations):
- `strategy/signals.py` — signal generation
- `strategy/backtester.py` — trade-level simulation placeholder
- `strategy/metrics.py` — performance metrics

---

## 2026-02-16 — Timezone Fixes

**Commits:** `91b23c1`, `59ef21e`

Timestamps from EODHD intraday data came through in UTC. Polars datetime handling produced off-by-one-bar errors when NYSE market hours (09:30–16:00 ET) were applied without explicit timezone conversion. Two fix iterations were needed:
1. Convert timestamps to US/Eastern before market-hours filtering.
2. Strip timezone info after conversion (Polars requires timezone-naive datetimes for `group_by_dynamic` resampling).

---

## 2026-02-17 — Signal Evaluation Pipeline + Grid Search Setup

**Commits:** `eab906b` (signals), `0aef166` (deps), `024d3d0` (config refactor), `bab6b55` (evaluation), `4964a81` (tests), `71a4ede` (parallelize)

### Signal module (`analysis/signals.py`)
Full stateful z-score signal machine: entry when `|z| ≥ z_entry`, exit when `|z| ≤ z_exit`, stop-loss when `|z| ≥ z_stop`, time-stop at `max_holding_bars`. Direction (+1 / −1) tracked per position. `generate_pair_signals_for_day()` runs over lookback + current day, outputs `signal_binary` and `zscore` per bar.

### Evaluation module (`analysis/evaluation.py`)
Bar-level forward return computation: for each bar where a signal is active, compute the return over N future bars (holding horizons). Mean net return per bar and Spearman IC (information coefficient) used as ranking metrics. Note: this is a *biased proxy* for trade-level PnL (acknowledged at this stage as a useful approximation for parameter ranking, not for actual performance measurement).

### Config refactor
All parameters moved to typed dataclasses in `utils/config.py`: `PathsConfig`, `ApiConfig`, `CointegrationConfig`, `SignalConfig`, `PortfolioConfig`, `EvaluationConfig`, `UniverseConfig`. Timeframe conversion helpers added (`zscore_window_bars`, `max_holding_bars`, `holding_horizons_bars`).

### Grid search setup (`notebooks/parameter_search.ipynb`)
Grid over: `z_entry` (1.5–3.5), `z_exit` (0.0–1.0), `z_stop` (3.5–5.0), `zscore_window_days` (3–15), `max_holding_minutes` (30–120), `rolling_window_days` (42–84). Evaluation on H2 2022 (Jul–Dec).

**Optimization**: cointegration results pre-computed and cached by `(formation_window_days, trading_day)` before the grid loop, eliminating redundant Parquet reads. Without this, the grid loop took prohibitively long.

**Parallelization**: workers dispatched via `multiprocessing` with `spawn` start method to avoid a fork/thread-pool incompatibility in Polars (Polars maintains its own thread pool and fork causes deadlocks).

### Tests
Full test suite added: `test_cointegration.py`, `test_config.py`, `test_evaluation.py`, `test_signals.py`.

---

## 2026-02-20 — Grid Worker + Signal Bug Fixes

**Commits:** `d1424d1` (signal bugs), `85150ac` (debug script), `856e7cf` (grid worker), `472262` (gitignore)

### Bug fixes in signals and evaluation (`analysis/signals.py`, `analysis/evaluation.py`)
Two bugs fixed:
1. Signal state machine could set `signal = +1` on a bar where the previous signal was already `+1` (re-entry on continuation rather than fresh crossing). Fixed by gating entries on `prev_signal == 0`.
2. Evaluation forward-return computation included bars from the lookback window in the average. Fixed by filtering to current-day bars only before computing forward returns.

### Grid worker script (`scripts/grid_worker.py`)
Extracted the parallel grid computation into a standalone script callable from the notebook via `subprocess`. This allows the parameter search to restart without re-running notebook cells and enables checkpointing.

---

## 2026-02-21 — Parameter Search Results + Strategy Analysis

**Commits:** `b7b1bec` (search results), `d559af3` (config dates), `b34140a` (analysis notebook), `a117d09` (hypothesis analysis), `2e0b87b` (remove old notebook), `83347622` (rename), `08c2d96` (week 7 summary), `362bf48` (progress update)

### Grid search results (key findings)

The grid ran over H2 2022 on 100-ticker universe. Main findings:

- **z_entry is the dominant parameter** (Pearson r = +0.33 with mean net return per bar): entries below z_entry = 2.0 are consistently unprofitable after costs; 2.5–3.0 are in the profitable zone; higher values reduce trade frequency without proportionally improving per-trade returns.
- **Recommended parameters:** `z_entry = 3.0`, `z_exit = 0.5`, `z_stop = 4.5`, `zscore_window_days = 10`, `rolling_window_days = 84`, `max_holding_minutes = 60` (this last parameter was later superseded by multi-day holding).
- IC is positive (>0) at `z_entry ≥ 2.5`, confirming signal has predictive power above those thresholds.
- Formation window: 84 calendar days (~60 trading days) preferred over 42 days; longer windows show more stable cointegration relationships.

### Hypothesis analysis
Written up as annotated notebook cells in `parameter_search.ipynb`. Confirmed:
1. Signal has positive IC at `z_entry ≥ 2.5`
2. Within-sector pairs are necessary (cross-sector pairs are noisy)
3. Half-lives of 14–28 bars (at 15-min) indicate overnight holding is needed for full reversion
4. Transaction costs are the binding constraint — 80 bps round-trip (20 bps/leg) tested in grid is too high for profitable trading

### Week 7 progress file
Documented the state of the project and identified the key challenge: transitioning from bar-level signal evaluation to a genuine trade-level P&L backtest.

---

## 2026-02-24 — Full Portfolio Backtester (Two Iterations)

**Commits:** `4b4d2b2` (initial backtester), `3e9df99` (multi-day holding + bug fixes)

### Iteration 1: EOD-only backtester (`4b4d2b2` — "backtester that loses money")

Built the first genuine trade-level backtester in `strategy/backtester.py`:
- Day-by-day portfolio simulation
- SOD cointegration recalibration over trailing `rolling_window_days`
- Bar-by-bar signal loop: entry on fresh z_entry crossing, exit on signal → 0
- **Execution model**: entry/exit at midpoint of *next* bar (no lookahead); EOD forced close at close price
- Dollar-neutral position sizing: `notional_per_pair = portfolio_value / max_pairs`
- Four-leg transaction costs
- `max_pairs` slot limit (top pairs by p-value get priority)

**Result at `cost_bps=5.0`**: losing strategy (as suggested by the grid search cost analysis). This is correct behavior — gross PnL per trade is too small at 5 bps to be profitable.

Key design decision at this stage: positions were **force-closed at EOD every day**. This was identified as potentially incorrect given the estimated half-lives (14–28 bars at 15-min = 3.5–7 hours), which span multiple sessions.

### Iteration 2: Multi-day holding + bug fixes (`3e9df99`)

Three significant changes:

**1. Multi-day position holding (overnight carry)**
Positions are no longer force-closed at the end of each trading day. Only two conditions force a close:
- Last bar of the final trading day in the backtest
- `max_holding_days` exceeded (default = 5 trading days)

This allows spreads with 3.5–7 hour half-lives to mean-revert across sessions. `min_bars_remaining = 8` gates new entries with <8 bars left in the day to prevent entering positions that cannot revert before the next forced close.

**2. Last-bar fallback bug fix**
When z_exit fires at the final 15-min bar (15:45) of a non-final trading day, `next_bar_ts` is `None`. The original code fell back to the current bar's midpoint — executing intraday instead of carrying overnight. Fixed by computing `force_close` before the signal-transition check, and guarding signal checks with `not is_last_bar`.

**3. Direction-flip exit**
If a position's z-score crosses z_exit and immediately re-enters z_entry in the opposite direction within the same bar (rare overnight gap scenario), the backtester would miss the exit. Added detection: `sig != 0 and prev_sig != 0 and sig != pos.direction → should_exit = True`.

**Tests**: 26 tests passing across all modules after these changes.

---

## 2026-02-28 — Validity Investigation + Notebook Finalization

*(Current session — no commit yet)*

### Phase 1 validity investigation (`scripts/debug.py`)

Motivated by a Sharpe ~3.5 result at 1.5 bps that appeared too good. Ran a comprehensive diagnostic to rule out bugs inflating the results:

**Findings:**
- **No lookahead bias**: confirmed entry at bar i → execution at bar i+1 throughout.
- **Last-bar fallback impact**: only 12 of ~1,700 z_exit trades hit the 15:45 bar; those 12 have *lower* average net PnL than normal exits ($144 vs $186) — the bug was a minor drag, not inflation.
- **Same-day z_exit 98.3% win rate**: mathematically correct by construction. When z_exit fires, the spread has by definition reverted toward zero, so gross PnL > 0 unless costs dominate. Not a bug.
- **Stale signal carry-over at bar_idx=0**: fixed — added z-score threshold check at the first bar to prevent entries on signals that carried over from the lookback window rather than fresh crossings.
- **Cost sensitivity**: total gross PnL = $63K over the period; transaction costs at 1.5 bps = $22K; break-even ~3.5 bps. Strategy unprofitable at 5+ bps.
- **Annual stability**: profitable in most individual years (2022 H2 through 2024), with some losing months but no catastrophic drawdowns.
- **Entry time distribution**: 52% of entries at 9:45 (second bar), reflecting strong morning z-score divergence — consistent with overnight gap risk as a genuine signal source.

### Quantitative results summary (1.5 bps, 2022-07-01 → 2024-12-31)

| Metric | Value |
|---|---|
| Total return | ~+41% |
| Annualised Sharpe | ~3.5 |
| Max drawdown | ~−8.6% |
| Total trades | ~2,300 |
| Win rate | ~64% |
| Avg hold | ~1,280 min (~21 h) |
| z_exit exits | ~82% of trades |
| z_stop exits | ~18% of trades |
| Break-even cost | ~3.5 bps |

Note: the notebook's default `cost_bps=5.0` produces a losing result (~−11% analytically). The signal has positive alpha but is cost-sensitive.

### Strategy notebook updates (`notebooks/strategy.ipynb`)

- `BACKTEST_PARAMS` now explicitly includes `max_holding_days=5` and `min_bars_remaining=8`
- Parameter table updated with a rationale column for every parameter
- Portfolio summary section replaced single `pct_eod` metric with full EXIT REASON BREAKDOWN (z_exit / z_stop / max_hold / eod) using `value_counts()` helper

### Week 8 progress file
Written covering: what was built (full backtester + bug fixes + diagnostics), current quantitative results, and six future challenges (cost realism, walk-forward validation, overnight gap risk, cointegration instability, recalibration performance, correlated pairs).

---

## Open Questions / Next Experiments

### 5-minute bar frequency
**Question**: does 5-min data harvest faster reversion signals, or introduce more noise?

**Infrastructure status**: fully ready — processed 5-min data exists for all 102 tickers, `run_backtest(timeframe="5min")` requires zero code changes.

**Theoretical trade-offs:**
- Pro: faster z_exit detection (~15 min earlier), finer stop-loss granularity (3× less overshoot at z_stop), `min_bars_remaining=8` = 40 min instead of 2 hours
- Con: more microstructure noise → more false z_entry crossings → more trades → higher cost drag; z_stop may fire more often on noise (currently the dominant drag at -$72K total)

**Key metric to watch**: if n_trades increases substantially (>50%) while avg_gross_pnl per trade stays flat, noise is over-trading the cost budget. Break-even cost analysis will be the primary evaluation criterion.

**Suggested experiment**: run `timeframe="5min"` at z_entry = 3.0, 3.5, 4.0 and compare break-even bps. Higher z_entry filters noise at the cost of fewer trades.

### Walk-forward validation
Parameters selected on H2 2022 grid search. Full 2022-07-01 → 2024-12-31 backtest has lookahead contamination (parameters were chosen knowing the evaluation period). Need rolling walk-forward: train on 6-month window, test on following 3 months, roll forward.

### Cost realism
Real-world costs for institutional pairs trading on large-cap US equities: ~1–3 bps all-in (spread + market impact) at small sizes, higher at scale. The strategy is viable only below ~3.5 bps. Increasing `z_entry` to 3.5+ or switching to daily bars would improve gross-per-trade to cost ratio.
