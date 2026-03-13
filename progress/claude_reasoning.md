# Claude Reasoning Log

---

## 2026-03-12 — Add `_pv` parameter to strategy_comparison.ipynb

**What:** Added optional `_pv` (use_pvalue_weights) parameter parsing to `strategy_comparison.ipynb`.

**Why:** `strategy.ipynb` now appends `_pv{True/False}` to filenames when saving results. The comparison notebook's `_PARAM_RE` regex didn't include this token, causing any `_pv`-suffixed files to silently fail parsing and be skipped with a WARNING.

**Changes:**
- Added `(?:_pv(?P<pv>[^_]+))?` as an optional group in `_PARAM_RE` (between `_cb` and the start date)
- Added `"pv": d.get("pv")` to `_parse_params` return dict (returns `None` for old files without the param)
- Added `"pv": "pv"` to `PARAM_DISPLAY` so it appears in labels when it differs across runs

---

## 2026-03-10 — Parallelize formation_search.py with ProcessPoolExecutor

**Change:** Replaced the sequential combo loop in `scripts/formation_search.py` with `ProcessPoolExecutor`. Added `--workers` CLI arg (default: `os.cpu_count()`).

**macOS-specific design:** macOS uses the "spawn" multiprocessing start method (not "fork"), which means worker processes cannot inherit parent memory. All module-level imports re-run in each worker. Two consequences:
1. Worker functions must be defined at module level (top-level picklable callables) — `_eval_combo_worker` and `_init_worker` added.
2. `price_cache` is sent to each worker once via the `initializer=` argument to `ProcessPoolExecutor`, not re-pickled per task. This avoids N_tasks × cache_size pickle overhead.

**Progress reporting:** Uses `as_completed()` so results print as they finish (out of submission order, but tagged with combo index).

**`if __name__ == "__main__"` guard** was already present — required on macOS so spawned workers don't re-invoke `main()`.

---

## 2026-03-09 — Fix position sizing bug (B leg oversized by price_b/price_a)

**Bug:** `shares_b = hedge_ratio * shares_a = hedge_ratio * notional / price_a`. This divides by `price_a` instead of `price_b`, oversizing leg B by a factor of `price_b / price_a`. For low-price/high-price pairs (e.g. F ~$10 vs COST ~$980), leg B was ~95× too large.

**Symptom:** F/COST pair created $9,437 COST shares (~$9.3M notional) from a $50k notional budget. Produced $219k and $89k gross P&L outliers in Dec 2024.

**Root cause:** β is a log-price OLS coefficient — not a dollar hedge ratio. Dollar-neutral sizing requires `shares_b = β × notional / price_b`, not `β × notional / price_a`.

**Fix:** [strategy/backtester.py](strategy/backtester.py) line 650: `shares_b = hedge_ratio * notional_per_pair / entry_price_b`. Also corrected the docstring.

**Tests added** (`TestPositionSizing` in [tests/test_backtester.py](tests/test_backtester.py)): leg A notional == notional_per_pair; |leg B notional| == |β| * notional_per_pair; regression test with price_a=$10/price_b=$1000; invariance to price ratio; negative hedge_ratio sign check. Key infra: must also patch `strategy.backtester.get_all_tickers` to return `["A","B"]` so synthetic tickers reach `_price_cache`.

---

## 2026-03-09 — Config-driven run_backtest + require_split_window

**Motivation:** `run_backtest` had an ever-growing explicit parameter list. Any new cointegration or signal tunable required a signature change. Solution: collapse all parameters into a single `Config` instance; `run_backtest` now takes only `config: Config | None = None`.

**Changes:**

- `utils/config.py`:
  - `CointegrationConfig`: added `require_split_window: bool = False` — when True, a pair must cointegrate in both halves of the formation window independently. Added inline comments to all fields.
  - `SignalConfig`: changed `z_stop: float = 4.0` → `z_stop: float | None = 4.0` (None disables stop-loss); added `fixed_exit_norm: bool = True` (default True — uses entry-time rolling stats for exit). Added inline comments to all fields.
  - `PortfolioConfig`: added `start_date`, `end_date`, `timeframe`, `min_bars_remaining`, `execution_lag_minutes`, `execution_price_field` fields so all backtest execution params live in the config.

- `strategy/backtester.py`:
  - Replaced all individual explicit params with `config: Config | None = None`.
  - Resolver block at function top reads all params from `cfg = config or CONFIG`.
  - `find_cointegrated_pairs` now receives `p_value_threshold`, `min_half_life`, `max_half_life`, `require_split_window` from config.
  - `generate_pair_signals_for_day` now receives `fixed_exit_norm` from config.
  - Import updated to `from utils.config import BARS_PER_DAY, CONFIG, Config, MINUTES_PER_BAR, get_all_tickers`.

- `tests/test_backtester.py`:
  - Removed `RECOMMENDED_PARAMS` dict; added `_make_config` and `_recommended_config` helpers that return `Config` instances.
  - All `run_backtest` call sites updated to use `config=`.
  - Added `TestConfigParamEffects` class with 4 new tests: `test_no_config_uses_module_default`, `test_cost_bps_from_config_affects_net_pnl`, `test_max_pairs_from_config_limits_positions`, `test_zscore_window_days_from_config_runs_without_error`. All 40 tests pass.

- `scripts/debug.py`: Replaced `run_backtest(start_date=..., end_date=..., cost_bps=1.5)` with a `Config()` object.

- `notebooks/strategy.ipynb`: Replaced `CONFIG.signal.zscore_window_days = 5` + `BACKTEST_PARAMS` dict + `run_backtest(**BACKTEST_PARAMS)` pattern with a `cfg = Config()` setup cell. All downstream cells updated to reference `cfg.*` attributes instead of `BACKTEST_PARAMS[...]`.

---

## 2026-03-09 — Fix rolling volatility expansion in exit logic

**Problem diagnosed in `neg_zexit_pnl.ipynb`:** 36.6% of z-exit trades are unprofitable, contributing -130.3% of total z-exit P&L. Root cause: when rolling sigma expands between entry and exit, the z-score exit threshold fires even though the raw spread hasn't actually reverted. For z_entry=3.5, z_exit=2.5, a sigma_exit/sigma_entry ratio >1.40 causes a loss.

**Fix:** Added `fixed_exit_norm` flag. When enabled, the mean-reversion exit z-score is computed as `(spread[t] - mu_entry) / sigma_entry` using rolling stats snapshotted at entry time, rather than the current rolling stats. Stop-loss and time-stop still use rolling z-score.

**Changes:**
- `analysis/signals.py`: Added `compute_rolling_stats(spread, window)` helper; refactored `compute_zscore` to use it. Modified `generate_signals` to accept `spread`, `rolling_mean`, `rolling_std`, and `fixed_exit_norm` params — on entry snapshots mu/sigma, on exit computes fixed-norm z when flag is set. Updated `generate_pair_signals_for_day` to compute rolling stats explicitly and pass them through.
- `utils/config.py`: Added `fixed_exit_norm: bool = False` to `SignalConfig` with explanatory comment.
- `scripts/grid_search_v2.py`: Added `"fixed_exit_norm": [True, False]` to `SIGNAL_GRID`; wired the param through `_eval_combo` → `generate_pair_signals_for_day`.

---

## 2026-03-08 — Portfolio parquet save with params in filename (strategy.ipynb)

Added a new cell after the existing `save_results` call in `notebooks/strategy.ipynb`.
Saves `daily_pnl_df` to `results/portfolio/portfolio_<params>.parquet`.
- `_pval()` strips trailing `.0` (2.0→"2") and maps None→"none"
- Includes `CONFIG.signal.zscore_window_days` (set separately before BACKTEST_PARAMS)
- `mkdir(parents=True, exist_ok=True)` auto-creates the directory

---

## 2026-03-08 — Presentation reorganization

**Changed file:** `progress/presentation.md`

Reorganized 8 slides into 7-slide structure per user request. Key changes:
- Slide 4: removed specific zscore_window value (belongs in param selection); added transition note
- Slide 5: removed "Walk-forward design" subsection (was pair selection architecture, not execution)
- Slide 6: removed "Key findings so far" (results language in a methodology slide); added "see slide 7" transition
- Slide 7: added "Parameter findings" section (moved from slide 6) + kept P&L breakdown, cost sensitivity, issues

Principle: each slide stays on its topic; cross-slide references use explicit transitions rather than embedding the content.

---

## 2026-03-08 — Grid search v2: parameter sweep for directional correctness

**Created files:**
- `scripts/grid_search_v2.py` — two-phase grid search script
- `notebooks/grid_search_v2.ipynb` — results analysis notebook

**Design rationale:**

The core insight driving the design: cointegration (ADF tests) is expensive; signal
parameter evaluation is cheap. So the script separates:
1. **pairs cache build** — for each calendar day in eval period, run `find_cointegrated_pairs`
   with `price_cache` (avoids Parquet I/O) over the rolling formation window ending the
   day before. Cached as `dict[date_str, pl.DataFrame]`.
2. **parallel signal sweep** — each worker gets the full pairs cache and price data,
   iterates days, generates signals, evaluates metrics. Workers use `get_context("spawn")`
   to avoid polars/rayon fork-safety issues on macOS.

**Phase 1 (directional correctness):** Fixes formation params at defaults (42-day window,
half-life 4–18 bars, p≤0.05). Sweeps 3×4×3×3×4 = 144 signal combos. Primary question:
is IC > 0 at any horizon? If yes, the cointegration signal is directionally valid.

**Phase 2 (full grid):** Also sweeps 3×3×3×2 = 54 formation combos → 144 × 54 = 7,776
total combos. Much slower but identifies whether formation params matter.

**Key design choices:**
- `day_prices` sliced from full price cache with zscore lookback buffer (`zscore_window_days * 2 + 7`)
  so rolling z-score has enough history on each eval day
- `full_prices` (not day_prices) passed to `evaluate_all` so forward-return lookups work
  beyond the eval day's last bar
- `mean_gross_return` derived from `mean_net_return + 4 * cost_bps / 10_000` since
  `compute_net_forward_returns` deducts round-trip costs
- `breakeven_bps` = `mean_gross_return * 10000 / 4` (cost per leg at which net return = 0)

**Metrics reported:**
- `mean_ic_gross`: directional signal quality (Spearman IC, no costs)
- `ic_t_stat`: statistical significance
- `mean_net_return`, `mean_gross_return`, `breakeven_bps`: profitability
- `mean_hit_rate`: direction accuracy
- `total_n_obs`: signal activity (low count → unreliable stats)

---

## 2026-03-05 — Universe expansion: 100 → 200 tickers

**Goal:** Double breadth to increase the number of intra-sector pair candidates (pairs scale as C(n,2) — doubling sector size gives ~4× more pairs).

**Selection criteria:**
1. Continuous US listing since at least 2015 (required for historical backtesting)
2. Large/mid-cap with liquid 1-min data available on EODHD
3. Fit naturally into an existing sector peer group (no cross-sector contamination)

**Additions by sector (100 new tickers total):**

| Sector | Before | After | New pairs |
|--------|--------|-------|-----------|
| Technology | 10 | 20 | 45 → 190 |
| Semiconductors | 9 | 18 | 36 → 153 |
| Financials | 10 | 20 | 45 → 190 |
| Healthcare | 10 | 20 | 45 → 190 |
| Consumer Discretionary | 10 | 20 | 45 → 190 |
| Consumer Staples | 9 | 18 | 36 → 153 |
| Energy | 9 | 18 | 36 → 153 |
| Industrials | 10 | 20 | 45 → 190 |
| Communication Services | 9 | 18 | 36 → 153 |
| Utilities | 7 | 14 | 21 → 91 |
| REITs | 7 | 14 | 21 → 91 |
| **Total** | **100** | **200** | **411 → 1,744** |

**New tickers added:**
- Technology: IBM, CSCO, NOW, INTU, PYPL, SNPS, CDNS, ACN, FISV, FIS
- Semiconductors: ADI, MRVL, SWKS, NXPI, QRVO, MPWR, ASML, ENTG, TER
- Financials: AXP, COF, MET, PRU, TFC, FITB, RF, CFG, MTB, KEY
- Healthcare: CVS, CI, HUM, MDT, SYK, BSX, ISRG, REGN, VRTX, BIIB
- Consumer Discretionary: ROST, ORLY, AZO, DPZ, YUM, MAR, HLT, F, GM, CCL
- Consumer Staples: EL, MO, GIS, HSY, SYY, CLX, KMB, CHD, STZ
- Energy: HAL, BKR, DVN, FANG, MRO, WMB, KMI, APA, OKE
- Industrials: NOC, GD, FDX, ETN, EMR, PH, ITW, AME, ROK, XYL
- Communication Services: PARA, LYV, TTWO, IPG, OMC, NWSA, SIRI, MTCH, LBTYA
- Utilities: PPL, ED, FE, ES, WEC, XEL, AWK
- REITs: VTR, WELL, EXR, AVB, EQR, DLR, ARE

**Partial-data caveats (flagged in config comments):**
- PYPL: spun off from eBay Jul-2015 — first few weeks may be partial
- QRVO: formed Jan-2015 from RF Micro Devices + TriQuint merger — early 2015 may be partial
- TFC: current ticker since Dec-2019 (BB&T+SunTrust); pre-merger data is BB&T's history, not a gap
- BKR: current form since Jul-2017 (GE Oil & Gas merger with Baker Hughes); pre-2017 data is old BHI
- MTCH: IPO Nov-2015 — first few weeks partial

**Deliberately excluded (with reasons):**
- SNOW, NET, RBLX, ZM: IPO'd 2019–2020, insufficient history for 2015 backtest
- PXD (Pioneer Natural Resources): acquired by XOM Oct-2023; delisted, no active market
- HES (Hess): pending acquisition by CVX; price corrupted by deal premium
- ATVI (Activision Blizzard): acquired by MSFT Oct-2023; delisted
- TWTR (Twitter): taken private Oct-2022; delisted
- WBD (Warner Bros Discovery): formed Apr-2022; only ~3 years of data
- FOX/FOXA (Fox Corp): split from 21st Century Fox Mar-2019; limited history
- DISH (DISH Network): merger with DirecTV expected to close; corporate uncertainty
- CPB (Campbell Soup): potential acquisition by Sovos Brands complicates data
- SNAP: IPO Mar-2017; mid-cap, borderline liquidity for 1-min data
- HPE/HPQ: both spun off from Hewlett-Packard Nov-2015; included PYPL from same vintage but HPE/HPQ are lower-conviction pairs with existing tech names

**Note on the min_obs filter:** The 60-bar minimum observation check in `find_cointegrated_pairs()` naturally handles tickers with partial early data — any window where a ticker has <60 aligned bars with its pair is skipped automatically. No special treatment needed in backtester.

---

## 2026-03-05 — Chunk-based 1-min cache (memory-bounded speedup)

**Problem:** 1-min execution data was re-read from 100 parquet files every trading day (0.171s/day × 850 days = ~145s = 2.4 min). Full in-memory cache (~2.6 GB) would hit ~9 GB at 300 tickers × 10 years.

**Solution:** Chunk-based preload in `strategy/backtester.py`:
- Before the day loop: `_1min_chunk = {}`, `_1min_chunk_through = dt.date.min`
- On each day: if `day > _1min_chunk_through`, load the next 30 trading days at once (`_load_day_prices(get_all_tickers(), day, chunk_end, "1min")`)
- Per-day: filter from chunk with `pl.col("timestamp").dt.date() == day` (no parquet I/O)

**Memory:** 30 days × 100 tickers × ~1.8 MB/day ≈ **55 MB** per chunk (bounded regardless of history length). At 300 tickers: ~165 MB. At 300 tickers × 11 years: still ~165 MB (chunk doesn't scale with history).

**Speedup:** 21.0s → 12.0s for 3-month window. Projected full run: **4:42 → ~2:42** (~2 min saved, ~36% reduction from prior baseline).

**Correctness:** All 106 tests pass. Entry/exit times, tickers, prices, exit reasons bit-for-bit identical to saved baseline.

**Cumulative speedup from all optimizations:** 8:09 → ~2:42 (~67% reduction).

---

## 2026-03-05 — Replace statsmodels OLS/ADF with numpy (2× speedup on cointegration)

**Problem:** Profiling 3-month backtest showed cointegration was 63% of runtime (27s/43s):
- 78,912 `statsmodels.OLS.fit()` calls (3 per pair: hedge ratio, half-life, ADF internal)
- 96,313 Polars `lazy.collect()` calls from per-pair `DataFrame.join()`

**Changes in `analysis/cointegration.py`:**

1. **`_compute_hedge_ratio()`**: Replaced `add_constant(x)` + `OLS(y, X).fit()` with `np.linalg.lstsq(X, y)`. Identical results to machine epsilon (~4e-16).

2. **`_compute_half_life()`**: Same OLS→lstsq replacement.

3. **New `_fast_adfuller(y)`**: Hand-rolled ADF with constant, maxlag=1:
   - Build `X = [ones, y_lag, dy_lag]`, solve via `np.linalg.lstsq`
   - Compute t-stat: `beta[1] / sqrt(s² * (X'X)⁻¹[1,1])`
   - P-value: `mackinnonp(tstat, regression='c', N=1)` (kept from statsmodels — cheap polynomial)
   - Verified identical to `adfuller(..., maxlag=1, regression='c', autolag=None)` at ~6e-16

4. **Pre-extract numpy arrays**: Before the pair loop, convert each ticker's price DataFrame to `(timestamps_int64, prices_float64)` numpy arrays.

5. **`np.intersect1d` alignment**: Replace `df_a.join(df_b, on='timestamp', how='inner')` with `np.intersect1d(ts_a, ts_b, return_indices=True)`. Eliminates 96,313 Polars lazy collects.

**Result:** 8:09 → 4:42 wall time (~42% faster). Trade count identical (5,363). Entry/exit times, tickers, prices, exit reasons are bit-for-bit identical. Tiny differences (~1e-10 in net_pnl) arise from numpy lstsq vs statsmodels pinv computing the same hedge_ratio via different FP paths — economically irrelevant (relative error ~1e-12). Saved parquets updated to new baseline.

Imports removed: `OLS`, `add_constant` from statsmodels, `adfuller` from statsmodels.tsa.stattools.
Import added: `mackinnonp` from statsmodels.tsa.adfvalues.

---

## 2026-03-05 — Speed optimizations: price cache + numpy + vectorized signals

**Changes (3 files):**

**`analysis/cointegration.py`:**
- Added `price_cache: dict | None = None` to `find_cointegrated_pairs()`
- When provided: filters in-memory dict by date range instead of reading parquet files per call

**`analysis/signals.py`:**
- `generate_signals()`: `to_numpy()` instead of `to_list()`, `np.zeros(n, dtype=np.int32)`, NaN check `z != z`, cached local `_z_entry/_z_exit/_z_stop/_max_holding`, split `abs(z) > z_stop` to avoid abs()
- `generate_pair_signals_for_day()`: replaced list-of-dicts + schema-cast with per-pair `pl.DataFrame({...})` + `pl.concat(pair_dfs)`

**`strategy/backtester.py`:**
- Preload all universe prices once before day loop into `_price_cache`
- Per day: filter from RAM (`pl.col("timestamp").dt.date()`) instead of re-reading parquet
- Pass `price_cache=_price_cache` to `find_cointegrated_pairs()` (skips all parquet I/O)

**`tests/test_backtester.py`:**
- Updated `test_none_lag_does_not_load_1min`: replaced stale `call_count["n"] == 2` assertion (count is now 100+ due to preload) with targeted 1-min-specific counter

**Result:** All 106 tests pass. Backtest output is **exactly identical** (5,363 trades, all values match trade-by-trade vs saved parquet).

---

## 2026-03-05 — Remove z_stop (stop-loss destroys all profitability)

**Problem:** Strategy still losing money ($-28K, Sharpe -0.81) after log-spread switch.
Exit breakdown: 69.6% z_exit, 30.4% z_stop.

**Trade economics at z_stop=4.0, cost_bps=2.0:**
| Exit | Avg gross P&L | Avg net P&L | Win rate | n trades |
|------|-------------|-------------|----------|---------|
| z_exit | +$9.12 | +$4.87 | 51.7% | 7,195 |
| z_stop | -$15.8 | -$20.1 | 30.0% | 3,148 |

z_exit trades contribute +$35K total; z_stop trades contribute -$63K total. Net = -$28K.

**Root cause:** At 15-min bars, a 4-sigma intraday log-spread move is mostly transient
noise, not a regime break. The position would recover if held, but z_stop forces it out
at maximum adverse excursion plus transaction cost. Each z_stop trade loses ~$20 net.

**Fix: set z_stop=None throughout.**

`analysis/signals.py` — `generate_signals()`:
- Removed `z_stop = z_stop if z_stop is not None else CONFIG.signal.z_stop` fallback
- Stop-loss check is already guarded: `if z_stop is not None and abs(z) > z_stop`
- `z_stop=None` → stop-loss branch never fires

`strategy/backtester.py` — `run_backtest()`:
- Removed `z_stop = z_stop if z_stop is not None else CONFIG.signal.z_stop` resolution
- Exit reason labeling guarded: `if z_stop is not None and z_at_exit is not None and abs(z_at_exit) > z_stop`
- `z_stop=None` passes through to signal generator unchanged

`notebooks/strategy.ipynb`:
- Changed `z_stop = 4.0` → `z_stop = None` in BACKTEST_PARAMS
- Updated markdown table and added "no stop-loss" rationale block

All 106 tests pass.

**Expected impact:** Remove the -$63K z_stop drag. With only z_exit exits (avg +$4.87,
51.7% WR), total should flip from -$28K to ~+$35K over the backtest period.

---

## 2026-03-05 — Switch to log-price spread throughout

**Problem:** Linear (price-level) OLS spread has tiny sigma relative to stock price.
Trade economics showed: avg gross = 0.91 bps of notional_A vs avg cost = 16.86 bps.
Delta-spread median = $0.0025/share. Even z_exit trades (best-case) gross 7.59 bps vs
16.80 bps cost — 2.2× below breakeven. The strategy cannot be profitable at any realistic
cost with linear spreads at 15-min frequency.

**Root cause:** Linear spread `A - hr*B` has dollar sigma ~$0.05–0.50/day for
cointegrated equity pairs. Normalized by notional_A (~$2,500–5,000), that's ~1–10 bps/day.
A 2.5-sigma entry only captures a fraction of daily sigma → gross per trade << cost.

**Fix: switch to log-price spread everywhere:**

`analysis/cointegration.py` — `test_cointegration()`:
- Compute `log_a = np.log(prices_a)`, `log_b = np.log(prices_b)` before OLS regression
- `hedge_ratio` is now a log-elasticity (beta): `log(A) ~ alpha + beta*log(B)`
- ADF test on log-spread residuals tests log-cointegration (stationarity of log(A/B^beta))

`analysis/signals.py` — `compute_spread()`:
- Added `import math`
- Changed `prices_a - hedge_ratio * prices_b` → `prices_a.log(math.e) - hedge_ratio * prices_b.log(math.e)`
- Log-spread is in percentage space; a unit change ≈ 1% divergence

`analysis/evaluation.py` — `compute_forward_returns()`:
- Added `import math`
- Changed guard: `if close_a_t <= 0.0` → `if any price <= 0` (all four prices must be positive for log)
- Changed computation: `sig * (spread_tn - spread_t) / close_a_t` → `sig * (log_spread_tn - log_spread_t)`
- Log-spread difference is already a percentage return; no normalization needed

`tests/test_evaluation.py`:
- Updated 4 tests with new expected values (log expressions instead of linear arithmetic)
- Renamed `test_normalized_by_price_a` → `test_log_spread_difference`

**Expected impact:**
- Spread sigma for cointegrated pairs is typically 0.5–2%/day in log space
- At z_entry=2.5 sigma, expected gross per trade = ~50–200 bps (vs current 0.91 bps)
- Breakeven shifts from ~0.2 bps/leg (impossible) to ~4–12 bps/leg (realistic)

All 106 tests pass after the change.

**Next steps:** Re-run `notebooks/parameter_search_15min.ipynb` with the new log-spread
to get updated grid metrics. The previous grid results in `results/params/params_15min.parquet`
are now stale (computed with linear spread normalization by price_a).



Record of file edits and the reasoning behind each change.

---

## 2026-03-04 — Align grid search ↔ backtest (intraday-only)

**Problem:** Grid search and backtest used incompatible parameter conventions, explaining why the backtest lost money despite IC-positive grid results. Key mismatches:
- `max_holding_bars=None` hardcoded in backtester → time stop never fired in signal generator
- `max_holding_days=5` (calendar days, multi-day overnight) vs grid's `max_holding_minutes=60-120` (intraday)
- `zscore_window_days=10` in backtest vs `2` in grid winners (100% of top-20 filtered combos)
- `formation_window_days=84` vs `126` (grid winners)
- `z_entry=3.0` vs `2.5`, `z_exit=0.5` vs `1.0`, `z_stop=4.5` vs `3.5`

Grid results filtered to: `mean_net_return > 0` AND `hit_rate > 0.5`, ranked by `ic_gross`. 91/1,296 combos passed. Top combo: ic_gross=0.042, hit_rate=51.5%.

**`strategy/backtester.py`:**
- Replaced `max_holding_days: int = 5` with `max_holding_minutes: int = 120`
- Added `max_holding_bars_val = max_holding_minutes // MINUTES_PER_BAR[timeframe]`
- Passed `max_holding_bars=max_holding_bars_val` to `generate_pair_signals_for_day()` (was hardcoded `None`)
- Simplified force-close to `force_close = is_last_bar` (EOD every day; was only last trading day or multi-day expiry)
- Exit reason simplified to `"eod"` (removed `"max_hold"` branch)
- Removed `entry_day_idx` from `PositionState` (no longer needed without multi-day tracking)

**`notebooks/strategy.ipynb`:**
- `CONFIG.signal.zscore_window_days = 2` (was 10)
- `BACKTEST_PARAMS`: rolling_window_days=126, z_entry=2.5, z_exit=1.0, z_stop=3.5, max_holding_minutes=120 (replacing max_holding_days=5)
- Updated markdown cell with new parameter rationale table

**`tests/test_evaluation.py`:**
- Fixed pre-existing failures: `ic_net_weighted` → `mean_net_return` in two places (column was renamed but tests not updated)

Result: 106 tests passing.

---

## 2026-03-05 — Parameter search analysis after evaluation fix; new recommended params

**Context:** After fixing evaluation.py, user re-ran the 1,296-combo grid and saved results to
`results/params/params_15min.parquet`. Notebook `notebooks/parameter_search_analysis.ipynb`
created to analyze the corrected results.

**Key findings from corrected grid:**
- 0% of combos profitable at 20 bps grid cost (as expected after fix)
- ~40% of combos have positive mean_gross_return — signal HAS directional value
- Best breakeven cost: 4.77 bps/leg (window=5, z_entry=3.0, z_exit=0.0, z_stop=4.0, max_hold=60, formation=84)
- Best IC: 0.078 (window=10, z_entry=2.5, z_exit=0.0, z_stop=4.0, max_hold=120, formation=126)
- IC vs breakeven are anti-correlated: high IC region (window=10, formation=126) has worst
  breakeven (-0.9 bps); best breakeven region (window=5, formation=84) has near-zero IC

**Recommended parameters** (best IC>0 + highest breakeven):
- zscore_window_days=5, rolling_window_days=84, z_entry=3.0, z_exit=0.5, z_stop=4.0, max_hold=120
- breakeven=3.65 bps/leg, IC=0.003

**`notebooks/parameter_search_analysis.ipynb`:** Created. 9 cells covering: corrected
distributions, IC vs breakeven scatter, parameter sensitivity heatmaps, cost sensitivity
curve, top combos by each criterion, recommendation.

**`notebooks/strategy.ipynb`:** Updated BACKTEST_PARAMS to recommended params above.

---

## 2026-03-04 — Root cause: evaluation.py normalises returns by spread_t, not price_a

**Problem:** The backtest continues to lose money (Sharpe ~-7, return ~-68%) even with
zscore_window_days=10 and IC-positive grid params. Avg gross per z_exit trade = $2.15 vs
avg cost = $5.05 → ratio 0.43.

**Root cause — wrong normalisation in `analysis/evaluation.py` line 100:**
```python
# Old (wrong):
fwd_return = float(sig) * (spread_tn - spread_t) / abs(spread_t)
# Fixed:
fwd_return = float(sig) * (spread_tn - spread_t) / close_a_t
```

The OLS spread for a tightly cointegrated pair (e.g. AAPL/MSFT) is
`spread = price_a - beta * price_b ≈ $1.40`.  `price_a ≈ $180`.
Normalising by spread_t produces a "% return on spread value", which is ~128× too
large vs the correct "% return on notional" (= delta_spread / price_a).

The same error inflates the apparent transaction cost:
- Grid cost = 80 bps of spread_t = 0.80% × $1.40 = $0.011 per "unit"
- Real cost   = 20 bps of notional = 0.20% × $5,000 = $10 per trade

This means the grid's `mean_net_return > 0` filter was essentially noise — it was
checking whether gross exceeded 80 bps of the spread, not 20 bps of notional.

**Quantified impact:**
- Grid breakeven (old, in spread-bps): 25 bps/leg of spread value
- Real breakeven (in notional bps):    25 × (spread/price_a) ≈ 25 × (1.40/180) ≈ 0.19 bps/leg
- Backtest uses: 5 bps/leg → far above the real breakeven of 0.19 bps

The IC is still valid (rank-correlation, scale-invariant). Mean_net_return and hit_rate
filtered results were misleading.

**`analysis/evaluation.py`:**
- Changed `if spread_t == 0.0:` → `if close_a_t <= 0.0:` (new denominator never zero for equities)
- Changed `/ abs(spread_t)` → `/ close_a_t` with explanatory comment

**Next steps required:**
1. Re-run `notebooks/parameter_search_15min.ipynb` with the fixed evaluation.
2. Expect most / all combos to show `mean_net_return < 0` at cost_bps=20 (= 80 bps round-trip).
3. Re-run at lower cost (e.g. 1 bps) to find the true breakeven parameter region.
4. Consider: at 5 bps (20 bps round-trip), the intraday 15-min strategy may be
   fundamentally unviable — sigma_spread is ~2 cents on a $100 stock, generating ~$2 gross
   vs $10 round-trip cost. Daily bars (sigma ~sqrt(26)× larger) or much lower costs are
   needed for dollar profitability.

---

## 2026-03-04 — Fix zscore_window_days; re-derive params using IC as primary criterion

**Problem:** zscore_window_days=2 destroyed the backtest (-80% return, avg gross $0.28 vs avg cost $4.24). Root cause: 2-day rolling std is tiny, so z=2.5 fires on minute dollar spread deviations. Transaction cost ($10 round-trip on $5k notional) dwarfs the captured gross P&L. IC is scale-agnostic so the grid didn't detect this. 20,182 trades generated (vs 4,583 before) — massive overtrading on noise.

**Fix:** Switch to zscore_window_days=10, which has the highest IC of all 1,296 combos (IC=0.070 vs 0.047 for window=2). Longer window → larger dollar spread deviation at z=2.5 → gross P&L can clear costs. The `hit_rate > 0.5` filter was misleading — it excluded all window=10 combos but window=10 is the best by IC.

**New params (best IC at window=10):**
- zscore_window_days: 2 → 10
- z_exit: 0.0 → 0.5 (rank-1 IC combo at window=10)
- z_stop: 3.5 → 4.0 (top-3 IC combos at window=10)
- max_holding_minutes: 240 → 120 (11/20 top-IC combos at window=10)
- z_entry: 2.5, formation_window_days: 126 — unchanged

**`notebooks/strategy.ipynb`:** Updated BACKTEST_PARAMS and CONFIG.signal.zscore_window_days.

---

## 2026-03-06 — Fix `max_hold` exit reason misclassification

**Bug:** All time-stop exits were being recorded as `z_exit`, so the backtest showed
only `eod` and `z_exit` exit reasons. `max_hold` never appeared.

**Root cause (two-step):**

1. The signal generator (`analysis/signals.py`) applies the time stop by setting
   `signal = 0` when `bars_held >= max_holding_bars`. It does not emit a reason code
   — it just turns the signal off, indistinguishable from a z_exit crossing.

2. The backtester infers exit reason by checking `elapsed_minutes >= max_holding_minutes`,
   but `pos.entry_time = next_bar_ts` (1 bar after the signal bar). The signal generator
   counts `bars_held` from the signal bar, so when the time stop fires after
   `max_holding_bars` bars, only `(max_holding_bars - 1)` bar-intervals have elapsed
   from `pos.entry_time`. For `max_holding_minutes=180` and 15-min bars:
   - Time stop fires at elapsed = (12−1)×15 = **165 min**
   - Original threshold: `>= 180 min` → never triggered → misclassified as `z_exit`

**Fix (`strategy/backtester.py`):**

Use `(max_holding_bars_val - 1) * MINUTES_PER_BAR[timeframe]` as the `max_hold`
detection threshold instead of `max_holding_minutes`:

```python
max_hold_threshold = (max_holding_bars_val - 1) * minutes_per_bar
elif elapsed_minutes >= max_hold_threshold:
    exit_reason = "max_hold"
```

**Verification:** Smoke test over Q3 2022 shows `max_hold=785`, `z_exit=164`, `eod=11`
(previously all `max_hold` were absorbed into `z_exit`).

---

## 2026-03-06 — z_exit does not guarantee spread reversion (rolling window drift)

**Finding:** A `z_exit` trade can have negative gross P&L even though the exit condition
fired correctly. Verified by reconstructing the AMD/PYPL trade on 2020-07-09 (entry 10:15,
exit 12:15, direction=-1, gross=-$20.49, exit_reason=z_exit).

**Root cause: rolling window mean/std drift**

The z-score is normalised against a 130-bar rolling window. Over a 2-hour hold, as new
bars enter and old bars leave the window, the rolling `μ` and `σ` change. In this trade:

| | Signal bar (10:00) | Exit signal bar (12:00) |
|---|---|---|
| Raw log-spread | 4.42751 | 4.42685 |
| Rolling μ | 4.38159 | 4.38503 |
| Rolling σ | **0.00906** | **0.01388** |
| z-score | 5.07 | 3.01 |

The spread barely moved (Δ = −0.00066). The z-score fell from 5.07 → 3.01 entirely
because `σ` expanded by 53% — normalisation reference frame shifted under the position.
At execution prices (10:17 and 12:17 1-min bars), the spread actually **widened** by
+0.0051 log units.

**Hedge ratio β = −0.079797 (negative):** AMD and PYPL are not naturally cointegrated —
this is a cross-sector pair (semiconductors + fintech) with a spurious short-term
relationship. Negative β means `spread = log(AMD) + 0.08·log(PYPL)`, so
`shares_b = β × shares_a` is negative (short PYPL, not long). Both legs moved against
the position: AMD up, PYPL down.

**Implication:** The 86.7% z_exit win rate reflects that in most cases the spread
physically reverts AND z_exit fires. The ~13% z_exit losses are predominantly
window-drift artifacts — the exit threshold is crossed due to σ expansion, not actual
mean reversion. A more robust exit would track the raw spread price against a fixed
normalisation baseline (e.g., spread at entry) rather than an adaptive rolling window.

**Notebook used for verification:** Ad-hoc script using `load_processed()`,
`scipy.stats.linregress()`, and 1-min price lookup. Reconstruction cross-checks P&L
formula in `_compute_pnl()` (`strategy/backtester.py` line 72).

---

## 2026-03-06 — Staleness hypothesis analysis + snapshot

**Context:** After fixing `max_hold` classification, re-ran strategy.ipynb and found
79% of trades (6,172/7,820) hit the time stop. `z_exit` is the only profitable exit
(+$55,294; 86.7% win rate). `max_hold` destroys -$96,899. Hypothesis: many entries are
"stale" — z-score already elevated at entry from a prior session, not a fresh divergence.

**Snapshot saved:**
- `results/snapshots/trades_15min_z35_zh180_zx175_84d_2022-2024.parquet`
- `results/snapshots/daily_pnl_15min_z35_zh180_zx175_84d_2022-2024.parquet`
- Naming convention: `{label}_{z_entry}_{max_hold_min}_{z_exit}_{formation_days}d_{date_range}`

**Analysis notebook:** `notebooks/staleness_analysis.ipynb`

Two tests:
- **Test 1:** Entry hour as staleness proxy — max_hold rate by entry hour
- **Test 3:** Z-score at bar-0 of each trading day, reconstructed via rolling z-score
  from `analysis.preprocessing.load_processed`. Hedge ratio recovered from `shares_b/shares_a`.
  Binary split: `|z_at_open| >= z_entry` → stale (carry-over), else → fresh (intraday).

**Key formula for z_at_open:**
  - `lookback = ZSCORE_WINDOW` bars ending at bar immediately before first bar of trade_date
  - `z = (log_spread[first_bar] - mean(window)) / std(window, ddof=1)`

---

## 2026-03-07 — Add min_half_life / max_half_life to parameter grid

**Motivation:** Half-life of mean reversion is the key determinant of whether a pair can revert within the max_holding window. Current config (`min=5, max=120` bars) allows pairs with intraday half-life up to 1,800 min — 10× the 180-min hold window. Testing tighter upper bounds directly tests the "79% max_hold rate" hypothesis.

**Changes:**

`scripts/grid_worker.py`:
- Cache key type: `tuple[int, dt.date]` → `tuple[int, int, int, dt.date]` (adds min_half_life, max_half_life)
- `_eval_combo`: reads `params["min_half_life"]` and `params["max_half_life"]`; uses `(formation_days, min_hl, max_hl, day)` as cache key

`notebooks/parameter_search_15min.ipynb`:
- GRID: added `min_half_life: [1, 5]` and `max_half_life: [12, 24, 48, 120]`
- Precomputation loop: iterates over all `(formation_days, min_hl, max_hl)` combos; passes `min_half_life` and `max_half_life` to `find_cointegrated_pairs()`; uses 4-tuple cache key
- Results display and sensitivity analysis cells updated to include new params

**Grid size impact:** previous 288 → now 2,304 combos (8× factor from 2×4 half-life combinations). User should narrow the half-life values as needed before running.

**Expected finding:** `max_half_life=12` (≤180 min) should show higher z_exit rate and better breakeven vs `max_half_life=120`, if the 79% max_hold rate is driven by slow-reverting pairs.

---

## 2026-03-07 — Sector consolidation: 11 → 8 sectors

**Goal:** Increase pairs evaluated for cointegration by merging fundamentally similar sectors. Pairs scale as C(n,2), so merging two n=14–18 groups into one n=28–38 group roughly doubles the pair count for those groups.

**Merges applied:**
1. **Technology + Semiconductors → "Technology"** (20+18 = 38 tickers, 703 pairs vs 343): Semiconductors are a sub-sector of GICS Information Technology. NVDA/AMD/INTC were already in "Technology" despite being pure-play semis — the prior split was internally inconsistent.
2. **Consumer Discretionary + Consumer Staples → "Consumer"** (20+18 = 38 tickers, 703 pairs vs 343): Retail overlap between groups (WMT/COST vs AMZN/HD) blurs the boundary. Combined consumer demand is a shared fundamental driver.
3. **Utilities + REITs → "Real Assets"** (14+14 = 28 tickers, 378 pairs vs 182): Both sectors are rate-sensitive, asset-heavy, dividend-focused, and respond identically to macro rate moves. Infrastructure REITs (AMT, CCI, EQIX) behave more like utilities than equity REITs.

**Result:** 11 → 8 sectors, 1,744 → 2,660 total pairs (+52%), same 200 tickers.

---

## 2026-03-08 — Rewrote presentation.md

**What changed:**
- Universe: updated from 100 tickers / 11 sectors → 200 tickers / 8 sectors; sector table now reflects consolidated groupings
- Half-life filter: corrected units from "trading days" to **bars**; added empirical sweet spot (24 bars / ~6h from param search)
- Signal parameters (Slide 4): updated state machine to match current config (z_entry=4.0, z_exit=2.5, z_stop=5.0, max_hold=120 min); added param search finding that z_entry ≥ 4.0 is required
- Formation window: updated to 42 calendar days (matching config) and noted tradeoff vs 84-day option
- Execution model (Slide 5): corrected midpoint framing — it is an **optimistic** upper bound, not a lower bound
- Slide 6 framing: reframed from "break-even is 0.5 bps" to the cleaner split: z_exit cohort earns +$64,976 at 86.7% win rate with ~7 bps/leg break-even; time-stop cohort destroys $59,964
- Slide 8: replaced stale "to-do" items with completed work (sector consolidation, half-life calibration, z_stop), and kept remaining items (z_entry validation, stability filter, walk-forward OOS)


## 2026-03-08 — Backtest config comparison + disconnect analysis

**Files created:**
- `notebooks/backtest_configs_analysis.ipynb` — runs 5 backtest configs, compares to param search, saves plots
- `progress/disconnect_analysis.md` — full written analysis of the param search vs backtest disconnect
- `results/experiments/trades_C*.parquet` / `pnl_C*.parquet` — cached backtest results for 5 configs

**5 configs tested (2022-07-01 → 2024-07-01, $100k, 5 bps/leg):**

| Config | Trades | Net P&L | Sharpe |
|---|---|---|---|
| C1 Baseline (z_e=2.5, z_x=0.0) | 6,860 | −$78,381 | −2.78 |
| C2 ParamSearch (z_e=4.0, z_x=2.5) | 115 | −$65 | −0.02 |
| C3 Moderate (z_e=3.0, z_x=0.0) | 6,464 | −$69,885 | −2.27 |
| C4 Tight+Partial (z_e=3.5, z_x=1.5) | 908 | −$16,703 | −1.23 |
| C5 FastZ (z_e=3.0, z_x=1.0, zscore=2d) | 6,883 | −$77,836 | −2.17 |

**Key findings:**
1. `z_exit=0.0` is the primary bug — produces 4% z_exit rate, rest time out. Raising to 1.0–1.5 jumps z_exit rate to 18%+ at 90%+ win rate
2. C2 validates param search breakeven metric exactly (near breakeven at 5 bps), but 115 trades = not a strategy
3. C4 best net P&L: fewer trades, high entry threshold, partial exit, wide z_stop
4. C5 best z_exit gross (+$89,881 at 89.7% win) — killed by $101k in costs
5. z_stop losses dominate C1/C3, not max_hold losses (max_hold is actually slightly positive gross in C1)

**Recommended next config:** z_entry=3.5, z_exit=1.0, z_stop=5.0, max_hold=240, formation=63d, max_hl=24, zscore=2d, max_pairs=20


## 2026-03-09 — Split-window ADF + formation parameter search

**Files modified/created:**
- `analysis/cointegration.py` — added `_split_window_stats`, `_hedge_ratio_cv`, extended `find_cointegrated_pairs`
- `scripts/formation_search.py` — Stage A quality sweep (54 combos, no signal backtest)
- `notebooks/formation_search.ipynb` — results analysis notebook

### `analysis/cointegration.py` changes

**`_split_window_stats(spread, p_threshold) -> (float, float, bool)`**
- Splits spread into two halves, runs `_fast_adfuller` on each independently
- Guard: `mid < 10` returns NaN/False rather than computing ADF on too-short windows
- Motivation (Gregory-Hansen): aggregate ADF can pass even when cointegrating relationship breaks at the midpoint; requiring both halves to pass independently detects structural breaks within the formation window
- Spread recomputed inline in the `find_cointegrated_pairs` loop (`log_a - hedge_ratio * log_b`) rather than refactoring `test_cointegration` — avoids changing an existing interface and is cheap (one numpy subtraction)

**`_hedge_ratio_cv(log_a, log_b) -> float`**
- Estimates β on each third of the formation window (3 estimates instead of 2) — gives a more robust CV with 3 points than just comparing two halves
- Guard: `third < 10` (minimum 30 obs total for a valid CV estimate)
- CV = `std(betas, ddof=0) / |mean(betas)|` — measures percentage drift in the cointegrating vector
- Returns NaN if mean ≈ 0 (would produce division instability)

**`find_cointegrated_pairs` signature extension**
- New params: `split_window_p_threshold` (defaults to `p_value_threshold` if None), `require_split_window` (hard-filter on split_consistent when True)
- All 3 empty-DataFrame schemas updated with 4 new columns: `adf_p_first_half`, `adf_p_second_half`, `split_consistent`, `hedge_ratio_cv`
- `require_split_window=False` by default — new metrics are always computed but don't filter by default, preserving existing behavior

### `scripts/formation_search.py` design

**Two-stage rationale:** Formation param search (ADF on ~2,660 pairs × 54 combos × N eval days) is too slow to combine with signal backtest. Stage A evaluates pair quality only, Stage B runs signal sweep on the ~10–20% of combos that survive quality thresholds.

**Formation grid:** 3×3×3×2 = 54 combos. `rolling_window_days ∈ {21, 42, 84}`, `min_half_life ∈ {2, 4, 8}`, `max_half_life ∈ {12, 24, 48}`, `p_value_threshold ∈ {0.01, 0.05}`.

**Quality thresholds:**
- `mean_n_pairs ≥ 5`: minimum portfolio diversification — below 5 pairs/day, the strategy has single-pair concentration risk
- `pct_days_with_pairs ≥ 50%`: pairs must be available on most days — below 50%, formation parameters are too restrictive to run a live strategy
- `split_consistency_rate ≥ 40%`: at least 40% of pairs pass both half-window ADF tests — a threshold below which the cointegration is mostly spurious. Intraday cointegration is inherently weaker than daily, so 40% (not 70%+) is appropriate

**Composite score:** `split_consistency_rate × log(mean_n_pairs) × mean_n_sectors`
- Multiplicative: all three factors must be non-trivially large to score well
- `log(n_pairs)` rather than `n_pairs`: diminishing returns from extra pairs (10→20 pairs is more valuable than 100→110)
- `mean_n_sectors`: diversification across sectors prevents correlated pair losses in sector-specific macro moves

### `notebooks/formation_search.ipynb` workflow

Final cell generates the Python dict for copy-paste into `FORMATION_GRID` in `grid_search_v2.py`, forming the bridge from Stage A → Stage B. The notebook is designed to be run after `formation_search.py` produces `results/formation_search.parquet`.

---

## 2026-03-09 — Portfolio comparison notebook

**Created:** `notebooks/portfolio_comparison.ipynb`

Compares all `results/portfolio/portfolio_tf15min*.parquet` files. Key design choices:
- Parses param key-value pairs from filename with regex `([a-z]+)([\d.]+)` and date range suffix
- Computes which params vary across files; legend shows only those (fixed params omitted)
- Varying params in this run: `zx`, `zs`, `mp`, `mbr`, `cb`, `start`
- Sections: full equity curves (each portfolio from its own start), normalized equity in common
  window (max of all starts → min of all ends), cumulative gross/net P&L, drawdown, rolling 30d
  Sharpe, summary stats table (total/ann return, Sharpe, max DD, Calmar, trades, win days),
  bar chart of ann_ret/Sharpe/max_DD

---

## 2026-03-09 — Negative z_exit P&L analysis notebook

**Created:** `notebooks/neg_zexit_pnl.ipynb`

**Motivation:** Trades with `exit_reason=z_exit` should be profitable — the z-score crossed back
through the exit threshold, indicating spread reversion. But a subset have `gross_pnl < 0`.
Examples: AMAT/MCHP 2021-01-11 (dir=-1, pnl=-$83), AEP/WEC 2021-01-13, TXN/KLAC.

**Root cause identified: rolling window normalization drift.**
The z-score is `z_t = (S_t - mu_t) / sigma_t` where `mu` and `sigma` are computed over the
last 130 bars (5 days at 15min). The z-score threshold crossing does NOT guarantee the raw
log-spread reverted. Mathematical condition for a false z_exit (short spread, dir=-1):

```
At entry: S_entry = mu_entry + 3.5 * sigma_entry
At exit:  S_exit  = mu_exit  + 2.5 * sigma_exit
P&L < 0 when S_exit > S_entry, which occurs when sigma_exit / sigma_entry > 3.5/2.5 = 1.40
```

Any 40%+ intraday vol expansion between entry and exit causes z_exit to fire before actual
reversion. Rolling mean drift compounds this.

**Notebook sections:**
1. Setup: loads `results/trades/trades_15min.parquet`, reconstructs `hedge_ratio = shares_b/shares_a`,
   computes `log_spread_entry`, `log_spread_exit`, `log_spread_change`
2. Cross-tab: P&L sign vs spread reversion flag (`direction × log_spread_change > 0`)
3. Mathematical mechanism explanation with breakeven derivation (z_entry/z_exit = 1.40)
4. `build_spread_zscore_series()`: loads 15-min history, computes 130-bar rolling z-score for
   specific trades with entry/exit signal reconstruction
5. AMAT/MCHP 2021-01-11 reconstruction: 3-panel plot (z-score, raw spread+bands, rolling std)
6. `reconstruct_trade()`: generic wrapper for any trade; diagnoses std_ratio vs breakeven
7. AEP/WEC 2021-01-13 reconstruction
8. Systematic classification across all negative z_exit trades using execution prices
9. P&L decomposition table (positive vs negative z_exit count, total, mean)
10. Distribution plots (P&L, spread change, hold duration)
11. Summary: 4 potential fixes (fixed normalization at entry, absolute spread target,
    tighter max_holding, require min reversion gate) with trade-offs table

---

## 2026-03-09 — IC evaluation improvements: horizon gating, pooled IC, TARGET_N_BARS

**Files modified:**
- `analysis/evaluation.py` — added pooled IC to `evaluate_all()`
- `scripts/grid_search_v2.py` — horizon gating in `_eval_combo()`, added weighted/pooled IC to aggregation
- `notebooks/grid_search_v2.ipynb` — TARGET_N_BARS filtering, IC curve plot, pooled vs mean IC comparison

### Three problems addressed

**Problem 1: Phantom forward returns (highest priority)**
`evaluate_all` was called with ALL four horizons [4, 8, 16, 26] bars for every combo regardless of
`max_holding_minutes`. A combo with `max_hold=120min` (8 bars) was getting IC computed at 16 and 26
bars — returns the state machine can never capture. Since spread mean-reverts slowly, IC at longer
horizons looks better than at 4-8 bars, making slow-reverting pairs appear attractive.

Fix: in `_eval_combo`, compute `valid_horizons = [h for h in all_horizons if h <= max_hold_bars]`
and pass only those to `evaluate_all`. Fallback to minimum horizon if none fit.

**Problem 2: Mean-of-daily-IC weights low-signal days equally**
IC computed on 2 pairs is meaningless noise but contributed equal weight to `mean_ic_gross`.

Fix A: `weighted_mean_ic_gross` — weight daily IC by `sqrt(n_obs)`.
Fix B: `pooled_ic_gross` — pool all (signal, return) pairs across all dates per horizon, compute
one Spearman correlation. Accumulated in `evaluate_all` via per-horizon lists; broadcast as constant
to all rows sharing the same n_bars.

**Problem 3: Notebook horizon filtering**
All selection cells now use `TARGET_N_BARS = min(df['n_bars'].unique())`. Added `best_pooled` table
and IC curve plot + pooled-vs-mean scatter.

---

## 2026-03-08 — Correction: strategy.ipynb parameter overrides

Discovered that strategy.ipynb overrides config.py with different parameters:
- z_entry=4.0, z_exit=2.5, z_stop=5.0, max_hold=300min, cost_bps=0.0 (gross only), start=2020-07-01
- The notebook intentionally runs at 0 bps to measure gross signal edge separately from costs
- The actual strategy config matches C2 in the experiment (z_entry=4, z_exit=2.5, z_stop=5, max_hold=300)
- C1 "Baseline" in experiments was wrong — it reflected config.py defaults, not what the strategy actually runs
- disconnect_analysis.md updated with a correction section at the top


## 2026-03-11: Fix three performance bugs in grid_search_v2.py _eval_combo

### What changed
**File:** `scripts/grid_search_v2.py`

1. **`_init_worker`**: Added `os.environ["POLARS_MAX_THREADS"] = "1"` before setting worker globals. Without this, each of N worker processes spawns its own rayon thread pool (~8 threads each), so N workers × 8 threads = ~80 threads compete on ~14 cores causing massive context-switch overhead (confirmed by 795/2216 `sample` stack frames in `pthread_cond_wait` inside `rechunk_mut_par`).

2. **`_eval_combo` day_prices filter**: Changed `pl.col("timestamp").dt.date() >= lookback_start` (Python `date` object) to `pl.col("timestamp").dt.date() >= pl.lit(lookback_start)`. Without `pl.lit()`, Polars falls back to row-by-row Python object conversion which was consuming 862/2216 profiling samples in `datetime_to_py_object`.

3. **Double filter eliminated**: The old dict comprehension applied `df.filter(...)` twice per ticker per date — once in the comprehension value and once in the `if len(df.filter(...)) > 0` guard. Replaced with an explicit loop that filters once and reuses the result.

### Why
Profiling via macOS `sample` on a 27-hour-old stuck run revealed these as the top two hotspots. The current run cannot benefit (workers already initialized), so user should kill and restart.

## 2026-03-11: Add incremental checkpointing to grid_search_v2.py

### What changed
**File:** `scripts/grid_search_v2.py`

- `output_path = Path(args.output)` moved earlier (before the formation loop) so checkpoint_dir can be derived from it.
- `checkpoint_dir` = `{output_stem}_checkpoints/` sibling directory created at startup.
- Per formation combo, a checkpoint parquet is written with a descriptive name (rolling window, half-life range, p-value, index).
- Inside the `as_completed` loop, checkpoint is overwritten every 25 completed signal combos so partial results survive a crash.
- Final checkpoint written after each formation combo's pool exits.
- Final merged output still written unchanged at the end from `all_results`.
- Removed duplicate `output_path = Path(args.output)` that was previously at the bottom of main().

### Why
The grid search runs for many hours; without checkpoints any crash loses all progress. Each formation combo's checkpoint file is a self-contained parquet loadable for analysis while the run is in flight.

## 2026-03-11: Add timestamped logging to grid_search_v2.py

### What changed
**File:** `scripts/grid_search_v2.py`

- Added `import time`, `from datetime import datetime`, `from typing import TextIO` to imports.
- Added module-level `_log_fh: TextIO | None` and `_log(msg)` function: stamps every message with `[YYYY-MM-DD HH:MM:SS]`, prints to stdout, and flushes to the log file if open.
- Added `_fmt_duration(seconds) -> str` helper for HH:MM:SS formatting.
- `main()` opens `{output_stem}.log` in append mode before calling `_run()`, closes it in a `finally` block. Append mode means successive runs accumulate in one log file.
- Extracted all post-output_path logic into `_run()` so the try/finally in `main()` cleanly wraps everything.
- All `print()` calls replaced with `_log()` throughout `_run()` and `_send_completion_imessage()`.
- Added `run_start` and `sweep_start` monotonic timers; each checkpoint line now shows `elapsed HH:MM:SS | ETA ~HH:MM:SS` so it's easy to read progress from the log timestamps alone.
- Final save line includes total wall-clock duration.

### Why
Grid search runs many hours with no console feedback between checkpoints; log file + timestamps make it easy to check progress and estimate completion time without being at the terminal.

## 2026-03-11: Write README.md

### What changed
**File:** `README.md` (new)

### Why
Added top-level README covering the full pipeline: data fetching via EODHD, preprocessing, formation search, grid search, backtest via notebook or debug.py, and results directory layout. Includes setup instructions, env var config (EODHD_KEY, NOTIFY_IMESSAGE_TO), and pointer to final_report.ipynb. All commands use `uv run python -m` to match the pyproject.toml build setup.

## 2026-03-12 — Added exit reason analysis to strategy_comparison.ipynb

Added two new cells after the Drawdown plot in `notebooks/strategy_comparison.ipynb`:

1. **Summary table** (`exit_df`): groups trades by `(Strategy, exit_reason)` and shows count, % of trades, total gross P&L, total net P&L, avg gross/net per trade, and win rate. Rendered via `pandas Styler` with formatted columns.

2. **Bar chart** (side-by-side): two subplots — one for gross P&L by exit reason, one for net P&L — with one bar group per strategy, so cost impact is immediately visible across `z_exit`, `z_stop`, and `max_hold`.

Motivation: user requested an explicit breakdown of exit reasons including total gross and net P&L, which makes it easy to see which exit type drives the large cost drag in the `cb=3.0` run.

## 2026-03-12 — Reordered exit reason table index

Changed the multi-index in the exit reason table from `(Strategy, Exit Reason)` to `(Exit Reason, Strategy)` so all strategies appear side-by-side under each exit reason, making cross-strategy comparison easier. Also updated the chart's `loc` lookup to match the new index order `(reason, lbl)`.

## 2026-03-12 — Added formation group filter to grid_search_v2.ipynb

Inserted two cells after the data load cell (cell-2):
1. Markdown explaining the filter feature
2. Code cell that:
   - Defines `FORMATION_PARAMS` = the 4 pair-selection columns
   - Prints a table of all available formation combos (with index) for easy reference
   - Exposes `FORMATION_FILTER` dict — user sets this to a specific combo or leaves it `None` for all
   - Re-assigns `df` filtered to that combo so all downstream cells (IC distributions, top combos, heatmaps, etc.) automatically reflect the selection without any further changes
   - Sets `FORMATION_LABEL` for potential use in chart titles

No other cells were modified. The pattern (narrow df early, everything else just works) avoids duplicating any analysis logic.

## 2026-03-12 — Updated IS/OOS periods in final_plan.md

Changed A2 and Sections 8/9 to reflect the correct split:
- IS: 2017-2019 (matches actual param fitting: formation on 2017-2018, signals on 2019)
- OOS: 2020-2024 (5 years; 2020 included but flagged explicitly as COVID stress year)
- Extended OOS: 2025-01-01 through 2026-02-28 (recent history)
Also aligned A1 ablation to use 2019 (IS period) rather than 2021-2024.

## 2026-03-12 — Updated benchmark approach in final_plan.md

Changed A6 and Section 2 from equal-weight long-only to:
- Primary benchmark: risk-free rate (max 3-month T-bill over full period — conservative single constant)
- Market neutrality check: SPY beta regression (beta, alpha, correlation) to validate benchmark choice
- SPY shown alongside equity curve for regime context only, not as performance comparison
Rationale: long-only benchmarks are inappropriate for dollar-neutral market-neutral strategies.

## 2026-03-12 — Created scripts/fetch_spy.py

New standalone script to fetch daily adjusted SPY prices from EODHD.
Saves to data/benchmark/spy_daily.parquet (separate from data/raw/ universe data).
Reuses EODHD_KEY env var and CONFIG.api.base_url from utils/config.py but does not import fetch_data.py — intentionally kept independent so it can be run without touching the universe pipeline.
Columns: date, open, high, low, close, adjusted_close, volume.
Defaults to 2017-01-01 → 2026-02-28 to cover IS + OOS + extended OOS.
--force flag to overwrite; otherwise skips if file already exists.

## 2026-03-12 — Implemented A3: p-value weighted position sizing

Added `use_pvalue_weights: bool = False` to `PortfolioConfig` in `utils/config.py`.
In `strategy/backtester.py`:
- Read `use_pvalue_weights` from config at top of `run_backtest()`
- At SOD: if enabled, sort pairs by p_value ascending, take top max_pairs, compute weights as `-log(p_value)` normalized to sum to 1, store as `_pair_notionals` dict keyed by (ticker_a, ticker_b)
- At entry: look up pair's notional from dict (fallback to equal notional_per_pair if not found)

Design: -log(p) avoids explosion for very small p-values; normalizing over top max_pairs (not all pairs) ensures total notional = portfolio_value when all slots fill, consistent with equal-weight baseline.

## 2026-03-12 — Created scripts/fetch_tbill.py (A6 benchmark)

Added yfinance as a dependency (uv add yfinance). Fetches ^IRX (CBOE 13-Week Treasury Bill index) from Yahoo Finance — annualised yield in %, no API key required. Saves to data/benchmark/tbill_3m_daily.parquet (same benchmark dir as spy_daily.parquet). Outputs max yield as a convenience since the plan uses peak rate as the conservative Sharpe hurdle. 2017-2026 gives max of ~5.35%.

Tried pandas_datareader (FRED) first but it's broken on pandas 3.0 (deprecate_kwarg API change). yfinance ^IRX is equivalent data (13-week T-bill yield) and has no auth requirement.

## 2026-03-12: Built notebooks/final_report.ipynb

Created `scripts/build_final_report.py` — a builder script that programmatically generates the final report notebook using Python's json library. This avoids JSON escaping issues that would arise from writing the notebook as raw JSON.

The notebook has 53 cells (24 code, 29 markdown) covering all 12 sections mapped to the 10 rubric criteria:
- Section 0: Setup — imports, file paths, helper functions (sharpe, max_dd, summary_stats, load_portfolio, load_trades)
- Section 1: Strategy overview + hypothesis table (H1–H7)
- Section 2: Benchmarks — T-bill max yield, SPY beta regression, equity curve with regime overlay
- Section 3: Data description — sector composition bar chart, coverage table
- Section 4: Formation indicator analysis — mean_n_pairs vs window, composite score heatmap
- Section 5: Signal process analysis — IC histogram, IC t-stat vs z_entry, hit rate, breakeven bps
- Section 6: Rule ablation — 4 configs overlaid equity curves + summary stats table
- Section 7: Parameter optimization — formation heatmap, signal IC heatmap, zscore_window sensitivity
- Section 8: Walk-forward analysis — IS/OOS/Extended split, equity curve with dividers
- Section 9: Overfitting assessment — Sharpe comparison bar chart, parameter stability
- Section 10: Full backtest results — 3-panel equity curve, trade stats, monthly returns heatmap
- Section 11: Extensions — p-value weighting illustration, Mahalanobis + Kalman conceptual descriptions
- Section 12: Conclusions — hypothesis results table, key findings, limitations, future work

Key data mappings used:
- IS/OOS split from single full-period file: daily_pnl_..._mp20_mh9999_mbr8_cb3_2017-01-01_2026-02-28.parquet
- Ablation: 4 IS-period files (zs=999/6/6/6, mh=9999/9999/9999/120)
- Benchmark: data/benchmark/tbill_3m_daily.parquet + spy_daily.parquet
- Formation: results/formation_search.parquet (81 rows)
- Signal: results/grid_search_v2.parquet (2916 rows)

Bug fixed: 
 in triple-quoted builder strings was expanded to literal newlines inside set_title() calls, causing SyntaxError. Fixed by removing 
 from those specific title strings.


## 2026-03-12: Expanded Section 1 of final_report.ipynb with content from strategy_summary.ipynb

Replaced the lean Section 1 overview with rich content from strategy_summary.ipynb, adapted to match the actual work done:

1. Introduction paragraph (adapted: 154 tickers/8 sector groups, 15-min bars, correct params)
2. Point 1: Cointegration-based pair selection (adapted: half-life 2-24 bars = 30min-6h, 42-day window)
3. Point 2: Z-score signal generation (adapted: z_entry=3.0, z_exit=2.25, z_stop=6.0, 5-day rolling window)
4. Point 3: Market-neutral position construction (adapted: 20 pairs, $1M capital, 3 bps/leg)
5. Point 4: Intraday implementation (adapted: 15-min resampled, 154 tickers, 8 sector groups)
6. Hypothesis Testing Framework — full H₀/H₁/Test/Metrics/Expected for H1-H7 (adapted to actual work)
7. Hypothesis cross-reference summary code cell (updated metrics to match actual tests)
8. Literature Review — 6 papers across 3 subsections:
   - Foundational: Engle & Granger (1987), Elliott et al. (2005)
   - Pairs strategies: Gatev et al. (2006), Do & Faff (2010), Do & Faff (2012)
   - Cointegration methods: Avellaneda & Lee (2010), Caldeira & Moura (2013), Alexander & Dimitriu (2005)

Notebook grew from 53 to 63 cells (24 code, 39 markdown).
Key adaptations from strategy_summary: removed references to old H5/H8 hypotheses, updated all parameter values, replaced '11 GICS sectors' with '8 sector groups', updated breakeven discussion to reflect 3 bps vs 15-20 bps historical benchmark.

