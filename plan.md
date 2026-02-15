# Hypothesis Testing Implementation Plan

## Current State

**Existing code:**
- `analysis/preprocessing.py` — Data loading, cleaning, split adjustment, resampling (1min/5min/15min/1hour/daily). `load_processed()` provides efficient Parquet loading with date filtering.
- `analysis/cointegration.py` — Engle-Granger cointegration testing, pair discovery (`find_cointegrated_pairs()`), rolling window analysis (`find_cointegrated_pairs_rolling()`), pair stability tracking (`track_pair_stability()`).
- `utils/config.py` — All strategy parameters (z_score_entry=2.5, z_score_stop=4.0, transaction_cost_bps=20, max_pairs=10, etc.), 100-ticker universe across 11 GICS sectors, helper functions.
- `scripts/fetch_data.py` — EODHD API data fetcher (data already fetched: 2022-01-01 to 2025-12-31).

**What's missing (needed for hypothesis testing):**
- Signal generation module (z-score calculation, entry/exit logic)
- Backtesting engine (trade simulation, position tracking, P&L)
- Performance metrics (Sharpe, drawdown, win rate, etc.)
- Hypothesis test implementations

**Available data:** 100 tickers, 1003 daily bars each (2022-01-01 to 2025-12-31), plus intraday at 1min/5min/15min/1hour. 104 cointegrated pairs found on 15min data for H1 2023.

---

## Phase 0: Build Core Infrastructure

### Step 0.1: Create `strategy/signals.py` — Signal Generation

Compute z-scores and generate entry/exit signals from cointegrated pair data.

```python
# Key functions:
def compute_spread(prices_a, prices_b, hedge_ratio) -> pl.Series
    # spread = prices_a - hedge_ratio * prices_b

def compute_zscore(spread, window) -> pl.Series
    # z = (spread - rolling_mean) / rolling_std

def generate_signals(zscore, z_entry, z_exit, z_stop) -> pl.DataFrame
    # Returns DataFrame with columns: [timestamp, zscore, signal]
    # signal: 1 (long spread), -1 (short spread), 0 (no position)
    # Entry: z < -z_entry → long, z > +z_entry → short
    # Exit: |z| <= z_exit (mean reversion), |z| > z_stop (stop loss)
```

Uses: `analysis/preprocessing.load_processed()` for price data, `utils/config.CONFIG` for default parameters.

### Step 0.2: Create `strategy/backtester.py` — Backtesting Engine

Simulate trades on historical data with realistic transaction costs.

```python
@dataclass
class Trade:
    pair: tuple[str, str]
    direction: int  # 1=long spread, -1=short spread
    entry_time: datetime
    exit_time: datetime
    entry_zscore: float
    exit_zscore: float
    pnl: float
    pnl_net: float  # after transaction costs
    exit_reason: str  # "mean_reversion", "stop_loss", "time_stop"
    holding_minutes: int

class PairBacktester:
    def __init__(self, z_entry, z_exit, z_stop, max_holding_minutes,
                 transaction_cost_bps, zscore_window)

    def backtest_pair(self, ticker_a, ticker_b, hedge_ratio,
                      timeframe, start_date, end_date) -> list[Trade]
        # Simulate all trades for one pair over a period
        # Uses signals.generate_signals() for entry/exit
        # Applies transaction costs: 4 * cost_bps * notional per round-trip
        # Enforces max_holding_minutes time stop

class PortfolioBacktester:
    def __init__(self, capital, max_pairs, **pair_backtester_kwargs)

    def backtest(self, pairs_df, timeframe, start_date, end_date) -> BacktestResult
        # Runs PairBacktester on all pairs in pairs_df
        # Enforces max_pairs concurrent limit
        # Equal-weight allocation: capital / max_pairs per pair
        # Returns aggregated results
```

Uses: `strategy/signals.py`, `analysis/preprocessing.load_processed()`, `utils/config.CONFIG`.

### Step 0.3: Create `strategy/metrics.py` — Performance Analytics

Calculate risk-adjusted return metrics from trade results.

```python
def calculate_metrics(trades: list[Trade], capital: float) -> dict:
    # Returns dict with:
    # - total_return, annualized_return
    # - sharpe_ratio (annualized)
    # - sortino_ratio
    # - max_drawdown
    # - win_rate (% profitable trades)
    # - profit_factor (gross_profit / gross_loss)
    # - avg_trade_pnl, avg_holding_time
    # - num_trades, trades_per_day
    # - var_95, cvar_95 (Value at Risk, Conditional VaR)
    # - tail_ratio (95th percentile gain / 5th percentile loss)

def build_equity_curve(trades: list[Trade], capital: float) -> pl.DataFrame:
    # Returns time series of portfolio equity for drawdown and plotting
```

Uses: numpy for calculations.

### Step 0.4: Create `strategy/__init__.py`

Empty init file to make the strategy directory a proper package.

---

## Phase 1: Core Validation (Hypotheses 1-4)

### Step 1.1: Hypothesis 1 — Within-Sector Cointegration

**File:** `notebooks/hypothesis_tests.ipynb` (new notebook for all hypothesis tests)

**What it tests:** Within-sector pairs are more likely to be cointegrated than random cross-sector pairs.

**Implementation:**
1. Use existing `find_cointegrated_pairs()` to get within-sector cointegration pass rate (already have: 104/411 pairs on 15min)
2. Add a cross-sector pair testing function that generates ~1000 random cross-sector pair combinations and runs `test_cointegration()` on each
3. Compare pass rates (p < 0.05) between within-sector and cross-sector
4. Chi-square test for independence (`scipy.stats.chi2_contingency`)
5. Calculate odds ratio with 95% CI

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs()`, `analysis/cointegration.test_cointegration()`, `utils/config.get_tickers_by_sector()`, `analysis/preprocessing.load_processed()`

**No backtester needed** — this is purely a cointegration rate comparison.

### Step 1.2: Hypothesis 2 — Z-Score Entry Threshold Optimization

**What it tests:** An optimal z-score entry threshold exists that maximizes Sharpe ratio.

**Implementation:**
1. Use `find_cointegrated_pairs()` to get pairs for formation period (e.g., 2023-01-01 to 2023-06-30)
2. Grid search: z_entry ∈ {1.5, 2.0, 2.5, 3.0, 3.5}
3. For each threshold, run `PortfolioBacktester.backtest()` on trading period (2023-07-01 to 2023-12-31)
4. Compare Sharpe, win rate, trade frequency, avg P&L across thresholds
5. Bootstrap resampling (1000 iterations) for confidence intervals on Sharpe

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs()`, `analysis/preprocessing.load_processed()`
**New code used:** `strategy/backtester.PortfolioBacktester`, `strategy/metrics.calculate_metrics`

### Step 1.3: Hypothesis 3 — Transaction Cost Sensitivity

**What it tests:** Strategy profitability sensitivity to transaction cost assumptions.

**Implementation:**
1. Use best-performing pairs from formation period
2. Grid search: transaction_cost_bps ∈ {5, 10, 15, 20, 25, 30}
3. For each cost level, run `PortfolioBacktester.backtest()` with that cost
4. Plot Sharpe ratio and annualized return vs. cost
5. Find breakeven cost level (interpolate where Sharpe = 0)
6. Calculate sensitivity coefficient: ΔSharpe / Δcost

**Existing code used:** Same as H2
**New code used:** `strategy/backtester.PortfolioBacktester` (varying `transaction_cost_bps`)

### Step 1.4: Hypothesis 4 — In-Sample vs Out-of-Sample Performance Degradation

**What it tests:** Whether in-sample optimized parameters degrade out-of-sample.

**Implementation:**
1. Split data: IS = 2023-01-01 to 2024-03-31 (60%), OOS = 2024-04-01 to 2025-06-30 (40%)
2. IS optimization: grid search z_entry, zscore_window, z_stop on IS data
3. Run IS-optimal parameters on OOS data
4. Compare IS vs OOS Sharpe, return, drawdown, win rate
5. Calculate degradation: (Sharpe_IS - Sharpe_OOS) / Sharpe_IS
6. Bootstrap confidence intervals for both IS and OOS Sharpe

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs()`
**New code used:** `strategy/backtester.PortfolioBacktester`, `strategy/metrics.calculate_metrics`

---

## Phase 2: Robustness Testing (Hypotheses 5-8)

### Step 2.1: Hypothesis 5 — Half-Life Stability Over Time

**What it tests:** Whether half-life estimates are stable across time windows for cointegrated pairs.

**Implementation:**
1. Use existing `track_pair_stability()` with quarterly windows across 2023-2025
2. For each pair, calculate coefficient of variation (CV = std/mean) of half-life
3. Classify pairs: stable (CV < 0.5) vs unstable (CV >= 0.5)
4. Compare backtest performance of stable vs unstable pairs
5. Correlation between formation-period half-life and trading-period half-life

**Existing code used:** `analysis/cointegration.track_pair_stability()` — this is directly designed for this test
**New code used:** `strategy/backtester.PairBacktester` for performance comparison

### Step 2.2: Hypothesis 6 — Stop Loss Effectiveness

**What it tests:** Whether z_stop = 4.0 improves risk-adjusted returns vs no stop loss.

**Implementation:**
1. Run backtests with z_stop ∈ {3.0, 3.5, 4.0, 4.5, 5.0, inf (no stop)}
2. Compare: max drawdown, Sharpe, Sortino, CVaR, tail ratio
3. Analyze percentage of trades stopped out vs mean-reverted at each level
4. Compare P&L distribution tails with and without stops

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs()`
**New code used:** `strategy/backtester.PortfolioBacktester` (varying `z_stop`), `strategy/metrics.calculate_metrics`

### Step 2.3: Hypothesis 7 — Portfolio Diversification Benefits

**What it tests:** Whether trading 10 pairs reduces volatility vs fewer pairs.

**Implementation:**
1. Run backtests with max_pairs ∈ {3, 5, 10, 15, 20}
2. For each portfolio size, select top N pairs by p-value
3. Compare: portfolio volatility, Sharpe, max drawdown
4. Calculate average pairwise correlation between pair P&Ls
5. Compute diversification ratio and marginal benefit of adding pairs

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs()` (sorted by p-value)
**New code used:** `strategy/backtester.PortfolioBacktester` (varying `max_pairs`), `strategy/metrics`

### Step 2.4: Hypothesis 8 — Walk-Forward Robustness

**What it tests:** Whether the strategy delivers consistent returns across rolling walk-forward windows.

**Implementation:**
1. Use existing `find_cointegrated_pairs_rolling(window_months=6, step_months=3)` for formation
2. For each formation window, trade on the subsequent 3-month holdout
3. Calculate OOS Sharpe for each period
4. Report: mean OOS Sharpe, std, % positive periods, % with Sharpe > 1.0
5. Time series plot of OOS Sharpe to identify regime changes
6. Windows: 2023-01 through 2025-06 (approximately 8 walk-forward periods)

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs_rolling()` — directly designed for this
**New code used:** `strategy/backtester.PortfolioBacktester`, `strategy/metrics.calculate_metrics`

---

## Phase 3: Refinement (Hypothesis 9)

### Step 3.1: Hypothesis 9 — P-Value as Pair Quality Signal

**What it tests:** Whether lower ADF p-values predict higher pair profitability.

**Implementation:**
1. Sort all cointegrated pairs by p-value into quintiles (Q1 = lowest p-value)
2. Run separate backtests for each quintile
3. Compare Sharpe, win rate, avg P&L across quintiles
4. Test for monotonic relationship: Sharpe(Q1) > Sharpe(Q2) > ... > Sharpe(Q5)
5. Regression: pair Sharpe on p-value rank

**Existing code used:** `analysis/cointegration.find_cointegrated_pairs()` (results already sorted by p-value)
**New code used:** `strategy/backtester.PairBacktester`, `strategy/metrics.calculate_metrics`

---

## Implementation Order

```
1. strategy/__init__.py          (trivial)
2. strategy/signals.py           (depends on: analysis/preprocessing, utils/config)
3. strategy/metrics.py           (standalone, depends on: Trade dataclass)
4. strategy/backtester.py        (depends on: signals.py, metrics.py)
5. H1: Within-Sector Coint.     (no backtester needed — uses cointegration.py only)
6. H5: Half-Life Stability       (uses track_pair_stability() + backtester)
7. H2: Z-Score Threshold Opt.    (uses backtester grid search)
8. H3: Transaction Cost Sens.    (uses backtester grid search)
9. H6: Stop Loss Effectiveness   (uses backtester grid search)
10. H7: Portfolio Diversification (uses backtester with varying max_pairs)
11. H9: P-Value Quality Signal   (uses backtester with quintile split)
12. H4: IS vs OOS Degradation    (uses backtester + grid search optimization)
13. H8: Walk-Forward Robustness  (uses rolling cointegration + backtester)
```

Steps 1-4 build infrastructure. Steps 5-13 are hypothesis tests, each implemented as a section in `notebooks/hypothesis_tests.ipynb`. H1 can start immediately since it needs no backtester.

## Data Periods

- **Formation period:** 2023-01-01 to 2023-06-30 (6 months, pair discovery)
- **Trading period:** 2023-07-01 to 2023-12-31 (6 months, backtesting)
- **Walk-forward range:** 2023-01-01 to 2025-06-30 (~8 rolling windows)
- **IS/OOS split:** IS = 2023-01 to 2024-03 (60%), OOS = 2024-04 to 2025-06 (40%)
- **Primary timeframe:** 15min (balance between frequency and cost sensitivity)
