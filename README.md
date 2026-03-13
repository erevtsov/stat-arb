# Statistical Arbitrage: Intraday Pairs Trading

An intraday mean-reversion pairs trading strategy for US equities, built on Engle-Granger cointegration analysis. Each trading day, cointegrated pairs within the same GICS sector are identified using a rolling formation window; z-score normalized spreads generate entry and exit signals with dollar-neutral position sizing.

---

## Repository Structure

```
stat-arb/
├── analysis/
│   ├── cointegration.py    # Engle-Granger ADF tests, hedge ratio OLS, half-life AR(1)
│   ├── signals.py          # Log-spread construction, rolling z-score, signal generation
│   ├── evaluation.py       # Forward-return IC, hit rate, breakeven cost metrics
│   └── preprocessing.py    # Raw 1-min bar loading, resampling to 5min/15min, splits/divs
│
├── strategy/
│   └── backtester.py       # Day-by-day portfolio simulation, position state machine
│
├── scripts/
│   ├── fetch_data.py       # EODHD API downloader (1-min OHLCV, EOD, splits, dividends)
│   ├── formation_search.py # Stage A: sweep formation params on pair quality metrics
│   ├── grid_search_v2.py   # Stage B: sweep signal params on IC/return metrics (parallel)
│   └── debug.py            # Quick single-run backtest for development
│
├── utils/
│   └── config.py           # Centralized config dataclasses (paths, signal, portfolio params)
│
├── notebooks/
│   ├── strategy.ipynb          # Interactive backtest runner + full performance tearsheet
│   ├── strategy_summary.ipynb  # Strategy theory, hypotheses, literature review
│   ├── final_report.ipynb      # Complete project report (all rubric sections)
│   └── grid_search_v2.ipynb    # Grid search results explorer
│
├── results/
│   ├── portfolio/          # Daily P&L parquets, one file per backtest config
│   ├── trades/             # Trade-level logs
│   ├── params/             # Signal grid search results
│   ├── formation_search.parquet  # Formation parameter quality sweep results
│   └── grid_search_v2_checkpoints/  # Incremental signal sweep checkpoints
│
├── data/
│   ├── raw/                # 1-min OHLCV, EOD, splits, dividends (from EODHD)
│   └── processed/          # Resampled 5min/15min parquets per ticker
│
├── tests/                  # pytest unit tests for backtester and signal logic
└── progress/               # Development notes, plan, reasoning log
```

---

## Setup

**Install dependencies** (requires Python ≥ 3.11):

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

**Configure environment** — create a `.env` file in the project root:

```bash
# Required for data fetching
EODHD_KEY=your_eodhd_api_key_here

# Optional: iMessage completion notifications (macOS only)
NOTIFY_IMESSAGE_TO=+1xxxxxxxxxx
```

The `EODHD_KEY` is only needed to run `scripts/fetch_data.py`. All other scripts and notebooks work with pre-downloaded data.

---

## Data Pipeline

### 1. Fetch raw data

Downloads 1-min OHLCV bars, daily EOD, splits, and dividends for all universe tickers via EODHD API:

```bash
uv run python scripts/fetch_data.py              # fetch all tickers (5 workers)
uv run python scripts/fetch_data.py --workers 10 # faster with more threads
uv run python scripts/fetch_data.py --force      # re-fetch and overwrite existing
```

Raw data lands in `data/raw/`. The universe (~100 large-cap US equities across 11 GICS sectors) is defined in `utils/config.py`.

### 2. Preprocess

Resampling and corporate action adjustment happens automatically inside `analysis/preprocessing.py` via `load_processed()` — called on demand by all downstream scripts. There is no separate preprocessing step to run.

---

## Running the Formation Search

Stage A sweeps formation parameters (rolling window length, half-life bounds, ADF p-value threshold) on pair *quality* metrics only, without running any signal backtest. Use this to identify which formation configs produce stable, plentiful pairs.

```bash
uv run python -m scripts.formation_search \
    --start 2017-01-01 --end 2018-12-31 \
    --timeframe 15min
```

Output: `results/formation_search.parquet` — one row per formation combo, ranked by composite quality score. The top configs are then fed into the grid search.

---

## Running the Grid Search

Stage B sweeps signal parameters (z-score window, entry/exit/stop thresholds, max holding time) in parallel across workers. Two phases:

- **Phase 1** — fixed formation params (from `FORMATION_DEFAULTS`), ~324 signal combos:
  ```bash
  uv run python -m scripts.grid_search_v2 \
      --phase 1 --start 2019-01-01 --end 2019-09-30
  ```

- **Phase 2** — full formation × signal grid (use after formation search):
  ```bash
  uv run python -m scripts.grid_search_v2 \
      --phase 2 --start 2019-01-01 --end 2019-09-30
  ```

Output: `results/grid_search_v2.parquet` + incremental checkpoints in `results/grid_search_v2_checkpoints/`. A timestamped log is written to `results/grid_search_v2.log`.

**Note:** The grid search can run for many hours. Checkpoints are written every 25 completed combos so progress survives interruptions.

---

## Running a Backtest

Open `notebooks/strategy.ipynb` and configure the `Config` object:

```python
from utils.config import Config
from strategy.backtester import run_backtest

cfg = Config()
cfg.portfolio.start_date            = "2021-01-01"
cfg.portfolio.end_date              = "2024-12-31"
cfg.portfolio.capital               = 1_000_000.0
cfg.portfolio.max_pairs             = 20
cfg.portfolio.transaction_cost_bps  = 0.0
cfg.cointegration.rolling_window_days = 42
cfg.signal.z_entry                  = 3.0
cfg.signal.z_exit                   = 1.5
cfg.signal.z_stop                   = 4.0

trades, daily_pnl = run_backtest(config=cfg)
```

For a quick command-line run, edit and execute `scripts/debug.py`:

```bash
uv run python scripts/debug.py
```

---

## Results

| Path | Contents |
|------|----------|
| `results/portfolio/` | Daily P&L parquet per backtest run; filename encodes all params |
| `results/trades/` | Trade-level logs with entry/exit timestamps, tickers, P&L |
| `results/params/params_15min.parquet` | Signal grid search: IC, hit rate, net return per combo |
| `results/formation_search.parquet` | Formation quality sweep: pairs/day, half-life, composite score |
| `results/grid_search_v2.parquet` | Full signal sweep output (when complete) |

---

## Final Report

`notebooks/final_report.ipynb` is a single self-contained report covering all grading criteria: strategy summary, data description, indicator testing, signal testing, incremental rule ablation, parameter optimization, IS/OOS walk-forward analysis, overfitting assessment, and extensions (p-value weighted sizing, Mahalanobis distance pairs, Kalman filter hedge ratio).

See `progress/final_plan.md` for the full report plan and implementation checklist.
