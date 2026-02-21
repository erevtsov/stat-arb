"""Centralized configuration for the statistical arbitrage trading system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

BARS_PER_DAY: dict[str, int] = {
    "1min": 390,
    "5min": 78,
    "15min": 26,
    "1hour": 7,
    "daily": 1,
}

MINUTES_PER_BAR: dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "1hour": 60,
    "daily": 390,
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    data_dir: str = str(PROJECT_ROOT / "data")
    raw_dir: str = str(PROJECT_ROOT / "data" / "raw" / "1min")
    raw_eod_dir: str = str(PROJECT_ROOT / "data" / "raw" / "eod")
    splits_dir: str = str(PROJECT_ROOT / "data" / "raw" / "splits")
    dividends_dir: str = str(PROJECT_ROOT / "data" / "raw" / "dividends")
    processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    results_dir: str = str(PROJECT_ROOT / "results")


@dataclass
class ApiConfig:
    base_url: str = "https://eodhd.com/api"
    api_key: str = "your_key_here"  # override via EODHD_KEY env var at runtime
    exchange: str = "US"
    max_workers: int = 5
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    intraday_chunk_days: int = 120
    request_delay: float = 0.2


@dataclass
class CointegrationConfig:
    p_value_threshold: float = 0.05
    min_half_life: int = 5  # trading days
    max_half_life: int = 120  # trading days
    rolling_window_days: int = 84  # ~60 trading days in calendar days
    rolling_step_days: int = 1  # daily recomputation


@dataclass
class SignalConfig:
    zscore_window_days: int = 5  # converted to bars at runtime
    z_entry: float = 2.5
    z_exit: float = 0.0
    z_stop: float = 4.0
    max_holding_minutes: int = 120  # converted to bars at runtime


@dataclass
class PortfolioConfig:
    capital: float = 100_000.0
    max_pairs: int = 10
    transaction_cost_bps: float = 20.0  # per leg


@dataclass
class EvaluationConfig:
    holding_horizons_days: list[float] = field(
        default_factory=lambda: [0.15, 0.31, 0.62, 1.0]
    )  # fractions of a trading day → [4, 8, 16, 26] bars at 15min


@dataclass
class UniverseConfig:
    sector_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    cointegration: CointegrationConfig = field(default_factory=CointegrationConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

CONFIG = Config()
# Override API key from environment if set
CONFIG.api.api_key = os.environ.get("EODHD_KEY", CONFIG.api.api_key)

CONFIG.universe.sector_mapping = {
    # ── Technology (10) ─────────────────────────────────────
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "ORCL": "Technology",
    # ── Semiconductors (9) ──────────────────────────────────
    "AVGO": "Semiconductors",
    "TXN": "Semiconductors",
    "QCOM": "Semiconductors",
    "MU": "Semiconductors",
    "LRCX": "Semiconductors",
    "AMAT": "Semiconductors",
    "KLAC": "Semiconductors",
    "MCHP": "Semiconductors",
    "ON": "Semiconductors",
    # ── Financials (10) ─────────────────────────────────────
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    # ── Healthcare / Pharma (10) ────────────────────────────
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    "MRK": "Healthcare",
    "ABT": "Healthcare",
    "TMO": "Healthcare",
    "LLY": "Healthcare",
    "ABBV": "Healthcare",
    "BMY": "Healthcare",
    "AMGN": "Healthcare",
    # ── Consumer Discretionary (10) ─────────────────────────
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    # ── Consumer Staples (9) ────────────────────────────────
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PM": "Consumer Staples",
    "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "KHC": "Consumer Staples",
    # ── Energy (9) ──────────────────────────────────────────
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "OXY": "Energy",
    # ── Industrials (10) ────────────────────────────────────
    "CAT": "Industrials",
    "DE": "Industrials",
    "UNP": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "BA": "Industrials",
    "RTX": "Industrials",
    "LMT": "Industrials",
    "GE": "Industrials",
    "MMM": "Industrials",
    # ── Communication Services (9) ──────────────────────────
    "GOOG": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "NFLX": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "TMUS": "Communication Services",
    "CHTR": "Communication Services",
    "EA": "Communication Services",
    # ── Utilities (7) ───────────────────────────────────────
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "SRE": "Utilities",
    "EXC": "Utilities",
    # ── REITs (7) ───────────────────────────────────────────
    "PLD": "REITs",
    "AMT": "REITs",
    "CCI": "REITs",
    "EQIX": "REITs",
    "SPG": "REITs",
    "PSA": "REITs",
    "O": "REITs",
}


# ---------------------------------------------------------------------------
# Timeframe conversion helpers
# ---------------------------------------------------------------------------


def zscore_window_bars(timeframe: str) -> int:
    """Convert zscore_window_days to bars for the given timeframe."""
    return CONFIG.signal.zscore_window_days * BARS_PER_DAY[timeframe]


def max_holding_bars(timeframe: str) -> int:
    """Convert max_holding_minutes to bars for the given timeframe."""
    return CONFIG.signal.max_holding_minutes // MINUTES_PER_BAR[timeframe]


def holding_horizons_bars(timeframe: str) -> list[int]:
    """Convert holding_horizons_days fractions to bar counts for the given timeframe."""
    return [
        round(d * BARS_PER_DAY[timeframe])
        for d in CONFIG.evaluation.holding_horizons_days
    ]


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------


def get_sector(ticker: str) -> str:
    """Return sector for a ticker, or 'Unknown' if not mapped."""
    return CONFIG.universe.sector_mapping.get(ticker, "Unknown")


def get_tickers_by_sector(sector: str) -> list[str]:
    """Return all tickers belonging to a given sector."""
    return [t for t, s in CONFIG.universe.sector_mapping.items() if s == sector]


def get_all_sectors() -> list[str]:
    """Return sorted list of unique sectors."""
    return sorted(set(CONFIG.universe.sector_mapping.values()))


def get_all_tickers() -> list[str]:
    """Return sorted list of all tickers in the universe."""
    return sorted(CONFIG.universe.sector_mapping.keys())
