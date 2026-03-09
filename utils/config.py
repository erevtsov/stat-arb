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
    min_half_life: int = 4  # bars
    max_half_life: int = 24  # bars (12 has worked best)
    rolling_window_days: int = 42  # ~60 trading days in calendar days
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
    transaction_cost_bps: float = 5.0  # per leg


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
    # ── Technology (38: former Technology + Semiconductors) ──
    # Software / IT services
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
    "IBM": "Technology",
    "CSCO": "Technology",
    "NOW": "Technology",
    "INTU": "Technology",
    "PYPL": "Technology",  # spun off from eBay Jul-2015; early weeks may be partial
    "SNPS": "Technology",
    "CDNS": "Technology",
    "ACN": "Technology",
    "FISV": "Technology",
    "FIS": "Technology",
    # Semiconductors (merged from own sector — same GICS IT parent)
    "AVGO": "Technology",
    "TXN": "Technology",
    "QCOM": "Technology",
    "MU": "Technology",
    "LRCX": "Technology",
    "AMAT": "Technology",
    "KLAC": "Technology",
    "MCHP": "Technology",
    "ON": "Technology",
    "ADI": "Technology",
    "MRVL": "Technology",
    "SWKS": "Technology",
    "NXPI": "Technology",
    "QRVO": "Technology",  # formed Jan-2015 (RF Micro + TriQuint merger); early 2015 partial
    "MPWR": "Technology",
    "ASML": "Technology",
    "ENTG": "Technology",
    "TER": "Technology",
    # ── Financials (20) ─────────────────────────────────────
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
    "AXP": "Financials",
    "COF": "Financials",
    "MET": "Financials",
    "PRU": "Financials",
    "TFC": "Financials",  # BB&T+SunTrust merged Dec-2019; pre-2020 data is BB&T history
    "FITB": "Financials",
    "RF": "Financials",
    "CFG": "Financials",
    "MTB": "Financials",
    "KEY": "Financials",
    # ── Healthcare (20) ─────────────────────────────────────
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
    "CVS": "Healthcare",
    "CI": "Healthcare",
    "HUM": "Healthcare",
    "MDT": "Healthcare",
    "SYK": "Healthcare",
    "BSX": "Healthcare",
    "ISRG": "Healthcare",
    "REGN": "Healthcare",
    "VRTX": "Healthcare",
    "BIIB": "Healthcare",
    # ── Consumer (38: former Discretionary + Staples) ────────
    # Discretionary
    "AMZN": "Consumer",
    "TSLA": "Consumer",
    "HD": "Consumer",
    "MCD": "Consumer",
    "NKE": "Consumer",
    "SBUX": "Consumer",
    "LOW": "Consumer",
    "TJX": "Consumer",
    "BKNG": "Consumer",
    "CMG": "Consumer",
    "ROST": "Consumer",
    "ORLY": "Consumer",
    "AZO": "Consumer",
    "DPZ": "Consumer",
    "YUM": "Consumer",
    "MAR": "Consumer",
    "HLT": "Consumer",
    "F": "Consumer",
    "GM": "Consumer",
    "CCL": "Consumer",
    # Staples
    "WMT": "Consumer",
    "PG": "Consumer",
    "COST": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",
    "PM": "Consumer",
    "CL": "Consumer",
    "MDLZ": "Consumer",
    "KHC": "Consumer",
    "EL": "Consumer",
    "MO": "Consumer",
    "GIS": "Consumer",
    "HSY": "Consumer",
    "SYY": "Consumer",
    "CLX": "Consumer",
    "KMB": "Consumer",
    "CHD": "Consumer",
    "STZ": "Consumer",
    # ── Energy (18) ─────────────────────────────────────────
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "OXY": "Energy",
    "HAL": "Energy",
    "BKR": "Energy",  # Baker Hughes in current form since Jul-2017 (GE O&G merger)
    "DVN": "Energy",
    "FANG": "Energy",
    "MRO": "Energy",
    "WMB": "Energy",
    "KMI": "Energy",
    "APA": "Energy",
    "OKE": "Energy",
    # ── Industrials (20) ────────────────────────────────────
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
    "NOC": "Industrials",
    "GD": "Industrials",
    "FDX": "Industrials",
    "ETN": "Industrials",
    "EMR": "Industrials",
    "PH": "Industrials",
    "ITW": "Industrials",
    "AME": "Industrials",
    "ROK": "Industrials",
    "XYL": "Industrials",
    # ── Communication Services (18) ─────────────────────────
    "GOOG": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "NFLX": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "TMUS": "Communication Services",
    "CHTR": "Communication Services",
    "EA": "Communication Services",
    "PARA": "Communication Services",  # continuous listing: CBS Corp → VIAC → PARA
    "LYV": "Communication Services",
    "TTWO": "Communication Services",
    "IPG": "Communication Services",
    "OMC": "Communication Services",
    "NWSA": "Communication Services",
    "SIRI": "Communication Services",
    "MTCH": "Communication Services",  # IPO Nov-2015; first weeks may be partial
    "LBTYA": "Communication Services",
    # ── Real Assets (28: former Utilities + REITs) ───────────
    # Utilities
    "NEE": "Real Assets",
    "DUK": "Real Assets",
    "SO": "Real Assets",
    "D": "Real Assets",
    "AEP": "Real Assets",
    "SRE": "Real Assets",
    "EXC": "Real Assets",
    "PPL": "Real Assets",
    "ED": "Real Assets",
    "FE": "Real Assets",
    "ES": "Real Assets",
    "WEC": "Real Assets",
    "XEL": "Real Assets",
    "AWK": "Real Assets",
    # REITs
    "PLD": "Real Assets",
    "AMT": "Real Assets",
    "CCI": "Real Assets",
    "EQIX": "Real Assets",
    "SPG": "Real Assets",
    "PSA": "Real Assets",
    "O": "Real Assets",
    "VTR": "Real Assets",
    "WELL": "Real Assets",
    "EXR": "Real Assets",
    "AVB": "Real Assets",
    "EQR": "Real Assets",
    "DLR": "Real Assets",
    "ARE": "Real Assets",
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
