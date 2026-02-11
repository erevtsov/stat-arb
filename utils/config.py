"""Centralized configuration for the statistical arbitrage trading system."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # --- Directory paths ---
    "data_dir": str(PROJECT_ROOT / "data"),
    "raw_dir": str(PROJECT_ROOT / "data" / "raw" / "1min"),
    "raw_eod_dir": str(PROJECT_ROOT / "data" / "raw" / "eod"),
    "splits_dir": str(PROJECT_ROOT / "data" / "raw" / "splits"),
    "dividends_dir": str(PROJECT_ROOT / "data" / "raw" / "dividends"),
    "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "results_dir": str(PROJECT_ROOT / "results"),

    # --- EODHD API ---
    "eodhd_base_url": "https://eodhd.com/api",
    "exchange": "US",

    # --- Strategy parameters ---
    "z_score_entry": 2.5,
    "z_score_exit": 0.0,
    "z_score_stop": 4.0,
    "max_holding_minutes": 120,
    "transaction_cost_bps": 20,  # per leg

    # --- Portfolio parameters ---
    "capital": 100_000,
    "max_pairs": 10,

    # --- Cointegration parameters ---
    "coint_p_value_threshold": 0.05,
    "min_half_life": 5,       # minimum half-life in days
    "max_half_life": 120,     # maximum half-life in days

    # --- Spread / z-score parameters ---
    "zscore_window": 60,      # rolling window in bars (default for 1-hour)

    # --- Universe: 100 US equities, stratified by GICS sector ---
    #
    # ~9-10 liquid large/mid-cap names per sector chosen to maximise
    # the number of plausible within-sector pairs for cointegration.
    "sector_mapping": {
        # ── Technology (10) ─────────────────────────────────────
        "AAPL":  "Technology",
        "MSFT":  "Technology",
        "GOOGL": "Technology",
        "META":  "Technology",
        "NVDA":  "Technology",
        "AMD":   "Technology",
        "INTC":  "Technology",
        "CRM":   "Technology",
        "ADBE":  "Technology",
        "ORCL":  "Technology",
        # ── Semiconductors (9) ──────────────────────────────────
        "AVGO":  "Semiconductors",
        "TXN":   "Semiconductors",
        "QCOM":  "Semiconductors",
        "MU":    "Semiconductors",
        "LRCX":  "Semiconductors",
        "AMAT":  "Semiconductors",
        "KLAC":  "Semiconductors",
        "MCHP":  "Semiconductors",
        "ON":    "Semiconductors",
        # ── Financials (10) ─────────────────────────────────────
        "JPM":   "Financials",
        "BAC":   "Financials",
        "GS":    "Financials",
        "MS":    "Financials",
        "WFC":   "Financials",
        "C":     "Financials",
        "BLK":   "Financials",
        "SCHW":  "Financials",
        "USB":   "Financials",
        "PNC":   "Financials",
        # ── Healthcare / Pharma (10) ────────────────────────────
        "JNJ":   "Healthcare",
        "PFE":   "Healthcare",
        "UNH":   "Healthcare",
        "MRK":   "Healthcare",
        "ABT":   "Healthcare",
        "TMO":   "Healthcare",
        "LLY":   "Healthcare",
        "ABBV":  "Healthcare",
        "BMY":   "Healthcare",
        "AMGN":  "Healthcare",
        # ── Consumer Discretionary (10) ─────────────────────────
        "AMZN":  "Consumer Discretionary",
        "TSLA":  "Consumer Discretionary",
        "HD":    "Consumer Discretionary",
        "MCD":   "Consumer Discretionary",
        "NKE":   "Consumer Discretionary",
        "SBUX":  "Consumer Discretionary",
        "LOW":   "Consumer Discretionary",
        "TJX":   "Consumer Discretionary",
        "BKNG":  "Consumer Discretionary",
        "CMG":   "Consumer Discretionary",
        # ── Consumer Staples (9) ────────────────────────────────
        "WMT":   "Consumer Staples",
        "PG":    "Consumer Staples",
        "COST":  "Consumer Staples",
        "KO":    "Consumer Staples",
        "PEP":   "Consumer Staples",
        "PM":    "Consumer Staples",
        "CL":    "Consumer Staples",
        "MDLZ":  "Consumer Staples",
        "KHC":   "Consumer Staples",
        # ── Energy (9) ──────────────────────────────────────────
        "XOM":   "Energy",
        "CVX":   "Energy",
        "COP":   "Energy",
        "SLB":   "Energy",
        "EOG":   "Energy",
        "MPC":   "Energy",
        "PSX":   "Energy",
        "VLO":   "Energy",
        "OXY":   "Energy",
        # ── Industrials (10) ────────────────────────────────────
        "CAT":   "Industrials",
        "DE":    "Industrials",
        "UNP":   "Industrials",
        "HON":   "Industrials",
        "UPS":   "Industrials",
        "BA":    "Industrials",
        "RTX":   "Industrials",
        "LMT":   "Industrials",
        "GE":    "Industrials",
        "MMM":   "Industrials",
        # ── Communication Services (9) ──────────────────────────
        "GOOG":  "Communication Services",
        "DIS":   "Communication Services",
        "CMCSA": "Communication Services",
        "NFLX":  "Communication Services",
        "T":     "Communication Services",
        "VZ":    "Communication Services",
        "TMUS":  "Communication Services",
        "CHTR":  "Communication Services",
        "EA":    "Communication Services",
        # ── Utilities (7) ───────────────────────────────────────
        "NEE":   "Utilities",
        "DUK":   "Utilities",
        "SO":    "Utilities",
        "D":     "Utilities",
        "AEP":   "Utilities",
        "SRE":   "Utilities",
        "EXC":   "Utilities",
        # ── REITs (7) ───────────────────────────────────────────
        "PLD":   "REITs",
        "AMT":   "REITs",
        "CCI":   "REITs",
        "EQIX":  "REITs",
        "SPG":   "REITs",
        "PSA":   "REITs",
        "O":     "REITs",
    },
}


def get_sector(ticker: str) -> str:
    """Return sector for a ticker, or 'Unknown' if not mapped."""
    return CONFIG["sector_mapping"].get(ticker, "Unknown")


def get_tickers_by_sector(sector: str) -> list[str]:
    """Return all tickers belonging to a given sector."""
    return [
        t for t, s in CONFIG["sector_mapping"].items()
        if s == sector
    ]


def get_all_sectors() -> list[str]:
    """Return sorted list of unique sectors."""
    return sorted(set(CONFIG["sector_mapping"].values()))


def get_all_tickers() -> list[str]:
    """Return sorted list of all tickers in the universe."""
    return sorted(CONFIG["sector_mapping"].keys())
