"""Centralized configuration for the statistical arbitrage trading system."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # --- Directory paths ---
    "data_dir": str(PROJECT_ROOT / "data"),
    "raw_dir": str(PROJECT_ROOT / "data" / "raw" / "1min"),
    "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "results_dir": str(PROJECT_ROOT / "results"),

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

    # --- Sector mapping ---
    "sector_mapping": {
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
        "JPM": "Financials",
        "BAC": "Financials",
        "GS": "Financials",
        "MS": "Financials",
        "WFC": "Financials",
        "C": "Financials",
        "BLK": "Financials",
        "SCHW": "Financials",
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
        "UNH": "Healthcare",
        "MRK": "Healthcare",
        "ABT": "Healthcare",
        "TMO": "Healthcare",
        "LLY": "Healthcare",
        "AMZN": "Consumer",
        "WMT": "Consumer",
        "HD": "Consumer",
        "COST": "Consumer",
        "NKE": "Consumer",
        "MCD": "Consumer",
        "SBUX": "Consumer",
        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "SLB": "Energy",
        "EOG": "Energy",
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
