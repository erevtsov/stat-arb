"""Tests for utils/config.py."""

import pytest

from utils.config import (
    CONFIG,
    Config,
    PathsConfig,
    ApiConfig,
    CointegrationConfig,
    SignalConfig,
    PortfolioConfig,
    EvaluationConfig,
    UniverseConfig,
    zscore_window_bars,
    max_holding_bars,
    holding_horizons_bars,
    get_sector,
    get_all_tickers,
    get_all_sectors,
)


class TestConfigInstantiation:
    def test_config_constructs(self):
        cfg = Config()
        assert isinstance(cfg, Config)

    def test_sub_configs_have_correct_types(self):
        cfg = Config()
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.api, ApiConfig)
        assert isinstance(cfg.cointegration, CointegrationConfig)
        assert isinstance(cfg.signal, SignalConfig)
        assert isinstance(cfg.portfolio, PortfolioConfig)
        assert isinstance(cfg.evaluation, EvaluationConfig)
        assert isinstance(cfg.universe, UniverseConfig)

    def test_singleton_is_config_instance(self):
        assert isinstance(CONFIG, Config)


class TestZscoreWindowBars:
    def test_15min(self):
        # 5 days * 26 bars/day = 130
        assert zscore_window_bars("15min") == 130

    def test_1hour(self):
        # 5 days * 7 bars/day = 35
        assert zscore_window_bars("1hour") == 35

    def test_1min(self):
        # 5 days * 390 bars/day = 1950
        assert zscore_window_bars("1min") == 1950

    def test_daily(self):
        # 5 days * 1 bar/day = 5
        assert zscore_window_bars("daily") == 5


class TestMaxHoldingBars:
    def test_15min(self):
        # 120 minutes // 15 = 8
        assert max_holding_bars("15min") == 8

    def test_1hour(self):
        # 120 minutes // 60 = 2
        assert max_holding_bars("1hour") == 2

    def test_1min(self):
        # 120 minutes // 1 = 120
        assert max_holding_bars("1min") == 120


class TestHoldingHorizonsBars:
    def test_15min_returns_list_of_ints(self):
        horizons = holding_horizons_bars("15min")
        assert isinstance(horizons, list)
        assert all(isinstance(h, int) for h in horizons)

    def test_15min_first_approx_4(self):
        horizons = holding_horizons_bars("15min")
        assert horizons[0] == pytest.approx(4, abs=1)

    def test_15min_last_approx_26(self):
        horizons = holding_horizons_bars("15min")
        assert horizons[-1] == pytest.approx(26, abs=1)

    def test_15min_has_4_horizons(self):
        assert len(holding_horizons_bars("15min")) == 4


class TestGetSector:
    def test_known_ticker(self):
        assert get_sector("AAPL") == "Technology"

    def test_financial_ticker(self):
        assert get_sector("JPM") == "Financials"

    def test_unknown_ticker(self):
        assert get_sector("XYZ") == "Unknown"

    def test_case_sensitive(self):
        # Lowercase should return Unknown (tickers are uppercase)
        assert get_sector("aapl") == "Unknown"


class TestGetAllTickers:
    def test_count_is_100(self):
        tickers = get_all_tickers()
        assert len(tickers) == 100

    def test_returns_sorted_list(self):
        tickers = get_all_tickers()
        assert tickers == sorted(tickers)

    def test_all_tickers_have_sector(self):
        for ticker in get_all_tickers():
            assert get_sector(ticker) != "Unknown", f"{ticker} has no sector"

    def test_no_duplicates(self):
        tickers = get_all_tickers()
        assert len(tickers) == len(set(tickers))
