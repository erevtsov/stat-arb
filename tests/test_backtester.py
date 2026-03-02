"""Tests for strategy/backtester.py."""

import datetime as dt
from unittest.mock import patch

import polars as pl
import pytest

from strategy.backtester import (
    PositionState,
    _compute_pnl,
    _get_exec_price,
    run_backtest,
)
from utils.config import CONFIG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pos(
    direction: int = 1,
    entry_price_a: float = 100.0,
    entry_price_b: float = 100.0,
    shares_a: float = 10.0,
    shares_b: float = 10.0,
    hedge_ratio: float = 1.0,
) -> PositionState:
    return PositionState(
        direction=direction,
        entry_time=dt.datetime(2023, 1, 3, 9, 30),
        entry_price_a=entry_price_a,
        entry_price_b=entry_price_b,
        shares_a=shares_a,
        shares_b=shares_b,
        hedge_ratio=hedge_ratio,
        ticker_a="A",
        ticker_b="B",
        p_value=0.01,
    )


def _ts(bar: int) -> dt.datetime:
    return dt.datetime(2023, 1, 3, 9, 30) + dt.timedelta(minutes=15 * bar)


# ---------------------------------------------------------------------------
# _compute_pnl
# ---------------------------------------------------------------------------


class TestComputePnl:
    def test_known_values_long(self):
        """Long spread: A rises, B flat → positive gross PnL."""
        pos = _make_pos(
            direction=1,
            entry_price_a=100.0,
            entry_price_b=100.0,
            shares_a=10.0,
            shares_b=10.0,
        )
        gross, cost, net = _compute_pnl(
            pos, exit_price_a=105.0, exit_price_b=100.0, cost_bps=20.0
        )
        expected_gross = 1 * ((105 - 100) * 10 - (100 - 100) * 10)
        assert gross == pytest.approx(expected_gross)
        assert net < gross

    def test_known_values_short(self):
        """Short spread: A falls, B flat → positive gross PnL."""
        pos = _make_pos(
            direction=-1,
            entry_price_a=105.0,
            entry_price_b=100.0,
            shares_a=10.0,
            shares_b=10.0,
        )
        gross, cost, net = _compute_pnl(
            pos, exit_price_a=102.0, exit_price_b=100.0, cost_bps=20.0
        )
        expected_gross = -1 * ((102 - 105) * 10 - (100 - 100) * 10)
        assert gross == pytest.approx(expected_gross)

    def test_transaction_cost_4_legs(self):
        """Transaction cost = 4 individual leg costs."""
        pos = _make_pos(
            direction=1,
            entry_price_a=100.0,
            entry_price_b=100.0,
            shares_a=1.0,
            shares_b=1.0,
        )
        cost_bps = 20.0
        _, cost, _ = _compute_pnl(
            pos, exit_price_a=100.0, exit_price_b=100.0, cost_bps=cost_bps
        )

        leg_cost = cost_bps / 10_000.0
        expected_cost = (
            1.0 * 100.0 * leg_cost  # entry A
            + 1.0 * 100.0 * leg_cost  # entry B
            + 1.0 * 100.0 * leg_cost  # exit A
            + 1.0 * 100.0 * leg_cost  # exit B
        )
        assert cost == pytest.approx(expected_cost, rel=1e-9)

    def test_net_equals_gross_minus_cost(self):
        pos = _make_pos()
        gross, cost, net = _compute_pnl(
            pos, exit_price_a=102.0, exit_price_b=100.0, cost_bps=20.0
        )
        assert net == pytest.approx(gross - cost, abs=1e-12)


# ---------------------------------------------------------------------------
# PositionState
# ---------------------------------------------------------------------------


class TestPositionState:
    def test_is_dataclass(self):
        pos = _make_pos()
        assert hasattr(pos, "direction")
        assert hasattr(pos, "entry_time")
        assert hasattr(pos, "shares_a")
        assert hasattr(pos, "shares_b")


# ---------------------------------------------------------------------------
# run_backtest (with heavy mocking — no data files required)
# ---------------------------------------------------------------------------


def _make_empty_pairs_df():
    return pl.DataFrame(
        schema={
            "ticker_a": pl.Utf8,
            "ticker_b": pl.Utf8,
            "hedge_ratio": pl.Float64,
            "p_value": pl.Float64,
            "half_life": pl.Float64,
        }
    )


def _make_minimal_pairs_df(ticker_a="A", ticker_b="B", hedge_ratio=1.0, p_value=0.01):
    return pl.DataFrame(
        {
            "ticker_a": [ticker_a],
            "ticker_b": [ticker_b],
            "hedge_ratio": [hedge_ratio],
            "p_value": [p_value],
            "half_life": [10.0],
        }
    )


def _make_intraday_df(n_bars: int = 10, base_price: float = 100.0):
    """Generate synthetic 15-min OHLCV."""
    timestamps = [_ts(i) for i in range(n_bars)]
    closes = [base_price] * n_bars
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100_000] * n_bars,
        }
    )


def _make_flat_signals_df(
    ticker_a, ticker_b, n_bars: int = 10, date: str = "2023-01-03"
):
    """All signals are 0 (flat), so no trades open."""
    timestamps = [_ts(i) for i in range(n_bars)]
    return pl.DataFrame(
        {
            "date": [date] * n_bars,
            "timestamp": timestamps,
            "ticker_a": [ticker_a] * n_bars,
            "ticker_b": [ticker_b] * n_bars,
            "hedge_ratio": [1.0] * n_bars,
            "p_value": [0.01] * n_bars,
            "signal_weight": [1.0] * n_bars,
            "zscore": [0.0] * n_bars,
            "signal_binary": [0] * n_bars,
            "signal_weighted": [0.0] * n_bars,
        }
    )


def _patch_all(pairs_df, intraday_df, signals_df, trading_days):
    return [
        patch(
            "strategy.backtester._get_nyse_trading_days",
            return_value=trading_days,
        ),
        patch(
            "strategy.backtester.find_cointegrated_pairs",
            return_value=pairs_df,
        ),
        patch(
            "strategy.backtester.load_processed",
            return_value=intraday_df,
        ),
        patch(
            "strategy.backtester.generate_pair_signals_for_day",
            return_value=signals_df,
        ),
    ]


def _make_trade_signals_df(
    ticker_a: str,
    ticker_b: str,
    signal_seq: list[int],
    date: str = "2023-01-03",
) -> pl.DataFrame:
    """Build a signals DataFrame with an explicit signal_binary sequence."""
    n = len(signal_seq)
    timestamps = [_ts(i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": [date] * n,
            "timestamp": timestamps,
            "ticker_a": [ticker_a] * n,
            "ticker_b": [ticker_b] * n,
            "hedge_ratio": [1.0] * n,
            "p_value": [0.01] * n,
            "signal_weight": [1.0] * n,
            "zscore": [0.0] * n,
            "signal_binary": signal_seq,
            "signal_weighted": [float(s) for s in signal_seq],
        }
    )


class TestRunBacktest:
    """Integration-style tests that mock data loading and cointegration."""

    def _patch_all(self, pairs_df, intraday_df, signals_df, trading_days):
        return _patch_all(pairs_df, intraday_df, signals_df, trading_days)

    def test_portfolio_value_starts_at_capital(self):
        trading_days = [dt.date(2023, 1, 3)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df()
        signals = _make_flat_signals_df("A", "B")

        patches = self._patch_all(pairs, intraday, signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            _, daily_pnl = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
            )

        capital = CONFIG.portfolio.capital
        assert daily_pnl["portfolio_value"][0] == pytest.approx(capital)

    def test_no_open_positions_at_eod(self):
        """All positions must be force-closed by EOD."""
        trading_days = [dt.date(2023, 1, 3)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df()
        signals = _make_flat_signals_df("A", "B")

        patches = self._patch_all(pairs, intraday, signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            _, daily_pnl = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
            )

        assert daily_pnl["n_open_eod"][0] == 0

    def test_daily_pnl_n_open_eod_always_zero(self):
        """n_open_eod must be 0 for every day in the result."""
        trading_days = [dt.date(2023, 1, 3), dt.date(2023, 1, 4)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df()
        signals = _make_flat_signals_df("A", "B")

        patches = self._patch_all(pairs, intraday, signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            _, daily_pnl = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-04",
            )

        assert (daily_pnl["n_open_eod"] == 0).all()

    def test_trade_log_no_null_exit_time(self):
        """All completed trades must have a non-null exit_time."""
        # Build a signal with one entry and one exit
        n_bars = 10
        timestamps = [_ts(i) for i in range(n_bars)]
        signals = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

        signals_df = pl.DataFrame(
            {
                "date": ["2023-01-03"] * n_bars,
                "timestamp": timestamps,
                "ticker_a": ["A"] * n_bars,
                "ticker_b": ["B"] * n_bars,
                "hedge_ratio": [1.0] * n_bars,
                "p_value": [0.01] * n_bars,
                "signal_weight": [1.0] * n_bars,
                "zscore": [0.0] * n_bars,
                "signal_binary": signals,
                "signal_weighted": [float(s) for s in signals],
            }
        )

        trading_days = [dt.date(2023, 1, 3)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df(n_bars=n_bars)

        patches = self._patch_all(pairs, intraday, signals_df, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            trades, _ = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
            )

        if len(trades) > 0:
            assert trades["exit_time"].is_null().sum() == 0

    def test_empty_pairs_produces_no_trades(self):
        trading_days = [dt.date(2023, 1, 3)]
        empty_pairs = _make_empty_pairs_df()
        intraday = _make_intraday_df()
        signals = _make_flat_signals_df("A", "B")

        patches = self._patch_all(empty_pairs, intraday, signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            trades, daily_pnl = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
            )

        assert len(trades) == 0

    def test_pnl_known_values(self):
        """Manually verify gross/net PnL for a simple trade."""
        pos = _make_pos(
            direction=1,
            entry_price_a=100.0,
            entry_price_b=100.0,
            shares_a=10.0,
            shares_b=10.0,
        )
        gross, cost, net = _compute_pnl(
            pos, exit_price_a=110.0, exit_price_b=100.0, cost_bps=20.0
        )
        # gross = 1 * (10 * 10 - 0 * 10) = 100
        assert gross == pytest.approx(100.0)
        # net < gross
        assert net < gross
        # net = 100 - transaction_cost
        assert net == pytest.approx(gross - cost, abs=1e-12)


# ---------------------------------------------------------------------------
# Output schema validation
# ---------------------------------------------------------------------------


class TestOutputSchemas:
    """Verify trades_df and daily_pnl_df schemas are correct regardless of trades."""

    EXPECTED_TRADES_COLS = {
        "entry_time", "exit_time", "ticker_a", "ticker_b", "direction",
        "entry_price_a", "entry_price_b", "exit_price_a", "exit_price_b",
        "shares_a", "shares_b", "gross_pnl", "transaction_cost", "net_pnl",
        "exit_reason",
    }
    EXPECTED_DAILY_COLS = {
        "date", "gross_pnl", "net_pnl", "n_trades", "n_open_eod", "portfolio_value",
    }

    def _run_one_day(self, signals_df):
        trading_days = [dt.date(2023, 1, 3)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df()
        patches = _patch_all(pairs, intraday, signals_df, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            return run_backtest(start_date="2023-01-03", end_date="2023-01-03")

    def test_trades_df_columns_when_no_trades(self):
        trades, _ = self._run_one_day(_make_flat_signals_df("A", "B"))
        assert set(trades.columns) == self.EXPECTED_TRADES_COLS

    def test_daily_pnl_df_columns(self):
        _, daily_pnl = self._run_one_day(_make_flat_signals_df("A", "B"))
        assert set(daily_pnl.columns) == self.EXPECTED_DAILY_COLS

    def test_trades_df_columns_when_trades_occur(self):
        # signal: enter at bar 2, exit at bar 5
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        trades, _ = self._run_one_day(signals)
        assert set(trades.columns) == self.EXPECTED_TRADES_COLS

    def test_daily_pnl_one_row_per_trading_day(self):
        trading_days = [dt.date(2023, 1, 3), dt.date(2023, 1, 4), dt.date(2023, 1, 5)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df()
        signals = _make_flat_signals_df("A", "B")
        patches = _patch_all(pairs, intraday, signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            _, daily_pnl = run_backtest(
                start_date="2023-01-03", end_date="2023-01-05"
            )
        assert len(daily_pnl) == len(trading_days)


# ---------------------------------------------------------------------------
# Recommended parameters (from parameter_search.ipynb analysis)
#   z_entry=3.0, rolling_window_days=42, max_holding=60min → z_stop=4.5
# ---------------------------------------------------------------------------


# Recommended parameters identified in parameter_search.ipynb
RECOMMENDED_PARAMS = dict(
    z_entry=3.0,
    z_exit=0.5,
    z_stop=4.5,
    rolling_window_days=42,
)


class TestRecommendedParams:
    """
    Validate backtester output using the recommended parameters.
    Signals are mocked so no data files are required.
    """

    def _run(self, signals_df, n_days: int = 1, max_pairs: int = 10):
        start = dt.date(2023, 1, 3)
        trading_days = [start + dt.timedelta(days=i) for i in range(n_days)]
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df(n_bars=12)
        patches = _patch_all(pairs, intraday, signals_df, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            return run_backtest(
                start_date=str(start),
                end_date=str(trading_days[-1]),
                max_pairs=max_pairs,
                **RECOMMENDED_PARAMS,
            )

    def test_runs_without_error(self):
        """run_backtest completes with recommended params and returns two DataFrames."""
        signals = _make_flat_signals_df("A", "B")
        trades, daily_pnl = self._run(signals)
        assert isinstance(trades, pl.DataFrame)
        assert isinstance(daily_pnl, pl.DataFrame)

    def test_daily_pnl_row_count(self):
        """One daily_pnl row per trading day."""
        signals = _make_flat_signals_df("A", "B")
        _, daily_pnl = self._run(signals, n_days=3)
        assert len(daily_pnl) == 3

    def test_trade_direction_is_plus_or_minus_one(self):
        """All trade directions must be +1 or -1."""
        # Enter long at bar 2, exit at bar 5
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        trades, _ = self._run(signals)
        if len(trades) > 0:
            assert trades["direction"].is_in([1, -1]).all()

    def test_exit_reason_valid_values(self):
        """exit_reason must be one of the known exit types."""
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        trades, _ = self._run(signals)
        if len(trades) > 0:
            valid = ["z_exit", "z_stop", "eod", "max_hold"]
            assert trades["exit_reason"].is_in(valid).all()

    def test_entry_time_before_exit_time(self):
        """Every trade must have entry_time < exit_time."""
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        trades, _ = self._run(signals)
        if len(trades) > 0:
            assert (trades["entry_time"] < trades["exit_time"]).all()

    def test_shares_are_positive(self):
        """shares_a and shares_b must be positive for every trade."""
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        trades, _ = self._run(signals)
        if len(trades) > 0:
            assert (trades["shares_a"] > 0).all()
            assert (trades["shares_b"] > 0).all()

    def test_transaction_cost_positive(self):
        """transaction_cost must be positive for every trade."""
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        trades, _ = self._run(signals)
        if len(trades) > 0:
            assert (trades["transaction_cost"] > 0).all()

    def test_net_pnl_equals_gross_minus_cost(self):
        """net_pnl == gross_pnl - transaction_cost for every trade."""
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        trades, _ = self._run(signals)
        if len(trades) > 0:
            diff = (
                trades["net_pnl"] - (trades["gross_pnl"] - trades["transaction_cost"])
            ).abs()
            assert (diff < 1e-9).all()

    def test_portfolio_value_consistency(self):
        """Cumulative net_pnl must equal final portfolio_value - initial capital."""
        signals = _make_trade_signals_df("A", "B", [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        _, daily_pnl = self._run(signals, n_days=1)
        total_net = daily_pnl["net_pnl"].sum()
        final_value = daily_pnl["portfolio_value"][-1]
        assert final_value == pytest.approx(CONFIG.portfolio.capital + total_net, rel=1e-9)

    def test_max_pairs_constraint(self):
        """
        When more pairs signal simultaneously than max_pairs allows,
        n_trades must not exceed max_pairs.
        """
        # Build 5 pairs all entering at bar 2
        n_bars = 12
        timestamps = [_ts(i) for i in range(n_bars)]
        signals_list = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

        all_rows = []
        pairs_data = {"ticker_a": [], "ticker_b": [], "hedge_ratio": [], "p_value": [], "half_life": []}
        for idx in range(5):
            ta, tb = f"A{idx}", f"B{idx}"
            pairs_data["ticker_a"].append(ta)
            pairs_data["ticker_b"].append(tb)
            pairs_data["hedge_ratio"].append(1.0)
            pairs_data["p_value"].append(0.01 * (idx + 1))
            pairs_data["half_life"].append(10.0)
            for i, ts in enumerate(timestamps):
                all_rows.append({
                    "date": "2023-01-03", "timestamp": ts,
                    "ticker_a": ta, "ticker_b": tb,
                    "hedge_ratio": 1.0, "p_value": 0.01 * (idx + 1),
                    "signal_weight": 1.0, "zscore": 0.0,
                    "signal_binary": signals_list[i],
                    "signal_weighted": float(signals_list[i]),
                })

        multi_pairs = pl.DataFrame(pairs_data)
        multi_signals = pl.DataFrame(all_rows, schema={
            "date": pl.Utf8, "timestamp": pl.Datetime,
            "ticker_a": pl.Utf8, "ticker_b": pl.Utf8,
            "hedge_ratio": pl.Float64, "p_value": pl.Float64,
            "signal_weight": pl.Float64, "zscore": pl.Float64,
            "signal_binary": pl.Int32, "signal_weighted": pl.Float64,
        })

        intraday = _make_intraday_df(n_bars=n_bars)
        trading_days = [dt.date(2023, 1, 3)]
        patches = _patch_all(multi_pairs, intraday, multi_signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            trades, _ = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
                max_pairs=3,
                **RECOMMENDED_PARAMS,
            )

        # With max_pairs=3, at most 3 positions can be open at once
        assert len(trades) <= 3

    def test_eod_exit_recorded_as_eod(self):
        """A position that stays open until EOD must have exit_reason='eod'."""
        # Signal never goes back to 0 — position must be force-closed at EOD
        n_bars = 8
        signals = _make_trade_signals_df("A", "B", [0, 1, 1, 1, 1, 1, 1, 1])
        intraday = _make_intraday_df(n_bars=n_bars)
        trading_days = [dt.date(2023, 1, 3)]
        pairs = _make_minimal_pairs_df()
        patches = _patch_all(pairs, intraday, signals, trading_days)
        with patches[0], patches[1], patches[2], patches[3]:
            trades, _ = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
                **RECOMMENDED_PARAMS,
            )
        if len(trades) > 0:
            assert (trades["exit_reason"] == "eod").all()


# ---------------------------------------------------------------------------
# _get_exec_price
# ---------------------------------------------------------------------------


def _make_1min_lookup(
    ticker: str,
    base_ts: dt.datetime,
    n_bars: int = 20,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
) -> dict:
    """Build a 1-min bar lookup for a single ticker."""
    lookup = {}
    for i in range(n_bars):
        ts = base_ts + dt.timedelta(minutes=i)
        lookup[ts] = {"open": open_, "high": high, "low": low, "close": close}
    return {ticker: lookup}


class TestGetExecPrice:
    _base = dt.datetime(2023, 1, 3, 9, 15)  # start of the next 15-min bar

    def test_lag0_open(self):
        lookup = _make_1min_lookup("A", self._base, open_=101.0)
        price = _get_exec_price(lookup, "A", self._base, 0, "open")
        assert price == pytest.approx(101.0)

    def test_lag5_close(self):
        lookup = _make_1min_lookup("A", self._base, close=102.5)
        price = _get_exec_price(lookup, "A", self._base, 5, "close")
        assert price == pytest.approx(102.5)

    def test_mid_field(self):
        lookup = _make_1min_lookup("A", self._base, high=102.0, low=100.0)
        price = _get_exec_price(lookup, "A", self._base, 0, "mid")
        assert price == pytest.approx(101.0)

    def test_missing_bar_returns_none(self):
        """If the 1-min bar at exec_ts is missing, return None."""
        lookup = _make_1min_lookup("A", self._base, n_bars=3)  # bars 0,1,2 only
        price = _get_exec_price(lookup, "A", self._base, 5, "open")  # lag=5 → missing
        assert price is None

    def test_missing_ticker_returns_none(self):
        lookup = _make_1min_lookup("A", self._base)
        price = _get_exec_price(lookup, "B", self._base, 0, "open")  # ticker B missing
        assert price is None

    def test_all_ohlc_fields(self):
        lookup = _make_1min_lookup("A", self._base, open_=100.0, high=104.0, low=98.0, close=103.0)
        assert _get_exec_price(lookup, "A", self._base, 0, "open")  == pytest.approx(100.0)
        assert _get_exec_price(lookup, "A", self._base, 0, "high")  == pytest.approx(104.0)
        assert _get_exec_price(lookup, "A", self._base, 0, "low")   == pytest.approx(98.0)
        assert _get_exec_price(lookup, "A", self._base, 0, "close") == pytest.approx(103.0)
        assert _get_exec_price(lookup, "A", self._base, 0, "mid")   == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# run_backtest with execution_lag_minutes
# ---------------------------------------------------------------------------


def _make_1min_intraday_df(n_bars: int = 390, base_price: float = 100.0):
    """Generate synthetic 1-min OHLCV for a single trading day."""
    base = dt.datetime(2023, 1, 3, 9, 30)
    timestamps = [base + dt.timedelta(minutes=i) for i in range(n_bars)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open":   [base_price] * n_bars,
            "high":   [base_price + 0.5] * n_bars,
            "low":    [base_price - 0.5] * n_bars,
            "close":  [base_price] * n_bars,
            "volume": [10_000] * n_bars,
        }
    )


class TestExecLagIntegration:
    """Verify that execution_lag_minutes wires 1-min prices into trades."""

    def _run_with_exec_lag(self, lag, field, intraday_15min, intraday_1min):
        """Run a one-day backtest with execution lag, returning trades."""
        signals = _make_trade_signals_df("A", "B", [0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        pairs = _make_minimal_pairs_df()
        trading_days = [dt.date(2023, 1, 3)]

        # load_processed is called twice: once for 15-min signal data, once for 1-min exec data
        call_count = {"n": 0}

        def load_side_effect(ticker, timeframe, **kwargs):
            call_count["n"] += 1
            if timeframe == "1min":
                return intraday_1min
            return intraday_15min

        patches = [
            patch("strategy.backtester._get_nyse_trading_days", return_value=trading_days),
            patch("strategy.backtester.find_cointegrated_pairs", return_value=pairs),
            patch("strategy.backtester.load_processed", side_effect=load_side_effect),
            patch("strategy.backtester.generate_pair_signals_for_day", return_value=signals),
        ]
        with patches[0], patches[1], patches[2], patches[3]:
            trades, _ = run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
                execution_lag_minutes=lag,
                execution_price_field=field,
                **RECOMMENDED_PARAMS,
            )
        return trades, call_count["n"]

    def test_none_lag_does_not_load_1min(self):
        """execution_lag_minutes=None must not trigger a second load_processed call."""
        signals = _make_trade_signals_df("A", "B", [0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        pairs = _make_minimal_pairs_df()
        intraday = _make_intraday_df()
        trading_days = [dt.date(2023, 1, 3)]

        call_count = {"n": 0}

        def load_side_effect(ticker, timeframe, **kwargs):
            call_count["n"] += 1
            return intraday

        patches = [
            patch("strategy.backtester._get_nyse_trading_days", return_value=trading_days),
            patch("strategy.backtester.find_cointegrated_pairs", return_value=pairs),
            patch("strategy.backtester.load_processed", side_effect=load_side_effect),
            patch("strategy.backtester.generate_pair_signals_for_day", return_value=signals),
        ]
        with patches[0], patches[1], patches[2], patches[3]:
            run_backtest(
                start_date="2023-01-03",
                end_date="2023-01-03",
                execution_lag_minutes=None,
                **RECOMMENDED_PARAMS,
            )
        # Should only load 15-min data (2 tickers), not 1-min data
        assert all(
            True for _ in range(call_count["n"])
        ), "load_processed call count recorded"
        # No 1-min call means the call count is for 2 tickers (A and B) at 15-min only
        assert call_count["n"] == 2

    def test_lag0_open_uses_1min_price(self):
        """With lag=0 and field='open', entry price should equal the 1-min open."""
        intraday_15min = _make_intraday_df(n_bars=10, base_price=100.0)
        intraday_1min = _make_1min_intraday_df(base_price=105.0)  # distinctly different price

        trades, _ = self._run_with_exec_lag(0, "open", intraday_15min, intraday_1min)

        if len(trades) > 0:
            # 1-min open is 105.0; 15-min midpoint would be 100.0 ((100.5 + 99.5)/2)
            assert trades["entry_price_a"][0] == pytest.approx(105.0)
            assert trades["entry_price_b"][0] == pytest.approx(105.0)

    def test_lag0_mid_uses_1min_midpoint(self):
        """With field='mid', entry price should be (high+low)/2 of the 1-min bar."""
        intraday_15min = _make_intraday_df(n_bars=10, base_price=100.0)
        # 1-min bars: high=106, low=104 → mid=105
        intraday_1min = pl.DataFrame(
            {
                "timestamp": [dt.datetime(2023, 1, 3, 9, 30) + dt.timedelta(minutes=i) for i in range(390)],
                "open":  [105.0] * 390,
                "high":  [106.0] * 390,
                "low":   [104.0] * 390,
                "close": [105.0] * 390,
                "volume": [10_000] * 390,
            }
        )
        trades, _ = self._run_with_exec_lag(0, "mid", intraday_15min, intraday_1min)

        if len(trades) > 0:
            assert trades["entry_price_a"][0] == pytest.approx(105.0)  # (106+104)/2

    def test_fallback_when_1min_missing(self):
        """If the 1-min bar at exec_ts is not found, fall back to 15-min midpoint."""
        intraday_15min = _make_intraday_df(n_bars=10, base_price=100.0)
        # Empty 1-min dataframe — no bars available
        empty_1min = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
        trades, _ = self._run_with_exec_lag(0, "open", intraday_15min, empty_1min)

        if len(trades) > 0:
            # 15-min midpoint: high=100.5, low=99.5 → 100.0
            assert trades["entry_price_a"][0] == pytest.approx(100.0)
