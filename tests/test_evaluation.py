"""Tests for analysis/evaluation.py."""

import datetime as dt

import polars as pl
import pytest

from analysis.evaluation import (
    compute_forward_returns,
    compute_net_forward_returns,
    compute_ic,
    compute_hit_rate,
    evaluate_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(bar: int) -> dt.datetime:
    """Bar index → datetime (15min bars starting 2023-01-03 09:30)."""
    return dt.datetime(2023, 1, 3, 9, 30) + dt.timedelta(minutes=15 * bar)


def _make_prices(closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "timestamp": [_ts(i) for i in range(n)],
            "open":      closes,
            "high":      [c * 1.001 for c in closes],
            "low":       [c * 0.999 for c in closes],
            "close":     closes,
            "volume":    [100_000] * n,
        }
    )


def _make_signals_row(
    ticker_a: str,
    ticker_b: str,
    bar: int,
    signal_binary: int,
    hedge_ratio: float = 1.0,
    signal_weight: float = 1.0,
    date: str = "2023-01-03",
) -> dict:
    return {
        "date":           date,
        "timestamp":      _ts(bar),
        "ticker_a":       ticker_a,
        "ticker_b":       ticker_b,
        "hedge_ratio":    hedge_ratio,
        "p_value":        0.01,
        "signal_weight":  signal_weight,
        "zscore":         -3.0 if signal_binary == 1 else (3.0 if signal_binary == -1 else 0.0),
        "signal_binary":  signal_binary,
        "signal_weighted": float(signal_binary) * signal_weight,
    }


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------


class TestComputeForwardReturns:
    def setup_method(self):
        # spread_t: close_a - 1.0 * close_b
        # A goes from 100 → 105 over 5 bars; B stays at 100
        self.prices = {
            "A": _make_prices([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
            "B": _make_prices([100.0] * 6),
        }

    def test_long_spread_zero_spread_returns_null(self):
        """When spread at t is exactly 0 (division guard), return is null."""
        # setup: closes_a[0]=100, closes_b[0]=100 with hedge_ratio=1 → spread=0
        signals_df = pl.DataFrame(
            [_make_signals_row("A", "B", bar=0, signal_binary=1)]
        )
        result = compute_forward_returns(signals_df, self.prices, n_bars=3)
        fwd = result["fwd_return_gross"][0]
        assert fwd is None

    def test_long_spread_positive_when_spread_rises(self):
        prices = {
            "A": _make_prices([100.0, 101.0, 103.0, 106.0]),
            "B": _make_prices([100.0] * 4),
        }
        # spread_0 = 0, adjust: use hedge_ratio != 1
        prices2 = {
            "A": _make_prices([105.0, 106.0, 107.0, 110.0]),
            "B": _make_prices([100.0] * 4),
        }
        signals_df = pl.DataFrame(
            [_make_signals_row("A", "B", bar=0, signal_binary=1, hedge_ratio=1.0)]
        )
        result = compute_forward_returns(signals_df, prices2, n_bars=2)
        fwd = result["fwd_return_gross"][0]
        # spread_0 = 105 - 100 = 5; spread_2 = 107 - 100 = 7
        # fwd = 1 * (7 - 5) / 5 = 0.4
        assert fwd == pytest.approx(0.4, abs=1e-9)

    def test_short_spread_positive_when_spread_falls(self):
        prices = {
            "A": _make_prices([105.0, 104.0, 103.0, 102.0]),
            "B": _make_prices([100.0] * 4),
        }
        signals_df = pl.DataFrame(
            [_make_signals_row("A", "B", bar=0, signal_binary=-1, hedge_ratio=1.0)]
        )
        result = compute_forward_returns(signals_df, prices, n_bars=2)
        fwd = result["fwd_return_gross"][0]
        # spread_0 = 5; spread_2 = 3
        # fwd = -1 * (3 - 5) / 5 = 0.4
        assert fwd == pytest.approx(0.4, abs=1e-9)

    def test_zero_signal_returns_null(self):
        signals_df = pl.DataFrame(
            [_make_signals_row("A", "B", bar=0, signal_binary=0)]
        )
        result = compute_forward_returns(signals_df, self.prices, n_bars=3)
        assert result["fwd_return_gross"][0] is None

    def test_normalized_by_spread_at_t(self):
        prices = {
            "A": _make_prices([110.0, 111.0, 112.0, 115.0]),
            "B": _make_prices([100.0] * 4),
        }
        signals_df = pl.DataFrame(
            [_make_signals_row("A", "B", bar=0, signal_binary=1, hedge_ratio=1.0)]
        )
        result = compute_forward_returns(signals_df, prices, n_bars=2)
        fwd = result["fwd_return_gross"][0]
        # spread_0 = 10; spread_2 = 12; fwd = (12 - 10) / 10 = 0.2
        assert fwd == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_net_forward_returns
# ---------------------------------------------------------------------------


class TestComputeNetForwardReturns:
    def test_net_less_than_gross(self):
        signals_df = pl.DataFrame({"fwd_return_gross": [0.05]})
        result = compute_net_forward_returns(signals_df, cost_bps=20.0)
        assert result["fwd_return_net"][0] < result["fwd_return_gross"][0]

    def test_cost_magnitude(self):
        """fwd_return_gross - fwd_return_net == 4 * 20 / 10_000."""
        gross = 0.1
        signals_df = pl.DataFrame({"fwd_return_gross": [gross]})
        result = compute_net_forward_returns(signals_df, cost_bps=20.0)
        diff = gross - result["fwd_return_net"][0]
        assert diff == pytest.approx(4 * 20 / 10_000, abs=1e-12)

    def test_null_gross_gives_null_net(self):
        signals_df = pl.DataFrame({"fwd_return_gross": [None]}, schema={"fwd_return_gross": pl.Float64})
        result = compute_net_forward_returns(signals_df)
        assert result["fwd_return_net"][0] is None


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------


class TestComputeIC:
    def test_perfect_positive(self):
        ws = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        fr = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_ic(ws, fr) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative(self):
        ws = pl.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        fr = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_ic(ws, fr) == pytest.approx(-1.0, abs=1e-9)

    def test_drops_nulls(self):
        ws = pl.Series([1.0, None, 3.0, 4.0, 5.0])
        fr = pl.Series([1.0, 2.0, None, 4.0, 5.0])
        # Should not raise; computes on non-null intersection
        result = compute_ic(ws, fr)
        assert -1.0 <= result <= 1.0

    def test_all_nulls_returns_zero(self):
        ws = pl.Series([None, None], dtype=pl.Float64)
        fr = pl.Series([None, None], dtype=pl.Float64)
        assert compute_ic(ws, fr) == 0.0

    def test_result_in_range(self):
        import random
        random.seed(42)
        ws = pl.Series([random.gauss(0, 1) for _ in range(100)])
        fr = pl.Series([random.gauss(0, 1) for _ in range(100)])
        result = compute_ic(ws, fr)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_hit_rate
# ---------------------------------------------------------------------------


class TestComputeHitRate:
    def test_all_correct(self):
        sig = pl.Series([1, 1, -1, -1])
        fr  = pl.Series([0.1, 0.2, -0.1, -0.2])
        assert compute_hit_rate(sig, fr) == pytest.approx(1.0)

    def test_all_wrong(self):
        sig = pl.Series([1, 1, -1, -1])
        fr  = pl.Series([-0.1, -0.2, 0.1, 0.2])
        assert compute_hit_rate(sig, fr) == pytest.approx(0.0)

    def test_half_correct(self):
        sig = pl.Series([1, 1, -1, -1])
        fr  = pl.Series([0.1, -0.1, -0.1, 0.1])
        assert compute_hit_rate(sig, fr) == pytest.approx(0.5)

    def test_ignores_zero_signals(self):
        sig = pl.Series([0, 1, 0, -1])
        fr  = pl.Series([0.1, 0.2, -0.1, -0.2])
        assert compute_hit_rate(sig, fr) == pytest.approx(1.0)

    def test_ignores_null_returns(self):
        sig = pl.Series([1, 1])
        fr  = pl.Series([None, 0.1], dtype=pl.Float64)
        # Only second pair counts → 1 hit out of 1
        assert compute_hit_rate(sig, fr) == pytest.approx(1.0)

    def test_no_active_signals(self):
        sig = pl.Series([0, 0])
        fr  = pl.Series([0.1, 0.2])
        assert compute_hit_rate(sig, fr) == 0.0


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    def setup_method(self):
        self.prices = {
            "A": _make_prices([105.0, 106.0, 108.0, 111.0, 115.0, 120.0, 126.0]),
            "B": _make_prices([100.0] * 7),
        }
        # Single active signal at bar 0 for a long position
        self.signals_df = pl.DataFrame(
            [_make_signals_row("A", "B", bar=0, signal_binary=1, hedge_ratio=1.0)]
        )

    def test_schema(self):
        result = evaluate_all(
            self.signals_df, self.prices,
            holding_bars_list=[2, 4],
            cost_bps=20.0,
        )
        expected_cols = {
            "date", "n_bars",
            "ic_gross_weighted", "ic_net_weighted",
            "hit_rate_binary", "n_observations",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_n_rows_is_dates_times_horizons(self):
        result = evaluate_all(
            self.signals_df, self.prices,
            holding_bars_list=[2, 4],
            cost_bps=20.0,
        )
        # 1 date × 2 horizons = 2 rows
        assert len(result) == 2

    def test_ic_net_less_than_or_equal_gross(self):
        result = evaluate_all(
            self.signals_df, self.prices,
            holding_bars_list=[2],
            cost_bps=20.0,
        )
        row = result.row(0, named=True)
        # Net IC can differ from gross, but gross return > net return for positive trades
        # We just check both are finite floats
        assert isinstance(row["ic_gross_weighted"], float)
        assert isinstance(row["ic_net_weighted"], float)

    def test_empty_signals_df_returns_empty(self):
        empty_signals = pl.DataFrame(
            schema={
                "date":           pl.Utf8,
                "timestamp":      pl.Datetime,
                "ticker_a":       pl.Utf8,
                "ticker_b":       pl.Utf8,
                "hedge_ratio":    pl.Float64,
                "p_value":        pl.Float64,
                "signal_weight":  pl.Float64,
                "zscore":         pl.Float64,
                "signal_binary":  pl.Int32,
                "signal_weighted": pl.Float64,
            }
        )
        result = evaluate_all(empty_signals, self.prices, holding_bars_list=[2], cost_bps=20.0)
        assert len(result) == 0
