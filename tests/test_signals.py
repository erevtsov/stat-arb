"""Tests for analysis/signals.py."""

import datetime as dt

import polars as pl
import pytest

from analysis.signals import (
    compute_spread,
    compute_zscore,
    generate_signals,
    compute_pvalue_weights,
    generate_pair_signals_for_day,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zscore_series(values: list) -> pl.Series:
    return pl.Series("zscore", values, dtype=pl.Float64)


def _make_pairs_df(p_values: list[float]) -> pl.DataFrame:
    """Create a minimal pairs DataFrame sorted by p_value ascending."""
    n = len(p_values)
    return pl.DataFrame(
        {
            "ticker_a":    [f"A{i}" for i in range(n)],
            "ticker_b":    [f"B{i}" for i in range(n)],
            "hedge_ratio": [1.0] * n,
            "p_value":     sorted(p_values),
        }
    )


def _make_price_df(
    timestamps: list,
    closes: list[float],
) -> pl.DataFrame:
    """Create a minimal OHLCV DataFrame."""
    n = len(closes)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open":      closes,
            "high":      [c * 1.001 for c in closes],
            "low":       [c * 0.999 for c in closes],
            "close":     closes,
            "volume":    [1_000_000] * n,
        }
    )


def _make_timestamps(n: int) -> list:
    """Generate n consecutive 15-min timestamps starting 2023-01-03 09:30."""
    base = dt.datetime(2023, 1, 3, 9, 30, 0)
    return [base + dt.timedelta(minutes=15 * i) for i in range(n)]


# ---------------------------------------------------------------------------
# compute_pvalue_weights
# ---------------------------------------------------------------------------


class TestComputePvalueWeights:
    def test_sums_to_one(self):
        pairs_df = _make_pairs_df([0.01, 0.02, 0.03])
        result = compute_pvalue_weights(pairs_df)
        total = sum(result["signal_weight"].to_list())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_sums_to_one_five_pairs(self):
        pairs_df = _make_pairs_df([0.01, 0.02, 0.03, 0.04, 0.05])
        result = compute_pvalue_weights(pairs_df)
        total = sum(result["signal_weight"].to_list())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_rank_order(self):
        """Pair with lowest p_value (rank 1) must have highest weight."""
        pairs_df = _make_pairs_df([0.01, 0.03, 0.05])
        result = compute_pvalue_weights(pairs_df)
        weights = result["signal_weight"].to_list()
        # Weights should be strictly decreasing (pairs sorted ascending by p_value)
        assert weights[0] > weights[1] > weights[2]

    def test_single_pair(self):
        pairs_df = _make_pairs_df([0.04])
        result = compute_pvalue_weights(pairs_df)
        assert result["signal_weight"][0] == pytest.approx(1.0)

    def test_weights_monotone_decreasing(self):
        pairs_df = _make_pairs_df([0.001, 0.01, 0.02, 0.03, 0.04])
        result = compute_pvalue_weights(pairs_df)
        weights = result["signal_weight"].to_list()
        for a, b in zip(weights, weights[1:]):
            assert a > b

    def test_empty_df(self):
        pairs_df = _make_pairs_df([])
        # Should not raise; signal_weight column should exist with null values
        result = compute_pvalue_weights(pairs_df)
        assert "signal_weight" in result.columns


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    def test_long_entry(self):
        zvals = [0.0] * 5 + [-3.0] + [0.0] * 4
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        assert sig[5] == 1

    def test_short_entry(self):
        zvals = [0.0] * 5 + [3.0] + [0.0] * 4
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        assert sig[5] == -1

    def test_exit_mean_reversion_long(self):
        # Enter long at bar 5, then spread reverts to 0 at bar 8
        zvals = [0.0] * 5 + [-3.0, -2.0, -1.0, 0.0, 0.0]
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        # Bar 5: enter long
        assert sig[5] == 1
        # Bar 8: z=0 >= -z_exit(0) → exit
        assert sig[8] == 0

    def test_exit_mean_reversion_short(self):
        zvals = [0.0] * 5 + [3.0, 2.0, 1.0, 0.0, 0.0]
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        assert sig[5] == -1
        assert sig[8] == 0

    def test_stoploss(self):
        # Enter long, then |z| > z_stop
        zvals = [0.0] * 3 + [-3.0, -4.5]
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        assert sig[3] == 1   # entered
        assert sig[4] == 0   # stopped out

    def test_time_stop(self):
        # Enter long at bar 2, hold for max_holding_bars=3
        zvals = [0.0, 0.0, -3.0, -2.8, -2.6, -2.7, 0.0]
        sig = generate_signals(
            _make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0,
            max_holding_bars=3,
        )
        assert sig[2] == 1   # entered at bar 2
        assert sig[5] == 0   # exited at bar 5 (held bars 3,4,5 = 3 bars)

    def test_no_double_entry(self):
        """Once in a position, further crossings should not re-enter."""
        zvals = [-3.0, -3.0, -3.0, -3.0, -3.0]
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        # Should stay +1 throughout (no re-entry toggle)
        assert list(sig) == [1, 1, 1, 1, 1]

    def test_null_values_reset_position(self):
        zvals = [None, None, -3.0, None, -3.0]
        sig = generate_signals(_make_zscore_series(zvals), z_entry=2.5, z_exit=0.0, z_stop=4.0)
        assert sig[0] == 0
        assert sig[3] == 0   # null resets


# ---------------------------------------------------------------------------
# compute_zscore
# ---------------------------------------------------------------------------


class TestComputeZscore:
    def test_leading_nulls(self):
        spread = pl.Series("spread", list(range(1, 21)), dtype=pl.Float64)
        window = 5
        zscore = compute_zscore(spread, window=window)
        nulls = zscore.is_null().to_list()
        # First (window-1) = 4 values should be null
        assert all(nulls[:window - 1])
        assert not nulls[window - 1]

    def test_requires_window(self):
        spread = pl.Series("spread", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="window is required"):
            compute_zscore(spread)


# ---------------------------------------------------------------------------
# generate_pair_signals_for_day
# ---------------------------------------------------------------------------


class TestGeneratePairSignalsForDay:
    def setup_method(self):
        self.n_bars = 30
        self.timestamps = _make_timestamps(self.n_bars)
        self.date = dt.date(2023, 1, 3)

        # Two tickers: A0 and B0 with a spread that dips below -2.5
        closes_a = [100.0] * 5 + [97.0] * 5 + [100.0] * 20
        closes_b = [100.0] * self.n_bars

        self.prices = {
            "A0": _make_price_df(self.timestamps, closes_a),
            "B0": _make_price_df(self.timestamps, closes_b),
        }

        self.pairs_df = compute_pvalue_weights(
            pl.DataFrame(
                {
                    "ticker_a":    ["A0"],
                    "ticker_b":    ["B0"],
                    "hedge_ratio": [1.0],
                    "p_value":     [0.01],
                }
            )
        )

    def test_output_schema(self):
        result = generate_pair_signals_for_day(
            pairs_df=self.pairs_df,
            date=self.date,
            intraday_prices=self.prices,
            zscore_window=10,
        )
        expected_cols = {
            "date", "timestamp", "ticker_a", "ticker_b",
            "hedge_ratio", "p_value", "signal_weight",
            "zscore", "signal_binary", "signal_weighted",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_signal_weighted_equals_binary_times_weight(self):
        result = generate_pair_signals_for_day(
            pairs_df=self.pairs_df,
            date=self.date,
            intraday_prices=self.prices,
            zscore_window=10,
        )
        for row in result.iter_rows(named=True):
            expected = float(row["signal_binary"]) * row["signal_weight"]
            assert row["signal_weighted"] == pytest.approx(expected, abs=1e-12)

    def test_n_rows_equals_n_bars_times_n_pairs(self):
        result = generate_pair_signals_for_day(
            pairs_df=self.pairs_df,
            date=self.date,
            intraday_prices=self.prices,
            zscore_window=10,
        )
        # 1 pair × n_bars rows
        assert len(result) == self.n_bars

    def test_missing_ticker_skipped(self):
        """Pairs with missing tickers should be silently skipped."""
        pairs_with_missing = compute_pvalue_weights(
            pl.DataFrame(
                {
                    "ticker_a":    ["MISSING"],
                    "ticker_b":    ["B0"],
                    "hedge_ratio": [1.0],
                    "p_value":     [0.01],
                }
            )
        )
        result = generate_pair_signals_for_day(
            pairs_df=pairs_with_missing,
            date=self.date,
            intraday_prices=self.prices,
            zscore_window=10,
        )
        assert len(result) == 0
