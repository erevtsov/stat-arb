"""Tests for find_cointegrated_pairs_rolling in analysis/cointegration.py."""

import datetime as dt
from unittest.mock import patch, MagicMock

import polars as pl
import pytest

from analysis.cointegration import find_cointegrated_pairs_rolling


# Empty DataFrame to return from the mocked find_cointegrated_pairs
_EMPTY_PAIRS = pl.DataFrame(
    schema={
        "ticker_a":    pl.Utf8,
        "ticker_b":    pl.Utf8,
        "hedge_ratio": pl.Float64,
        "p_value":     pl.Float64,
        "half_life":   pl.Float64,
    }
)


@patch("analysis.cointegration.find_cointegrated_pairs", return_value=_EMPTY_PAIRS)
class TestRollingWindowDays:
    """Tests for find_cointegrated_pairs_rolling with window_days / step_days."""

    def test_key_format(self, mock_find):
        """Keys must be 'YYYY-MM-DD_YYYY-MM-DD'."""
        result = find_cointegrated_pairs_rolling(
            window_days=84,
            step_days=7,
            start_date="2023-01-01",
            end_date="2023-03-01",
        )
        for key in result.keys():
            parts = key.split("_")
            assert len(parts) == 2, f"Bad key format: {key}"
            start_str, end_str = parts
            # Both must be parseable as dates
            dt.date.fromisoformat(start_str)
            dt.date.fromisoformat(end_str)

    def test_window_span(self, mock_find):
        """Each window's end - start must equal window_days calendar days."""
        result = find_cointegrated_pairs_rolling(
            window_days=84,
            step_days=7,
            start_date="2023-01-01",
            end_date="2023-06-01",
        )
        for key in result.keys():
            start_str, end_str = key.split("_")
            start = dt.date.fromisoformat(start_str)
            end   = dt.date.fromisoformat(end_str)
            span  = (end - start).days
            assert span == 84, f"Window span {span} != 84 for key {key}"

    def test_step_days(self, mock_find):
        """Consecutive window start dates must differ by exactly step_days."""
        result = find_cointegrated_pairs_rolling(
            window_days=42,
            step_days=7,
            start_date="2023-01-01",
            end_date="2023-06-01",
        )
        starts = sorted(
            dt.date.fromisoformat(k.split("_")[0]) for k in result.keys()
        )
        if len(starts) >= 2:
            for a, b in zip(starts, starts[1:]):
                diff = (b - a).days
                assert diff == 7, f"Step {diff} != 7 between {a} and {b}"

    def test_correct_number_of_windows(self, mock_find):
        """Number of windows should match manual calculation."""
        window_days = 42
        step_days = 7
        start_date = "2023-01-01"
        end_date = "2023-03-31"

        result = find_cointegrated_pairs_rolling(
            window_days=window_days,
            step_days=step_days,
            start_date=start_date,
            end_date=end_date,
        )

        # Count expected windows manually
        start = dt.date.fromisoformat(start_date)
        end   = dt.date.fromisoformat(end_date)
        expected = 0
        cursor = start
        while cursor + dt.timedelta(days=window_days) <= end:
            expected += 1
            cursor += dt.timedelta(days=step_days)

        assert len(result) == expected, (
            f"Expected {expected} windows, got {len(result)}"
        )

    def test_no_relativedelta_dependency(self, mock_find):
        """Ensure no import of relativedelta (we switched to timedelta)."""
        import analysis.cointegration as coint_module
        import inspect
        src = inspect.getsource(coint_module)
        assert "relativedelta" not in src, (
            "find_cointegrated_pairs_rolling should not use relativedelta"
        )
