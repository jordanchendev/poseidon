"""Tests for new condition evaluators: bollinger_width_percentile, indicator_comparison,
and resolve_column_name extensions."""

import numpy as np
import pandas as pd
import pytest

from poseidon.strategies.dsl.conditions import resolve_column_name
from poseidon.strategies.dsl.executor import evaluate_condition


class TestBollingerWidthPercentile:
    """Tests for the bollinger_width_percentile condition evaluator."""

    @pytest.fixture
    def bb_features(self):
        """DataFrame with known Bollinger Band values for percentile testing."""
        n = 200
        # Create widths that go from wide (10.0) to narrow (1.0) to moderate
        # so row_idx can target specific percentile positions.
        bb_upper = np.linspace(110, 101, n)
        bb_lower = np.linspace(90, 99.5, n)
        # widths go from 20 -> 1.5 roughly
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 100, n),
                "bb_upper_20": bb_upper,
                "bb_lower_20": bb_lower,
            }
        )
        return df

    def test_squeeze_at_10th_percentile(self, bb_features):
        """Width at ~10th percentile with threshold 0.2 -> True (squeeze detected)."""
        # The last rows have the narrowest width. Use last row.
        row_idx = len(bb_features) - 1
        condition = {
            "type": "bollinger_width_percentile",
            "params": {"period": 20, "lookback": 168},
            "threshold": 0.2,
        }
        result = evaluate_condition(condition, bb_features, row_idx)
        assert result is True

    def test_no_squeeze_at_50th_percentile(self):
        """Width at ~50th percentile with threshold 0.2 -> False (no squeeze)."""
        n = 100
        # Constant width except the current row is at median
        widths = np.linspace(1.0, 10.0, n)
        mid = n // 2
        df = pd.DataFrame(
            {
                "close": np.ones(n) * 100,
                "bb_upper_20": 100 + widths / 2,
                "bb_lower_20": 100 - widths / 2,
            }
        )
        condition = {
            "type": "bollinger_width_percentile",
            "params": {"period": 20, "lookback": 100},
            "threshold": 0.2,
        }
        # At the middle row, percentile should be ~0.5
        result = evaluate_condition(condition, df, row_idx=mid)
        assert result is False

    def test_row_idx_less_than_lookback_uses_available(self):
        """row_idx < lookback -> uses available data, no error."""
        n = 10
        # Narrow width at the last position
        widths = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        df = pd.DataFrame(
            {
                "close": np.ones(n) * 100,
                "bb_upper_20": 100 + widths / 2,
                "bb_lower_20": 100 - widths / 2,
            }
        )
        condition = {
            "type": "bollinger_width_percentile",
            "params": {"period": 20, "lookback": 168},  # lookback > available rows
            "threshold": 0.2,
        }
        # Should not raise, should use all 10 rows
        result = evaluate_condition(condition, df, row_idx=9)
        # Width at row 9 is 1.0 (narrowest), should be in squeeze
        assert result is True

    def test_no_look_ahead(self):
        """bollinger_width_percentile does NOT access rows beyond row_idx."""
        n = 20
        # First 10 rows: wide width (10.0), rows 10-19: very narrow (0.5)
        widths = np.array([10.0] * 10 + [0.5] * 10)
        df = pd.DataFrame(
            {
                "close": np.ones(n) * 100,
                "bb_upper_20": 100 + widths / 2,
                "bb_lower_20": 100 - widths / 2,
            }
        )
        condition = {
            "type": "bollinger_width_percentile",
            "params": {"period": 20, "lookback": 20},
            "threshold": 0.2,
        }
        # At row_idx=9 (last wide row), all 10 visible rows have width=10
        # So percentile of 10 among [10,10,...,10] = 0/10 = 0.0 (none are strictly less)
        # 0.0 < 0.2 is True, BUT logically all widths are equal so it's a boundary case.
        # The key assertion: result should NOT be affected by the narrow rows after idx 9.
        # If look-ahead were present, the current width (10) would be at ~50th percentile.
        result_at_9 = evaluate_condition(condition, df, row_idx=9)
        # All widths up to idx 9 are 10.0 (equal), so (widths < current_width).sum() = 0
        # percentile = 0/10 = 0.0, which IS < 0.2 -> True
        assert result_at_9 is True


class TestIndicatorComparison:
    """Tests for the indicator_comparison condition evaluator."""

    @pytest.fixture
    def comparison_df(self):
        return pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "ema_7": [105.0, 100.0, 95.0],
                "ema_26": [100.0, 100.0, 100.0],
            }
        )

    def test_ema7_above_ema26(self, comparison_df):
        """ema_7=105 > ema_26=100, direction='above' -> True."""
        condition = {
            "type": "indicator_comparison",
            "indicator_a": "ema",
            "indicator_b": "ema",
            "params": {"period_a": 7, "period_b": 26},
            "direction": "above",
        }
        result = evaluate_condition(condition, comparison_df, row_idx=0)
        assert result is True

    def test_ema7_below_ema26(self, comparison_df):
        """ema_7=95 < ema_26=100, direction='above' -> False."""
        condition = {
            "type": "indicator_comparison",
            "indicator_a": "ema",
            "indicator_b": "ema",
            "params": {"period_a": 7, "period_b": 26},
            "direction": "above",
        }
        result = evaluate_condition(condition, comparison_df, row_idx=2)
        assert result is False


class TestResolveColumnName:
    """Tests for resolve_column_name extensions."""

    def test_macd_histogram(self):
        assert resolve_column_name("macd_histogram", {}) == "macd_histogram"

    def test_macd_signal(self):
        assert resolve_column_name("macd_signal", {}) == "macd_signal"

    def test_cum_return_period_6(self):
        assert resolve_column_name("cum_return", {"period": 6}) == "cum_return_6d"

    def test_cum_return_period_12(self):
        assert resolve_column_name("cum_return", {"period": 12}) == "cum_return_12d"
