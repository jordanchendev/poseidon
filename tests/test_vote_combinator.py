"""Tests for DSL vote combinator in executor.py."""

import pandas as pd
import pytest

from poseidon.strategies.dsl.executor import evaluate_condition


@pytest.fixture
def features_df():
    """Synthetic feature DataFrame with known indicator values."""
    return pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "rsi_14": [70.0, 30.0, 55.0],
            "sma_20": [99.0, 100.0, 101.0],
            "ema_7": [101.0, 100.0, 103.0],
            "ema_26": [100.0, 100.5, 102.0],
            "return_1d": [0.01, -0.02, 0.015],
        }
    )


class TestVoteCombinator:
    """Tests for the vote combinator logic in evaluate_condition."""

    def test_3_of_5_true_with_min_votes_3_returns_true(self, features_df):
        """3 of 5 conditions true with min_votes=3 -> returns True."""
        # Conditions at row_idx=2: rsi_14=55 > 50 (True), rsi_14=55 > 60 (False),
        # rsi_14=55 > 40 (True), rsi_14=55 > 80 (False), rsi_14=55 > 45 (True)
        condition = {
            "vote": {
                "conditions": [
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 60},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 40},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 80},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 45},
                ],
                "min_votes": 3,
            }
        }
        assert evaluate_condition(condition, features_df, row_idx=2) is True

    def test_2_of_5_true_with_min_votes_3_returns_false(self, features_df):
        """2 of 5 conditions true with min_votes=3 -> returns False."""
        # At row_idx=2: rsi_14=55 > 50 (True), rsi_14=55 > 60 (False),
        # rsi_14=55 > 54 (True), rsi_14=55 > 80 (False), rsi_14=55 > 70 (False)
        condition = {
            "vote": {
                "conditions": [
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 60},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 54},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 80},
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
                ],
                "min_votes": 3,
            }
        }
        assert evaluate_condition(condition, features_df, row_idx=2) is False

    def test_empty_conditions_returns_false(self, features_df):
        """Empty conditions list -> returns False."""
        condition = {
            "vote": {
                "conditions": [],
                "min_votes": 1,
            }
        }
        assert evaluate_condition(condition, features_df, row_idx=0) is False

    def test_vote_nested_inside_all_combinator(self, features_df):
        """Vote combinator nested inside an 'all' combinator works recursively."""
        # At row_idx=2: rsi_14=55 > 50 (True), rsi_14=55 > 40 (True) = 2 votes >= 1
        # AND rsi_14=55 > 54 (True)
        condition = {
            "all": [
                {
                    "vote": {
                        "conditions": [
                            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 40},
                        ],
                        "min_votes": 1,
                    }
                },
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 54},
            ]
        }
        assert evaluate_condition(condition, features_df, row_idx=2) is True

    def test_max_depth_exceeded_inside_vote_raises(self, features_df):
        """max_depth exceeded inside vote -> raises ValueError."""
        condition = {
            "vote": {
                "conditions": [
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                ],
                "min_votes": 1,
            }
        }
        with pytest.raises(ValueError, match="max depth"):
            evaluate_condition(condition, features_df, row_idx=0, max_depth=1, _current_depth=1)
