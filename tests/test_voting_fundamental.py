"""Tests for VotingStrategy fundamental sub-signals (feature_above condition type).

Validates that quality_profitability_z and foreign_net_buy_ratio can be used
as voting sub-signals via the feature_above condition type in conditions.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.strategies.dsl.conditions import eval_feature_above
from poseidon.strategies.voting_strategy import VotingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = (
    Path(__file__).parent.parent
    / "config"
    / "strategies"
    / "voting_tw_stock_fundamental.json"
)


def make_features_with_fundamental(
    n_rows: int = 200,
    *,
    quality_profitability_z: float = 1.0,
    foreign_net_buy_ratio: float = 0.02,
    include_quality_z: bool = True,
    include_net_buy: bool = True,
    **overrides: float,
) -> pd.DataFrame:
    """Create synthetic DataFrame with both TA and fundamental columns.

    Args:
        n_rows: Number of rows.
        quality_profitability_z: Value for the quality Z-score column.
        foreign_net_buy_ratio: Value for the foreign net buy ratio column.
        include_quality_z: Whether to include quality_profitability_z column.
        include_net_buy: Whether to include foreign_net_buy_ratio column.
        **overrides: Override any column value at the last row.
    """
    data: dict[str, list[float]] = {
        # TA columns
        "close": [100.0] * n_rows,
        "ema_7": [100.0] * n_rows,
        "ema_26": [99.0] * n_rows,
        "rsi_8": [55.0] * n_rows,
        "macd_histogram": [0.5] * n_rows,
        "bb_upper_20": [105.0] * n_rows,
        "bb_lower_20": [95.0] * n_rows,
        "atr_14": [2.0] * n_rows,
        "cum_return_6d": [0.01] * n_rows,
    }
    # Fundamental columns (conditionally included)
    if include_quality_z:
        data["quality_profitability_z"] = [quality_profitability_z] * n_rows
    if include_net_buy:
        data["foreign_net_buy_ratio"] = [foreign_net_buy_ratio] * n_rows

    df = pd.DataFrame(data)
    for col, val in overrides.items():
        df.loc[df.index[-1], col] = val
    return df


def _make_config_with_fundamental_signals() -> dict:
    """Return a VotingStrategy config dict matching voting_tw_stock_fundamental.json."""
    return {
        "name": "voting_tw_stock_fundamental",
        "symbol": "2330",
        "market": "tw_stock",
        "interval": "1d",
        "min_votes": 4,
        "position_pct": 0.08,
        "cooldown_bars": 0,
        "conviction_gap": 0,
        "sub_signals": [
            {
                "type": "indicator_above",
                "indicator": "rsi",
                "params": {"period": 8},
                "threshold": 50,
            },
            {
                "type": "indicator_above",
                "indicator": "cum_return",
                "params": {"period": 6},
                "threshold": 0,
            },
            {
                "type": "indicator_above",
                "indicator": "macd_histogram",
                "params": {},
                "threshold": 0,
            },
            {
                "type": "bollinger_width_percentile",
                "params": {"period": 20, "lookback": 168},
                "threshold": 0.85,
            },
            {
                "type": "feature_above",
                "column": "quality_profitability_z",
                "threshold": 0.5,
            },
            {
                "type": "feature_above",
                "column": "foreign_net_buy_ratio",
                "threshold": 0.0,
            },
        ],
    }


# ---------------------------------------------------------------------------
# TestFundamentalSubSignals — feature_above condition evaluation
# ---------------------------------------------------------------------------


class TestFundamentalSubSignals:
    """Unit tests for VotingStrategy fundamental sub-signals."""

    def test_quality_z_above_threshold_contributes_vote(self):
        """quality_profitability_z = 1.5 > threshold 0.5 -> eval_feature_above returns True."""
        features = make_features_with_fundamental(quality_profitability_z=1.5)
        condition = {"column": "quality_profitability_z", "threshold": 0.5}
        row_idx = len(features) - 1
        assert eval_feature_above(condition, features, row_idx) is True

    def test_quality_z_below_threshold_no_vote(self):
        """quality_profitability_z = 0.2 < threshold 0.5 -> eval_feature_above returns False."""
        features = make_features_with_fundamental(quality_profitability_z=0.2)
        condition = {"column": "quality_profitability_z", "threshold": 0.5}
        row_idx = len(features) - 1
        assert eval_feature_above(condition, features, row_idx) is False

    def test_net_buy_above_threshold_contributes_vote(self):
        """foreign_net_buy_ratio = 0.03 > threshold 0.0 -> eval_feature_above returns True."""
        features = make_features_with_fundamental(foreign_net_buy_ratio=0.03)
        condition = {"column": "foreign_net_buy_ratio", "threshold": 0.0}
        row_idx = len(features) - 1
        assert eval_feature_above(condition, features, row_idx) is True

    def test_feature_above_returns_false_on_nan(self):
        """eval_feature_above returns False when column value is NaN (T-67-04 mitigation)."""
        features = make_features_with_fundamental(quality_profitability_z=float("nan"))
        condition = {"column": "quality_profitability_z", "threshold": 0.5}
        row_idx = len(features) - 1
        assert eval_feature_above(condition, features, row_idx) is False

    def test_feature_above_returns_false_on_missing_column(self):
        """eval_feature_above returns False when column does not exist in DataFrame."""
        features = make_features_with_fundamental(include_quality_z=False)
        condition = {"column": "quality_profitability_z", "threshold": 0.5}
        row_idx = len(features) - 1
        assert eval_feature_above(condition, features, row_idx) is False

    def test_config_json_loads_correctly(self):
        """voting_tw_stock_fundamental.json loads correctly with expected structure."""
        with open(_CONFIG_PATH) as f:
            config = json.load(f)

        assert config["name"] == "voting_tw_stock_fundamental"
        assert config["market"] == "tw_stock"
        assert config["interval"] == "1d"
        assert len(config["sub_signals"]) == 6

        # Last 2 sub_signals should be feature_above
        feature_signals = [
            s for s in config["sub_signals"] if s["type"] == "feature_above"
        ]
        assert len(feature_signals) == 2

        columns = {s["column"] for s in feature_signals}
        assert columns == {"quality_profitability_z", "foreign_net_buy_ratio"}

    def test_feature_above_signals_count_as_votes(self):
        """VotingStrategy with fundamental config counts feature_above as votes.

        When quality_profitability_z=1.5 (above 0.5) and
        foreign_net_buy_ratio=0.03 (above 0.0), both fundamental sub-signals
        contribute 1 vote each via _count_votes().
        """
        config = _make_config_with_fundamental_signals()
        strategy = VotingStrategy(config=config)

        # Create features where ALL 6 conditions should be True:
        # - rsi_8=55 > 50
        # - cum_return_6d=0.01 > 0
        # - macd_histogram=0.5 > 0
        # - BB width percentile (narrow bands -> below 0.85 threshold? need squeeze)
        # - quality_profitability_z=1.5 > 0.5
        # - foreign_net_buy_ratio=0.03 > 0.0
        features = make_features_with_fundamental(
            quality_profitability_z=1.5,
            foreign_net_buy_ratio=0.03,
        )
        # Make BB squeeze (narrow bands -> percentile low -> below threshold -> NOT a vote)
        # BB width percentile threshold=0.85 means "BB width percentile > 0.85" to be True
        # For BB squeeze, we want narrow bands where percentile is high...
        # Actually, bollinger_width_percentile checks if current width is in a low percentile
        # (squeeze). It returns True if width_percentile < threshold.
        # So narrow bands -> low width -> low percentile rank -> True when threshold=0.85.
        # With the synthetic data, we need the last row to have narrow bands.
        features.loc[features.index[-1], "bb_upper_20"] = 101.0
        features.loc[features.index[-1], "bb_lower_20"] = 99.0

        # Count votes directly via _count_votes
        row_idx = len(features) - 1
        votes = strategy._count_votes(strategy._sub_signals, features, row_idx)

        # At minimum, feature_above conditions should count:
        # quality_profitability_z (1.5 > 0.5) = 1 vote
        # foreign_net_buy_ratio (0.03 > 0.0) = 1 vote
        # Plus TA votes: rsi=55>50, cum_return_6d=0.01>0, macd=0.5>0 = 3 more
        # BB squeeze depends on percentile calculation (may or may not count)
        # So at least 5 votes from the clearly-true conditions
        assert votes >= 5, f"Expected at least 5 votes (3 TA + 2 fundamental), got {votes}"

        # Verify the 2 fundamental signals contribute by removing them and comparing
        config_no_fundamental = _make_config_with_fundamental_signals()
        config_no_fundamental["sub_signals"] = [
            s
            for s in config_no_fundamental["sub_signals"]
            if s["type"] != "feature_above"
        ]
        strategy_no_fundamental = VotingStrategy(config=config_no_fundamental)
        votes_no_fundamental = strategy_no_fundamental._count_votes(
            strategy_no_fundamental._sub_signals, features, row_idx
        )
        # The difference should be exactly 2 (the 2 fundamental sub-signals)
        assert votes - votes_no_fundamental == 2
