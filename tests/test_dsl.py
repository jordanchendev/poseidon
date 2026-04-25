"""Tests for DSL engine — schema validation, condition evaluators, and tree executor."""

import pandas as pd
import pytest
from pydantic import ValidationError

from poseidon.strategies.dsl.conditions import CONDITION_REGISTRY, resolve_column_name
from poseidon.strategies.dsl.executor import evaluate_condition
from poseidon.strategies.dsl.schema import RuleConfig, RuleEntry


@pytest.fixture
def sample_features():
    """5-row DataFrame mimicking FeatureEngine output."""
    return pd.DataFrame(
        {
            "close": [100.0, 102.0, 98.0, 105.0, 103.0],
            "open": [99.0, 100.0, 102.0, 98.0, 105.0],
            "high": [101.0, 103.0, 102.5, 106.0, 104.0],
            "low": [98.5, 99.5, 97.0, 97.5, 102.0],
            "volume": [1000, 1200, 800, 3000, 1100],
            "sma_5": [100.0, 100.5, 100.2, 101.0, 101.6],
            "sma_20": [99.0, 99.2, 99.5, 99.8, 100.0],
            "sma_60": [97.0, 97.5, 97.8, 98.0, 98.5],
            "rsi_14": [55.0, 60.0, 25.0, 72.0, 50.0],
            "macd_line": [1.0, 1.2, 0.0, 1.5, 0.7],
            "return_1d": [0.01, 0.02, -0.04, 0.07, -0.02],
            "atr_14": [2.0, 2.1, 2.5, 2.3, 2.2],
            "std_vol_20": [0.015, 0.016, 0.018, 0.017, 0.016],
        }
    )


# ---------------------------------------------------------------------------
# TestDSLSchema
# ---------------------------------------------------------------------------


class TestDSLSchema:
    def test_rule_config_valid(self):
        config = RuleConfig(
            name="test",
            symbol="2330",
            market="tw_stock",
            rules=[
                RuleEntry(
                    condition={
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    action="long",
                )
            ],
        )
        assert config.name == "test"
        assert len(config.rules) == 1

    def test_rule_config_missing_rules(self):
        with pytest.raises(ValidationError):
            RuleConfig(name="test", symbol="2330", market="tw_stock", rules=[])

    def test_rule_entry_valid(self):
        entry = RuleEntry(
            condition={"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
            action="short",
            quantity_pct=0.05,
            confidence=0.8,
        )
        assert entry.action == "short"
        assert entry.quantity_pct == 0.05
        assert entry.confidence == 0.8


# ---------------------------------------------------------------------------
# TestColumnResolution
# ---------------------------------------------------------------------------


class TestColumnResolution:
    def test_resolve_rsi(self):
        assert resolve_column_name("rsi", {"period": 14}) == "rsi_14"

    def test_resolve_sma(self):
        assert resolve_column_name("sma", {"period": 60}) == "sma_60"

    def test_resolve_ma_alias(self):
        assert resolve_column_name("ma", {"period": 60}) == "sma_60"

    def test_resolve_macd(self):
        assert resolve_column_name("macd", {}) == "macd_line"


# ---------------------------------------------------------------------------
# TestDSLExecutor
# ---------------------------------------------------------------------------


class TestDSLExecutor:
    def test_evaluate_leaf_indicator_above(self, sample_features):
        # Row 3: rsi_14 = 72 > 70
        cond = {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70}
        assert evaluate_condition(cond, sample_features, 3) is True
        # Row 0: rsi_14 = 55, not > 70
        assert evaluate_condition(cond, sample_features, 0) is False

    def test_evaluate_leaf_indicator_below(self, sample_features):
        # Row 2: rsi_14 = 25 < 30
        cond = {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30}
        assert evaluate_condition(cond, sample_features, 2) is True
        # Row 0: rsi_14 = 55, not < 30
        assert evaluate_condition(cond, sample_features, 0) is False

    def test_evaluate_leaf_volume_spike(self, sample_features):
        # Row 3: volume=3000, avg of rows 0-2 = (1000+1200+800)/3 ≈ 1000, 3000 > 2*1000
        cond = {"type": "volume_spike", "params": {"period": 20, "multiplier": 2.0}}
        assert evaluate_condition(cond, sample_features, 3) is True
        # Row 4: volume=1100, avg of rows 0-3 = (1000+1200+800+3000)/4 = 1500, 1100 < 2*1500
        assert evaluate_condition(cond, sample_features, 4) is False

    def test_evaluate_leaf_price_change_pct(self, sample_features):
        # Row 3: return_1d = 0.07, direction=up, threshold=0.05 -> True
        cond = {"type": "price_change_pct", "threshold": 0.05, "direction": "up"}
        assert evaluate_condition(cond, sample_features, 3) is True
        # Row 0: return_1d = 0.01, not > 0.05
        assert evaluate_condition(cond, sample_features, 0) is False

    def test_evaluate_all_combinator(self, sample_features):
        # Row 2: rsi_14=25 (below 30) AND close=98 > sma_60=97.8
        cond = {
            "all": [
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                {"type": "indicator_above", "indicator": "price_vs_ma", "params": {"period": 60}, "threshold": 0},
            ]
        }
        assert evaluate_condition(cond, sample_features, 2) is True

        # Row 0: rsi_14=55, not below 30 -> all fails
        assert evaluate_condition(cond, sample_features, 0) is False

    def test_evaluate_any_combinator(self, sample_features):
        cond = {
            "any": [
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
            ]
        }
        # Row 2: rsi=25 < 30 -> True
        assert evaluate_condition(cond, sample_features, 2) is True
        # Row 3: rsi=72 > 70 -> True
        assert evaluate_condition(cond, sample_features, 3) is True
        # Row 4: rsi=50, neither -> False
        assert evaluate_condition(cond, sample_features, 4) is False

    def test_evaluate_none_combinator(self, sample_features):
        cond = {
            "none": [
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
            ]
        }
        # Row 0: rsi=55, not > 70 -> none satisfied -> True
        assert evaluate_condition(cond, sample_features, 0) is True
        # Row 3: rsi=72 > 70 -> one satisfied -> False
        assert evaluate_condition(cond, sample_features, 3) is False

    def test_nested_all_any(self, sample_features):
        """Two levels: all -> [leaf, any -> [leaf, leaf]]."""
        cond = {
            "all": [
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                {
                    "any": [
                        {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
                        {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                    ]
                },
            ]
        }
        # Row 2: rsi=25 -> below 30 (first leaf True), any: [False, True] -> True. All: True
        assert evaluate_condition(cond, sample_features, 2) is True

    def test_nested_all_any_fails(self, sample_features):
        """Outer all fails because first child is false."""
        cond = {
            "all": [
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
                {
                    "any": [
                        {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                        {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                    ]
                },
            ]
        }
        # Row 2: rsi=25, first child: 25 > 70 -> False. all short-circuits -> False
        assert evaluate_condition(cond, sample_features, 2) is False

    def test_unknown_condition_type_raises(self, sample_features):
        cond = {"type": "nonexistent", "indicator": "rsi", "threshold": 30}
        with pytest.raises(ValueError, match="Unknown condition type"):
            evaluate_condition(cond, sample_features, 0)

    def test_indicator_crosses_respects_indicator_param(self, sample_features):
        """indicator_crosses should use the 'indicator' field, not hardcode sma."""
        cond = {
            "type": "indicator_crosses",
            "indicator": "sma",
            "params": {"fast": 5, "slow": 20},
            "direction": "up",
        }
        # Row 1: sma_5=100.5 > sma_20=99.2, prev: sma_5=100.0 > sma_20=99.0
        # Already above so not a cross-up
        assert evaluate_condition(cond, sample_features, 1) is False

    def test_indicator_crosses_missing_fast_raises(self, sample_features):
        """Missing fast param should raise ValueError."""
        cond = {
            "type": "indicator_crosses",
            "indicator": "sma",
            "params": {"slow": 20},
            "direction": "up",
        }
        with pytest.raises(ValueError, match=r"requires params.fast and params.slow"):
            evaluate_condition(cond, sample_features, 1)

    def test_indicator_crosses_missing_slow_raises(self, sample_features):
        """Missing slow param should raise ValueError."""
        cond = {
            "type": "indicator_crosses",
            "indicator": "sma",
            "params": {"fast": 5},
            "direction": "up",
        }
        with pytest.raises(ValueError, match=r"requires params.fast and params.slow"):
            evaluate_condition(cond, sample_features, 1)

    def test_max_depth_exceeded_raises(self, sample_features):
        # Build deeply nested condition
        cond = {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50}
        for _ in range(12):
            cond = {"all": [cond]}
        with pytest.raises(ValueError, match="max depth"):
            evaluate_condition(cond, sample_features, 0)


# ---------------------------------------------------------------------------
# R2 Feature Condition Tests — feature_above / feature_below
# ---------------------------------------------------------------------------


@pytest.fixture
def r2_features():
    """DataFrame with R2 feature columns for testing feature_above/below."""
    return pd.DataFrame(
        {
            "close": [100.0, 102.0, 98.0],
            "foreign_net_buy_ratio": [0.02, 0.005, float("nan")],
            "pe_ratio": [15.0, 25.0, float("nan")],
            "funding_rate_daily": [0.001, -0.002, float("nan")],
        }
    )


class TestFeatureAbove:
    """Tests for feature_above condition type."""

    def test_above_threshold_returns_true(self, r2_features):
        cond = {"type": "feature_above", "column": "foreign_net_buy_ratio", "threshold": 0.01}
        assert evaluate_condition(cond, r2_features, 0) is True  # 0.02 > 0.01

    def test_below_threshold_returns_false(self, r2_features):
        cond = {"type": "feature_above", "column": "foreign_net_buy_ratio", "threshold": 0.01}
        assert evaluate_condition(cond, r2_features, 1) is False  # 0.005 < 0.01

    def test_nan_returns_false(self, r2_features):
        cond = {"type": "feature_above", "column": "foreign_net_buy_ratio", "threshold": 0.01}
        assert evaluate_condition(cond, r2_features, 2) is False  # NaN

    def test_missing_column_returns_false(self, r2_features):
        cond = {"type": "feature_above", "column": "nonexistent_column", "threshold": 0.01}
        assert evaluate_condition(cond, r2_features, 0) is False

    def test_registered_in_registry(self):
        assert "feature_above" in CONDITION_REGISTRY


class TestFeatureBelow:
    """Tests for feature_below condition type."""

    def test_below_threshold_returns_true(self, r2_features):
        cond = {"type": "feature_below", "column": "pe_ratio", "threshold": 20.0}
        assert evaluate_condition(cond, r2_features, 0) is True  # 15.0 < 20.0

    def test_above_threshold_returns_false(self, r2_features):
        cond = {"type": "feature_below", "column": "pe_ratio", "threshold": 20.0}
        assert evaluate_condition(cond, r2_features, 1) is False  # 25.0 > 20.0

    def test_nan_returns_false(self, r2_features):
        cond = {"type": "feature_below", "column": "pe_ratio", "threshold": 20.0}
        assert evaluate_condition(cond, r2_features, 2) is False  # NaN

    def test_missing_column_returns_false(self, r2_features):
        cond = {"type": "feature_below", "column": "nonexistent_column", "threshold": 20.0}
        assert evaluate_condition(cond, r2_features, 0) is False

    def test_registered_in_registry(self):
        assert "feature_below" in CONDITION_REGISTRY
