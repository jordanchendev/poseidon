"""Tests for ModelStrategy and RuleStrategy."""

import pandas as pd
import pytest

from poseidon.ml.base import BaseModel as MLBaseModel
from poseidon.signals.schemas import SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.model_strategy import ModelStrategy
from poseidon.strategies.rule_strategy import RuleStrategy

# ---------------------------------------------------------------------------
# DummyModel stub
# ---------------------------------------------------------------------------


class DummyModel(MLBaseModel):
    """Minimal BaseModel for testing — predict() returns a pre-set DataFrame."""

    name = "dummy"
    description = "test stub"

    def __init__(self, predictions_df: pd.DataFrame | None = None):
        self._predictions_df = predictions_df

    def train(self, features, targets, params):
        return {}

    def predict(self, features):
        if self._predictions_df is not None:
            return self._predictions_df
        return pd.DataFrame(
            {"prediction": ["hold"] * len(features), "confidence": [0.5] * len(features)},
            index=features.index,
        )

    def validate(self, features, targets):
        return {}

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()

    def get_default_params(self):
        return {}

    def get_feature_list(self):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features():
    return pd.DataFrame(
        {"close": [100.0, 102.0, 98.0], "sma_20": [99.0, 99.5, 99.8], "rsi_14": [55.0, 60.0, 25.0]},
    )


# ---------------------------------------------------------------------------
# TestModelStrategy
# ---------------------------------------------------------------------------


class TestModelStrategy:
    def test_model_strategy_evaluate_filters_hold(self, sample_features):
        preds = pd.DataFrame(
            {"prediction": ["long", "hold", "short"], "confidence": [0.9, 0.5, 0.8]},
            index=sample_features.index,
        )
        model = DummyModel(predictions_df=preds)
        ms = ModelStrategy(name="test", model=model, symbol="2330", market="tw_stock")
        signals = ms.evaluate(sample_features)
        assert len(signals) == 2
        assert signals[0].action == SignalAction.LONG
        assert signals[1].action == SignalAction.SHORT

    def test_model_strategy_signal_fields(self, sample_features):
        preds = pd.DataFrame(
            {"prediction": ["long"], "confidence": [0.85]},
            index=sample_features.index[:1],
        )
        model = DummyModel(predictions_df=preds)
        ms = ModelStrategy(name="test", model=model, symbol="2330", market="tw_stock", interval="1h")
        signals = ms.evaluate(sample_features[:1])
        s = signals[0]
        assert s.symbol == "2330"
        assert s.market == "tw_stock"
        assert s.interval == "1h"
        assert s.strategy_id == ms.strategy_id
        assert s.confidence == 0.85

    def test_model_strategy_evaluate_all_hold(self, sample_features):
        model = DummyModel()  # defaults to all hold
        ms = ModelStrategy(name="test", model=model, symbol="2330", market="tw_stock")
        signals = ms.evaluate(sample_features)
        assert len(signals) == 0

    def test_model_strategy_evaluate_single_row(self):
        features = pd.DataFrame({"close": [100.0]})
        preds = pd.DataFrame(
            {"prediction": ["long"], "confidence": [0.85]},
            index=features.index,
        )
        model = DummyModel(predictions_df=preds)
        ms = ModelStrategy(name="test", model=model, symbol="AAPL", market="us_stock")
        signals = ms.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG
        assert signals[0].confidence == 0.85

    def test_model_strategy_validate_config_valid(self, sample_features):
        model = DummyModel()
        ms = ModelStrategy(name="test", model=model, symbol="2330", market="tw_stock")
        assert ms.validate_config() is True

    def test_model_strategy_validate_config_missing_model(self):
        ms = ModelStrategy.__new__(ModelStrategy)
        ms.model = None
        ms.symbol = "2330"
        ms.market = "tw_stock"
        with pytest.raises(ValueError, match="model instance"):
            ms.validate_config()

    def test_model_strategy_is_base_strategy(self):
        assert issubclass(ModelStrategy, BaseStrategy)
        model = DummyModel()
        ms = ModelStrategy(name="test", model=model, symbol="2330", market="tw_stock")
        assert ms.strategy_type == StrategyType.MODEL


# ---------------------------------------------------------------------------
# TestRuleStrategy
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_features():
    """5-row DataFrame with feature columns for DSL testing."""
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
            "return_1d": [0.01, 0.02, -0.04, 0.07, -0.02],
            "atr_14": [2.0, 2.1, 2.5, 2.3, 2.2],
        }
    )


def _make_rule_config(rules, **kwargs):
    defaults = {"name": "test_rule", "symbol": "2330", "market": "tw_stock", "interval": "1d"}
    defaults.update(kwargs)
    defaults["rules"] = rules
    return defaults


class TestRuleStrategy:
    def test_rule_strategy_evaluate_simple_condition(self, rule_features):
        """Row 2 has rsi_14=25 (below 30), last row has rsi_14=50 (not below 30).
        But we only check last row, so this should NOT trigger."""
        # Use a 1-row DataFrame where rsi_14 < 30
        features = rule_features.iloc[2:3].reset_index(drop=True)
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    "action": "long",
                    "quantity_pct": 0.1,
                    "confidence": 0.9,
                }
            ]
        )
        rs = RuleStrategy(config=config)
        signals = rs.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG

    def test_rule_strategy_evaluate_no_trigger(self, rule_features):
        """Last row rsi_14=50, not below 30."""
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    "action": "long",
                }
            ]
        )
        rs = RuleStrategy(config=config)
        signals = rs.evaluate(rule_features)
        assert len(signals) == 0

    def test_rule_strategy_evaluate_nested_condition(self, rule_features):
        """Two-level nesting: all -> [leaf, any -> [leaf, leaf]]."""
        features = rule_features.iloc[2:3].reset_index(drop=True)  # rsi=25
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "all": [
                            {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                            {
                                "any": [
                                    {
                                        "type": "indicator_above",
                                        "indicator": "rsi",
                                        "params": {"period": 14},
                                        "threshold": 70,
                                    },
                                    {
                                        "type": "indicator_below",
                                        "indicator": "rsi",
                                        "params": {"period": 14},
                                        "threshold": 30,
                                    },
                                ]
                            },
                        ]
                    },
                    "action": "long",
                }
            ]
        )
        rs = RuleStrategy(config=config)
        signals = rs.evaluate(features)
        assert len(signals) == 1

    def test_rule_strategy_signal_fields(self, rule_features):
        features = rule_features.iloc[2:3].reset_index(drop=True)
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    "action": "long",
                    "quantity_pct": 0.1,
                    "confidence": 0.9,
                }
            ]
        )
        rs = RuleStrategy(config=config)
        signals = rs.evaluate(features)
        s = signals[0]
        assert s.symbol == "2330"
        assert s.market == "tw_stock"
        assert s.interval == "1d"
        assert s.action == SignalAction.LONG
        assert s.quantity_pct == 0.1
        assert s.confidence == 0.9

    def test_rule_strategy_multiple_rules(self, rule_features):
        """Two rules that both trigger."""
        features = rule_features.iloc[2:3].reset_index(drop=True)  # rsi=25
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    "action": "long",
                },
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 50,
                    },
                    "action": "short",
                },
            ]
        )
        rs = RuleStrategy(config=config)
        signals = rs.evaluate(features)
        assert len(signals) == 2
        assert signals[0].action == SignalAction.LONG
        assert signals[1].action == SignalAction.SHORT

    def test_rule_strategy_validate_config_valid(self):
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    "action": "long",
                }
            ]
        )
        rs = RuleStrategy(config=config)
        assert rs.validate_config() is True

    def test_rule_strategy_validate_config_no_rules(self):
        """Cannot create RuleConfig with empty rules (Pydantic validation)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_rule_config_obj = {"name": "test", "symbol": "2330", "market": "tw_stock", "rules": []}
            RuleStrategy(config=_make_rule_config_obj)

    def test_rule_strategy_is_base_strategy(self):
        assert issubclass(RuleStrategy, BaseStrategy)
        config = _make_rule_config(
            [
                {
                    "condition": {
                        "type": "indicator_below",
                        "indicator": "rsi",
                        "params": {"period": 14},
                        "threshold": 30,
                    },
                    "action": "long",
                }
            ]
        )
        rs = RuleStrategy(config=config)
        assert rs.strategy_type == StrategyType.RULE
