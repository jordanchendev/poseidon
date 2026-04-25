"""Tests for regime label generation and RegimeRouter."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from poseidon.backtest.regime_labels import generate_regime_labels
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.regime_router import DEFAULT_REGIME_CONFIGS, RegimeRouter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_regime_model(regime_label: str = "high_vol") -> MagicMock:
    """Create a mock XGBoostRegimeModel that returns a fixed regime prediction."""
    model = MagicMock()
    model.predict.return_value = pd.DataFrame({"prediction": [regime_label]})
    return model


def _minimal_voting_config() -> dict:
    """Minimal valid VotingStrategy config for factory creation."""
    return {
        "name": "test_voting",
        "symbol": "BTCUSDT",
        "market": "crypto_spot",
        "interval": "1d",
        "min_votes": 4,
        "position_pct": 0.08,
        "sub_signals": [
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 40},
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 20},
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 10},
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 5},
        ],
    }


# ---------------------------------------------------------------------------
# Label generation tests
# ---------------------------------------------------------------------------


class TestGenerateRegimeLabels:
    """Tests for generate_regime_labels."""

    def test_label_generation_percentiles(self):
        """generate_regime_labels with known vol data produces 3 classes and thresholds."""
        vol_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        features = pd.DataFrame({"realized_vol_20": vol_data})
        labels, thresholds = generate_regime_labels(features)

        unique_classes = set(labels.dropna().unique())
        assert unique_classes == {0, 1, 2}, f"Expected 3 classes (0,1,2), got {unique_classes}"
        assert "low_threshold" in thresholds
        assert "high_threshold" in thresholds
        assert "low_pct" in thresholds
        assert "high_pct" in thresholds

    def test_label_generation_returns_correct_classes(self):
        """For uniform vol data, ~33% of labels are each class."""
        np.random.seed(42)
        vol_data = np.random.uniform(0, 1, 300)
        features = pd.DataFrame({"realized_vol_20": vol_data})
        labels, _ = generate_regime_labels(features)

        counts = labels.value_counts()
        total = len(labels)
        for cls in [0, 1, 2]:
            pct = counts.get(cls, 0) / total
            assert 0.2 < pct < 0.5, f"Class {cls} has {pct:.1%}, expected ~33%"

    def test_label_generation_threshold_persistence(self):
        """Returned thresholds dict has correct keys with float values."""
        features = pd.DataFrame({"realized_vol_20": [0.1, 0.2, 0.3, 0.4, 0.5]})
        _, thresholds = generate_regime_labels(features)

        for key in ["low_pct", "high_pct", "low_threshold", "high_threshold"]:
            assert key in thresholds, f"Missing key: {key}"
            assert isinstance(thresholds[key], float), f"{key} should be float, got {type(thresholds[key])}"


# ---------------------------------------------------------------------------
# RegimeRouter tests
# ---------------------------------------------------------------------------


class TestRegimeRouter:
    """Tests for RegimeRouter."""

    def test_regime_router_applies_config(self):
        """RegimeRouter with enabled=True and high_vol regime applies correct overrides."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("high_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
            enabled=True,
        )

        # Create minimal feature data (just needs to be non-empty for regime prediction)
        features = pd.DataFrame({"close": [100.0], "rsi_14": [55.0], "atr_14": [2.0]})

        # After evaluate, the underlying strategy should have high_vol config
        router.evaluate(features)
        assert router._strategy._min_votes == DEFAULT_REGIME_CONFIGS["high_vol"]["min_votes"]
        assert router._strategy._position_pct == DEFAULT_REGIME_CONFIGS["high_vol"]["position_pct"]

    def test_regime_router_preserves_state(self):
        """RegimeRouter does not re-instantiate VotingStrategy between calls."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("high_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
            enabled=True,
        )

        strategy_id_before = id(router._strategy)

        features = pd.DataFrame({"close": [100.0], "rsi_14": [55.0], "atr_14": [2.0]})
        router.evaluate(features)

        # Change regime prediction
        mock_model.predict.return_value = pd.DataFrame({"prediction": ["low_vol"]})
        router.evaluate(features)

        strategy_id_after = id(router._strategy)
        assert strategy_id_before == strategy_id_after, "Strategy object was re-instantiated"

    def test_disabled_router_passthrough(self):
        """Disabled RegimeRouter uses base config values regardless of regime prediction."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("high_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
            enabled=False,
        )

        features = pd.DataFrame({"close": [100.0], "rsi_14": [55.0], "atr_14": [2.0]})
        router.evaluate(features)

        # Should use base config values, not high_vol overrides
        assert router._strategy._min_votes == config["min_votes"]
        assert router._strategy._position_pct == config["position_pct"]

    def test_regime_router_is_base_strategy(self):
        """RegimeRouter is a BaseStrategy with VOTING strategy_type."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("medium_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
        )

        assert isinstance(router, BaseStrategy)
        assert router.strategy_type == StrategyType.VOTING

    def test_regime_router_reset(self):
        """RegimeRouter.reset() delegates to underlying strategy.reset()."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("medium_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
        )

        # Set some trailing stop state on the inner strategy
        router._strategy._position_direction = "long"
        router._strategy._position_high_watermark = 100.0

        router.reset()

        assert router._strategy._position_direction is None
        assert router._strategy._position_high_watermark is None


class TestRegimeRouterBear:
    """Tests for RegimeRouter bear parameter overrides."""

    def test_default_regime_configs_have_bear_params(self):
        """All 3 regime configs must include bear_min_votes and bear_position_pct."""
        for regime_name, cfg in DEFAULT_REGIME_CONFIGS.items():
            assert "bear_min_votes" in cfg, f"{regime_name} missing bear_min_votes"
            assert "bear_position_pct" in cfg, f"{regime_name} missing bear_position_pct"

    def test_high_vol_regime_config(self):
        """High vol regime config per D-20."""
        expected = {"min_votes": 5, "position_pct": 0.05, "bear_min_votes": 4, "bear_position_pct": 0.04}
        assert DEFAULT_REGIME_CONFIGS["high_vol"] == expected

    def test_medium_vol_regime_config(self):
        """Medium vol regime config per D-20."""
        expected = {"min_votes": 4, "position_pct": 0.08, "bear_min_votes": 4, "bear_position_pct": 0.06}
        assert DEFAULT_REGIME_CONFIGS["medium_vol"] == expected

    def test_low_vol_regime_config(self):
        """Low vol regime config per D-20."""
        expected = {"min_votes": 3, "position_pct": 0.10, "bear_min_votes": 5, "bear_position_pct": 0.05}
        assert DEFAULT_REGIME_CONFIGS["low_vol"] == expected

    def test_evaluate_sets_bear_min_votes(self):
        """RegimeRouter.evaluate() sets _bear_min_votes from regime config."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("high_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
            enabled=True,
        )

        features = pd.DataFrame({"close": [100.0], "rsi_14": [55.0], "atr_14": [2.0]})
        router.evaluate(features)
        assert router._strategy._bear_min_votes == DEFAULT_REGIME_CONFIGS["high_vol"]["bear_min_votes"]

    def test_evaluate_sets_bear_position_pct(self):
        """RegimeRouter.evaluate() sets _bear_position_pct from regime config."""
        config = _minimal_voting_config()
        mock_model = _make_mock_regime_model("high_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
            enabled=True,
        )

        features = pd.DataFrame({"close": [100.0], "rsi_14": [55.0], "atr_14": [2.0]})
        router.evaluate(features)
        assert router._strategy._bear_position_pct == DEFAULT_REGIME_CONFIGS["high_vol"]["bear_position_pct"]

    def test_disabled_router_sets_bear_from_base_config(self):
        """Disabled RegimeRouter uses base_config bear values."""
        config = _minimal_voting_config()
        config["bear_min_votes"] = 3
        config["bear_position_pct"] = 0.05
        mock_model = _make_mock_regime_model("high_vol")

        router = RegimeRouter(
            base_config=config,
            regime_model=mock_model,
            enabled=False,
        )

        features = pd.DataFrame({"close": [100.0], "rsi_14": [55.0], "atr_14": [2.0]})
        router.evaluate(features)
        assert router._strategy._bear_min_votes == 3
        assert router._strategy._bear_position_pct == 0.05
