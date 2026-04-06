"""Tests for component capability metadata (Phase 34, COMP-01 through COMP-04).

Verifies:
- All 5 base classes carry capability metadata with conservative defaults
- Subclass override isolation (mutable list safety)
- Strategy and portfolio strategy registries work correctly
"""

import pytest


# ---------- COMP-01: Conservative defaults on all 5 base classes ----------


class TestBaseClassDefaults:
    """Each base class must have supports_backtest=True, supports_live=False,
    bias_risk=[], stateful=False as class-level attributes."""

    def test_base_feature_defaults(self):
        from poseidon.data.features.base import BaseFeature

        assert BaseFeature.supports_backtest is True
        assert BaseFeature.supports_live is False
        assert BaseFeature.bias_risk == []
        assert BaseFeature.stateful is False

    def test_base_model_defaults(self):
        from poseidon.ml.base import BaseModel

        assert BaseModel.supports_backtest is True
        assert BaseModel.supports_live is False
        assert BaseModel.bias_risk == []
        assert BaseModel.stateful is False

    def test_base_strategy_defaults(self):
        from poseidon.strategies.base import BaseStrategy

        assert BaseStrategy.supports_backtest is True
        assert BaseStrategy.supports_live is False
        assert BaseStrategy.bias_risk == []
        assert BaseStrategy.stateful is False

    def test_base_rule_defaults(self):
        from poseidon.risk.base import BaseRule

        assert BaseRule.supports_backtest is True
        assert BaseRule.supports_live is False
        assert BaseRule.bias_risk == []
        assert BaseRule.stateful is False

    def test_portfolio_strategy_defaults(self):
        from poseidon.strategies.portfolio.base import PortfolioStrategy

        assert PortfolioStrategy.supports_backtest is True
        assert PortfolioStrategy.supports_live is False
        assert PortfolioStrategy.bias_risk == []
        assert PortfolioStrategy.stateful is False


# ---------- COMP-01: Subclass override isolation ----------


class TestSubclassIsolation:
    """Subclass overrides must not affect parent defaults."""

    def test_subclass_override_does_not_affect_parent(self):
        from poseidon.strategies.base import BaseStrategy

        class LiveStrategy(BaseStrategy):
            name = "live_test"
            strategy_type = "model"
            supports_live = True

            def evaluate(self, features):
                return []

            def validate_config(self):
                return True

        assert LiveStrategy.supports_live is True
        assert BaseStrategy.supports_live is False  # parent unchanged

    def test_subclass_bias_risk_does_not_mutate_parent(self):
        from poseidon.data.features.base import BaseFeature

        class LeakyFeature(BaseFeature):
            name = "leaky"
            description = "test"
            bias_risk = ["lookahead"]

            def compute(self, ohlcv, **params):
                return ohlcv

        assert LeakyFeature.bias_risk == ["lookahead"]
        assert BaseFeature.bias_risk == []  # parent unchanged


# ---------- COMP-03: Strategy registry ----------


class TestStrategyRegistry:
    """Strategy registry must support register, get, list, and KeyError."""

    def test_register_and_list(self):
        from poseidon.strategies.registry import (
            _registry,
            list_strategies,
            register_strategy,
        )
        from poseidon.strategies.base import BaseStrategy

        # Clear to avoid side effects
        _registry.clear()

        @register_strategy
        class DummyStrategy(BaseStrategy):
            name = "dummy_strat"
            strategy_type = "model"

            def evaluate(self, features):
                return []

            def validate_config(self):
                return True

        assert "dummy_strat" in list_strategies()
        _registry.clear()

    def test_get_strategy_found(self):
        from poseidon.strategies.registry import (
            _registry,
            get_strategy,
            register_strategy,
        )
        from poseidon.strategies.base import BaseStrategy

        _registry.clear()

        @register_strategy
        class FindMe(BaseStrategy):
            name = "find_me"
            strategy_type = "model"

            def evaluate(self, features):
                return []

            def validate_config(self):
                return True

        assert get_strategy("find_me") is FindMe
        _registry.clear()

    def test_get_strategy_raises_key_error(self):
        from poseidon.strategies.registry import _registry, get_strategy

        _registry.clear()
        with pytest.raises(KeyError, match="no_such_strategy"):
            get_strategy("no_such_strategy")


# ---------- COMP-03: Portfolio strategy registry ----------


class TestPortfolioStrategyRegistry:
    """Portfolio strategy registry mirrors strategy registry pattern."""

    def test_register_and_list(self):
        from poseidon.strategies.portfolio.registry import (
            _registry,
            list_portfolio_strategies,
            register_portfolio_strategy,
        )
        from poseidon.strategies.portfolio.base import PortfolioStrategy

        _registry.clear()

        @register_portfolio_strategy
        class DummyPortfolio(PortfolioStrategy):
            name = "dummy_port"

            def select_stocks(self, universe_df, as_of=None):
                return []

            def validate_config(self):
                return True

        assert "dummy_port" in list_portfolio_strategies()
        _registry.clear()

    def test_get_portfolio_strategy_raises_key_error(self):
        from poseidon.strategies.portfolio.registry import (
            _registry,
            get_portfolio_strategy,
        )

        _registry.clear()
        with pytest.raises(KeyError, match="no_such"):
            get_portfolio_strategy("no_such")
