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


# ---------- COMP-02: Concrete component annotations ----------


class TestConcreteStrategyAnnotations:
    """Verify live-capable and stateful strategies are correctly annotated."""

    def test_voting_strategy_supports_live(self):
        from poseidon.strategies.voting_strategy import VotingStrategy

        assert VotingStrategy.supports_live is True
        assert VotingStrategy.stateful is True

    def test_regime_router_supports_live(self):
        from poseidon.strategies.regime_router import RegimeRouter

        assert RegimeRouter.supports_live is True
        assert RegimeRouter.stateful is True

    def test_model_strategy_defaults(self):
        from poseidon.strategies.model_strategy import ModelStrategy

        assert ModelStrategy.supports_live is False  # backtest only
        assert ModelStrategy.stateful is False

    def test_rule_strategy_defaults(self):
        from poseidon.strategies.rule_strategy import RuleStrategy

        assert RuleStrategy.supports_live is False
        assert RuleStrategy.stateful is False

    def test_crypto_trend_stateful(self):
        from poseidon.strategies.portfolio.crypto_trend import CryptoTrendStrategy

        assert CryptoTrendStrategy.supports_live is True
        assert CryptoTrendStrategy.stateful is True

    def test_revenue_breakout_supports_live(self):
        from poseidon.strategies.portfolio.revenue_breakout import (
            RevenueBreakoutStrategy,
        )

        assert RevenueBreakoutStrategy.supports_live is True
        assert RevenueBreakoutStrategy.stateful is False


class TestConcreteRiskRuleAnnotations:
    """All 6 risk rules must have supports_live=True."""

    @pytest.mark.parametrize(
        "rule_cls_path",
        [
            "poseidon.risk.rules.position_limit.PositionLimitRule",
            "poseidon.risk.rules.loss_limit.LossLimitRule",
            "poseidon.risk.rules.frequency.FrequencyRule",
            "poseidon.risk.rules.confidence_threshold.ConfidenceThresholdRule",
            "poseidon.risk.rules.leverage_cap.LeverageCapRule",
            "poseidon.risk.rules.var_limit.VaRLimitRule",
        ],
    )
    def test_risk_rule_supports_live(self, rule_cls_path):
        import importlib

        module_path, cls_name = rule_cls_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        assert cls.supports_live is True, f"{cls_name}.supports_live should be True"


class TestConcreteMLModelAnnotations:
    """All ML models must have supports_live=True."""

    def test_xgboost_model_supports_live(self):
        from poseidon.ml.implementations.xgboost_model import XGBoostModel

        assert XGBoostModel.supports_live is True

    def test_transformer_model_supports_live(self):
        from poseidon.ml.implementations.transformer_model import TransformerModel

        assert TransformerModel.supports_live is True

    def test_xgboost_regime_model_supports_live(self):
        from poseidon.ml.implementations.xgboost_regime import XGBoostRegimeModel

        assert XGBoostRegimeModel.supports_live is True


class TestHMMRegimeFeatureAnnotation:
    """HMM regime feature must have bias_risk and stateful annotations."""

    def test_hmm_regime_bias_risk(self):
        from poseidon.data.features.hmm_regime import HMMRegime

        assert HMMRegime.bias_risk == ["future_data_leakage"]
        assert HMMRegime.stateful is True


class TestRegistryDiscovery:
    """Verify registries discover concrete components after import."""

    def test_list_strategies_has_expected_names(self):
        # Force imports to trigger registration
        import poseidon.strategies.voting_strategy  # noqa: F401
        import poseidon.strategies.regime_router  # noqa: F401
        import poseidon.strategies.model_strategy  # noqa: F401
        import poseidon.strategies.rule_strategy  # noqa: F401

        from poseidon.strategies.registry import list_strategies

        names = list_strategies()
        assert "voting_strategy" in names
        assert "regime_router" in names
        assert "model_strategy" in names
        assert "rule_strategy" in names

    def test_list_portfolio_strategies_has_expected_names(self):
        import poseidon.strategies.portfolio.revenue_breakout  # noqa: F401
        import poseidon.strategies.portfolio.crypto_trend  # noqa: F401

        from poseidon.strategies.portfolio.registry import list_portfolio_strategies

        names = list_portfolio_strategies()
        assert "revenue_breakout" in names
        assert "crypto_trend" in names
