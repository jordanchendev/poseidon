"""Tests for pipeline enforcement — capability validation and BacktestRunner integration.

Tests COMP-05 enforcement behavior:
- validate_live_components rejects supports_live=False
- validate_backtest_components rejects supports_backtest=False
- warn_bias_risks logs warnings for components with bias_risk
- BacktestRunner.__init__ enforces at construction time
- get_all_capabilities returns structured data from registries
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from poseidon.capabilities.validation import (
    validate_backtest_components,
    validate_live_components,
    warn_bias_risks,
)
from poseidon.capabilities.registry import ComponentCapability, get_all_capabilities


# --------------- Stub components for testing ---------------


class SafeComponent:
    name = "test_safe"
    supports_live = True
    supports_backtest = True
    bias_risk: list[str] = []
    stateful = False


class LiveUnsafeComponent:
    name = "test_live_unsafe"
    supports_live = False
    supports_backtest = True
    bias_risk: list[str] = []
    stateful = False


class BacktestUnsafeComponent:
    name = "test_no_backtest"
    supports_live = True
    supports_backtest = False
    bias_risk: list[str] = []
    stateful = False


class BiasedComponent:
    name = "test_biased"
    supports_live = True
    supports_backtest = True
    bias_risk = ["lookahead_normalization", "future_data_leakage"]
    stateful = False


# --------------- validate_live_components tests ---------------


class TestValidateLiveComponents:
    def test_rejects_live_unsafe(self):
        with pytest.raises(ValueError, match="test_live_unsafe"):
            validate_live_components([LiveUnsafeComponent()])

    def test_accepts_live_safe(self):
        validate_live_components([SafeComponent()])  # should not raise

    def test_empty_list_ok(self):
        validate_live_components([])  # should not raise

    def test_rejects_first_unsafe_in_mixed(self):
        with pytest.raises(ValueError, match="test_live_unsafe"):
            validate_live_components([SafeComponent(), LiveUnsafeComponent()])


# --------------- validate_backtest_components tests ---------------


class TestValidateBacktestComponents:
    def test_rejects_backtest_unsafe(self):
        with pytest.raises(ValueError, match="test_no_backtest"):
            validate_backtest_components([BacktestUnsafeComponent()])

    def test_accepts_backtest_safe(self):
        validate_backtest_components([SafeComponent()])  # should not raise


# --------------- warn_bias_risks tests ---------------


class TestWarnBiasRisks:
    def test_logs_warning_for_biased(self, caplog):
        with caplog.at_level(logging.WARNING, logger="poseidon.capabilities.validation"):
            warn_bias_risks([BiasedComponent()])
        assert "lookahead_normalization" in caplog.text
        assert "future_data_leakage" in caplog.text
        assert "test_biased" in caplog.text

    def test_no_warning_for_clean(self, caplog):
        with caplog.at_level(logging.WARNING, logger="poseidon.capabilities.validation"):
            warn_bias_risks([SafeComponent()])
        assert caplog.text == ""


# --------------- BacktestRunner enforcement tests ---------------


class TestBacktestRunnerEnforcement:
    """Test that BacktestRunner validates components at construction time."""

    def _make_strategy(self, supports_backtest=True, bias_risk=None):
        """Create a mock strategy with capability metadata."""
        from poseidon.strategies.base import StrategyType

        strategy = MagicMock()
        strategy.name = "mock_strategy"
        strategy.supports_backtest = supports_backtest
        strategy.supports_live = False
        strategy.bias_risk = bias_risk or []
        strategy.stateful = False
        strategy.strategy_type = StrategyType.RULE
        strategy.symbol = "BTCUSDT"
        strategy.market = "crypto_spot"
        strategy.interval = "1d"
        return strategy

    def _make_runner_deps(self):
        """Create mock dependencies for BacktestRunner."""
        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()
        return feature_engine, risk_engine, cost_model

    def test_rejects_backtest_unsafe_strategy(self):
        from poseidon.backtest.runner import BacktestRunner

        strategy = self._make_strategy(supports_backtest=False)
        fe, re, cm = self._make_runner_deps()

        with pytest.raises(ValueError, match="mock_strategy"):
            BacktestRunner(strategy, fe, re, cm)

    def test_warns_on_bias_risk_strategy(self, caplog):
        from poseidon.backtest.runner import BacktestRunner

        strategy = self._make_strategy(bias_risk=["future_data_leakage"])
        fe, re, cm = self._make_runner_deps()

        with caplog.at_level(logging.WARNING, logger="poseidon.capabilities.validation"):
            runner = BacktestRunner(strategy, fe, re, cm)

        assert runner is not None
        assert "future_data_leakage" in caplog.text

    def test_no_warning_for_clean_strategy(self, caplog):
        from poseidon.backtest.runner import BacktestRunner

        strategy = self._make_strategy()
        fe, re, cm = self._make_runner_deps()

        with caplog.at_level(logging.WARNING, logger="poseidon.capabilities.validation"):
            BacktestRunner(strategy, fe, re, cm)

        # No bias warning should appear
        assert "BIAS WARNING" not in caplog.text


# --------------- get_all_capabilities tests ---------------


class TestGetAllCapabilities:
    def test_returns_list_of_component_capabilities(self):
        """get_all_capabilities returns structured data with required fields."""
        # Patch all registries with minimal test data
        mock_feature_cls = type("MockFeature", (), {
            "name": "test_feat",
            "supports_backtest": True,
            "supports_live": False,
            "bias_risk": [],
            "stateful": False,
        })
        mock_strategy_cls = type("MockStrategy", (), {
            "name": "test_strat",
            "supports_backtest": True,
            "supports_live": True,
            "bias_risk": ["lookahead"],
            "stateful": False,
        })

        with patch("poseidon.capabilities.registry._get_feature_registry", return_value={"test_feat": mock_feature_cls}), \
             patch("poseidon.capabilities.registry._get_strategy_registry", return_value={"test_strat": mock_strategy_cls}), \
             patch("poseidon.capabilities.registry._get_model_registry", return_value={}), \
             patch("poseidon.capabilities.registry._get_rule_registry", return_value={}), \
             patch("poseidon.capabilities.registry._get_portfolio_registry", return_value={}):
            caps = get_all_capabilities()

        assert len(caps) == 2
        cap_names = {c.name for c in caps}
        assert "test_feat" in cap_names
        assert "test_strat" in cap_names

        # Verify fields
        for cap in caps:
            assert isinstance(cap, ComponentCapability)
            assert hasattr(cap, "name")
            assert hasattr(cap, "component_type")
            assert hasattr(cap, "supports_backtest")
            assert hasattr(cap, "supports_live")
            assert hasattr(cap, "bias_risk")
            assert hasattr(cap, "stateful")
