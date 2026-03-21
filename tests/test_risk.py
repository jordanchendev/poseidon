"""Tests for risk engine, rules, and virtual portfolio."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.risk.engine import RiskEngine
from poseidon.risk.portfolio import VirtualPortfolio
from poseidon.risk.rules import RULE_REGISTRY
from poseidon.risk.rules.confidence_threshold import ConfidenceThresholdRule
from poseidon.risk.rules.frequency import FrequencyRule
from poseidon.risk.rules.leverage_cap import LeverageCapRule
from poseidon.risk.rules.loss_limit import LossLimitRule
from poseidon.risk.rules.position_limit import PositionLimitRule
from poseidon.signals.schemas import (
    InstrumentType,
    Signal,
    SignalAction,
    SignalStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_signal(**overrides) -> Signal:
    """Create a Signal with sensible defaults for testing."""
    defaults = {
        "id": uuid4(),
        "symbol": "2330",
        "market": "tw_stock",
        "instrument": InstrumentType.SPOT,
        "action": SignalAction.LONG,
        "confidence": 0.8,
        "quantity_pct": 0.1,
        "signal_time": datetime.now(timezone.utc),
        "interval": "1d",
    }
    defaults.update(overrides)
    return Signal(**defaults)


def _fill_portfolio(portfolio: VirtualPortfolio, count: int) -> None:
    """Add *count* distinct open positions to a portfolio."""
    for i in range(count):
        sig = make_signal(symbol=f"SYM{i}", quantity_pct=0.05)
        portfolio.apply_signal(sig)


# ---------------------------------------------------------------------------
# TestRiskRules
# ---------------------------------------------------------------------------


class TestRiskRules:
    """Tests for individual risk rule checks."""

    # -- ConfidenceThresholdRule --

    def test_confidence_threshold_rejects_low(self) -> None:
        rule = ConfidenceThresholdRule()
        rule.load_params({"min_confidence": 0.5})
        signal = make_signal(confidence=0.3)
        portfolio = VirtualPortfolio()

        result = rule.check(signal, portfolio)

        assert result.passed is False
        assert result.rule_name == "confidence_threshold"
        assert "0.30" in (result.reason or "")
        assert "0.50" in (result.reason or "")

    def test_confidence_threshold_passes_high(self) -> None:
        rule = ConfidenceThresholdRule()
        rule.load_params({"min_confidence": 0.5})
        signal = make_signal(confidence=0.7)
        portfolio = VirtualPortfolio()

        result = rule.check(signal, portfolio)

        assert result.passed is True
        assert result.rule_name == "confidence_threshold"

    def test_confidence_threshold_passes_exact(self) -> None:
        rule = ConfidenceThresholdRule()
        rule.load_params({"min_confidence": 0.5})
        signal = make_signal(confidence=0.5)
        portfolio = VirtualPortfolio()

        result = rule.check(signal, portfolio)

        assert result.passed is True

    # -- PositionLimitRule --

    def test_position_limit_rejects_when_full(self) -> None:
        rule = PositionLimitRule()
        rule.load_params({"max_positions": 10})
        signal = make_signal()
        portfolio = VirtualPortfolio()
        _fill_portfolio(portfolio, 10)

        result = rule.check(signal, portfolio)

        assert result.passed is False
        assert result.rule_name == "position_limit"
        assert "10" in (result.reason or "")

    def test_position_limit_passes_when_below(self) -> None:
        rule = PositionLimitRule()
        rule.load_params({"max_positions": 10})
        signal = make_signal()
        portfolio = VirtualPortfolio()
        _fill_portfolio(portfolio, 5)

        result = rule.check(signal, portfolio)

        assert result.passed is True

    # -- LeverageCapRule --

    def test_leverage_cap_rejects_over_exposure(self) -> None:
        rule = LeverageCapRule()
        rule.load_params({"max_total_exposure": 1.0})
        portfolio = VirtualPortfolio()
        # Add positions totalling 0.8 exposure
        for i in range(8):
            portfolio.apply_signal(
                make_signal(symbol=f"S{i}", quantity_pct=0.1)
            )
        # New signal adds 0.3 => total 1.1 > 1.0
        signal = make_signal(symbol="NEW", quantity_pct=0.3)

        result = rule.check(signal, portfolio)

        assert result.passed is False
        assert result.rule_name == "leverage_cap"
        assert "1.10" in (result.reason or "")

    def test_leverage_cap_passes_under(self) -> None:
        rule = LeverageCapRule()
        rule.load_params({"max_total_exposure": 1.0})
        portfolio = VirtualPortfolio()
        for i in range(5):
            portfolio.apply_signal(
                make_signal(symbol=f"S{i}", quantity_pct=0.1)
            )
        signal = make_signal(symbol="NEW", quantity_pct=0.3)

        result = rule.check(signal, portfolio)

        assert result.passed is True

    # -- FrequencyRule (placeholder) --

    def test_frequency_rule_placeholder(self) -> None:
        rule = FrequencyRule()
        rule.load_params({})
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = rule.check(signal, portfolio)

        assert result.passed is True
        assert result.rule_name == "frequency"

    # -- LossLimitRule (placeholder) --

    def test_loss_limit_rule_placeholder(self) -> None:
        rule = LossLimitRule()
        rule.load_params({})
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = rule.check(signal, portfolio)

        assert result.passed is True
        assert result.rule_name == "loss_limit"

    # -- load_params defaults --

    def test_all_rules_load_empty_params(self) -> None:
        """Every rule must load default params without error."""
        for name, cls in RULE_REGISTRY.items():
            rule = cls()
            rule.load_params({})
            assert isinstance(rule, BaseRule), f"{name} is not a BaseRule"


# ---------------------------------------------------------------------------
# TestRiskEngine
# ---------------------------------------------------------------------------


class _AlwaysPassRule(BaseRule):
    """Stub rule that always passes."""

    name = "always_pass"

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        return RuleResult(passed=True, rule_name=self.name)

    def load_params(self, params: dict) -> None:
        pass


class _AlwaysRejectRule(BaseRule):
    """Stub rule that always rejects."""

    name = "always_reject"

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        return RuleResult(
            passed=False, rule_name=self.name, reason="forced rejection"
        )

    def load_params(self, params: dict) -> None:
        pass


class TestRiskEngine:
    """Tests for RiskEngine chain-of-responsibility."""

    def test_evaluate_passes_all_rules(self) -> None:
        engine = RiskEngine(rules=[_AlwaysPassRule(), _AlwaysPassRule()])
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = engine.evaluate(signal, portfolio)

        assert result.status == SignalStatus.PASSED
        assert result.reject_reason is None

    def test_evaluate_rejects_on_first_failure(self) -> None:
        engine = RiskEngine(rules=[_AlwaysPassRule(), _AlwaysRejectRule()])
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = engine.evaluate(signal, portfolio)

        assert result.status == SignalStatus.REJECTED
        assert result.reject_reason is not None
        assert "always_reject" in result.reject_reason

    def test_evaluate_short_circuits(self) -> None:
        """Second rule should never be called if first rejects."""
        spy_rule = MagicMock(spec=BaseRule)
        spy_rule.enabled = True
        spy_rule.check.return_value = RuleResult(
            passed=True, rule_name="spy"
        )

        engine = RiskEngine(rules=[_AlwaysRejectRule(), spy_rule])
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = engine.evaluate(signal, portfolio)

        assert result.status == SignalStatus.REJECTED
        spy_rule.check.assert_not_called()

    def test_empty_engine_passes(self) -> None:
        engine = RiskEngine(rules=[])
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = engine.evaluate(signal, portfolio)

        assert result.status == SignalStatus.PASSED

    def test_disabled_rule_skipped(self) -> None:
        reject_rule = _AlwaysRejectRule()
        reject_rule.enabled = False
        engine = RiskEngine(rules=[reject_rule])
        signal = make_signal()
        portfolio = VirtualPortfolio()

        result = engine.evaluate(signal, portfolio)

        assert result.status == SignalStatus.PASSED


# ---------------------------------------------------------------------------
# TestVirtualPortfolio
# ---------------------------------------------------------------------------


class TestVirtualPortfolio:
    """Tests for VirtualPortfolio position tracking."""

    def test_apply_long_signal(self) -> None:
        portfolio = VirtualPortfolio()
        signal = make_signal(
            symbol="2330", market="tw_stock", action=SignalAction.LONG
        )

        portfolio.apply_signal(signal)

        pos = portfolio.get_position("tw_stock", "2330")
        assert pos is not None
        assert pos.side == "long"
        assert pos.symbol == "2330"

    def test_apply_short_signal(self) -> None:
        portfolio = VirtualPortfolio()
        signal = make_signal(
            symbol="BTCUSDT", market="crypto_spot", action=SignalAction.SHORT
        )

        portfolio.apply_signal(signal)

        pos = portfolio.get_position("crypto_spot", "BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

    def test_apply_close_signal(self) -> None:
        portfolio = VirtualPortfolio()
        # Open a position
        portfolio.apply_signal(
            make_signal(symbol="2330", market="tw_stock", action=SignalAction.LONG)
        )
        assert portfolio.open_position_count == 1

        # Close it
        portfolio.apply_signal(
            make_signal(symbol="2330", market="tw_stock", action=SignalAction.CLOSE)
        )

        assert portfolio.open_position_count == 0
        assert portfolio.get_position("tw_stock", "2330") is None

    def test_close_nonexistent_position(self) -> None:
        portfolio = VirtualPortfolio()
        signal = make_signal(action=SignalAction.CLOSE)

        # Should not raise
        portfolio.apply_signal(signal)
        assert portfolio.open_position_count == 0

    def test_open_position_count(self) -> None:
        portfolio = VirtualPortfolio()
        symbols = ["2330", "AAPL", "BTCUSDT"]
        for sym in symbols:
            portfolio.apply_signal(make_signal(symbol=sym))

        assert portfolio.open_position_count == 3

    def test_total_exposure(self) -> None:
        portfolio = VirtualPortfolio()
        portfolio.apply_signal(make_signal(symbol="A", quantity_pct=0.2))
        portfolio.apply_signal(make_signal(symbol="B", quantity_pct=0.3))
        portfolio.apply_signal(make_signal(symbol="C", quantity_pct=0.15))

        assert abs(portfolio.total_exposure() - 0.65) < 1e-9

    def test_hold_signal_no_effect(self) -> None:
        portfolio = VirtualPortfolio()
        signal = make_signal(action=SignalAction.HOLD)

        portfolio.apply_signal(signal)

        assert portfolio.open_position_count == 0


# ---------------------------------------------------------------------------
# TestRuleRegistry
# ---------------------------------------------------------------------------


class TestRuleRegistry:
    """Tests for the RULE_REGISTRY mapping."""

    def test_registry_has_five_entries(self) -> None:
        assert len(RULE_REGISTRY) == 5

    def test_registry_keys(self) -> None:
        expected = {
            "position_limit",
            "loss_limit",
            "frequency",
            "confidence_threshold",
            "leverage_cap",
        }
        assert set(RULE_REGISTRY.keys()) == expected

    def test_all_inherit_base_rule(self) -> None:
        for name, cls in RULE_REGISTRY.items():
            assert issubclass(cls, BaseRule), f"{name} does not inherit BaseRule"


# ---------------------------------------------------------------------------
# TestPublicAPI
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Tests for risk package public API imports."""

    def test_full_import(self) -> None:
        from poseidon.risk import (
            RULE_REGISTRY,
            BaseRule,
            RiskEngine,
            RuleResult,
            VirtualPortfolio,
        )

        assert BaseRule is not None
        assert RuleResult is not None
        assert RiskEngine is not None
        assert VirtualPortfolio is not None
        assert RULE_REGISTRY is not None
