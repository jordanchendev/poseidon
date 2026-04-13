"""Tests for autoresearch immutability guard (AUTO-04, AUTO-06).

Verifies that:
- ContextVar defaults to False and can be toggled
- autoresearch_context() sets/resets the flag properly (even on exception)
- @autoresearch_guard prevents post-init __setattr__ on protected classes
- Normal (non-autoresearch) attribute mutation is not blocked
- FeatureEngine, BacktestRunner, RiskEngine are all protected
- VotingStrategy is NOT protected (strategy config is mutable)
"""

import pandas as pd
import pytest

from poseidon.autoresearch.guard import (
    ImmutabilityViolationError,
    _AUTORESEARCH_ACTIVE,
    autoresearch_context,
    autoresearch_guard,
)


# ---------------------------------------------------------------------------
# ContextVar basics
# ---------------------------------------------------------------------------


class TestContextVar:
    def test_defaults_to_false(self):
        assert _AUTORESEARCH_ACTIVE.get(False) is False

    def test_context_manager_sets_flag(self):
        assert _AUTORESEARCH_ACTIVE.get(False) is False
        with autoresearch_context():
            assert _AUTORESEARCH_ACTIVE.get(False) is True
        assert _AUTORESEARCH_ACTIVE.get(False) is False

    def test_context_manager_resets_on_exception(self):
        """Flag must reset even when body raises (try/finally)."""
        with pytest.raises(ValueError, match="boom"):
            with autoresearch_context():
                assert _AUTORESEARCH_ACTIVE.get(False) is True
                raise ValueError("boom")
        assert _AUTORESEARCH_ACTIVE.get(False) is False


# ---------------------------------------------------------------------------
# Guard decorator on a dummy class
# ---------------------------------------------------------------------------


@autoresearch_guard
class _DummyProtected:
    def __init__(self, value: int):
        self.value = value


class TestGuardDecorator:
    def test_init_works_during_autoresearch(self):
        """Construction must succeed even when flag is True."""
        with autoresearch_context():
            obj = _DummyProtected(42)
            assert obj.value == 42

    def test_setattr_blocked_after_init_during_autoresearch(self):
        """Post-init __setattr__ raises ImmutabilityViolationError."""
        with autoresearch_context():
            obj = _DummyProtected(42)
            with pytest.raises(ImmutabilityViolationError, match="Cannot modify"):
                obj.value = 99

    def test_setattr_allowed_outside_autoresearch(self):
        """Normal mode: mutation is not blocked."""
        obj = _DummyProtected(42)
        obj.value = 99
        assert obj.value == 99

    def test_setattr_allowed_before_autoresearch(self):
        """Object created before autoresearch can be mutated before autoresearch."""
        obj = _DummyProtected(10)
        obj.value = 20
        assert obj.value == 20

    def test_ar_prefix_allowed_during_autoresearch(self):
        """Internal _ar_ attributes can always be set."""
        with autoresearch_context():
            obj = _DummyProtected(42)
            obj._ar_custom = "internal"
            assert obj._ar_custom == "internal"


# ---------------------------------------------------------------------------
# Guard applied to real protected classes (AUTO-06)
# ---------------------------------------------------------------------------


class TestImmutabilityEnforcement:
    """AUTO-06: Protected classes raise ImmutabilityViolationError."""

    def test_immutability_enforcement_feature_engine(self):
        from poseidon.data.feature_engine import FeatureEngine

        with autoresearch_context():
            fe = FeatureEngine()
            with pytest.raises(ImmutabilityViolationError):
                fe.some_attr = "forbidden"

    def test_immutability_enforcement_backtest_runner(self):
        from poseidon.backtest.runner import BacktestRunner

        # BacktestRunner requires constructor args -- use minimal mocks
        with autoresearch_context():
            runner = BacktestRunner(
                strategy=None,
                feature_engine=None,
                risk_engine=None,
                cost_model=None,
            )
            with pytest.raises(ImmutabilityViolationError):
                runner.strategy = "forbidden"

    def test_immutability_enforcement_risk_engine(self):
        from poseidon.risk.engine import RiskEngine

        with autoresearch_context():
            re = RiskEngine()
            with pytest.raises(ImmutabilityViolationError):
                re._rules = []


# ---------------------------------------------------------------------------
# VotingStrategy is NOT guarded (AUTO-04 verification)
# ---------------------------------------------------------------------------


class TestStrategyConfigMutable:
    """AUTO-04: Only strategy JSON config is mutable during autoresearch."""

    def test_strategy_config_mutable(self):
        """VotingStrategy attributes remain settable during autoresearch."""
        from poseidon.strategies.voting_strategy import VotingStrategy

        config = {
            "name": "test_mutable",
            "symbol": "BTCUSDT",
            "market": "crypto_spot",
            "interval": "1h",
            "min_votes": 4,
            "position_pct": 0.08,
            "sub_signals": [
                {
                    "name": "momentum_short",
                    "strategy_type": "rule",
                    "conditions": {
                        "type": "comparison",
                        "left": {"type": "feature", "name": "close"},
                        "op": ">",
                        "right": {"type": "feature", "name": "sma_5"},
                    },
                },
                {
                    "name": "momentum_long",
                    "strategy_type": "rule",
                    "conditions": {
                        "type": "comparison",
                        "left": {"type": "feature", "name": "close"},
                        "op": ">",
                        "right": {"type": "feature", "name": "sma_10"},
                    },
                },
                {
                    "name": "ema_cross",
                    "strategy_type": "rule",
                    "conditions": {
                        "type": "comparison",
                        "left": {"type": "feature", "name": "ema_12"},
                        "op": ">",
                        "right": {"type": "feature", "name": "ema_26"},
                    },
                },
            ],
        }
        with autoresearch_context():
            vs = VotingStrategy(config=config)
            # Should NOT raise -- VotingStrategy is not @autoresearch_guard decorated
            vs.min_votes = 5
            assert vs.min_votes == 5


class TestMutableAttrsWhitelist:
    """Test mutable_attrs parameter on autoresearch_guard."""

    def test_mutable_attrs_allowed_during_autoresearch(self):
        """Whitelisted attrs can be mutated during autoresearch."""

        @autoresearch_guard(mutable_attrs=frozenset({"counter"}))
        class _WithMutable:
            def __init__(self):
                self.counter = 0
                self.config = "immutable"

        with autoresearch_context():
            obj = _WithMutable()
            obj.counter = 42  # should NOT raise
            assert obj.counter == 42
            with pytest.raises(ImmutabilityViolationError):
                obj.config = "changed"  # should raise

    def test_backtest_runner_funding_state_mutable(self):
        """BacktestRunner funding state attrs are whitelisted for mutation."""
        from poseidon.backtest.runner import BacktestRunner

        with autoresearch_context():
            runner = BacktestRunner(
                strategy=None,
                feature_engine=None,
                risk_engine=None,
                cost_model=None,
            )
            # These must NOT raise -- they are per-run runtime state
            runner._funding_costs_total = 1.23
            runner._funding_costs_by_trade = [0.5, 0.73]
            runner._last_funding_settlement = None
            assert runner._funding_costs_total == 1.23

            # Non-whitelisted attrs must still raise
            with pytest.raises(ImmutabilityViolationError):
                runner.strategy = "forbidden"


class TestBacktestRunnerInternalState:
    """Regression test for internal state writes during autoresearch."""

    def test_backtest_runner_can_store_portfolio_during_autoresearch(self):
        """BacktestRunner.run() should complete under guard and preserve portfolio access."""
        from poseidon.backtest.cost_model import COST_MODELS
        from poseidon.backtest.runner import BacktestRunner
        from poseidon.data.feature_engine import FeatureEngine
        from poseidon.risk.engine import RiskEngine
        from poseidon.strategies.voting_strategy import VotingStrategy

        idx = pd.date_range("2024-01-01", periods=220, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 220,
                "high": [101.0] * 220,
                "low": [99.0] * 220,
                "close": [100.0] * 220,
                "volume": [1000.0] * 220,
            },
            index=idx,
        )
        config = {
            "name": "test_voting",
            "symbol": "BTCUSDT",
            "market": "crypto_spot",
            "interval": "1h",
            "min_votes": 4,
            "position_pct": 0.08,
            "cooldown_bars": 0,
            "conviction_gap": 0,
            "sub_signals": [
                {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 6}, "threshold": 0},
                {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 12}, "threshold": 0},
                {
                    "type": "indicator_comparison",
                    "indicator_a": "ema",
                    "indicator_b": "ema",
                    "params": {"period_a": 7, "period_b": 26},
                    "direction": "above",
                },
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 8}, "threshold": 50},
                {"type": "indicator_above", "indicator": "macd_histogram", "params": {}, "threshold": 0},
                {"type": "bollinger_width_percentile", "params": {"period": 20, "lookback": 168}, "threshold": 0.2},
            ],
        }

        with autoresearch_context():
            runner = BacktestRunner(
                strategy=VotingStrategy(config=config),
                feature_engine=FeatureEngine(),
                risk_engine=RiskEngine(),
                cost_model=COST_MODELS["crypto_spot"],
            )
            result = runner.run(ohlcv)

        assert result.status == "completed"
        assert result.error_message is None
        assert runner.portfolio is not None
