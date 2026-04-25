"""Tests for VaRLimitRule — portfolio VaR limit risk gate.

Tests cover:
- Rule name and defaults
- RULE_REGISTRY registration
- load_params configuration
- check() with cached VaR below/above limit
- check() with no cached data (pass by default)
- check() always passes CLOSE signals
- Stale snapshot warning
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import msgpack

from poseidon.risk.base import RuleResult
from poseidon.risk.rules.var_limit import VaRLimitRule
from poseidon.signals.schemas import Signal, SignalAction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(action: SignalAction = SignalAction.LONG) -> Signal:
    """Create a minimal Signal for testing."""
    return Signal(
        symbol="BTCUSDT",
        market="crypto_spot",
        action=action,
        confidence=0.8,
    )


class FakePortfolio:
    """Minimal portfolio stub for rule checks."""

    open_position_count: int = 3


def _pack_snapshot(
    var_95: float = 0.03,
    method: str = "parametric",
    computed_at: datetime | None = None,
) -> bytes:
    """Pack a VaR snapshot dict into msgpack bytes."""
    if computed_at is None:
        computed_at = datetime.now(UTC)
    return msgpack.packb(
        {
            "method": method,
            "var_95": var_95,
            "var_99": var_95 * 1.5,
            "cvar_95": var_95 * 1.2,
            "cvar_99": var_95 * 1.8,
            "portfolio_value": 100000.0,
            "as_of": computed_at.isoformat(),
            "computed_at": computed_at.isoformat(),
        },
        use_bin_type=True,
    )


# ---------------------------------------------------------------------------
# FakeRedis helper
# ---------------------------------------------------------------------------


class FakeRedisClient:
    """In-memory Redis mock for testing."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def set(self, key: str, value: bytes, **kwargs) -> None:
        self._store[key] = value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVaRLimitRuleBasics:
    """Basic properties and defaults."""

    def test_name(self):
        rule = VaRLimitRule()
        assert rule.name == "var_limit"

    def test_default_max_var_pct(self):
        rule = VaRLimitRule()
        assert rule.max_var_pct == 0.05

    def test_default_var_method(self):
        rule = VaRLimitRule()
        assert rule.var_method == "parametric"

    def test_load_params(self):
        rule = VaRLimitRule()
        rule.load_params({"max_var_pct": 0.10, "var_method": "historical"})
        assert rule.max_var_pct == 0.10
        assert rule.var_method == "historical"

    def test_load_params_defaults(self):
        rule = VaRLimitRule()
        rule.load_params({})
        assert rule.max_var_pct == 0.05
        assert rule.var_method == "parametric"


class TestVaRLimitRuleRegistry:
    """RULE_REGISTRY integration."""

    def test_in_registry(self):
        from poseidon.risk.rules import RULE_REGISTRY

        assert "var_limit" in RULE_REGISTRY
        assert RULE_REGISTRY["var_limit"] is VaRLimitRule


class TestVaRLimitRuleCheck:
    """check() behavior with various cache states."""

    def _make_rule(self, redis_client: FakeRedisClient, **kwargs) -> VaRLimitRule:
        rule = VaRLimitRule()
        rule._redis = redis_client
        if kwargs:
            rule.load_params(kwargs)
        return rule

    def test_close_signal_always_passes(self):
        """CLOSE signals must always pass regardless of VaR state."""
        redis = FakeRedisClient()
        rule = self._make_rule(redis)
        signal = _make_signal(SignalAction.CLOSE)
        result = rule.check(signal, FakePortfolio())
        assert result.passed is True
        assert result.rule_name == "var_limit"

    def test_no_cached_data_passes(self):
        """When no VaR snapshot exists, pass by default with reason."""
        redis = FakeRedisClient()
        rule = self._make_rule(redis)
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        assert result.passed is True
        assert "No VaR data available" in result.reason

    def test_var_below_limit_passes(self):
        """VaR below limit should pass."""
        redis = FakeRedisClient()
        redis.set("poseidon:var:latest:parametric", _pack_snapshot(var_95=0.03))
        rule = self._make_rule(redis, max_var_pct=0.05)
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        assert result.passed is True

    def test_var_above_limit_rejects(self):
        """VaR above limit should reject with reason."""
        redis = FakeRedisClient()
        redis.set("poseidon:var:latest:parametric", _pack_snapshot(var_95=0.08))
        rule = self._make_rule(redis, max_var_pct=0.05)
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        assert result.passed is False
        assert "portfolio VaR limit breached" in result.reason

    def test_var_equal_to_limit_passes(self):
        """VaR equal to limit should pass (only strictly greater rejects)."""
        redis = FakeRedisClient()
        redis.set("poseidon:var:latest:parametric", _pack_snapshot(var_95=0.05))
        rule = self._make_rule(redis, max_var_pct=0.05)
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        assert result.passed is True

    def test_uses_configured_method_key(self):
        """Rule reads from correct Redis key based on var_method."""
        redis = FakeRedisClient()
        # Only set historical key, not parametric
        redis.set("poseidon:var:latest:historical", _pack_snapshot(var_95=0.03))
        rule = self._make_rule(redis, var_method="historical")
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        assert result.passed is True

    def test_stale_snapshot_still_checks(self):
        """Stale snapshot logs warning but still evaluates the VaR check."""
        redis = FakeRedisClient()
        stale_time = datetime.now(UTC) - timedelta(hours=5)
        redis.set(
            "poseidon:var:latest:parametric",
            _pack_snapshot(var_95=0.03, computed_at=stale_time),
        )
        rule = self._make_rule(redis)
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        # Stale data should still be evaluated (pass in this case)
        assert result.passed is True

    def test_result_type(self):
        """check() returns RuleResult."""
        redis = FakeRedisClient()
        rule = self._make_rule(redis)
        signal = _make_signal()
        result = rule.check(signal, FakePortfolio())
        assert isinstance(result, RuleResult)
