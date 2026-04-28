"""Unit tests for poseidon.risk.reject_reason (Phase 87 / TRUTH-03, D-13/D-14).

Validates the canonical 4-key reject_reason TypedDict shape and the
build_reject_reason factory that every write callsite must use.
"""

from __future__ import annotations

from poseidon.risk.reject_reason import RejectReason, build_reject_reason


def test_typeddict_shape_has_four_keys():
    """RejectReason TypedDict must define exactly 4 keys (D-13)."""
    annotations = RejectReason.__annotations__
    assert set(annotations.keys()) == {"check_name", "rule", "shortfall", "details"}


def test_factory_with_all_four_keys():
    payload = build_reject_reason(
        check_name="max_exposure",
        rule="Total exposure 110% exceeds 100% limit",
        shortfall={"projected_pct": 1.10, "limit_pct": 1.00},
        details="strategy=test_strategy, ts=2026-04-29T00:00:00Z",
    )
    assert payload["check_name"] == "max_exposure"
    assert payload["rule"] == "Total exposure 110% exceeds 100% limit"
    assert payload["shortfall"] == {"projected_pct": 1.10, "limit_pct": 1.00}
    assert payload["details"] == "strategy=test_strategy, ts=2026-04-29T00:00:00Z"


def test_factory_with_default_none_optional_fields():
    """shortfall and details default to None when not given."""
    payload = build_reject_reason(check_name="stop_loss", rule="No stop-loss configured")
    assert payload["check_name"] == "stop_loss"
    assert payload["rule"] == "No stop-loss configured"
    assert payload["shortfall"] is None
    assert payload["details"] is None


def test_factory_preserves_structured_shortfall():
    """Numeric shortfall dicts must round-trip exactly (JSONB roundtrip)."""
    shortfall = {
        "symbol": "ETHUSDT",
        "actual_leverage": 25,
        "max_leverage": 10,
    }
    payload = build_reject_reason(
        check_name="leverage_limit",
        rule="ETHUSDT leverage 25x exceeds max 10x",
        shortfall=shortfall,
    )
    assert payload["shortfall"] == shortfall
    # Different object identity is fine, but the dict equality must hold
    assert payload["shortfall"] is shortfall or payload["shortfall"] == shortfall


def test_factory_returns_dict_compatible_with_orderrecord_jsonb():
    """OrderRecord.reject_reason is JSONB. The factory output must be a
    plain dict[str, Any] and JSON-serializable (no special types)."""
    import json

    payload = build_reject_reason(
        check_name="broker_error",
        rule="broker_error: TimeoutError",
        details="Request timeout after 30s",
    )
    assert isinstance(payload, dict)
    serialized = json.dumps(payload)  # must not raise
    roundtripped = json.loads(serialized)
    assert roundtripped == payload
