"""The stored protective-exit context shape (``OrderRecord.protective_context_json``)."""

import copy
import uuid

import pytest

from poseidon.orders.protective_context import ProtectiveContext, ProtectiveContextError

# Hex letters in every id, so an upper-cased copy is a non-canonical spelling of the same UUID.
LOT_A = str(uuid.UUID(int=0xA1))
LOT_B = str(uuid.UUID(int=0xA2))
DECISION = str(uuid.UUID(int=0xD3))
HOLDING_A = str(uuid.UUID(int=0xB4))
HOLDING_B = str(uuid.UUID(int=0xB5))
PREDECESSOR = str(uuid.UUID(int=0xC6))

BASE = {
    "account_scope": "paper:pilot",
    "account_generation": "generation-1",
    "identity": {"market": "tw_stock", "symbol": "2330", "instrument": "spot", "side": "long"},
    "origin": "stop_loss",
    "trigger_generation": "2026-09-26T12:45:00Z",
    "dedupe_sha256": "a" * 64,
}
ORDINARY = {**BASE, "source_lot_ids": [LOT_A, LOT_B], "source_decision_ids": [DECISION]}
LEGACY = {
    **BASE,
    "source_lot_ids": [],
    "source_decision_ids": [],
    "legacy_exception": True,
    "source_holding_ids": [HOLDING_A, HOLDING_B],
    "source_holding_quantities": {HOLDING_A: 5.0, HOLDING_B: 3},
    "source_holding_risk": {
        HOLDING_A: {"entry_price": 100.0, "stop_loss_pct": 0.1},
        HOLDING_B: {"entry_price": 90, "stop_loss_pct": 0.05},
    },
    "legacy_context_sha256": "b" * 64,
}


def successor(context, remaining):
    return {**context, "predecessor_order_ids": [PREDECESSOR], "remaining_source_quantities": remaining}


@pytest.mark.parametrize(
    "context",
    [
        ORDINARY,
        LEGACY,
        successor(ORDINARY, {LOT_A: 2.0, LOT_B: 0}),
        successor(LEGACY, {HOLDING_A: 5.0, HOLDING_B: 3}),
    ],
    ids=["ordinary", "legacy", "ordinary-successor", "legacy-successor"],
)
def test_stored_context_variants_parse(context):
    parsed = ProtectiveContext.parse(copy.deepcopy(context))

    assert parsed.legacy is (context.get("legacy_exception") is True)
    assert parsed.successor is ("predecessor_order_ids" in context)
    assert parsed.source_ids == (context["source_holding_ids"] if parsed.legacy else context["source_lot_ids"])


def _without(context, key):
    return {name: value for name, value in context.items() if name != key}


@pytest.mark.parametrize(
    "context",
    [
        None,
        [],
        {**ORDINARY, "unexpected": 1},
        _without(ORDINARY, "dedupe_sha256"),
        {**ORDINARY, "identity": {**ORDINARY["identity"], "venue": "twse"}},
        {**ORDINARY, "trigger_generation": ""},
        {**ORDINARY, "trigger_generation": " padded"},
        {**ORDINARY, "source_lot_ids": []},
        {**ORDINARY, "source_lot_ids": [LOT_A, LOT_A.upper()]},
        {**ORDINARY, "source_lot_ids": ["not-a-uuid"]},
        {**ORDINARY, "source_decision_ids": []},
        {**ORDINARY, "source_decision_ids": [LOT_B, LOT_A]},
        {**ORDINARY, "legacy_exception": True},
        _without(LEGACY, "legacy_context_sha256"),
        {**LEGACY, "legacy_exception": False},
        {**LEGACY, "legacy_context_sha256": None},
        {**LEGACY, "source_lot_ids": [LOT_A]},
        {**LEGACY, "source_holding_ids": [HOLDING_B, HOLDING_A]},
        {**LEGACY, "source_holding_ids": [HOLDING_A.upper(), HOLDING_B]},
        {**LEGACY, "source_holding_quantities": {HOLDING_A: 5.0}},
        {**LEGACY, "source_holding_quantities": {HOLDING_A: 0, HOLDING_B: 3}},
        {**LEGACY, "source_holding_quantities": {HOLDING_A: True, HOLDING_B: 3}},
        {**LEGACY, "source_holding_quantities": {HOLDING_A: float("inf"), HOLDING_B: 3}},
        {**LEGACY, "source_holding_risk": {**LEGACY["source_holding_risk"], HOLDING_A: {"entry_price": 100.0}}},
        {
            **LEGACY,
            "source_holding_risk": {
                **LEGACY["source_holding_risk"],
                HOLDING_A: {"entry_price": float("nan"), "stop_loss_pct": 0.1},
            },
        },
        {
            **LEGACY,
            "source_holding_risk": {
                **LEGACY["source_holding_risk"],
                HOLDING_A: {"entry_price": 100.0, "stop_loss_pct": 0.1, "mutable": True},
            },
        },
    ],
)
def test_malformed_contexts_fail_as_shape_errors(context):
    with pytest.raises(ProtectiveContextError) as caught:
        ProtectiveContext.parse(copy.deepcopy(context))

    assert caught.value.generation is False


@pytest.mark.parametrize(
    "context",
    [
        {**ORDINARY, "predecessor_order_ids": [PREDECESSOR]},
        {**ORDINARY, "remaining_source_quantities": {LOT_A: 1.0, LOT_B: 1.0}},
        {**successor(ORDINARY, {LOT_A: 1.0, LOT_B: 1.0}), "predecessor_order_ids": []},
        {**successor(ORDINARY, {LOT_A: 1.0, LOT_B: 1.0}), "predecessor_order_ids": [PREDECESSOR, LOT_A]},
        {**successor(ORDINARY, {LOT_A: 1.0, LOT_B: 1.0}), "predecessor_order_ids": [PREDECESSOR.upper()]},
        {**successor(ORDINARY, {LOT_A: 1.0, LOT_B: 1.0}), "predecessor_order_ids": None},
        successor(ORDINARY, {LOT_A: 1.0}),
        successor(ORDINARY, {LOT_A: 0, LOT_B: 0.0}),
        successor(ORDINARY, {LOT_A: -1.0, LOT_B: 2.0}),
        successor(LEGACY, {LOT_A: 5.0, LOT_B: 3}),
    ],
)
def test_bad_successor_links_fail_as_generation_errors(context):
    with pytest.raises(ProtectiveContextError) as caught:
        ProtectiveContext.parse(copy.deepcopy(context))

    assert caught.value.generation is True
