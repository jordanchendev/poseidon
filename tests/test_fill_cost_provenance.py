"""Revisioned fill-cost provenance and Decimal accounting contracts."""

import ast
import inspect
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from poseidon.decision_loop.manifest import ValidationError
from poseidon.decision_loop.outcome_accounting import FillCostService, OutcomeAccounting
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.outcome import FillCostComponent, FillCostRevision

NOW = datetime(2026, 1, 6, 16, tzinfo=UTC)


@pytest.fixture
def cost_session(phase99_session_factory):
    session = phase99_session_factory()
    transaction = session.begin()
    try:
        yield session
    finally:
        if transaction.is_active:
            transaction.rollback()
        session.close()


def _fill(session, *, symbol="PH99"):
    order = OrderRecord(
        id=uuid.uuid4(),
        strategy_name="phase99-cost",
        symbol=symbol,
        market="tw_stock",
        action="sell",
        order_type="market",
        target_weight=0,
        quantity=1,
        price=None,
        side="long",
        status="filled",
        broker_mode="paper",
        order_origin="manual",
        account_scope="paper:phase99-cost",
        account_generation="phase99-cost-generation",
        client_order_ref=f"phase99-cost-{uuid.uuid4().hex}",
        instrument="spot",
    )
    session.add(order)
    session.flush()
    fill = OrderFillRecord(
        id=uuid.uuid4(),
        order_id=order.id,
        fill_price=110,
        fill_quantity=1,
        fill_time=NOW,
        broker_fill_id=f"phase99-fill-{uuid.uuid4().hex}",
        projection_status="applied",
    )
    session.add(fill)
    session.flush()
    return fill


def _component(
    component_type,
    amount,
    *,
    classification="actual",
    native_currency="USD",
    reporting_currency="USD",
    source=None,
    fx_source=None,
    fx_rate=None,
    reason=None,
):
    return {
        "component_type": component_type,
        "native_amount": amount,
        "native_currency": native_currency,
        "reporting_amount": amount,
        "reporting_currency": reporting_currency,
        "classification": classification,
        "source": source or f"phase99:{component_type}",
        "fx_source": fx_source,
        "fx_rate": fx_rate,
        "fx_as_of": NOW if fx_rate is not None else None,
        "reason": reason,
    }


def test_fill_cost_replay_correction_and_all_component_provenance(cost_session):
    fill = _fill(cost_session)
    components = [
        _component("commission", "1"),
        _component("tax", "0.5", classification="estimated"),
        _component("funding", "-0.25"),
        _component("borrow", "0.125"),
        _component(
            "fx",
            "1.1",
            native_currency="EUR",
            reporting_currency="USD",
            fx_source="wm-close-v1",
            fx_rate="1.1",
        ),
        _component("other", None, classification="not_applicable"),
        _component(
            "other",
            None,
            classification="unavailable",
            source="phase99:other-unavailable",
            reason="not posted",
        ),
    ]
    service = FillCostService(cost_session)
    first = service.append_revision(
        order_fill_id=fill.id,
        reporting_currency="USD",
        cost_model_version="paper-cost-v1",
        components=components,
    )
    replay = service.append_revision(
        order_fill_id=fill.id,
        reporting_currency="USD",
        cost_model_version="paper-cost-v1",
        components=list(reversed(components)),
    )
    assert (replay.id, replay.content_sha256) == (first.id, first.content_sha256)
    assert first.revision_no == 1

    stored = cost_session.scalars(
        select(FillCostComponent)
        .where(FillCostComponent.fill_cost_revision_id == first.id)
        .order_by(FillCostComponent.component_type)
    ).all()
    assert {row.component_type for row in stored} == {"commission", "tax", "funding", "borrow", "fx", "other"}
    assert all(row.native_amount is None or isinstance(row.native_amount, Decimal) for row in stored)
    assert next(row for row in stored if row.component_type == "funding").reporting_amount == Decimal("-0.25")
    assert next(row for row in stored if row.component_type == "tax").classification == "estimated"
    fx = next(row for row in stored if row.component_type == "fx")
    assert (fx.fx_source, fx.fx_rate, fx.fx_as_of) == ("wm-close-v1", Decimal("1.1"), NOW)

    corrected = [dict(item) for item in components]
    corrected[0]["native_amount"] = corrected[0]["reporting_amount"] = "2"
    second = service.append_revision(
        order_fill_id=fill.id,
        reporting_currency="USD",
        cost_model_version="paper-cost-v1",
        components=corrected,
    )
    assert (second.revision_no, second.previous_fill_cost_revision_id, second.previous_revision_no) == (
        2,
        first.id,
        1,
    )
    assert cost_session.query(FillCostRevision).filter_by(fill_key_sha256=first.fill_key_sha256).count() == 2


def test_cross_currency_cost_requires_complete_frozen_fx(cost_session):
    fill = _fill(cost_session)
    missing_fx = _component(
        "commission",
        "1",
        native_currency="EUR",
        reporting_currency="USD",
    )
    with pytest.raises(ValidationError, match="FX"):
        FillCostService(cost_session).append_revision(
            order_fill_id=fill.id,
            reporting_currency="USD",
            cost_model_version="paper-cost-v1",
            components=[missing_fx],
        )


def test_cost_and_pnl_implementation_is_decimal_and_caller_transaction_owned():
    import poseidon.decision_loop.outcome_accounting as accounting

    source = inspect.getsource(accounting)
    tree = ast.parse(source)
    assert "Decimal" in source
    assert "float(" not in source
    for service in (FillCostService, OutcomeAccounting):
        service_tree = ast.parse(inspect.getsource(service))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"commit", "rollback"}
            for node in ast.walk(service_tree)
        )
