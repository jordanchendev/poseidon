"""Real PostgreSQL serialization proof for same-fill cost corrections."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from sqlalchemy import func, select

from poseidon.decision_loop.outcome_accounting import FillCostService
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.outcome import FillCostRevision

NOW = datetime(2026, 1, 6, 16, tzinfo=UTC)


def _seed_fill(factory):
    with factory() as session, session.begin():
        order = OrderRecord(
            id=uuid.uuid4(),
            strategy_name="phase99-cost-race",
            symbol="PH99",
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
            account_scope="paper:phase99-cost-race",
            account_generation="phase99-cost-race-generation",
            client_order_ref=f"phase99-cost-race-{uuid.uuid4().hex}",
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
            broker_fill_id=f"phase99-cost-race-{uuid.uuid4().hex}",
            projection_status="applied",
        )
        session.add(fill)
        session.flush()
        return fill.id


def _append(session, fill_id, amount, barrier):
    barrier.wait(timeout=10)
    with session.begin():
        row = FillCostService(session).append_revision(
            order_fill_id=fill_id,
            reporting_currency="USD",
            cost_model_version="paper-cost-v1",
            components=[
                {
                    "component_type": "commission",
                    "native_amount": amount,
                    "native_currency": "USD",
                    "reporting_amount": amount,
                    "reporting_currency": "USD",
                    "classification": "actual",
                    "source": "broker-statement",
                }
            ],
        )
        result = row.id, row.content_sha256, row.input_sha256
    return result


def test_concurrent_fill_cost_corrections_form_one_immediate_predecessor_chain(
    phase99_session_factory,
    phase99_two_sessions,
    phase99_barrier,
):
    fill_id = _seed_fill(phase99_session_factory)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_append, session, fill_id, amount, phase99_barrier)
            for session, amount in zip(phase99_two_sessions, ("1", "2"), strict=True)
        ]
        inserted = [future.result(timeout=20) for future in futures]

    with phase99_session_factory() as session:
        rows = session.scalars(
            select(FillCostRevision)
            .where(FillCostRevision.order_fill_id == fill_id)
            .order_by(FillCostRevision.revision_no)
        ).all()
        assert [row.revision_no for row in rows] == [1, 2]
        assert (rows[1].previous_fill_cost_revision_id, rows[1].previous_revision_no) == (rows[0].id, 1)
        branches = session.execute(
            select(FillCostRevision.previous_fill_cost_revision_id, func.count())
            .where(FillCostRevision.previous_fill_cost_revision_id.is_not(None))
            .group_by(FillCostRevision.previous_fill_cost_revision_id)
            .having(func.count() > 1)
        ).all()
        assert branches == []

    replayed = []
    with phase99_session_factory() as session, session.begin():
        for amount in ("1", "2"):
            row = FillCostService(session).append_revision(
                order_fill_id=fill_id,
                reporting_currency="USD",
                cost_model_version="paper-cost-v1",
                components=[
                    {
                        "component_type": "commission",
                        "native_amount": amount,
                        "native_currency": "USD",
                        "reporting_amount": amount,
                        "reporting_currency": "USD",
                        "classification": "actual",
                        "source": "broker-statement",
                    }
                ],
            )
            replayed.append((row.id, row.content_sha256, row.input_sha256))
    assert set(replayed) == set(inserted)

