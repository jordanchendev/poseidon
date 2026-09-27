"""Durable decision order intents stop before the broker boundary."""

import copy
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from poseidon.decision_loop.decisions import DecisionService
from poseidon.decision_loop.execution import (
    DecisionExecutionService,
    ExecutionConflictError,
    internal_state_watermark,
)
from poseidon.decision_loop.transactions import materialize_order_intents
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.base import Base
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import (
    create_decision,
    decision_inputs,
    executable_intent,
    manager,
    synthetic_policy,
    synthetic_reconciliation_policy,
    worker,
)

NOW = datetime(2026, 9, 26, 12, 45, tzinfo=UTC)
PRICE_2330 = {("tw_stock", "2330", "spot"): 100.0}


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'execution.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def _claimed(
    session,
    *,
    action="enter",
    target_weight=0.1,
    symbols=("2330",),
    reconciliation_status="matched",
    account="paper:pilot",
    market="tw_stock",
    instrument="spot",
    reconciliation=None,
):
    reconciliation = reconciliation or synthetic_reconciliation_policy(account_generation="generation-1")
    policy_json = synthetic_policy(
        account_scope=account,
        market=market,
        reconciliation=reconciliation,
    )
    universe = [{"symbol": symbol, "market": market, "instrument": instrument} for symbol in symbols]
    run, version, snapshot_ids = decision_inputs(
        session,
        policy=policy_json,
        non_evaluated_status="evaluated",
        universe=universe,
        market=market,
    )
    intents = [
        executable_intent(
            snapshot_id,
            symbol=symbol,
            market=market,
            instrument=instrument,
            action=action,
            target_weight=target_weight,
        )
        for snapshot_id, symbol in zip(snapshot_ids, symbols, strict=True)
    ]
    decision = create_decision(
        session,
        run.id,
        snapshot_ids,
        account_scope=account,
        principal=worker(account),
        original_json={
            "selected_evaluation_ids": snapshot_ids,
            "final_action": action,
            "order_intents": intents,
        },
        risk_snapshot_json={"hard_failures": [], "allowed_actions": [action]},
    )
    DecisionService(session).approve(
        decision.id,
        {"expected_revision": 1},
        principal=manager(scope=account),
        idempotency_key=f"approve-{decision.id}",
        now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
    )
    DecisionService(session).claim_execution(
        decision.id,
        2,
        principal=worker(account),
        now=NOW,
    )
    session.add(
        PaperBrokerAccount(
            account_scope=account,
            account_generation=reconciliation["account_generation"],
            opening_cash=reconciliation["opening_cash"],
            currency=reconciliation["currency"],
            created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
        )
    )
    session.add(
        AccountReconciliation(
            account_scope=account,
            account_generation=reconciliation["account_generation"],
            as_of=datetime(2026, 9, 26, 12, 44, tzinfo=UTC),
            broker_state_watermark="broker:0",
            internal_state_watermark="internal:0",
            broker_snapshot_sha256="a" * 64,
            broker_snapshot_json={},
            internal_snapshot_json={},
            difference_json={},
            policy_sha256=decision.policy_sha256,
            status=reconciliation_status,
        )
    )
    session.commit()
    return decision.id, version.id


def _reconcile_current(session, *, account="paper:pilot", generation="generation-1", watermark=None):
    previous = session.scalar(
        select(AccountReconciliation)
        .where(
            AccountReconciliation.account_scope == account,
            AccountReconciliation.account_generation == generation,
        )
        .order_by(AccountReconciliation.as_of.desc())
        .limit(1)
    )
    session.flush()
    session.add(
        AccountReconciliation(
            account_scope=account,
            account_generation=generation,
            as_of=datetime(2026, 9, 26, 12, 44, 30, tzinfo=UTC),
            broker_state_watermark=previous.broker_state_watermark,
            internal_state_watermark=watermark or internal_state_watermark(session, account, generation),
            broker_snapshot_sha256="c" * 64,
            broker_snapshot_json={},
            internal_snapshot_json={},
            difference_json={},
            policy_sha256=previous.policy_sha256,
            status="matched",
        )
    )


def test_two_symbol_intents_are_canonical_and_replay_keeps_first_economics(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session, symbols=("2330", "2317"))

    prices = {
        ("tw_stock", "2330", "spot"): 100.0,
        ("tw_stock", "2317", "spot"): 50.0,
    }
    first = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=prices,
        now=NOW,
    )
    replay = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=999_999.0,
        prices={key: 1.0 for key in prices},
        now=NOW,
    )

    assert replay == first
    with sessions() as session:
        orders = session.scalars(select(OrderRecord).order_by(OrderRecord.client_order_ref)).all()
        assert [order.symbol for order in orders] == ["2317", "2330"]
        assert [order.quantity for order in orders] == [200.0, 100.0]
        assert len({order.client_order_ref for order in orders}) == 2
        assert orders[0].client_order_ref.endswith("-0000")
        assert orders[1].client_order_ref.endswith("-0001")
        assert all(order.status == "pending_submit" for order in orders)
        assert all(order.order_origin == "decision" for order in orders)
        assert all(order.account_generation == "generation-1" for order in orders)
        assert [order.price for order in orders] == [50.0, 100.0]
        assert [order.intent_json["frozen_intent"]["symbol"] for order in orders] == ["2317", "2330"]
        assert [order.intent_json["economics"]["price"] for order in orders] == [50.0, 100.0]
        assert [order.intent_json["economics"]["account_nav"] for order in orders] == [100_000.0] * 2
        assert [order.intent_json["economics"]["materialized_quantity"] for order in orders] == [200.0, 100.0]
        assert all(not (order.signal_id is None and order.order_origin == "signal") for order in orders)
        assert all(len(order.intent_sha256) == 64 for order in orders)


@pytest.mark.parametrize(
    ("field", "value"),
    [("price", 101.0), ("quantity", 99.0), ("broker_mode", "live")],
)
def test_replay_rejects_corrupt_executable_columns(sessions, field, value):
    with sessions() as session:
        decision_id, _ = _claimed(session)
    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    with sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(response["order_ids"][0]))
        setattr(order, field, value)
        session.commit()

    with pytest.raises(ExecutionConflictError, match="executable content changed"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=999_999.0,
            prices={("tw_stock", "2330", "spot"): 1.0},
            now=NOW,
        )


def test_service_flushes_only_and_runner_commits_before_return(sessions, monkeypatch):
    with sessions() as session:
        decision_id, _ = _claimed(session)

    with sessions() as session:
        monkeypatch.setattr(session, "commit", lambda: (_ for _ in ()).throw(AssertionError("service committed")))
        response = DecisionExecutionService(session).materialize(
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )
        assert response["status"] == "pending_submit"
        session.rollback()

    committed = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    with sessions() as other_session:
        visible = other_session.scalars(select(OrderRecord).where(OrderRecord.decision_id == decision_id)).all()
    assert [str(row.id) for row in visible] == committed["order_ids"]


@pytest.mark.parametrize(
    "failure",
    ["missing_price", "no_account", "account_terms", "no_reconciliation", "mismatch", "policy_drift", "corrupt"],
)
def test_exposure_increase_fails_closed_without_partial_orders(sessions, failure):
    with sessions() as session:
        decision_id, version_id = _claimed(session)
        decision = session.get(DecisionRecord, decision_id)
        if failure == "no_account":
            session.query(PaperBrokerAccount).delete()
        elif failure == "account_terms":
            session.query(PaperBrokerAccount).one().currency = "USD"
        elif failure == "no_reconciliation":
            session.query(AccountReconciliation).delete()
        elif failure == "mismatch":
            session.query(AccountReconciliation).one().status = "mismatch"
        elif failure == "policy_drift":
            version = session.get(StrategyVersion, version_id)
            changed = copy.deepcopy(version.policy_json)
            changed["hard_limits"]["max_position_weight"] = 0.19
            session.execute(
                update(StrategyVersion)
                .where(StrategyVersion.id == version.id)
                .values(
                    policy_json=changed,
                    content_sha256=strategy_version_digest(version.config_json, changed, version.artifact_json),
                )
                .execution_options(synchronize_session=False)
            )
        elif failure == "corrupt":
            decision.final_json = copy.deepcopy(decision.final_json)
            decision.final_json["order_intents"][0]["target_weight"] = -1.0
        session.commit()

    prices = {} if failure == "missing_price" else PRICE_2330
    with pytest.raises(ExecutionConflictError):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=prices,
            now=NOW,
        )
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0


def test_current_hard_risk_failure_is_audited_without_an_order(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
        decision = session.get(DecisionRecord, decision_id)
        decision.risk_snapshot_json = copy.deepcopy(decision.risk_snapshot_json)
        decision.risk_snapshot_json["hard_failures"] = ["current limit failure"]
        session.commit()

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )

    assert response["status"] == "risk_blocked"
    with sessions() as session:
        decision = session.get(DecisionRecord, decision_id)
        assert decision.status == "risk_blocked"
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0
        assert session.query(DecisionEvent).filter_by(decision_id=decision_id, event_type="risk_blocked").count() == 1


def test_nav_delta_pending_reservation_and_reduce_without_matched_reconciliation(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(
            session,
            action="reduce",
            target_weight=0.05,
            reconciliation_status="mismatch",
        )
        session.add(
            PositionLot(
                account_scope="paper:pilot",
                account_generation="generation-1",
                market="tw_stock",
                symbol="2330",
                instrument="spot",
                side="long",
                opening_fill_id=uuid.uuid4(),
                opening_decision_id=decision_id,
                original_quantity=100.0,
                open_quantity=100.0,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 80.0},
                opened_at=datetime(2026, 9, 25, tzinfo=UTC),
            )
        )
        session.commit()

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )

    with sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(response["order_ids"][0]))
        lot = session.scalar(select(PositionLot).where(PositionLot.opening_decision_id == decision_id))
        assert order.action == "sell"
        assert order.quantity == 50.0
        assert order.reserved_quantity == 50.0
        assert order.reserved_cash_json["amount"] == 0.0
        assert lot.reserved_close_quantity == 50.0


def test_whole_share_floor(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session, target_weight=0.00199)

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    with sessions() as session:
        assert session.get(OrderRecord, uuid.UUID(response["order_ids"][0])).quantity == 1.0


def test_nonpositive_quantity_fails_closed(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session, target_weight=0.0001)
    with pytest.raises(ExecutionConflictError, match="positive quantity"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )


def test_reduce_floors_the_delta_instead_of_the_target(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session, action="reduce", target_weight=0.19)
        session.add(
            PositionLot(
                account_scope="paper:pilot",
                account_generation="generation-1",
                market="tw_stock",
                symbol="2330",
                instrument="spot",
                side="long",
                opening_fill_id=uuid.uuid4(),
                opening_decision_id=decision_id,
                original_quantity=2.0,
                open_quantity=2.0,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 80.0},
                opened_at=datetime(2026, 9, 25, tzinfo=UTC),
                created_at=datetime(2026, 9, 25, tzinfo=UTC),
                updated_at=datetime(2026, 9, 25, tzinfo=UTC),
            )
        )
        session.commit()

    with pytest.raises(ExecutionConflictError, match="positive quantity"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=1_000.0,
            prices=PRICE_2330,
            now=NOW,
        )
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0
        assert session.query(PositionLot).one().reserved_close_quantity == 0.0


def test_newer_cash_state_invalidates_matched_reconciliation(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
        session.add(
            PaperCashMovement(
                account_scope="paper:pilot",
                account_generation="generation-1",
                currency="TWD",
                amount=1.0,
                movement_type="adjustment",
                state_version=1,
                occurred_at=NOW,
                created_at=NOW,
            )
        )
        session.commit()

    with pytest.raises(ExecutionConflictError, match="predates outstanding state"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )


def test_internal_watermark_mismatch_invalidates_matched_reconciliation(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
        _reconcile_current(session, watermark="internal:stale")
        session.commit()

    with pytest.raises(ExecutionConflictError, match="internal watermark changed"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )


@pytest.mark.parametrize(
    ("state_kind", "message"),
    [
        ("order", "predates outstanding state"),
        ("projection_pending_fill", "predates outstanding state"),
        ("lot", "predates outstanding state"),
        ("paper_order", "predates outstanding state"),
        ("paper_fill", "predates outstanding state"),
        ("account_watermark", "broker watermark changed"),
    ],
)
def test_newer_account_state_invalidates_matched_reconciliation(sessions, state_kind, message):
    source_order_id = uuid.uuid4()
    paper_order_id = uuid.uuid4()
    before_reconciliation = datetime(2026, 9, 26, 12, 40, tzinfo=UTC)
    with sessions() as session:
        decision_id, _ = _claimed(session)
        if state_kind in {"order", "projection_pending_fill"}:
            session.add(
                OrderRecord(
                    id=source_order_id,
                    strategy_name="state-watermark-fixture",
                    symbol="2330",
                    market="tw_stock",
                    action="buy",
                    order_type="market",
                    target_weight=0.1,
                    quantity=1.0,
                    price=100.0,
                    side="long",
                    status="filled",
                    broker_mode="paper",
                    order_origin="decision",
                    decision_id=decision_id,
                    account_scope="paper:pilot",
                    account_generation="generation-1",
                    client_order_ref=f"DL-state-{source_order_id.hex}",
                    instrument="spot",
                    created_at=NOW if state_kind == "order" else before_reconciliation,
                    updated_at=NOW if state_kind == "order" else before_reconciliation,
                )
            )
            session.flush()
            if state_kind == "projection_pending_fill":
                session.add(
                    OrderFillRecord(
                        id=uuid.uuid4(),
                        order_id=source_order_id,
                        fill_price=100.0,
                        fill_quantity=1.0,
                        fill_time=NOW,
                        broker_fill_id=f"state-{source_order_id.hex}",
                        projection_status="projection_pending",
                        created_at=NOW,
                    )
                )
        elif state_kind == "lot":
            session.add(
                PositionLot(
                    account_scope="paper:pilot",
                    account_generation="generation-1",
                    market="tw_stock",
                    symbol="2330",
                    instrument="spot",
                    side="long",
                    opening_fill_id=uuid.uuid4(),
                    opening_decision_id=decision_id,
                    original_quantity=1.0,
                    open_quantity=1.0,
                    reserved_close_quantity=0.0,
                    cost_basis_json={"price": 100.0},
                    opened_at=NOW,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        elif state_kind in {"paper_order", "paper_fill"}:
            session.add(
                PaperBrokerOrder(
                    id=paper_order_id,
                    account_scope="paper:pilot",
                    account_generation="generation-1",
                    client_order_ref=f"PB-state-{paper_order_id.hex}",
                    broker_order_id=f"broker-{paper_order_id.hex}",
                    market="tw_stock",
                    symbol="2330",
                    instrument="spot",
                    action="buy",
                    side="long",
                    order_type="market",
                    quantity=1.0,
                    price=100.0,
                    status="filled",
                    state_version=1,
                    accepted_at=before_reconciliation,
                    created_at=before_reconciliation,
                    updated_at=NOW if state_kind == "paper_order" else before_reconciliation,
                )
            )
            session.flush()
            if state_kind == "paper_fill":
                session.add(
                    PaperBrokerFill(
                        id=uuid.uuid4(),
                        paper_broker_order_id=paper_order_id,
                        account_scope="paper:pilot",
                        account_generation="generation-1",
                        broker_fill_id=f"paper-fill-{paper_order_id.hex}",
                        market="tw_stock",
                        symbol="2330",
                        instrument="spot",
                        side="long",
                        fill_price=100.0,
                        fill_quantity=1.0,
                        fill_time=NOW,
                        state_version=1,
                        created_at=NOW,
                    )
                )
        else:
            session.query(PaperBrokerAccount).one().state_version = 1
        session.flush()
        _reconcile_current(session)
        session.commit()

    with pytest.raises(ExecutionConflictError, match=message):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )


def test_gross_risk_includes_unrelated_settled_exposure(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
        session.add(
            PositionLot(
                account_scope="paper:pilot",
                account_generation="generation-1",
                market="tw_stock",
                symbol="2317",
                instrument="spot",
                side="long",
                opening_fill_id=uuid.uuid4(),
                opening_decision_id=decision_id,
                original_quantity=950.0,
                open_quantity=950.0,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 90.0},
                opened_at=datetime(2026, 9, 25, tzinfo=UTC),
                created_at=datetime(2026, 9, 25, tzinfo=UTC),
                updated_at=datetime(2026, 9, 25, tzinfo=UTC),
            )
        )
        _reconcile_current(session)
        session.commit()

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices={**PRICE_2330, ("tw_stock", "2317", "spot"): 100.0},
        now=NOW,
    )
    assert response["status"] == "risk_blocked"
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0


def test_gross_risk_includes_unrelated_pending_reservations(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
        session.add(
            OrderRecord(
                id=uuid.uuid4(),
                strategy_name="unrelated-pending-decision",
                symbol="2317",
                market="tw_stock",
                action="buy",
                order_type="market",
                target_weight=0.95,
                quantity=950.0,
                price=100.0,
                side="long",
                status="pending_submit",
                broker_mode="paper",
                order_origin="decision",
                account_scope="paper:pilot",
                account_generation="generation-1",
                client_order_ref="DL-unrelated-pending-0000",
                instrument="spot",
                intent_json={"frozen_intent": {"action": "enter"}},
                intent_sha256="0" * 64,
                reserved_cash_json={"currency": "TWD", "amount": 0.0},
                reserved_quantity=950.0,
                reservation_status="reserved",
                reconciliation_status="pending",
                created_at=datetime(2026, 9, 26, 12, 43, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 43, tzinfo=UTC),
            )
        )
        _reconcile_current(session)
        session.commit()

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices={**PRICE_2330, ("tw_stock", "2317", "spot"): 100.0},
        now=NOW,
    )
    assert response["status"] == "risk_blocked"
    assert "max_gross_exposure" in response["failures"]
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0


def test_pending_reservation_reduces_nav_target_delta(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
        session.add(
            OrderRecord(
                id=uuid.uuid4(),
                strategy_name="prior-decision",
                symbol="2330",
                market="tw_stock",
                action="buy",
                order_type="market",
                target_weight=0.05,
                quantity=50.0,
                price=100.0,
                side="long",
                status="pending_submit",
                broker_mode="paper",
                order_origin="decision",
                account_scope="paper:pilot",
                account_generation="generation-1",
                client_order_ref="DL-prior-0000",
                instrument="spot",
                intent_json={"frozen_intent": {"action": "enter"}},
                intent_sha256="0" * 64,
                reserved_cash_json={"currency": "TWD", "amount": 5_000.0},
                reserved_quantity=50.0,
                reservation_status="reserved",
                reconciliation_status="pending",
                created_at=datetime(2026, 9, 26, 12, 43, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 43, tzinfo=UTC),
            )
        )
        _reconcile_current(session)
        session.commit()

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    with sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(response["order_ids"][0]))
        assert order.quantity == 50.0
        assert order.intent_json["economics"]["pending_increase_quantity"] == 50.0


def test_planned_cash_fails_closed_before_partial_intents(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session, symbols=("2330", "2317"))
        account = session.query(PaperBrokerAccount).one()
        account.state_version = 1
        account.updated_at = datetime(2026, 9, 26, 12, 40, tzinfo=UTC)
        reconciliation = session.query(AccountReconciliation).one()
        reconciliation.broker_state_watermark = "broker:1"
        session.add(
            PaperCashMovement(
                account_scope="paper:pilot",
                account_generation="generation-1",
                currency="TWD",
                amount=-85_000.0,
                movement_type="adjustment",
                state_version=1,
                occurred_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.commit()

    with pytest.raises(ExecutionConflictError, match="exceed available paper cash"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices={**PRICE_2330, ("tw_stock", "2317", "spot"): 100.0},
            now=NOW,
        )
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0


def test_perp_uses_owner_step_multiplier_and_semantics(sessions):
    reconciliation = synthetic_reconciliation_policy(
        account_generation="perp-generation-1",
        currency="USDT",
        perp_instrument_rules={
            "BTC-USDT": {
                "quantity_step": 0.001,
                "contract_multiplier": 0.01,
                "margin_semantics": "full_notional",
                "funding_semantics": "excluded",
            }
        },
    )
    with sessions() as session:
        decision_id, _ = _claimed(
            session,
            symbols=("BTC",),
            market="crypto_perp",
            instrument="BTC-USDT",
            reconciliation=reconciliation,
        )

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices={("crypto_perp", "BTC", "BTC-USDT"): 50_000.0},
        now=NOW,
    )
    with sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(response["order_ids"][0]))
        assert order.quantity == 20.0
        assert order.reserved_cash_json == {"currency": "USDT", "amount": 10_000.0}
        assert order.intent_json["economics"]["sizing_rules"]["margin_semantics"] == "full_notional"


def test_perp_reduction_keeps_decimal_delta_before_step_floor(sessions):
    reconciliation = synthetic_reconciliation_policy(
        account_generation="perp-generation-1",
        currency="USDT",
        perp_instrument_rules={
            "BTC-USDT": {
                "quantity_step": 0.1,
                "contract_multiplier": 1.0,
                "margin_semantics": "full_notional",
                "funding_semantics": "excluded",
            }
        },
    )
    with sessions() as session:
        decision_id, _ = _claimed(
            session,
            action="reduce",
            target_weight=0.1,
            symbols=("BTC",),
            market="crypto_perp",
            instrument="BTC-USDT",
            reconciliation=reconciliation,
        )
        session.add(
            PositionLot(
                account_scope="paper:pilot",
                account_generation="perp-generation-1",
                market="crypto_perp",
                symbol="BTC",
                instrument="BTC-USDT",
                side="long",
                opening_fill_id=uuid.uuid4(),
                opening_decision_id=decision_id,
                original_quantity=0.3,
                open_quantity=0.3,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 50.0},
                opened_at=datetime(2026, 9, 25, tzinfo=UTC),
            )
        )
        session.commit()

    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100.0,
        prices={("crypto_perp", "BTC", "BTC-USDT"): 50.0},
        now=NOW,
    )
    with sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(response["order_ids"][0]))
        assert order.quantity == 0.1
        assert order.intent_json["economics"]["delta_quantity"] == -0.1


def test_runner_rolls_back_a_post_flush_failure(sessions, monkeypatch):
    with sessions() as session:
        decision_id, _ = _claimed(session)

    original = DecisionExecutionService.materialize

    def fail_after_flush(service, *args, **kwargs):
        original(service, *args, **kwargs)
        raise RuntimeError("injected after flush")

    monkeypatch.setattr(DecisionExecutionService, "materialize", fail_after_flush)
    with pytest.raises(RuntimeError, match="injected"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0


def test_runner_rolls_back_close_lot_reservation_after_post_flush_failure(sessions, monkeypatch):
    with sessions() as session:
        decision_id, _ = _claimed(session, action="exit", target_weight=0.0)
        lot = PositionLot(
            account_scope="paper:pilot",
            account_generation="generation-1",
            market="tw_stock",
            symbol="2330",
            instrument="spot",
            side="long",
            opening_fill_id=uuid.uuid4(),
            opening_decision_id=decision_id,
            original_quantity=100.0,
            open_quantity=100.0,
            reserved_close_quantity=0.0,
            cost_basis_json={"price": 80.0},
            opened_at=datetime(2026, 9, 25, tzinfo=UTC),
        )
        session.add(lot)
        session.commit()
        lot_id = lot.id

    original = DecisionExecutionService.materialize

    def fail_after_flush(service, *args, **kwargs):
        original(service, *args, **kwargs)
        raise RuntimeError("injected after close reservation flush")

    monkeypatch.setattr(DecisionExecutionService, "materialize", fail_after_flush)
    with pytest.raises(RuntimeError, match="close reservation"):
        materialize_order_intents(
            sessions,
            decision_id,
            principal=worker(),
            account_nav=100_000.0,
            prices=PRICE_2330,
            now=NOW,
        )
    with sessions() as session:
        assert session.query(OrderRecord).filter_by(decision_id=decision_id).count() == 0
        assert session.get(PositionLot, lot_id).reserved_close_quantity == 0.0
