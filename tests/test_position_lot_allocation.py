"""PostgreSQL FIFO projection, replay, isolation, and reservation proof."""

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.orm import sessionmaker

from poseidon.decision_loop.decisions import DecisionService
from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.manifest import ManifestService, content_sha256
from poseidon.decision_loop.recovery import RecoverySelector
from poseidon.models.data_manifest import DataManifest
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.fill_allocation import FillAllocation
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import (
    executable_intent,
    manager,
    manifest_request,
    synthetic_policy,
    synthetic_reconciliation_policy,
    worker,
)
from tests.test_execution_concurrency_postgres import ENGINE

NOW = datetime(2026, 9, 26, 12, 45, tzinfo=UTC)


@pytest.fixture
def seed():
    sessions = sessionmaker(ENGINE, expire_on_commit=False)
    scope = f"paper:phase98:lots:{uuid.uuid4().hex}"
    ids = {name: [] for name in ("decisions", "runs", "versions", "strategies", "manifests")}

    def fill(
        *,
        quantity=5,
        price=100,
        action="enter",
        side="long",
        symbol="2330",
        market="tw_stock",
        instrument="spot",
        account=None,
        generation="generation-1",
        time=NOW,
        fill_quantity=None,
        status="filled",
    ):
        account = scope if account is None else account
        reconciliation = synthetic_reconciliation_policy(account_generation=generation)
        if market == "crypto_perp":
            reconciliation.update(
                currency="USDT",
                perp_instrument_rules={
                    instrument: {
                        "quantity_step": 0.001,
                        "contract_multiplier": 2.0,
                        "margin_semantics": "full_notional",
                        "funding_semantics": "excluded",
                    }
                },
            )
        policy = synthetic_policy(account_scope=account, market=market, reconciliation=reconciliation)
        with sessions() as session, session.begin():
            strategy = StrategyRecord(
                name=f"lot-fixture-{uuid.uuid4().hex}",
                strategy_type="technical",
                symbol=symbol,
                market=market,
                interval="1d",
            )
            session.add(strategy)
            session.flush()
            config = {"fixture": strategy.name}
            version = StrategyVersion(
                strategy_id=strategy.id,
                version_no=1,
                config_json=config,
                policy_json=policy,
                artifact_json={},
                content_sha256=strategy_version_digest(config, policy, {}),
            )
            session.add(version)
            session.flush()
            request = manifest_request(market=market, symbol=symbol)
            request.update(account_scope=account, universe_id=policy["universe_id"])
            request["evidence"][0]["payload"]["marker"] = strategy.name
            request["evidence"][0]["content_sha256"] = content_sha256(request["evidence"][0]["payload"])
            manifest = ManifestService(session).freeze(request)
            universe = [{"symbol": symbol, "market": market, "instrument": instrument}]
            run = EvaluationService(session).evaluate_run(
                version.id,
                manifest.id,
                universe,
                [
                    {
                        **universe[0],
                        "status": "evaluated",
                        "reason_codes": [],
                        "recommendation_json": {"research_status": "not_required", "side": side},
                        "valid_until": "2026-09-26T15:00:00Z",
                    }
                ],
            )
            snapshot = session.scalar(select(EvaluationSnapshot).where(EvaluationSnapshot.evaluation_run_id == run.id))
            intent = executable_intent(
                str(snapshot.id),
                symbol=symbol,
                market=market,
                instrument=instrument,
                side=side,
                action=action,
                target_weight=0.0 if action == "exit" else 0.1,
            )
            decision = DecisionService(session).create_decision(
                run.id,
                principal=worker(account),
                account_scope=account,
                original_json={
                    "selected_evaluation_ids": [str(snapshot.id)],
                    "final_action": action,
                    "order_intents": [intent],
                },
                portfolio_snapshot_json={},
                risk_snapshot_json={"hard_failures": [], "allowed_actions": [action]},
                now=NOW - timedelta(minutes=45),
            )
            DecisionService(session).approve(
                decision.id,
                {"expected_revision": 1},
                principal=manager(scope=account),
                idempotency_key=f"lot-{decision.id}",
                now=NOW - timedelta(minutes=15),
            )
            DecisionService(session).claim_execution(decision.id, 2, principal=worker(account), now=NOW)
            if (
                session.scalar(
                    select(PaperBrokerAccount).where(
                        PaperBrokerAccount.account_scope == account, PaperBrokerAccount.account_generation == generation
                    )
                )
                is None
            ):
                session.add(
                    PaperBrokerAccount(
                        account_scope=account,
                        account_generation=generation,
                        opening_cash=reconciliation["opening_cash"],
                        currency=reconciliation["currency"],
                    )
                )
            multiplier = 2 if market == "crypto_perp" else 1
            economics = {"price": price, "materialized_quantity": quantity, "contract_multiplier": multiplier}
            if market == "crypto_perp":
                economics["sizing_rules"] = reconciliation["perp_instrument_rules"][instrument]
            stored = {"frozen_intent": intent, "economics": economics}
            order = OrderRecord(
                id=uuid.uuid4(),
                strategy_name=strategy.name,
                symbol=symbol,
                market=market,
                instrument=instrument,
                side=side,
                action="buy" if (side == "long") == (action in {"enter", "add"}) else "sell",
                order_type="market",
                target_weight=intent["target_weight"],
                quantity=quantity,
                price=price,
                status=status,
                broker_mode="paper",
                order_origin="decision",
                decision_id=decision.id,
                account_scope=account,
                account_generation=generation,
                execution_key=decision.execution_key,
                client_order_ref=f"DL-{decision.execution_key.hex}-0000",
                intent_json=stored,
                intent_sha256=content_sha256(stored),
                broker_order_id=f"PAPER-{uuid.uuid4().hex}",
                reserved_cash_json={"currency": reconciliation["currency"], "amount": 0},
                reserved_quantity=quantity,
                reservation_status="reserved",
                reconciliation_status="resolved",
                submit_attempted_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            )
            session.add(order)
            fill_record = OrderFillRecord(
                id=uuid.uuid4(),
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=price,
                fill_quantity=quantity if fill_quantity is None else fill_quantity,
                fill_time=time,
                projection_status="projection_pending",
                created_at=NOW,
            )
            session.add(fill_record)
            if action in {"reduce", "exit"}:
                remaining = quantity
                for lot in session.scalars(
                    select(PositionLot)
                    .filter_by(
                        account_scope=account,
                        account_generation=generation,
                        market=market,
                        symbol=symbol,
                        instrument=instrument,
                        side=side,
                    )
                    .order_by(PositionLot.opened_at, PositionLot.id)
                ):
                    reserved = min(remaining, lot.open_quantity - lot.reserved_close_quantity)
                    lot.reserved_close_quantity += reserved
                    remaining -= reserved
            for name, value in (
                ("decisions", decision.id),
                ("runs", run.id),
                ("versions", version.id),
                ("strategies", strategy.id),
                ("manifests", manifest.id),
            ):
                ids[name].append(value)
        return fill_record.id

    yield SimpleNamespace(sessions=sessions, fill=fill, account=scope)
    with sessions() as session, session.begin():
        own_orders = select(OrderRecord.id).where(OrderRecord.account_scope.startswith(scope))
        own_fills = select(OrderFillRecord.id).where(OrderFillRecord.order_id.in_(own_orders))
        session.execute(delete(FillAllocation).where(FillAllocation.closing_fill_id.in_(own_fills)))
        session.execute(delete(PositionLot).where(PositionLot.account_scope.startswith(scope)))
        session.execute(delete(OrderFillRecord).where(OrderFillRecord.order_id.in_(own_orders)))
        session.execute(delete(OrderRecord).where(OrderRecord.account_scope.startswith(scope)))
        session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id.in_(ids["decisions"])))
        session.execute(delete(DecisionRecord).where(DecisionRecord.id.in_(ids["decisions"])))
        session.execute(delete(EvaluationSnapshot).where(EvaluationSnapshot.evaluation_run_id.in_(ids["runs"])))
        session.execute(delete(EvaluationRun).where(EvaluationRun.id.in_(ids["runs"])))
        session.execute(delete(StrategyVersion).where(StrategyVersion.id.in_(ids["versions"])))
        session.execute(delete(StrategyRecord).where(StrategyRecord.id.in_(ids["strategies"])))
        session.execute(delete(DataManifest).where(DataManifest.id.in_(ids["manifests"])))
        session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope.startswith(scope)))


def project(seed, fill_id):
    from poseidon.positions.lots import apply_fill_projection

    return apply_fill_projection(seed.sessions, fill_id)


def test_fifo_open_add_partial_full_close_and_exact_replay(seed):
    first = seed.fill(quantity=3, price=100)
    second = seed.fill(quantity=4, price=120, action="add", time=NOW + timedelta(seconds=1))
    first_result = project(seed, first)
    assert project(seed, first) == first_result
    second_result = project(seed, second)
    assert first_result["lot_ids"] != second_result["lot_ids"]
    close = seed.fill(quantity=5, action="reduce")
    result = project(seed, close)
    assert project(seed, close) == result
    with seed.sessions() as session:
        lots = session.scalars(
            select(PositionLot)
            .where(PositionLot.account_scope == seed.account)
            .order_by(PositionLot.opened_at, PositionLot.id)
        ).all()
        assert [lot.open_quantity for lot in lots] == [0, 2]
        allocations = session.scalars(select(FillAllocation).where(FillAllocation.closing_fill_id == close)).all()
        assert sorted(allocation.quantity for allocation in allocations) == [2, 3]
        assert all(allocation.realized_cost_json["entry_cost"] > 0 for allocation in allocations)
    project(seed, seed.fill(quantity=2, action="exit"))
    with seed.sessions() as session:
        assert (
            sum(session.scalars(select(PositionLot.open_quantity).where(PositionLot.account_scope == seed.account)))
            == 0
        )


def test_fifo_equal_fill_time_uses_lot_uuid(seed):
    openings = [seed.fill(quantity=2), seed.fill(quantity=2, action="add")]
    for fill_id in reversed(openings):
        project(seed, fill_id)
    with seed.sessions() as session:
        expected = session.scalar(
            select(PositionLot.id).where(PositionLot.account_scope == seed.account).order_by(PositionLot.id)
        )
    close = seed.fill(quantity=1, action="reduce")
    project(seed, close)
    with seed.sessions() as session:
        assert (
            session.scalar(select(FillAllocation.position_lot_id).where(FillAllocation.closing_fill_id == close))
            == expected
        )


@pytest.mark.parametrize("dimension", ["account", "generation", "market", "symbol", "instrument", "side"])
def test_fifo_two_symbol_full_identity_isolation(seed, dimension):
    base = dict(quantity=2, market="crypto_perp", symbol="BTC", instrument="BTC-USDT")
    other = {
        "account": seed.account + ":other",
        "generation": "generation-2",
        "market": "tw_stock",
        "symbol": "ETH",
        "instrument": "BTC-USDC",
        "side": "short",
    }[dimension]
    first = seed.fill(**base)
    variant = {**base, dimension: other}
    second = seed.fill(**variant)
    project(seed, first)
    second_lot = project(seed, second)["lot_ids"][0]
    project(seed, seed.fill(**{**base, "quantity": 1, "action": "reduce"}))
    with seed.sessions() as session:
        assert session.get(PositionLot, uuid.UUID(second_lot)).open_quantity == 2


@pytest.mark.parametrize("side", ["long", "short"])
def test_fifo_correct_opposite_action_and_positive_allocation(seed, side):
    project(seed, seed.fill(quantity=3, side=side))
    close = seed.fill(quantity=2, side=side, action="reduce")
    project(seed, close)
    with seed.sessions() as session:
        fill = session.get(OrderFillRecord, close)
        order = session.get(OrderRecord, fill.order_id)
        assert order.action == ("sell" if side == "long" else "buy")
        assert session.scalar(select(FillAllocation.quantity).where(FillAllocation.closing_fill_id == close)) == 2


@pytest.mark.parametrize("corruption", ["quantity", "price", "time", "broker_fill_id"])
def test_fifo_applied_replay_content_conflict_is_quantity_neutral(seed, corruption):
    from poseidon.positions.lots import FillProjectionConflictError

    fill_id = seed.fill()
    project(seed, fill_id)
    with seed.sessions() as session, session.begin():
        fill = session.get(OrderFillRecord, fill_id)
        setattr(
            fill,
            {
                "quantity": "fill_quantity",
                "price": "fill_price",
                "time": "fill_time",
                "broker_fill_id": "broker_fill_id",
            }[corruption],
            {"quantity": 4, "price": 101, "time": NOW + timedelta(seconds=1), "broker_fill_id": "changed"}[corruption],
        )
    with pytest.raises(FillProjectionConflictError):
        project(seed, fill_id)
    with seed.sessions() as session:
        assert session.scalar(select(PositionLot.open_quantity).where(PositionLot.opening_fill_id == fill_id)) == 5
        order = session.get(OrderRecord, session.get(OrderFillRecord, fill_id).order_id)
        assert order.reconciliation_status == "required"


@pytest.mark.parametrize(
    "corruption", ["unknown_status", "missing_client", "missing_broker", "zero", "negative", "infinite", "nan"]
)
def test_fifo_invalid_projection_fails_without_partial_mutation(seed, corruption):
    from poseidon.positions.lots import FillProjectionConflictError

    fill_id = seed.fill()
    with seed.sessions() as session, session.begin():
        fill = session.get(OrderFillRecord, fill_id)
        order = session.get(OrderRecord, fill.order_id)
        if corruption == "unknown_status":
            fill.projection_status = "unknown"
        elif corruption == "missing_client":
            order.client_order_ref = None
        elif corruption == "missing_broker":
            fill.broker_fill_id = None
        else:
            fill.fill_quantity = {"zero": 0, "negative": -1, "infinite": float("inf"), "nan": float("nan")}[corruption]
    with pytest.raises(FillProjectionConflictError):
        project(seed, fill_id)
    with seed.sessions() as session:
        assert session.scalar(select(PositionLot.id).where(PositionLot.opening_fill_id == fill_id)) is None


def test_fifo_overclose_rolls_back_all_lots_and_marks_reconciliation_required(seed):
    from poseidon.positions.lots import FillProjectionConflictError

    project(seed, seed.fill(quantity=2))
    project(seed, seed.fill(quantity=2, action="add"))
    close = seed.fill(quantity=5, action="exit")
    with pytest.raises(FillProjectionConflictError):
        project(seed, close)
    with seed.sessions() as session:
        assert list(
            session.scalars(select(PositionLot.open_quantity).where(PositionLot.account_scope == seed.account))
        ) == [2, 2]
        assert session.scalar(select(FillAllocation.id).where(FillAllocation.closing_fill_id == close)) is None
        assert session.get(OrderFillRecord, close).projection_status == "projection_pending"
        assert (
            session.get(OrderRecord, session.get(OrderFillRecord, close).order_id).reconciliation_status == "required"
        )


def test_projection_pending_restart_and_flush_only_service(seed, monkeypatch):
    from poseidon.positions.lots import FillProjectionService

    fill_id = seed.fill()
    with seed.sessions() as session:
        assert any(
            action.operation == "project_fill" and action.persisted_id == fill_id
            for action in RecoverySelector(session).select(now=NOW)
        )
        monkeypatch.setattr(session, "commit", lambda: pytest.fail("allocator committed caller transaction"))
        FillProjectionService(session).apply(fill_id)
        session.rollback()
    result = project(seed, fill_id)
    assert project(seed, fill_id) == result
    with seed.sessions() as session:
        assert session.get(OrderFillRecord, fill_id).projection_status == "applied"
        order = session.get(OrderRecord, session.get(OrderFillRecord, fill_id).order_id)
        assert order.reservation_status == "released"


def test_fifo_partial_close_reservation_rebuild_preserves_pending_siblings(seed):
    project(seed, seed.fill(quantity=3))
    project(seed, seed.fill(quantity=7, action="add", time=NOW + timedelta(seconds=1)))
    partial = seed.fill(quantity=5, fill_quantity=2, action="reduce", status="partially_filled")
    sibling = seed.fill(quantity=3, action="reduce")
    project(seed, partial)
    with seed.sessions() as session:
        lots = session.scalars(
            select(PositionLot)
            .where(PositionLot.account_scope == seed.account)
            .order_by(PositionLot.opened_at, PositionLot.id)
        ).all()
        assert [(lot.open_quantity, lot.reserved_close_quantity) for lot in lots] == [(1, 1), (7, 5)]
    project(seed, sibling)
    with seed.sessions() as session:
        lots = session.scalars(
            select(PositionLot)
            .where(PositionLot.account_scope == seed.account)
            .order_by(PositionLot.opened_at, PositionLot.id)
        ).all()
        assert [(lot.open_quantity, lot.reserved_close_quantity) for lot in lots] == [(0, 0), (5, 3)]
        order = session.get(OrderRecord, session.get(OrderFillRecord, partial).order_id)
        assert order.reservation_status == "reserved"


def test_concurrent_postgres_close_fifo_never_double_allocates(seed):
    from poseidon.positions.lots import FillProjectionService

    project(seed, seed.fill(quantity=5))
    closing = [seed.fill(quantity=3, action="reduce"), seed.fill(quantity=2, action="reduce")]
    barrier = Barrier(2)

    def close(fill_id):
        with seed.sessions() as session, session.begin():
            backend_id = session.scalar(text("SELECT pg_backend_pid()"))
            barrier.wait(timeout=10)
            return backend_id, FillProjectionService(session).apply(fill_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20) for future in [pool.submit(close, fill_id) for fill_id in closing]]
    assert len(results) == 2
    assert results[0][0] != results[1][0]
    with seed.sessions() as session:
        assert session.scalar(select(PositionLot.open_quantity).where(PositionLot.account_scope == seed.account)) == 0
        rows = session.scalars(
            select(FillAllocation).join(PositionLot).where(PositionLot.account_scope == seed.account)
        ).all()
        assert sum(row.quantity for row in rows) == 5
        assert len(rows) == 2


def test_concurrent_postgres_same_fill_replay_has_one_lot(seed):
    fill_id = seed.fill()
    barrier = Barrier(2)

    def apply():
        barrier.wait(timeout=10)
        return project(seed, fill_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20) for future in [pool.submit(apply), pool.submit(apply)]]
    assert results[0] == results[1]
    with seed.sessions() as session:
        assert len(session.scalars(select(PositionLot).where(PositionLot.opening_fill_id == fill_id)).all()) == 1


@pytest.mark.parametrize("replay", ["opening", "closing"])
@pytest.mark.parametrize("corruption", ["open_quantity", "reservation"])
def test_fifo_replay_derived_lot_corruption_fails_closed(seed, replay, corruption):
    from poseidon.positions.lots import FillProjectionConflictError

    opening = seed.fill(quantity=5)
    project(seed, opening)
    closing = seed.fill(quantity=2, action="reduce")
    project(seed, closing)
    with seed.sessions() as session, session.begin():
        lot = session.scalar(select(PositionLot).where(PositionLot.opening_fill_id == opening))
        if corruption == "open_quantity":
            lot.open_quantity = 4  # Within DB bounds, inconsistent with the allocation of 2.
        else:
            lot.reserved_close_quantity = 1  # No remaining close order owns it.
    with pytest.raises(FillProjectionConflictError):
        project(seed, opening if replay == "opening" else closing)
    with seed.sessions() as session:
        assert session.scalar(select(FillAllocation.quantity).where(FillAllocation.closing_fill_id == closing)) == 2
        replay_fill = session.get(OrderFillRecord, opening if replay == "opening" else closing)
        assert session.get(OrderRecord, replay_fill.order_id).reconciliation_status == "required"


@pytest.mark.parametrize("corruption", ["sizing_rules", "sibling_intent", "sibling_action"])
def test_fifo_malformed_persisted_structure_records_required_marker(seed, corruption):
    from poseidon.positions.lots import FillProjectionConflictError

    if corruption == "sizing_rules":
        fill_id = seed.fill(market="crypto_perp", symbol="BTC", instrument="BTC-USDT")
        with seed.sessions() as session, session.begin():
            order = session.get(OrderRecord, session.get(OrderFillRecord, fill_id).order_id)
            changed = copy.deepcopy(order.intent_json)
            changed["economics"]["sizing_rules"] = []
            order.intent_json = changed
            order.intent_sha256 = content_sha256(changed)
    else:
        project(seed, seed.fill())
        fill_id = seed.fill(quantity=1, action="reduce")
        sibling = seed.fill(quantity=1, action="reduce")
        with seed.sessions() as session, session.begin():
            order = session.get(OrderRecord, session.get(OrderFillRecord, sibling).order_id)
            if corruption == "sibling_intent":
                order.intent_json = None
            else:
                changed = copy.deepcopy(order.intent_json)
                changed["frozen_intent"]["action"] = []
                order.intent_json = changed
    with pytest.raises(FillProjectionConflictError):
        project(seed, fill_id)
    with seed.sessions() as session:
        fill = session.get(OrderFillRecord, fill_id)
        assert fill.projection_status == "projection_pending"
        assert session.get(OrderRecord, fill.order_id).reconciliation_status == "required"
        assert session.scalar(select(FillAllocation.id).where(FillAllocation.closing_fill_id == fill_id)) is None


def test_fifo_missing_account_records_required_marker(seed):
    from poseidon.positions.lots import FillProjectionConflictError

    fill_id = seed.fill()
    with seed.sessions() as session, session.begin():
        session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == seed.account))
    with pytest.raises(FillProjectionConflictError):
        project(seed, fill_id)
    with seed.sessions() as session:
        fill = session.get(OrderFillRecord, fill_id)
        assert fill.projection_status == "projection_pending"
        assert session.get(OrderRecord, fill.order_id).reconciliation_status == "required"


def test_fifo_out_of_order_projection_pending_opening_defers_close(seed):
    from poseidon.positions.lots import FillProjectionConflictError

    earlier = seed.fill(quantity=2, time=NOW)
    later = seed.fill(quantity=2, action="add", time=NOW + timedelta(seconds=1))
    project(seed, later)
    close = seed.fill(quantity=1, action="reduce", time=NOW + timedelta(seconds=2))
    with pytest.raises(FillProjectionConflictError):
        project(seed, close)
    with seed.sessions() as session:
        assert session.get(OrderFillRecord, close).projection_status == "projection_pending"
        assert session.scalar(select(FillAllocation.id).where(FillAllocation.closing_fill_id == close)) is None
    earlier_lot = project(seed, earlier)["lot_ids"][0]
    project(seed, close)
    with seed.sessions() as session:
        assert (
            str(session.scalar(select(FillAllocation.position_lot_id).where(FillAllocation.closing_fill_id == close)))
            == earlier_lot
        )
