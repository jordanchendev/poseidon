"""Real PostgreSQL row-lock proof for decision execution claims."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, delete, event, select, text
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionService
from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.execution import ExecutionConflictError, internal_state_watermark
from poseidon.decision_loop.manifest import ManifestService, content_sha256
from poseidon.decision_loop.releases import ReleaseService, require_active_pointer
from poseidon.decision_loop.transactions import materialize_order_intents
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.data_manifest import DataManifest
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.position_lot import PositionLot
from poseidon.models.signal import SignalRecord
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import (
    executable_intent,
    manifest_request,
    synthetic_policy,
    synthetic_reconciliation_policy,
)

DATABASE_URL = os.environ["POSEIDON_DATABASE_URL"]
ENGINE = create_engine(DATABASE_URL)
if ENGINE.dialect.name != "postgresql":
    raise RuntimeError("test_execution_concurrency_postgres.py requires PostgreSQL")

CLAIM_TIME = datetime(2026, 9, 26, 12, 45, tzinfo=UTC)


@dataclass(frozen=True)
class ClaimSeed:
    account: str
    decision_ids: tuple[uuid.UUID, uuid.UUID]
    strategy_ids: tuple[uuid.UUID, uuid.UUID]
    version_ids: tuple[uuid.UUID, uuid.UUID]
    manifest_id: uuid.UUID
    run_ids: tuple[uuid.UUID, uuid.UUID]

    @property
    def worker(self):
        return AuthPrincipal("service:phase98-claim", frozenset({"decision-worker"}), frozenset({self.account}))


@pytest.fixture
def claim_seed():
    marker = uuid.uuid4().hex
    account = f"paper:phase98:claim:{marker}"
    universe_id = f"phase98-claim-{marker}"
    strategy_ids = []
    version_ids = []
    run_ids = []
    decision_ids = []
    with Session(ENGINE) as session:
        request = manifest_request()
        request["account_scope"] = account
        request["universe_id"] = universe_id
        request["evidence"][0]["payload"]["marker"] = marker
        request["evidence"][0]["content_sha256"] = content_sha256(request["evidence"][0]["payload"])
        manifest = ManifestService(session).freeze(request)

        for index in range(2):
            strategy = StrategyRecord(
                name=f"phase98-claim-{marker}-{index}",
                strategy_type="technical",
                symbol="2330",
                market="tw_stock",
                interval="1d",
            )
            session.add(strategy)
            session.flush()
            policy_json = synthetic_policy(
                account_scope=account,
                universe_id=universe_id,
                reconciliation=synthetic_reconciliation_policy(account_generation=f"phase98-claim-generation-{marker}"),
            )
            version = StrategyVersion(
                strategy_id=strategy.id,
                version_no=1,
                config_json={"marker": marker, "index": index},
                policy_json=policy_json,
                artifact_json={},
                content_sha256=strategy_version_digest(
                    {"marker": marker, "index": index},
                    policy_json,
                    {},
                ),
            )
            session.add(version)
            session.flush()
            session.execute(text("ALTER TABLE strategies DISABLE TRIGGER trg_strategies_release_authority"))
            try:
                strategy.active_version_id = version.id
                session.flush()
            finally:
                session.execute(text("ALTER TABLE strategies ENABLE TRIGGER trg_strategies_release_authority"))
            universe = [{"symbol": "2330", "market": "tw_stock", "instrument": "spot"}]
            run = EvaluationService(session).evaluate_run(
                version.id,
                manifest.id,
                universe,
                [
                    {
                        **universe[0],
                        "status": "evaluated",
                        "recommendation_json": {"research_status": "not_required", "side": "long"},
                        "reason_codes": [],
                        "valid_until": "2026-09-26T15:00:00Z",
                    }
                ],
            )
            session.flush()
            snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=run.id).one()
            decision = DecisionService(session).create_decision(
                run.id,
                principal=AuthPrincipal(
                    "service:phase98-create",
                    frozenset({"decision-worker"}),
                    frozenset({account}),
                ),
                account_scope=account,
                original_json={
                    "selected_evaluation_ids": [str(snapshot.id)],
                    "final_action": "enter",
                    "order_intents": [executable_intent(str(snapshot.id))],
                },
                portfolio_snapshot_json={"cash": 100000.0, "positions": []},
                risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
                now=datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
            )
            DecisionService(session).approve(
                decision.id,
                {"expected_revision": 1},
                principal=AuthPrincipal(
                    "human:phase98-manager",
                    frozenset({"portfolio_manager"}),
                    frozenset({account}),
                ),
                idempotency_key=f"phase98-claim-approve-{index}",
                now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
            )
            strategy_ids.append(strategy.id)
            version_ids.append(version.id)
            run_ids.append(run.id)
            decision_ids.append(decision.id)
        value = ClaimSeed(
            account,
            tuple(decision_ids),
            tuple(strategy_ids),
            tuple(version_ids),
            manifest.id,
            tuple(run_ids),
        )
        session.commit()

    yield value

    with Session(ENGINE) as session:
        session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id.in_(value.decision_ids)))
        session.execute(delete(DecisionRecord).where(DecisionRecord.id.in_(value.decision_ids)))
        session.execute(delete(EvaluationSnapshot).where(EvaluationSnapshot.evaluation_run_id.in_(value.run_ids)))
        session.execute(delete(EvaluationRun).where(EvaluationRun.id.in_(value.run_ids)))
        session.execute(delete(SignalRecord).where(SignalRecord.strategy_id.in_(value.strategy_ids)))
        session.execute(text("ALTER TABLE strategies DISABLE TRIGGER trg_strategies_release_authority"))
        try:
            for strategy_id in value.strategy_ids:
                strategy = session.get(StrategyRecord, strategy_id)
                if strategy is not None:
                    strategy.active_version_id = None
            session.flush()
        finally:
            session.execute(text("ALTER TABLE strategies ENABLE TRIGGER trg_strategies_release_authority"))
        session.execute(delete(StrategyVersion).where(StrategyVersion.id.in_(value.version_ids)))
        session.execute(delete(DataManifest).where(DataManifest.id == value.manifest_id))
        session.execute(delete(StrategyRecord).where(StrategyRecord.id.in_(value.strategy_ids)))
        session.commit()


def test_concurrent_named_claim_has_one_key_revision_and_event(claim_seed):
    barrier = Barrier(2)

    def claim():
        with Session(ENGINE) as session:
            barrier.wait(timeout=10)
            try:
                result = DecisionService(session).claim_execution(
                    claim_seed.decision_ids[0],
                    2,
                    principal=claim_seed.worker,
                    now=CLAIM_TIME,
                )
                session.commit()
                return "claimed", result["execution_key"]
            except DecisionConflictError as error:
                session.rollback()
                return "conflict", str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20) for future in [pool.submit(claim), pool.submit(claim)]]

    assert sorted(status for status, _ in results) == ["claimed", "conflict"]
    with Session(ENGINE) as session:
        decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
        assert decision.status == "execution_claimed"
        assert decision.revision == 3
        assert decision.execution_key is not None
        events = session.query(DecisionEvent).filter_by(
            decision_id=decision.id,
            event_type="execution_claimed",
        )
        assert events.count() == 1
        assert events.one().expected_revision == 2


def test_claim_fails_closed_after_pointer_is_cleared(claim_seed):
    with Session(ENGINE) as session, session.begin():
        session.execute(text("ALTER TABLE strategies DISABLE TRIGGER trg_strategies_release_authority"))
        try:
            session.get(StrategyRecord, claim_seed.strategy_ids[0]).active_version_id = None
            session.flush()
        finally:
            session.execute(text("ALTER TABLE strategies ENABLE TRIGGER trg_strategies_release_authority"))

    with Session(ENGINE) as session, session.begin(), pytest.raises(DecisionConflictError, match="active pointer"):
        DecisionService(session).claim_execution(
            claim_seed.decision_ids[0], 2, principal=claim_seed.worker, now=CLAIM_TIME
        )

    with Session(ENGINE) as session:
        decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
        assert decision.status == "approved"
        assert decision.execution_key is None


@pytest.mark.parametrize("mode", ["legacy", "shadow"])
def test_pause_with_fresh_signal_cannot_reach_ordinary_legacy_writer(monkeypatch, claim_seed, mode):
    from poseidon.signals.repository import SignalRepository
    from poseidon.workers import cpu_tasks

    with Session(ENGINE) as session, session.begin():
        _pause(session, claim_seed, key=f"pause-fresh-signal-{mode}")
        session.add(
            SignalRecord(
                strategy_id=claim_seed.strategy_ids[0],
                symbol="2330",
                market="tw_stock",
                instrument="spot",
                action="long",
                confidence=0.9,
                quantity_pct=0.1,
                signal_time=CLAIM_TIME,
                interval="1d",
                status="passed",
            )
        )

    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        SimpleNamespace(decision_loop_execution_mode=mode),
    )
    monkeypatch.setattr(
        SignalRepository,
        "latest_passed",
        lambda *args, **kwargs: pytest.fail("paused ordinary writer read a fresh signal"),
    )
    with Session(ENGINE) as session:
        baseline_orders = session.query(OrderRecord).count()

    result = cpu_tasks.portfolio_monthly_rebalance.run()

    assert result == {"skipped": "release_authority_required"}
    with Session(ENGINE) as session:
        assert session.query(OrderRecord).count() == baseline_orders


def _release_manager(claim_seed):
    return AuthPrincipal(
        "human:phase100-release-manager",
        frozenset({"release_manager"}),
        frozenset({claim_seed.account}),
    )


def _pause(session, claim_seed, *, key):
    return ReleaseService(session).pause(
        claim_seed.strategy_ids[0],
        expected_revision=0,
        expected_active_version_id=claim_seed.version_ids[0],
        principal=_release_manager(claim_seed),
        idempotency_key=key,
        reason="prove ordinary execution pause barrier",
    )


def _delete_pause_event(session, strategy_id):
    session.execute(text("ALTER TABLE strategy_release_events DISABLE TRIGGER trg_strategy_release_events_append_only"))
    try:
        session.execute(delete(StrategyReleaseEvent).where(StrategyReleaseEvent.strategy_id == strategy_id))
    finally:
        session.execute(
            text("ALTER TABLE strategy_release_events ENABLE TRIGGER trg_strategy_release_events_append_only")
        )


def test_pause_wins_strategy_lock_before_claim_without_execution_side_effects(monkeypatch, claim_seed):
    pause_locked = Event()
    claim_started = Event()
    claim_lock_attempted = Event()
    release_pause = Event()
    original_lock = ReleaseService._lock_strategy

    def hold_pause_lock(service, strategy_id):
        strategy = original_lock(service, strategy_id)
        if service.session.info.get("phase100_pause_winner"):
            pause_locked.set()
            assert claim_started.wait(timeout=10)
            assert release_pause.wait(timeout=10)
        return strategy

    def observe_claim_lock(connection, cursor, statement, parameters, context, executemany):
        if connection.info.get("phase100_claim_waiter") and "FOR UPDATE" in statement:
            claim_lock_attempted.set()

    def pause():
        with Session(ENGINE) as session, session.begin():
            session.info["phase100_pause_winner"] = True
            return _pause(session, claim_seed, key="pause-before-claim")

    def claim():
        assert pause_locked.wait(timeout=10)
        with Session(ENGINE) as session, session.begin():
            session.connection().info["phase100_claim_waiter"] = True
            claim_started.set()
            try:
                DecisionService(session).claim_execution(
                    claim_seed.decision_ids[0],
                    2,
                    principal=claim_seed.worker,
                    now=CLAIM_TIME,
                )
            except DecisionConflictError as error:
                return str(error)
            finally:
                session.connection().info.pop("phase100_claim_waiter", None)
        raise AssertionError("claim unexpectedly crossed a committed pause")

    monkeypatch.setattr(ReleaseService, "_lock_strategy", hold_pause_lock)
    event.listen(ENGINE, "before_cursor_execute", observe_claim_lock)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pause_future = pool.submit(pause)
            claim_future = pool.submit(claim)
            assert pause_locked.wait(timeout=10)
            assert claim_lock_attempted.wait(timeout=10)
            assert claim_future.done() is False
            release_pause.set()
            response = pause_future.result(timeout=20)
            conflict = claim_future.result(timeout=20)

        assert response["event_type"] == "pause"
        assert conflict == "decision active pointer changed"
        with Session(ENGINE) as session:
            decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
            strategy = session.get(StrategyRecord, claim_seed.strategy_ids[0])
            assert (strategy.active_version_id, strategy.release_revision) == (None, 1)
            assert (decision.status, decision.revision, decision.execution_key) == ("approved", 2, None)
            assert (
                session.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="execution_claimed").count()
                == 0
            )
            assert session.query(OrderRecord).filter_by(decision_id=decision.id).count() == 0
            assert session.query(PositionLot).filter_by(account_scope=claim_seed.account).count() == 0
    finally:
        release_pause.set()
        event.remove(ENGINE, "before_cursor_execute", observe_claim_lock)
        with Session(ENGINE) as session, session.begin():
            _delete_pause_event(session, claim_seed.strategy_ids[0])


def test_materialization_wins_strategy_lock_then_pause_blocks_replay(monkeypatch, claim_seed):
    with Session(ENGINE) as session, session.begin():
        decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
        generation = session.get(StrategyVersion, decision.strategy_version_id).policy_json["reconciliation"][
            "account_generation"
        ]
        DecisionService(session).claim_execution(decision.id, 2, principal=claim_seed.worker, now=CLAIM_TIME)
        session.add(
            PaperBrokerAccount(
                account_scope=claim_seed.account,
                account_generation=generation,
                opening_cash=100_000.0,
                currency="TWD",
                created_at=CLAIM_TIME,
                updated_at=CLAIM_TIME,
            )
        )
        session.flush()
        session.add(
            AccountReconciliation(
                account_scope=claim_seed.account,
                account_generation=generation,
                as_of=CLAIM_TIME,
                broker_state_watermark="broker:0",
                internal_state_watermark=internal_state_watermark(session, claim_seed.account, generation),
                broker_snapshot_sha256="d" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=decision.policy_sha256,
                status="matched",
            )
        )

    materialize_locked = Event()
    pause_started = Event()
    pause_lock_attempted = Event()
    release_materialize = Event()

    def hold_materialize_lock(session, decision, *, lock, error_type):
        strategy = require_active_pointer(session, decision, lock=lock, error_type=error_type)
        if lock and session.info.get("phase100_materialize_winner"):
            materialize_locked.set()
            assert pause_started.wait(timeout=10)
            assert release_materialize.wait(timeout=10)
        return strategy

    def observe_pause_lock(connection, cursor, statement, parameters, context, executemany):
        if connection.info.get("phase100_pause_waiter") and "FOR UPDATE" in statement:
            pause_lock_attempted.set()

    def materialize_session():
        session = Session(ENGINE)
        session.info["phase100_materialize_winner"] = True
        return session

    def materialize():
        return materialize_order_intents(
            materialize_session,
            claim_seed.decision_ids[0],
            principal=claim_seed.worker,
            account_nav=100_000.0,
            prices={("tw_stock", "2330", "spot"): 100.0},
            now=CLAIM_TIME,
        )

    def pause():
        assert materialize_locked.wait(timeout=10)
        with Session(ENGINE) as session, session.begin():
            session.connection().info["phase100_pause_waiter"] = True
            pause_started.set()
            try:
                return _pause(session, claim_seed, key="pause-after-materialize")
            finally:
                session.connection().info.pop("phase100_pause_waiter", None)

    monkeypatch.setattr("poseidon.decision_loop.releases.require_active_pointer", hold_materialize_lock)
    event.listen(ENGINE, "before_cursor_execute", observe_pause_lock)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            materialize_future = pool.submit(materialize)
            pause_future = pool.submit(pause)
            assert materialize_locked.wait(timeout=10)
            assert pause_lock_attempted.wait(timeout=10)
            assert pause_future.done() is False
            release_materialize.set()
            materialized = materialize_future.result(timeout=20)
            paused = pause_future.result(timeout=20)

        assert materialized["status"] == "pending_submit"
        assert paused["event_type"] == "pause"
        with pytest.raises(ExecutionConflictError, match="active pointer"):
            materialize()

        with Session(ENGINE) as session:
            strategy = session.get(StrategyRecord, claim_seed.strategy_ids[0])
            orders = session.scalars(
                select(OrderRecord).where(OrderRecord.decision_id == claim_seed.decision_ids[0])
            ).all()
            assert (strategy.active_version_id, strategy.release_revision) == (None, 1)
            assert len(orders) == 1
            assert orders[0].reservation_status == "reserved"
            assert orders[0].reserved_quantity == 100.0
            assert orders[0].reserved_cash_json == {"currency": "TWD", "amount": 10_000.0}
            assert session.query(PositionLot).filter_by(account_scope=claim_seed.account).count() == 0
    finally:
        release_materialize.set()
        event.remove(ENGINE, "before_cursor_execute", observe_pause_lock)
        with Session(ENGINE) as session, session.begin():
            session.execute(delete(OrderRecord).where(OrderRecord.decision_id == claim_seed.decision_ids[0]))
            session.execute(
                delete(AccountReconciliation).where(AccountReconciliation.account_scope == claim_seed.account)
            )
            session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == claim_seed.account))
            _delete_pause_event(session, claim_seed.strategy_ids[0])


def test_claim_next_skips_an_older_ineligible_approved_row(claim_seed):
    with Session(ENGINE) as session:
        ineligible = session.get(DecisionRecord, claim_seed.decision_ids[0])
        eligible = session.get(DecisionRecord, claim_seed.decision_ids[1])
        ineligible.final_json = {
            "selected_evaluation_ids": ineligible.final_json["selected_evaluation_ids"],
            "final_action": "hold",
        }
        ineligible.created_at = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
        eligible.created_at = datetime(2026, 9, 26, 12, 1, tzinfo=UTC)
        session.commit()

    with Session(ENGINE) as session:
        result = DecisionService(session).claim_next_approved(
            claim_seed.account,
            principal=claim_seed.worker,
            now=CLAIM_TIME,
        )
        session.commit()

    assert result["decision_id"] == str(claim_seed.decision_ids[1])
    with Session(ENGINE) as session:
        ineligible = session.get(DecisionRecord, claim_seed.decision_ids[0])
        assert (ineligible.status, ineligible.execution_key) == ("approved", None)


def test_skip_locked_claims_distinct_eligible_rows_without_waiting(claim_seed):
    selected = Barrier(2)

    def claim_next():
        with Session(ENGINE) as session:
            result = DecisionService(session).claim_next_approved(
                claim_seed.account,
                principal=claim_seed.worker,
                now=CLAIM_TIME,
            )
            selected.wait(timeout=10)
            session.commit()
            return result["decision_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed_ids = [future.result(timeout=20) for future in [pool.submit(claim_next), pool.submit(claim_next)]]

    assert set(claimed_ids) == {str(decision_id) for decision_id in claim_seed.decision_ids}
    with Session(ENGINE) as session:
        decisions = session.scalars(select(DecisionRecord).where(DecisionRecord.id.in_(claim_seed.decision_ids))).all()
        assert all(decision.status == "execution_claimed" for decision in decisions)
        assert len({decision.execution_key for decision in decisions}) == 2
        assert (
            session.query(DecisionEvent)
            .filter(
                DecisionEvent.decision_id.in_(claim_seed.decision_ids),
                DecisionEvent.event_type == "execution_claimed",
            )
            .count()
            == 2
        )


def test_same_account_materialization_serializes_reservations(claim_seed):
    principal = claim_seed.worker
    reduce_decision_ids = []
    source_order_id = uuid.uuid4()
    source_fill_id = uuid.uuid4()
    lot_id = uuid.uuid4()
    with Session(ENGINE) as session:
        approved = session.scalars(
            select(DecisionRecord).where(DecisionRecord.id.in_(claim_seed.decision_ids)).order_by(DecisionRecord.id)
        ).all()
        generation = session.get(StrategyVersion, approved[0].strategy_version_id).policy_json["reconciliation"][
            "account_generation"
        ]
        for index, existing in enumerate(approved):
            snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=existing.evaluation_run_id).one()
            decision = DecisionService(session).create_decision(
                existing.evaluation_run_id,
                principal=principal,
                account_scope=claim_seed.account,
                original_json={
                    "selected_evaluation_ids": [str(snapshot.id)],
                    "final_action": "exit",
                    "order_intents": [executable_intent(str(snapshot.id), action="exit", target_weight=0.0)],
                },
                portfolio_snapshot_json={"cash": 90_000.0, "positions": [{"symbol": "2330", "quantity": 100.0}]},
                risk_snapshot_json={"hard_failures": [], "allowed_actions": ["exit"]},
                now=datetime(2026, 9, 26, 12, 31, tzinfo=UTC),
            )
            DecisionService(session).approve(
                decision.id,
                {"expected_revision": 1},
                principal=AuthPrincipal(
                    "human:phase98-manager",
                    frozenset({"portfolio_manager"}),
                    frozenset({claim_seed.account}),
                ),
                idempotency_key=f"phase98-exit-approve-{index}-{decision.id}",
                now=datetime(2026, 9, 26, 12, 35, tzinfo=UTC),
            )
            DecisionService(session).claim_execution(decision.id, 2, principal=principal, now=CLAIM_TIME)
            reduce_decision_ids.append(decision.id)
        session.add(
            PaperBrokerAccount(
                account_scope=claim_seed.account,
                account_generation=generation,
                opening_cash=100_000.0,
                currency="TWD",
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.add(
            AccountReconciliation(
                account_scope=claim_seed.account,
                account_generation=generation,
                as_of=datetime(2026, 9, 26, 12, 44, tzinfo=UTC),
                broker_state_watermark="broker:0",
                internal_state_watermark="internal:2026-09-26T12:40:00Z",
                broker_snapshot_sha256="b" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=approved[0].policy_sha256,
                status="matched",
            )
        )
        session.add(
            OrderRecord(
                id=source_order_id,
                strategy_name="phase98-opening-lot",
                symbol="2330",
                market="tw_stock",
                action="buy",
                order_type="market",
                target_weight=0.1,
                quantity=100.0,
                price=100.0,
                side="long",
                status="filled",
                broker_mode="paper",
                order_origin="decision",
                decision_id=claim_seed.decision_ids[0],
                account_scope=claim_seed.account,
                account_generation=generation,
                client_order_ref=f"DL-source-{source_order_id.hex}",
                instrument="spot",
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.flush()
        session.add(
            OrderFillRecord(
                id=source_fill_id,
                order_id=source_order_id,
                fill_price=100.0,
                fill_quantity=100.0,
                fill_time=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                broker_fill_id=f"phase98-source-{source_fill_id.hex}",
                projection_status="applied",
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.flush()
        session.add(
            PositionLot(
                id=lot_id,
                account_scope=claim_seed.account,
                account_generation=generation,
                market="tw_stock",
                symbol="2330",
                instrument="spot",
                side="long",
                opening_fill_id=source_fill_id,
                opening_decision_id=claim_seed.decision_ids[0],
                original_quantity=100.0,
                open_quantity=100.0,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 100.0},
                opened_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.commit()

    barrier = Barrier(2)

    def materialize(decision_id):
        barrier.wait(timeout=10)
        try:
            return materialize_order_intents(
                lambda: Session(ENGINE),
                decision_id,
                principal=principal,
                account_nav=100_000.0,
                prices={("tw_stock", "2330", "spot"): 100.0},
                now=CLAIM_TIME,
            )["status"]
        except ExecutionConflictError as error:
            return str(error)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = [
                future.result(timeout=20)
                for future in [pool.submit(materialize, decision_id) for decision_id in reduce_decision_ids]
            ]
        assert sorted(statuses) == ["pending_submit", "reduce/exit requires a negative quantity delta"]
        with Session(ENGINE) as session:
            orders = session.scalars(
                select(OrderRecord).where(
                    OrderRecord.account_scope == claim_seed.account,
                    OrderRecord.account_generation == generation,
                )
            ).all()
            materialized = [order for order in orders if order.decision_id in reduce_decision_ids]
            assert len(materialized) == 1
            assert materialized[0].reserved_quantity == 100.0
            assert session.get(PositionLot, lot_id).reserved_close_quantity == 100.0
    finally:
        with Session(ENGINE) as session:
            session.execute(
                delete(OrderRecord).where(
                    OrderRecord.account_scope == claim_seed.account,
                    OrderRecord.id != source_order_id,
                )
            )
            session.execute(delete(PositionLot).where(PositionLot.id == lot_id))
            session.execute(delete(OrderFillRecord).where(OrderFillRecord.id == source_fill_id))
            session.execute(delete(OrderRecord).where(OrderRecord.id == source_order_id))
            session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id.in_(reduce_decision_ids)))
            session.execute(delete(DecisionRecord).where(DecisionRecord.id.in_(reduce_decision_ids)))
            session.execute(
                delete(AccountReconciliation).where(AccountReconciliation.account_scope == claim_seed.account)
            )
            session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == claim_seed.account))
            session.commit()


def test_same_account_materialization_serializes_cash_capacity(claim_seed):
    principal = claim_seed.worker
    source_order_id = uuid.uuid4()
    custom_run_id = None
    custom_snapshot_id = None
    custom_decision_id = None
    with Session(ENGINE) as session:
        first_decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
        custom_universe = [{"symbol": "2317", "market": "tw_stock", "instrument": "spot"}]
        custom_run = EvaluationService(session).evaluate_run(
            claim_seed.version_ids[1],
            claim_seed.manifest_id,
            custom_universe,
            [
                {
                    **custom_universe[0],
                    "status": "evaluated",
                    "recommendation_json": {"research_status": "not_required", "side": "long"},
                    "reason_codes": [],
                    "valid_until": "2026-09-26T15:00:00Z",
                }
            ],
        )
        session.flush()
        custom_snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=custom_run.id).one()
        custom_decision = DecisionService(session).create_decision(
            custom_run.id,
            principal=principal,
            account_scope=claim_seed.account,
            original_json={
                "selected_evaluation_ids": [str(custom_snapshot.id)],
                "final_action": "enter",
                "order_intents": [
                    executable_intent(str(custom_snapshot.id), symbol="2317", action="enter", target_weight=0.1)
                ],
            },
            portfolio_snapshot_json={"cash": 15_000.0, "positions": []},
            risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
            now=datetime(2026, 9, 26, 12, 1, tzinfo=UTC),
        )
        manager = AuthPrincipal(
            "human:phase98-manager",
            frozenset({"portfolio_manager"}),
            frozenset({claim_seed.account}),
        )
        DecisionService(session).approve(
            custom_decision.id,
            {"expected_revision": 1},
            principal=manager,
            idempotency_key=f"phase98-cash-approve-{custom_decision.id}",
            now=datetime(2026, 9, 26, 12, 35, tzinfo=UTC),
        )
        DecisionService(session).claim_execution(first_decision.id, 2, principal=principal, now=CLAIM_TIME)
        DecisionService(session).claim_execution(custom_decision.id, 2, principal=principal, now=CLAIM_TIME)
        generation = session.get(StrategyVersion, first_decision.strategy_version_id).policy_json["reconciliation"][
            "account_generation"
        ]
        session.add(
            PaperBrokerAccount(
                account_scope=claim_seed.account,
                account_generation=generation,
                opening_cash=100_000.0,
                currency="TWD",
                state_version=1,
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.add(
            PaperCashMovement(
                account_scope=claim_seed.account,
                account_generation=generation,
                currency="TWD",
                amount=-85_000.0,
                movement_type="adjustment",
                state_version=1,
                occurred_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.add(
            OrderRecord(
                id=source_order_id,
                strategy_name="phase98-cash-watermark",
                symbol="2303",
                market="tw_stock",
                action="buy",
                order_type="market",
                target_weight=0.0,
                quantity=1.0,
                price=100.0,
                side="long",
                status="filled",
                broker_mode="paper",
                order_origin="decision",
                decision_id=first_decision.id,
                account_scope=claim_seed.account,
                account_generation=generation,
                client_order_ref=f"DL-cash-source-{source_order_id.hex}",
                instrument="spot",
                created_at=CLAIM_TIME,
                updated_at=CLAIM_TIME,
            )
        )
        session.flush()
        session.add(
            AccountReconciliation(
                account_scope=claim_seed.account,
                account_generation=generation,
                as_of=CLAIM_TIME,
                broker_state_watermark="broker:1",
                internal_state_watermark=internal_state_watermark(session, claim_seed.account, generation),
                broker_snapshot_sha256="c" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=first_decision.policy_sha256,
                status="matched",
            )
        )
        session.commit()
        custom_run_id = custom_run.id
        custom_snapshot_id = custom_snapshot.id
        custom_decision_id = custom_decision.id

    barrier = Barrier(2)
    decision_ids = (claim_seed.decision_ids[0], custom_decision_id)

    def materialize(decision_id):
        barrier.wait(timeout=10)
        try:
            return materialize_order_intents(
                lambda: Session(ENGINE),
                decision_id,
                principal=principal,
                account_nav=100_000.0,
                prices={
                    ("tw_stock", "2330", "spot"): 100.0,
                    ("tw_stock", "2317", "spot"): 100.0,
                },
                now=CLAIM_TIME,
            )["status"]
        except ExecutionConflictError as error:
            return str(error)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = [
                future.result(timeout=20) for future in [pool.submit(materialize, item) for item in decision_ids]
            ]
        # Winner's reservation invalidates the prior match before loser's capacity arithmetic.
        assert sorted(outcomes) == ["account reconciliation internal watermark changed", "pending_submit"]
        with Session(ENGINE) as session:
            orders = session.scalars(select(OrderRecord).where(OrderRecord.decision_id.in_(decision_ids))).all()
            materialized = [order for order in orders if order.id != source_order_id]
            assert len(materialized) == 1
            assert materialized[0].reserved_cash_json == {"currency": "TWD", "amount": 10_000.0}
    finally:
        with Session(ENGINE) as session:
            session.execute(delete(OrderRecord).where(OrderRecord.account_scope == claim_seed.account))
            session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id == custom_decision_id))
            session.execute(delete(DecisionRecord).where(DecisionRecord.id == custom_decision_id))
            session.execute(delete(EvaluationSnapshot).where(EvaluationSnapshot.id == custom_snapshot_id))
            session.execute(delete(EvaluationRun).where(EvaluationRun.id == custom_run_id))
            session.execute(
                delete(AccountReconciliation).where(AccountReconciliation.account_scope == claim_seed.account)
            )
            session.execute(delete(PaperCashMovement).where(PaperCashMovement.account_scope == claim_seed.account))
            session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == claim_seed.account))
            session.commit()
