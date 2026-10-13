"""Real PostgreSQL serialization proof for Phase 100 release authority."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from queue import Queue
from threading import Barrier, Event

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from poseidon.decision_loop.decisions import DecisionConflictError
from poseidon.decision_loop.execution import DecisionExecutionService
from poseidon.decision_loop.releases import ReleaseService
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from tests.test_strategy_release import NOW, _seed_gate_graph, release_manager
from tests.test_decision_service import synthetic_reconciliation_policy


DATABASE_URL = os.environ["POSEIDON_DATABASE_URL"]
ENGINE = create_engine(DATABASE_URL)
if ENGINE.dialect.name != "postgresql":
    raise RuntimeError("test_strategy_release_concurrency_postgres.py requires PostgreSQL")


@pytest.fixture
def graph():
    with Session(ENGINE) as session:
        with session.begin():
            value = _seed_gate_graph(
                session,
                policy_changes={
                    "reconciliation": synthetic_reconciliation_policy(
                        account_generation=f"phase100-concurrency-{uuid.uuid4().hex}",
                    )
                },
            )
    return value


def _release(session, graph, *, key, expected_revision=0, expected_pointer=None):
    return ReleaseService(session).release(
        graph.strategy_id,
        graph.candidate_id,
        graph.campaign_id,
        graph.review_id,
        graph.reconciliation_id,
        expected_revision=expected_revision,
        expected_active_version_id=expected_pointer,
        principal=release_manager(),
        idempotency_key=key,
        reason="concurrent release",
        now=NOW,
    )


def _concurrent(call_a, call_b):
    barrier = Barrier(2)

    def run(call):
        try:
            with Session(ENGINE) as session, session.begin():
                barrier.wait(timeout=10)
                response = call(session)
            return "ok", response
        except DecisionConflictError as error:
            return "conflict", str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, call) for call in (call_a, call_b)]
        return [future.result(timeout=30) for future in futures]


def test_same_key_same_release_replays_one_retained_response(graph):
    call = lambda session: _release(session, graph, key="same-release-race")

    results = _concurrent(call, call)

    assert [status for status, _ in results] == ["ok", "ok"]
    assert results[0][1] == results[1][1]
    with Session(ENGINE) as session:
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 1


def test_different_key_release_race_has_one_success_and_one_stale_conflict(graph):
    results = _concurrent(
        lambda session: _release(session, graph, key="release-race-a"),
        lambda session: _release(session, graph, key="release-race-b"),
    )

    assert sorted(status for status, _ in results) == ["conflict", "ok"]
    with Session(ENGINE) as session:
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (graph.candidate_id, 1)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 1


def test_release_and_pause_serialize_to_one_revision_winner(graph):
    with Session(ENGINE) as session:
        with session.begin():
            _release(session, graph, key="initial-release")
        with session.begin():
            followup = _seed_gate_graph(
                session,
                strategy_id=graph.strategy_id,
                incumbent_id=graph.candidate_id,
            )

    results = _concurrent(
        lambda session: _release(
            session,
            followup,
            key="followup-release",
            expected_revision=1,
            expected_pointer=graph.candidate_id,
        ),
        lambda session: ReleaseService(session).pause(
            graph.strategy_id,
            expected_revision=1,
            expected_active_version_id=graph.candidate_id,
            principal=release_manager(),
            idempotency_key="competing-pause",
            reason="concurrent operator pause",
        ),
    )

    assert sorted(status for status, _ in results) == ["conflict", "ok"]
    with Session(ENGINE) as session:
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert strategy.release_revision == 2
        assert strategy.active_version_id in {None, followup.candidate_id}
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 2


def test_pause_waits_for_ordinary_strategy_lock_holder(graph):
    with Session(ENGINE) as session, session.begin():
        _release(session, graph, key="pause-lock-initial-release")

    holder_ready = Event()
    pause_started = Event()
    pause_lock_attempted = Event()
    release_holder = Event()

    def observe_pause_lock(connection, cursor, statement, parameters, context, executemany):
        if connection.info.get("phase100_pause_waiter") and "FOR UPDATE" in statement:
            pause_lock_attempted.set()

    def hold_strategy_lock():
        with Session(ENGINE) as session, session.begin():
            session.scalar(
                select(StrategyRecord).where(StrategyRecord.id == graph.strategy_id).with_for_update()
            )
            holder_ready.set()
            assert pause_started.wait(timeout=10)
            assert release_holder.wait(timeout=10)

    def pause():
        assert holder_ready.wait(timeout=10)
        with Session(ENGINE) as session, session.begin():
            session.connection().info["phase100_pause_waiter"] = True
            pause_started.set()
            try:
                return ReleaseService(session).pause(
                    graph.strategy_id,
                    expected_revision=1,
                    expected_active_version_id=graph.candidate_id,
                    principal=release_manager(),
                    idempotency_key="pause-after-ordinary-lock",
                    reason="serialize behind ordinary authority reader",
                )
            finally:
                session.connection().info.pop("phase100_pause_waiter", None)

    event.listen(ENGINE, "before_cursor_execute", observe_pause_lock)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(hold_strategy_lock)
            waiter = pool.submit(pause)
            assert pause_started.wait(timeout=10)
            assert pause_lock_attempted.wait(timeout=10)
            assert waiter.done() is False
            release_holder.set()
            holder.result(timeout=20)
            response = waiter.result(timeout=20)
    finally:
        release_holder.set()
        event.remove(ENGINE, "before_cursor_execute", observe_pause_lock)

    assert response["event_type"] == "pause"
    with Session(ENGINE) as session:
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (None, 2)


@pytest.mark.parametrize("operation", ("release", "rollback"))
def test_release_transition_waits_for_current_account_reconciliation(graph, operation):
    if operation == "rollback":
        with Session(ENGINE) as session, session.begin():
            _release(session, graph, key="reconciliation-lock-initial-release")
            ReleaseService(session).pause(
                graph.strategy_id,
                expected_revision=1,
                expected_active_version_id=graph.candidate_id,
                principal=release_manager(),
                idempotency_key="reconciliation-lock-pause",
                reason="prepare rollback reconciliation race",
            )

    holder_ready = Event()
    newer_committed = Event()
    progress = Queue()

    def observe_before(connection, cursor, statement, parameters, context, executemany):
        if not connection.info.get("phase100_reconciliation_waiter"):
            return
        normalized = " ".join(statement.lower().split())
        if "pg_advisory_xact_lock" in normalized:
            progress.put("account_lock")
        elif "insert into strategy_release_events" in normalized:
            assert newer_committed.wait(timeout=10)

    def observe_after(connection, cursor, statement, parameters, context, executemany):
        if not connection.info.get("phase100_reconciliation_waiter"):
            return
        normalized = " ".join(statement.lower().split())
        if (
            "from account_reconciliations" in normalized
            and "order by account_reconciliations.as_of desc" in normalized
        ):
            progress.put("stale_read")

    def append_newer_reconciliation():
        observed = None
        with Session(ENGINE) as session, session.begin():
            current = session.get(AccountReconciliation, graph.reconciliation_id)
            DecisionExecutionService(session)._advisory_lock(
                current.account_scope,
                current.account_generation,
            )
            session.add(
                AccountReconciliation(
                    account_scope=current.account_scope,
                    account_generation=current.account_generation,
                    as_of=NOW + timedelta(seconds=1),
                    broker_state_watermark=f"broker:concurrent-{operation}",
                    internal_state_watermark=f"internal:concurrent-{operation}",
                    broker_snapshot_sha256=("6" if operation == "release" else "7") * 64,
                    broker_snapshot_json={},
                    internal_snapshot_json={},
                    difference_json={"unresolved": ["concurrent fixture"]},
                    policy_sha256=current.policy_sha256,
                    status="unresolved",
                )
            )
            session.flush()
            holder_ready.set()
            observed = progress.get(timeout=10)
        newer_committed.set()
        return observed

    def transition():
        assert holder_ready.wait(timeout=10)
        try:
            with Session(ENGINE) as session, session.begin():
                session.connection().info["phase100_reconciliation_waiter"] = True
                try:
                    if operation == "release":
                        _release(session, graph, key="release-after-reconciliation-lock")
                    else:
                        ReleaseService(session).rollback(
                            graph.strategy_id,
                            graph.candidate_id,
                            graph.reconciliation_id,
                            expected_revision=2,
                            expected_active_version_id=None,
                            principal=release_manager(),
                            idempotency_key="rollback-after-reconciliation-lock",
                            reason="reject stale reconciliation race",
                            now=NOW + timedelta(seconds=2),
                        )
                finally:
                    session.connection().info.pop("phase100_reconciliation_waiter", None)
            return "ok"
        except DecisionConflictError:
            return "conflict"

    event.listen(ENGINE, "before_cursor_execute", observe_before)
    event.listen(ENGINE, "after_cursor_execute", observe_after)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(append_newer_reconciliation)
            waiter = pool.submit(transition)
            observed = holder.result(timeout=20)
            result = waiter.result(timeout=20)
    finally:
        newer_committed.set()
        event.remove(ENGINE, "after_cursor_execute", observe_after)
        event.remove(ENGINE, "before_cursor_execute", observe_before)

    assert observed == "account_lock"
    assert result == "conflict"
    with Session(ENGINE) as session:
        strategy = session.get(StrategyRecord, graph.strategy_id)
        expected = (None, 2) if operation == "rollback" else (None, 0)
        assert (strategy.active_version_id, strategy.release_revision) == expected
