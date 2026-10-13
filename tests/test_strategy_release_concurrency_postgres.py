"""Real PostgreSQL serialization proof for Phase 100 release authority."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from poseidon.decision_loop.decisions import DecisionConflictError
from poseidon.decision_loop.releases import ReleaseService
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from tests.test_strategy_release import NOW, _seed_gate_graph, release_manager


DATABASE_URL = os.environ["POSEIDON_DATABASE_URL"]
ENGINE = create_engine(DATABASE_URL)
if ENGINE.dialect.name != "postgresql":
    raise RuntimeError("test_strategy_release_concurrency_postgres.py requires PostgreSQL")


@pytest.fixture
def graph():
    with Session(ENGINE) as session:
        with session.begin():
            value = _seed_gate_graph(session)
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
    release_holder = Event()

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
            pause_started.set()
            response = ReleaseService(session).pause(
                graph.strategy_id,
                expected_revision=1,
                expected_active_version_id=graph.candidate_id,
                principal=release_manager(),
                idempotency_key="pause-after-ordinary-lock",
                reason="serialize behind ordinary authority reader",
            )
            return response

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(hold_strategy_lock)
        waiter = pool.submit(pause)
        assert pause_started.wait(timeout=10)
        release_holder.set()
        holder.result(timeout=20)
        response = waiter.result(timeout=20)

    assert response["event_type"] == "pause"
    with Session(ENGINE) as session:
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (None, 2)
