"""Real PostgreSQL locking and constraint proof for Phase 97 decisions."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionService
from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.manifest import ManifestService, content_sha256
from poseidon.models.data_manifest import DataManifest
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import manifest_request, synthetic_policy

DATABASE_URL = os.environ["POSEIDON_DATABASE_URL"]
ENGINE = create_engine(DATABASE_URL)
if ENGINE.dialect.name != "postgresql":
    raise RuntimeError("test_decision_concurrency_postgres.py requires PostgreSQL")


@dataclass(frozen=True)
class Seed:
    account: str
    strategy_id: uuid.UUID
    version_id: uuid.UUID
    manifest_id: uuid.UUID
    run_id: uuid.UUID
    snapshot_id: uuid.UUID

    @property
    def worker(self):
        return AuthPrincipal("service:phase97-worker", frozenset({"decision-worker"}), frozenset({self.account}))

    @property
    def manager(self):
        return AuthPrincipal("human:phase97-manager", frozenset({"portfolio_manager"}), frozenset({self.account}))


@pytest.fixture
def seed():
    marker = uuid.uuid4().hex
    account = f"paper:phase97:{marker}"
    universe_id = f"phase97-{marker}"
    with Session(ENGINE) as session:
        request = manifest_request()
        request["account_scope"] = account
        request["universe_id"] = universe_id
        request["evidence"][0]["payload"]["marker"] = marker
        request["evidence"][0]["content_sha256"] = content_sha256(request["evidence"][0]["payload"])
        manifest = ManifestService(session).freeze(request)
        strategy = StrategyRecord(
            name=f"phase97-concurrency-{marker}",
            strategy_type="technical",
            symbol="2330",
            market="tw_stock",
            interval="1d",
        )
        session.add(strategy)
        session.flush()
        policy_json = synthetic_policy(account_scope=account, universe_id=universe_id)
        version = StrategyVersion(
            strategy_id=strategy.id,
            version_no=1,
            config_json={"marker": marker},
            policy_json=policy_json,
            artifact_json={},
            content_sha256=strategy_version_digest({"marker": marker}, policy_json, {}),
        )
        session.add(version)
        session.flush()
        universe = [{"symbol": "2330", "market": "tw_stock", "instrument": "spot"}]
        snapshots = [
            {
                **universe[0],
                "status": "evaluated",
                "recommendation_json": {"research_status": "not_required"},
                "reason_codes": [],
                "valid_until": "2026-09-26T15:00:00Z",
            }
        ]
        run = EvaluationService(session).evaluate_run(version.id, manifest.id, universe, snapshots)
        session.flush()
        snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=run.id).one()
        value = Seed(account, strategy.id, version.id, manifest.id, run.id, snapshot.id)
        session.commit()

    yield value

    with Session(ENGINE) as session:
        decision_ids = select(DecisionRecord.id).where(DecisionRecord.strategy_version_id == value.version_id)
        session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id.in_(decision_ids)))
        session.execute(delete(DecisionRecord).where(DecisionRecord.strategy_version_id == value.version_id))
        session.execute(delete(EvaluationSnapshot).where(EvaluationSnapshot.evaluation_run_id == value.run_id))
        session.execute(delete(EvaluationRun).where(EvaluationRun.id == value.run_id))
        session.execute(delete(StrategyVersion).where(StrategyVersion.id == value.version_id))
        session.execute(delete(DataManifest).where(DataManifest.id == value.manifest_id))
        session.execute(delete(StrategyRecord).where(StrategyRecord.id == value.strategy_id))
        session.commit()


def create_kwargs(seed, cash=100000.0):
    return {
        "principal": seed.worker,
        "account_scope": seed.account,
        "original_json": {"selected_evaluation_ids": [str(seed.snapshot_id)], "final_action": "hold"},
        "portfolio_snapshot_json": {"cash": cash, "positions": []},
        "risk_snapshot_json": {"hard_failures": [], "allowed_actions": ["hold", "reduce", "exit"]},
        "now": datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
    }


def concurrent_calls(call_a, call_b):
    barrier = Barrier(2)

    def run(call):
        with Session(ENGINE) as session:
            barrier.wait(timeout=10)
            try:
                result = call(session)
                session.commit()
                return "ok", result
            except DecisionConflictError as error:
                session.rollback()
                return "conflict", str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, call) for call in (call_a, call_b)]
        return [future.result(timeout=20) for future in futures]


def create_once(seed, cash=100000.0):
    with Session(ENGINE) as session:
        decision = DecisionService(session).create_decision(seed.run_id, **create_kwargs(seed, cash))
        decision_id = decision.id
        session.commit()
        return decision_id


def test_same_creation_digest_converges_to_one_decision_and_event(seed):
    results = concurrent_calls(
        lambda session: DecisionService(session).create_decision(seed.run_id, **create_kwargs(seed)).id,
        lambda session: DecisionService(session).create_decision(seed.run_id, **create_kwargs(seed)).id,
    )
    assert {status for status, _ in results} == {"ok"}
    assert len({result for _, result in results}) == 1
    with Session(ENGINE) as session:
        decisions = session.query(DecisionRecord).filter_by(strategy_version_id=seed.version_id).all()
        assert len(decisions) == 1
        events = session.query(DecisionEvent).filter_by(decision_id=decisions[0].id).all()
        assert [(event.event_type, event.expected_revision) for event in events] == [("created", 0)]


def test_changed_creation_race_leaves_one_live_and_one_superseded(seed):
    results = concurrent_calls(
        lambda session: DecisionService(session).create_decision(seed.run_id, **create_kwargs(seed, 100000)).id,
        lambda session: DecisionService(session).create_decision(seed.run_id, **create_kwargs(seed, 90000)).id,
    )
    assert [status for status, _ in results] == ["ok", "ok"]
    assert len({result for _, result in results}) == 2
    with Session(ENGINE) as session:
        decisions = session.query(DecisionRecord).filter_by(strategy_version_id=seed.version_id).all()
        assert sorted(decision.status for decision in decisions) == ["pending_approval", "superseded"]
        superseded = next(decision for decision in decisions if decision.status == "superseded")
        assert superseded.revision == 2
        events = session.query(DecisionEvent).filter_by(decision_id=superseded.id).all()
        assert sorted((event.event_type, event.expected_revision) for event in events) == [
            ("created", 0),
            ("superseded", 1),
        ]


def test_same_key_same_body_approval_replays_one_response(seed):
    decision_id = create_once(seed)

    def call(session):
        return DecisionService(session).approve(
            decision_id,
            {"expected_revision": 1},
            principal=seed.manager,
            idempotency_key="same-approval",
            now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )

    results = concurrent_calls(call, call)
    assert [status for status, _ in results] == ["ok", "ok"]
    assert results[0][1] == results[1][1]
    with Session(ENGINE) as session:
        events = session.query(DecisionEvent).filter_by(decision_id=decision_id, event_type="approved").all()
        assert len(events) == 1
        assert events[0].expected_revision == 1
        assert events[0].request_sha256 == content_sha256(
            {
                "operation": "approve",
                "decision_id": str(decision_id),
                "expected_revision": 1,
                "actor_id": seed.manager.actor_id,
                "body": {"expected_revision": 1, "final_action": None, "override_reason": None},
            }
        )


@pytest.mark.parametrize("case", ["changed_body", "different_key"])
def test_competing_approval_is_one_success_and_one_conflict(seed, case):
    decision_id = create_once(seed)

    def approve(session, *, key, body):
        return DecisionService(session).approve(
            decision_id,
            body,
            principal=seed.manager,
            idempotency_key=key,
            now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )

    def first(session):
        return approve(session, key="race-key", body={"expected_revision": 1})

    def second(session):
        if case == "changed_body":
            return approve(session, key="race-key", body={"expected_revision": 1, "override_reason": "changed"})
        return approve(session, key="other-key", body={"expected_revision": 1})

    results = concurrent_calls(first, second)
    assert sorted(status for status, _ in results) == ["conflict", "ok"]
    with Session(ENGINE) as session:
        decision = session.get(DecisionRecord, decision_id)
        assert (decision.status, decision.revision) == ("approved", 2)
        events = session.query(DecisionEvent).filter_by(decision_id=decision_id, event_type="approved").all()
        assert len(events) == 1


@pytest.mark.parametrize("operation", ["approve", "expire"])
def test_changed_creation_serializes_with_other_transition(seed, operation):
    old_id = create_once(seed)

    def transition(session):
        service = DecisionService(session)
        if operation == "approve":
            return service.approve(
                old_id,
                {"expected_revision": 1},
                principal=seed.manager,
                idempotency_key="create-transition-race",
                now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
            )
        return [row.id for row in service.expire_due(datetime(2026, 9, 26, 13, 0, tzinfo=UTC))]

    results = concurrent_calls(
        lambda session: DecisionService(session).create_decision(seed.run_id, **create_kwargs(seed, 90000)).id,
        transition,
    )
    assert results[0][0] == "ok"
    with Session(ENGINE) as session:
        decisions = session.query(DecisionRecord).filter_by(strategy_version_id=seed.version_id).all()
        assert len(decisions) == 2
        for decision in decisions:
            events = (
                session.query(DecisionEvent)
                .filter_by(decision_id=decision.id)
                .order_by(DecisionEvent.expected_revision)
                .all()
            )
            revisions = [event.expected_revision for event in events]
            assert revisions == list(range(len(revisions)))
            assert decision.revision == len(revisions)


def test_database_constraints_reject_duplicate_pair_and_orphan(seed):
    decision_id = create_once(seed)
    with Session(ENGINE) as session:
        session.add(
            DecisionEvent(
                decision_id=decision_id,
                event_type="constraint_probe",
                actor_id="test",
                expected_revision=90,
                idempotency_key="constraint-key",
                request_sha256="1" * 64,
                payload_json={},
            )
        )
        session.commit()

    invalid = [
        DecisionEvent(
            decision_id=decision_id,
            event_type="duplicate_replay",
            actor_id="test",
            expected_revision=91,
            idempotency_key="constraint-key",
            request_sha256="2" * 64,
            payload_json={},
        ),
        DecisionEvent(
            decision_id=decision_id,
            event_type="invalid_pair",
            actor_id="test",
            expected_revision=92,
            idempotency_key="missing-request-digest",
            request_sha256=None,
            payload_json={},
        ),
        DecisionEvent(
            decision_id=uuid.uuid4(),
            event_type="orphan",
            actor_id="test",
            expected_revision=0,
            payload_json={},
        ),
    ]
    for event in invalid:
        with Session(ENGINE) as session:
            session.add(event)
            with pytest.raises(IntegrityError):
                session.flush()
            session.rollback()
