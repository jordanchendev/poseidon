"""Fail-closed decision execution claim contracts."""

import copy
import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionService
from poseidon.models.base import Base
from poseidon.models.decision_event import DecisionEvent
from tests.test_decision_service import (
    create_decision,
    decision_inputs,
    executable_intent,
    manager,
    synthetic_policy,
    synthetic_reconciliation_policy,
    worker,
)

CLAIM_TIME = datetime(2026, 9, 26, 12, 45, tzinfo=UTC)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    tables = [
        Base.metadata.tables[name]
        for name in (
            "data_manifests",
            "research_revisions",
            "strategy_versions",
            "evaluation_runs",
            "evaluation_snapshots",
            "decision_records",
            "decision_events",
        )
    ]
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as session:
        yield session
    engine.dispose()


def approved_execution(db, *, reconciliation=True):
    policy = synthetic_policy(
        reconciliation=synthetic_reconciliation_policy() if reconciliation else None,
    )
    run, version, snapshot_ids = decision_inputs(db, policy=policy)
    decision = create_decision(
        db,
        run.id,
        snapshot_ids,
        original_json={
            "selected_evaluation_ids": [snapshot_ids[0]],
            "final_action": "enter",
            "order_intents": [executable_intent(snapshot_ids[0])],
        },
        risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
    )
    DecisionService(db).approve(
        decision.id,
        {"expected_revision": 1},
        principal=manager(),
        idempotency_key=f"approve-{decision.id}",
        now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
    )
    return decision, version


def test_claim_requires_worker_role_and_exact_account_scope(db):
    decision, _ = approved_execution(db)
    principals = (
        manager(),
        AuthPrincipal("legacy-api-key", frozenset(), frozenset()),
        worker("paper:other"),
    )

    for principal in principals:
        with pytest.raises(HTTPException) as error:
            DecisionService(db).claim_execution(
                decision.id,
                2,
                principal=principal,
                now=CLAIM_TIME,
            )
        assert error.value.status_code == 403

    assert decision.execution_key is None
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="execution_claimed").count() == 0


def test_claim_flushes_one_key_revision_and_event_without_committing(db, monkeypatch):
    decision, _ = approved_execution(db)

    def reject_commit():
        raise AssertionError("DecisionService must not commit the caller-owned session")

    monkeypatch.setattr(db, "commit", reject_commit)
    response = DecisionService(db).claim_execution(
        decision.id,
        2,
        principal=worker(),
        now=CLAIM_TIME,
    )

    assert response == {
        "decision_id": str(decision.id),
        "execution_key": str(decision.execution_key),
        "status": "execution_claimed",
        "revision": 3,
        "claimed_at": "2026-09-26T12:45:00Z",
    }
    event = db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="execution_claimed").one()
    assert event.actor_id == worker().actor_id
    assert event.expected_revision == 2
    assert event.payload_json == {"response": response}


@pytest.mark.parametrize(
    "case,reason",
    [
        ("stale_revision", "revision"),
        ("superseded", "approved"),
        ("expired", "expired"),
        ("risk_blocked", "approved"),
        ("already_claimed", "approved"),
        ("policy_drift", "policy changed"),
        ("missing_intent", "requires order_intents"),
        ("missing_reconciliation", "reconciliation policy"),
        ("hard_risk", "hard risk"),
    ],
)
def test_ineligible_decisions_never_receive_an_execution_key(db, case, reason):
    decision, version = approved_execution(db, reconciliation=case != "missing_reconciliation")
    expected_revision = 2
    now = CLAIM_TIME
    if case == "stale_revision":
        expected_revision = 1
    elif case in {"superseded", "risk_blocked"}:
        decision.status = case
    elif case == "expired":
        now = datetime(2026, 9, 26, 15, 0, tzinfo=UTC)
    elif case == "already_claimed":
        decision.status = "execution_claimed"
        decision.execution_key = uuid.uuid4()
        decision.claimed_at = CLAIM_TIME
    elif case == "policy_drift":
        changed_policy = copy.deepcopy(version.policy_json) | {"decision_ttl_seconds": 7200}
        db.execute(
            update(type(version))
            .where(type(version).id == version.id)
            .values(policy_json=changed_policy)
            .execution_options(synchronize_session=False)
        )
        db.expire(version, ["policy_json"])
    elif case == "missing_intent":
        decision.final_json = {
            "selected_evaluation_ids": decision.final_json["selected_evaluation_ids"],
            "final_action": "enter",
        }
    elif case == "hard_risk":
        decision.risk_snapshot_json = copy.deepcopy(decision.risk_snapshot_json) | {
            "hard_failures": ["synthetic-current-hard-failure"]
        }

    original_key = decision.execution_key
    with pytest.raises(DecisionConflictError, match=reason):
        DecisionService(db).claim_execution(
            decision.id,
            expected_revision,
            principal=worker(),
            now=now,
        )

    assert decision.execution_key == original_key
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="execution_claimed").count() == 0


def test_claim_next_selects_only_the_authorized_account(db):
    decision, _ = approved_execution(db)

    assert DecisionService(db).claim_next_approved(
        "paper:pilot",
        principal=worker(),
        now=CLAIM_TIME,
    )["decision_id"] == str(decision.id)
    assert decision.execution_key is not None

    with pytest.raises(HTTPException) as error:
        DecisionService(db).claim_next_approved(
            "paper:other",
            principal=worker(),
            now=CLAIM_TIME,
        )
    assert error.value.status_code == 403
