"""Real PostgreSQL row-lock proof for decision execution claims."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import create_engine, delete, select
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
