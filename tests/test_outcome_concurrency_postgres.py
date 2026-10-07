"""Real PostgreSQL serialization proofs for append-only outcomes."""

import inspect
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from poseidon.decision_loop.evaluation import snapshot_payload
from poseidon.decision_loop.manifest import ManifestService, content_sha256
from poseidon.decision_loop.outcomes import OutcomeService
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.outcome import OutcomeLabelContract, OutcomeRecord
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest

NOW = datetime(2026, 1, 6, 16, tzinfo=UTC)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def _seed_dependencies(factory):
    marker = uuid.uuid4().hex
    with factory() as session, session.begin():
        strategy = StrategyRecord(
            name=f"phase99-outcome-race-{marker}",
            strategy_type="technical",
            symbol="PH99",
            market="tw_stock",
            interval="1d",
        )
        session.add(strategy)
        session.flush()
        policy = {"fixture": marker}
        version = StrategyVersion(
            strategy_id=strategy.id,
            version_no=1,
            config_json={"fixture": marker},
            policy_json=policy,
            artifact_json={},
            content_sha256=strategy_version_digest({"fixture": marker}, policy, {}),
        )
        session.add(version)
        manifest = ManifestService(session).freeze(
            {
                "market": "tw_stock",
                "interval": "1d",
                "as_of": _iso(NOW),
                "account_scope": f"paper:phase99-race:{marker}",
                "capability_json": {"calendar": True},
                "required_data": {"calendar": {"max_age_seconds": 60}},
                "evidence": [
                    {
                        "id": "calendar",
                        "kind": "calendar",
                        "source_uri": f"fixture://{marker}",
                        "event_time": _iso(NOW),
                        "available_at": _iso(NOW),
                        "recorded_at": _iso(NOW),
                        "payload": {"identity": marker},
                        "content_sha256": content_sha256({"identity": marker}),
                    }
                ],
            }
        )
        contract_json = {
            "calendar": {"identity": marker, "evidence_id": "calendar"},
            "horizons": {
                "signal": [{"key": "s1", "anchor": "decision_as_of", "session_offset": 1}],
                "trade": [{"key": "t1", "anchor": "decision_as_of", "session_offset": 1}],
                "research": [{"key": "r1", "anchor": "decision_as_of", "session_offset": 1}],
            },
            "benchmark": {"symbol": "SPY", "evidence_id": "benchmark"},
            "reporting_currency": "USD",
            "required_cost_components": ["commission"],
            "cost_model_version": "paper-cost-v1",
            "fx_model_version": "wm-close-v1",
            "pnl_tolerance": "0.000000000000000001",
            "research_assessment": {
                "expiry_horizon_key": "r1",
                "statuses": ["confirmed", "not_confirmed", "unavailable"],
            },
            "counterfactual_assumption_versions": ["paper-execution-v1"],
        }
        contract = OutcomeLabelContract(
            version=f"race-{marker}",
            contract_json=contract_json,
            contract_sha256=content_sha256(contract_json),
        )
        session.add(contract)
        session.flush()
        run = EvaluationRun(
            strategy_version_id=version.id,
            manifest_id=manifest.id,
            decision_as_of=NOW,
            universe_json=[{"symbol": "PH99", "market": "tw_stock", "instrument": "spot"}],
            input_sha256=content_sha256({"run": marker}),
            status="complete",
            coverage_json={"total": 1, "terminal": 1, "by_status": {"evaluated": 1}},
        )
        session.add(run)
        session.flush()
        snapshot_data = {
            "symbol": "PH99",
            "market": "tw_stock",
            "instrument": "spot",
            "status": "evaluated",
            "recommendation_json": {"research_status": "not_required"},
            "technical_json": {},
            "research_revision_ids": [],
            "reason_codes": [],
            "valid_until": None,
        }
        snapshot = EvaluationSnapshot(
            evaluation_run_id=run.id,
            content_sha256=content_sha256(snapshot_payload(snapshot_data)),
            **snapshot_data,
        )
        session.add(snapshot)
        session.flush()
        decision = DecisionRecord(
            evaluation_run_id=run.id,
            strategy_version_id=version.id,
            account_scope=f"paper:phase99-race:{marker}",
            decision_as_of=NOW,
            valid_until=NOW + timedelta(days=1),
            status="approved",
            revision=1,
            creation_sha256=content_sha256({"decision": marker}),
            policy_sha256=content_sha256(policy),
            original_json={},
            final_json={},
            portfolio_snapshot_json={},
            risk_snapshot_json={},
        )
        session.add(decision)
        session.flush()
        return snapshot.id, contract.id, manifest.id, decision.id


def _item(dependencies, input_marker):
    snapshot_id, contract_id, manifest_id, decision_id = dependencies
    logical = content_sha256({"logical": str(snapshot_id), "kind": "trade"})
    return {
        "evaluation_snapshot_id": snapshot_id,
        "kind": "trade",
        "label_contract_id": contract_id,
        "horizon_key": "t1",
        "manifest_id": manifest_id,
        "decision_id": decision_id,
        "logical_key_sha256": logical,
        "input_sha256": content_sha256({"logical": logical, "input": input_marker}),
        "maturity_at": NOW,
        "status": "provisional",
        "reason_code": input_marker,
        "metrics_json": {
            "actual": {"status": "provisional", "reason": input_marker},
            "counterfactual": [],
        },
    }


def _append(session, item, barrier):
    barrier.wait(timeout=10)
    with session.begin():
        service = OutcomeService(session)
        service._lock_digests([item["logical_key_sha256"]])
        row = service._append(item)
        result = row.id, row.content_sha256, row.input_sha256
    return result


def test_concurrent_same_input_returns_one_identical_outcome(
    phase99_session_factory,
    phase99_two_sessions,
    phase99_barrier,
):
    dependencies = _seed_dependencies(phase99_session_factory)
    item = _item(dependencies, "same")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_append, session, item, phase99_barrier) for session in phase99_two_sessions]
        results = [future.result(timeout=20) for future in futures]
    assert results[0] == results[1]
    with phase99_session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(OutcomeRecord).where(
                OutcomeRecord.logical_key_sha256 == item["logical_key_sha256"]
            )
        ) == 1


def test_concurrent_outcome_corrections_form_one_unbranched_chain(
    phase99_session_factory,
    phase99_two_sessions,
    phase99_barrier,
):
    dependencies = _seed_dependencies(phase99_session_factory)
    initial = _item(dependencies, "initial")
    with phase99_session_factory() as session, session.begin():
        service = OutcomeService(session)
        service._lock_digests([initial["logical_key_sha256"]])
        service._append(initial)
    corrections = [_item(dependencies, marker) for marker in ("correction-a", "correction-b")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_append, session, item, phase99_barrier)
            for session, item in zip(phase99_two_sessions, corrections, strict=True)
        ]
        inserted = [future.result(timeout=20) for future in futures]
    with phase99_session_factory() as session:
        rows = session.scalars(
            select(OutcomeRecord)
            .where(OutcomeRecord.logical_key_sha256 == initial["logical_key_sha256"])
            .order_by(OutcomeRecord.revision_no)
        ).all()
        assert [row.revision_no for row in rows] == [1, 2, 3]
        assert [row.previous_outcome_id for row in rows] == [None, rows[0].id, rows[1].id]
        assert session.execute(
            select(OutcomeRecord.previous_outcome_id, func.count())
            .where(OutcomeRecord.previous_outcome_id.is_not(None))
            .group_by(OutcomeRecord.previous_outcome_id)
            .having(func.count() > 1)
        ).all() == []

    replayed = []
    with phase99_session_factory() as session, session.begin():
        for item in corrections:
            service = OutcomeService(session)
            service._lock_digests([item["logical_key_sha256"]])
            row = service._append(item)
            replayed.append((row.id, row.content_sha256, row.input_sha256))
    assert set(replayed) == set(inserted)

    with phase99_session_factory() as session, session.begin():
        rows = session.scalars(
            select(OutcomeRecord)
            .where(OutcomeRecord.logical_key_sha256 == initial["logical_key_sha256"])
            .order_by(OutcomeRecord.revision_no)
        ).all()
        branch = OutcomeRecord(
            **{
                **corrections[0],
                "input_sha256": content_sha256({"branch": str(rows[0].id)}),
                "content_sha256": content_sha256({"branch-content": str(rows[0].id)}),
                "revision_no": 2,
                "previous_outcome_id": rows[0].id,
                "previous_revision_no": 1,
            }
        )
        with pytest.raises(IntegrityError), session.begin_nested():
            session.add(branch)
            session.flush()


def test_batch_locking_sorts_digests_before_any_append():
    source = inspect.getsource(OutcomeService)
    assert "for digest in sorted(logical_digests)" in source
    assert source.index("self._lock_digests") < source.index("return [self._append(item)")
