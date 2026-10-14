"""Mode gating and safety of the Phase 100-07 decision-loop housekeeping tasks."""

import uuid
from datetime import UTC, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from poseidon.models.decision_record import DecisionRecord
from poseidon.research.campaign import HoldoutReuseDenied
from poseidon.workers import cpu_tasks
from tests.test_decision_service import create_decision, decision_inputs
from tests.test_decision_service import db as decision_db

db = decision_db
ACCOUNT = "paper:pilot"
HOLDOUT = {
    "manifest_set": [{"manifest_id": "00000000-0000-0000-0000-00000000a001", "content_sha256": "c" * 64}],
    "evidence_slice": {"fold": "holdout", "slice": "2025-H2"},
    "window": {"start": "2025-07-01", "end": "2025-12-31"},
    "universe": ["2330"],
    "label": {"version": "forward-return-v1", "horizons": ["1_session"]},
    "benchmark": {"symbol": "0050"},
    "purge_gap": {"gap_eligible_sessions": 20, "overlap_method": "purged_embargo"},
}
LOOP_TASKS = {
    "freeze_market_manifest": ("tw_stock", "1d", "2026-09-26T12:00:00Z"),
    "produce_decision_run": (str(uuid.uuid4()), str(uuid.uuid4()), "2026-09-26T12:00:00Z"),
    "produce_decision_runs_for_market": ("tw_stock", "1d", "2026-09-26T12:00:00Z"),
    "expire_due_decisions": (),
    "label_decision_outcomes": (),
    "run_review_campaign": (str(uuid.uuid4()), HOLDOUT),
    "scheduled_recover_decision_execution": (),
    "scheduled_reconcile_paper_account": (),
}


def _settings(mode, **changes):
    values = {
        "decision_loop_execution_mode": mode,
        "decision_loop_approved_account_scope": ACCOUNT,
        "decision_loop_approved_market": "tw_stock",
        "decision_loop_approved_account_generation": "generation-1",
        "decision_loop_worker_principal_id": "decision-worker:producer",
        "decision_loop_label_definition_version": "label-v1",
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", ["legacy", "halted"])
@pytest.mark.parametrize("task_name", sorted(LOOP_TASKS))
def test_every_loop_task_is_a_write_free_no_op_outside_shadow_and_decision(monkeypatch, mode, task_name):
    monkeypatch.setattr(cpu_tasks, "settings", _settings(mode))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail(f"{task_name} opened a session in {mode}"))
    for downstream in ("recover_decision_execution", "reconcile_paper_account", "produce_decision_run"):
        task = getattr(cpu_tasks, downstream)
        if downstream != task_name:
            monkeypatch.setattr(task, "run", lambda *args, _name=downstream, **kwargs: pytest.fail(f"{_name} ran"))
            monkeypatch.setattr(task, "delay", lambda *args, _name=downstream, **kwargs: pytest.fail(f"{_name} queued"))
    monkeypatch.setattr(
        "poseidon.research.campaign.run_paired_review_in_transactions",
        lambda *args, **kwargs: pytest.fail("review ran outside shadow/decision"),
    )
    assert getattr(cpu_tasks, task_name).run(*LOOP_TASKS[task_name]) == {"skipped": mode}


def test_new_loop_tasks_never_retry():
    for task_name in LOOP_TASKS:
        task = getattr(cpu_tasks, task_name)
        assert task.max_retries == 0, task_name
        assert not getattr(task, "autoretry_for", ()), task_name


def test_expire_due_decisions_expires_only_due_decisions(monkeypatch, db):
    run, _version, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    db.commit()
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision"))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: Session(db.get_bind()))
    valid_until = decision.valid_until
    valid_until = valid_until.replace(tzinfo=UTC) if valid_until.tzinfo is None else valid_until
    assert cpu_tasks.expire_due_decisions.run((valid_until - timedelta(minutes=1)).isoformat())["expired"] == []
    result = cpu_tasks.expire_due_decisions.run((valid_until + timedelta(minutes=1)).isoformat())
    assert result["expired"] == [str(decision.id)]
    db.expire_all()
    assert db.get(DecisionRecord, decision.id).status == "expired"


def test_label_decision_outcomes_requires_a_label_contract_version(monkeypatch):
    monkeypatch.setattr(cpu_tasks, "settings", _settings("shadow", decision_loop_label_definition_version=""))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail("unset label version opened a session"))
    assert cpu_tasks.label_decision_outcomes.run() == {"skipped": "label_definition_version_unset"}


def _seed_outcome_inputs(session_factory, cutoff):
    from poseidon.decision_loop.evaluation import snapshot_payload
    from poseidon.decision_loop.manifest import ManifestService, content_sha256
    from poseidon.models.evaluation_run import EvaluationRun
    from poseidon.models.evaluation_snapshot import EvaluationSnapshot
    from poseidon.models.outcome import OutcomeLabelContract
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
    from tests.test_outcomes import ANCHOR, _contract_json, _evaluation_manifest_request, _outcome_manifest_request

    marker = uuid.uuid4().hex
    symbol = f"L{marker[:7]}"
    with session_factory() as session:
        strategy = StrategyRecord(
            name=f"phase100-loop-{marker}",
            strategy_type="technical",
            symbol=symbol,
            market="tw_stock",
            interval="1d",
        )
        session.add(strategy)
        session.flush()
        version = StrategyVersion(
            strategy_id=strategy.id,
            version_no=1,
            config_json={"fixture": marker},
            policy_json={"fixture": marker},
            artifact_json={},
            content_sha256=strategy_version_digest({"fixture": marker}, {"fixture": marker}, {}),
        )
        session.add(version)
        evaluation_request = _evaluation_manifest_request(marker)
        evaluation_manifest = ManifestService(session).freeze(evaluation_request)
        outcome_request = _outcome_manifest_request(cutoff, calendar_identity=f"XNYS:2026a:{marker}")
        for item in outcome_request["evidence"]:
            if item["kind"] == "price":
                item["payload"]["symbol"] = symbol
                item["content_sha256"] = content_sha256(item["payload"])
        ManifestService(session).freeze(outcome_request)
        contract_json = _contract_json(marker)
        session.add(
            OutcomeLabelContract(
                version=f"label-{marker}",
                contract_json=contract_json,
                contract_sha256=content_sha256(contract_json),
            )
        )
        session.flush()
        run = EvaluationRun(
            strategy_version_id=version.id,
            manifest_id=evaluation_manifest.id,
            decision_as_of=ANCHOR,
            universe_json=[{"symbol": symbol, "market": "tw_stock", "instrument": "spot"}],
            input_sha256=content_sha256({"loop-evaluation": marker}),
            status="complete",
            coverage_json={"total": 1, "terminal": 1, "by_status": {"evaluated": 1}},
        )
        session.add(run)
        session.flush()
        body = {
            "symbol": symbol,
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
            evaluation_run_id=run.id, content_sha256=content_sha256(snapshot_payload(body)), **body
        )
        session.add(snapshot)
        session.commit()
        return snapshot.id, f"label-{marker}"


@pytest.mark.postgresql
def test_label_decision_outcomes_never_labels_before_maturity(monkeypatch, phase99_session_factory):
    from poseidon.decision_loop.manifest import iso_time
    from poseidon.models.outcome import OutcomeRecord
    from tests.test_outcomes import ANCHOR

    cutoff = ANCHOR + timedelta(hours=4)  # Friday evening: the next session has not happened.
    snapshot_id, version = _seed_outcome_inputs(phase99_session_factory, cutoff)
    monkeypatch.setattr(cpu_tasks, "settings", _settings("shadow", decision_loop_label_definition_version=version))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", phase99_session_factory)
    result = cpu_tasks.label_decision_outcomes.run(iso_time(cutoff, "cutoff"))
    assert result["outcomes"] == 0
    assert result["snapshots"] == 0
    with phase99_session_factory() as session:
        assert session.query(OutcomeRecord).filter_by(evaluation_snapshot_id=snapshot_id).count() == 0


@pytest.mark.postgresql
def test_label_decision_outcomes_labels_only_mature_horizons_once(monkeypatch, phase99_session_factory):
    from poseidon.decision_loop.manifest import iso_time
    from poseidon.models.outcome import OutcomeRecord
    from tests.test_outcomes import MONDAY

    snapshot_id, version = _seed_outcome_inputs(phase99_session_factory, MONDAY)
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision", decision_loop_label_definition_version=version))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", phase99_session_factory)
    first = cpu_tasks.label_decision_outcomes.run(iso_time(MONDAY, "cutoff"))
    assert first["snapshots"] == 1
    with phase99_session_factory() as session:
        kinds = [row.kind for row in session.query(OutcomeRecord).filter_by(evaluation_snapshot_id=snapshot_id)]
    assert kinds == ["signal"]
    replay = cpu_tasks.label_decision_outcomes.run(iso_time(MONDAY, "cutoff"))
    assert replay["snapshots"] == 0
    assert replay["outcomes"] == 0


@pytest.mark.postgresql
def test_label_decision_outcomes_fails_clearly_without_the_contract_row(monkeypatch, phase99_session_factory):
    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        _settings("shadow", decision_loop_label_definition_version=f"missing-{uuid.uuid4().hex}"),
    )
    monkeypatch.setattr(cpu_tasks, "SessionLocal", phase99_session_factory)
    with pytest.raises(ValueError, match="must resolve exactly once"):
        cpu_tasks.label_decision_outcomes.run("2026-01-05T16:00:00Z")


class _CampaignSession:
    def __init__(self, campaign, manifests=None):
        self.campaign = campaign
        self.manifests = manifests or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, model, key):
        from poseidon.models.experiment_campaign import ExperimentCampaign

        if model is ExperimentCampaign:
            return self.campaign if key == self.campaign.id else None
        return self.manifests.get(key)


def _campaign(manifest_set=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        contract_json={"manifest_set": manifest_set if manifest_set is not None else HOLDOUT["manifest_set"]},
    )


def test_run_review_campaign_surfaces_holdout_reuse_as_task_failure(monkeypatch):
    campaign = _campaign()
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision"))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _CampaignSession(campaign))
    monkeypatch.setattr(cpu_tasks, "_holdout_manifest_ids", lambda session, campaign, holdout: [])
    owner = uuid.uuid4()

    def deny(*_args, **_kwargs):
        raise HoldoutReuseDenied("d" * 64, owner)

    monkeypatch.setattr("poseidon.research.campaign.run_paired_review_in_transactions", deny)
    with pytest.raises(HoldoutReuseDenied) as error:
        cpu_tasks.run_review_campaign.run(str(campaign.id), HOLDOUT)
    assert error.value.owner_campaign_id == owner


def test_run_review_campaign_reads_holdout_payload_only_after_consumption(monkeypatch):
    campaign = _campaign()
    monkeypatch.setattr(cpu_tasks, "settings", _settings("shadow"))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _CampaignSession(campaign))
    monkeypatch.setattr(cpu_tasks, "_holdout_manifest_ids", lambda session, campaign, holdout: ["manifest"])
    panel_reads = []
    monkeypatch.setattr(cpu_tasks, "_holdout_panel", lambda ids, holdout: panel_reads.append(ids) or "panel")
    observed = {}

    def review(session_factory, campaign_id, digest, loader):
        observed["reads_before_consumption"] = len(panel_reads)
        observed["panel"] = loader(permit=object())
        return SimpleNamespace(id=uuid.uuid4(), status="inconclusive")

    monkeypatch.setattr("poseidon.research.campaign.run_paired_review_in_transactions", review)
    result = cpu_tasks.run_review_campaign.run(str(campaign.id), HOLDOUT)
    assert observed == {"reads_before_consumption": 0, "panel": "panel"}
    assert result["status"] == "inconclusive"
    assert len(result["holdout_identity_sha256"]) == 64


def test_run_review_campaign_rejects_holdout_outside_frozen_contract_before_consumption(monkeypatch):
    campaign = _campaign(manifest_set=[{"manifest_id": str(uuid.uuid4()), "content_sha256": "e" * 64}])
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision"))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _CampaignSession(campaign))
    monkeypatch.setattr(
        "poseidon.research.campaign.run_paired_review_in_transactions",
        lambda *args, **kwargs: pytest.fail("holdout consumed despite failed preflight"),
    )
    with pytest.raises(ValueError, match="not part of the frozen campaign contract"):
        cpu_tasks.run_review_campaign.run(str(campaign.id), HOLDOUT)


def test_run_review_campaign_requires_declared_review_panel_manifest(monkeypatch):
    manifest_id = uuid.UUID(HOLDOUT["manifest_set"][0]["manifest_id"])
    manifest = SimpleNamespace(id=manifest_id, content_sha256="c" * 64, capability_json={"ohlcv": True})
    campaign = _campaign()
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision"))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _CampaignSession(campaign, {manifest_id: manifest}))
    monkeypatch.setattr(
        "poseidon.research.campaign.run_paired_review_in_transactions",
        lambda *args, **kwargs: pytest.fail("holdout consumed despite failed preflight"),
    )
    with pytest.raises(ValueError, match="review_panel"):
        cpu_tasks.run_review_campaign.run(str(campaign.id), HOLDOUT)


def test_scheduled_recovery_delegates_only_in_producing_modes(monkeypatch):
    calls = []
    monkeypatch.setattr(cpu_tasks, "settings", _settings("shadow"))
    monkeypatch.setattr(cpu_tasks.recover_decision_execution, "run", lambda: calls.append("recover") or {"enqueued": 0})
    assert cpu_tasks.scheduled_recover_decision_execution.run() == {"enqueued": 0}
    assert calls == ["recover"]


class _AccountSession:
    def __init__(self, account_id):
        self.account_id = account_id

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def scalar(self, _statement):
        return self.account_id


def test_scheduled_reconciliation_targets_only_the_approved_account(monkeypatch):
    account_id = uuid.uuid4()
    calls = []
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision"))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _AccountSession(account_id))
    monkeypatch.setattr(cpu_tasks.reconcile_paper_account, "run", lambda value: calls.append(value) or {"ok": True})
    assert cpu_tasks.scheduled_reconcile_paper_account.run() == {"ok": True}
    assert calls == [str(account_id)]

    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _AccountSession(None))
    assert cpu_tasks.scheduled_reconcile_paper_account.run() == {"skipped": "paper_account_missing"}
    monkeypatch.setattr(cpu_tasks, "settings", _settings("decision", decision_loop_approved_account_scope=""))
    assert cpu_tasks.scheduled_reconcile_paper_account.run() == {"skipped": "account_not_approved"}
    assert calls == [str(account_id)]


def test_beat_schedules_loop_tasks_without_touching_legacy_entries():
    from celery.schedules import crontab

    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert schedule["evaluate-strategies-tw-daily"] == {
        "task": "poseidon.workers.cpu_tasks.evaluate_active_strategies",
        "schedule": crontab(hour=6, minute=10),
        "kwargs": {"market": "tw_stock", "interval": "1d"},
    }
    expected = {
        "decision-loop-recover-execution": "scheduled_recover_decision_execution",
        "decision-loop-reconcile-paper-account": "scheduled_reconcile_paper_account",
        "decision-loop-expire-decisions": "expire_due_decisions",
        "decision-loop-produce-tw-daily": "produce_decision_runs_for_market",
        "decision-loop-label-outcomes": "label_decision_outcomes",
    }
    for entry, task in expected.items():
        assert schedule[entry]["task"] == f"poseidon.workers.cpu_tasks.{task}"
        assert callable(getattr(cpu_tasks, task).run)
    assert schedule["decision-loop-produce-tw-daily"]["kwargs"] == {"market": "tw_stock", "interval": "1d"}
    assert schedule["decision-loop-produce-tw-daily"]["schedule"] == crontab(hour=6, minute=40)
    assert schedule["decision-loop-label-outcomes"]["schedule"] == crontab(hour=7, minute=30)
