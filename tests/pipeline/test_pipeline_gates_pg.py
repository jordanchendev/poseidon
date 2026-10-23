"""Step 3b against PostgreSQL: gate stages under the real dispatcher (family ledger, holdout events)."""

from __future__ import annotations

import uuid

import pytest

from poseidon.pipeline.family import HOLDOUT_EVALUATED
from poseidon.pipeline.plan import DEFAULT_STAGE_PLAN, GATE_STAGES
from tests.pipeline.conftest import STEP2_PLAN, Harness, synthetic_registry
from tests.pipeline.gate_support import passing, stub_train_gates

pytestmark = pytest.mark.usefixtures("pipeline_engine")


def _harness(pipeline_sessions, tmp_path, clock, registry):
    return Harness(pipeline_sessions, tmp_path / "artifacts", clock, registry=registry)


def test_the_default_plan_stops_at_the_first_failing_gate(pipeline_sessions, tmp_path, clock):
    harness = _harness(pipeline_sessions, tmp_path, clock, synthetic_registry())
    run_id = harness.create(stage_plan=None)
    harness.drain()
    run = harness.run(run_id)
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    assert run.config_json["stage_plan"] == list(DEFAULT_STAGE_PLAN)
    # 20 synthetic days cannot satisfy regime_split's 250-day volatility median (nor the holdout's 20 trades).
    assert run.status == "failed"
    failed = [name for name in GATE_STAGES if name in attempts and attempts[name].verdict == "fail"]
    assert len(failed) == 1 and run.reason.startswith(f"{failed[0]}: ")
    gate = attempts[failed[0]]
    assert gate.artifact_json["report"]["summary"].startswith(f"{failed[0]} ❌ ")
    later = GATE_STAGES[GATE_STAGES.index(failed[0]) + 1 :]
    assert not any(name in attempts for name in later)  # fail is final
    assert "holdout" not in attempts and HOLDOUT_EVALUATED not in harness.events(run_id)
    for name in GATE_STAGES[: GATE_STAGES.index(failed[0])]:
        assert attempts[name].verdict == "pass"
        assert attempts[name].artifact_json["report"]["summary"].startswith(f"{name} ✅ ")


def test_a_run_whose_gates_all_pass_completes_with_the_gate_reason(pipeline_sessions, tmp_path, clock):
    registry = stub_train_gates(holdout=passing("holdout", ("optimize", *GATE_STAGES[:5])))
    harness = _harness(pipeline_sessions, tmp_path, clock, registry)
    run_id = harness.create(stage_plan=None)
    harness.drain()
    run = harness.run(run_id)
    assert (run.status, run.reason, run.current_stage) == ("completed", "passed all validation gates", None)


def test_the_holdout_look_is_an_event_and_later_runs_see_it(pipeline_sessions, tmp_path, clock):
    harness = _harness(pipeline_sessions, tmp_path, clock, stub_train_gates())
    first = harness.create(stage_plan=None)
    harness.drain()
    attempts = {row.stage: row for row in harness.attempts(first)}
    assert attempts["holdout"].status == "completed" and attempts["holdout"].verdict in ("pass", "fail")
    assert harness.events(first).count(HOLDOUT_EVALUATED) == 1
    family = harness.run(first).family_id
    report = attempts["holdout"].artifact_json["report"]
    assert report["inputs"]["family_holdout_evaluations"]["other_runs"] == 0
    assert report["summary"].startswith("holdout ")

    second = harness.create(stage_plan=None, family_id=str(family))
    harness.drain()
    holdout = {row.stage: row for row in harness.attempts(second)}["holdout"]
    evaluations = holdout.artifact_json["report"]["inputs"]["family_holdout_evaluations"]
    assert evaluations["other_runs"] == 1 and evaluations["other_run_ids"] == [str(first)]

    from poseidon.pipeline.models import PipelineEvent

    with pipeline_sessions() as session:
        event = session.query(PipelineEvent).filter_by(run_id=second, event_type=HOLDOUT_EVALUATED).one()
        assert event.stage_run_id == holdout.id and event.payload_json["family_runs_already_evaluated"] == 1


def test_dsr_counts_every_trial_of_the_family_in_the_db(pipeline_sessions, tmp_path, clock):
    from poseidon.pipeline.service import family_trial_count

    harness = _harness(pipeline_sessions, tmp_path, clock, synthetic_registry())
    first = harness.create(stage_plan=STEP2_PLAN)
    harness.drain()
    family = harness.run(first).family_id
    plan = [*STEP2_PLAN, "param_stability", "dsr"]
    second = harness.create(stage_plan=plan, family_id=str(family))
    harness.drain()
    attempts = {row.stage: row for row in harness.attempts(second)}
    if attempts["param_stability"].verdict != "pass":
        pytest.skip("synthetic neighbourhood not stable; dsr not reached")
    inputs = attempts["dsr"].artifact_json["report"]["inputs"]
    with pipeline_sessions() as session:
        total = family_trial_count(session, family)
    assert inputs["n_trials_used"] == total == attempts["dsr"].artifact_json["report"]["gate_report"]["n_trials"]
    assert inputs["n_trials_other_runs"] > 0 and inputs["n_trials_this_run"] > 0


def test_wfe_trials_land_in_the_ledger_under_the_wfe_attempt(pipeline_sessions, tmp_path, clock):
    from poseidon.pipeline.models import PipelineTrial

    registry = stub_train_gates()
    registry["wfe"] = synthetic_registry()["wfe"]
    harness = _harness(pipeline_sessions, tmp_path, clock, registry)
    plan = [*STEP2_PLAN, *GATE_STAGES[:5]]
    run_id = harness.create(stage_plan=plan)
    harness.drain()
    wfe = {row.stage: row for row in harness.attempts(run_id)}["wfe"]
    assert wfe.status == "completed" and wfe.verdict in ("pass", "fail")
    report = wfe.artifact_json["report"]
    with pipeline_sessions() as session:
        rows = session.query(PipelineTrial).filter_by(stage_run_id=wfe.id).order_by(PipelineTrial.trial_no).all()
    if report["reason_code"] == "insufficient_history":
        assert rows == []  # 20 synthetic days are far short of 4 x (730 + 182) days
    else:
        assert [row.trial_no for row in rows] == list(range(1, report["inputs"]["n_trials_recorded"] + 1))


def test_gate_failure_is_final(pipeline_sessions, tmp_path, clock):
    from poseidon.pipeline.service import PipelineConflict, PipelineService

    harness = _harness(pipeline_sessions, tmp_path, clock, synthetic_registry())
    run_id = harness.create(stage_plan=None)
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "failed"
    with pipeline_sessions() as session, pytest.raises(PipelineConflict):
        PipelineService(session, registry=harness.dispatcher.registry, clock=clock).rerun(
            run_id,
            {"expected_revision": run.revision, "reason": "try again"},
            actor_id="human:researcher",
            idempotency_key=uuid.uuid4().hex,
        )
