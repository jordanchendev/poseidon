"""Dispatcher end-to-end on real PostgreSQL with a direct-call (Celery-free) harness."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import pytest
from sqlalchemy import text

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline.models import PipelineRun, PipelineTrial
from poseidon.pipeline.service import PipelineConflict, PipelineService, family_trial_count
from poseidon.pipeline.stages import (
    STAGE_REGISTRY,
    BaseStage,
    StageResult,
    TransientStageError,
)
from tests.pipeline.conftest import STEP1_PLAN, Harness, synthetic_registry, tx_spec, tx_v1_spec

pytestmark = pytest.mark.postgresql


def _registry(**stages):
    return synthetic_registry(**stages)


class ScriptedBacktest(BaseStage):
    """Replaces ``backtest``; each call pops the next behaviour."""

    name = "backtest"
    requires = ("spec_validate",)

    def __init__(self, *behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        behaviour = self.behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        if callable(behaviour):
            return behaviour(ctx)
        return behaviour


def _rerun(harness, run_id, **body):
    run = harness.run(run_id)
    with harness.sessions() as session:
        response, _ = PipelineService(session, registry=harness.dispatcher.registry, clock=harness.clock).rerun(
            run_id,
            {"from_stage": None, "expected_revision": run.revision, "reason": "retry", **body},
            actor_id="human:researcher",
            idempotency_key=uuid.uuid4().hex,
        )
        session.commit()
    harness.queue.append(run_id)
    return response


def _abort(harness, run_id):
    run = harness.run(run_id)
    with harness.sessions() as session:
        response, _ = PipelineService(session, clock=harness.clock).abort(
            run_id,
            {"expected_revision": run.revision, "reason": "owner abort"},
            actor_id="human:researcher",
            idempotency_key=uuid.uuid4().hex,
        )
        session.commit()
    return response


def _summary(harness, run_id):
    return [(row.stage, row.attempt, row.trigger, row.status, row.verdict) for row in harness.attempts(run_id)]


def test_hand_written_tx_spec_walks_the_step1_plan_to_completed(harness):
    run_id = harness.create()
    results = harness.drain()

    run = harness.run(run_id)
    assert (run.status, run.current_stage, run.reason) == ("completed", None, "all stages passed")
    assert run.finished_at is not None and run.config_json["stage_plan"] == STEP1_PLAN
    assert [result["verdict"] for result in results] == ["pass", "pass", "pass"]
    assert _summary(harness, run_id) == [
        ("spec_generate", 1, "advance", "completed", "pass"),
        ("spec_validate", 1, "advance", "completed", "pass"),
        ("backtest", 1, "advance", "completed", "pass"),
    ]
    attempts = harness.attempts(run_id)
    assert all(row.code_version == "test-sha" and len(row.input_sha256) == 64 for row in attempts)
    assert all(row.artifact_sha256 == content_sha256(row.artifact_json) for row in attempts)
    backtest = attempts[-1]
    report = backtest.artifact_json["report"]
    assert report["metrics"]["n_trades"] > 0 and report["data"]["part"] == "train"
    assert report["data"]["last_bar"] < run.config_json["data"]["holdout_start"]
    files = backtest.artifact_json["files"]
    assert [entry["name"] for entry in files] == ["daily_pnl.csv", "trades.csv"]
    directory = Path(urlparse(backtest.artifact_uri).path)
    assert directory.parts[-3:] == (str(run_id), "backtest", "1")
    for entry in files:
        data = (directory / entry["name"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["bytes"] == len(data)
    assert harness.events(run_id) == [
        "run_created",
        "stage_claimed",
        "stage_completed",
        "stage_claimed",
        "stage_completed",
        "stage_claimed",
        "stage_completed",
    ]
    # A redelivered message for a finished run is a no-op.
    assert harness.dispatcher.advance(run_id)["skipped"] == "terminal"
    assert len(harness.attempts(run_id)) == 3


def test_same_spec_and_config_reproduce_the_backtest_hashes(harness):
    first, second = harness.create(), harness.create()
    harness.drain()
    a, b = harness.attempts(first)[-1], harness.attempts(second)[-1]
    assert (a.input_sha256, a.artifact_sha256) == (b.input_sha256, b.artifact_sha256)
    other = harness.create(spec=tx_spec(name="another"))
    harness.drain()
    assert harness.attempts(other)[-1].input_sha256 != a.input_sha256


def test_invalid_spec_fails_validation_and_the_run_is_final(harness):
    run_id = harness.create(spec=tx_spec(market="tw_stock"))
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "failed" and run.reason.startswith("spec_validate: spec invalid")
    validate = harness.attempts(run_id)[1]
    assert validate.verdict == "fail" and validate.artifact_json["report"]["errors"]
    assert [row.stage for row in harness.attempts(run_id)] == ["spec_generate", "spec_validate"]
    with pytest.raises(PipelineConflict, match="final"):
        _rerun(harness, run_id)


def test_unsupported_stops_the_run_and_rerun_is_allowed(harness, monkeypatch):
    # Without the LLM key the generation attempt is an error (no LLM call), which is rerunnable.
    monkeypatch.delenv("STRATEGY_PIPELINE_OPENROUTER_KEY", raising=False)
    run_id = harness.create(spec=None)
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "error" and "STRATEGY_PIPELINE_OPENROUTER_KEY is not set" in run.reason

    response = _rerun(harness, run_id)
    assert response["status"] == "queued" and response["current_stage"] == "spec_generate"
    harness.drain()
    assert _summary(harness, run_id) == [
        ("spec_generate", 1, "advance", "completed", "error"),
        ("spec_generate", 2, "rerun", "completed", "error"),
    ]

    v1 = harness.create(spec=tx_v1_spec())
    harness.drain()
    assert (harness.run(v1).status, harness.run(v1).reason) == (
        "unsupported",
        "spec_validate: pipeline requires dsl_version 2",
    )
    assert harness.attempts(v1)[1].verdict == "unsupported"


def test_shadow_paper_is_registered_but_unsupported(pipeline_sessions, tmp_path, clock):
    from poseidon.pipeline.plan import GATE_STAGES, V1_STAGE_PLAN
    from tests.pipeline.gate_support import passing, stub_train_gates

    registry = stub_train_gates(holdout=passing("holdout", ("optimize", *GATE_STAGES[:5])))
    harness = Harness(pipeline_sessions, tmp_path, clock, registry=registry)
    run_id = harness.create(stage_plan=list(V1_STAGE_PLAN))
    harness.drain()
    run = harness.run(run_id)
    assert run.config_json["stage_plan"][-1] == "awaiting_release"
    assert (run.status, run.reason) == ("unsupported", "shadow_paper: not implemented")
    assert [row.verdict for row in harness.attempts(run_id)] == ["pass"] * 10 + ["unsupported"]


def test_rerun_reuses_upstream_passes_and_rejects_another_from_stage(pipeline_sessions, tmp_path, clock):
    harness = Harness(
        pipeline_sessions,
        tmp_path,
        clock,
        registry=_registry(backtest=ScriptedBacktest(RuntimeError("boom"), StageResult.passed({"ok": 1}))),
    )
    run_id = harness.create()
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "error" and "RuntimeError: boom" in run.reason
    errored = harness.attempts(run_id)[-1]
    assert errored.error_json["kind"] == "exception" and errored.verdict == "error"

    from poseidon.pipeline.service import PipelineRequestInvalid

    with pytest.raises(PipelineRequestInvalid):
        _rerun(harness, run_id, from_stage="spec_validate")
    _rerun(harness, run_id, from_stage="backtest")
    harness.drain()
    assert harness.run(run_id).status == "completed"
    assert _summary(harness, run_id) == [
        ("spec_generate", 1, "advance", "completed", "pass"),
        ("spec_validate", 1, "advance", "completed", "pass"),
        ("backtest", 1, "advance", "completed", "error"),
        ("backtest", 2, "rerun", "completed", "pass"),
    ]


def test_transient_errors_retry_automatically_up_to_two_attempts(pipeline_sessions, tmp_path, clock):
    stage = ScriptedBacktest(TransientStageError("blip"), StageResult.passed({"ok": 1}))
    harness = Harness(pipeline_sessions, tmp_path, clock, registry=_registry(backtest=stage))
    run_id = harness.create()
    harness.drain()
    assert harness.run(run_id).status == "completed"
    assert _summary(harness, run_id)[2:] == [
        ("backtest", 1, "advance", "completed", "error"),
        ("backtest", 2, "retry", "completed", "pass"),
    ]

    stage = ScriptedBacktest(TransientStageError("blip"), TransientStageError("blip again"))
    harness = Harness(pipeline_sessions, tmp_path, clock, registry=_registry(backtest=stage))
    run_id = harness.create()
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "error" and "blip again" in run.reason
    assert [row.trigger for row in harness.attempts(run_id)[2:]] == ["advance", "retry"]


def test_abort_mid_run_fences_the_running_attempt(pipeline_sessions, tmp_path, clock):
    started, stopped = threading.Event(), threading.Event()

    def block_until_stopped(ctx):
        started.set()
        deadline = time.monotonic() + 10
        while not ctx.should_stop():  # the heartbeat flips this after the abort
            if time.monotonic() > deadline:
                raise AssertionError("heartbeat never signalled the abort")
            time.sleep(0.01)
        stopped.set()
        return StageResult.passed({"should": "never be stored"})

    harness = Harness(
        pipeline_sessions, tmp_path, clock, registry=_registry(backtest=ScriptedBacktest(block_until_stopped))
    )
    run_id = harness.create()
    harness.dispatcher.advance(harness.queue.pop(0))
    harness.dispatcher.advance(harness.queue.pop(0))  # spec_generate, spec_validate
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.dispatcher.advance, harness.queue.pop(0))
        assert started.wait(10)
        response = _abort(harness, run_id)
        assert response["status"] == "running" and response["abort_requested_at"] is not None
        outcome = future.result(timeout=15)
    assert stopped.is_set()
    assert outcome["fenced"] == "aborted"
    backtest = harness.attempts(run_id)[-1]
    assert (backtest.status, backtest.verdict, backtest.artifact_json) == ("aborted", None, None)
    assert harness.run(run_id).status == "aborted"
    assert harness.events(run_id)[-2:] == ["run_abort_requested", "stage_aborted"]
    assert harness.queue == []

    _rerun(harness, run_id)  # aborted is rerunnable
    assert _summary(harness, run_id)[-1] == ("backtest", 2, "rerun", "pending", None)


def test_abort_of_a_queued_run_closes_pending_attempts_at_once(harness):
    run_id = harness.create(spec=None)
    harness.drain()
    _rerun(harness, run_id)  # leaves a pending spec_generate attempt
    response = _abort(harness, run_id)
    assert response["status"] == "aborted"
    assert _summary(harness, run_id)[-1] == ("spec_generate", 2, "rerun", "aborted", None)
    assert harness.dispatcher.advance(harness.queue.pop(0))["skipped"] == "terminal"


def test_advance_skips_a_run_another_transaction_holds(harness):
    run_id = harness.create()
    with harness.sessions() as holder:
        holder.execute(text("SELECT id FROM pipeline_runs WHERE id = :id FOR UPDATE"), {"id": run_id})
        assert harness.dispatcher.advance(run_id) == {"run_id": str(run_id), "skipped": "locked"}
        holder.rollback()
    assert harness.attempts(run_id) == []


def test_two_concurrent_advances_claim_exactly_once(pipeline_sessions, tmp_path, clock):
    barrier = threading.Barrier(2)

    def slow(ctx):
        time.sleep(0.5)
        return StageResult.passed({"slow": True})

    class SlowGenerate(BaseStage):
        name = "spec_generate"

        def run(self, ctx):
            return slow(ctx)

    harness = Harness(pipeline_sessions, tmp_path, clock, registry=_registry(spec_generate=SlowGenerate()))
    run_id = harness.create()
    harness.queue.clear()

    def advance():
        barrier.wait(timeout=10)
        return harness.dispatcher.advance(run_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(advance), pool.submit(advance))]
    claimed = [result for result in results if "verdict" in result]
    skipped = [result for result in results if "skipped" in result]
    assert len(claimed) == 1 and len(skipped) == 1
    assert skipped[0]["skipped"] in {"locked", "running"}
    assert _summary(harness, run_id) == [("spec_generate", 1, "advance", "completed", "pass")]
    assert harness.events(run_id).count("stage_claimed") == 1


def test_lease_expiry_is_swept_to_worker_lost_and_a_second_attempt(harness, clock):
    run_id = harness.create()
    harness.queue.clear()
    claim, _ = harness.dispatcher._claim(run_id)  # the worker "dies" after Txn A
    lease_expires = harness.attempts(run_id)[0].lease_expires_at

    assert harness.dispatcher.sweep(now=lease_expires - timedelta(seconds=1))["worker_lost"] == 0
    swept = harness.dispatcher.sweep(now=lease_expires + timedelta(seconds=1))
    assert swept["worker_lost"] == 1 and harness.queue == [run_id]
    lost, retry = harness.attempts(run_id)
    assert (lost.status, lost.verdict, lost.reason, lost.error_json["kind"]) == (
        "completed",
        "error",
        "worker_lost",
        "worker_lost",
    )
    assert (retry.attempt, retry.trigger, retry.status) == (2, "retry", "pending")
    assert harness.run(run_id).status == "queued"

    # The lost worker wakes up late: its Txn B is fenced and changes nothing.
    late = harness.dispatcher._commit(claim, StageResult.passed({"late": True}), None)
    assert late["fenced"] == "lease_lost"
    assert harness.attempts(run_id)[0].verdict == "error"

    harness.drain()
    assert harness.run(run_id).status == "completed"
    assert _summary(harness, run_id)[:2] == [
        ("spec_generate", 1, "advance", "completed", "error"),
        ("spec_generate", 2, "retry", "completed", "pass"),
    ]


def test_second_worker_loss_exhausts_automatic_attempts(harness):
    run_id = harness.create()
    harness.queue.clear()
    for _ in range(2):
        harness.dispatcher._claim(run_id)
        expires = next(row for row in harness.attempts(run_id) if row.status == "running").lease_expires_at
        harness.dispatcher.sweep(now=expires + timedelta(seconds=1))
    run = harness.run(run_id)
    assert (run.status, run.reason) == ("error", "spec_generate: worker_lost")
    assert [row.trigger for row in harness.attempts(run_id)] == ["advance", "retry"]
    _rerun(harness, run_id)
    harness.drain()
    assert harness.run(run_id).status == "completed"


def test_waiting_stage_resumes_after_wait_until(pipeline_sessions, tmp_path, clock):
    def wait_an_hour(ctx):
        return StageResult.waiting(clock.now + timedelta(hours=1), reason="next session close")

    stage = ScriptedBacktest(wait_an_hour, StageResult.passed({"paper_day": 1}))
    harness = Harness(pipeline_sessions, tmp_path, clock, registry=_registry(backtest=stage))
    run_id = harness.create()
    harness.drain()
    assert harness.run(run_id).status == "waiting"
    assert harness.attempts(run_id)[-1].status == "waiting"
    assert harness.dispatcher.advance(run_id)["skipped"] == "waiting"
    assert harness.dispatcher.sweep(now=clock.now)["waiting_due"] == 0

    clock.advance(hours=1, seconds=1)
    assert harness.dispatcher.sweep(now=clock.now)["waiting_due"] == 1
    harness.drain()
    assert harness.run(run_id).status == "completed"
    assert _summary(harness, run_id)[-1] == ("backtest", 1, "advance", "completed", "pass")
    assert stage.calls == 2


def test_sweeper_requeues_idle_queued_runs(harness, clock):
    run_id = harness.create()
    harness.queue.clear()
    assert harness.dispatcher.sweep(now=clock.now + timedelta(minutes=4))["requeued_idle"] == 0
    assert harness.dispatcher.sweep(now=clock.now + timedelta(minutes=6))["requeued_idle"] == 1
    assert harness.queue == [run_id]


def test_family_trial_count_includes_trials_of_errored_attempts(pipeline_sessions, tmp_path, clock):
    harness = Harness(
        pipeline_sessions, tmp_path, clock, registry=_registry(backtest=ScriptedBacktest(RuntimeError("x")))
    )
    run_id = harness.create()
    harness.drain()
    errored = harness.attempts(run_id)[-1]
    variant = harness.create(family_id=str(run_id), spec=tx_spec(name="variant"))
    harness.queue.clear()
    with pipeline_sessions() as session:
        for trial_no in range(3):
            session.add(
                PipelineTrial(
                    family_id=run_id,
                    run_id=run_id,
                    stage_run_id=errored.id,
                    trial_no=trial_no,
                    params_json={"n": trial_no},
                    params_sha256=content_sha256({"n": trial_no}),
                    state="fail",
                    window_sha256="c" * 64,
                )
            )
        session.commit()
        assert session.get(PipelineRun, variant).family_id == run_id
        assert family_trial_count(session, run_id) == 3


def test_default_registry_is_untouched_by_tests():
    assert STAGE_REGISTRY["backtest"].__class__.__name__ == "BacktestStage"
    assert STAGE_REGISTRY["backtest"].data_source is None and STAGE_REGISTRY["optimize"].data_source is None
