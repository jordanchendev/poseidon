"""Step 2 through the dispatcher on real PostgreSQL: real backtest, optimize and the trial ledger."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from poseidon.pipeline.models import PipelineTrial
from poseidon.pipeline.service import PipelineService, family_trial_count
from tests.pipeline.conftest import STEP2_PLAN, Harness, SyntheticTxSource, orb_template, synthetic_registry

pytestmark = pytest.mark.postgresql


def _trials(harness, run_id):
    with harness.sessions() as session:
        rows = session.query(PipelineTrial).filter_by(run_id=run_id).order_by(PipelineTrial.trial_no).all()
        for row in rows:
            session.expunge(row)
        return rows


def _optimize_attempt(harness, run_id):
    return next(row for row in harness.attempts(run_id) if row.stage == "optimize")


def test_step2_plan_completes_and_every_trial_is_in_the_ledger(harness):
    run_id = harness.create(stage_plan=STEP2_PLAN)
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "completed", run.reason
    assert [(row.stage, row.verdict) for row in harness.attempts(run_id)] == [
        ("spec_generate", "pass"),
        ("spec_validate", "pass"),
        ("backtest", "pass"),
        ("optimize", "pass"),
    ]
    optimize = _optimize_attempt(harness, run_id)
    report = optimize.artifact_json["report"]
    trials = _trials(harness, run_id)
    assert len(trials) == report["n_trials"] == 6
    assert {row.stage_run_id for row in trials} == {optimize.id}
    assert {row.family_id for row in trials} == {run.family_id}
    assert [row.trial_no for row in trials] == list(range(1, 7))
    assert all(
        row.state in {"complete", "pruned"} and row.window_sha256 == report["data"]["window_sha256"] for row in trials
    )
    complete = [row for row in trials if row.state == "complete"]
    assert all(row.n_obs and row.kurtosis is not None for row in complete)
    assert report["best"]["trial_no"] == max(complete, key=lambda row: row.sharpe).trial_no
    assert {entry["name"] for entry in optimize.artifact_json["files"]} >= {"trials.json", "best_trades.csv"}
    assert "trials in" in optimize.reason
    with harness.sessions() as session:
        assert family_trial_count(session, run.family_id) == 6
        count, rows = PipelineService(session).family_trials(run.family_id)
    assert count == len(rows) == 6


def test_identical_runs_reproduce_backtest_and_optimize_hashes(harness):
    first = harness.create(stage_plan=STEP2_PLAN)
    second = harness.create(stage_plan=STEP2_PLAN)
    harness.drain()
    a = {row.stage: row for row in harness.attempts(first)}
    b = {row.stage: row for row in harness.attempts(second)}
    for stage in ("backtest", "optimize"):
        assert (a[stage].input_sha256, a[stage].artifact_sha256) == (b[stage].input_sha256, b[stage].artifact_sha256)
    assert len(_trials(harness, first)) == len(_trials(harness, second)) == 6
    with harness.sessions() as session:  # two runs, two families, each counted on its own
        assert family_trial_count(session, harness.run(first).family_id) == 6


def test_failed_trials_are_in_the_db_ledger(harness, monkeypatch):
    from poseidon.pipeline import optimization
    from poseidon.pipeline.backtesting import BacktestFailed

    real = optimization.run_backtest

    def flaky(strategy, bars, config, **kwargs):
        if strategy.config.risk.take_profit.value == 1.0:
            raise BacktestFailed("synthetic failure")
        return real(strategy, bars, config, **kwargs)

    monkeypatch.setattr(optimization, "run_backtest", flaky)
    run_id = harness.create(stage_plan=STEP2_PLAN)
    harness.drain()
    assert harness.run(run_id).status == "completed"
    trials = _trials(harness, run_id)
    failed = [row for row in trials if row.state == "fail"]
    assert len(trials) == 6 and len(failed) == 2
    assert all(row.sharpe is None and row.params_json["tp_r"] == 1.0 for row in failed)


def test_trials_of_an_errored_optimize_attempt_still_count(harness, monkeypatch):
    from poseidon.pipeline import optimization

    real = optimization.TrialRunner.evaluate
    calls = {"n": 0}

    def explode_on_third(self, params, source):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("worker bug")
        return real(self, params, source)

    monkeypatch.setattr(optimization.TrialRunner, "evaluate", explode_on_third)
    run_id = harness.create(stage_plan=STEP2_PLAN)
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "error" and "worker bug" in run.reason
    assert len(_trials(harness, run_id)) == 2
    with harness.sessions() as session:
        assert family_trial_count(session, run.family_id) == 2


def test_abort_during_optimize_stops_within_one_heartbeat(pipeline_sessions, tmp_path, clock):
    heartbeat = 0.25
    params = {
        "or_min": {"type": "int", "low": 15, "high": 60, "step": 15, "default": 30},
        "tp_r": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5, "default": 2.0},
    }
    harness = Harness(
        pipeline_sessions,
        tmp_path,
        clock,
        registry=synthetic_registry(SyntheticTxSource(days_before=40)),
        heartbeat_seconds=heartbeat,
    )
    run_id = harness.create(spec=orb_template(params), stage_plan=STEP2_PLAN)
    for _ in range(3):  # spec_generate, spec_validate, backtest
        harness.dispatcher.advance(harness.queue.pop(0))

    from poseidon.pipeline import optimization

    trial_started = threading.Event()
    real = optimization.run_backtest
    timings: dict[str, float] = {}

    def tracked(strategy, bars, config, **kwargs):
        if len(_trials(harness, run_id)) >= 1:
            trial_started.set()
        return real(strategy, bars, config, **kwargs)

    optimization_run_backtest = optimization.run_backtest
    optimization.run_backtest = tracked
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(harness.dispatcher.advance, harness.queue.pop(0))
            assert trial_started.wait(60), "optimize never reached its second trial"
            time.sleep(0.05)  # inside the second trial's backtest
            rows_at_abort = len(_trials(harness, run_id))
            from tests.pipeline.test_pipeline_dispatcher_pg import _abort

            timings["abort"] = time.monotonic()
            _abort(harness, run_id)
            outcome = future.result(timeout=60)
            timings["stopped"] = time.monotonic()
    finally:
        optimization.run_backtest = optimization_run_backtest

    latency = timings["stopped"] - timings["abort"]
    assert outcome["fenced"] == "aborted"
    assert latency < heartbeat + 1.0, f"stage stopped {latency:.2f}s after the abort"
    optimize = _optimize_attempt(harness, run_id)
    assert (optimize.status, optimize.verdict) == ("aborted", None)
    assert harness.run(run_id).status == "aborted"
    # At most the in-flight trial lands after the abort; the search (20 points) did not go on.
    assert rows_at_abort <= len(_trials(harness, run_id)) <= rows_at_abort + 1 < 20
    print(f"abort->stopped latency {latency:.3f}s (heartbeat {heartbeat}s)")
