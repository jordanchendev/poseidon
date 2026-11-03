"""Do the gates discriminate? Three controls through the full chain (dispatcher + PostgreSQL, no Thalassa).

* planted edge: synthetic TX-like day-session bars (2020-03 .. 2026-10) where
  the 30-minute opening-range break is followed by a drift; the matching ORB
  spec (2 params) must pass every gate, benchmark included.
* random entry: on the *same* bars, a strategy that enters at seeded random
  bars in a random direction must end ``failed`` at some gate.
* pure beta: bars with no pattern and only an upward drift proportional to
  variance (buy-and-hold rises ~50 points a day); a strategy that is long over
  every overnight gap makes money but has no alpha, and must fail
  ``benchmark`` (in the run, or -- if an earlier gate stops it -- evaluated
  outside the run for the record).

Several minutes each, so opt-in: ``POSEIDON_PIPELINE_GATE_CONTROLS=1`` (plus
``POSEIDON_PIPELINE_TEST_DATABASE_URL``). ``POSEIDON_PIPELINE_GATE_CONTROLS_OUT``
names a directory for the JSON evidence (margins vs thresholds).
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from poseidon.pipeline.family import HOLDOUT_EVALUATED
from poseidon.pipeline.plan import GATE_STAGES, V1_STAGE_PLAN
from poseidon.pipeline.registry import build_default_registry
from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.gate_support import (
    DriftOnlySource,
    PlantedOrbSource,
    RandomEntryFactory,
    diagnose,
    gate_margins,
    overnight_long_spec,
    planted_orb_spec,
    random_entry_spec,
)

pytestmark = [
    pytest.mark.skipif(os.environ.get("POSEIDON_PIPELINE_GATE_CONTROLS") != "1", reason="opt-in (minutes per run)"),
    pytest.mark.usefixtures("pipeline_engine"),
]

# 2026-10-23 09:00 Asia/Taipei: the frozen window is 2020-03-02 .. 2026-10-23.
CONTROL_CLOCK = datetime(2026, 10, 23, 1, 0, tzinfo=UTC)


def _run(pipeline_sessions, tmp_path, spec, *, factory=None, source=None):
    registry = build_default_registry(data_source=source or PlantedOrbSource(), strategy_factory=factory)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(CONTROL_CLOCK), registry=registry)
    # the gates' verdicts only: a plan that ends at holdout (the default continues into shadow_paper)
    plan = list(V1_STAGE_PLAN[: V1_STAGE_PLAN.index("holdout") + 1])
    run_id = harness.create(spec=spec, stage_plan=plan, idea_text="control")
    started = time.monotonic()
    timings = []
    while harness.queue:
        t0 = time.monotonic()
        out = harness.dispatcher.advance(harness.queue.pop(0))
        timings.append((out.get("stage"), out.get("verdict"), round(time.monotonic() - t0, 1)))
    wall = round(time.monotonic() - started, 1)
    run = harness.run(run_id)
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    reports = {name: (row.artifact_json or {}).get("report") for name, row in attempts.items()}
    return harness, run, attempts, reports, timings, wall


def _evidence(name: str, payload: dict) -> None:
    text = json.dumps(payload, indent=1, default=str, ensure_ascii=False)
    print(f"\n=== {name} ===\n{text}")
    out = os.environ.get("POSEIDON_PIPELINE_GATE_CONTROLS_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{name}.json").write_text(text)


def _optimize_summary(reports):
    optimize = reports.get("optimize") or {}
    return {
        "n_trials": optimize.get("n_trials"),
        "best": (optimize.get("best") or {}).get("params"),
        "best_metrics": {
            k: (optimize.get("best") or {}).get("metrics", {}).get(k)
            for k in ("total_pnl_twd", "sharpe_daily", "sharpe_annualised", "n_trades", "n_obs", "win_rate")
        },
        "neighbours": [(n["param"], n["direction"], n["total_pnl_twd"]) for n in optimize.get("neighbours") or []],
    }


def test_planted_edge_passes_every_gate(pipeline_sessions, tmp_path):
    harness, run, attempts, reports, timings, wall = _run(pipeline_sessions, tmp_path, planted_orb_spec())
    payload = {
        "run_status": run.status,
        "run_reason": run.reason,
        "wall_seconds": wall,
        "stage_seconds": timings,
        "optimize": _optimize_summary(reports),
        "verdicts": {name: attempts[name].verdict for name in attempts},
        "margins": gate_margins({name: reports.get(name) for name in GATE_STAGES if name in reports}),
        "holdout_events": harness.events(run.id).count(HOLDOUT_EVALUATED),
    }
    _evidence("planted_edge", payload)
    assert (run.status, run.reason) == ("completed", "passed all validation gates"), payload
    assert all(attempts[name].verdict == "pass" for name in GATE_STAGES)
    assert payload["holdout_events"] == 1


def test_random_entry_is_stopped_by_a_gate(pipeline_sessions, tmp_path):
    harness, run, attempts, reports, timings, wall = _run(
        pipeline_sessions, tmp_path, random_entry_spec(), factory=RandomEntryFactory
    )
    reached = [name for name in GATE_STAGES if name in attempts]
    failed_at = reached[-1] if reached else None
    later_train = [name for name in GATE_STAGES[:-1] if name not in attempts]
    started = time.monotonic()
    diagnostics = diagnose(pipeline_sessions, run.id, later_train, harness.dispatcher.registry)
    payload = {
        "run_status": run.status,
        "run_reason": run.reason,
        "wall_seconds": wall,
        "stage_seconds": timings,
        "failed_at": failed_at,
        "optimize": _optimize_summary(reports),
        "margins": gate_margins({name: reports.get(name) for name in reached}),
        "diagnostics_outside_the_run": {
            "note": "later train gates run directly for reporting only (no attempts, holdout never evaluated)",
            "seconds": round(time.monotonic() - started, 1),
            "verdicts": {name: d["verdict"] for name, d in diagnostics.items()},
            "margins": gate_margins({name: d["report"] for name, d in diagnostics.items()}),
        },
        "holdout_events": harness.events(run.id).count(HOLDOUT_EVALUATED),
    }
    _evidence("random_entry", payload)
    assert run.status == "failed", payload
    assert failed_at is not None and attempts[failed_at].verdict == "fail"
    assert run.reason.startswith(f"{failed_at}: ")
    assert "holdout" not in attempts and payload["holdout_events"] == 0


def test_pure_beta_fails_benchmark(pipeline_sessions, tmp_path):
    harness, run, attempts, reports, timings, wall = _run(
        pipeline_sessions, tmp_path, overnight_long_spec(), source=DriftOnlySource()
    )
    reached = [name for name in GATE_STAGES if name in attempts]
    failed_at = next((name for name in reached if attempts[name].verdict != "pass"), None)
    later_train = [name for name in GATE_STAGES[:-1] if name not in attempts]
    diagnostics = (
        diagnose(pipeline_sessions, run.id, later_train, harness.dispatcher.registry, artifacts_root=tmp_path / "diag")
        if later_train
        else {}
    )
    benchmark_report = reports.get("benchmark") or (diagnostics.get("benchmark") or {}).get("report")
    benchmark_verdict = (
        attempts["benchmark"].verdict if "benchmark" in attempts else diagnostics["benchmark"]["verdict"]
    )
    payload = {
        "run_status": run.status,
        "run_reason": run.reason,
        "wall_seconds": wall,
        "stage_seconds": timings,
        "failed_at": failed_at,
        "optimize": _optimize_summary(reports),
        "margins": gate_margins({name: reports.get(name) for name in reached}),
        "benchmark_in_run": "benchmark" in attempts,
        "benchmark_verdict": benchmark_verdict,
        "benchmark": gate_margins({"benchmark": benchmark_report}).get("benchmark"),
        "diagnostics_outside_the_run": {
            "verdicts": {name: d["verdict"] for name, d in diagnostics.items()},
            "margins": gate_margins({name: d["report"] for name, d in diagnostics.items()}),
        },
        "holdout_events": harness.events(run.id).count(HOLDOUT_EVALUATED),
    }
    _evidence("pure_beta", payload)
    assert run.status == "failed", payload
    assert benchmark_verdict == "fail", payload
    assert benchmark_report["gate_report"]["beta"] > 0, payload
    assert "holdout" not in attempts and payload["holdout_events"] == 0
