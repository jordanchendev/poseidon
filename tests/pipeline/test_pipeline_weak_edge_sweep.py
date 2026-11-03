"""Experiment: how well do the six gates discriminate *weak* planted edges?

The planted-edge control (``test_pipeline_gate_controls``) at several drift
strengths and data seeds. Each case runs the real dispatcher through a plan
ending at holdout; for a failed run, the later *train* gates are evaluated
outside the run for reporting (the locked holdout is never evaluated for a
failed candidate). Records evidence only -- no assertion on the verdict.

Opt-in: ``POSEIDON_PIPELINE_WEAK_EDGE=1`` + ``POSEIDON_PIPELINE_TEST_DATABASE_URL``.
``POSEIDON_WEAK_EDGE_DRIFTS`` (comma list, points) and ``POSEIDON_WEAK_EDGE_SEEDS``
(comma list) choose the cases; JSON evidence goes to ``POSEIDON_WEAK_EDGE_OUT``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from poseidon.pipeline.plan import GATE_STAGES, V1_STAGE_PLAN
from poseidon.pipeline.registry import build_default_registry
from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.gate_support import PlantedOrbSource, diagnose, gate_margins, planted_orb_spec
from tests.pipeline.test_pipeline_gate_controls import CONTROL_CLOCK

pytestmark = [
    pytest.mark.skipif(os.environ.get("POSEIDON_PIPELINE_WEAK_EDGE") != "1", reason="opt-in experiment"),
    pytest.mark.usefixtures("pipeline_engine"),
]


def _floats(name: str, default: str) -> list[float]:
    return [float(x) for x in os.environ.get(name, default).split(",") if x.strip()]


CASES = [
    (drift, int(seed))
    for seed in _floats("POSEIDON_WEAK_EDGE_SEEDS", "11")
    for drift in _floats("POSEIDON_WEAK_EDGE_DRIFTS", "30,20,15,10,5,0")
]


@pytest.mark.parametrize(("drift", "seed"), CASES, ids=[f"d{d:g}-s{s}" for d, s in CASES])
def test_weak_edge_case(pipeline_sessions, tmp_path, drift, seed):
    source = PlantedOrbSource(drift_points=drift, seed=seed)
    registry = build_default_registry(data_source=source)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(CONTROL_CLOCK), registry=registry)
    plan = list(V1_STAGE_PLAN[: V1_STAGE_PLAN.index("holdout") + 1])
    run_id = harness.create(spec=planted_orb_spec(), stage_plan=plan, idea_text=f"weak edge d{drift:g} s{seed}")
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

    reached = [name for name in GATE_STAGES if name in attempts]
    failed_at = next((name for name in reached if attempts[name].verdict != "pass"), None)
    later_train = [name for name in GATE_STAGES[:-1] if name not in attempts]
    t0 = time.monotonic()
    diagnostics = (
        diagnose(
            pipeline_sessions,
            run.id,
            later_train,
            harness.dispatcher.registry,
            artifacts_root=tmp_path / "diag-artifacts",
        )
        if later_train
        else {}
    )
    diag_seconds = round(time.monotonic() - t0, 1)

    optimize = reports.get("optimize") or {}
    best = optimize.get("best") or {}
    metrics = best.get("metrics") or {}
    n_trades = metrics.get("n_trades") or 0
    payload = {
        "drift_points": drift,
        "seed": seed,
        "planted_days": source.planted_days,
        "run_status": run.status,
        "run_reason": run.reason,
        "failed_at": failed_at,
        "failed_summary": (reports.get(failed_at) or {}).get("summary") if failed_at else None,
        "wall_seconds": wall,
        "diagnostics_seconds": diag_seconds,
        "stage_seconds": timings,
        "optimize_best": {
            "params": best.get("params"),
            "total_pnl_twd": metrics.get("total_pnl_twd"),
            "sharpe_daily": metrics.get("sharpe_daily"),
            "win_rate": metrics.get("win_rate"),
            "n_trades": n_trades,
            "avg_trade_twd": (metrics.get("total_pnl_twd") / n_trades) if n_trades else None,
        },
        "verdicts_in_run": {name: attempts[name].verdict for name in reached},
        "margins_in_run": gate_margins({name: reports.get(name) for name in reached}),
        "verdicts_outside_run": {name: d["verdict"] for name, d in diagnostics.items()},
        "margins_outside_run": gate_margins({name: d["report"] for name, d in diagnostics.items()}),
    }
    text = json.dumps(payload, indent=1, default=str, ensure_ascii=False)
    print(f"\n=== d{drift:g} s{seed} ===\n{text}")
    out = os.environ.get("POSEIDON_WEAK_EDGE_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"d{drift:g}-s{seed}.json").write_text(text)
    assert run.status in ("completed", "failed"), payload
