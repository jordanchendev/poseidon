"""Daily controls through the real v3 stages (design 2026-10-24 §7, §10 step 4).

Each control is a spec v3 document created through ``PipelineService`` and driven by the dispatcher on
PostgreSQL through every stage of ``plan.V3_DAILY_STAGE_PLAN`` with the real stage bodies: refdata ->
``v3.compiler`` -> engine -> the registered ``daily_equity`` thresholds. Only the refdata is synthetic
(``v3_wiring_support.build_long_store``, 1999-2026):

* planted edge -- the market drifts up only while it is "on" by the 200-day MA band rule; 3x TMF
  switched off below the band must pass every gate. The run is created on 2026-09-01 and the synthetic
  refdata runs to 2026-10-21, so its daily shadow paper (``registry.compile_strategy``) books the
  ~35 trading days after the data window at its first wake-up, matches its replay and passes; the run
  parks in ``awaiting_release``;
* static leverage -- always long at 2-4x on a structure-free drifting market: must fail
  ``benchmark`` (alpha over the 1x underlying is only beta minus costs), after passing the train
  gates before it;
* forced liquidation -- 4x long into a train day whose low is 24.3 % under the previous close:
  must fail ``backtest`` on the liquidation gate.

Every gate is the registered one, ``execution_timing`` included (signal_close and a one-day signal lag
against the spec's next_open). Seeds were fixed before looking at gate results.
Set ``POSEIDON_V3_CONTROLS_OUT`` to keep each control's stage trace as JSON.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.v3_wiring_support import (
    CLOCK_START,
    build_long_store,
    create_v3,
    liquidation_spec,
    planted_edge_spec,
    static_leverage_spec,
    trace,
    v3_registry,
)

pytestmark = pytest.mark.postgresql
OUT = os.environ.get("POSEIDON_V3_CONTROLS_OUT")
PLANTED_SEED, DRIFT_SEED = 5, 3
CRASH_DAY = "2010-06-01"
PAPER_CLOCK_START = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)  # 10:00 Taipei: the paper starts that day


def drive(sessions, root: Path, store, spec, name: str, start=CLOCK_START):
    harness = Harness(sessions, root / "artifacts", Clock(start), registry=v3_registry(store))
    run_id = create_v3(harness, spec, store=store, idea=f"control: {name}")
    harness.drain(limit=40)
    run = harness.run(run_id)
    rows = trace(harness, run_id)
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(
            json.dumps(
                {"run": {"status": run.status, "stage": run.current_stage, "reason": run.reason}, "trace": rows},
                indent=2,
            )
        )
    return harness, run, {row["stage"]: row for row in rows}, rows


def report(harness, run, stage):
    row = next(r for r in harness.attempts(run.id) if r.stage == stage)
    return (row.artifact_json or {}).get("report") or {}


@pytest.mark.usefixtures("pipeline_engine")
def test_planted_edge_passes_every_gate_into_shadow_paper(pipeline_sessions, tmp_path):
    store = build_long_store(tmp_path / "refdata", kind="ma_band", seed=PLANTED_SEED)
    harness, run, stages, rows = drive(
        pipeline_sessions, tmp_path, store, planted_edge_spec(), "planted_edge", start=PAPER_CLOCK_START
    )
    assert (run.status, run.current_stage) == ("waiting", "awaiting_release"), rows
    plan = run.config_json["stage_plan"]
    assert run.config_json["data"]["end"] == "2026-09-01"
    assert [row["stage"] for row in rows] == plan
    assert all(stages[name]["verdict"] == "pass" for name in plan[:-1]), rows
    assert stages["awaiting_release"]["status"] == "waiting"
    benchmark = report(harness, run, "benchmark")["gate_report"]
    assert (
        benchmark["alpha_per_day"] > 0
        and benchmark["alpha_t_hac"] >= run.config_json["thresholds"]["benchmark_alpha_t_min"]
    )
    timing = report(harness, run, "execution_timing")["gate_report"]
    assert set(timing["scenarios"]) == {"signal_close", "signal_lag_1d"} and timing["criterion_enabled"]
    holdout = report(harness, run, "holdout")
    assert holdout["inputs"]["holdout_metrics"]["n_days"] >= 250
    assert holdout["inputs"]["holdout_metrics"]["first_day"] >= run.config_json["data"]["holdout_start"]
    wfe = report(harness, run, "wfe")
    assert wfe["inputs"]["layout"] == {"windows": 5, "is_days": 1460, "oos_days": 365}
    paper = report(harness, run, "shadow_paper")
    assert paper["mode"] == "daily_target" and paper["engine"]["instruments"]["core"]["product"] == "TX"
    assert paper["status"] == "pass" and paper["paper_start_day"] == "2026-09-01" and paper["cutoff"] == "2026-10-21"
    assert paper["trading_days"] >= run.config_json["thresholds"]["paper_min_trading_days"]
    assert paper["consistency"]["ratio"] == 1.0 and paper["protection"]["tripped"] is None
    assert paper["engine"]["train_start"] == "2001-01-02"


@pytest.mark.usefixtures("pipeline_engine")
def test_static_leverage_fails_benchmark(pipeline_sessions, tmp_path):
    store = build_long_store(tmp_path / "refdata", kind="drift", seed=DRIFT_SEED)
    harness, run, stages, rows = drive(pipeline_sessions, tmp_path, store, static_leverage_spec(), "static_leverage")
    assert (run.status, run.current_stage) == ("failed", "benchmark"), rows
    plan = run.config_json["stage_plan"]
    assert all(stages[name]["verdict"] == "pass" for name in plan[: plan.index("benchmark")]), rows
    gate = report(harness, run, "benchmark")
    assert gate["reason_code"] in ("alpha_not_positive", "alpha_not_significant"), gate
    assert gate["gate_report"]["beta"] > 1.5  # leveraged beta, nothing else


@pytest.mark.usefixtures("pipeline_engine")
def test_forced_liquidation_fails_backtest(pipeline_sessions, tmp_path):
    store = build_long_store(tmp_path / "refdata", kind="drift", seed=DRIFT_SEED, crash=CRASH_DAY)
    harness, run, _stages, rows = drive(pipeline_sessions, tmp_path, store, liquidation_spec(), "forced_liquidation")
    assert (run.status, run.current_stage) == ("failed", "backtest"), rows
    liquidation = report(harness, run, "backtest")["liquidation"]
    assert liquidation["verdict"] == "fail" and liquidation["reason_code"] in ("forced_liquidation", "ruin"), (
        liquidation
    )
    days = [event["date"] for event in liquidation["forced_liquidations"]] + [liquidation["ruined"]]
    assert CRASH_DAY in days
