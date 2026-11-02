"""Runs frozen at CONFIG_VERSION 4 (before the benchmark gate) keep their plan, settings and input hashes.

``fixtures/run6_frozen_v4.json`` is the live runner's run #6 as stored in
``poseidon_pipeline`` on 2026-10-23 (mid shadow_paper): its frozen
``config_json``, ``config_sha256`` and every stage attempt's ``input_sha256`` /
``artifact_sha256``. The upgraded code must recompute the very same input
hashes, or the dispatcher fails the run with "stage ... input changed".
"""

from __future__ import annotations

import copy
import json
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from poseidon.core.canonical import content_sha256
from poseidon.pipeline.dispatcher import select_next_stage, upstream_names
from poseidon.pipeline.gates import benchmark, holdout
from poseidon.pipeline.gates.base import ResultSummary
from poseidon.pipeline.plan import (
    ALL_GATES_PASSED,
    CONFIG_VERSION,
    CORE_GATE_STAGES,
    StagePlanError,
    completion_reason,
    default_config,
    validate_stage_plan,
)
from poseidon.pipeline.shadow_paper import PAPER_VERSION
from poseidon.pipeline.stages import STAGE_REGISTRY, StageContext, UpstreamResult
from poseidon.pipeline.state import Verdict

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "run6_frozen_v4.json").read_text())
CONFIG = FIXTURE["config_json"]
ROWS = {row["stage"]: row for row in FIXTURE["stages"]}
GATE_AND_PAPER = ["param_stability", "dsr", "cost_stress", "regime_split", "wfe", "holdout", "shadow_paper"]


def _ctx(name: str, config: dict, config_sha256: str, rows: dict) -> StageContext:
    stage = STAGE_REGISTRY[name]
    upstream = {
        req: UpstreamResult(req, 1, rows[req]["input_sha256"], rows[req]["artifact_sha256"], None)
        for req in upstream_names(stage, config)
    }
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="",
        input_spec_json=None,
        config=config,
        config_sha256=config_sha256,
        stage=name,
        attempt=1,
        upstream=upstream,
    )


def test_the_fixture_is_run_6_as_frozen():
    assert FIXTURE["run_no"] == 6 and content_sha256(CONFIG) == FIXTURE["config_sha256"]
    assert CONFIG["version"] == 4 < CONFIG_VERSION
    assert "benchmark" not in CONFIG["stage_plan"] and "benchmark" not in CONFIG["gates"]
    assert not {"benchmark_alpha_t_min", "holdout_alpha_positive"} & set(CONFIG["thresholds"])
    assert ROWS["shadow_paper"]["verdict"] is None  # waiting
    assert all(ROWS[name]["verdict"] == "pass" for name in GATE_AND_PAPER[:-1])


@pytest.mark.parametrize("name", GATE_AND_PAPER)
def test_upgraded_code_recomputes_run_6_input_hashes(name):
    stage = STAGE_REGISTRY[name]
    ctx = _ctx(name, CONFIG, FIXTURE["config_sha256"], ROWS)
    assert "benchmark" not in ctx.upstream
    if name in ("holdout", "shadow_paper"):
        assert "benchmark" in stage.requires  # new runs need it; run #6 does not
    assert content_sha256(stage.input_payload(ctx)) == ROWS[name]["input_sha256"]


def test_shadow_paper_input_is_config_upstream_and_paper_version_only():
    payload = STAGE_REGISTRY["shadow_paper"].input_payload(_ctx("shadow_paper", CONFIG, FIXTURE["config_sha256"], ROWS))
    assert payload == {
        "stage": "shadow_paper",
        "config_sha256": FIXTURE["config_sha256"],
        "upstream": {name: ROWS[name]["artifact_sha256"] for name in ("backtest", "optimize", *CORE_GATE_STAGES)},
        "paper_version": PAPER_VERSION,
    }


def test_new_runs_require_benchmark_upstream():
    config = default_config(stage_plan=None, as_of=date(2026, 10, 24))
    assert config["version"] == CONFIG_VERSION == 5 and "benchmark" in config["stage_plan"]
    assert "benchmark" in upstream_names(STAGE_REGISTRY["holdout"], config)
    assert "benchmark" in upstream_names(STAGE_REGISTRY["shadow_paper"], config)
    assert "benchmark" not in upstream_names(STAGE_REGISTRY["holdout"], CONFIG)


def test_run_6_plan_stays_valid_at_its_version_and_never_schedules_benchmark():
    plan = CONFIG["stage_plan"]
    assert validate_stage_plan(plan, STAGE_REGISTRY, version=4) == plan
    with pytest.raises(StagePlanError, match="requires benchmark"):
        validate_stage_plan(plan, STAGE_REGISTRY)  # the same plan for a NEW run is refused
    with pytest.raises(StagePlanError, match="does not exist at config version 4"):
        validate_stage_plan([*plan[:8], "benchmark", *plan[8:]], STAGE_REGISTRY, version=4)
    assert completion_reason(plan) == ALL_GATES_PASSED
    rows = [
        SimpleNamespace(stage=name, attempt=1, status="completed", verdict=Verdict.PASS.value)
        for name in plan[: plan.index("shadow_paper")]
    ]
    assert select_next_stage(plan, rows)[0] == "shadow_paper"


def test_missing_new_threshold_keys_mean_criteria_off():
    thresholds = CONFIG["thresholds"]
    negative = {"alpha_twd_per_day": -1_000.0, "alpha_t_hac": -2.0, "beta": 0.9}
    result = holdout.evaluate(
        holdout.HoldoutInputs(
            ResultSummary(3_807_603.46, 0.0956789, 1247),
            ResultSummary(4_129_556.27, 0.1281378, 316),
            341,
            negative,
        ),
        thresholds,
    )
    assert result.passed and result.report["alpha_positive_required"] is False
    assert result.report["criteria"]["se"]["enabled"] is False
    off = benchmark.evaluate(benchmark.BenchmarkInputs([-5.0, -1.0, -3.0, -2.0], [1.0, 3.0, 2.0, 5.0], 5), thresholds)
    assert off.passed and off.report["criterion_enabled"] is False


@pytest.mark.usefixtures("pipeline_engine")
def test_a_v4_run_goes_from_holdout_to_shadow_paper_without_benchmark(pipeline_sessions, tmp_path, clock, monkeypatch):
    """Dispatcher + PostgreSQL: a run frozen at version 4 under the upgraded code."""

    from poseidon.pipeline import service
    from poseidon.pipeline.plan import GATE_STAGES
    from tests.pipeline.conftest import Harness
    from tests.pipeline.gate_support import passing, stub_train_gates

    def frozen_v4(*, stage_plan, as_of):
        config = default_config(stage_plan=None, as_of=as_of)
        config["version"] = 4
        config["stage_plan"] = [name for name in config["stage_plan"] if name != "benchmark"]
        config["gates"].pop("benchmark")
        for key in ("benchmark_alpha_t_min", "holdout_alpha_positive"):
            config["thresholds"].pop(key)
        return copy.deepcopy(config)

    monkeypatch.setattr(service, "default_config", frozen_v4)
    registry = stub_train_gates(holdout=passing("holdout", ("optimize", *GATE_STAGES[:-1])))
    harness = Harness(pipeline_sessions, tmp_path, clock, registry=registry)
    run_id = harness.create(stage_plan=None)
    harness.drain()
    run = harness.run(run_id)
    assert run.config_json["version"] == 4 and "benchmark" not in run.config_json["stage_plan"]
    assert (run.status, run.current_stage) == ("waiting", "shadow_paper")
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    assert "benchmark" not in attempts
    expected = {
        "stage": "shadow_paper",
        "config_sha256": run.config_sha256,
        "upstream": {name: attempts[name].artifact_sha256 for name in ("backtest", "optimize", *CORE_GATE_STAGES)},
        "paper_version": PAPER_VERSION,
    }
    assert attempts["shadow_paper"].input_sha256 == content_sha256(expected)
