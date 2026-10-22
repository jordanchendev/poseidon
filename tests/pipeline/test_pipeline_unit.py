"""Database-free pipeline contracts: state machine, plan, validator, stub, isolation."""

from __future__ import annotations

import subprocess
import sys
import types
import uuid
from datetime import UTC, date, datetime

import pytest

from poseidon.pipeline import artifacts
from poseidon.pipeline.plan import V1_STAGE_PLAN, StagePlanError, default_config, validate_stage_plan
from poseidon.pipeline.spec_validation import DslV1Validator, get_spec_validator
from poseidon.pipeline.stages import STAGE_REGISTRY, StageContext, UpstreamResult
from poseidon.pipeline.state import (
    IllegalTransition,
    RunStatus,
    StageStatus,
    Verdict,
    check_run_transition,
    check_stage_transition,
    transition_run,
    transition_stage,
)
from tests.pipeline.conftest import REPO_ROOT, STEP1_PLAN, runner_env, tx_spec

NOW = datetime(2026, 10, 22, tzinfo=UTC)


# --------------------------------------------------------------- state machine


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("queued", "running"),
        ("running", "queued"),
        ("running", "waiting"),
        ("running", "failed"),
        ("running", "aborted"),
        ("waiting", "running"),
        ("error", "queued"),
        ("unsupported", "queued"),
        ("aborted", "queued"),
    ],
)
def test_legal_run_transitions(old, new):
    assert check_run_transition(old, new) == RunStatus(new)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("failed", "queued"),  # owner decision: fail is final
        ("completed", "queued"),
        ("completed", "running"),
        ("queued", "failed"),
        ("waiting", "completed"),
        ("aborted", "running"),
    ],
)
def test_illegal_run_transitions_raise(old, new):
    with pytest.raises(IllegalTransition):
        check_run_transition(old, new)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("completed", "running"),
        ("completed", "aborted"),
        ("aborted", "pending"),
        ("pending", "completed"),
        ("pending", "waiting"),
        ("waiting", "completed"),
    ],
)
def test_illegal_stage_transitions_raise(old, new):
    with pytest.raises(IllegalTransition):
        check_stage_transition(old, new)


def test_transition_helpers_keep_revision_finished_and_verdict_coherent():
    run = types.SimpleNamespace(status="running", revision=3, updated_at=None, finished_at=None, reason=None)
    transition_run(run, RunStatus.FAILED, now=NOW, reason="spec_validate: bad")
    assert (run.status, run.revision, run.finished_at, run.reason) == ("failed", 4, NOW, "spec_validate: bad")
    with pytest.raises(IllegalTransition):
        transition_run(run, RunStatus.QUEUED, now=NOW)

    attempt = types.SimpleNamespace(status="running", verdict=None, updated_at=None, finished_at=None, reason=None)
    with pytest.raises(IllegalTransition):
        transition_stage(attempt, StageStatus.COMPLETED, now=NOW)  # completed needs a verdict
    with pytest.raises(IllegalTransition):
        transition_stage(attempt, StageStatus.WAITING, now=NOW, verdict=Verdict.PASS)
    transition_stage(attempt, StageStatus.COMPLETED, now=NOW, verdict=Verdict.PASS)
    assert (attempt.status, attempt.verdict, attempt.finished_at) == ("completed", "pass", NOW)


# ------------------------------------------------------------------ plan/config


def test_default_config_carries_the_full_v1_plan_and_frozen_window():
    config = default_config(stage_plan=None, as_of=date(2026, 10, 22))
    assert tuple(config["stage_plan"]) == V1_STAGE_PLAN
    assert config["data"]["end"] == "2026-10-22"
    assert config["thresholds"]["wfe_min"] == 0.5
    assert config["max_trials"] == 200
    assert tuple(STAGE_REGISTRY) == V1_STAGE_PLAN


@pytest.mark.parametrize(
    "plan",
    [
        [],
        ["spec_generate", "nope"],
        ["spec_generate", "spec_generate"],
        ["spec_validate", "spec_generate"],
        ["spec_generate", "backtest"],  # backtest requires spec_validate
    ],
)
def test_invalid_stage_plans_are_rejected(plan):
    with pytest.raises(StagePlanError):
        validate_stage_plan(plan, STAGE_REGISTRY)


def test_step1_and_full_plans_validate():
    assert validate_stage_plan(STEP1_PLAN, STAGE_REGISTRY) == STEP1_PLAN
    assert validate_stage_plan(list(V1_STAGE_PLAN), STAGE_REGISTRY) == list(V1_STAGE_PLAN)


# -------------------------------------------------------------------- validator


def test_v1_validator_accepts_the_hand_written_tx_spec():
    ok, errors, unsupported = get_spec_validator("dsl_v1").validate_spec(tx_spec())
    assert (ok, errors, unsupported) == (True, (), ())


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        (tx_spec(market="tw_stock"), "must target tw_futures/TX"),
        (tx_spec(rules=[]), "rules"),
        (
            tx_spec(rules=[{"condition": {"type": "indicator_above", "indicator": "rsi"}, "action": "long"}]),
            "threshold",
        ),
        (tx_spec(rules=[{"condition": {"all": []}, "action": "long"}]), "non-empty"),
        (
            tx_spec(rules=[{"condition": {"type": "feature_above", "column": "x", "threshold": 1}, "action": "buy"}]),
            "action",
        ),
        ("not-an-object", "JSON object"),
    ],
)
def test_v1_validator_reports_errors(spec, fragment):
    ok, errors, unsupported = DslV1Validator().validate_spec(spec)
    assert not ok and unsupported == ()
    assert any(fragment in error for error in errors), errors


@pytest.mark.parametrize("condition_type", ["opening_range_breakout", "ml_prediction"])
def test_v1_validator_reports_missing_capabilities_as_unsupported(condition_type):
    spec = tx_spec(rules=[{"condition": {"type": condition_type, "threshold": 0.5}, "action": "long"}])
    ok, errors, unsupported = DslV1Validator().validate_spec(spec)
    assert not ok and errors == ()
    assert condition_type in unsupported[0]


# ----------------------------------------------------------------- stub + files


def _backtest_context(spec_sha):
    config = default_config(stage_plan=STEP1_PLAN, as_of=date(2026, 10, 22))
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="x",
        input_spec_json=None,
        config=config,
        config_sha256="0" * 64,
        stage="backtest",
        attempt=1,
        upstream={"spec_validate": UpstreamResult("spec_validate", 1, "1" * 64, "2" * 64, {"spec_sha256": spec_sha})},
    )


def test_backtest_stub_is_keyed_by_input_hash():
    from poseidon.decision_loop.manifest import content_sha256

    stage = STAGE_REGISTRY["backtest"]
    reports = []
    for spec_sha in ("a" * 64, "a" * 64, "b" * 64):
        ctx = _backtest_context(spec_sha)
        ctx.input_sha256 = content_sha256(stage.input_payload(ctx))
        result = stage.run(ctx)
        assert result.verdict == Verdict.PASS
        reports.append((ctx.input_sha256, result.report, dict(result.files)))
    assert reports[0] == reports[1]
    assert reports[0][0] != reports[2][0] and reports[0][1] != reports[2][1]


def test_artifact_limits(tmp_path):
    with pytest.raises(artifacts.ArtifactError):
        artifacts.write_files(tmp_path, {"../escape.json": b"{}"})
    with pytest.raises(artifacts.ArtifactError):
        artifacts.build_artifact({"blob": "x" * (300 * 1024)}, [])
    manifest = artifacts.write_files(tmp_path, {"trades.json": b"[]"})
    artifact, digest = artifacts.build_artifact({"k": 1}, manifest)
    assert artifact["files"][0]["sha256"] == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    assert len(digest) == 64 and (tmp_path / "trades.json").read_bytes() == b"[]"
    assert artifacts.build_artifact(None, []) == (None, None)


# -------------------------------------------------------------------- isolation


def test_pipeline_celery_app_schedules_only_the_sweeper():
    from poseidon.pipeline.celery_app import PIPELINE_QUEUE, pipeline_celery_app

    conf = pipeline_celery_app.conf
    assert PIPELINE_QUEUE == "poseidon_pipeline"
    assert set(conf.beat_schedule) == {"sweep-pipelines"}
    assert conf.beat_schedule["sweep-pipelines"]["task"] == "poseidon.pipeline.tasks.sweep_pipelines"
    assert conf.beat_schedule["sweep-pipelines"]["schedule"] == 300.0
    assert conf.redbeat_key_prefix == "poseidon-pipeline:redbeat:"
    assert conf.task_default_queue == PIPELINE_QUEUE
    assert conf.task_acks_late is True and conf.worker_prefetch_multiplier == 1
    assert list(conf.imports) == ["poseidon.pipeline.tasks"]


def test_runner_entry_points_never_import_the_production_celery_app():
    code = (
        "import sys, poseidon.pipeline.tasks, poseidon.pipeline.app; "
        "print('poseidon.workers.celery_app' in sys.modules, 'poseidon.main' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=runner_env()
    ).stdout
    assert output.split() == ["False", "False"]


def test_production_api_does_not_mount_pipelines_by_default():
    from poseidon.core.config import settings
    from poseidon.main import app

    assert settings.pipeline_api_enabled is False
    assert not any(getattr(route, "path", "").startswith("/api/v1/pipelines") for route in app.routes)


def test_guard_refuses_a_non_pipeline_database():
    from poseidon.pipeline.guard import PipelineIsolationError, assert_isolated_database

    assert_isolated_database("postgresql://u:p@h:5433/poseidon_pipeline", expected="poseidon_pipeline")
    with pytest.raises(PipelineIsolationError, match="expects database 'poseidon_pipeline', configured 'poseidon'"):
        assert_isolated_database("postgresql://u:p@h:5433/poseidon", expected="poseidon_pipeline")


@pytest.mark.parametrize("expected", ["", "   "])
def test_guard_refuses_an_unset_or_empty_expectation(expected):
    from poseidon.pipeline.guard import PipelineIsolationError, assert_isolated_database

    with pytest.raises(PipelineIsolationError, match="POSEIDON_PIPELINE_EXPECTED_DATABASE is unset or empty"):
        assert_isolated_database("postgresql://u:p@h:5433/poseidon_pipeline", expected=expected)


def test_guard_reads_the_expectation_from_settings(monkeypatch):
    from poseidon.core.config import settings
    from poseidon.pipeline.guard import PipelineIsolationError, assert_isolated_database

    monkeypatch.setattr(settings, "pipeline_expected_database", "")
    with pytest.raises(PipelineIsolationError, match="unset or empty"):
        assert_isolated_database("postgresql://u:p@h:5433/poseidon_pipeline")
    monkeypatch.setattr(settings, "pipeline_expected_database", "poseidon_pipeline")
    assert_isolated_database("postgresql://u:p@h:5433/poseidon_pipeline")


def _run(args, **env):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=runner_env(**env))


@pytest.mark.parametrize("expected", [None, ""])  # None: variable absent from the environment
def test_guard_cli_fails_closed_when_the_expectation_is_unset(expected):
    result = _run(["-m", "poseidon.pipeline.guard"], POSEIDON_PIPELINE_EXPECTED_DATABASE=expected)
    assert result.returncode == 1
    assert "refusing to start: POSEIDON_PIPELINE_EXPECTED_DATABASE is unset or empty" in result.stderr


def test_guard_cli_passes_only_for_the_named_database():
    assert _run(["-m", "poseidon.pipeline.guard"]).returncode == 0
    production = _run(["-m", "poseidon.pipeline.guard"], POSEIDON_DATABASE_URL="postgresql://u:p@h:5433/poseidon")
    assert production.returncode == 1 and "configured 'poseidon'" in production.stderr


@pytest.mark.parametrize("module", ["poseidon.pipeline.app", "poseidon.pipeline.celery_app", "poseidon.pipeline.tasks"])
@pytest.mark.parametrize("expected", [None, ""])
def test_runner_entry_points_refuse_to_import_without_an_expectation(module, expected):
    result = _run(["-c", f"import {module}"], POSEIDON_PIPELINE_EXPECTED_DATABASE=expected)
    assert result.returncode != 0
    assert "PipelineIsolationError" in result.stderr and "unset or empty" in result.stderr


def test_alembic_streams_are_separate():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    ini = str(REPO_ROOT / "alembic.ini")
    main = ScriptDirectory.from_config(Config(ini))
    pipeline_config = Config(ini, ini_section="pipeline")
    pipeline = ScriptDirectory.from_config(pipeline_config)
    assert main.get_heads() == ["046"]
    assert pipeline.get_heads() == ["p001"]
    assert all(not revision.revision.startswith("p") for revision in main.walk_revisions())
    assert pipeline_config.get_main_option("version_table") == "alembic_version_pipeline"
    assert Config(ini).get_main_option("version_table") is None
