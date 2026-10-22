"""Real-PostgreSQL proofs for pipeline constraints, partial indexes and triggers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline.models import PipelineBase, PipelineEvent, PipelineRun, PipelineTrial, StageRun

pytestmark = pytest.mark.postgresql
NOW = datetime(2026, 10, 22, 2, 0, tzinfo=UTC)


def _run(**overrides) -> PipelineRun:
    run_id = uuid.uuid4()
    config = {"stage_plan": ["spec_generate"]}
    values = {
        "id": run_id,
        "family_id": run_id,
        "source": "api",
        "idea_text": "idea",
        "config_json": config,
        "config_sha256": content_sha256(config),
        "created_by": "human:researcher",
        "status": "queued",
    }
    values.update(overrides)
    return PipelineRun(**values)


def _attempt(run_id, **overrides) -> StageRun:
    values = {
        "id": uuid.uuid4(),
        "run_id": run_id,
        "stage": "spec_generate",
        "seq": 0,
        "attempt": 1,
        "trigger": "advance",
        "status": "pending",
    }
    values.update(overrides)
    return StageRun(**values)


def _running(run_id, **overrides) -> StageRun:
    values = {
        "status": "running",
        "lease_owner": uuid.uuid4(),
        "lease_expires_at": NOW + timedelta(minutes=5),
        "input_sha256": "a" * 64,
        "code_version": "test",
    }
    values.update(overrides)
    return _attempt(run_id, **values)


def _completed(run_id, **overrides) -> StageRun:
    values = {"status": "completed", "verdict": "pass", "finished_at": NOW, "input_sha256": "a" * 64}
    values.update(overrides)
    return _attempt(run_id, **values)


def _seed(sessions, *rows):
    with sessions(expire_on_commit=False) as session:
        for row in rows:
            session.add(row)
            session.flush()
        session.commit()


def _rejects(sessions, statement, params, message):
    with sessions() as session:
        with pytest.raises(DBAPIError) as error:
            session.execute(text(statement), params)
            session.flush()
        session.rollback()
    assert message in str(error.value)


def _integrity(sessions, *rows):
    with sessions() as session:
        with pytest.raises(IntegrityError):
            for row in rows:
                session.add(row)
                session.flush()
        session.rollback()


def test_migrated_schema_matches_the_models(pipeline_engine):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    with pipeline_engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"version_table": "alembic_version_pipeline"})
        diffs = compare_metadata(context, PipelineBase.metadata)
    assert diffs == []
    with pipeline_engine.connect() as connection:
        for table in PipelineBase.metadata.sorted_tables:
            database = set(
                connection.execute(
                    text(
                        "SELECT conname FROM pg_constraint WHERE conrelid = CAST(:table AS regclass) AND contype = 'c'"
                    ),
                    {"table": table.name},
                ).scalars()
            )
            model = {
                constraint.name
                for constraint in table.constraints
                if constraint.__class__.__name__ == "CheckConstraint"
            }
            assert database == model, table.name


def test_run_check_constraints(pipeline_sessions):
    _integrity(pipeline_sessions, _run(market="tw_stock"))
    _integrity(pipeline_sessions, _run(config_sha256="a" * 63))
    _integrity(pipeline_sessions, _run(status="done"))
    _integrity(pipeline_sessions, _run(status="failed"))  # terminal without finished_at
    _integrity(pipeline_sessions, _run(idea_text="   "))
    _integrity(pipeline_sessions, _run(create_idempotency_key="k"))  # key without request hash


def test_stage_run_constraints_and_partial_unique_indexes(pipeline_sessions):
    run = _run()
    _seed(pipeline_sessions, run)
    _integrity(pipeline_sessions, _attempt(run.id, status="completed", finished_at=NOW))  # completed w/o verdict
    _integrity(pipeline_sessions, _attempt(run.id, verdict="pass"))  # verdict on a pending row
    _integrity(pipeline_sessions, _attempt(run.id, status="running"))  # running without a lease
    _integrity(pipeline_sessions, _attempt(run.id, status="waiting"))  # waiting without wait_until
    _integrity(pipeline_sessions, _completed(run.id, artifact_json={"report": {}}))  # artifact without its hash
    _integrity(pipeline_sessions, _completed(run.id, artifact_json={"r": "x" * 270_000}, artifact_sha256="d" * 64))
    _integrity(pipeline_sessions, _completed(run.id), _completed(run.id))  # UQ(run, stage, attempt)
    # At most one open attempt per (run, stage).
    _integrity(pipeline_sessions, _attempt(run.id), _attempt(run.id, attempt=2))
    # At most one running attempt per run, across stages.
    _integrity(pipeline_sessions, _running(run.id), _running(run.id, stage="spec_validate", seq=1))
    # A closed attempt does not block the next open one.
    _seed(pipeline_sessions, _completed(run.id, verdict="error"), _attempt(run.id, attempt=2))


def test_completed_and_aborted_stage_runs_are_immutable_and_undeletable(pipeline_sessions):
    run = _run()
    done = _completed(run.id)
    aborted = _attempt(run.id, stage="spec_validate", seq=1, status="aborted", finished_at=NOW)
    pending = _attempt(run.id, stage="backtest", seq=2)
    _seed(pipeline_sessions, run, done, aborted, pending)
    for row in (done, aborted):
        _rejects(
            pipeline_sessions,
            "UPDATE stage_runs SET reason = 'rewritten' WHERE id = :id",
            {"id": row.id},
            "immutable",
        )
    _rejects(pipeline_sessions, "DELETE FROM stage_runs WHERE id = :id", {"id": pending.id}, "cannot be deleted")
    _rejects(
        pipeline_sessions,
        "UPDATE stage_runs SET attempt = 9 WHERE id = :id",
        {"id": pending.id},
        "identity is immutable",
    )
    with pipeline_sessions() as session:  # open attempts still move
        session.execute(text("UPDATE stage_runs SET reason = 'ok' WHERE id = :id"), {"id": pending.id})
        session.commit()
    with pipeline_sessions() as session:  # ORM layer refuses first
        row = session.get(StageRun, done.id)
        row.reason = "rewritten"
        with pytest.raises(ValueError, match="immutable"):
            session.flush()


def test_runs_cannot_be_deleted_or_change_frozen_columns(pipeline_sessions):
    run = _run()
    _seed(pipeline_sessions, run)
    _rejects(pipeline_sessions, "DELETE FROM pipeline_runs WHERE id = :id", {"id": run.id}, "cannot be deleted")
    _rejects(
        pipeline_sessions,
        "UPDATE pipeline_runs SET config_json = '{}'::jsonb WHERE id = :id",
        {"id": run.id},
        "frozen",
    )
    _rejects(
        pipeline_sessions,
        "UPDATE pipeline_runs SET idea_text = 'other' WHERE id = :id",
        {"id": run.id},
        "frozen",
    )
    with pipeline_sessions() as session:
        session.execute(
            text("UPDATE pipeline_runs SET status = 'running', revision = 2 WHERE id = :id"), {"id": run.id}
        )
        session.commit()
    with pipeline_sessions() as session:
        loaded = session.get(PipelineRun, run.id)
        session.delete(loaded)
        with pytest.raises(ValueError, match="cannot be deleted"):
            session.flush()


def test_trials_and_events_are_append_only(pipeline_sessions):
    run = _run()
    attempt = _completed(run.id, verdict="error")
    trial = PipelineTrial(
        family_id=run.id,
        run_id=run.id,
        stage_run_id=attempt.id,
        trial_no=0,
        params_json={"x": 1},
        params_sha256=content_sha256({"x": 1}),
        state="fail",
        window_sha256="b" * 64,
    )
    event = PipelineEvent(run_id=run.id, event_type="run_created", actor_id="a", expected_revision=1, payload_json={})
    _seed(pipeline_sessions, run, attempt, trial, event)
    for table, row in (("pipeline_trials", trial), ("pipeline_events", event)):
        _rejects(
            pipeline_sessions, f"UPDATE {table} SET created_at = now() WHERE id = :id", {"id": row.id}, "append-only"
        )
        _rejects(pipeline_sessions, f"DELETE FROM {table} WHERE id = :id", {"id": row.id}, "append-only")
    with pipeline_sessions() as session:
        loaded = session.get(PipelineEvent, event.id)
        loaded.payload_json = {"changed": True}
        with pytest.raises(ValueError, match="append-only"):
            session.flush()
    _integrity(
        pipeline_sessions,
        PipelineEvent(
            run_id=run.id, event_type="x", actor_id="a", expected_revision=1, payload_json={}, idempotency_key="k"
        ),
    )
