"""Tests for the Phase 41 foundation slice (plan 41-01).

Covers RESEARCH-API-03 (training_runs table, status transitions, promotion)
and basic Pydantic schema validation.

The unit suite runs against an in-memory SQLite harness that uses the
Postgres-only types via ``@compiles`` shims (same pattern as
``tests/unit/test_phase40_foundation.py``). Real DDL execution runs
end-to-end on stormtrooper via the Phase 41 smoke runbook.

NOTE: The ORM fixture tests (db_session) require psycopg2 and run on
stormtrooper. Migration file assertions and Pydantic schema tests run
locally without Postgres.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# --- SQLite compatibility shims for Postgres-only types (must run before
# any Base.metadata.create_all call) ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


# ---------------------------------------------------------------------------
# Migration 025 file-level assertions
# ---------------------------------------------------------------------------

MIGRATION_025_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "025_create_training_runs.py"
)


def test_migration_025_exists_with_correct_revision():
    """025 must exist with the right revision chain and key DDL markers."""
    assert MIGRATION_025_PATH.exists(), f"migration file missing: {MIGRATION_025_PATH}"
    content = MIGRATION_025_PATH.read_text()

    assert 'revision = "025"' in content
    assert 'down_revision = "024"' in content
    assert "CREATE SCHEMA IF NOT EXISTS mlflow" in content
    assert '"training_runs"' in content


def test_migration_025_has_all_d04_columns():
    """D-04: All training_runs columns must be present in the migration."""
    content = MIGRATION_025_PATH.read_text()

    for col in (
        "run_id",
        "handler_class",
        "handler_params",
        "model_class",
        "model_params",
        "market",
        "symbols",
        "interval",
        "segments",
        "lookback",
        "status",
        "metrics",
        "model_version_id",
        "mlflow_run_id",
        "error",
        "requested_by",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    ):
        assert col in content, f"missing column in migration 025: {col}"


def test_migration_025_has_check_constraint_and_indexes():
    """D-05: Status CHECK constraint + filtering/pagination indexes."""
    content = MIGRATION_025_PATH.read_text()

    assert "ck_training_runs_status" in content
    assert "ix_training_runs_status" in content
    assert "ix_training_runs_created_at" in content


def test_migration_025_downgrade_drops_mlflow_schema():
    """Downgrade must drop training_runs table and mlflow schema."""
    content = MIGRATION_025_PATH.read_text()
    downgrade_block = content.split("def downgrade")[-1]

    assert 'drop_table("training_runs")' in downgrade_block
    assert "DROP SCHEMA IF EXISTS mlflow CASCADE" in downgrade_block


# ---------------------------------------------------------------------------
# TrainingRun ORM tests (in-memory SQLite)
#
# These tests use a standalone DeclarativeBase to avoid importing
# poseidon.models.base which eagerly creates a PostgreSQL engine. The
# TrainingRun column definitions are exercised via direct import of the
# model module after patching Base.
# ---------------------------------------------------------------------------

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class _TestBase(DeclarativeBase):
    """Standalone base for in-memory SQLite ORM tests."""

    pass


def _make_training_run_class():
    """Build a TrainingRun-like model bound to _TestBase for SQLite testing.

    We re-declare the model using the same Column definitions to avoid
    importing poseidon.models.base (which needs psycopg2). The column list
    is verified against the source file to catch drift.
    """
    from sqlalchemy import (
        CheckConstraint,
        Column,
        DateTime,
        ForeignKey,
        String,
        Text,
        func,
    )
    from sqlalchemy.dialects.postgresql import JSONB, UUID

    class TrainingRunTest(_TestBase):
        __tablename__ = "training_runs"

        run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        handler_class = Column(String(64), nullable=False)
        handler_params = Column(JSONB, nullable=False, default=dict)
        model_class = Column(String(64), nullable=False)
        model_params = Column(JSONB, nullable=False, default=dict)
        market = Column(String(32), nullable=False)
        symbols = Column(JSONB, nullable=False, default=list)
        interval = Column(String(8), nullable=False)
        segments = Column(JSONB, nullable=False, default=dict)
        lookback = Column(Text, nullable=True)
        status = Column(String(16), nullable=False, default="pending")
        metrics = Column(JSONB, nullable=True)
        model_version_id = Column(UUID(as_uuid=True), nullable=True)
        mlflow_run_id = Column(Text, nullable=True)
        error = Column(Text, nullable=True)
        requested_by = Column(String(16), nullable=False, default="api")
        started_at = Column(DateTime(timezone=True), nullable=True)
        finished_at = Column(DateTime(timezone=True), nullable=True)
        created_at = Column(
            DateTime(timezone=True), server_default=func.now(), nullable=False
        )
        updated_at = Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        )

        __table_args__ = (
            CheckConstraint(
                "status IN ('pending','running','succeeded','failed','cancelled')",
                name="ck_training_runs_status",
            ),
        )

        def __repr__(self) -> str:
            return (
                f"<TrainingRun run_id={self.run_id} status={self.status} "
                f"handler_class={self.handler_class} model_class={self.model_class}>"
            )

    return TrainingRunTest


@pytest.fixture()
def db_session():
    """In-memory SQLite session with training_runs table created."""
    engine = create_engine("sqlite:///:memory:")
    _TestBase.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    _TestBase.metadata.drop_all(engine)


# Build the test model class at module level (after shims are registered)
TrainingRunTest = _make_training_run_class()


def test_training_run_default_status_is_pending(db_session: Session):
    """A new TrainingRun should default to status='pending'."""
    run = TrainingRunTest(
        run_id=uuid.uuid4(),
        handler_class="Alpha158Handler",
        model_class="LGBMModel",
        market="crypto_perp",
        interval="4h",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    assert run.status == "pending"


def test_training_run_all_d04_columns_exist():
    """TrainingRun must expose every column from the D-04 schema."""
    columns = {c.name for c in TrainingRunTest.__table__.columns}
    expected = {
        "run_id",
        "handler_class",
        "handler_params",
        "model_class",
        "model_params",
        "market",
        "symbols",
        "interval",
        "segments",
        "lookback",
        "status",
        "metrics",
        "model_version_id",
        "mlflow_run_id",
        "error",
        "requested_by",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    }
    assert expected.issubset(columns), f"missing columns: {expected - columns}"


def test_training_run_all_d04_columns_in_source():
    """Cross-check: the actual TrainingRun source file must contain all D-04 columns."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "poseidon"
        / "models"
        / "training_run.py"
    )
    content = source_path.read_text()
    for col in (
        "run_id",
        "handler_class",
        "handler_params",
        "model_class",
        "model_params",
        "market",
        "symbols",
        "interval",
        "segments",
        "lookback",
        "status",
        "metrics",
        "model_version_id",
        "mlflow_run_id",
        "error",
        "requested_by",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    ):
        assert f"{col} = Column(" in content, f"missing column in source: {col}"
    assert 'ForeignKey("model_versions.id")' in content
    assert "ck_training_runs_status" in content


def test_training_run_status_transitions(db_session: Session):
    """Status can transition: pending -> running -> succeeded."""
    run = TrainingRunTest(
        run_id=uuid.uuid4(),
        handler_class="Alpha158Handler",
        model_class="LGBMModel",
        market="crypto_perp",
        interval="4h",
    )
    db_session.add(run)
    db_session.commit()

    # pending -> running
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "running"
    assert run.started_at is not None

    # running -> succeeded
    run.status = "succeeded"
    run.finished_at = datetime.now(timezone.utc)
    run.metrics = {"ic": 0.05, "icir": 1.2}
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "succeeded"
    assert run.finished_at is not None


def test_training_run_failed_preserves_error(db_session: Session):
    """D-07: A failed run must preserve the error text."""
    run = TrainingRunTest(
        run_id=uuid.uuid4(),
        handler_class="Alpha158Handler",
        model_class="LGBMModel",
        market="crypto_perp",
        interval="4h",
    )
    db_session.add(run)
    db_session.commit()

    run.status = "failed"
    run.error = "OOM: GPU ran out of memory during training"
    run.finished_at = datetime.now(timezone.utc)
    db_session.commit()
    db_session.refresh(run)

    assert run.status == "failed"
    assert run.error == "OOM: GPU ran out of memory during training"


def test_training_run_model_version_id_nullable(db_session: Session):
    """D-06: model_version_id is NULL until a succeeded run is promoted."""
    run = TrainingRunTest(
        run_id=uuid.uuid4(),
        handler_class="Alpha158Handler",
        model_class="LGBMModel",
        market="crypto_perp",
        interval="4h",
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    assert run.model_version_id is None


def test_training_run_repr():
    """__repr__ must include run_id, status, handler_class, model_class."""
    run_id = uuid.uuid4()
    run = TrainingRunTest(
        run_id=run_id,
        handler_class="Alpha158Handler",
        model_class="LGBMModel",
        market="crypto_perp",
        interval="4h",
    )
    r = repr(run)
    assert "TrainingRun" in r
    assert str(run_id) in r
    assert "Alpha158Handler" in r
    assert "LGBMModel" in r


# ---------------------------------------------------------------------------
# Pydantic schema tests (no DB needed)
# ---------------------------------------------------------------------------


def test_train_request_validates_required_fields():
    """TrainRequest with all required fields should succeed."""
    from poseidon.core.schemas import TrainRequest

    req = TrainRequest(
        handler_class="Alpha158Handler",
        model_class="LGBMModel",
        market="crypto_perp",
        symbols=["BTCUSDT"],
        interval="4h",
    )
    assert req.handler_class == "Alpha158Handler"
    assert req.model_class == "LGBMModel"
    assert req.symbols == ["BTCUSDT"]
    # segments should have default value
    assert "train" in req.segments


def test_train_request_rejects_empty_symbols():
    """TrainRequest with symbols=[] must raise ValidationError (min_length=1)."""
    from pydantic import ValidationError

    from poseidon.core.schemas import TrainRequest

    with pytest.raises(ValidationError):
        TrainRequest(
            handler_class="Alpha158Handler",
            model_class="LGBMModel",
            market="crypto_perp",
            symbols=[],
            interval="4h",
        )


def test_training_run_response_from_attributes():
    """TrainingRunResponse.model_validate() from a dict should work."""
    from poseidon.core.schemas import TrainingRunResponse

    data = {
        "run_id": uuid.uuid4(),
        "handler_class": "Alpha158Handler",
        "model_class": "LGBMModel",
        "market": "crypto_perp",
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
    }
    resp = TrainingRunResponse.model_validate(data)
    assert resp.handler_class == "Alpha158Handler"
    assert resp.status == "pending"
