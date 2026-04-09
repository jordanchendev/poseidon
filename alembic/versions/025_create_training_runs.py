"""Create training_runs table + mlflow schema (Phase 41 D-04, D-15).

Revision ID: 025
Revises: 024
Create Date: 2026-04-09

Phase 41 decisions (.planning/phases/41-research-api/41-CONTEXT.md):

- D-04: training_runs schema — UUID PK, handler/model class + params,
  market/symbols/interval/segments, status lifecycle, metrics JSONB,
  FK to model_versions, mlflow_run_id, error, requested_by, timestamps.
- D-05: Status enum — pending / running / succeeded / failed / cancelled.
  Enforced by a CHECK constraint (same pattern as backfill_jobs).
- D-15: MLflow schema — ``CREATE SCHEMA IF NOT EXISTS mlflow``. MLflow
  auto-creates its own internal tables on first use; Poseidon does NOT
  manage MLflow's internal tables.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade():
    # D-15: MLflow stores its metadata in the ``mlflow`` schema.
    # We create the schema; MLflow auto-creates its tables on first use.
    op.execute("CREATE SCHEMA IF NOT EXISTS mlflow")

    # D-04: training_runs table
    op.create_table(
        "training_runs",
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("handler_class", sa.String(64), nullable=False),
        sa.Column(
            "handler_params",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("model_class", sa.String(64), nullable=False),
        sa.Column(
            "model_params",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column(
            "symbols",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column(
            "segments",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("lookback", sa.Text, nullable=True),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="pending"
        ),
        sa.Column("metrics", JSONB, nullable=True),
        sa.Column(
            "model_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column("mlflow_run_id", sa.Text, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column(
            "requested_by",
            sa.String(16),
            nullable=False,
            server_default="api",
        ),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "finished_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # D-05: Status CHECK constraint
    op.create_check_constraint(
        "ck_training_runs_status",
        "training_runs",
        "status IN ('pending','running','succeeded','failed','cancelled')",
    )

    # Indexes for GET /runs filtering and pagination
    op.create_index(
        "ix_training_runs_status", "training_runs", ["status"]
    )
    op.create_index(
        "ix_training_runs_created_at", "training_runs", ["created_at"]
    )


def downgrade():
    op.drop_table("training_runs")
    op.execute("DROP SCHEMA IF EXISTS mlflow CASCADE")
