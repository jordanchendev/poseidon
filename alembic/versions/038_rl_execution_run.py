"""Create rl_execution_runs table (Phase 90 D-25 — Wave 4a).

Revision ID: 038
Revises: 037
Create Date: 2026-05-03

Phase 90 decisions (.planning/phases/90-rl-order-execution/90-CONTEXT.md):

- D-25: Run lifecycle adopts the v8.0 Phase 41 ``training_runs`` pattern
  (pending -> running -> succeeded / failed / cancelled, durable Postgres
  row, cooperative cancel via row re-read). Schema mirrors
  ``025_create_training_runs.py`` with the column drift documented in
  ``.planning/phases/90-rl-order-execution/90-PATTERNS.md``:

  Dropped (RL has no model registry link):
    handler_class, handler_params, model_class, model_params,
    market, symbols, interval, segments, lookback, mlflow_run_id,
    model_version_id, metrics

  Added (Phase 90 specific):
    algos       JSONB list of requested algos (twap/vwap/ppo/opds)
    dates       JSONB explicit trigger-day list (null = full window)
    mode        full | preflight
    summary     JSONB per-algo summary written at finalize
    verdict     deploy | refine | kill (only set on success)
    result_dir  path to local_dev/rl-execution/runs/<run_id>/

  Kept (lifecycle invariants, identical to training_runs):
    status, error, requested_by, started_at, finished_at,
    created_at, updated_at, status CHECK constraint, and the same
    two indexes on (status, created_at).

The ``rl_execution_runs`` table is additive — existing tables are
untouched, so this migration is reversible (downgrade drops only the
new table) and rerun-safe (alembic version_num row gates re-application).
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade():
    # D-25: rl_execution_runs table (Phase 41 lifecycle pattern).
    op.create_table(
        "rl_execution_runs",
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "algos",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("dates", JSONB, nullable=True),
        sa.Column("mode", sa.String(16), nullable=False, server_default="full"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("summary", JSONB, nullable=True),
        sa.Column("verdict", sa.String(16), nullable=True),
        sa.Column("result_dir", sa.Text, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column(
            "requested_by",
            sa.String(16),
            nullable=False,
            server_default="api",
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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

    # D-25: Status CHECK constraint — same five lifecycle states as
    # training_runs (Phase 41 D-05 verbatim).
    op.create_check_constraint(
        "ck_rl_execution_runs_status",
        "rl_execution_runs",
        "status IN ('pending','running','succeeded','failed','cancelled')",
    )

    # Indexes for GET /runs filtering and pagination (mirror Phase 41).
    op.create_index("ix_rl_execution_runs_status", "rl_execution_runs", ["status"])
    op.create_index(
        "ix_rl_execution_runs_created_at",
        "rl_execution_runs",
        ["created_at"],
    )


def downgrade():
    op.drop_index("ix_rl_execution_runs_created_at", table_name="rl_execution_runs")
    op.drop_index("ix_rl_execution_runs_status", table_name="rl_execution_runs")
    op.drop_table("rl_execution_runs")
