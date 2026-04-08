"""Create ingest_state and backfill_jobs; drop backfill_progress.

Revision ID: 020
Revises: 019
Create Date: 2026-04-08

Phase 38 data-foundation (D-01, D-10):
- CREATE TABLE ingest_state — cursor state per (symbol, market, interval)
- CREATE TABLE backfill_jobs — job_id/status/cursor/progress (replaces backfill_progress)
- DROP TABLE backfill_progress — clean slate per decision D-10
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade():
    # ingest_state: cursor bookkeeping for self-healing ingest
    op.create_table(
        "ingest_state",
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("last_successful_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_attempt_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "first_backfill_done",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("symbol", "market", "interval", name="pk_ingest_state"),
    )

    # backfill_jobs: replaces backfill_progress (Phase 38 D-10)
    op.create_table(
        "backfill_jobs",
        sa.Column(
            "job_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("cursor", JSONB, nullable=True),
        sa.Column("progress", JSONB, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="ck_backfill_jobs_status",
        ),
    )
    op.create_index("ix_backfill_jobs_status", "backfill_jobs", ["status"])

    # Drop legacy backfill_progress table (D-10 clean slate)
    op.drop_table("backfill_progress")


def downgrade():
    # Recreate backfill_progress with original v7.0 schema
    op.create_table(
        "backfill_progress",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("last_fetched_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("target_start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "symbol", "market", "interval", name="uq_backfill_symbol_market_interval"
        ),
    )

    op.drop_index("ix_backfill_jobs_status", table_name="backfill_jobs")
    op.drop_table("backfill_jobs")
    op.drop_table("ingest_state")
