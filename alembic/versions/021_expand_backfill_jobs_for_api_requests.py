"""Expand backfill_jobs for request-level API jobs (Phase 39).

Revision ID: 021
Revises: 020
Create Date: 2026-04-09

Phase 39 backfill-api-coverage:

A single user-triggered backfill job may now cover multiple symbols and
intervals (D-03/D-08 in 39-CONTEXT.md). The Phase 38 substrate assumed one
row per (symbol, interval) tuple and enforced NOT NULL on both columns.
This migration relaxes that assumption while preserving the durable row
contract Phase 38 committed to.

Changes:
- backfill_jobs.symbol   -> NULLABLE (multi-symbol request jobs)
- backfill_jobs.interval -> NULLABLE (multi-interval request jobs)
- + symbols       JSONB NOT NULL DEFAULT '[]'::jsonb
- + intervals     JSONB NOT NULL DEFAULT '[]'::jsonb
- + requested_by  VARCHAR(16) NOT NULL DEFAULT 'api'
- + started_at    TIMESTAMPTZ NULL
- + finished_at   TIMESTAMPTZ NULL
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("backfill_jobs", "symbol", existing_type=sa.String(32), nullable=True)
    op.alter_column("backfill_jobs", "interval", existing_type=sa.String(8), nullable=True)

    op.add_column(
        "backfill_jobs",
        sa.Column(
            "symbols",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "backfill_jobs",
        sa.Column(
            "intervals",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "backfill_jobs",
        sa.Column(
            "requested_by",
            sa.String(16),
            nullable=False,
            server_default="api",
        ),
    )
    op.add_column(
        "backfill_jobs",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "backfill_jobs",
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade():
    op.drop_column("backfill_jobs", "finished_at")
    op.drop_column("backfill_jobs", "started_at")
    op.drop_column("backfill_jobs", "requested_by")
    op.drop_column("backfill_jobs", "intervals")
    op.drop_column("backfill_jobs", "symbols")

    # Restore NOT NULL constraints (down path assumes no multi-tuple rows
    # remain; operators should purge pending Phase 39 jobs before rollback).
    op.alter_column("backfill_jobs", "interval", existing_type=sa.String(8), nullable=False)
    op.alter_column("backfill_jobs", "symbol", existing_type=sa.String(32), nullable=False)
