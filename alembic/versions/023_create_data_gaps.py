"""Create data_gaps table (Phase 40 plan 40-01, D-04/D-07/D-09).

Revision ID: 023
Revises: 022
Create Date: 2026-04-09

Phase 40 decisions (.planning/phases/40-data-health-observability/40-CONTEXT.md):

- D-04: ``data_gaps`` schema = (gap_id UUID PK, market, symbol, interval,
  gap_start, gap_end, missing_bars, detected_at, healed_at). The unique
  index on (market, symbol, interval, gap_start) is mandatory so the daily
  audit's ``INSERT ... ON CONFLICT DO NOTHING`` contract stays idempotent
  across reruns (D-07).
- D-07: Heal lifecycle — rows are NEVER deleted. When a later audit run
  sees the window is now fully populated, it updates ``healed_at = now()``.
- D-09: Audit scope is every ``(market, symbol, interval)`` present in
  ``data_coverage_mv``, so the schema must not impose any foreign-key
  constraint back to the symbols universe (it auto-tracks ``symbols.yaml``
  via the MV instead).

A second partial index ``ix_data_gaps_open`` accelerates the default
``GET /api/data/gaps?open_only=true`` path (D-02) — operators query open
gaps far more often than healed ones.

Upgrade issues ``op.create_table("data_gaps", ...)`` followed by
``op.create_index("ix_data_gaps_tuple_start", ..., unique=True)`` and the
partial open-gap index. Downgrade drops them in reverse order.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "data_gaps",
        sa.Column(
            "gap_id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("gap_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gap_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("missing_bars", sa.Integer(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("healed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # D-04/D-07: this UNIQUE index is the idempotency anchor. Without it the
    # daily audit's ``INSERT ... ON CONFLICT (market, symbol, interval,
    # gap_start) DO NOTHING`` contract has no target to match on.
    op.create_index(
        "ix_data_gaps_tuple_start",
        "data_gaps",
        ["market", "symbol", "interval", "gap_start"],
        unique=True,
    )

    # D-02: the dashboard and the default ``open_only=true`` query want the
    # list of unhealed gaps. A partial index keeps that read cheap and keeps
    # the healed history out of the hot path.
    op.create_index(
        "ix_data_gaps_open",
        "data_gaps",
        ["market", "symbol", "interval"],
        postgresql_where=sa.text("healed_at IS NULL"),
    )


def downgrade():
    # Drop in reverse of create order: partial index, unique index, table.
    op.drop_index("ix_data_gaps_open", table_name="data_gaps")
    op.drop_index("ix_data_gaps_tuple_start", table_name="data_gaps")
    op.drop_table("data_gaps")
