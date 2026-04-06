"""Create universe_snapshots table.

Revision ID: 018
Revises: 017
Create Date: 2026-04-06

Changes:
- CREATE TABLE universe_snapshots for storing point-in-time universe resolution results
- Composite index on (market, snapshot_time DESC) for efficient lookups
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "018"
down_revision = "017"


def upgrade():
    op.create_table(
        "universe_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbols", postgresql.JSONB(), nullable=False),
        sa.Column("source_type", sa.String(64), nullable=False),
        sa.Column("filter_config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_universe_snapshots_market", "universe_snapshots", ["market"])
    op.create_index(
        "ix_universe_snapshots_snapshot_time", "universe_snapshots", ["snapshot_time"]
    )
    op.create_index(
        "ix_universe_snapshots_market_time",
        "universe_snapshots",
        ["market", sa.text("snapshot_time DESC")],
    )


def downgrade():
    op.drop_index("ix_universe_snapshots_market_time", "universe_snapshots")
    op.drop_index("ix_universe_snapshots_snapshot_time", "universe_snapshots")
    op.drop_index("ix_universe_snapshots_market", "universe_snapshots")
    op.drop_table("universe_snapshots")
