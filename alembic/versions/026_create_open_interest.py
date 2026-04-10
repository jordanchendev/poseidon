"""Create open_interest hypertable (Phase 43.1 G-01).

Revision ID: 026
Revises: 025
Create Date: 2026-04-10

Open interest data for crypto perpetual contracts. Stores OI snapshots
at configurable intervals (5m, 1h, 4h) for liquidity sweep detection.
PK is (time, symbol, interval) to support multiple timeframes per symbol.
"""

import sqlalchemy as sa
from alembic import op

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "open_interest",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("open_interest", sa.Numeric, nullable=False),
        sa.Column("open_interest_value", sa.Numeric, nullable=True),
        sa.PrimaryKeyConstraint("time", "symbol", "interval", name="pk_open_interest"),
    )

    # Index for fast per-symbol time-ordered queries
    op.create_index(
        "idx_open_interest_symbol_time",
        "open_interest",
        ["symbol", sa.text("time DESC")],
    )

    # Convert to TimescaleDB hypertable for efficient time-series queries
    op.execute(
        "SELECT create_hypertable('open_interest', 'time', if_not_exists => TRUE)"
    )


def downgrade():
    op.drop_table("open_interest")
