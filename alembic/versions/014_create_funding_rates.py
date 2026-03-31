"""Create funding_rates table for perpetual contract funding rate history.

Revision ID: 014
Revises: 013
Create Date: 2026-03-31

Creates:
- funding_rates table (TimescaleDB hypertable on 'time')
  Columns: time, symbol, funding_rate, mark_price, index_price
  PK: (time, symbol)
"""

import sqlalchemy as sa
from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "funding_rates",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("funding_rate", sa.Float, nullable=False),
        sa.Column("mark_price", sa.Float, nullable=True),
        sa.Column("index_price", sa.Float, nullable=True),
        sa.PrimaryKeyConstraint("time", "symbol", name="pk_funding_rates"),
    )

    # Create index for efficient symbol+time queries
    op.create_index(
        "idx_funding_rates_symbol_time",
        "funding_rates",
        ["symbol", sa.text("time DESC")],
    )

    # Convert to TimescaleDB hypertable for time-series optimization
    op.execute(
        "SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.drop_table("funding_rates")
