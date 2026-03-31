"""Add entry_type to trade_logs and convert shares to Float.

Revision ID: 015
Revises: 014
Create Date: 2026-03-31

Changes:
- trade_logs.shares: Integer -> Float (support fractional shares for perps)
- trade_logs.entry_type: new column (String(16), default 'trade')
  Values: 'trade' (normal buy/sell), 'funding' (funding rate settlement)
"""

import sqlalchemy as sa
from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trade_logs",
        sa.Column("entry_type", sa.String(16), server_default="trade", nullable=False),
    )
    op.alter_column(
        "trade_logs",
        "shares",
        type_=sa.Float,
        existing_type=sa.Integer,
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "trade_logs",
        "shares",
        type_=sa.Integer,
        existing_type=sa.Float,
        existing_nullable=False,
    )
    op.drop_column("trade_logs", "entry_type")
