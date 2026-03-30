"""Create nav_snapshots and trade_logs tables.

Revision ID: 012
Revises: 011
Create Date: 2026-03-30
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade():
    # --- nav_snapshots table ---
    op.create_table(
        "nav_snapshots",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("snapshot_date", sa.Date, nullable=False, unique=True),
        sa.Column("total_nav", sa.Float, nullable=False),
        sa.Column("holdings_value", sa.Float, nullable=False),
        sa.Column("cash", sa.Float, nullable=False),
        sa.Column("holdings_count", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # --- trade_logs table ---
    op.create_table(
        "trade_logs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("exit_price", sa.Float, nullable=False),
        sa.Column("entry_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("shares", sa.Integer, nullable=False),
        sa.Column("realized_pnl", sa.Float, nullable=False),
        sa.Column("holding_days", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_trade_logs_strategy_symbol", "trade_logs", ["strategy_name", "symbol"]
    )


def downgrade():
    op.drop_index("ix_trade_logs_strategy_symbol")
    op.drop_table("trade_logs")
    op.drop_table("nav_snapshots")
