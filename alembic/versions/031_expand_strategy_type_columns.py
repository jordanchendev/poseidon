"""Expand strategy_type columns for portfolio strategies.

Revision ID: 031
Revises: 030
Create Date: 2026-04-19

`portfolio_strategy` is longer than the legacy 16-character column width used
by both `strategies.strategy_type` and `backtests.strategy_type`. Widen both
columns so public CRUD and persisted backtest results work on Postgres.
"""

import sqlalchemy as sa
from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        "strategies",
        "strategy_type",
        existing_type=sa.String(length=16),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "backtests",
        "strategy_type",
        existing_type=sa.String(length=16),
        type_=sa.String(length=32),
        existing_nullable=False,
    )


def downgrade():
    op.alter_column(
        "backtests",
        "strategy_type",
        existing_type=sa.String(length=32),
        type_=sa.String(length=16),
        existing_nullable=False,
    )
    op.alter_column(
        "strategies",
        "strategy_type",
        existing_type=sa.String(length=32),
        type_=sa.String(length=16),
        existing_nullable=False,
    )
