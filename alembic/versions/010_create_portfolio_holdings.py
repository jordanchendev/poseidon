"""Create portfolio_holdings table.

Revision ID: 010
Revises: 009
Create Date: 2026-03-30
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portfolio_holdings",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("shares", sa.Integer, nullable=True),
        sa.Column("entry_price", sa.Float, nullable=True),
        sa.Column("entry_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "closed",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("close_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_loss_pct", sa.Float, nullable=True),
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
            nullable=False,
        ),
    )
    op.create_index(
        "ix_portfolio_holdings_strategy_closed",
        "portfolio_holdings",
        ["strategy_name", "closed"],
    )


def downgrade():
    op.drop_index("ix_portfolio_holdings_strategy_closed")
    op.drop_table("portfolio_holdings")
