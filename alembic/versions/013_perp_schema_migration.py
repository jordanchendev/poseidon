"""Perp schema migration: int->float quantities, add side column.

Revision ID: 013
Revises: 012
Create Date: 2026-03-31

Changes:
- orders.quantity: Integer -> Float
- orders.side: new column (String(16), default 'long')
- order_fills.fill_quantity: Integer -> Float
- portfolio_holdings.shares: Integer -> Float
- portfolio_holdings.side: new column (String(16), default 'long')
"""

import sqlalchemy as sa
from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- orders table --
    op.alter_column(
        "orders",
        "quantity",
        type_=sa.Float,
        existing_type=sa.Integer,
        existing_nullable=False,
    )
    op.add_column(
        "orders",
        sa.Column("side", sa.String(16), server_default="long", nullable=False),
    )

    # -- order_fills table --
    op.alter_column(
        "order_fills",
        "fill_quantity",
        type_=sa.Float,
        existing_type=sa.Integer,
        existing_nullable=False,
    )

    # -- portfolio_holdings table --
    op.alter_column(
        "portfolio_holdings",
        "shares",
        type_=sa.Float,
        existing_type=sa.Integer,
        existing_nullable=True,
    )
    op.add_column(
        "portfolio_holdings",
        sa.Column("side", sa.String(16), server_default="long", nullable=False),
    )


def downgrade() -> None:
    # -- portfolio_holdings table --
    op.drop_column("portfolio_holdings", "side")
    op.alter_column(
        "portfolio_holdings",
        "shares",
        type_=sa.Integer,
        existing_type=sa.Float,
        existing_nullable=True,
    )

    # -- order_fills table --
    op.alter_column(
        "order_fills",
        "fill_quantity",
        type_=sa.Integer,
        existing_type=sa.Float,
        existing_nullable=False,
    )

    # -- orders table --
    op.drop_column("orders", "side")
    op.alter_column(
        "orders",
        "quantity",
        type_=sa.Integer,
        existing_type=sa.Float,
        existing_nullable=False,
    )
