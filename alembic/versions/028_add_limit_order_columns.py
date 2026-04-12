"""Add limit order columns to signals table.

Revision ID: 028
Revises: 027
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("order_type", sa.String(16), nullable=True))
    op.add_column("signals", sa.Column("order_price", sa.Float, nullable=True))
    op.add_column("signals", sa.Column("stop_loss_price", sa.Float, nullable=True))
    op.add_column("signals", sa.Column("take_profit_price", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("signals", "take_profit_price")
    op.drop_column("signals", "stop_loss_price")
    op.drop_column("signals", "order_price")
    op.drop_column("signals", "order_type")
