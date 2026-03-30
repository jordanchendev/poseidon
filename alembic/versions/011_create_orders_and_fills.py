"""Create orders and order_fills tables.

Revision ID: 011
Revises: 010
Create Date: 2026-03-30
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade():
    # --- orders table ---
    op.create_table(
        "orders",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column(
            "order_type",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'market'"),
        ),
        sa.Column("target_weight", sa.Float, nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("price", sa.Float, nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("broker_order_id", sa.String(64), nullable=True),
        sa.Column("broker_mode", sa.String(16), nullable=False),
        sa.Column("reject_reason", sa.Text, nullable=True),
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
        "ix_orders_strategy_status", "orders", ["strategy_name", "status"]
    )
    op.create_index("ix_orders_symbol", "orders", ["symbol"])

    # --- order_fills table ---
    op.create_table(
        "order_fills",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "order_id",
            UUID(as_uuid=True),
            sa.ForeignKey("orders.id"),
            nullable=False,
        ),
        sa.Column("fill_price", sa.Float, nullable=False),
        sa.Column("fill_quantity", sa.Integer, nullable=False),
        sa.Column("fill_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("broker_fill_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_order_fills_order_id", "order_fills", ["order_id"])


def downgrade():
    # Drop order_fills first (FK dependency on orders)
    op.drop_index("ix_order_fills_order_id")
    op.drop_table("order_fills")

    op.drop_index("ix_orders_symbol")
    op.drop_index("ix_orders_strategy_status")
    op.drop_table("orders")
