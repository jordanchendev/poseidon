"""Create strategies table.

Revision ID: 006
Revises: 005
Create Date: 2026-03-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "strategies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("strategy_type", sa.String(16), nullable=False),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False, server_default="1d"),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("model_version_id", UUID(as_uuid=True), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("name", name="uq_strategies_name"),
    )
    op.create_index("idx_strategies_active", "strategies", ["active"])
    op.create_index("idx_strategies_market_symbol", "strategies", ["market", "symbol"])


def downgrade():
    op.drop_table("strategies")
