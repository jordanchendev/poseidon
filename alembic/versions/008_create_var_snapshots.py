"""Create var_snapshots hypertable.

Revision ID: 008
Revises: 007
Create Date: 2026-03-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "var_snapshots",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("method", sa.String(32), nullable=False),
        sa.Column("var_95", sa.Numeric, nullable=False),
        sa.Column("var_99", sa.Numeric, nullable=False),
        sa.Column("cvar_95", sa.Numeric, nullable=False),
        sa.Column("cvar_99", sa.Numeric, nullable=False),
        sa.Column("portfolio_value", sa.Numeric, nullable=False),
        sa.Column("holding_period", sa.Integer, nullable=False, server_default="1"),
        sa.Column("details", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("time", "method", name="pk_var_snapshots"),
    )
    op.execute(
        "SELECT create_hypertable('var_snapshots', 'time', "
        "chunk_time_interval => INTERVAL '1 week')"
    )


def downgrade():
    op.drop_table("var_snapshots")
