"""Create quality_scores hypertable.

Revision ID: 009
Revises: 008
Create Date: 2026-03-28
"""

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "quality_scores",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("score", sa.Numeric, nullable=False),
        sa.Column("completeness", sa.Numeric, nullable=False),
        sa.Column("consistency", sa.Numeric, nullable=False),
        sa.Column("anomaly_free", sa.Numeric, nullable=False),
        sa.Column("timeliness", sa.Numeric, nullable=False),
        sa.PrimaryKeyConstraint(
            "time", "symbol", "interval", name="pk_quality_scores"
        ),
    )
    op.execute(
        "SELECT create_hypertable('quality_scores', 'time', "
        "chunk_time_interval => INTERVAL '1 week')"
    )


def downgrade():
    op.drop_table("quality_scores")
