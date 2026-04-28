"""Add cash_flow column to nav_snapshots (TRUTH-04, D-05).

Revision ID: 032
Revises: 031
Create Date: 2026-04-29

Adds NUMERIC(18, 2) NOT NULL DEFAULT 0 column for cash flow accounting.
Positive=deposit, negative=withdrawal. Required for Modified Dietz TWR
to isolate the 2026-04-28 $9.87M deposit from total_return_pct.
"""

import sqlalchemy as sa

from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "nav_snapshots",
        sa.Column(
            "cash_flow",
            sa.Numeric(18, 2),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade():
    op.drop_column("nav_snapshots", "cash_flow")
