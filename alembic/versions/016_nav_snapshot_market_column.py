"""Add market column to nav_snapshots.

Revision ID: 016
Revises: 015
Create Date: 2026-03-31

Changes:
- nav_snapshots.market: new nullable String(50) column (default NULL for existing rows)
- Replace unique constraint on snapshot_date with (snapshot_date, market)
"""

import sqlalchemy as sa
from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "nav_snapshots",
        sa.Column("market", sa.String(50), nullable=True),
    )
    # Drop old unique constraint on snapshot_date only (if exists)
    op.execute(
        "ALTER TABLE nav_snapshots DROP CONSTRAINT IF EXISTS uq_nav_snapshot_date"
    )
    # Create new composite unique constraint
    op.create_unique_constraint(
        "uq_nav_snapshot_date_market",
        "nav_snapshots",
        ["snapshot_date", "market"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_nav_snapshot_date_market", "nav_snapshots", type_="unique")
    op.drop_column("nav_snapshots", "market")
