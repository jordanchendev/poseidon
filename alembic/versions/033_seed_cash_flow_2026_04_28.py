"""Seed $9,870,000.00 cash deposit on 2026-04-28 (TRUTH-04, D-07).

Revision ID: 033
Revises: 032
Create Date: 2026-04-29

The 2026-04-28 health check identified a $9.87M cash deposit that
contaminated total_return_pct (98.70% bug). Per CONTEXT D-07, seed the
deposit into the cash_flow column on the corresponding NAV snapshot row.

Audit (2026-04-29) found exactly one nav_snapshot on 2026-04-28 with
market='crypto_perp' (total_nav=$9,970,236.89, holdings_value=0.0,
holdings_count=1). Seed targets that single row.

Idempotent: WHERE cash_flow = 0 ensures re-running is a no-op.
"""

from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        UPDATE nav_snapshots
        SET cash_flow = 9870000.00
        WHERE snapshot_date = DATE '2026-04-28'
          AND market = 'crypto_perp'
          AND cash_flow = 0
        """
    )


def downgrade():
    op.execute(
        """
        UPDATE nav_snapshots
        SET cash_flow = 0
        WHERE snapshot_date = DATE '2026-04-28'
          AND market = 'crypto_perp'
          AND cash_flow = 9870000.00
        """
    )
