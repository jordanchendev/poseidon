"""Convert orders.reject_reason from Text to JSONB and wrap legacy values (TRUTH-03).

Revision ID: 034
Revises: 033
Create Date: 2026-04-29

reject_reason is restructured to a 4-key dict:
    {check_name, rule, shortfall, details}

Migration strategy (avoids USING-cast type-incompat issues):
1. Add new JSONB column reject_reason_new (nullable)
2. Backfill: text values become {check_name='legacy', rule='legacy',
   shortfall=null, details=<original text>}; NULLs stay NULL.
3. Drop old reject_reason
4. Rename reject_reason_new -> reject_reason

Audit (2026-04-29): only 3 non-null reject_reason values exist (the same 3
ETHUSDT 2026-03-31 rejected orders that migration 035 will overwrite),
but wrap-on-migrate semantics are still applied for any future re-runs.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade():
    # Add new JSONB column
    op.add_column(
        "orders",
        sa.Column("reject_reason_new", postgresql.JSONB(), nullable=True),
    )

    # Wrap legacy text values into 4-key dict
    op.execute(
        """
        UPDATE orders
        SET reject_reason_new = jsonb_build_object(
            'check_name', 'legacy',
            'rule', 'legacy',
            'shortfall', NULL,
            'details', reject_reason
        )
        WHERE reject_reason IS NOT NULL
        """
    )

    # Drop old text column
    op.drop_column("orders", "reject_reason")

    # Rename new column to canonical name
    op.alter_column(
        "orders",
        "reject_reason_new",
        new_column_name="reject_reason",
    )


def downgrade():
    # Reverse: JSONB -> Text. Extract details field to preserve original text.
    op.add_column(
        "orders",
        sa.Column("reject_reason_old", sa.Text(), nullable=True),
    )

    op.execute(
        """
        UPDATE orders
        SET reject_reason_old = reject_reason->>'details'
        WHERE reject_reason IS NOT NULL
        """
    )

    op.drop_column("orders", "reject_reason")
    op.alter_column(
        "orders",
        "reject_reason_old",
        new_column_name="reject_reason",
    )
