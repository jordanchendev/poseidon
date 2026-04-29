"""Add signal_id FK to orders with zero-downtime pattern (Phase 89-01).

Revision ID: 036
Revises: 035
Create Date: 2026-04-29

Phase 89-01 (D-04, D-27): closes the F8 wiring breach where Phase 88 found
25 orders with zero matching signals. This adds the audit-grade FK so the
mini-audit can join orders.signal_id -> signals.id directly instead of
guessing via timestamp + symbol heuristics.

Mirrors migration 017 (trade_logs.signal_id) zero-downtime pattern:
1. Add nullable column (instant, no rewrite).
2. Add FK constraint NOT VALID (SHARE UPDATE EXCLUSIVE only — concurrent writes ok).
3. VALIDATE CONSTRAINT (allows concurrent writes during scan).
4. Index for the mini-audit join.

Backfill is intentionally skipped (D-15: legacy untouched). The 25 orphan
orders from Phase 88 stay orphan — Plan 89-02 prevents new orphans by
wiring perp_rebalance through SignalRepository, and adds order_origin tags
for protective close paths.
"""

import sqlalchemy as sa

from alembic import op

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Add nullable column (instant, no lock).
    op.add_column(
        "orders",
        sa.Column("signal_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
    )
    # Step 2: Add FK constraint as NOT VALID (no full-table scan, brief lock only).
    op.execute(
        """
        ALTER TABLE orders
        ADD CONSTRAINT fk_orders_signal_id
        FOREIGN KEY (signal_id) REFERENCES signals(id)
        NOT VALID
        """
    )
    # Step 3: Validate constraint (allows concurrent writes during scan).
    op.execute(
        """
        ALTER TABLE orders
        VALIDATE CONSTRAINT fk_orders_signal_id
        """
    )
    # Step 4: Index for the mini-audit join (orders.signal_id -> signals.id).
    op.create_index("ix_orders_signal_id", "orders", ["signal_id"])


def downgrade():
    op.drop_index("ix_orders_signal_id", "orders")
    op.drop_constraint("fk_orders_signal_id", "orders", type_="foreignkey")
    op.drop_column("orders", "signal_id")
