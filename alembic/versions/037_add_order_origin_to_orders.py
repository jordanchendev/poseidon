"""Add order_origin column to orders for audit-friendly origin tagging (Phase 89-02 W4).

Revision ID: 037
Revises: 036
Create Date: 2026-04-29

Phase 89-02 Task 3 (D-04 / W4 audit whitelist): tags how each order was
triggered so the mini-audit (89-03 Plan) can distinguish:

* signal_id NULL + origin in {stop_loss, liquidation, manual} -> legitimate Cat-B
* signal_id NULL + origin == 'signal'                          -> wiring breach

Without this column the auditor cannot tell whether an orphan order is a
by-design protective close (e.g. portfolio_stop_loss_monitor / perp
liquidation_monitor) or a genuine missing-wiring breach (the F8 finding
that Phase 88 surfaced with 25 orphan orders).

Schema change pattern -- direct add (instant, no rewrite) because:
* The new column has a server_default = 'signal', so existing rows get
  the legacy value (correct semantics: pre-89 orders predate origin tagging
  and were predominantly signal-driven from the perp_rebalance path that
  was already wired to a defunct strategy).
* No FK constraint, no full-table validate scan needed.
* Index is non-blocking (CREATE INDEX is fast on small tables; orders is
  ~25 rows in production at the time of this migration).

Backfill is intentionally skipped -- legacy 25 orphan orders carry the
default 'signal' tag, which combined with their NULL signal_id will show
up in the post-89 audit as historical Cat-B (out of new audit window per
D-15). New orders post-89 land with the correct origin tag from
execute_rebalance kwargs added in this same plan.
"""

import sqlalchemy as sa

from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Add column with NOT NULL DEFAULT 'signal' (rewrites are cheap on
    # this small table; in postgres 11+ this is metadata-only).
    op.add_column(
        "orders",
        sa.Column(
            "order_origin",
            sa.String(length=32),
            nullable=False,
            server_default="signal",
        ),
    )
    # Step 2: Index for the audit query "GROUP BY order_origin WHERE signal_id IS NULL".
    op.create_index("ix_orders_order_origin", "orders", ["order_origin"])


def downgrade():
    op.drop_index("ix_orders_order_origin", table_name="orders")
    op.drop_column("orders", "order_origin")
