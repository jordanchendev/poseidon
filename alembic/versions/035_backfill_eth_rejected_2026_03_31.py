"""Backfill structured reject_reason for 3 ETHUSDT rejected orders on 2026-03-31 (TRUTH-03, D-16).

Revision ID: 035
Revises: 034
Create Date: 2026-04-29

Per CONTEXT D-16, the 3 ETHUSDT orders rejected on 2026-03-31 between
09:36 and 09:42 UTC predate structured reject_reason. Root cause cannot
be reconstructed faithfully (audit (2026-04-29) confirmed legacy text was:
- 09:36:08 -> 'max_exposure: projected 125.00% exceeds limit 100.00%'
- 09:40:23 -> 'max_exposure: projected 105.00% exceeds limit 100.00%'
- 09:42:56 -> 'stop_loss: no stop-loss configured for buy orders'

Migration 034 already wrapped these as {check_name='legacy', rule='pre_phase87',
details=<text>}. This migration overwrites those 3 rows with the explicit
backfill marker per D-16, acknowledging incomplete provenance rather than
fabricating shortfall numbers.

Idempotent: WHERE rule IS DISTINCT FROM 'pre_phase87_backfill' guards re-runs.
"""

from alembic import op

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None

BACKFILL_PAYLOAD = """
jsonb_build_object(
    'check_name', 'unknown_legacy',
    'rule', 'pre_phase87_backfill',
    'shortfall', NULL,
    'details', 'Reconstructed from order log on 2026-04-28; root cause not preserved at rejection time'
)
"""


def upgrade():
    op.execute(
        f"""
        UPDATE orders
        SET reject_reason = {BACKFILL_PAYLOAD}
        WHERE symbol = 'ETHUSDT'
          AND status = 'rejected'
          AND created_at >= TIMESTAMP WITH TIME ZONE '2026-03-31 09:30:00+00'
          AND created_at <= TIMESTAMP WITH TIME ZONE '2026-03-31 09:50:00+00'
          AND (reject_reason IS NULL OR reject_reason->>'rule' IS DISTINCT FROM 'pre_phase87_backfill')
        """
    )


def downgrade():
    # Restore the legacy-wrapped form (what migration 034 produced).
    # Original text values are preserved verbatim per audit findings.
    op.execute(
        """
        UPDATE orders
        SET reject_reason = jsonb_build_object(
            'check_name', 'legacy',
            'rule', 'pre_phase87',
            'shortfall', NULL,
            'details', 'max_exposure: projected 125.00% exceeds limit 100.00%'
        )
        WHERE symbol = 'ETHUSDT' AND status = 'rejected'
          AND created_at = TIMESTAMP WITH TIME ZONE '2026-03-31 09:36:08.528531+00'
        """
    )
    op.execute(
        """
        UPDATE orders
        SET reject_reason = jsonb_build_object(
            'check_name', 'legacy',
            'rule', 'pre_phase87',
            'shortfall', NULL,
            'details', 'max_exposure: projected 105.00% exceeds limit 100.00%'
        )
        WHERE symbol = 'ETHUSDT' AND status = 'rejected'
          AND created_at = TIMESTAMP WITH TIME ZONE '2026-03-31 09:40:23.467213+00'
        """
    )
    op.execute(
        """
        UPDATE orders
        SET reject_reason = jsonb_build_object(
            'check_name', 'legacy',
            'rule', 'pre_phase87',
            'shortfall', NULL,
            'details', 'stop_loss: no stop-loss configured for buy orders'
        )
        WHERE symbol = 'ETHUSDT' AND status = 'rejected'
          AND created_at = TIMESTAMP WITH TIME ZONE '2026-03-31 09:42:56.290423+00'
        """
    )
