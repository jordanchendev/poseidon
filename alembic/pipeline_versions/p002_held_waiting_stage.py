"""Let ``awaiting_release`` wait without a ``wait_until`` (held for a human release).

Revision ID: p002
Revises: p001
Create Date: 2026-10-23

Step 7 of the strategy-pipeline plan: ``awaiting_release`` is a waiting stage
the sweeper must never advance, so it carries no ``wait_until``. Every other
waiting attempt still needs one. Run with ``alembic -n pipeline upgrade head``
against the isolated ``poseidon_pipeline`` database.
"""

from alembic import op

revision = "p002"
down_revision = "p001"
branch_labels = None
depends_on = None

OLD = "status <> 'waiting' OR wait_until IS NOT NULL"
NEW = "status <> 'waiting' OR wait_until IS NOT NULL OR stage IN ('awaiting_release')"


def upgrade():
    op.drop_constraint("ck_stage_runs_waiting", "stage_runs", type_="check")
    op.create_check_constraint("ck_stage_runs_waiting", "stage_runs", NEW)


def downgrade():
    # Fails while an awaiting_release attempt is held; abort those runs first.
    op.drop_constraint("ck_stage_runs_waiting", "stage_runs", type_="check")
    op.create_check_constraint("ck_stage_runs_waiting", "stage_runs", OLD)
