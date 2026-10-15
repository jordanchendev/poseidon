"""Admit a documented first-activation bootstrap release event.

Revision ID: 046
Revises: 044
Create Date: 2026-10-15

045 is reserved for the v21 research agenda (SEED-002); 046 = bootstrap (Phase 101
D-19/D-27); 047 = hardening. 046 never modifies 042/043/044 objects other than
replacing the two release-event CHECKs, and its downgrade never deletes rows.

A bootstrap event is the only way to set the first active StrategyVersion without
campaign evidence. Its shape forbids campaign, review, holdout, and
reconciliation ids, requires a NULL prior pointer and expected_revision 0, and
still flows through the unchanged 044 pointer trigger.
"""

import sqlalchemy as sa

from alembic import op

revision = "046"
down_revision = "044"
branch_labels = None
depends_on = None

TYPE_CHECK = "ck_strategy_release_events_type"
SHAPE_CHECK = "ck_strategy_release_events_shape"
TABLE = "strategy_release_events"

TYPE_044 = "event_type IN ('release', 'pause', 'rollback', 'reject')"
TYPE_046 = "event_type IN ('release', 'pause', 'rollback', 'reject', 'bootstrap')"

SHAPE_044 = (
    "(event_type = 'release' AND candidate_version_id IS NOT NULL "
    "AND to_version_id = candidate_version_id AND source_release_event_id IS NULL "
    "AND campaign_id IS NOT NULL AND review_id IS NOT NULL "
    "AND holdout_use_id IS NOT NULL AND reconciliation_id IS NOT NULL) OR "
    "(event_type = 'pause' AND candidate_version_id IS NULL "
    "AND to_version_id IS NULL AND source_release_event_id IS NULL) OR "
    "(event_type = 'rollback' AND candidate_version_id IS NOT NULL "
    "AND to_version_id = candidate_version_id AND source_release_event_id IS NOT NULL "
    "AND reconciliation_id IS NOT NULL) OR "
    "(event_type = 'reject' AND candidate_version_id IS NOT NULL "
    "AND to_version_id IS NOT DISTINCT FROM from_version_id "
    "AND source_release_event_id IS NULL)"
)
SHAPE_046 = SHAPE_044 + (
    " OR "
    "(event_type = 'bootstrap' AND candidate_version_id IS NOT NULL "
    "AND to_version_id = candidate_version_id AND from_version_id IS NULL "
    "AND expected_revision = 0 AND source_release_event_id IS NULL "
    "AND campaign_id IS NULL AND review_id IS NULL "
    "AND holdout_use_id IS NULL AND reconciliation_id IS NULL)"
)


def _replace_checks(type_expr: str, shape_expr: str) -> None:
    op.drop_constraint(TYPE_CHECK, TABLE, type_="check")
    op.create_check_constraint(TYPE_CHECK, TABLE, type_expr)
    op.drop_constraint(SHAPE_CHECK, TABLE, type_="check")
    op.create_check_constraint(SHAPE_CHECK, TABLE, shape_expr)


def upgrade():
    _replace_checks(TYPE_046, SHAPE_046)


def downgrade():
    connection = op.get_bind()
    connection.execute(sa.text("LOCK TABLE strategy_release_events IN SHARE MODE"))
    has_bootstrap = connection.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM strategy_release_events WHERE event_type = 'bootstrap' LIMIT 1)")
    ).scalar()
    if has_bootstrap:
        raise RuntimeError("migration 046 downgrade refused: bootstrap release history exists")
    _replace_checks(TYPE_044, SHAPE_044)
