"""Spec v3 runs: any pipeline market, the main instrument (or PORTFOLIO) as ``symbol``.

Revision ID: p003
Revises: p002
Create Date: 2026-10-24

Strategy Pipeline v2 step 4 (design 2026-10-24 §8): ``ck_pipeline_runs_instrument`` pinned every
run to ``tw_futures`` / ``TX``. A CONFIG_VERSION 6 run stores its main instrument (TMF, MTX, TX,
later an ETF) and its market; a multi-instrument run stores market ``multi`` and symbol
``PORTFOLIO`` (the full instrument list is in ``config_json``). No new table. Run with
``alembic -n pipeline upgrade head`` against the isolated ``poseidon_pipeline`` database.
"""

from alembic import op

revision = "p003"
down_revision = "p002"
branch_labels = None
depends_on = None

OLD = "market = 'tw_futures' AND symbol = 'TX'"
NEW = (
    "market IN ('tw_futures', 'tw_stock', 'multi') AND btrim(symbol) <> '' "
    "AND (market = 'multi') = (symbol = 'PORTFOLIO')"
)


def upgrade():
    op.drop_constraint("ck_pipeline_runs_instrument", "pipeline_runs", type_="check")
    op.create_check_constraint("ck_pipeline_runs_instrument", "pipeline_runs", NEW)


def downgrade():
    # Fails while any run is not tw_futures/TX (a spec v3 run); runs are never deleted, so a
    # database that has created v3 runs cannot go back below p003.
    op.drop_constraint("ck_pipeline_runs_instrument", "pipeline_runs", type_="check")
    op.create_check_constraint("ck_pipeline_runs_instrument", "pipeline_runs", OLD)
