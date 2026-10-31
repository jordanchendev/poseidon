"""Add research correction links and persisted claim precheck results.

Revision ID: 048
Revises: 047
Create Date: 2026-10-21

SEED-002 C3 (research correction policy, local_dev/design/2026-10-21-research-correction-policy.md)
and the C2 precheck persistence TODO. Additive only: three nullable columns on research_revisions,
two CHECK constraints, one partial unique index. No existing row is touched; every existing row has
NULL in the new columns and satisfies both CHECKs, so no pre-check is needed.

- supersedes_revision_id: FK to research_revisions.id. Set on a correction; the superseded row keeps
  status 'completed' and its content_sha256 (evaluation replay re-verifies it).
- correction_reason: closed enum, set when a correction is completed.
- precheck_json: host-owned claim precheck results of the completion; never part of research_json,
  so content_sha256 stays the hash of the human-submitted body.
- uq_research_revisions_supersedes_completed: at most one completed correction per superseded
  revision (linear chain, serialises racing completions).

Uses batch_alter_table so the same upgrade/downgrade runs on SQLite (tests) and PostgreSQL.
``CONSTRAINTS`` and ``INDEX_WHERE`` are pure literals kept equal to the ORM by
tests/unit/test_decision_loop_schema.py.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None

TABLE = "research_revisions"
FK_NAME = "fk_research_revisions_supersedes_revision_id"
INDEX_NAME = "uq_research_revisions_supersedes_completed"
INDEX_WHERE = "supersedes_revision_id IS NOT NULL AND status = 'completed'"
CONSTRAINTS = (
    (
        "ck_research_revisions_correction_reason",
        "correction_reason IS NULL OR correction_reason IN ('source_error', 'excerpt_error', 'numeric_error', "
        "'label_policy', 'stale_evidence', 'other')",
    ),
    (
        "ck_research_revisions_correction_link",
        "(correction_reason IS NULL OR supersedes_revision_id IS NOT NULL) "
        "AND (supersedes_revision_id IS NULL OR status <> 'completed' OR correction_reason IS NOT NULL)",
    ),
)


def upgrade():
    with op.batch_alter_table(TABLE) as batch:
        batch.add_column(sa.Column("supersedes_revision_id", postgresql.UUID(as_uuid=True), nullable=True))
        batch.add_column(sa.Column("correction_reason", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("precheck_json", postgresql.JSONB(), nullable=True))
        batch.create_foreign_key(FK_NAME, TABLE, ["supersedes_revision_id"], ["id"])
        for name, expr in CONSTRAINTS:
            batch.create_check_constraint(name, expr)
    op.create_index(
        INDEX_NAME,
        TABLE,
        ["supersedes_revision_id"],
        unique=True,
        postgresql_where=sa.text(INDEX_WHERE),
        sqlite_where=sa.text(INDEX_WHERE),
    )


def downgrade():
    op.drop_index(INDEX_NAME, table_name=TABLE)
    with op.batch_alter_table(TABLE) as batch:
        for name, _expr in reversed(CONSTRAINTS):
            batch.drop_constraint(name, type_="check")
        batch.drop_constraint(FK_NAME, type_="foreignkey")
        batch.drop_column("precheck_json")
        batch.drop_column("correction_reason")
        batch.drop_column("supersedes_revision_id")
