"""The owner's original text behind a Sisyphus-drafted idea, kept on the run beside the draft.

Revision ID: p004
Revises: p003
Create Date: 2026-10-26

A drafted idea's ``idea_text`` is the spec the owner confirmed (what spec_generate builds from); the
owner's own material travels with it from macminim4 (``strategy-pending`` -> bridge -> API) and is
stored in three nullable columns, all set or all NULL, the sha256 matching the text. They are frozen
like ``idea_text``. Runs without an original are unchanged. Run with ``alembic -n pipeline upgrade
head`` against the isolated ``poseidon_pipeline`` database.
"""

import sqlalchemy as sa

from alembic import op

revision = "p004"
down_revision = "p003"
branch_labels = None
depends_on = None

ORIGINAL_CHECK = (
    "(original_text IS NULL AND original_sha256 IS NULL AND drafted_by IS NULL) OR "
    "(original_text IS NOT NULL AND original_sha256 IS NOT NULL AND drafted_by IS NOT NULL "
    "AND btrim(original_text) <> '' AND drafted_by IN ('sisyphus') "
    "AND original_sha256 = encode(sha256(convert_to(original_text, 'UTF8')), 'hex'))"
)
FROZEN_P001 = (
    "id",
    "run_no",
    "family_id",
    "source",
    "idea_text",
    "market",
    "symbol",
    "input_spec_json",
    "config_json",
    "config_sha256",
    "created_by",
    "create_idempotency_key",
    "create_request_sha256",
    "created_at",
)
FROZEN_P004 = (*FROZEN_P001, "original_text", "original_sha256", "drafted_by")


def _guard(columns) -> str:
    changed = " OR ".join(f"NEW.{column} IS DISTINCT FROM OLD.{column}" for column in columns)
    return f"""
        CREATE OR REPLACE FUNCTION pipeline_guard_run() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION USING ERRCODE = '23514', MESSAGE = 'pipeline_runs cannot be deleted';
            END IF;
            IF {changed} THEN
                RAISE EXCEPTION USING ERRCODE = '23514',
                    MESSAGE = 'pipeline_runs identity and config are frozen';
            END IF;
            RETURN NEW;
        END;
        $$
    """


def upgrade():
    op.add_column("pipeline_runs", sa.Column("original_text", sa.Text(), nullable=True))
    op.add_column("pipeline_runs", sa.Column("original_sha256", sa.String(64), nullable=True))
    op.add_column("pipeline_runs", sa.Column("drafted_by", sa.String(16), nullable=True))
    op.create_check_constraint("ck_pipeline_runs_original", "pipeline_runs", ORIGINAL_CHECK)
    op.execute(sa.text(_guard(FROZEN_P004)))


def downgrade():
    # Refused while a run holds an original: runs are never deleted and the original is part of the record.
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pipeline_runs WHERE original_text IS NOT NULL) THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'ck_pipeline_runs_original: runs hold an original text; restore from a dump instead';
                END IF;
            END;
            $$
            """
        )
    )
    op.execute(sa.text(_guard(FROZEN_P001)))
    op.drop_constraint("ck_pipeline_runs_original", "pipeline_runs", type_="check")
    op.drop_column("pipeline_runs", "drafted_by")
    op.drop_column("pipeline_runs", "original_sha256")
    op.drop_column("pipeline_runs", "original_text")
