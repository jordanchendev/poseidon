"""Strategy-pipeline foundation: runs, stage attempts, trial ledger, events.

Revision ID: p001
Revises: (none -- first revision of the separate ``pipeline`` stream)
Create Date: 2026-10-22

Run with ``alembic -n pipeline upgrade head`` against the isolated
``poseidon_pipeline`` database. The version table is
``alembic_version_pipeline``; the main 0xx stream is untouched. Step 9 of the
strategy-pipeline plan folds these objects into the main stream.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "p001"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB()
TS = sa.DateTime(timezone=True)

RUN_FROZEN_COLUMNS = (
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
STAGE_RUN_IDENTITY_COLUMNS = ("id", "run_id", "stage", "seq", "attempt", "trigger", "created_at")


def _uuid_pk():
    return sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()"))


def _created_at():
    return sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()"))


def _distinct(columns) -> str:
    return " OR ".join(f"NEW.{column} IS DISTINCT FROM OLD.{column}" for column in columns)


def upgrade():
    op.create_table(
        "pipeline_runs",
        _uuid_pk(),
        sa.Column("run_no", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("family_id", UUID, sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("idea_text", sa.Text(), nullable=False),
        sa.Column("market", sa.String(16), nullable=False, server_default="tw_futures"),
        sa.Column("symbol", sa.String(16), nullable=False, server_default="TX"),
        sa.Column("input_spec_json", JSONB, nullable=True),
        sa.Column("config_json", JSONB, nullable=False),
        sa.Column("config_sha256", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(256), nullable=False),
        sa.Column("create_idempotency_key", sa.String(256), nullable=True),
        sa.Column("create_request_sha256", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("current_stage", sa.String(32), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("abort_requested_at", TS, nullable=True),
        _created_at(),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", TS, nullable=True),
        sa.UniqueConstraint("run_no", name="uq_pipeline_runs_run_no"),
        sa.UniqueConstraint("created_by", "create_idempotency_key", name="uq_pipeline_runs_create_idempotency"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'waiting', 'completed', 'failed', 'unsupported', 'error', 'aborted')",
            name="ck_pipeline_runs_status",
        ),
        sa.CheckConstraint("source IN ('telegram', 'scout', 'api')", name="ck_pipeline_runs_source"),
        sa.CheckConstraint("market = 'tw_futures' AND symbol = 'TX'", name="ck_pipeline_runs_instrument"),
        sa.CheckConstraint("char_length(config_sha256) = 64", name="ck_pipeline_runs_config_sha256"),
        sa.CheckConstraint(
            "(create_idempotency_key IS NULL AND create_request_sha256 IS NULL) OR "
            "(create_idempotency_key IS NOT NULL AND create_request_sha256 IS NOT NULL "
            "AND char_length(create_request_sha256) = 64)",
            name="ck_pipeline_runs_idempotency_pair",
        ),
        sa.CheckConstraint("btrim(idea_text) <> ''", name="ck_pipeline_runs_idea_text"),
        sa.CheckConstraint("btrim(created_by) <> ''", name="ck_pipeline_runs_created_by"),
        sa.CheckConstraint("revision >= 1", name="ck_pipeline_runs_revision"),
        sa.CheckConstraint(
            "(status IN ('completed', 'failed', 'unsupported', 'error', 'aborted')) = (finished_at IS NOT NULL)",
            name="ck_pipeline_runs_finished",
        ),
    )
    op.create_index("ix_pipeline_runs_status_updated", "pipeline_runs", ["status", "updated_at"])
    op.create_index("ix_pipeline_runs_family", "pipeline_runs", ["family_id"])

    op.create_table(
        "stage_runs",
        _uuid_pk(),
        sa.Column("run_id", UUID, sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("trigger", sa.String(8), nullable=False),
        sa.Column("code_version", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("verdict", sa.String(16), nullable=True),
        sa.Column("input_sha256", sa.String(64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("artifact_json", JSONB, nullable=True),
        sa.Column("artifact_sha256", sa.String(64), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("error_json", JSONB, nullable=True),
        sa.Column("lease_owner", UUID, nullable=True),
        sa.Column("lease_expires_at", TS, nullable=True),
        sa.Column("heartbeat_at", TS, nullable=True),
        sa.Column("wait_until", TS, nullable=True),
        sa.Column("started_at", TS, nullable=True),
        sa.Column("finished_at", TS, nullable=True),
        _created_at(),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("run_id", "stage", "attempt", name="uq_stage_runs_attempt"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'waiting', 'completed', 'aborted')", name="ck_stage_runs_status"
        ),
        sa.CheckConstraint(
            "verdict IS NULL OR verdict IN ('pass', 'fail', 'unsupported', 'error')", name="ck_stage_runs_verdict"
        ),
        sa.CheckConstraint("(status = 'completed') = (verdict IS NOT NULL)", name="ck_stage_runs_verdict_completed"),
        sa.CheckConstraint("trigger IN ('advance', 'retry', 'rerun')", name="ck_stage_runs_trigger"),
        sa.CheckConstraint("attempt >= 1 AND seq >= 0", name="ck_stage_runs_ordinals"),
        sa.CheckConstraint(
            "(input_sha256 IS NULL OR char_length(input_sha256) = 64)", name="ck_stage_runs_input_sha256"
        ),
        sa.CheckConstraint(
            "(artifact_json IS NULL AND artifact_sha256 IS NULL) OR "
            "(artifact_json IS NOT NULL AND artifact_sha256 IS NOT NULL AND char_length(artifact_sha256) = 64)",
            name="ck_stage_runs_artifact_pair",
        ),
        sa.CheckConstraint(
            "artifact_json IS NULL OR octet_length(artifact_json::text) <= 262144",
            name="ck_stage_runs_artifact_size",
        ),
        sa.CheckConstraint(
            "status <> 'running' OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND input_sha256 IS NOT NULL AND code_version IS NOT NULL)",
            name="ck_stage_runs_running_lease",
        ),
        sa.CheckConstraint("status <> 'waiting' OR wait_until IS NOT NULL", name="ck_stage_runs_waiting"),
        sa.CheckConstraint(
            "(status IN ('completed', 'aborted')) = (finished_at IS NOT NULL)", name="ck_stage_runs_finished"
        ),
    )
    op.create_index(
        "uq_stage_runs_open_attempt",
        "stage_runs",
        ["run_id", "stage"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running', 'waiting')"),
    )
    op.create_index(
        "uq_stage_runs_one_running",
        "stage_runs",
        ["run_id"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index("ix_stage_runs_status_lease", "stage_runs", ["status", "lease_expires_at"])

    op.create_table(
        "pipeline_trials",
        _uuid_pk(),
        sa.Column("family_id", UUID, sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("run_id", UUID, sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("stage_run_id", UUID, sa.ForeignKey("stage_runs.id"), nullable=False),
        sa.Column("trial_no", sa.Integer(), nullable=False),
        sa.Column("params_json", JSONB, nullable=False),
        sa.Column("params_sha256", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("n_obs", sa.Integer(), nullable=True),
        sa.Column("skew", sa.Float(), nullable=True),
        sa.Column("kurtosis", sa.Float(), nullable=True),
        sa.Column("window_sha256", sa.String(64), nullable=False),
        _created_at(),
        sa.UniqueConstraint("stage_run_id", "trial_no", name="uq_pipeline_trials_number"),
        sa.CheckConstraint("state IN ('complete', 'pruned', 'fail')", name="ck_pipeline_trials_state"),
        sa.CheckConstraint(
            "char_length(params_sha256) = 64 AND char_length(window_sha256) = 64",
            name="ck_pipeline_trials_hashes",
        ),
        sa.CheckConstraint("trial_no >= 0 AND (n_obs IS NULL OR n_obs >= 0)", name="ck_pipeline_trials_ordinals"),
    )
    op.create_index("ix_pipeline_trials_family", "pipeline_trials", ["family_id"])

    op.create_table(
        "pipeline_events",
        _uuid_pk(),
        sa.Column("run_id", UUID, sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column("stage_run_id", UUID, sa.ForeignKey("stage_runs.id"), nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("actor_id", sa.String(256), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=True),
        sa.Column("request_sha256", sa.String(64), nullable=True),
        sa.Column("payload_json", JSONB, nullable=False),
        _created_at(),
        sa.UniqueConstraint("run_id", "idempotency_key", name="uq_pipeline_events_idempotency_key"),
        sa.CheckConstraint(
            "(idempotency_key IS NULL AND request_sha256 IS NULL) OR "
            "(idempotency_key IS NOT NULL AND request_sha256 IS NOT NULL)",
            name="ck_pipeline_events_idempotency_pair",
        ),
        sa.CheckConstraint(
            "(request_sha256 IS NULL OR char_length(request_sha256) = 64)",
            name="ck_pipeline_events_request_sha256",
        ),
    )
    op.create_index("ix_pipeline_events_run_created", "pipeline_events", ["run_id", "created_at"])

    # --- Database-layer immutability (mirrors the ORM listeners) ---
    op.execute(
        sa.text(
            """
            CREATE FUNCTION pipeline_reject_fact_mutation() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION USING ERRCODE = '23514',
                    MESSAGE = format('%s is append-only', TG_TABLE_NAME);
            END;
            $$
            """
        )
    )
    for table_name in ("pipeline_trials", "pipeline_events"):
        op.execute(
            sa.text(
                f"""
                CREATE TRIGGER trg_{table_name}_append_only
                BEFORE UPDATE OR DELETE ON {table_name}
                FOR EACH ROW EXECUTE FUNCTION pipeline_reject_fact_mutation()
                """
            )
        )

    op.execute(
        sa.text(
            f"""
            CREATE FUNCTION pipeline_guard_stage_run() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION USING ERRCODE = '23514', MESSAGE = 'stage_runs cannot be deleted';
                END IF;
                IF OLD.status IN ('completed', 'aborted') THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = format('stage_runs row is %s and immutable', OLD.status);
                END IF;
                IF {_distinct(STAGE_RUN_IDENTITY_COLUMNS)} THEN
                    RAISE EXCEPTION USING ERRCODE = '23514', MESSAGE = 'stage_runs identity is immutable';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_stage_runs_guard
            BEFORE UPDATE OR DELETE ON stage_runs
            FOR EACH ROW EXECUTE FUNCTION pipeline_guard_stage_run()
            """
        )
    )

    op.execute(
        sa.text(
            f"""
            CREATE FUNCTION pipeline_guard_run() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION USING ERRCODE = '23514', MESSAGE = 'pipeline_runs cannot be deleted';
                END IF;
                IF {_distinct(RUN_FROZEN_COLUMNS)} THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'pipeline_runs identity and config are frozen';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_pipeline_runs_guard
            BEFORE UPDATE OR DELETE ON pipeline_runs
            FOR EACH ROW EXECUTE FUNCTION pipeline_guard_run()
            """
        )
    )


def downgrade():
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_pipeline_runs_guard ON pipeline_runs"))
    op.execute(sa.text("DROP TRIGGER IF EXISTS trg_stage_runs_guard ON stage_runs"))
    for table_name in ("pipeline_trials", "pipeline_events"):
        op.execute(sa.text(f"DROP TRIGGER IF EXISTS trg_{table_name}_append_only ON {table_name}"))
    op.drop_table("pipeline_events")
    op.drop_table("pipeline_trials")
    op.drop_table("stage_runs")
    op.drop_table("pipeline_runs")
    for function_name in ("pipeline_guard_run", "pipeline_guard_stage_run", "pipeline_reject_fact_mutation"):
        op.execute(sa.text(f"DROP FUNCTION IF EXISTS {function_name}()"))
