"""Create guarded strategy release authority.

Revision ID: 044
Revises: 043
Create Date: 2026-10-13

The transaction-local guard prevents unsupported direct writes, but it is not
role separation: a compromised same-role credential could forge both an event
and the guard. Database-role hardening remains Phase 101 work.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade():
    op.create_unique_constraint(
        "uq_strategy_versions_strategy_id_id",
        "strategy_versions",
        ["strategy_id", "id"],
    )
    op.add_column("strategies", sa.Column("active_version_id", UUID(as_uuid=True), nullable=True))
    op.add_column(
        "strategies",
        sa.Column("release_revision", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.create_foreign_key(
        "fk_strategies_active_version",
        "strategies",
        "strategy_versions",
        ["id", "active_version_id"],
        ["strategy_id", "id"],
    )

    op.add_column("campaign_reviews", sa.Column("holdout_use_id", UUID(as_uuid=True), nullable=True))
    op.add_column("campaign_reviews", sa.Column("input_identity_json", JSONB, nullable=True))
    op.create_unique_constraint(
        "uq_holdout_uses_campaign_id_id",
        "holdout_uses",
        ["campaign_id", "id"],
    )
    op.create_unique_constraint(
        "uq_campaign_reviews_campaign_id_id",
        "campaign_reviews",
        ["campaign_id", "id"],
    )
    op.create_foreign_key(
        "fk_campaign_reviews_holdout_use",
        "campaign_reviews",
        "holdout_uses",
        ["campaign_id", "holdout_use_id"],
        ["campaign_id", "id"],
    )
    op.create_check_constraint(
        "ck_campaign_reviews_release_binding",
        "campaign_reviews",
        "(holdout_use_id IS NULL AND input_identity_json IS NULL) OR "
        "(holdout_use_id IS NOT NULL AND input_identity_json IS NOT NULL)",
    )

    op.create_table(
        "strategy_release_events",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "strategy_id",
            UUID(as_uuid=True),
            sa.ForeignKey("strategies.id"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(16), nullable=False),
        sa.Column("candidate_version_id", UUID(as_uuid=True), nullable=True),
        sa.Column("from_version_id", UUID(as_uuid=True), nullable=True),
        sa.Column("to_version_id", UUID(as_uuid=True), nullable=True),
        sa.Column("source_release_event_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiment_campaigns.id"),
            nullable=True,
        ),
        sa.Column("review_id", UUID(as_uuid=True), nullable=True),
        sa.Column("holdout_use_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "reconciliation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("account_reconciliations.id"),
            nullable=True,
        ),
        sa.Column("actor_id", sa.String(256), nullable=False),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=False),
        sa.Column("resulting_revision", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(1024), nullable=False),
        sa.Column("gate_json", JSONB, nullable=False),
        sa.Column("response_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "strategy_id",
            "id",
            name="uq_strategy_release_events_strategy_id_id",
        ),
        sa.UniqueConstraint(
            "strategy_id",
            "idempotency_key",
            name="uq_strategy_release_events_idempotency",
        ),
        sa.UniqueConstraint(
            "strategy_id",
            "resulting_revision",
            name="uq_strategy_release_events_revision",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id", "candidate_version_id"],
            ["strategy_versions.strategy_id", "strategy_versions.id"],
            name="fk_strategy_release_events_candidate_version",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id", "from_version_id"],
            ["strategy_versions.strategy_id", "strategy_versions.id"],
            name="fk_strategy_release_events_from_version",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id", "to_version_id"],
            ["strategy_versions.strategy_id", "strategy_versions.id"],
            name="fk_strategy_release_events_to_version",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id", "source_release_event_id"],
            ["strategy_release_events.strategy_id", "strategy_release_events.id"],
            name="fk_strategy_release_events_source_release",
        ),
        sa.ForeignKeyConstraint(
            ["campaign_id", "review_id"],
            ["campaign_reviews.campaign_id", "campaign_reviews.id"],
            name="fk_strategy_release_events_review",
        ),
        sa.ForeignKeyConstraint(
            ["campaign_id", "holdout_use_id"],
            ["holdout_uses.campaign_id", "holdout_uses.id"],
            name="fk_strategy_release_events_holdout_use",
        ),
        sa.CheckConstraint(
            "event_type IN ('release', 'pause', 'rollback', 'reject')",
            name="ck_strategy_release_events_type",
        ),
        sa.CheckConstraint(
            "resulting_revision = expected_revision + 1 AND expected_revision >= 0",
            name="ck_strategy_release_events_revision",
        ),
        sa.CheckConstraint(
            "char_length(request_sha256) = 64",
            name="ck_strategy_release_events_request_sha256",
        ),
        sa.CheckConstraint(
            "btrim(actor_id) <> '' AND btrim(account_scope) <> '' "
            "AND btrim(idempotency_key) <> '' AND btrim(reason) <> ''",
            name="ck_strategy_release_events_nonempty_text",
        ),
        sa.CheckConstraint(
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
            "AND source_release_event_id IS NULL)",
            name="ck_strategy_release_events_shape",
        ),
    )

    op.execute(
        sa.text(
            """
            CREATE FUNCTION phase100_reject_release_event_mutation() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION USING ERRCODE = '23514',
                    MESSAGE = 'strategy_release_events is append-only';
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_strategy_release_events_append_only
            BEFORE UPDATE OR DELETE ON strategy_release_events
            FOR EACH ROW EXECUTE FUNCTION phase100_reject_release_event_mutation()
            """
        )
    )

    op.execute(
        sa.text(
            """
            CREATE FUNCTION phase100_require_review_binding() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.holdout_use_id IS NULL OR NEW.input_identity_json IS NULL THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'new campaign reviews require exact release binding';
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
            CREATE TRIGGER trg_campaign_reviews_phase100_binding
            BEFORE INSERT ON campaign_reviews
            FOR EACH ROW EXECUTE FUNCTION phase100_require_review_binding()
            """
        )
    )

    op.execute(
        sa.text(
            """
            CREATE FUNCTION phase100_guard_strategy_authority() RETURNS trigger
            LANGUAGE plpgsql AS $$
            DECLARE
                guard_value text;
                release_event record;
            BEGIN
                IF NEW.active_version_id IS NOT DISTINCT FROM OLD.active_version_id
                   AND NEW.release_revision = OLD.release_revision
                   AND NEW.active IS NOT DISTINCT FROM OLD.active THEN
                    RETURN NEW;
                END IF;
                IF NEW.active IS DISTINCT FROM OLD.active THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'legacy strategies.active is inert';
                END IF;
                guard_value := current_setting('poseidon.release_event_id', true);
                IF guard_value IS NULL OR btrim(guard_value) = '' THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'strategy authority update requires a release event guard';
                END IF;
                BEGIN
                    SELECT event.*, event.xmin::text::bigint AS inserted_xid
                    INTO release_event
                    FROM strategy_release_events AS event
                    WHERE event.id = guard_value::uuid AND event.strategy_id = OLD.id;
                EXCEPTION WHEN invalid_text_representation THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'strategy authority guard is not a release event ID';
                END;
                IF NOT FOUND OR release_event.inserted_xid <> pg_current_xact_id()::text::bigint THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'strategy authority guard must name a same-transaction release event';
                END IF;
                IF release_event.expected_revision <> OLD.release_revision
                   OR release_event.resulting_revision <> NEW.release_revision
                   OR release_event.from_version_id IS DISTINCT FROM OLD.active_version_id
                   OR release_event.to_version_id IS DISTINCT FROM NEW.active_version_id THEN
                    RAISE EXCEPTION USING ERRCODE = '23514',
                        MESSAGE = 'release event does not match the strategy authority transition';
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
            CREATE TRIGGER trg_strategies_release_authority
            BEFORE UPDATE ON strategies
            FOR EACH ROW EXECUTE FUNCTION phase100_guard_strategy_authority()
            """
        )
    )


def downgrade():
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            LOCK TABLE strategies, campaign_reviews, strategy_release_events IN SHARE MODE
            """
        )
    )
    has_history = connection.execute(
        sa.text(
            """
            SELECT
                EXISTS (SELECT 1 FROM strategy_release_events LIMIT 1)
                OR EXISTS (
                    SELECT 1 FROM strategies
                    WHERE active_version_id IS NOT NULL OR release_revision <> 0
                    LIMIT 1
                )
                OR EXISTS (
                    SELECT 1 FROM campaign_reviews
                    WHERE holdout_use_id IS NOT NULL OR input_identity_json IS NOT NULL
                    LIMIT 1
                )
            """
        )
    ).scalar()
    if has_history:
        raise RuntimeError("migration 044 downgrade refused: release authority history exists")

    op.execute(sa.text("DROP TRIGGER trg_strategies_release_authority ON strategies"))
    op.execute(sa.text("DROP FUNCTION phase100_guard_strategy_authority()"))
    op.execute(sa.text("DROP TRIGGER trg_campaign_reviews_phase100_binding ON campaign_reviews"))
    op.execute(sa.text("DROP FUNCTION phase100_require_review_binding()"))
    op.execute(sa.text("DROP TRIGGER trg_strategy_release_events_append_only ON strategy_release_events"))
    op.execute(sa.text("DROP FUNCTION phase100_reject_release_event_mutation()"))

    op.drop_table("strategy_release_events")
    op.drop_constraint("ck_campaign_reviews_release_binding", "campaign_reviews", type_="check")
    op.drop_constraint("fk_campaign_reviews_holdout_use", "campaign_reviews", type_="foreignkey")
    op.drop_constraint("uq_campaign_reviews_campaign_id_id", "campaign_reviews", type_="unique")
    op.drop_constraint("uq_holdout_uses_campaign_id_id", "holdout_uses", type_="unique")
    op.drop_column("campaign_reviews", "input_identity_json")
    op.drop_column("campaign_reviews", "holdout_use_id")
    op.drop_constraint("fk_strategies_active_version", "strategies", type_="foreignkey")
    op.drop_column("strategies", "release_revision")
    op.drop_column("strategies", "active_version_id")
    op.drop_constraint("uq_strategy_versions_strategy_id_id", "strategy_versions", type_="unique")
