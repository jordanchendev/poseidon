"""Create immutable outcome and experiment campaign contracts.

Revision ID: 043
Revises: 042
Create Date: 2026-10-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None

PHASE99_TABLES = (
    "outcome_label_contracts",
    "research_assessments",
    "fill_cost_revisions",
    "fill_cost_components",
    "economic_reconciliations",
    "outcome_records",
    "experiment_campaigns",
    "campaign_events",
    "holdout_uses",
    "campaign_reviews",
)

EXPERIMENT_COLUMNS = (
    "campaign_id",
    "original_trial_id",
    "trial_role",
    "strategy_version_id",
    "ablation_arm",
    "paired_sample_key_sha256",
    "input_sha256",
    "result_sha256",
    "started_at",
    "completed_at",
    "terminal_state",
    "terminal_reason_json",
)


def _uuid_primary_key():
    return sa.Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )


def upgrade():
    op.create_table(
        "outcome_label_contracts",
        _uuid_primary_key(),
        sa.Column("version", sa.String(128), nullable=False),
        sa.Column("contract_json", JSONB, nullable=False),
        sa.Column("contract_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("contract_sha256", name="uq_outcome_label_contracts_sha256"),
        sa.CheckConstraint("btrim(version) <> ''", name="ck_outcome_label_contracts_version"),
        sa.CheckConstraint(
            "char_length(contract_sha256) = 64",
            name="ck_outcome_label_contracts_sha256",
        ),
    )

    op.create_table(
        "research_assessments",
        _uuid_primary_key(),
        sa.Column(
            "evaluation_snapshot_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluation_snapshots.id"),
            nullable=False,
        ),
        sa.Column(
            "research_revision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("research_revisions.id"),
            nullable=False,
        ),
        sa.Column(
            "manifest_id",
            UUID(as_uuid=True),
            sa.ForeignKey("data_manifests.id"),
            nullable=False,
        ),
        sa.Column("assessment_type", sa.String(24), nullable=False),
        sa.Column("assessment_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("citation_ids_json", JSONB, nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("assessment_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "evaluation_snapshot_id",
            "input_sha256",
            name="uq_research_assessments_replay",
        ),
        sa.CheckConstraint(
            "assessment_type IN ('expiry', 'invalidation')",
            name="ck_research_assessments_type",
        ),
        sa.CheckConstraint(
            "status IN ('confirmed', 'not_confirmed', 'unavailable')",
            name="ck_research_assessments_status",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(citation_ids_json) = 'array' AND jsonb_array_length(citation_ids_json) > 0",
            name="ck_research_assessments_citations",
        ),
        sa.CheckConstraint(
            "char_length(input_sha256) = 64 AND char_length(content_sha256) = 64",
            name="ck_research_assessments_hashes",
        ),
    )

    op.create_table(
        "fill_cost_revisions",
        _uuid_primary_key(),
        sa.Column(
            "order_fill_id",
            UUID(as_uuid=True),
            sa.ForeignKey("order_fills.id"),
            nullable=True,
        ),
        sa.Column(
            "paper_broker_fill_id",
            UUID(as_uuid=True),
            sa.ForeignKey("paper_broker_fills.id"),
            nullable=True,
        ),
        sa.Column("fill_key_sha256", sa.String(64), nullable=False),
        sa.Column("reporting_currency", sa.String(16), nullable=False),
        sa.Column("cost_model_version", sa.String(128), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("revision_no", sa.Integer, nullable=False),
        sa.Column(
            "previous_fill_cost_revision_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("previous_revision_no", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "fill_key_sha256",
            "input_sha256",
            name="uq_fill_cost_revisions_replay",
        ),
        sa.UniqueConstraint(
            "fill_key_sha256",
            "revision_no",
            name="uq_fill_cost_revisions_revision",
        ),
        sa.UniqueConstraint(
            "id",
            "fill_key_sha256",
            "revision_no",
            name="uq_fill_cost_revisions_identity_key_revision",
        ),
        sa.UniqueConstraint(
            "previous_fill_cost_revision_id",
            name="uq_fill_cost_revisions_previous",
        ),
        sa.ForeignKeyConstraint(
            ["previous_fill_cost_revision_id", "fill_key_sha256", "previous_revision_no"],
            [
                "fill_cost_revisions.id",
                "fill_cost_revisions.fill_key_sha256",
                "fill_cost_revisions.revision_no",
            ],
            name="fk_fill_cost_revisions_previous_identity_key_revision",
        ),
        sa.CheckConstraint(
            "(order_fill_id IS NOT NULL AND paper_broker_fill_id IS NULL) OR "
            "(order_fill_id IS NULL AND paper_broker_fill_id IS NOT NULL)",
            name="ck_fill_cost_revisions_fill_xor",
        ),
        sa.CheckConstraint(
            "(revision_no = 1 AND previous_fill_cost_revision_id IS NULL AND previous_revision_no IS NULL) "
            "OR (revision_no > 1 AND previous_fill_cost_revision_id IS NOT NULL "
            "AND previous_revision_no = revision_no - 1)",
            name="ck_fill_cost_revisions_revision_chain",
        ),
        sa.CheckConstraint(
            "char_length(fill_key_sha256) = 64 AND char_length(input_sha256) = 64 AND char_length(content_sha256) = 64",
            name="ck_fill_cost_revisions_hashes",
        ),
    )
    op.create_index(
        "uq_fill_cost_revisions_order_fill_revision",
        "fill_cost_revisions",
        ["order_fill_id", "revision_no"],
        unique=True,
        postgresql_where=sa.text("order_fill_id IS NOT NULL"),
    )
    op.create_index(
        "uq_fill_cost_revisions_paper_fill_revision",
        "fill_cost_revisions",
        ["paper_broker_fill_id", "revision_no"],
        unique=True,
        postgresql_where=sa.text("paper_broker_fill_id IS NOT NULL"),
    )

    op.create_table(
        "fill_cost_components",
        _uuid_primary_key(),
        sa.Column(
            "fill_cost_revision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("fill_cost_revisions.id"),
            nullable=False,
        ),
        sa.Column("component_type", sa.String(32), nullable=False),
        sa.Column("native_amount", sa.Numeric(38, 18), nullable=True),
        sa.Column("native_currency", sa.String(16), nullable=False),
        sa.Column("reporting_amount", sa.Numeric(38, 18), nullable=True),
        sa.Column("reporting_currency", sa.String(16), nullable=False),
        sa.Column("classification", sa.String(24), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("cost_model_version", sa.String(128), nullable=False),
        sa.Column("fx_source", sa.String(128), nullable=True),
        sa.Column("fx_rate", sa.Numeric(38, 18), nullable=True),
        sa.Column("fx_as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "fill_cost_revision_id",
            "component_type",
            "source",
            name="uq_fill_cost_components_identity",
        ),
        sa.CheckConstraint(
            "component_type IN ('commission', 'tax', 'funding', 'borrow', 'fx', 'other')",
            name="ck_fill_cost_components_type",
        ),
        sa.CheckConstraint(
            "classification IN ('actual', 'estimated', 'not_applicable', 'unavailable')",
            name="ck_fill_cost_components_classification",
        ),
        sa.CheckConstraint(
            "classification NOT IN ('actual', 'estimated') OR "
            "(native_amount IS NOT NULL AND reporting_amount IS NOT NULL)",
            name="ck_fill_cost_components_amounts",
        ),
        sa.CheckConstraint(
            "classification <> 'unavailable' OR (reason IS NOT NULL AND btrim(reason) <> '')",
            name="ck_fill_cost_components_unavailable_reason",
        ),
        sa.CheckConstraint(
            "classification NOT IN ('actual', 'estimated') OR native_currency = reporting_currency OR "
            "(fx_source IS NOT NULL AND btrim(fx_source) <> '' AND fx_rate IS NOT NULL "
            "AND fx_rate > 0 AND fx_as_of IS NOT NULL)",
            name="ck_fill_cost_components_cross_currency_fx",
        ),
    )

    op.create_table(
        "economic_reconciliations",
        _uuid_primary_key(),
        sa.Column(
            "account_reconciliation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("account_reconciliations.id"),
            nullable=False,
        ),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("reporting_currency", sa.String(16), nullable=False),
        sa.Column("cost_revision_ids_json", JSONB, nullable=False),
        sa.Column("fx_facts_json", JSONB, nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reason_codes_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "account_reconciliation_id",
            "input_sha256",
            name="uq_economic_reconciliations_replay",
        ),
        sa.CheckConstraint(
            "status IN ('matched', 'provisional', 'unavailable')",
            name="ck_economic_reconciliations_status",
        ),
        sa.CheckConstraint(
            "char_length(input_sha256) = 64 AND char_length(content_sha256) = 64",
            name="ck_economic_reconciliations_hashes",
        ),
    )

    op.create_table(
        "outcome_records",
        _uuid_primary_key(),
        sa.Column(
            "evaluation_snapshot_id",
            UUID(as_uuid=True),
            sa.ForeignKey("evaluation_snapshots.id"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column(
            "label_contract_id",
            UUID(as_uuid=True),
            sa.ForeignKey("outcome_label_contracts.id"),
            nullable=False,
        ),
        sa.Column("horizon_key", sa.String(128), nullable=False),
        sa.Column(
            "manifest_id",
            UUID(as_uuid=True),
            sa.ForeignKey("data_manifests.id"),
            nullable=False,
        ),
        sa.Column(
            "decision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("decision_records.id"),
            nullable=True,
        ),
        sa.Column("logical_key_sha256", sa.String(64), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("revision_no", sa.Integer, nullable=False),
        sa.Column(
            "previous_outcome_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("previous_revision_no", sa.Integer, nullable=True),
        sa.Column("maturity_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=False),
        sa.Column("metrics_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "logical_key_sha256",
            "input_sha256",
            name="uq_outcome_records_replay",
        ),
        sa.UniqueConstraint(
            "logical_key_sha256",
            "revision_no",
            name="uq_outcome_records_revision",
        ),
        sa.UniqueConstraint(
            "id",
            "logical_key_sha256",
            "revision_no",
            name="uq_outcome_records_identity_key_revision",
        ),
        sa.UniqueConstraint("previous_outcome_id", name="uq_outcome_records_previous"),
        sa.ForeignKeyConstraint(
            ["previous_outcome_id", "logical_key_sha256", "previous_revision_no"],
            ["outcome_records.id", "outcome_records.logical_key_sha256", "outcome_records.revision_no"],
            name="fk_outcome_records_previous_identity_key_revision",
        ),
        sa.CheckConstraint("kind IN ('signal', 'trade', 'research')", name="ck_outcome_records_kind"),
        sa.CheckConstraint(
            "(kind = 'trade' AND decision_id IS NOT NULL) OR (kind IN ('signal', 'research') AND decision_id IS NULL)",
            name="ck_outcome_records_kind_decision",
        ),
        sa.CheckConstraint(
            "status IN ('available', 'provisional', 'unavailable')",
            name="ck_outcome_records_status",
        ),
        sa.CheckConstraint(
            "(revision_no = 1 AND previous_outcome_id IS NULL AND previous_revision_no IS NULL) "
            "OR (revision_no > 1 AND previous_outcome_id IS NOT NULL "
            "AND previous_revision_no = revision_no - 1)",
            name="ck_outcome_records_revision_chain",
        ),
        sa.CheckConstraint(
            "char_length(logical_key_sha256) = 64 AND char_length(input_sha256) = 64 "
            "AND char_length(content_sha256) = 64",
            name="ck_outcome_records_hashes",
        ),
    )
    op.create_index(
        "ix_outcome_records_evaluation_kind",
        "outcome_records",
        ["evaluation_snapshot_id", "kind"],
    )

    op.create_table(
        "experiment_campaigns",
        _uuid_primary_key(),
        sa.Column(
            "incumbent_strategy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("strategy_versions.id"),
            nullable=False,
        ),
        sa.Column(
            "candidate_strategy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("strategy_versions.id"),
            nullable=False,
        ),
        sa.Column("incumbent_content_sha256", sa.String(64), nullable=False),
        sa.Column("candidate_content_sha256", sa.String(64), nullable=False),
        sa.Column("declared_difference_json", JSONB, nullable=False),
        sa.Column("hypothesis", sa.String(1024), nullable=False),
        sa.Column("contract_json", JSONB, nullable=False),
        sa.Column("contract_sha256", sa.String(64), nullable=False),
        sa.Column("created_by", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("contract_sha256", name="uq_experiment_campaigns_contract_sha256"),
        sa.CheckConstraint(
            "char_length(incumbent_content_sha256) = 64 "
            "AND char_length(candidate_content_sha256) = 64 "
            "AND char_length(contract_sha256) = 64",
            name="ck_experiment_campaigns_hashes",
        ),
        sa.CheckConstraint("btrim(hypothesis) <> ''", name="ck_experiment_campaigns_hypothesis"),
        sa.CheckConstraint("btrim(created_by) <> ''", name="ck_experiment_campaigns_created_by"),
    )

    op.create_table(
        "campaign_events",
        _uuid_primary_key(),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiment_campaigns.id"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("idempotency_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "campaign_id",
            "idempotency_sha256",
            name="uq_campaign_events_idempotency",
        ),
        sa.CheckConstraint("btrim(event_type) <> ''", name="ck_campaign_events_type"),
        sa.CheckConstraint(
            "char_length(idempotency_sha256) = 64",
            name="ck_campaign_events_idempotency_sha256",
        ),
    )

    op.create_table(
        "holdout_uses",
        _uuid_primary_key(),
        sa.Column("holdout_identity_sha256", sa.String(64), nullable=False),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiment_campaigns.id"),
            nullable=False,
        ),
        sa.Column("campaign_contract_sha256", sa.String(64), nullable=False),
        sa.Column("audit_json", JSONB, nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("holdout_identity_sha256", name="uq_holdout_uses_identity"),
        sa.CheckConstraint(
            "char_length(holdout_identity_sha256) = 64 AND char_length(campaign_contract_sha256) = 64",
            name="ck_holdout_uses_hashes",
        ),
    )

    op.create_table(
        "campaign_reviews",
        _uuid_primary_key(),
        sa.Column(
            "campaign_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiment_campaigns.id"),
            nullable=False,
        ),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("result_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("result_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "campaign_id",
            "input_sha256",
            name="uq_campaign_reviews_replay",
        ),
        sa.CheckConstraint(
            "status IN ('passed', 'failed', 'inconclusive', 'unavailable')",
            name="ck_campaign_reviews_status",
        ),
        sa.CheckConstraint(
            "char_length(input_sha256) = 64 AND char_length(result_sha256) = 64",
            name="ck_campaign_reviews_hashes",
        ),
    )

    experiment_columns = (
        sa.Column("campaign_id", UUID(as_uuid=True), nullable=True),
        sa.Column("original_trial_id", sa.String(128), nullable=True),
        sa.Column("trial_role", sa.String(32), nullable=True),
        sa.Column("strategy_version_id", UUID(as_uuid=True), nullable=True),
        sa.Column("ablation_arm", sa.String(32), nullable=True),
        sa.Column("paired_sample_key_sha256", sa.String(64), nullable=True),
        sa.Column("input_sha256", sa.String(64), nullable=True),
        sa.Column("result_sha256", sa.String(64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_state", sa.String(40), nullable=True),
        sa.Column("terminal_reason_json", JSONB, nullable=True),
    )
    for column in experiment_columns:
        op.add_column("experiments", column)
    op.create_foreign_key(
        "fk_experiments_campaign_id",
        "experiments",
        "experiment_campaigns",
        ["campaign_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_experiments_strategy_version_id",
        "experiments",
        "strategy_versions",
        ["strategy_version_id"],
        ["id"],
    )
    op.create_check_constraint(
        "ck_experiments_trial_role",
        "experiments",
        "trial_role IS NULL OR trial_role IN ('search', 'paired_evaluation')",
    )
    op.create_check_constraint(
        "ck_experiments_ablation_arm",
        "experiments",
        "ablation_arm IS NULL OR ablation_arm IN ('fundamental_only', 'technical_only', 'combined')",
    )
    op.create_check_constraint(
        "ck_experiments_terminal_state",
        "experiments",
        "terminal_state IS NULL OR terminal_state IN "
        "('succeeded', 'optimizer_failed', 'constraint_rejected', 'insufficient_data', "
        "'statistically_inconclusive', 'capability_unavailable')",
    )
    op.create_check_constraint(
        "ck_experiments_phase99_hashes",
        "experiments",
        "(paired_sample_key_sha256 IS NULL OR char_length(paired_sample_key_sha256) = 64) "
        "AND (input_sha256 IS NULL OR char_length(input_sha256) = 64) "
        "AND (result_sha256 IS NULL OR char_length(result_sha256) = 64)",
    )
    op.create_check_constraint(
        "ck_experiments_campaign_link_complete",
        "experiments",
        "campaign_id IS NULL OR (campaign_id IS NOT NULL AND original_trial_id IS NOT NULL AND trial_role IS NOT NULL "
        "AND strategy_version_id IS NOT NULL AND ablation_arm IS NOT NULL "
        "AND paired_sample_key_sha256 IS NOT NULL AND input_sha256 IS NOT NULL "
        "AND result_sha256 IS NOT NULL AND started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND terminal_state IS NOT NULL AND terminal_state IN "
        "('succeeded', 'optimizer_failed', 'constraint_rejected', 'insufficient_data', "
        "'statistically_inconclusive', 'capability_unavailable'))",
    )
    op.create_index(
        "uq_experiments_campaign_paired_cell",
        "experiments",
        [
            "campaign_id",
            "original_trial_id",
            "strategy_version_id",
            "ablation_arm",
            "paired_sample_key_sha256",
        ],
        unique=True,
        postgresql_where=sa.text("campaign_id IS NOT NULL"),
    )

    op.execute(
        sa.text(
            """
            CREATE FUNCTION phase99_reject_fact_mutation() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION USING ERRCODE = '23514',
                    MESSAGE = format('%s is append-only', TG_TABLE_NAME);
            END;
            $$
            """
        )
    )
    for table_name in PHASE99_TABLES:
        op.execute(
            sa.text(
                f"""
                CREATE TRIGGER trg_{table_name}_append_only
                BEFORE UPDATE OR DELETE ON {table_name}
                FOR EACH ROW EXECUTE FUNCTION phase99_reject_fact_mutation()
                """
            )
        )

    op.execute(
        sa.text(
            """
            CREATE FUNCTION phase99_guard_linked_experiment() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                IF TG_OP = 'UPDATE' AND OLD.campaign_id IS NULL AND NEW.campaign_id IS NULL THEN
                    RETURN NEW;
                END IF;
                IF TG_OP = 'DELETE' AND OLD.campaign_id IS NULL THEN
                    RETURN OLD;
                END IF;
                RAISE EXCEPTION USING ERRCODE = '23514',
                    MESSAGE = 'campaign-linked experiments are append-only';
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_experiments_phase99_append_only
            BEFORE UPDATE OR DELETE ON experiments
            FOR EACH ROW EXECUTE FUNCTION phase99_guard_linked_experiment()
            """
        )
    )


def downgrade():
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            LOCK TABLE
                outcome_label_contracts,
                research_assessments,
                fill_cost_revisions,
                fill_cost_components,
                economic_reconciliations,
                outcome_records,
                experiment_campaigns,
                campaign_events,
                holdout_uses,
                campaign_reviews,
                experiments
            IN SHARE MODE
            """
        )
    )
    has_history = connection.execute(
        sa.text(
            """
            SELECT
                EXISTS (SELECT 1 FROM outcome_label_contracts LIMIT 1)
                OR EXISTS (SELECT 1 FROM research_assessments LIMIT 1)
                OR EXISTS (SELECT 1 FROM fill_cost_revisions LIMIT 1)
                OR EXISTS (SELECT 1 FROM fill_cost_components LIMIT 1)
                OR EXISTS (SELECT 1 FROM economic_reconciliations LIMIT 1)
                OR EXISTS (SELECT 1 FROM outcome_records LIMIT 1)
                OR EXISTS (SELECT 1 FROM experiment_campaigns LIMIT 1)
                OR EXISTS (SELECT 1 FROM campaign_events LIMIT 1)
                OR EXISTS (SELECT 1 FROM holdout_uses LIMIT 1)
                OR EXISTS (SELECT 1 FROM campaign_reviews LIMIT 1)
                OR EXISTS (
                    SELECT 1 FROM experiments
                    WHERE experiments.campaign_id IS NOT NULL
                       OR experiments.original_trial_id IS NOT NULL
                       OR experiments.trial_role IS NOT NULL
                       OR experiments.strategy_version_id IS NOT NULL
                       OR experiments.ablation_arm IS NOT NULL
                       OR experiments.paired_sample_key_sha256 IS NOT NULL
                       OR experiments.input_sha256 IS NOT NULL
                       OR experiments.result_sha256 IS NOT NULL
                       OR experiments.started_at IS NOT NULL
                       OR experiments.completed_at IS NOT NULL
                       OR experiments.terminal_state IS NOT NULL
                       OR experiments.terminal_reason_json IS NOT NULL
                    LIMIT 1
                )
            """
        )
    ).scalar()
    if has_history:
        raise RuntimeError("migration 043 downgrade refused: outcome or experiment history exists")

    op.execute(sa.text("DROP TRIGGER trg_experiments_phase99_append_only ON experiments"))
    op.execute(sa.text("DROP FUNCTION phase99_guard_linked_experiment()"))
    for table_name in reversed(PHASE99_TABLES):
        op.execute(sa.text(f"DROP TRIGGER trg_{table_name}_append_only ON {table_name}"))
    op.execute(sa.text("DROP FUNCTION phase99_reject_fact_mutation()"))

    op.drop_index("uq_experiments_campaign_paired_cell", table_name="experiments")
    for constraint_name, constraint_type in (
        ("ck_experiments_campaign_link_complete", "check"),
        ("ck_experiments_phase99_hashes", "check"),
        ("ck_experiments_terminal_state", "check"),
        ("ck_experiments_ablation_arm", "check"),
        ("ck_experiments_trial_role", "check"),
        ("fk_experiments_strategy_version_id", "foreignkey"),
        ("fk_experiments_campaign_id", "foreignkey"),
    ):
        op.drop_constraint(constraint_name, "experiments", type_=constraint_type)
    for column_name in reversed(EXPERIMENT_COLUMNS):
        op.drop_column("experiments", column_name)

    for table_name in reversed(PHASE99_TABLES):
        op.drop_table(table_name)
