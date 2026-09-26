"""Create immutable decision-loop foundation records.

Revision ID: 040
Revises: 039
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "data_manifests",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("capability_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("sources_json", JSONB, nullable=False, server_default="[]"),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("content_sha256", name="uq_data_manifests_content_sha256"),
    )
    op.create_index("ix_data_manifests_market_as_of", "data_manifests", ["market", "as_of"])

    op.create_table(
        "research_revisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("scope_key", sa.String(256), nullable=False),
        sa.Column("manifest_id", UUID(as_uuid=True), sa.ForeignKey("data_manifests.id"), nullable=False),
        sa.Column(
            "previous_revision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("research_revisions.id"),
            nullable=True,
        ),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("policy_version", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("runtime_digest", sa.String(128), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=True),
        sa.Column("status", sa.String(24), nullable=False, server_default="prepared"),
        sa.Column("research_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("request_sha256", name="uq_research_revisions_request_sha256"),
    )
    op.create_index("ix_research_revisions_scope_created", "research_revisions", ["scope_key", "created_at"])

    op.create_table(
        "strategy_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("strategy_id", UUID(as_uuid=True), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("version_no", sa.Integer, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("config_json", JSONB, nullable=False),
        sa.Column("policy_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("artifact_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("strategy_id", "version_no", name="uq_strategy_versions_number"),
        sa.UniqueConstraint("strategy_id", "content_sha256", name="uq_strategy_versions_content"),
    )
    op.create_index("ix_strategy_versions_strategy_status", "strategy_versions", ["strategy_id", "status"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("strategy_version_id", UUID(as_uuid=True), sa.ForeignKey("strategy_versions.id"), nullable=False),
        sa.Column("manifest_id", UUID(as_uuid=True), sa.ForeignKey("data_manifests.id"), nullable=False),
        sa.Column("decision_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("universe_json", JSONB, nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("coverage_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("input_sha256", name="uq_evaluation_runs_input_sha256"),
    )
    op.create_index("ix_evaluation_runs_strategy_created", "evaluation_runs", ["strategy_version_id", "created_at"])

    op.create_table(
        "evaluation_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("evaluation_run_id", UUID(as_uuid=True), sa.ForeignKey("evaluation_runs.id"), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("recommendation_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("technical_json", JSONB, nullable=False, server_default="{}"),
        sa.Column("research_revision_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("reason_codes", JSONB, nullable=False, server_default="[]"),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "evaluation_run_id",
            "symbol",
            "instrument",
            name="uq_evaluation_snapshots_run_symbol_instrument",
        ),
    )
    op.create_index("ix_evaluation_snapshots_run_status", "evaluation_snapshots", ["evaluation_run_id", "status"])


def downgrade():
    # Production rollback disables writers and retains audit data; schema reversal
    # is reserved for disposable databases before audit records are in use.
    op.drop_table("evaluation_snapshots")
    op.drop_table("evaluation_runs")
    op.drop_table("strategy_versions")
    op.drop_table("research_revisions")
    op.drop_table("data_manifests")
