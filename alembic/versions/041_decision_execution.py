"""Create decision records and append-only events.

Revision ID: 041
Revises: 040
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "decision_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("evaluation_run_id", UUID(as_uuid=True), sa.ForeignKey("evaluation_runs.id"), nullable=False),
        sa.Column(
            "strategy_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("strategy_versions.id"),
            nullable=False,
        ),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("decision_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("revision", sa.Integer, nullable=False, server_default="1"),
        sa.Column("creation_sha256", sa.String(64), nullable=False),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("original_json", JSONB, nullable=False),
        sa.Column("final_json", JSONB, nullable=False),
        sa.Column("portfolio_snapshot_json", JSONB, nullable=False),
        sa.Column("risk_snapshot_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("creation_sha256", name="uq_decision_records_creation_sha256"),
    )
    op.create_index(
        "ix_decision_records_account_status_valid_created",
        "decision_records",
        ["account_scope", "status", "valid_until", "created_at"],
    )

    op.create_table(
        "decision_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("decision_id", UUID(as_uuid=True), sa.ForeignKey("decision_records.id"), nullable=False),
        sa.Column("event_type", sa.String(24), nullable=False),
        sa.Column("actor_id", sa.String(256), nullable=False),
        sa.Column("expected_revision", sa.Integer, nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=True),
        sa.Column("request_sha256", sa.String(64), nullable=True),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "decision_id",
            "event_type",
            "expected_revision",
            name="uq_decision_events_type_revision",
        ),
        sa.UniqueConstraint(
            "decision_id",
            "idempotency_key",
            name="uq_decision_events_idempotency_key",
        ),
        sa.CheckConstraint(
            "(idempotency_key IS NULL AND request_sha256 IS NULL) OR "
            "(idempotency_key IS NOT NULL AND request_sha256 IS NOT NULL)",
            name="ck_decision_events_idempotency_pair",
        ),
    )
    op.create_index(
        "ix_decision_events_decision_created",
        "decision_events",
        ["decision_id", "created_at"],
    )


def downgrade():
    op.drop_table("decision_events")
    op.drop_table("decision_records")
