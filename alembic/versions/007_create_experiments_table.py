"""Create experiments table and optuna schema.

Revision ID: 007
Revises: 006
Create Date: 2026-03-26

Per D-02: optuna schema created by migration; Optuna manages its own tables within it.
Per D-04: ExperimentRecord fields match the CONTEXT spec exactly.
Per D-05: optuna_study_name and optuna_trial_number provide optional linkage without FK.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade():
    # Create optuna schema for Optuna's internal storage
    op.execute("CREATE SCHEMA IF NOT EXISTS optuna")

    # Create experiments table
    op.create_table(
        "experiments",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("study_name", sa.String(128), nullable=False),
        sa.Column("config_json", JSONB, nullable=False),
        sa.Column("metrics_json", JSONB, nullable=True),
        sa.Column("composite_score", sa.Numeric, nullable=True),
        sa.Column("wfe_score", sa.Numeric, nullable=True),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="running"
        ),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("optuna_study_name", sa.String(128), nullable=True),
        sa.Column("optuna_trial_number", sa.Integer, nullable=True),
        sa.Column("holdout_boundary", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_experiments_market_interval", "experiments", ["market", "interval"]
    )
    op.create_index("ix_experiments_created_at", "experiments", ["created_at"])


def downgrade():
    op.drop_table("experiments")
    op.execute("DROP SCHEMA IF EXISTS optuna CASCADE")
