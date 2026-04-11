"""Create factor_analysis_runs table (Phase 47 FACTOR-04)."""

import sqlalchemy as sa
from alembic import op

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "factor_analysis_runs",
        sa.Column(
            "id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("run_type", sa.String(length=16), nullable=False),
        sa.Column(
            "config_json", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "results_json", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("market", sa.String(length=32), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "run_type IN ('ic', 'shapley', 'centrality')",
            name="ck_factor_analysis_runs_run_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed')",
            name="ck_factor_analysis_runs_status",
        ),
    )
    op.create_index(
        "ix_factor_analysis_runs_market",
        "factor_analysis_runs",
        ["market"],
    )
    op.create_index(
        "ix_factor_analysis_runs_created_at",
        "factor_analysis_runs",
        ["created_at"],
    )


def downgrade():
    op.drop_index("ix_factor_analysis_runs_created_at", table_name="factor_analysis_runs")
    op.drop_index("ix_factor_analysis_runs_market", table_name="factor_analysis_runs")
    op.drop_table("factor_analysis_runs")
