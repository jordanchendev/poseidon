"""Create durable RD-Agent research run records.

Revision ID: 039
Revises: 038
Create Date: 2026-09-21
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "rd_agent_runs",
        sa.Column("run_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("challenge", sa.Text, nullable=False),
        sa.Column("time_budget_hours", sa.Float, nullable=False, server_default="4"),
        sa.Column("cost_cap_usd", sa.Float, nullable=False, server_default="20"),
        sa.Column("use_gpu", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("cancel_reason", sa.Text, nullable=True),
        sa.Column("token_cost_acc_usd", sa.Float, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("summary", JSONB, nullable=True),
        sa.Column("verdict", sa.String(32), nullable=True),
        sa.Column("result_dir", sa.Text, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("requested_by", sa.String(16), nullable=False, server_default="api"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_check_constraint(
        "ck_rd_agent_runs_status",
        "rd_agent_runs",
        "status IN ('pending','running','succeeded','failed','cancelled')",
    )
    op.create_index("ix_rd_agent_runs_status", "rd_agent_runs", ["status"])
    op.create_index("ix_rd_agent_runs_created_at", "rd_agent_runs", ["created_at"])


def downgrade():
    op.drop_index("ix_rd_agent_runs_created_at", table_name="rd_agent_runs")
    op.drop_index("ix_rd_agent_runs_status", table_name="rd_agent_runs")
    op.drop_table("rd_agent_runs")
