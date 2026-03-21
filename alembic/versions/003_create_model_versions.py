"""Create model_versions table for ML model lifecycle tracking.

Revision ID: 003
Revises: 002
Create Date: 2026-03-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="training"),
        sa.Column("params", JSONB, nullable=False, server_default="{}"),
        sa.Column("metrics", JSONB, nullable=True),
        sa.Column("feature_list", JSONB, nullable=False, server_default="[]"),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("artifact_path", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_unique_constraint("uq_model_versions_name_version", "model_versions", ["name", "version"])
    op.create_index("idx_model_versions_name_status", "model_versions", ["name", "status"])


def downgrade():
    op.drop_table("model_versions")
