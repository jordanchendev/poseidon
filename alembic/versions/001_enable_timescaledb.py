"""Enable TimescaleDB extension.

Revision ID: 001
Revises:
Create Date: 2026-03-20
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")


def downgrade():
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE")
