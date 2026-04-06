"""Create protection_locks table.

Revision ID: 019
Revises: 018
Create Date: 2026-04-06

Changes:
- CREATE TABLE protection_locks for persistent protection lock storage
- Composite index for efficient active lock lookups
"""

import sqlalchemy as sa
from alembic import op

revision = "019"
down_revision = "018"


def upgrade():
    op.create_table(
        "protection_locks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("protection_type", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "locked_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_by", sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_protection_locks_protection_type",
        "protection_locks",
        ["protection_type"],
    )
    op.create_index(
        "ix_protection_locks_symbol",
        "protection_locks",
        ["symbol"],
    )
    op.create_index(
        "ix_protection_locks_market",
        "protection_locks",
        ["market"],
    )
    op.create_index(
        "ix_protection_locks_active",
        "protection_locks",
        ["active"],
    )
    op.create_index(
        "ix_protection_locks_active_lookup",
        "protection_locks",
        ["active", "protection_type", "symbol", "market"],
    )


def downgrade():
    op.drop_index("ix_protection_locks_active_lookup", "protection_locks")
    op.drop_index("ix_protection_locks_active", "protection_locks")
    op.drop_index("ix_protection_locks_market", "protection_locks")
    op.drop_index("ix_protection_locks_symbol", "protection_locks")
    op.drop_index("ix_protection_locks_protection_type", "protection_locks")
    op.drop_table("protection_locks")
