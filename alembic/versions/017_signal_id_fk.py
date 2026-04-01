"""Add signal_id FK to trade_logs with zero-downtime pattern.

Revision ID: 017
Revises: 016
Create Date: 2026-04-01

Changes:
- trade_logs.signal_id: nullable UUID FK to signals(id)
- Uses NOT VALID + VALIDATE CONSTRAINT for zero-downtime deployment
- Index ix_trade_logs_signal_id for query performance
"""

import sqlalchemy as sa
from alembic import op

revision = "017"
down_revision = "016"


def upgrade():
    # Step 1: Add nullable column (instant, no lock)
    op.add_column(
        "trade_logs",
        sa.Column("signal_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
    )
    # Step 2: Add FK constraint as NOT VALID (SHARE UPDATE EXCLUSIVE lock only)
    op.execute("""
        ALTER TABLE trade_logs
        ADD CONSTRAINT fk_trade_logs_signal_id
        FOREIGN KEY (signal_id) REFERENCES signals(id)
        NOT VALID
    """)
    # Step 3: Validate constraint (allows concurrent writes during scan)
    op.execute("""
        ALTER TABLE trade_logs
        VALIDATE CONSTRAINT fk_trade_logs_signal_id
    """)
    # Step 4: Index for query performance
    op.create_index("ix_trade_logs_signal_id", "trade_logs", ["signal_id"])


def downgrade():
    op.drop_index("ix_trade_logs_signal_id", "trade_logs")
    op.drop_constraint("fk_trade_logs_signal_id", "trade_logs", type_="foreignkey")
    op.drop_column("trade_logs", "signal_id")
