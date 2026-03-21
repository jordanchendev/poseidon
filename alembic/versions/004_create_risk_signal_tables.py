"""Create signals, risk_rules, and virtual_positions tables.

Revision ID: 004
Revises: 003
Create Date: 2026-03-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    # --- Signals table ---
    op.create_table(
        "signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=True),
        sa.Column("model_id", UUID(as_uuid=True), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("quantity_pct", sa.Float, nullable=True),
        sa.Column("signal_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("interval", sa.String(8), nullable=False, server_default="1d"),
        sa.Column("params", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("reject_reason", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_signals_market_status", "signals", ["market", "status"])
    op.create_index("idx_signals_strategy_time", "signals", ["strategy_id", "signal_time"])
    op.create_index("idx_signals_symbol_time", "signals", ["symbol", "signal_time"])

    # --- Risk rules table ---
    op.create_table(
        "risk_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("rule_type", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("params", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint("uq_risk_rules_name", "risk_rules", ["name"])

    # --- Virtual positions table ---
    op.create_table(
        "virtual_positions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("quantity_pct", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("entry_signal_id", UUID(as_uuid=True), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("close_signal_id", UUID(as_uuid=True), nullable=True),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_vp_market_symbol_closed", "virtual_positions", ["market", "symbol", "closed"])


def downgrade():
    op.drop_table("virtual_positions")
    op.drop_table("risk_rules")
    op.drop_table("signals")
