"""Create backtests, backtest_trades, and backtest_equity tables.

Revision ID: 005
Revises: 004
Create Date: 2026-03-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade():
    # --- Backtests (main) table ---
    op.create_table(
        "backtests",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("strategy_id", UUID(as_uuid=True), nullable=True),
        sa.Column("strategy_type", sa.String(16), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False, server_default="1d"),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("metrics", JSONB, nullable=True),
        sa.Column("walk_forward", JSONB, nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_backtests_strategy_status", "backtests", ["strategy_id", "status"])

    # --- Backtest trades table ---
    op.create_table(
        "backtest_trades",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("backtest_id", UUID(as_uuid=True), sa.ForeignKey("backtests.id"), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("entry_price", sa.Numeric, nullable=False),
        sa.Column("exit_price", sa.Numeric, nullable=True),
        sa.Column("quantity", sa.Numeric, nullable=False),
        sa.Column("pnl", sa.Numeric, nullable=True),
        sa.Column("fees", sa.Numeric, nullable=False, server_default="0"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("idx_bt_trades_backtest_id", "backtest_trades", ["backtest_id"])

    # --- Backtest equity table ---
    op.create_table(
        "backtest_equity",
        sa.Column("backtest_id", UUID(as_uuid=True), sa.ForeignKey("backtests.id"), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Numeric, nullable=False),
        sa.Column("drawdown", sa.Numeric, nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("backtest_id", "time"),
    )
    op.create_index("idx_bt_equity_backtest_id", "backtest_equity", ["backtest_id"])


def downgrade():
    op.drop_table("backtest_equity")
    op.drop_table("backtest_trades")
    op.drop_table("backtests")
