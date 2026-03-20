"""Create OHLCV, fundamentals, sentiment, backfill_progress tables.

Convert ohlcv to TimescaleDB hypertable and set compression policy.

Revision ID: 002
Revises: 001
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    # --- OHLCV table ---
    op.create_table(
        "ohlcv",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("open", sa.Numeric, nullable=False),
        sa.Column("high", sa.Numeric, nullable=False),
        sa.Column("low", sa.Numeric, nullable=False),
        sa.Column("close", sa.Numeric, nullable=False),
        sa.Column("volume", sa.Numeric, nullable=False),
    )
    op.create_primary_key("pk_ohlcv", "ohlcv", ["time", "symbol", "market", "interval"])

    # Convert to TimescaleDB hypertable (1 month chunks)
    op.execute(
        "SELECT create_hypertable('ohlcv', 'time', "
        "chunk_time_interval => INTERVAL '1 month', "
        "if_not_exists => TRUE)"
    )

    # Create index for most common query pattern
    op.execute(
        "CREATE INDEX idx_ohlcv_symbol_market_interval_time "
        "ON ohlcv (symbol, market, interval, time DESC)"
    )

    # Enable compression
    op.execute("""
        ALTER TABLE ohlcv SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, market, interval',
            timescaledb.compress_orderby = 'time DESC'
        )
    """)

    # Auto-compress chunks older than 7 days
    op.execute("SELECT add_compression_policy('ohlcv', INTERVAL '7 days')")

    # --- Fundamentals table ---
    op.create_table(
        "fundamentals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("data", JSONB, nullable=False),
    )
    op.create_unique_constraint("uq_fundamentals_symbol_market_date", "fundamentals", ["symbol", "market", "date"])

    # --- Sentiment table ---
    op.create_table(
        "sentiment",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- Backfill progress table ---
    op.create_table(
        "backfill_progress",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("last_fetched_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("target_start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_unique_constraint(
        "uq_backfill_symbol_market_interval", "backfill_progress", ["symbol", "market", "interval"]
    )


def downgrade():
    op.drop_table("backfill_progress")
    op.drop_table("sentiment")
    op.drop_table("fundamentals")
    # Remove compression policy before dropping hypertable
    op.execute("SELECT remove_compression_policy('ohlcv', if_exists => TRUE)")
    op.drop_table("ohlcv")
