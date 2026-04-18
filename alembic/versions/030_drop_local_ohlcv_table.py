"""Drop local OHLCV storage after Thalassa cutover.

Revision ID: 030
Revises: 029
Create Date: 2026-04-18

Poseidon no longer owns OHLCV data locally. Runtime code now reads market
data exclusively via Thalassa's RemoteDataRepository, so the legacy
``ohlcv`` hypertable and its derived observability views must be removed to
prevent drift between source-of-truth and local leftovers.
"""

import sqlalchemy as sa
from alembic import op

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SELECT remove_compression_policy('ohlcv', if_exists => TRUE)")
    op.execute(
        "SELECT remove_continuous_aggregate_policy('ohlcv_1d_cagg', if_exists => true)"
    )
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ohlcv_1d_cagg CASCADE")
    op.execute("DROP INDEX IF EXISTS ix_data_coverage_mv_tuple")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS data_coverage_mv CASCADE")
    op.drop_table("ohlcv")


def downgrade():
    op.create_table(
        "ohlcv",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("open", sa.Numeric(), nullable=False),
        sa.Column("high", sa.Numeric(), nullable=False),
        sa.Column("low", sa.Numeric(), nullable=False),
        sa.Column("close", sa.Numeric(), nullable=False),
        sa.Column("volume", sa.Numeric(), nullable=False),
    )
    op.create_primary_key("pk_ohlcv", "ohlcv", ["time", "symbol", "market", "interval"])
    op.execute(
        "CREATE INDEX idx_ohlcv_symbol_market_interval_time "
        "ON ohlcv (symbol, market, interval, time DESC)"
    )
    op.execute(
        """
        CREATE MATERIALIZED VIEW data_coverage_mv AS
        SELECT
            market,
            symbol,
            interval,
            MIN(time) AS first_ts,
            MAX(time) AS last_ts,
            COUNT(*) AS row_count,
            EXTRACT(EPOCH FROM (now() - MAX(time))) AS staleness_seconds
        FROM ohlcv
        GROUP BY market, symbol, interval
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX ix_data_coverage_mv_tuple
            ON data_coverage_mv (market, symbol, interval)
        """
    )
    op.execute(
        """
        CREATE MATERIALIZED VIEW ohlcv_1d_cagg
        WITH (timescaledb.continuous) AS
        SELECT
            market,
            symbol,
            interval AS interval_source,
            time_bucket('1 day', time) AS time_bucket_day,
            first(open, time) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, time) AS close,
            sum(volume) AS volume
        FROM ohlcv
        WHERE interval IN ('1m','5m','15m','30m','1h','4h')
        GROUP BY market, symbol, interval, time_bucket('1 day', time)
        WITH NO DATA
        """
    )
