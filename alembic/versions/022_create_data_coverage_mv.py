"""Create data_coverage_mv materialized view (Phase 39 plan 39-03).

Revision ID: 022
Revises: 021
Create Date: 2026-04-09

Phase 39 decisions (.planning/phases/39-backfill-api-coverage/39-CONTEXT.md):

- D-10: data_coverage_mv is the source of truth for per-tuple coverage and
  must have a unique index for ``REFRESH MATERIALIZED VIEW CONCURRENTLY``.
- D-11: GET /api/data/coverage reads from the MV and computes expected_count
  /gap_count/completeness_pct/health at request time.
- D-12: Refresh runs hourly plus immediately after successful backfill
  completion (the hourly beat task and the success-path hook are both in
  poseidon.workers.backfill_tasks).

MV columns:

- market / symbol / interval    (the tuple key)
- first_ts / last_ts            (MIN/MAX of ``ohlcv.time``)
- row_count                     (COUNT(*) of ``ohlcv`` rows in the tuple)
- staleness_seconds             (now() - last_ts)

The DDL targets Postgres/TimescaleDB; the unit suite exercises the Python
code paths with a SQLite stand-in (see tests/unit/test_data_coverage_api.py).
"""

from alembic import op

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade():
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

    # Unique index is a hard requirement for REFRESH MATERIALIZED VIEW
    # CONCURRENTLY. It also gives the (market, symbol, interval) lookup
    # path in GET /api/data/coverage a cheap seek.
    op.execute(
        """
        CREATE UNIQUE INDEX ix_data_coverage_mv_tuple
            ON data_coverage_mv (market, symbol, interval)
        """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_data_coverage_mv_tuple")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS data_coverage_mv")
