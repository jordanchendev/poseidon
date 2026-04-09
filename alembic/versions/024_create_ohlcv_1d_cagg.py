"""Create ohlcv_1d_cagg TimescaleDB continuous aggregate (Phase 40 D-17..D-20).

Revision ID: 024
Revises: 023
Create Date: 2026-04-09

Phase 40 decisions (.planning/phases/40-data-health-observability/40-CONTEXT.md):

- D-17: Rolls sub-daily bars up to daily via ``time_bucket('1 day', time)``.
  Columns carried: market, symbol, interval_source (the original interval),
  time_bucket_day, first(open), max(high), min(low), last(close), sum(volume).
  The source ``interval`` is stored as ``interval_source`` so the CAGG can
  carry both ``crypto_perp/4h -> 1d`` and ``crypto_spot/1h -> 1d`` rollups
  in the same view.
- D-18: Source rows are restricted to sub-daily intervals
  ``interval IN ('1m','5m','15m','30m','1h','4h')``. Raw 1d rows (tw_stock,
  tw_futures, us_stock) are intentionally EXCLUDED to prevent an identity
  feedback loop — raw 1d stays verbatim in the ``ohlcv`` hypertable.
- D-19: Hourly refresh via ``add_continuous_aggregate_policy`` with
  ``schedule_interval => INTERVAL '1 hour'``. No manual post-backfill
  refresh hook — 1d-granularity training workflows do not need sub-hour
  freshness.
- D-20: ``settings.cagg_1d_markets`` (default ``["crypto_perp",
  "crypto_spot"]``) controls which markets dispatch ``read_ohlcv(
  interval='1d')`` to this CAGG (Phase 40 plan 40-04). Markets outside the
  list fall through to raw 1d rows.

``$vwap`` / ``amount`` / ``turnover`` are explicitly deferred to v9.0
CAGG-V9-01 (D-23). ``DatasetBuilder`` continues to compute vwap downstream
via the HLC/3 proxy.

The ``WITH NO DATA`` clause is intentional — TimescaleDB requires it when
creating a continuous aggregate against a restricted source set, and the
hourly refresh policy will populate the view asynchronously. The initial
backfill for markets already in scope is triggered by the first scheduled
refresh or a manual ``CALL refresh_continuous_aggregate(...)`` on
stormtrooper.
"""

from alembic import op

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade():
    # D-17/D-18: restricted CAGG covering only sub-daily source intervals.
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

    # D-19: hourly refresh policy. TimescaleDB-native, runs on the
    # background worker — no Celery coupling, no custom beat entry.
    op.execute(
        """
        SELECT add_continuous_aggregate_policy(
            'ohlcv_1d_cagg',
            start_offset => INTERVAL '90 days',
            end_offset   => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour'
        )
        """
    )


def downgrade():
    # Drop the policy first, then the view (CASCADE handles any dependent
    # objects). ``if_exists => true`` keeps downgrade idempotent.
    op.execute(
        "SELECT remove_continuous_aggregate_policy('ohlcv_1d_cagg', if_exists => true)"
    )
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ohlcv_1d_cagg CASCADE")
