"""Tests for the Phase 40 foundation slice (plan 40-01).

Covers the locked Phase 40 decisions in
.planning/phases/40-data-health-observability/40-CONTEXT.md:

- Task 1: migration 023 (``data_gaps`` table) + DataGap ORM model
- Task 2: migration 024 (``ohlcv_1d_cagg`` continuous aggregate) shape
- Task 3: Settings fields (``freshness_sla``, ``uptime_kuma_push_url``,
  ``cagg_1d_markets``) + DataGapResponse / DataFreshnessResponse schemas

The unit suite runs against an in-memory SQLite harness that uses the
Postgres-only types via the same ``@compiles`` shims as
``tests/unit/test_data_coverage_api.py``. TimescaleDB-specific DDL (CAGG +
refresh policy) and Postgres server_default (``gen_random_uuid()``) are
exercised here only at the grep/assertion level; real DDL execution runs
end-to-end on stormtrooper via the Phase 40 smoke runbook (plan 40-06).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

# --- SQLite compatibility shims for Postgres-only types (must run before
# any Base.metadata.create_all call) ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


# ---------------------------------------------------------------------------
# Task 1: migration 023 + DataGap ORM
# ---------------------------------------------------------------------------

MIGRATION_023_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "023_create_data_gaps.py"
)


def test_migration_023_creates_data_gaps_table():
    """023 must exist with the right revision chain and create the data_gaps table."""
    assert MIGRATION_023_PATH.exists(), f"migration file missing: {MIGRATION_023_PATH}"
    content = MIGRATION_023_PATH.read_text()

    assert 'revision = "023"' in content
    assert 'down_revision = "022"' in content
    assert 'op.create_table(' in content
    assert '"data_gaps"' in content
    # Required columns (D-04 schema)
    for col in (
        "gap_id",
        "market",
        "symbol",
        "interval",
        "gap_start",
        "gap_end",
        "missing_bars",
        "detected_at",
        "healed_at",
    ):
        assert col in content, f"missing column in migration 023: {col}"


def test_migration_023_unique_index():
    """D-04: unique index on (market, symbol, interval, gap_start) enforces
    the idempotent ``ON CONFLICT DO NOTHING`` contract used by the daily audit
    (D-07)."""
    content = MIGRATION_023_PATH.read_text()

    # Must have the unique tuple-start index
    assert "ix_data_gaps_tuple_start" in content
    assert "unique=True" in content
    # Must reference all four columns of the uniqueness tuple
    for col in ("market", "symbol", "interval", "gap_start"):
        assert col in content

    # Must also have a partial "open gaps" index (healed_at IS NULL) so the
    # dashboard's open_only=true query is fast.
    assert "ix_data_gaps_open" in content
    assert "healed_at IS NULL" in content


def test_migration_023_downgrade_drops_in_reverse_order():
    content = MIGRATION_023_PATH.read_text()
    downgrade_block = content.split("def downgrade")[-1]
    # Must drop indexes before the table, in the reverse of create order.
    assert "ix_data_gaps_open" in downgrade_block
    assert "ix_data_gaps_tuple_start" in downgrade_block
    assert 'op.drop_table("data_gaps")' in downgrade_block


def test_data_gap_orm_tablename():
    """DataGap must be importable from its submodule and have the right table name."""
    from poseidon.models.data_gap import DataGap

    assert DataGap.__tablename__ == "data_gaps"


def test_data_gap_orm_columns():
    """DataGap must expose every column required by the D-04 schema."""
    from poseidon.models.data_gap import DataGap

    columns = {c.name for c in DataGap.__table__.columns}
    expected = {
        "gap_id",
        "market",
        "symbol",
        "interval",
        "gap_start",
        "gap_end",
        "missing_bars",
        "detected_at",
        "healed_at",
    }
    assert expected.issubset(columns), f"missing columns: {expected - columns}"


def test_data_gap_importable_from_models_package():
    """DataGap must be re-exported from poseidon.models so downstream
    plans (40-02/40-03/40-04) can import it without reaching into a submodule."""
    from poseidon.models import DataGap

    assert DataGap.__tablename__ == "data_gaps"


# ---------------------------------------------------------------------------
# Task 2: migration 024 — ohlcv_1d_cagg
# ---------------------------------------------------------------------------

MIGRATION_024_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "024_create_ohlcv_1d_cagg.py"
)


def test_migration_024_creates_continuous_aggregate():
    """024 must create ohlcv_1d_cagg as a TimescaleDB continuous aggregate."""
    assert MIGRATION_024_PATH.exists(), f"migration file missing: {MIGRATION_024_PATH}"
    content = MIGRATION_024_PATH.read_text()

    assert 'revision = "024"' in content
    assert 'down_revision = "023"' in content
    assert "CREATE MATERIALIZED VIEW ohlcv_1d_cagg" in content
    assert "WITH (timescaledb.continuous)" in content


def test_migration_024_restricts_source_intervals():
    """D-18: The CAGG must only roll up sub-daily intervals so raw 1d rows
    for tw_stock/tw_futures/us_stock are NOT double-rolled (identity feedback)."""
    content = MIGRATION_024_PATH.read_text()
    assert "interval IN ('1m','5m','15m','30m','1h','4h')" in content


def test_migration_024_uses_daily_time_bucket():
    """D-17: Must use ``time_bucket('1 day', time)`` for the rollup."""
    content = MIGRATION_024_PATH.read_text()
    assert "time_bucket('1 day', time)" in content


def test_migration_024_hourly_refresh_policy():
    """D-19: Must install an hourly ``add_continuous_aggregate_policy``."""
    content = MIGRATION_024_PATH.read_text()
    assert "add_continuous_aggregate_policy" in content
    assert "schedule_interval => INTERVAL '1 hour'" in content


def test_migration_024_downgrade_removes_policy_and_view():
    content = MIGRATION_024_PATH.read_text()
    downgrade_block = content.split("def downgrade")[-1]
    assert "remove_continuous_aggregate_policy" in downgrade_block
    assert "DROP MATERIALIZED VIEW" in downgrade_block
    assert "ohlcv_1d_cagg" in downgrade_block


# ---------------------------------------------------------------------------
# Task 3: Settings + schemas
# ---------------------------------------------------------------------------


def test_settings_freshness_sla_defaults():
    """D-10/D-11: Settings.freshness_sla must default to the locked Phase 40
    SLA values, keyed as ``"market:interval"`` -> seconds."""
    from poseidon.core.config import Settings

    s = Settings()
    assert isinstance(s.freshness_sla, dict)
    assert s.freshness_sla  # non-empty
    # D-11: crypto_perp/4h -> 5h (18000s)
    assert s.freshness_sla["crypto_perp:4h"] == 18000
    # D-11: crypto_spot/1h -> 2h (7200s)
    assert s.freshness_sla["crypto_spot:1h"] == 7200
    # D-11: crypto_spot/1d -> 30h (108000s)
    assert s.freshness_sla["crypto_spot:1d"] == 108000
    # D-11: tw_stock/1d -> 30h (covers weekends+holidays)
    assert s.freshness_sla["tw_stock:1d"] == 108000
    # D-11: tw_futures/1d -> 30h
    assert s.freshness_sla["tw_futures:1d"] == 108000
    # D-11: us_stock/1d -> 30h
    assert s.freshness_sla["us_stock:1d"] == 108000


def test_settings_uptime_kuma_push_url_default_empty():
    """D-13: uptime_kuma_push_url defaults to empty string (no-op for
    local/dev) and reads UPTIME_KUMA_PUSH_URL (unprefixed env var)."""
    from poseidon.core.config import Settings

    s = Settings()
    assert s.uptime_kuma_push_url == ""


def test_settings_uptime_kuma_push_url_env_override(monkeypatch):
    """The UPTIME_KUMA_PUSH_URL validation_alias must pick up an
    unprefixed env var (the watchdog runbook sets exactly this name)."""
    from poseidon.core.config import Settings

    monkeypatch.setenv("UPTIME_KUMA_PUSH_URL", "http://localhost:3001/api/push/fake-token")
    s = Settings()
    assert s.uptime_kuma_push_url == "http://localhost:3001/api/push/fake-token"


def test_settings_cagg_1d_markets_default():
    """D-20: cagg_1d_markets defaults to crypto_perp and crypto_spot only
    (the only markets with sub-daily raw data today)."""
    from poseidon.core.config import Settings

    s = Settings()
    assert s.cagg_1d_markets == ["crypto_perp", "crypto_spot"]


def test_data_gap_response_schema_instantiates():
    """D-02: DataGapResponse must accept the D-04 row shape directly."""
    from uuid import uuid4

    from poseidon.core.schemas import DataGapResponse

    resp = DataGapResponse(
        gap_id=uuid4(),
        market="crypto_perp",
        symbol="BTCUSDT",
        interval="4h",
        gap_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        gap_end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        missing_bars=6,
        detected_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        healed_at=None,
    )
    assert resp.market == "crypto_perp"
    assert resp.missing_bars == 6
    assert resp.healed_at is None


def test_data_freshness_response_schema_instantiates():
    """D-03/D-11/D-16: DataFreshnessResponse must carry per-(market, interval)
    SLA status derived from ingest_state.last_successful_ts."""
    from poseidon.core.schemas import DataFreshnessResponse

    resp = DataFreshnessResponse(
        market="crypto_perp",
        interval="4h",
        last_successful_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        expected_lag_seconds=18000,
        observed_lag_seconds=20000.5,
        status="violation",
    )
    assert resp.market == "crypto_perp"
    assert resp.expected_lag_seconds == 18000
    assert resp.status == "violation"
