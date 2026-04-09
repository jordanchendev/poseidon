"""Tests for the Phase 40 GET /api/data/freshness endpoint (plan 40-03 Task 3).

Covers the locked decisions in
.planning/phases/40-data-health-observability/40-CONTEXT.md (D-03, D-16):

- D-03: freshness is a stateless read of ingest_state + SLA dict
- D-16: no snapshot table, live computation from ingest_state.last_successful_ts
- Endpoint mounts under /api/data/freshness (NOT /api/v1/data/freshness)
- Response shape matches DataFreshnessResponse

Uses the same in-memory SQLite harness + @compiles shim pattern established
in test_data_coverage_api.py (Phase 39) and test_data_gaps_api.py (Phase 40-02).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility shims for Postgres-only types ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.ingest_state import IngestState  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.api.data import router as data_router  # noqa: E402

# --- Test DB / app wiring ---

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=_engine
)

_test_app = FastAPI()
_test_app.include_router(data_router, prefix="/api/data", tags=["data"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    return TestClient(_test_app)


# Predictable SLA dict for tests
TEST_SLA = {
    "crypto_perp:4h": 18000,  # 5h
    "crypto_spot:1h": 7200,  # 2h
}

# Use real current time so lag computations are consistent with the endpoint's
# internal datetime.now(timezone.utc) call.
REAL_NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _setup_db(monkeypatch):
    """Create ORM tables + stub MV + inject test SLA for every test."""
    Base.metadata.create_all(bind=_engine)

    # The data router's /coverage and /gaps endpoints reference data_coverage_mv
    # and data_gaps tables. Create minimal SQLite stand-ins so the router
    # import does not fail.
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS data_coverage_mv (
                    market TEXT, symbol TEXT, interval TEXT,
                    first_ts TEXT, last_ts TEXT, row_count INTEGER,
                    staleness_seconds REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS data_gaps (
                    gap_id VARCHAR(36) PRIMARY KEY,
                    market TEXT, symbol TEXT, interval TEXT,
                    gap_start TEXT, gap_end TEXT,
                    missing_bars INTEGER,
                    detected_at TEXT, healed_at TEXT
                )
                """
            )
        )
        # Create the unique index the gaps code needs
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ix_data_gaps_tuple_start
                ON data_gaps (market, symbol, interval, gap_start)
                """
            )
        )

    # Inject a small predictable SLA dict for all tests
    monkeypatch.setattr(
        "poseidon.core.config.settings.freshness_sla",
        TEST_SLA,
    )

    yield

    Base.metadata.drop_all(bind=_engine)
    with _engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_coverage_mv"))
        conn.execute(text("DROP TABLE IF EXISTS data_gaps"))


def _seed_rows():
    """Seed IngestState rows for testing: one ok, one violation, one no-SLA."""
    session = TestingSessionLocal()
    try:
        # crypto_perp BTCUSDT 4h — ok (1h old, SLA is 5h)
        session.add(
            IngestState(
                symbol="BTCUSDT",
                market="crypto_perp",
                interval="4h",
                last_successful_ts=REAL_NOW - timedelta(seconds=3600),
                first_backfill_done=True,
            )
        )
        # crypto_spot ETHUSDT 1h — violation (3h old, SLA is 2h)
        session.add(
            IngestState(
                symbol="ETHUSDT",
                market="crypto_spot",
                interval="1h",
                last_successful_ts=REAL_NOW - timedelta(seconds=10800),
                first_backfill_done=True,
            )
        )
        # unknown_market FOO 1d — no SLA → should NOT appear in response
        session.add(
            IngestState(
                symbol="FOO",
                market="unknown_market",
                interval="1d",
                last_successful_ts=REAL_NOW - timedelta(seconds=100),
                first_backfill_done=True,
            )
        )
        session.commit()
    finally:
        session.close()


class TestGetDataFreshness:
    """Tests 1-5: GET /api/data/freshness endpoint."""

    def test_returns_correct_statuses(self, client):
        """Test 1: with seeded rows, response has correct ok/violation statuses."""
        _seed_rows()
        resp = client.get("/api/data/freshness")
        assert resp.status_code == 200
        data = resp.json()

        # Should have 2 entries (crypto_perp:4h and crypto_spot:1h)
        assert len(data) == 2

        by_market = {r["market"]: r for r in data}
        assert by_market["crypto_perp"]["status"] == "ok"
        assert by_market["crypto_spot"]["status"] == "violation"

    def test_no_sla_tuples_excluded(self, client):
        """Test 2: tuples without SLA are NOT in the response."""
        _seed_rows()
        resp = client.get("/api/data/freshness")
        data = resp.json()
        markets = {r["market"] for r in data}
        assert "unknown_market" not in markets

    def test_empty_table_returns_empty(self, client):
        """Test 3: no IngestState rows returns []."""
        resp = client.get("/api/data/freshness")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_response_shape(self, client):
        """Test 4: response fields match DataFreshnessResponse exactly."""
        _seed_rows()
        resp = client.get("/api/data/freshness")
        data = resp.json()
        assert len(data) >= 1

        expected_keys = {
            "market",
            "interval",
            "last_successful_ts",
            "expected_lag_seconds",
            "observed_lag_seconds",
            "status",
        }
        for row in data:
            assert set(row.keys()) == expected_keys

    def test_market_filter(self, client):
        """Test 5: ?market=crypto_perp filters to that market only."""
        _seed_rows()
        resp = client.get("/api/data/freshness?market=crypto_perp")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["market"] == "crypto_perp"
        assert data[0]["interval"] == "4h"
