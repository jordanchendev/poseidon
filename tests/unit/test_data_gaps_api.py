"""Tests for the Phase 40 GET /api/data/gaps endpoint (plan 40-02 Task 2).

Covers the locked decisions in
.planning/phases/40-data-health-observability/40-CONTEXT.md (D-01, D-02):

- D-01: endpoint mounts under /api/data/gaps (NOT /api/v1/data/gaps)
- D-02: query filters market, symbol, interval, open_only=true default
- Response shape matches DataGapResponse with all required fields

Uses the same in-memory SQLite harness + @compiles shim pattern established
in test_data_coverage_api.py (Phase 39).
"""

from __future__ import annotations

import uuid
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
from poseidon.models.data_gap import DataGap  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.api.data import router as data_router  # noqa: E402

# --- Test DB / app wiring ---

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

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


@pytest.fixture(autouse=True)
def _setup_db():
    """Create ORM tables + SQLite stand-ins for tables the data router needs."""
    Base.metadata.create_all(bind=_engine)
    with _engine.begin() as conn:
        # Stand-in for data_coverage_mv (needed by /coverage endpoint that
        # shares the same router)
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS data_coverage_mv (
                    market VARCHAR(32) NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    interval VARCHAR(8) NOT NULL,
                    first_ts TIMESTAMP,
                    last_ts TIMESTAMP,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    staleness_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                    PRIMARY KEY (market, symbol, interval)
                )
                """
            )
        )
    yield
    with _engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_coverage_mv"))
    Base.metadata.drop_all(bind=_engine)


def _seed_gap(
    *,
    market: str = "crypto_perp",
    symbol: str = "BTCUSDT",
    interval: str = "4h",
    gap_start: datetime | None = None,
    gap_end: datetime | None = None,
    missing_bars: int = 2,
    healed_at: datetime | None = None,
) -> None:
    """Insert a DataGap row directly via the test session."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    if gap_start is None:
        gap_start = base + timedelta(hours=8)
    if gap_end is None:
        gap_end = base + timedelta(hours=20)

    session = TestingSessionLocal()
    try:
        gap = DataGap(
            gap_id=uuid.uuid4(),
            market=market,
            symbol=symbol,
            interval=interval,
            gap_start=gap_start,
            gap_end=gap_end,
            missing_bars=missing_bars,
            healed_at=healed_at,
        )
        session.add(gap)
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 1: open_only=true (default) filters out healed gaps
# ---------------------------------------------------------------------------


def test_get_gaps_default_open_only_filters_healed(client):
    """Two open gaps + one healed => default returns only two open gaps."""
    _seed_gap(symbol="BTCUSDT", gap_start=datetime(2024, 1, 1, 8, tzinfo=timezone.utc))
    _seed_gap(symbol="ETHUSDT", gap_start=datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
    _seed_gap(
        symbol="SOLUSDT",
        gap_start=datetime(2024, 1, 1, 16, tzinfo=timezone.utc),
        healed_at=datetime(2024, 1, 2, 3, tzinfo=timezone.utc),
    )

    resp = client.get("/api/data/gaps")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    # All returned rows should have healed_at = None
    for row in rows:
        assert row["healed_at"] is None


# ---------------------------------------------------------------------------
# Test 2: open_only=false returns all rows including healed
# ---------------------------------------------------------------------------


def test_get_gaps_open_only_false_returns_all(client):
    """open_only=false returns all three rows (two open + one healed)."""
    _seed_gap(symbol="BTCUSDT", gap_start=datetime(2024, 1, 1, 8, tzinfo=timezone.utc))
    _seed_gap(symbol="ETHUSDT", gap_start=datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
    _seed_gap(
        symbol="SOLUSDT",
        gap_start=datetime(2024, 1, 1, 16, tzinfo=timezone.utc),
        healed_at=datetime(2024, 1, 2, 3, tzinfo=timezone.utc),
    )

    resp = client.get("/api/data/gaps", params={"open_only": "false"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Test 3: market filter
# ---------------------------------------------------------------------------


def test_get_gaps_market_filter(client):
    """?market=crypto_perp returns only crypto_perp rows."""
    _seed_gap(market="crypto_perp", symbol="BTCUSDT")
    _seed_gap(market="crypto_spot", symbol="BTCUSDT",
              gap_start=datetime(2024, 1, 2, 8, tzinfo=timezone.utc))

    resp = client.get("/api/data/gaps", params={"market": "crypto_perp"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["market"] == "crypto_perp"


# ---------------------------------------------------------------------------
# Test 4: symbol filter
# ---------------------------------------------------------------------------


def test_get_gaps_symbol_filter(client):
    """?symbol=BTCUSDT returns only BTCUSDT rows."""
    _seed_gap(symbol="BTCUSDT")
    _seed_gap(symbol="ETHUSDT", gap_start=datetime(2024, 1, 2, 8, tzinfo=timezone.utc))

    resp = client.get("/api/data/gaps", params={"symbol": "BTCUSDT"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# Test 5: interval filter
# ---------------------------------------------------------------------------


def test_get_gaps_interval_filter(client):
    """?interval=4h returns only 4h rows."""
    _seed_gap(interval="4h")
    _seed_gap(interval="1h", gap_start=datetime(2024, 1, 2, 8, tzinfo=timezone.utc))

    resp = client.get("/api/data/gaps", params={"interval": "4h"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["interval"] == "4h"


# ---------------------------------------------------------------------------
# Test 6: response shape matches DataGapResponse
# ---------------------------------------------------------------------------


def test_get_gaps_response_shape(client):
    """Response includes all DataGapResponse fields."""
    _seed_gap()

    resp = client.get("/api/data/gaps")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]

    required_fields = [
        "gap_id", "market", "symbol", "interval",
        "gap_start", "gap_end", "missing_bars",
        "detected_at", "healed_at",
    ]
    for field in required_fields:
        assert field in row, f"Missing field: {field}"

    assert row["market"] == "crypto_perp"
    assert row["symbol"] == "BTCUSDT"
    assert row["interval"] == "4h"
    assert row["missing_bars"] == 2
    assert row["healed_at"] is None


# ---------------------------------------------------------------------------
# Test 7: empty table returns [] with 200
# ---------------------------------------------------------------------------


def test_get_gaps_empty_table_returns_empty_list(client):
    """No data_gaps rows => 200 with empty list."""
    resp = client.get("/api/data/gaps")
    assert resp.status_code == 200
    assert resp.json() == []
