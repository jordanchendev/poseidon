"""Tests for GET /api/data/funding-rates endpoint (API-06)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility: register before any model import ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.funding_rate import FundingRateRecord  # noqa: E402,F401 – registers table

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient
from poseidon.api.data import router as data_router

# --------------- Test DB setup (SQLite in-memory) ---------------

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


def _seed_funding_rates(session):
    """Seed 10 funding rate rows for BTC/USDT:USDT."""
    for i in range(10):
        day = 1 + (i * 8) // 24
        hour = (i * 8) % 24
        row = FundingRateRecord(
            time=datetime(2026, 1, day, hour, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            funding_rate=0.0001 * (i + 1),
            mark_price=40000.0 + i * 50,
            index_price=39990.0 + i * 50,
        )
        session.add(row)
    session.commit()


# --------------- Tests ---------------

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables and seed data before each test, drop after."""
    Base.metadata.create_all(bind=_engine)
    session = TestingSessionLocal()
    _seed_funding_rates(session)
    session.close()
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def client():
    return TestClient(_test_app)


def test_get_funding_rates_returns_data(client):
    """GET /api/data/funding-rates with valid symbol returns funding data."""
    resp = client.get("/api/data/funding-rates", params={"symbol": "BTC/USDT:USDT"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 10
    assert len(body["data"]) == 10
    # Verify fields
    point = body["data"][0]
    assert "time" in point
    assert "funding_rate" in point
    assert "mark_price" in point
    assert point["symbol"] == "BTC/USDT:USDT"


def test_get_funding_rates_empty(client):
    """Non-existent symbol returns 200 with empty data."""
    resp = client.get("/api/data/funding-rates", params={"symbol": "NONEXIST"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["data"] == []


def test_get_funding_rates_limit(client):
    """limit param restricts the number of returned records."""
    resp = client.get("/api/data/funding-rates", params={"symbol": "BTC/USDT:USDT", "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert len(body["data"]) == 3
