"""Tests for GET /api/data/ohlcv endpoint (API-01)."""

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
from poseidon.models.ohlcv import OHLCV  # noqa: E402,F401 – registers table

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


def _seed_ohlcv(session):
    """Seed 5 OHLCV rows for BTC/crypto/1d."""
    for i in range(5):
        row = OHLCV(
            time=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            market="crypto",
            instrument="spot",
            interval="1d",
            open=40000 + i * 100,
            high=40500 + i * 100,
            low=39500 + i * 100,
            close=40200 + i * 100,
            volume=1000 + i * 10,
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
    _seed_ohlcv(session)
    session.close()
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def client():
    return TestClient(_test_app)


def test_get_ohlcv_returns_data(client):
    """GET /api/data/ohlcv with valid params returns OHLCV data."""
    resp = client.get("/api/data/ohlcv", params={"symbol": "BTCUSDT", "market": "crypto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["market"] == "crypto"
    assert body["interval"] == "1d"
    assert body["count"] == 5
    assert len(body["data"]) == 5
    # Verify fields present
    point = body["data"][0]
    assert "time" in point
    assert "open" in point
    assert "close" in point
    assert "volume" in point


def test_get_ohlcv_empty(client):
    """Non-existent symbol returns 200 with empty data and count=0."""
    resp = client.get("/api/data/ohlcv", params={"symbol": "NONEXIST", "market": "crypto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["data"] == []


def test_get_ohlcv_date_range(client):
    """start/end filters narrow down results."""
    resp = client.get(
        "/api/data/ohlcv",
        params={
            "symbol": "BTCUSDT",
            "market": "crypto",
            "start": "2026-01-02T00:00:00",
            "end": "2026-01-04T00:00:00",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3  # Jan 2, 3, 4


def test_get_ohlcv_requires_params(client):
    """Missing required symbol/market returns 422."""
    resp = client.get("/api/data/ohlcv")
    assert resp.status_code == 422
