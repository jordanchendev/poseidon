"""Tests for signal list/detail API endpoints."""

import uuid
from datetime import datetime, timezone

import pytest
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


# --- Now import models (registers them with Base.metadata) ---
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.signal import SignalRecord  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from poseidon.api.signals import router as signals_router  # noqa: E402

# --------------- Test DB setup (SQLite in-memory, shared connection) ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(signals_router, prefix="/signals", tags=["signals"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


client = TestClient(_test_app)

PREFIX = "/signals"


# --------------- Helper ---------------


def _insert_signal(
    symbol: str = "BTCUSDT",
    market: str = "crypto_spot",
    status: str = "passed",
    action: str = "long",
    confidence: float = 0.85,
    **kwargs,
) -> str:
    """Insert a signal record directly into the DB and return its ID."""
    db = TestingSessionLocal()
    signal_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    record = SignalRecord(
        id=signal_id,
        strategy_id=kwargs.get("strategy_id"),
        model_id=kwargs.get("model_id"),
        symbol=symbol,
        market=market,
        instrument=kwargs.get("instrument", "spot"),
        action=action,
        confidence=confidence,
        quantity_pct=kwargs.get("quantity_pct"),
        signal_time=now,
        valid_until=None,
        interval=kwargs.get("interval", "1d"),
        params=kwargs.get("params", {}),
        status=status,
        reject_reason=kwargs.get("reject_reason"),
        metadata_=kwargs.get("metadata_", {}),
    )
    db.add(record)
    db.commit()
    db.close()
    return str(signal_id)


# --------------- Tests ---------------


def test_list_signals_empty():
    resp = client.get(PREFIX)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_signal_not_found():
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_list_signals_filter_by_market():
    _insert_signal(symbol="BTCUSDT", market="crypto_spot")
    _insert_signal(symbol="2330", market="tw_stock")

    resp = client.get(f"{PREFIX}?market=crypto_spot")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["market"] == "crypto_spot"
    assert data[0]["symbol"] == "BTCUSDT"


def test_list_signals_filter_by_status():
    _insert_signal(status="passed")
    _insert_signal(status="rejected", reject_reason="too risky")

    resp = client.get(f"{PREFIX}?status=passed")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["status"] == "passed"


def test_list_signals_filter_by_symbol():
    _insert_signal(symbol="BTCUSDT", market="crypto_spot")
    _insert_signal(symbol="ETHUSDT", market="crypto_spot")

    resp = client.get(f"{PREFIX}?symbol=ETHUSDT")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "ETHUSDT"


def test_get_signal_detail():
    signal_id = _insert_signal(
        symbol="BTCUSDT",
        market="crypto_spot",
        status="passed",
        confidence=0.92,
    )

    resp = client.get(f"{PREFIX}/{signal_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == signal_id
    assert data["symbol"] == "BTCUSDT"
    assert data["market"] == "crypto_spot"
    assert data["status"] == "passed"
    assert data["confidence"] == 0.92


def test_list_signals_respects_limit():
    for i in range(5):
        _insert_signal(symbol=f"SYM{i}", market="crypto_spot")

    resp = client.get(f"{PREFIX}?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3


def test_list_signals_returns_all():
    _insert_signal(symbol="BTCUSDT", market="crypto_spot", status="passed")
    _insert_signal(symbol="2330", market="tw_stock", status="rejected")

    resp = client.get(PREFIX)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
