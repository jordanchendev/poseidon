"""Tests for strategy CRUD API endpoints."""

import uuid

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
from poseidon.models.strategy import StrategyRecord  # noqa: E402,F401

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from poseidon.api.strategies import router as strategies_router  # noqa: E402

# --------------- Test DB setup (SQLite in-memory, shared connection) ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

# Create a local test app with just the strategies router (not yet mounted in main.py)
_test_app = FastAPI()
_test_app.include_router(strategies_router, prefix="/api/strategies", tags=["strategies"])


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

PREFIX = "/api/strategies"


# --------------- Helper ---------------


def _create_strategy(
    name: str = "test-strategy",
    strategy_type: str = "rule",
    config: dict | None = None,
    symbol: str = "2330",
    market: str = "tw_stock",
    interval: str = "1d",
    model_version_id: str | None = None,
):
    """POST a strategy and return the response."""
    body = {
        "name": name,
        "strategy_type": strategy_type,
        "config": config or {},
        "symbol": symbol,
        "market": market,
        "interval": interval,
    }
    if model_version_id:
        body["model_version_id"] = model_version_id
    return client.post(PREFIX, json=body)


# --------------- Tests ---------------


def test_list_strategies_empty():
    resp = client.get(PREFIX)
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_strategy():
    resp = _create_strategy(
        name="My Rule Strategy",
        strategy_type="rule",
        config={"rules": [{"condition": {"op": "gt", "field": "close", "value": 100}}]},
        symbol="BTCUSDT",
        market="crypto_spot",
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Rule Strategy"
    assert data["strategy_type"] == "rule"
    assert data["symbol"] == "BTCUSDT"
    assert data["market"] == "crypto_spot"
    assert data["interval"] == "1d"
    assert data["active"] is False
    assert data["model_version_id"] is None
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_create_strategy_invalid_type():
    resp = _create_strategy(strategy_type="invalid", name="bad-type")
    assert resp.status_code == 400
    assert "Invalid strategy_type" in resp.json()["detail"]


def test_create_strategy_duplicate_name():
    resp1 = _create_strategy(name="dup-strategy")
    assert resp1.status_code == 201
    resp2 = _create_strategy(name="dup-strategy")
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"]


def test_get_strategy():
    create_resp = _create_strategy(name="get-test")
    assert create_resp.status_code == 201
    strategy_id = create_resp.json()["id"]

    resp = client.get(f"{PREFIX}/{strategy_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == strategy_id
    assert resp.json()["name"] == "get-test"


def test_get_strategy_not_found():
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_update_strategy():
    create_resp = _create_strategy(name="update-test", symbol="2330", market="tw_stock")
    assert create_resp.status_code == 201
    strategy_id = create_resp.json()["id"]

    resp = client.put(
        f"{PREFIX}/{strategy_id}",
        json={"symbol": "BTCUSDT", "market": "crypto_spot", "config": {"updated": True}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["market"] == "crypto_spot"
    assert data["config"] == {"updated": True}
    # Unchanged fields preserved
    assert data["name"] == "update-test"
    assert data["strategy_type"] == "rule"


def test_update_strategy_not_found():
    random_id = str(uuid.uuid4())
    resp = client.put(f"{PREFIX}/{random_id}", json={"symbol": "ETHUSDT"})
    assert resp.status_code == 404


def test_delete_strategy():
    create_resp = _create_strategy(name="delete-test")
    assert create_resp.status_code == 201
    strategy_id = create_resp.json()["id"]

    resp = client.delete(f"{PREFIX}/{strategy_id}")
    assert resp.status_code == 204

    # Verify it's gone
    get_resp = client.get(f"{PREFIX}/{strategy_id}")
    assert get_resp.status_code == 404


def test_delete_strategy_not_found():
    random_id = str(uuid.uuid4())
    resp = client.delete(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_activate_strategy():
    create_resp = _create_strategy(name="activate-test")
    assert create_resp.status_code == 201
    strategy_id = create_resp.json()["id"]
    assert create_resp.json()["active"] is False

    resp = client.post(f"{PREFIX}/{strategy_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["active"] is True


def test_deactivate_strategy():
    create_resp = _create_strategy(name="deactivate-test")
    assert create_resp.status_code == 201
    strategy_id = create_resp.json()["id"]

    # Activate first
    client.post(f"{PREFIX}/{strategy_id}/activate")

    # Then deactivate
    resp = client.post(f"{PREFIX}/{strategy_id}/deactivate")
    assert resp.status_code == 200
    assert resp.json()["active"] is False


def test_activate_not_found():
    random_id = str(uuid.uuid4())
    resp = client.post(f"{PREFIX}/{random_id}/activate")
    assert resp.status_code == 404


def test_deactivate_not_found():
    random_id = str(uuid.uuid4())
    resp = client.post(f"{PREFIX}/{random_id}/deactivate")
    assert resp.status_code == 404


def test_list_strategies_returns_created():
    _create_strategy(name="strat-1", symbol="2330", market="tw_stock")
    _create_strategy(name="strat-2", symbol="BTCUSDT", market="crypto_spot")

    resp = client.get(PREFIX)
    assert resp.status_code == 200
    strategies = resp.json()
    assert len(strategies) == 2
    names = [s["name"] for s in strategies]
    assert "strat-1" in names
    assert "strat-2" in names
