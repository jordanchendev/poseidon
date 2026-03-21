"""Tests verifying consistent error response format across all endpoints.

Uses the real FastAPI ``app`` from ``poseidon.main`` so that registered
exception handlers are exercised.  The DB dependency is overridden
with an in-memory SQLite session.
"""

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


# --- Patch settings before importing app so verify_api_key uses our test key ---
from poseidon.core.config import settings  # noqa: E402

VALID_API_KEY = "test-api-key-for-auth-tests"
settings.api_key = VALID_API_KEY

# --- Now import models and app ---
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.strategy import StrategyRecord  # noqa: E402,F401
from poseidon.models.signal import SignalRecord  # noqa: E402,F401
from poseidon.models.risk_rule import RiskRuleRecord  # noqa: E402,F401
from poseidon.models.virtual_position import VirtualPositionRecord  # noqa: E402,F401

from fastapi.testclient import TestClient  # noqa: E402
from poseidon.main import app  # noqa: E402

# --------------- Test DB setup ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


API_KEY_HEADER = {"X-API-Key": VALID_API_KEY}


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after.

    Re-applies the dependency override and API key on every test to
    guard against other test modules overwriting them on the shared
    ``app`` and ``settings`` singletons.
    """
    settings.api_key = VALID_API_KEY
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


client = TestClient(app, raise_server_exceptions=False)


# --------------- Tests ---------------


def test_validation_error_format():
    """POST with invalid body returns consistent validation error format."""
    # Missing required fields (name, strategy_type, symbol, market)
    resp = client.post(
        "/strategies",
        json={},
        headers=API_KEY_HEADER,
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["status_code"] == 422
    assert "detail" in body
    assert "errors" in body
    assert isinstance(body["errors"], list)


def test_not_found_error_format():
    """GET a non-existent resource returns consistent 404 format."""
    random_id = str(uuid.uuid4())
    resp = client.get(
        f"/strategies/{random_id}",
        headers=API_KEY_HEADER,
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error_code"] == "HTTP_404"
    assert body["status_code"] == 404
    assert "detail" in body


def test_missing_api_key_error_format():
    """Request without API key returns consistent 401 format."""
    resp = client.get("/strategies")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error_code"] == "HTTP_401"
    assert body["status_code"] == 401
    assert "detail" in body


def test_invalid_api_key_error_format():
    """Request with invalid API key returns consistent 401 format."""
    resp = client.get(
        "/strategies",
        headers={"X-API-Key": "invalid-key"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["error_code"] == "HTTP_401"
    assert body["status_code"] == 401
    assert body["detail"] == "Invalid API key"


def test_error_response_content_type():
    """All error responses return application/json content type."""
    # 401 - missing key
    resp = client.get("/strategies")
    assert "application/json" in resp.headers.get("content-type", "")

    # 404 - not found
    random_id = str(uuid.uuid4())
    resp = client.get(
        f"/strategies/{random_id}",
        headers=API_KEY_HEADER,
    )
    assert "application/json" in resp.headers.get("content-type", "")

    # 422 - validation error
    resp = client.post("/strategies", json={}, headers=API_KEY_HEADER)
    assert "application/json" in resp.headers.get("content-type", "")


def test_risk_rules_not_found_error_format():
    """Risk router also follows consistent error format."""
    random_id = str(uuid.uuid4())
    resp = client.get(
        f"/api/risk-rules/{random_id}",
        headers=API_KEY_HEADER,
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error_code"] == "HTTP_404"
    assert body["status_code"] == 404
    assert "detail" in body
