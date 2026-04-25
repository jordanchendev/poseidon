"""Tests verifying all secured endpoints reject unauthenticated requests.

Uses the real FastAPI ``app`` from ``poseidon.main`` so that router-level
``dependencies=secured`` is exercised.  The DB dependency is overridden
with an in-memory SQLite session to avoid needing a real database.
"""

import pytest
from sqlalchemy import create_engine

# --- SQLite compatibility: register before any model import ---
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


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
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.main import app  # noqa: E402
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.risk_rule import RiskRuleRecord  # noqa: E402,F401
from poseidon.models.signal import SignalRecord  # noqa: E402,F401
from poseidon.models.strategy import StrategyRecord  # noqa: E402,F401
from poseidon.models.virtual_position import VirtualPositionRecord  # noqa: E402,F401

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

# --------------- Secured endpoints to test ---------------

SECURED_ENDPOINTS = [
    ("GET", "/api/data/backfill/status"),
    ("POST", "/api/data/fetch"),
    ("POST", "/api/sentiment"),
    ("GET", "/api/sentiment?symbol=TEST"),
    ("GET", "/api/risk-rules"),
    ("POST", "/api/risk-rules"),
    ("GET", "/api/risk-rules/types"),
    ("GET", "/api/risk-rules/portfolio"),
    ("GET", "/api/strategies"),
    ("POST", "/api/strategies"),
    ("GET", "/api/models"),
    ("POST", "/api/models/train"),
    ("GET", "/api/backtest"),
    ("POST", "/api/backtest/run"),
    ("GET", "/api/signals"),
]


# --------------- Tests ---------------


@pytest.mark.parametrize("method,path", SECURED_ENDPOINTS)
def test_secured_endpoint_rejects_without_api_key(method, path):
    """All secured endpoints must return 401 when X-API-Key header is missing."""
    resp = client.request(method, path)
    assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"


@pytest.mark.parametrize("method,path", SECURED_ENDPOINTS)
def test_secured_endpoint_rejects_invalid_api_key(method, path):
    """All secured endpoints must return 401 when X-API-Key is wrong."""
    resp = client.request(method, path, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"


def test_health_allows_unauthenticated_access():
    """GET /health must be accessible without any API key."""
    resp = client.get("/health")
    # May return 200 or 503 depending on DB availability, but NOT 401/403
    assert resp.status_code not in (401, 403), f"GET /health returned {resp.status_code}, should not require auth"


def test_secured_endpoint_accepts_valid_api_key():
    """At least one secured endpoint should pass auth with a valid API key.

    We verify the response is NOT 401/403 (auth rejection).  The actual
    status may be 200 or 500 depending on DB state, but that is
    irrelevant -- the point is that the auth layer did not block us.
    """
    resp = client.get(
        "/api/strategies",
        headers={"X-API-Key": VALID_API_KEY},
    )
    assert resp.status_code not in (401, 403), f"Valid API key was rejected with {resp.status_code}"
