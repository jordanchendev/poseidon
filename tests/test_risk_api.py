"""Tests for risk rule CRUD API endpoints and SignalPipeline integration."""

import uuid

import pytest
from sqlalchemy import JSON, create_engine, event, text
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
from poseidon.models.risk_rule import RiskRuleRecord  # noqa: E402,F401
from poseidon.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# --------------- Test DB setup (SQLite in-memory, shared connection) ---------------

# StaticPool + single connection ensures in-memory SQLite state is shared
# across all sessions (fixture vs. dependency override).
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


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


client = TestClient(app)

PREFIX = "/api/risk-rules"


# --------------- Helper ---------------


def _create_rule(
    rule_type: str = "position_limit",
    name: str = "test-rule",
    enabled: bool = True,
    priority: int = 0,
    params: dict | None = None,
):
    """POST a rule and return the response."""
    body = {
        "rule_type": rule_type,
        "name": name,
        "enabled": enabled,
        "priority": priority,
        "params": params or {},
    }
    return client.post(PREFIX, json=body)


# --------------- Tests ---------------


def test_list_rules_empty():
    resp = client.get(PREFIX)
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_rule():
    resp = _create_rule(
        rule_type="position_limit",
        name="Max 5 positions",
        params={"max_positions": 5},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["rule_type"] == "position_limit"
    assert data["name"] == "Max 5 positions"
    assert data["enabled"] is True
    assert data["priority"] == 0
    assert data["params"] == {"max_positions": 5}
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_create_rule_invalid_type():
    resp = _create_rule(rule_type="nonexistent", name="bad-rule")
    assert resp.status_code == 400
    assert "Unknown rule_type" in resp.json()["detail"]


def test_create_rule_duplicate_name():
    resp1 = _create_rule(name="dup-rule")
    assert resp1.status_code == 201
    resp2 = _create_rule(name="dup-rule")
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"]


def test_get_rule():
    create_resp = _create_rule(name="get-test")
    assert create_resp.status_code == 201
    rule_id = create_resp.json()["id"]

    resp = client.get(f"{PREFIX}/{rule_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == rule_id
    assert resp.json()["name"] == "get-test"


def test_get_rule_not_found():
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_update_rule_params():
    create_resp = _create_rule(name="update-params", params={"max_positions": 3})
    assert create_resp.status_code == 201
    rule_id = create_resp.json()["id"]

    resp = client.patch(f"{PREFIX}/{rule_id}", json={"params": {"max_positions": 10}})
    assert resp.status_code == 200
    assert resp.json()["params"] == {"max_positions": 10}
    # Other fields unchanged
    assert resp.json()["enabled"] is True
    assert resp.json()["name"] == "update-params"


def test_update_rule_toggle_enabled():
    create_resp = _create_rule(name="toggle-test", enabled=True)
    assert create_resp.status_code == 201
    rule_id = create_resp.json()["id"]

    resp = client.patch(f"{PREFIX}/{rule_id}", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    # Toggle back
    resp2 = client.patch(f"{PREFIX}/{rule_id}", json={"enabled": True})
    assert resp2.status_code == 200
    assert resp2.json()["enabled"] is True


def test_delete_rule():
    create_resp = _create_rule(name="delete-test")
    assert create_resp.status_code == 201
    rule_id = create_resp.json()["id"]

    resp = client.delete(f"{PREFIX}/{rule_id}")
    assert resp.status_code == 204

    # Verify it's gone
    get_resp = client.get(f"{PREFIX}/{rule_id}")
    assert get_resp.status_code == 404


def test_delete_rule_not_found():
    random_id = str(uuid.uuid4())
    resp = client.delete(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_list_rule_types():
    resp = client.get(f"{PREFIX}/types")
    assert resp.status_code == 200
    types = resp.json()
    assert isinstance(types, list)
    assert len(types) == 5
    assert "position_limit" in types
    assert "loss_limit" in types
    assert "frequency" in types
    assert "confidence_threshold" in types
    assert "leverage_cap" in types


def test_list_rules_ordered_by_priority():
    _create_rule(name="rule-prio-10", priority=10)
    _create_rule(name="rule-prio-1", priority=1, rule_type="loss_limit")
    _create_rule(name="rule-prio-5", priority=5, rule_type="frequency")

    resp = client.get(PREFIX)
    assert resp.status_code == 200
    rules = resp.json()
    assert len(rules) == 3
    assert rules[0]["name"] == "rule-prio-1"
    assert rules[1]["name"] == "rule-prio-5"
    assert rules[2]["name"] == "rule-prio-10"
