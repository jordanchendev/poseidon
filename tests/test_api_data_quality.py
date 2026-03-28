"""Tests for data quality API endpoint -- provider health monitoring."""

from unittest.mock import patch

import fakeredis
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fake_redis_client():
    """Fakeredis client for mocking Redis connections."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=False)
    yield client
    client.flushall()


@pytest.fixture
def client(fake_redis_client):
    """FastAPI TestClient with mocked Redis and API key."""
    with (
        patch("poseidon.api.data_quality._get_redis_client", return_value=fake_redis_client),
        patch("poseidon.core.config.settings.api_key", "test-key"),
    ):
        from poseidon.main import app

        c = TestClient(app)
        yield c


def test_import_router():
    """Test that the data_quality router can be imported."""
    from poseidon.api.data_quality import router

    assert router is not None


def test_provider_health_returns_all_providers(client):
    """GET /api/data-quality/providers returns all 3 providers with expected keys."""
    response = client.get("/api/data-quality/providers", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    data = response.json()
    assert "providers" in data

    providers = data["providers"]
    assert "finmind" in providers
    assert "yfinance" in providers
    assert "ccxt" in providers

    # Each provider should have the expected fields
    for name in ("finmind", "yfinance", "ccxt"):
        p = providers[name]
        assert "circuit_state" in p
        assert "failure_count" in p
        assert "quota_used" in p
        assert "quota_limit" in p
        assert "window_seconds" in p


def test_provider_health_default_circuit_state(client):
    """With fresh Redis, all providers should have CLOSED circuit state."""
    response = client.get("/api/data-quality/providers", headers={"X-API-Key": "test-key"})
    data = response.json()

    for name in ("finmind", "yfinance", "ccxt"):
        assert data["providers"][name]["circuit_state"] == "CLOSED"
        assert data["providers"][name]["failure_count"] == 0
        assert data["providers"][name]["quota_used"] == 0


def test_provider_health_requires_auth(client):
    """Endpoint should require API key authentication."""
    response = client.get("/api/data-quality/providers")
    assert response.status_code in (401, 403)
