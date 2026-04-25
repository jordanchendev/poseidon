"""Integration tests for risk metrics API endpoints (RAPI-01 through RAPI-04)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import msgpack
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create TestClient with API key auth bypassed."""
    from poseidon.api.auth import verify_api_key
    from poseidon.main import app

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- RAPI-01: GET /api/risk/var ---


def test_var_endpoint_returns_snapshots(client):
    """GET /api/risk/var returns VaR snapshots from Redis cache."""
    sample = {
        "method": "parametric",
        "var_95": 0.025,
        "var_99": 0.041,
        "cvar_95": 0.035,
        "cvar_99": 0.058,
        "portfolio_value": 100000.0,
        "holding_period": 1,
        "as_of": "2026-03-28T00:00:00",
        "computed_at": "2026-03-28T01:00:00",
    }
    packed = msgpack.packb(sample, use_bin_type=True)

    mock_redis = MagicMock()

    def side_effect(key):
        if key == "poseidon:var:latest:parametric":
            return packed
        return None

    mock_redis.get.side_effect = side_effect

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/var")

    assert resp.status_code == 200
    data = resp.json()
    assert "snapshots" in data
    assert len(data["snapshots"]) == 1
    assert data["snapshots"][0]["method"] == "parametric"
    assert data["snapshots"][0]["var_95"] == 0.025


def test_var_endpoint_empty_cache(client):
    """GET /api/risk/var returns empty list when no cached data."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/var")

    assert resp.status_code == 200
    assert resp.json()["snapshots"] == []


# --- RAPI-02: GET /api/risk/exposure ---


def test_exposure_endpoint(client):
    """GET /api/risk/exposure returns portfolio exposure breakdown."""
    mock_portfolio = MagicMock()
    mock_portfolio.positions = {
        "crypto_spot:BTCUSDT": MagicMock(market="crypto_spot", quantity_pct=0.3),
        "us_stock:AAPL": MagicMock(market="us_stock", quantity_pct=0.5),
    }
    mock_portfolio.total_exposure.return_value = 0.8

    with (
        patch("poseidon.core.database.SessionLocal") as mock_session_cls,
        patch("poseidon.risk.portfolio.VirtualPortfolio", return_value=mock_portfolio),
    ):
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db
        resp = client.get("/api/risk/exposure")

    assert resp.status_code == 200
    data = resp.json()
    assert "exposures" in data
    assert data["total"] == 0.8
    categories = {e["category"] for e in data["exposures"]}
    assert "crypto_spot" in categories
    assert "us_stock" in categories


# --- RAPI-03: POST /api/risk/stress-test/run ---


def test_stress_test_trigger(client):
    """POST /api/risk/stress-test/run returns task_id."""
    mock_task = MagicMock()
    mock_task.id = "abc-123-task"

    with patch("poseidon.workers.cpu_tasks.run_stress_test") as mock_fn:
        mock_fn.delay.return_value = mock_task
        resp = client.post(
            "/api/risk/stress-test/run",
            json={"scenario_name": "covid_2020"},
        )

    assert resp.status_code == 200
    assert resp.json()["task_id"] == "abc-123-task"
    mock_fn.delay.assert_called_once_with("covid_2020", None)


def test_stress_test_status_pending(client):
    """GET /api/risk/stress-test/{task_id} with PENDING state."""
    mock_result = MagicMock()
    mock_result.state = "PENDING"

    with patch("poseidon.api.risk_metrics.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = mock_result
        resp = client.get("/api/risk/stress-test/some-task-id")

    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    assert resp.json()["result"] is None


def test_stress_test_status_completed(client):
    """GET /api/risk/stress-test/{task_id} with SUCCESS state."""
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {"scenario": "covid_2020", "portfolio_impact": -0.15}

    with patch("poseidon.api.risk_metrics.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = mock_result
        resp = client.get("/api/risk/stress-test/completed-task")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["scenario"] == "covid_2020"


# --- RAPI-04: GET /api/risk/alerts ---


def test_alerts_endpoint(client):
    """GET /api/risk/alerts returns alert history from Redis Stream."""
    alert_data = {
        "event_type": "drawdown_breach",
        "level": "WARNING",
        "drawdown_pct": 0.06,
    }
    raw_entries = [
        (b"1711600000000-0", {b"data": json.dumps(alert_data).encode()}),
    ]

    mock_redis = MagicMock()
    mock_redis.xrevrange.return_value = raw_entries
    mock_redis.xlen.return_value = 1

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/alerts")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["alerts"]) == 1
    assert data["alerts"][0]["event_type"] == "drawdown_breach"
    assert data["total"] == 1


def test_alerts_pagination(client):
    """GET /api/risk/alerts?limit=10&offset=5 respects pagination."""
    # Create 15 fake entries
    entries = []
    for i in range(15):
        alert = {"event_type": "drawdown_breach", "level": "WARNING", "i": i}
        entries.append((f"171160000{i:04d}-0".encode(), {b"data": json.dumps(alert).encode()}))

    mock_redis = MagicMock()
    mock_redis.xrevrange.return_value = entries
    mock_redis.xlen.return_value = 100  # total in stream

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/alerts?limit=10&offset=5")

    assert resp.status_code == 200
    data = resp.json()
    # With 15 entries returned, offset=5, limit=10 -> entries[5:15] = 10
    assert len(data["alerts"]) == 10
    assert data["total"] == 100
