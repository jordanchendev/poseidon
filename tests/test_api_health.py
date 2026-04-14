"""Tests for the enhanced health check endpoint.

After Phase 54 Redis isolation, health.py uses ``get_redis("celery")`` from
``poseidon.core.redis`` instead of the old ``redis.from_url(settings.redis_url)``.
Tests mock ``poseidon.core.redis.get_redis`` to inject a fake Redis instance.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api.health import router as health_router

# --------------- Test app setup ---------------

_test_app = FastAPI()
_test_app.include_router(health_router, tags=["health"])

client = TestClient(_test_app)


# --------------- Helpers ---------------


def _mock_all_healthy():
    """Return context managers that mock all external dependencies as healthy."""
    # Mock DB session
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None  # no OHLCV data

    mock_session_local = MagicMock(return_value=mock_db)

    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True

    # Mock Celery inspect
    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {"worker1": []}
    mock_inspect.reserved.return_value = {"worker1": []}

    mock_celery = MagicMock()
    mock_celery.control.inspect.return_value = mock_inspect

    return mock_session_local, mock_redis_instance, mock_celery


# --------------- Tests ---------------


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_lightweight_skips_celery_inspect(mock_session_local, mock_get_redis, mock_celery):
    """Default /health stays lightweight for Docker liveness checks."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    resp = client.get("/health")
    data = resp.json()

    assert resp.status_code == 200
    assert data["status"] == "ok"
    assert data["components"]["database"] == "ok"
    assert data["components"]["redis"] == "ok"
    assert data["components"]["celery"]["status"] == "skipped"
    assert data["components"]["gpu"]["status"] == "skipped"
    mock_celery.control.inspect.assert_not_called()


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_returns_200(mock_session_local, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {}
    mock_inspect.reserved.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspect

    resp = client.get("/health?details=true")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_response_structure(mock_session_local, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {}
    mock_inspect.reserved.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspect

    resp = client.get("/health?details=true")
    data = resp.json()

    assert "components" in data
    components = data["components"]
    assert "database" in components
    assert "redis" in components
    assert "celery" in components
    assert "gpu" in components
    assert "data_freshness" in components


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_all_ok(mock_session_local, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {"worker1": []}
    mock_inspect.reserved.return_value = {"worker1": []}
    mock_inspect.active_queues.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspect

    resp = client.get("/health?details=true")
    data = resp.json()

    assert data["status"] == "ok"
    assert data["components"]["database"] == "ok"
    assert data["components"]["redis"] == "ok"
    assert isinstance(data["components"]["celery"], dict)
    assert "active_tasks" in data["components"]["celery"]
    assert "reserved_tasks" in data["components"]["celery"]


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_degraded_on_db_error(mock_session_local, mock_get_redis, mock_celery):
    mock_session_local.return_value.execute.side_effect = Exception("connection refused")

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {}
    mock_inspect.reserved.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspect

    resp = client.get("/health?details=true")
    data = resp.json()

    assert data["status"] == "degraded"
    assert data["components"]["database"].startswith("error:")


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_degraded_on_redis_error(mock_session_local, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_get_redis.side_effect = Exception("redis down")

    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {}
    mock_inspect.reserved.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspect

    resp = client.get("/health?details=true")
    data = resp.json()

    assert data["status"] == "degraded"
    assert data["components"]["redis"].startswith("error:")


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_gpu_worker_ping(mock_session_local, mock_get_redis, mock_celery):
    """GPU component reports available when GPU workers respond to ping."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    # Celery inspect for queue lengths
    mock_inspect_queue = MagicMock()
    mock_inspect_queue.active.return_value = {}
    mock_inspect_queue.reserved.return_value = {}

    # GPU inspect returns queue membership for GPU workers
    mock_inspect_gpu = MagicMock()
    mock_inspect_gpu.active_queues.return_value = {
        "celery@gpu-worker": [{"name": "gpu"}],
    }

    # First call is for queue lengths (timeout=2.0), second for GPU (timeout=3.0)
    mock_celery.control.inspect.side_effect = [mock_inspect_queue, mock_inspect_gpu]

    resp = client.get("/health?details=true")
    data = resp.json()

    gpu = data["components"]["gpu"]
    assert gpu["available"] is True
    assert "celery@gpu-worker" in gpu["workers"]


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_gpu_worker_unavailable(mock_session_local, mock_get_redis, mock_celery):
    """GPU component reports unavailable when no GPU workers respond."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    # Queue inspect
    mock_inspect_queue = MagicMock()
    mock_inspect_queue.active.return_value = {}
    mock_inspect_queue.reserved.return_value = {}

    # GPU inspect returns empty (no GPU workers)
    mock_inspect_gpu = MagicMock()
    mock_inspect_gpu.active_queues.return_value = {}

    mock_celery.control.inspect.side_effect = [mock_inspect_queue, mock_inspect_gpu]

    resp = client.get("/health?details=true")
    data = resp.json()

    gpu = data["components"]["gpu"]
    assert gpu["available"] is False


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.SessionLocal")
def test_health_data_freshness_null_when_no_data(
    mock_session_local, mock_get_redis, mock_celery
):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_session_local.return_value = mock_db

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    mock_inspect = MagicMock()
    mock_inspect.active.return_value = {}
    mock_inspect.reserved.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspect

    resp = client.get("/health?details=true")
    data = resp.json()

    assert data["components"]["data_freshness"]["latest_ohlcv"] is None
