"""Tests for the enhanced health check endpoint.

After Phase 54 Redis isolation, health.py uses ``get_redis("celery")`` from
``poseidon.core.redis`` instead of the old ``redis.from_url(settings.redis_url)``.
Tests mock ``poseidon.core.redis.get_redis`` to inject a fake Redis instance.

After Phase 55-02 session unification, health.py uses ``db_session()`` context
manager from ``poseidon.core.database`` instead of direct ``SessionLocal()``.

After Phase 61-02 Thalassa health integration, health.py uses ``httpx.get`` to
probe Thalassa connectivity. Tests mock ``poseidon.api.health.httpx`` for
isolation.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api.health import router as health_router

# --------------- Test app setup ---------------

_test_app = FastAPI()
_test_app.include_router(health_router, tags=["health"])

client = TestClient(_test_app)


# --------------- Helpers ---------------


def _mock_db_session(mock_db):
    """Create a mock context manager that yields mock_db for db_session()."""
    @contextmanager
    def _fake_db_session():
        yield mock_db
    return _fake_db_session


# --------------- Tests ---------------


@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.db_session")
def test_health_lightweight_skips_celery_inspect(mock_db_session_fn, mock_get_redis, mock_celery):
    """Default /health stays lightweight for Docker liveness checks."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_health_returns_200(mock_db_session_fn, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_health_response_structure(mock_db_session_fn, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_health_all_ok(mock_db_session_fn, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_health_degraded_on_db_error(mock_db_session_fn, mock_get_redis, mock_celery):
    # db_session() itself raises (simulating connection failure)
    mock_db_session_fn.side_effect = Exception("connection refused")

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
@patch("poseidon.api.health.db_session")
def test_health_degraded_on_redis_error(mock_db_session_fn, mock_get_redis, mock_celery):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_gpu_worker_ping(mock_db_session_fn, mock_get_redis, mock_celery):
    """GPU component reports available when GPU workers respond to ping."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_gpu_worker_unavailable(mock_db_session_fn, mock_get_redis, mock_celery):
    """GPU component reports unavailable when no GPU workers respond."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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
@patch("poseidon.api.health.db_session")
def test_health_data_freshness_null_when_no_data(
    mock_db_session_fn, mock_get_redis, mock_celery
):
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

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


# --------------- Thalassa connectivity tests (Phase 61-02) ---------------


@patch("poseidon.api.health.httpx")
@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.db_session")
def test_health_includes_thalassa_ok(
    mock_db_session_fn, mock_get_redis, mock_celery, mock_httpx
):
    """Thalassa component shows ok with latency when reachable."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    # Mock httpx.get to return a successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_httpx.get.return_value = mock_response

    resp = client.get("/health")
    data = resp.json()

    assert resp.status_code == 200
    assert data["status"] == "ok"
    thalassa = data["components"]["thalassa"]
    assert isinstance(thalassa, dict)
    assert thalassa["status"] == "ok"
    assert "latency_ms" in thalassa
    assert isinstance(thalassa["latency_ms"], (int, float))


@patch("poseidon.api.health.httpx")
@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.db_session")
def test_health_thalassa_unreachable_degrades(
    mock_db_session_fn, mock_get_redis, mock_celery, mock_httpx
):
    """Overall status degrades when Thalassa is unreachable."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    # Mock httpx.get to raise ConnectError
    mock_httpx.get.side_effect = httpx.ConnectError("Connection refused")

    resp = client.get("/health")
    data = resp.json()

    assert resp.status_code == 200
    assert data["status"] == "degraded"
    assert isinstance(data["components"]["thalassa"], str)
    assert data["components"]["thalassa"].startswith("error")


@patch("poseidon.api.health.httpx")
@patch("poseidon.api.health.celery_app")
@patch("poseidon.core.redis.get_redis")
@patch("poseidon.api.health.db_session")
def test_health_thalassa_timeout_degrades(
    mock_db_session_fn, mock_get_redis, mock_celery, mock_httpx
):
    """Overall status degrades when Thalassa times out."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar.return_value = None
    mock_db_session_fn.side_effect = _mock_db_session(mock_db)

    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_get_redis.return_value = mock_redis_instance

    # Mock httpx.get to raise ReadTimeout
    mock_httpx.get.side_effect = httpx.ReadTimeout("Thalassa timeout")

    resp = client.get("/health")
    data = resp.json()

    assert resp.status_code == 200
    assert data["status"] == "degraded"
    assert isinstance(data["components"]["thalassa"], str)
    assert data["components"]["thalassa"].startswith("error")
