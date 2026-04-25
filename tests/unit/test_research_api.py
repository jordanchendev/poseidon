"""Tests for the Phase 41 Research API endpoints (plan 41-02).

Covers RESEARCH-API-01 (POST /train), RESEARCH-API-02 (allowlist reject),
RESEARCH-API-04 (GET /runs, GET /runs/{run_id}), RESEARCH-API-05 (cancel).

Uses SQLite in-memory DB with type shims and a standalone FastAPI app
mounting the research_api router -- avoids importing the full main.py
which pulls heavy dependencies (torch etc.) not available on local Mac.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine

# --- SQLite compatibility shims for Postgres-only types ---
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.api.research_api import router as research_router  # noqa: E402
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.training_run import TrainingRun  # noqa: E402

# --- Test DB / app wiring --------------------------------------------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(research_router, prefix="/api/v1/models", tags=["research"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def _setup_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture(autouse=True)
def _stub_celery(monkeypatch):
    """Stub celery_app.send_task to prevent broker calls and record invocations."""
    calls: list[dict] = []

    def fake_send_task(task_name, args=None, kwargs=None, queue=None, **extra):
        calls.append({"task": task_name, "args": args, "queue": queue})

    import poseidon.api.research_api as _mod

    monkeypatch.setattr(_mod.celery_app, "send_task", fake_send_task)
    return calls


@pytest.fixture
def client():
    return TestClient(_test_app)


@pytest.fixture
def celery_calls(_stub_celery):
    """Expose recorded send_task calls for assertions."""
    return _stub_celery


# --- Valid payload for POST /train ----------------------------------------

_VALID_TRAIN_PAYLOAD = {
    "handler_class": "Alpha158Handler",
    "model_class": "LGBModel",
    "market": "crypto_perp",
    "symbols": ["BTCUSDT"],
    "interval": "4h",
    "segments": {
        "train": ["2023-01-01", "2024-06-30"],
        "valid": ["2024-07-01", "2024-12-31"],
        "test": ["2025-01-01", "2025-06-30"],
    },
}


# --- Tests: POST /train ---------------------------------------------------


def test_train_endpoint(client, celery_calls):
    """POST /api/v1/models/train returns 202 with run_id and status=pending (RESEARCH-API-01)."""
    resp = client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "pending"
    assert body["handler_class"] == "Alpha158Handler"
    assert body["model_class"] == "LGBModel"
    assert body["market"] == "crypto_perp"


def test_train_allowlist_reject(client):
    """POST with unknown handler_class returns 422 (RESEARCH-API-02)."""
    payload = {**_VALID_TRAIN_PAYLOAD, "handler_class": "EvilHandler"}
    resp = client.post("/api/v1/models/train", json=payload)
    assert resp.status_code == 422, resp.text
    assert "EvilHandler" in resp.json()["detail"]


def test_train_unknown_model_reject(client):
    """POST with unknown model_class returns 422 (RESEARCH-API-02)."""
    payload = {**_VALID_TRAIN_PAYLOAD, "model_class": "MaliciousModel"}
    resp = client.post("/api/v1/models/train", json=payload)
    assert resp.status_code == 422, resp.text
    assert "MaliciousModel" in resp.json()["detail"]


def test_train_dispatches_to_qlib_queue(client, celery_calls):
    """POST /train dispatches via send_task with correct task name and queue."""
    resp = client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    assert len(celery_calls) == 1
    call = celery_calls[0]
    assert call["task"] == "poseidon.workers.qlib_tasks.qlib_train"
    assert call["args"] == [run_id]
    assert call["queue"] == "qlib_queue"


# --- Tests: GET /runs ------------------------------------------------------


def test_get_runs_empty(client):
    """GET /runs returns empty list with total=0 when no runs exist (RESEARCH-API-04)."""
    resp = client.get("/api/v1/models/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["runs"] == []
    assert body["total"] == 0
    assert body["limit"] == 50
    assert body["offset"] == 0


def test_get_runs_with_data(client, celery_calls):
    """Create a run via POST, then GET /runs returns it."""
    client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)

    resp = client.get("/api/v1/models/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert len(body["runs"]) == 1
    assert body["runs"][0]["handler_class"] == "Alpha158Handler"


def test_get_runs_pagination(client, celery_calls):
    """GET /runs?limit=1&offset=0 returns exactly 1 item when multiple exist."""
    client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)
    payload2 = {**_VALID_TRAIN_PAYLOAD, "model_class": "LinearModel"}
    client.post("/api/v1/models/train", json=payload2)

    resp = client.get("/api/v1/models/runs?limit=1&offset=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2  # total is full count
    assert len(body["runs"]) == 1  # but only 1 returned
    assert body["limit"] == 1
    assert body["offset"] == 0


def test_get_runs_filter_by_status(client, celery_calls):
    """GET /runs?status=pending returns only pending runs."""
    client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)

    resp = client.get("/api/v1/models/runs?status=pending")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["runs"][0]["status"] == "pending"

    resp2 = client.get("/api/v1/models/runs?status=running")
    assert resp2.status_code == 200
    assert resp2.json()["total"] == 0


# --- Tests: GET /runs/{run_id} --------------------------------------------


def test_get_run_detail(client, celery_calls):
    """POST a run, GET /runs/{run_id} returns full detail (RESEARCH-API-04)."""
    resp = client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)
    run_id = resp.json()["run_id"]

    detail_resp = client.get(f"/api/v1/models/runs/{run_id}")
    assert detail_resp.status_code == 200
    body = detail_resp.json()
    assert body["run_id"] == run_id
    assert body["handler_class"] == "Alpha158Handler"
    assert body["model_class"] == "LGBModel"
    assert body["market"] == "crypto_perp"
    assert body["symbols"] == ["BTCUSDT"]
    assert body["interval"] == "4h"
    assert body["status"] == "pending"
    assert body["requested_by"] == "api"
    assert "created_at" in body
    assert "updated_at" in body


def test_get_run_not_found(client):
    """GET /runs/{random_uuid} returns 404."""
    random_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/models/runs/{random_id}")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


# --- Tests: POST /runs/{run_id}/cancel ------------------------------------


def test_cancel_pending_run(client, celery_calls):
    """POST /runs/{run_id}/cancel on pending run returns cancelled (RESEARCH-API-05)."""
    resp = client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)
    run_id = resp.json()["run_id"]

    cancel_resp = client.post(f"/api/v1/models/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200
    body = cancel_resp.json()
    assert body["run_id"] == run_id
    assert body["status"] == "cancelled"

    # Verify the status is persisted
    detail_resp = client.get(f"/api/v1/models/runs/{run_id}")
    assert detail_resp.json()["status"] == "cancelled"


def test_cancel_already_succeeded_run(client, celery_calls):
    """POST /cancel on a succeeded run returns 409 (RESEARCH-API-05)."""
    resp = client.post("/api/v1/models/train", json=_VALID_TRAIN_PAYLOAD)
    run_id = resp.json()["run_id"]

    # Manually flip status to 'succeeded' in DB
    db = TestingSessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.run_id == uuid.UUID(run_id)).one()
        run.status = "succeeded"
        db.commit()
    finally:
        db.close()

    cancel_resp = client.post(f"/api/v1/models/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 409
    assert "Cannot cancel" in cancel_resp.json()["detail"]


def test_cancel_nonexistent_run(client):
    """POST /cancel on a nonexistent run returns 404."""
    random_id = str(uuid.uuid4())
    resp = client.post(f"/api/v1/models/runs/{random_id}/cancel")
    assert resp.status_code == 404
