"""Phase 90 / Wave 4b — RL Execution REST API tests (Plan 90-05b).

Mirrors ``poseidon/tests/unit/test_research_api.py:1-83`` SQLite shim +
send_task stub harness with the router import + endpoint paths swapped.

Covers:
* T-90-03 — Pydantic Literal allowlist rejection (`test_invalid_algo_rejected`)
* send_task dispatch contract (`test_create_run_dispatches_to_qlib_queue`)
* CRUD + cancel lifecycle (4 tests)
* T-90-04 path-traversal mitigation indirectly via UUID-only path type.
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

from poseidon.api import rl_execution  # noqa: E402
from poseidon.api.rl_execution import router as rl_router  # noqa: E402  (grep contract)
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.rl_execution_run import RLExecutionRun  # noqa: E402

# --- Test DB / app wiring (SQLite in-memory + StaticPool) ----------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(rl_router, prefix="/research/rl-execution", tags=["rl-execution"])


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

    monkeypatch.setattr(rl_execution.celery_app, "send_task", fake_send_task)
    return calls


@pytest.fixture
def client():
    return TestClient(_test_app)


@pytest.fixture
def celery_calls(_stub_celery):
    """Expose recorded send_task calls for assertions."""
    return _stub_celery


# --- Helpers --------------------------------------------------------------


def _insert_run(status: str = "pending", algos: list[str] | None = None) -> str:
    """Insert an RLExecutionRun row directly (bypassing API) and return run_id."""
    db = TestingSessionLocal()
    try:
        run = RLExecutionRun(
            algos=algos or ["twap"],
            mode="full",
            status=status,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return str(run.run_id)
    finally:
        db.close()


# --- Tests ---------------------------------------------------------------


def test_create_run_dispatches_to_qlib_queue(client, celery_calls):
    """POST /run → 202, inserts row, dispatches to poseidon_qlib queue."""
    payload = {"algos": ["twap"], "notional_per_leg": 1_000_000}
    resp = client.post("/research/rl-execution/run", json=payload)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["algos"] == ["twap"]
    # send_task stub recorded one call with the canonical task name + queue.
    assert len(celery_calls) == 1
    assert celery_calls[0]["task"] == "poseidon.workers.qlib_tasks.rl_execute"
    assert celery_calls[0]["queue"] == "poseidon_qlib"
    assert celery_calls[0]["args"] == [body["run_id"]]


def test_get_run_returns_row(client):
    """GET /runs/{id} returns 200 + matching JSON for an existing row."""
    run_id = _insert_run(status="running", algos=["twap", "vwap"])
    resp = client.get(f"/research/rl-execution/runs/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["status"] == "running"
    assert body["algos"] == ["twap", "vwap"]


def test_list_runs_paginated(client):
    """GET /runs returns paginated list."""
    _insert_run(status="pending")
    _insert_run(status="succeeded")
    resp = client.get("/research/rl-execution/runs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert len(body["runs"]) == 2


def test_cancel_pending_run(client):
    """POST /runs/{id}/cancel sets status='cancelled' for pending row."""
    run_id = _insert_run(status="pending")
    resp = client.post(f"/research/rl-execution/runs/{run_id}/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "cancelled"
    # Verify DB reflects the change.
    db = TestingSessionLocal()
    try:
        run = db.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
        assert run.status == "cancelled"
    finally:
        db.close()


def test_cancel_succeeded_run_409(client):
    """POST /runs/{id}/cancel on terminal-status row returns 409."""
    run_id = _insert_run(status="succeeded")
    resp = client.post(f"/research/rl-execution/runs/{run_id}/cancel")
    assert resp.status_code == 409, resp.text


def test_get_results_409_until_succeeded(client):
    """GET /runs/{id}/results on a non-succeeded run returns 409."""
    run_id = _insert_run(status="running")
    resp = client.get(f"/research/rl-execution/runs/{run_id}/results")
    assert resp.status_code == 409, resp.text


def test_invalid_algo_rejected(client):
    """T-90-03 — Pydantic Literal allowlist rejects unknown algos with 422."""
    payload = {"algos": ["invalid_algo"], "notional_per_leg": 1000}
    resp = client.post("/research/rl-execution/run", json=payload)
    assert resp.status_code == 422, resp.text


def test_run_not_found_404(client):
    """GET /runs/{id} for non-existent UUID returns 404."""
    bogus = str(uuid.uuid4())
    resp = client.get(f"/research/rl-execution/runs/{bogus}")
    assert resp.status_code == 404, resp.text
