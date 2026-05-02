"""Phase 90 / Wave 4 — RL Execution REST API tests.

Wave 0 scaffold (Plan 90-01 Task 3). Wave 4 implements
``poseidon.api.rl_execution`` (a Celery dispatch endpoint that enqueues
jobs onto qlib_queue) — until then ``rl_router`` does not exist, and the
module top uses ``pytest.importorskip`` to defer the failure.

Harness shape mirrors ``poseidon/tests/unit/test_research_api.py:1-83``
(SQLite shim + send_task stub) so Wave 4 can swap in the real router and
its tests without re-doing wiring. The ``rl_router`` identifier appears
literally below to satisfy the plan's grep contract.
"""

from __future__ import annotations

import pytest

# --- SQLite compatibility shims for Postgres-only types ---
from sqlalchemy import create_engine
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


# Defer router import until Wave 4 ships it. Until then this importorskip
# leaves the test collectable but skipped — `pytest --collect-only` still
# discovers the file, satisfying the Wave 0 acceptance contract.
rl_execution = pytest.importorskip(
    "poseidon.api.rl_execution",
    reason="poseidon.api.rl_execution lands in Wave 4",
)
rl_router = rl_execution.router  # name referenced for plan grep contract + used in include_router below

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.models.base import Base, get_db  # noqa: E402

# --- Test DB / app wiring (SQLite in-memory + StaticPool) ----------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(rl_router, prefix="/api/v1/rl-execution", tags=["rl-execution"])


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


def test_create_run_dispatches_to_qlib_queue():
    """POST /api/v1/rl-execution/runs → enqueues onto qlib_queue."""
    pytest.skip("Wave 4 implements")
