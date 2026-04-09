"""Tests for the Phase 39 backfill API contract (POST/GET/cancel) on /api/data.

Covers the locked Phase 39 decisions in
.planning/phases/39-backfill-api-coverage/39-CONTEXT.md (D-01..D-05):

- POST /api/data/backfill creates a durable BackfillJob row and returns 202
- GET  /api/data/backfill/{job_id} returns full job detail from Postgres
- POST /api/data/backfill/{job_id}/cancel marks cancelled without deleting
- Request schema requires explicit symbols[] and intervals[]
- GET /api/data/backfill/status compatibility list endpoint still works
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility shims for Postgres-only types ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.backfill import BackfillJob  # noqa: E402,F401

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.api.data import router as data_router  # noqa: E402


# --- Test DB / app wiring --------------------------------------------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(data_router, prefix="/api/data", tags=["data"])


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


@pytest.fixture
def client():
    return TestClient(_test_app)


@pytest.fixture(autouse=True)
def _stub_celery_dispatch(monkeypatch):
    """Prevent the unit suite from touching a real broker.

    The POST handler calls ``backfill_chunk.delay(str(job_id))``; we stub the
    delay method so it records invocations but does not require Redis.
    """
    calls: list[tuple[str, ...]] = []

    def fake_delay(*args, **kwargs):  # noqa: ANN001, ANN002
        calls.append(args)

        class _R:
            id = "stub-task-id"

        return _R()

    from poseidon.workers import backfill_tasks

    monkeypatch.setattr(backfill_tasks.backfill_chunk, "delay", fake_delay)
    return calls


_VALID_PAYLOAD = {
    "market": "crypto_perp",
    "symbols": ["BTCUSDT"],
    "intervals": ["4h"],
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-12-31T00:00:00Z",
}


# --- Tests -----------------------------------------------------------------


def test_post_backfill_creates_job(client, _stub_celery_dispatch):
    """POST /api/data/backfill creates a durable BackfillJob row and returns 202."""
    resp = client.post("/api/data/backfill", json=_VALID_PAYLOAD)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in ("pending", "queued")

    # Durable row must exist in Postgres (per D-04)
    session = TestingSessionLocal()
    try:
        row = session.query(BackfillJob).filter_by(job_id=body["job_id"]).one()
        assert row.market == "crypto_perp"
        assert row.symbols == ["BTCUSDT"]
        assert row.intervals == ["4h"]
        assert row.status == "pending"
    finally:
        session.close()

    # Celery dispatch was invoked with the new job_id
    assert _stub_celery_dispatch, "backfill_chunk.delay should have been called"
    assert _stub_celery_dispatch[0][0] == body["job_id"]


def test_get_backfill_job_returns_detail(client):
    """GET /api/data/backfill/{job_id} returns the full detail response."""
    create = client.post("/api/data/backfill", json=_VALID_PAYLOAD)
    assert create.status_code == 202
    job_id = create.json()["job_id"]

    resp = client.get(f"/api/data/backfill/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in (
        "job_id",
        "status",
        "cursor",
        "progress",
        "error",
        "created_at",
        "updated_at",
    ):
        assert key in body, f"missing key {key} in {body}"
    assert body["job_id"] == job_id


def test_cancel_backfill_job_marks_cancelled(client):
    """POST /api/data/backfill/{job_id}/cancel flips status to cancelled."""
    create = client.post("/api/data/backfill", json=_VALID_PAYLOAD)
    job_id = create.json()["job_id"]

    resp = client.post(f"/api/data/backfill/{job_id}/cancel")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "cancelled"

    # GET reflects cancellation and state is preserved (not deleted)
    detail = client.get(f"/api/data/backfill/{job_id}").json()
    assert detail["status"] == "cancelled"
    assert detail["cursor"] is not None
    assert detail["progress"] is not None


def test_post_backfill_requires_explicit_symbols_and_intervals(client):
    """Omitting symbols or intervals must fail validation (no silent YAML fallback)."""
    no_symbols = dict(_VALID_PAYLOAD)
    no_symbols.pop("symbols")
    resp = client.post("/api/data/backfill", json=no_symbols)
    assert resp.status_code == 422

    no_intervals = dict(_VALID_PAYLOAD)
    no_intervals.pop("intervals")
    resp = client.post("/api/data/backfill", json=no_intervals)
    assert resp.status_code == 422


def test_status_endpoint_compatibility_list_still_works(client):
    """GET /api/data/backfill/status still returns a list response."""
    # Empty list is valid — dashboard just needs the shape/route to exist.
    resp = client.get("/api/data/backfill/status")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # After creating a job it should show up in the list.
    client.post("/api/data/backfill", json=_VALID_PAYLOAD)
    resp = client.get("/api/data/backfill/status")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list)
    assert len(rows) >= 1
