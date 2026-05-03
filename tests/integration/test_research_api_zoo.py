"""Phase 94 Wave 3 — Research API dispatch smoke for the expanded model zoo.

Validates D-15: POST /api/v1/models/train with each new model_class
(LocalformerModel, TRAModel, ALSTM) returns 202 + run_id and dispatches
exactly one Celery task to the ``poseidon_qlib`` queue. Per D-16 / OQ-3,
this is dispatch-ONLY — does NOT wait for training. The standalone
fit/predict on toy data is covered by tests/test_zoo_smoke.py (Wave 2).

Pattern source (Pattern S3):
- poseidon/tests/unit/test_research_api.py (Phase 41 anchor — SQLite shim +
  TestClient + send_task stub)
- poseidon/tests/test_rl_execution_api.py (Phase 90 Wave 4b parallel pattern)

Verified at execute time against ``src/poseidon/api/research_api.py``:
- Train route: ``/api/v1/models/train`` (router @post(``/train``) + main.py
  prefix ``/api/v1/models``)
- Task name: ``poseidon.workers.qlib_tasks.qlib_train``
- Queue: ``POSEIDON_QLIB_QUEUE = "poseidon_qlib"`` (from
  ``poseidon.workers.celery_app``)
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason=(
        "stormtrooper-only integration — set STORMTROOPER=1 inside qlib-research "
        "container; mirrors the gate on tests/test_zoo_smoke.py for a uniform "
        "phase-suite execution surface (VALIDATION.md §Test Infrastructure)"
    ),
)

# --- SQLite shim + FastAPI TestClient + send_task stub (Pattern S3) -----------
# Verbatim from poseidon/tests/unit/test_research_api.py (Phase 41 anchor).

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB  # noqa: E402
from sqlalchemy.dialects.postgresql import UUID as PG_UUID  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402


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
from poseidon.models.training_run import TrainingRun  # noqa: E402,F401  (registers ORM)

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
_test_app = FastAPI()
# Mount prefix verified against poseidon/src/poseidon/main.py:74 —
# ``app.include_router(research_api.router, prefix="/api/v1/models", ...)``.
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


@pytest.fixture
def celery_calls(monkeypatch):
    """Stub celery_app.send_task; capture dispatch calls for assertion."""
    calls: list[dict] = []

    def fake_send_task(task_name, args=None, kwargs=None, queue=None, **extra):
        calls.append({"task": task_name, "args": args, "queue": queue})

    import poseidon.api.research_api as _mod

    monkeypatch.setattr(_mod.celery_app, "send_task", fake_send_task)
    return calls


@pytest.fixture
def client():
    return TestClient(_test_app)


# --- Phase 94 dispatch tests (D-15 / D-16) ------------------------------------

# Payload shape aligned to TrainRequest schema (poseidon/src/poseidon/core/schemas.py:196)
# and the existing Phase 41 anchor _VALID_TRAIN_PAYLOAD shape.
_PHASE94_BASE_PAYLOAD = {
    "handler_class": "Alpha158Handler",
    "model_class": "LGBModel",  # overridden per-test
    "handler_params": {},
    "model_params": {},
    "market": "tw_stock",
    "symbols": ["2330"],
    "interval": "1d",
    "segments": {
        "train": ["2024-01-01", "2024-06-30"],
        "valid": ["2024-07-01", "2024-09-30"],
        "test": ["2024-10-01", "2024-12-31"],
    },
}

# Verified from src/poseidon/api/research_api.py:51-94 + workers/celery_app.py:10:
_TRAIN_ROUTE = "/api/v1/models/train"
_EXPECTED_TASK_NAME = "poseidon.workers.qlib_tasks.qlib_train"
_EXPECTED_QUEUE = "poseidon_qlib"  # POSEIDON_QLIB_QUEUE constant


def _post_train(client, celery_calls, model_class):
    """Helper: build payload, POST, verify 202 + dispatch contract."""
    payload = {**_PHASE94_BASE_PAYLOAD, "model_class": model_class}
    resp = client.post(_TRAIN_ROUTE, json=payload)
    # Surface response body on failure: a 422 here means allowlist did NOT
    # include the model_class (D-15 contract violation / Wave 1 regression).
    assert resp.status_code == 202, (
        f"Expected 202 for model_class={model_class}, got {resp.status_code}; body={resp.text}"
    )
    body = resp.json()
    assert "run_id" in body, f"response missing run_id: {body}"

    # D-16: dispatch-only — assert exactly one Celery task was sent.
    assert len(celery_calls) == 1, f"Expected 1 send_task call, got {len(celery_calls)}: {celery_calls}"
    call = celery_calls[0]
    assert call["task"] == _EXPECTED_TASK_NAME, call
    assert call["queue"] == _EXPECTED_QUEUE, call
    assert call["args"] == [body["run_id"]], call

    return resp, body["run_id"]


def test_dispatch_localformer(client, celery_calls):
    """D-15: POST /train with model_class='LocalformerModel' returns 202 + dispatches."""
    _post_train(client, celery_calls, "LocalformerModel")


def test_dispatch_tra(client, celery_calls):
    """D-15 + OQ-3: POST /train with model_class='TRAModel' returns 202 (dispatch-only).

    Per D-16, the actual TRA fit (which requires MTSDatasetH) is covered by
    tests/test_zoo_smoke.py::test_tra_smoke. This integration test stops at
    the dispatch boundary — it proves the allowlist accepts TRAModel without
    a 422.
    """
    _post_train(client, celery_calls, "TRAModel")


def test_dispatch_alstm(client, celery_calls):
    """D-15: POST /train with model_class='ALSTM' returns 202 + dispatches."""
    _post_train(client, celery_calls, "ALSTM")
