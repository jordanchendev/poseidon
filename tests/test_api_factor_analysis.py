"""Tests for the Phase 47 factor analysis API endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
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


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.api.factor_analysis import router as factor_analysis_router  # noqa: E402
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.factor_analysis_run import FactorAnalysisRun  # noqa: E402
from poseidon.models.model_version import ModelVersion  # noqa: E402
from poseidon.models.training_run import TrainingRun  # noqa: E402

ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

APP = FastAPI()
APP.include_router(
    factor_analysis_router,
    prefix="/api/v1/factor-analysis",
    tags=["factor-analysis"],
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


APP.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def _setup_db():
    Base.metadata.create_all(bind=ENGINE)
    yield
    Base.metadata.drop_all(bind=ENGINE)


@pytest.fixture(autouse=True)
def _stub_celery(monkeypatch):
    calls: list[dict] = []

    def fake_send_task(task_name, args=None, kwargs=None, queue=None, **extra):
        calls.append({"task": task_name, "args": args, "queue": queue})

    import poseidon.api.factor_analysis as factor_analysis_module

    monkeypatch.setattr(factor_analysis_module.celery_app, "send_task", fake_send_task)
    return calls


@pytest.fixture
def client():
    return TestClient(APP)


@pytest.fixture
def celery_calls(_stub_celery):
    return _stub_celery


def _seed_model_version(session: sessionmaker) -> ModelVersion:
    model_version = ModelVersion(
        id=uuid.uuid4(),
        name="qlib_lgbm_tw_stock",
        version=1,
        status="ready",
        params={},
        feature_list=["rsi_14"],
        artifact_path="/tmp/model-artifacts",
        created_at=datetime(2026, 4, 11, tzinfo=UTC),
    )
    session.add(model_version)
    session.commit()
    return model_version


def _seed_training_run(session: sessionmaker, *, model_version_id: uuid.UUID) -> TrainingRun:
    training_run = TrainingRun(
        run_id=uuid.uuid4(),
        handler_class="Alpha158Handler",
        handler_params={},
        model_class="LGBModel",
        model_params={},
        market="tw_stock",
        symbols=["2330"],
        interval="1d",
        segments={"train": ["2025-01-01", "2025-06-30"]},
        status="succeeded",
        model_version_id=model_version_id,
        requested_by="api",
    )
    session.add(training_run)
    session.commit()
    return training_run


def test_ic_trigger_creates_pending_run_and_dispatches_cpu_queue(client, celery_calls):
    payload = {
        "market": "tw_stock",
        "symbols": ["2330"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-31",
        "horizons": [1, 5],
        "features": ["rsi_14"],
        "interval": "1d",
    }

    response = client.post("/api/v1/factor-analysis/ic", json=payload)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert len(celery_calls) == 1
    assert celery_calls[0]["task"] == "poseidon.workers.cpu_tasks.factor_ic_analysis"
    assert celery_calls[0]["args"] == [body["id"]]
    assert celery_calls[0]["queue"] == "poseidon_cpu"


def test_shapley_trigger_rejects_invalid_uuid(client):
    response = client.post(
        "/api/v1/factor-analysis/shapley",
        json={"model_version_id": "not-a-uuid"},
    )

    assert response.status_code == 422, response.text


def test_shapley_trigger_creates_pending_run_and_dispatches_qlib_queue(client, celery_calls):
    session = TestingSessionLocal()
    try:
        model_version = _seed_model_version(session)
        _seed_training_run(session, model_version_id=model_version.id)
        model_version_id = str(model_version.id)
    finally:
        session.close()

    response = client.post(
        "/api/v1/factor-analysis/shapley",
        json={"model_version_id": model_version_id, "max_samples": 250},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert len(celery_calls) == 1
    assert celery_calls[0]["task"] == "poseidon.workers.qlib_tasks.factor_shapley_analysis"
    assert celery_calls[0]["args"] == [body["id"]]
    assert celery_calls[0]["queue"] == "poseidon_qlib"


def test_centrality_trigger_creates_pending_run_and_dispatches_cpu_queue(client, celery_calls):
    payload = {
        "market": "tw_stock",
        "sub_signals": [{"label": "rsi", "type": "feature_above", "column": "rsi_14", "threshold": 55}],
        "symbols": ["2330"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-31",
        "interval": "1d",
        "distance_threshold": 0.7,
    }

    response = client.post("/api/v1/factor-analysis/centrality", json=payload)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert len(celery_calls) == 1
    assert celery_calls[0]["task"] == "poseidon.workers.cpu_tasks.factor_centrality_analysis"
    assert celery_calls[0]["args"] == [body["id"]]
    assert celery_calls[0]["queue"] == "poseidon_cpu"


def test_run_list_and_detail_endpoints_return_serialized_runs(client):
    session = TestingSessionLocal()
    try:
        run = FactorAnalysisRun(
            id=uuid.uuid4(),
            run_type="ic",
            config_json={"market": "tw_stock"},
            results_json={"features": {"rsi_14": {"1": 0.12}}},
            status="succeeded",
            market="tw_stock",
            created_at=datetime(2026, 4, 11, tzinfo=UTC),
            updated_at=datetime(2026, 4, 11, 1, tzinfo=UTC),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = str(run.id)
    finally:
        session.close()

    list_response = client.get("/api/v1/factor-analysis/runs?market=tw_stock")

    assert list_response.status_code == 200, list_response.text
    list_body = list_response.json()
    assert list_body["total"] == 1
    assert list_body["runs"][0]["id"] == run_id
    assert list_body["runs"][0]["status"] == "succeeded"

    detail_response = client.get(f"/api/v1/factor-analysis/runs/{run_id}")

    assert detail_response.status_code == 200, detail_response.text
    detail_body = detail_response.json()
    assert detail_body["id"] == run_id
    assert detail_body["results_json"] == {"features": {"rsi_14": {"1": 0.12}}}
