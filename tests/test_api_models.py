"""Tests for model lifecycle API endpoints."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility: register before any model import ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


# --- Now import models (registers them with Base.metadata) ---
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.model_version import ModelVersion  # noqa: E402,F401

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from poseidon.api.models import router as models_router  # noqa: E402

# --------------- Test DB setup (SQLite in-memory, shared connection) ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(_engine, "connect")
def _register_sqlite_functions(dbapi_conn, connection_record):
    """Register PostgreSQL-compatible functions for SQLite."""
    dbapi_conn.create_function("gen_random_uuid", 0, lambda: str(uuid.uuid4()))


TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(models_router, prefix="/api/models", tags=["models"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Create tables before each test, drop after.

    Also patches artifact filesystem operations so tests don't touch /data/.
    """
    Base.metadata.create_all(bind=_engine)
    with (
        patch("poseidon.ml.artifacts.get_artifact_dir", return_value=tmp_path),
        patch("poseidon.ml.artifacts.ensure_version_dir", return_value=tmp_path / "mock"),
        patch("poseidon.ml.artifacts.set_active_symlink"),
    ):
        yield
    Base.metadata.drop_all(bind=_engine)


client = TestClient(_test_app)

PREFIX = "/api/models"


# --------------- Helpers ---------------


def _seed_model_version(
    name: str = "test-model",
    version: int = 1,
    status: str = "training",
    params: dict | None = None,
    feature_list: list | None = None,
    metrics: dict | None = None,
    error_message: str | None = None,
) -> str:
    """Insert a ModelVersion directly into the test DB and return its id."""
    db = TestingSessionLocal()
    mv_id = uuid.uuid4()
    mv = ModelVersion(
        id=mv_id,
        name=name,
        version=version,
        status=status,
        params=params or {},
        feature_list=feature_list or [],
        metrics=metrics,
        error_message=error_message,
    )
    db.add(mv)
    db.commit()
    db.close()
    return str(mv_id)


# --------------- Tests ---------------


@patch("poseidon.api.models.train_model")
def test_train_returns_202(mock_train):
    """POST /models/train creates a version and dispatches Celery task."""
    fake_task = MagicMock()
    fake_task.id = "celery-task-id-123"
    mock_train.delay.return_value = fake_task

    # Seed a version directly and mock ModelManager.create_version to return it,
    # since SQLite cannot handle gen_random_uuid() in server_default + RETURNING.
    mv_id = uuid.uuid4()
    fake_mv = MagicMock()
    fake_mv.id = mv_id
    fake_mv.version = 1
    fake_mv.name = "xgboost_tw_stock"

    with patch("poseidon.api.models.ModelManager") as MockMM:
        MockMM.return_value.create_version.return_value = fake_mv

        resp = client.post(
            f"{PREFIX}/train",
            json={
                "model_name": "xgboost_tw_stock",
                "symbol": "2330",
                "market": "tw_stock",
                "interval": "1d",
                "params": {"n_estimators": 100},
                "feature_list": ["rsi_14", "sma_20"],
            },
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "celery-task-id-123"
    assert "Training dispatched" in data["message"]

    # Verify Celery delay was called with the right model name
    mock_train.delay.assert_called_once()
    call_kwargs = mock_train.delay.call_args
    assert call_kwargs.kwargs["model_name"] == "xgboost_tw_stock"
    assert call_kwargs.kwargs["symbol"] == "2330"


@patch("poseidon.api.models.train_model")
def test_train_passes_label_mode_to_celery_params(mock_train):
    """POST /models/train injects label_mode into Celery params."""
    fake_task = MagicMock()
    fake_task.id = "celery-task-id-456"
    mock_train.delay.return_value = fake_task

    mv_id = uuid.uuid4()
    fake_mv = MagicMock()
    fake_mv.id = mv_id
    fake_mv.version = 1
    fake_mv.name = "xgboost_tw_stock"

    with patch("poseidon.api.models.ModelManager") as MockMM:
        MockMM.return_value.create_version.return_value = fake_mv

        resp = client.post(
            f"{PREFIX}/train",
            json={
                "model_name": "xgboost_tw_stock",
                "symbol": "2330",
                "market": "tw_stock",
                "params": {"n_estimators": 100},
                "label_mode": "regime",
            },
        )

    assert resp.status_code == 202
    mock_train.delay.assert_called_once()
    assert mock_train.delay.call_args.kwargs["params"] == {
        "n_estimators": 100,
        "label_mode": "regime",
    }


def test_list_models_empty():
    resp = client.get(PREFIX)
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_models_returns_seeded():
    _seed_model_version(name="model-a", version=1)
    _seed_model_version(name="model-a", version=2)

    resp = client.get(f"{PREFIX}?name=model-a")
    assert resp.status_code == 200
    versions = resp.json()
    assert len(versions) == 2
    assert versions[0]["version"] == 1
    assert versions[1]["version"] == 2


def test_list_all_models_limited():
    """Without name filter, returns newest first, limited to 50."""
    _seed_model_version(name="m1", version=1)
    _seed_model_version(name="m2", version=1)

    resp = client.get(PREFIX)
    assert resp.status_code == 200
    versions = resp.json()
    assert len(versions) == 2


def test_get_model_version():
    mv_id = _seed_model_version(name="get-test", version=1)

    resp = client.get(f"{PREFIX}/{mv_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "get-test"
    assert data["version"] == 1


def test_get_model_version_not_found():
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_shadow_invalid_transition():
    """Cannot shadow a model that is still in 'training' status."""
    mv_id = _seed_model_version(name="shadow-test", version=1, status="training")

    resp = client.post(f"{PREFIX}/{mv_id}/shadow")
    assert resp.status_code == 400
    assert "Cannot transition" in resp.json()["detail"]


def test_shadow_valid_transition():
    """A 'ready' model can be transitioned to 'shadow'."""
    mv_id = _seed_model_version(name="shadow-ok", version=1, status="ready")

    resp = client.post(f"{PREFIX}/{mv_id}/shadow")
    assert resp.status_code == 200
    assert resp.json()["status"] == "shadow"


def test_shadow_not_found():
    random_id = str(uuid.uuid4())
    resp = client.post(f"{PREFIX}/{random_id}/shadow")
    assert resp.status_code == 404


def test_activate_from_ready():
    """A 'ready' model can be activated."""
    mv_id = _seed_model_version(name="activate-test", version=1, status="ready")

    resp = client.post(f"{PREFIX}/{mv_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


def test_activate_from_shadow():
    """A 'shadow' model can be activated."""
    mv_id = _seed_model_version(name="activate-shadow", version=1, status="shadow")

    resp = client.post(f"{PREFIX}/{mv_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


def test_activate_invalid_transition():
    """A 'training' model cannot be activated directly."""
    mv_id = _seed_model_version(name="activate-bad", version=1, status="training")

    resp = client.post(f"{PREFIX}/{mv_id}/activate")
    assert resp.status_code == 400
    assert "Cannot transition" in resp.json()["detail"]


def test_activate_not_found():
    random_id = str(uuid.uuid4())
    resp = client.post(f"{PREFIX}/{random_id}/activate")
    assert resp.status_code == 404


@patch("poseidon.api.models.run_prediction")
def test_predict_dispatch_returns_202(mock_run_prediction):
    """POST /models/{id}/predict dispatches GPU task and returns 202."""
    fake_task = MagicMock()
    fake_task.id = "test-task-id"
    mock_run_prediction.delay.return_value = fake_task

    mv_id = _seed_model_version(name="predict-test", version=1, status="active")

    resp = client.post(
        f"{PREFIX}/{mv_id}/predict",
        json={"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "test-task-id"
    assert "Prediction dispatched" in data["message"]

    mock_run_prediction.delay.assert_called_once()
    call_kwargs = mock_run_prediction.delay.call_args.kwargs
    assert call_kwargs["symbol"] == "BTCUSDT"
    assert call_kwargs["market"] == "crypto_spot"
    assert call_kwargs["interval"] == "1h"


def test_predict_model_not_found_returns_404():
    """Predict endpoint returns 404 for unknown model version."""
    random_id = str(uuid.uuid4())
    resp = client.post(
        f"{PREFIX}/{random_id}/predict",
        json={"symbol": "2330", "market": "tw_stock"},
    )
    assert resp.status_code == 404


@patch("poseidon.api.models.run_prediction")
def test_predict_invalid_status_returns_400(mock_run_prediction):
    """Predict endpoint returns 400 for model with invalid status."""
    mv_id = _seed_model_version(name="predict-bad", version=1, status="training")

    resp = client.post(
        f"{PREFIX}/{mv_id}/predict",
        json={"symbol": "2330", "market": "tw_stock"},
    )
    assert resp.status_code == 400
    assert "cannot run prediction" in resp.json()["detail"]

    mock_run_prediction.delay.assert_not_called()
