"""Tests for Phase 46 model discovery and ML param bounds."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


from poseidon.backtest.voting_strategy_factory import PARAM_BOUNDS, get_param_bounds  # noqa: E402
from poseidon.ml.manager import ModelManager  # noqa: E402
from poseidon.models.base import Base  # noqa: E402
from poseidon.models.model_version import ModelVersion  # noqa: E402


ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


def setup_function():
    Base.metadata.create_all(bind=ENGINE)


def teardown_function():
    Base.metadata.drop_all(bind=ENGINE)


def _seed_model(
    session,
    *,
    name: str,
    status: str,
    created_at: datetime,
    version: int = 1,
) -> ModelVersion:
    model = ModelVersion(
        id=uuid.uuid4(),
        name=name,
        version=version,
        status=status,
        params={},
        feature_list=[],
        created_at=created_at,
    )
    session.add(model)
    session.commit()
    return model


def test_list_ready_models_filters_status_market_and_orders_latest_first():
    session = SessionLocal()
    manager = ModelManager(session)
    base_time = datetime(2026, 4, 11, tzinfo=timezone.utc)

    older = _seed_model(
        session,
        name="qlib_lgbm_tw_stock",
        status="ready",
        created_at=base_time,
        version=1,
    )
    newer = _seed_model(
        session,
        name="qlib_xgb_tw_stock",
        status="active",
        created_at=base_time + timedelta(hours=1),
        version=2,
    )
    _seed_model(
        session,
        name="qlib_lgbm_tw_stock",
        status="training",
        created_at=base_time + timedelta(hours=2),
        version=3,
    )
    _seed_model(
        session,
        name="qlib_lgbm_crypto_spot",
        status="ready",
        created_at=base_time + timedelta(hours=3),
        version=1,
    )

    models = manager.list_ready_models("tw_stock")

    assert [model.id for model in models] == [newer.id, older.id]
    assert all(model.status in {"ready", "active"} for model in models)
    assert all("tw_stock" in model.name for model in models)


def test_list_ready_models_returns_empty_when_no_matching_models():
    session = SessionLocal()
    manager = ModelManager(session)

    _seed_model(
        session,
        name="qlib_lgbm_us_stock",
        status="ready",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
    )

    assert manager.list_ready_models("tw_stock") == []


def test_list_ready_models_does_not_return_other_market_models():
    session = SessionLocal()
    manager = ModelManager(session)

    _seed_model(
        session,
        name="qlib_lgbm_tw_stock",
        status="active",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
    )
    crypto_model = _seed_model(
        session,
        name="qlib_catboost_crypto_spot",
        status="ready",
        created_at=datetime(2026, 4, 11, 1, tzinfo=timezone.utc),
    )

    models = manager.list_ready_models("crypto_spot")

    assert [model.id for model in models] == [crypto_model.id]
    assert all("tw_stock" not in model.name for model in models)


def test_param_bounds_contains_qlib_model_enabled():
    assert PARAM_BOUNDS["qlib_model_enabled"] == (0, 1, "int")


def test_get_param_bounds_includes_qlib_model_enabled():
    bounds = get_param_bounds("tw_stock")

    assert bounds["qlib_model_enabled"] == (0, 1, "int")
