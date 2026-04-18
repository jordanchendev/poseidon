"""Tests for backtest API endpoints."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
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
from poseidon.models.backtest import (  # noqa: E402,F401
    BacktestEquityRecord,
    BacktestRecord,
    BacktestTradeRecord,
)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from poseidon.api.backtests import router as backtests_router  # noqa: E402

# --------------- Test DB setup (SQLite in-memory, shared connection) ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

# Create a local test app with just the backtests router
_test_app = FastAPI()
_test_app.include_router(backtests_router, prefix="/api/backtest", tags=["backtest"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


client = TestClient(_test_app)

PREFIX = "/api/backtest"


# --------------- Tests ---------------


@patch("poseidon.api.backtests.run_backtest_task")
def test_post_run_returns_202(mock_task):
    """POST /backtest/run dispatches a Celery task and returns 202 with task_id."""
    mock_async_result = MagicMock()
    mock_async_result.id = "celery-task-id-123"
    mock_task.delay.return_value = mock_async_result

    strategy_id = str(uuid.uuid4())
    resp = client.post(
        f"{PREFIX}/run",
        json={"strategy_id": strategy_id, "initial_capital": 500_000.0},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "celery-task-id-123"
    assert "Backtest dispatched" in data["message"]

    mock_task.delay.assert_called_once_with(
        strategy_id,
        None,
        None,
        500_000.0,
        "fixed_notional",
        None,
        None,
        False,
    )


@patch("poseidon.api.backtests.run_optimization_task")
def test_post_optimize_returns_202(mock_task):
    """POST /backtest/optimize dispatches an optimization task and returns 202."""
    mock_async_result = MagicMock()
    mock_async_result.id = "celery-opt-id-456"
    mock_task.delay.return_value = mock_async_result

    strategy_id = str(uuid.uuid4())
    resp = client.post(
        f"{PREFIX}/optimize",
        json={
            "strategy_id": strategy_id,
            "param_grid": {"period": [10, 20, 30]},
            "method": "grid",
            "n_trials": 50,
            "target_metric": "sharpe_ratio",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"] == "celery-opt-id-456"
    assert "Optimization dispatched" in data["message"]

    mock_task.delay.assert_called_once_with(
        strategy_id,
        {"period": [10, 20, 30]},
        "grid",
        50,
        "sharpe_ratio",
        None,
        None,
    )


def test_list_backtests_empty():
    """GET /backtest returns 200 with empty list when no backtests exist."""
    resp = client.get(PREFIX)
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_backtest_not_found():
    """GET /backtest/{id} returns 404 for non-existent UUID."""
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}")
    assert resp.status_code == 404


def test_get_backtest_returns_record():
    """GET /backtest/{id} returns the backtest record when it exists."""
    # Seed a backtest record directly into the DB
    bt_id = uuid.uuid4()
    db = TestingSessionLocal()
    record = BacktestRecord(
        id=bt_id,
        strategy_id=uuid.uuid4(),
        strategy_type="rule",
        symbol="2330",
        market="tw_stock",
        interval="1d",
        config={"test": True},
        metrics={"sharpe_ratio": 1.5},
        status="completed",
    )
    db.add(record)
    db.commit()
    db.close()

    resp = client.get(f"{PREFIX}/{bt_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(bt_id)
    assert data["strategy_type"] == "rule"
    assert data["symbol"] == "2330"
    assert data["metrics"]["sharpe_ratio"] == 1.5
    assert data["status"] == "completed"


def test_list_backtests_returns_records():
    """GET /backtest returns seeded records."""
    db = TestingSessionLocal()
    for i in range(3):
        record = BacktestRecord(
            id=uuid.uuid4(),
            strategy_type="rule",
            symbol=f"SYM{i}",
            market="tw_stock",
            interval="1d",
            config={},
            status="completed",
        )
        db.add(record)
    db.commit()
    db.close()

    resp = client.get(PREFIX)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3


def test_list_backtests_with_limit():
    """GET /backtest?limit=1 respects the limit parameter."""
    db = TestingSessionLocal()
    for i in range(3):
        record = BacktestRecord(
            id=uuid.uuid4(),
            strategy_type="rule",
            symbol=f"SYM{i}",
            market="tw_stock",
            interval="1d",
            config={},
            status="completed",
        )
        db.add(record)
    db.commit()
    db.close()

    resp = client.get(f"{PREFIX}?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


# --------------- Trade endpoint tests ---------------


def test_get_backtest_trades_returns_list():
    """GET /backtest/{id}/trades returns trade list for a completed backtest."""
    bt_id = uuid.uuid4()
    db = TestingSessionLocal()
    db.add(BacktestRecord(
        id=bt_id,
        strategy_type="voting",
        symbol="BTCUSDT",
        market="crypto",
        interval="1d",
        config={},
        status="completed",
    ))
    from datetime import datetime as dt

    db.add(BacktestTradeRecord(
        id=uuid.uuid4(),
        backtest_id=bt_id,
        symbol="BTCUSDT",
        action="buy",
        entry_time=dt(2025, 1, 1),
        exit_time=dt(2025, 1, 5),
        entry_price=40000.0,
        exit_price=42000.0,
        quantity=0.1,
        pnl=200.0,
        fees=10.0,
        metadata_={},
    ))
    db.add(BacktestTradeRecord(
        id=uuid.uuid4(),
        backtest_id=bt_id,
        symbol="BTCUSDT",
        action="sell",
        entry_time=dt(2025, 1, 10),
        exit_time=dt(2025, 1, 15),
        entry_price=43000.0,
        exit_price=41000.0,
        quantity=0.1,
        pnl=-200.0,
        fees=10.0,
        metadata_={},
    ))
    db.commit()
    db.close()

    resp = client.get(f"{PREFIX}/{bt_id}/trades")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["backtest_id"] == str(bt_id)
    assert data[0]["symbol"] == "BTCUSDT"


def test_get_backtest_trades_not_found():
    """GET /backtest/{random_uuid}/trades returns 404."""
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}/trades")
    assert resp.status_code == 404


# --------------- Equity curve endpoint tests ---------------


def test_get_backtest_equity_curve_returns_data():
    """GET /backtest/{id}/equity-curve returns equity time-series."""
    bt_id = uuid.uuid4()
    db = TestingSessionLocal()
    db.add(BacktestRecord(
        id=bt_id,
        strategy_type="voting",
        symbol="ETHUSDT",
        market="crypto",
        interval="1d",
        config={},
        status="completed",
    ))
    from datetime import datetime as dt

    for i in range(3):
        db.add(BacktestEquityRecord(
            backtest_id=bt_id,
            time=dt(2025, 1, 1 + i),
            equity=100000.0 + i * 1000,
            drawdown=0.01 * i,
        ))
    db.commit()
    db.close()

    resp = client.get(f"{PREFIX}/{bt_id}/equity-curve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backtest_id"] == str(bt_id)
    assert data["count"] == 3
    assert len(data["data"]) == 3
    assert data["data"][0]["equity"] == 100000.0


def test_get_backtest_equity_curve_not_found():
    """GET /backtest/{random_uuid}/equity-curve returns 404."""
    random_id = str(uuid.uuid4())
    resp = client.get(f"{PREFIX}/{random_id}/equity-curve")
    assert resp.status_code == 404
