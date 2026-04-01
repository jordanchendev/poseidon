"""Tests for strategy performance aggregation endpoint."""

import uuid
from datetime import datetime

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
from poseidon.models.backtest import BacktestRecord  # noqa: E402,F401
from poseidon.models.strategy import StrategyRecord  # noqa: E402,F401

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from poseidon.api.strategies import router as strategies_router  # noqa: E402

# --------------- Test DB setup (SQLite in-memory, shared connection) ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(strategies_router, prefix="/api/strategies", tags=["strategies"])


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

PREFIX = "/api/strategies"


# --------------- Helper ---------------

def _seed_backtest(
    db,
    strategy_id: uuid.UUID,
    composite_score: float,
    completed_at: datetime,
    sharpe_ratio: float = 1.0,
    win_rate: float = 0.6,
    total_pnl: float = 5000.0,
) -> uuid.UUID:
    """Seed a completed BacktestRecord and return its id."""
    bt_id = uuid.uuid4()
    record = BacktestRecord(
        id=bt_id,
        strategy_id=strategy_id,
        strategy_type="voting",
        symbol="BTCUSDT",
        market="crypto",
        interval="1d",
        config={},
        metrics={
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "trade_count": 42,
            "max_drawdown": 0.15,
            "composite_score": composite_score,
        },
        status="completed",
        completed_at=completed_at,
    )
    db.add(record)
    return bt_id


# --------------- Tests ---------------


def test_get_performance_with_backtests():
    """GET /strategies/{id}/performance returns best + latest with 2 backtests."""
    strategy_id = uuid.uuid4()
    db = TestingSessionLocal()

    # Older backtest with higher composite score (best)
    _seed_backtest(
        db,
        strategy_id=strategy_id,
        composite_score=0.9,
        completed_at=datetime(2025, 1, 1),
        sharpe_ratio=2.0,
        total_pnl=10000.0,
    )
    # Newer backtest with lower composite score (latest)
    _seed_backtest(
        db,
        strategy_id=strategy_id,
        composite_score=0.7,
        completed_at=datetime(2025, 2, 1),
        sharpe_ratio=1.2,
        total_pnl=3000.0,
    )
    db.commit()
    db.close()

    resp = client.get(f"{PREFIX}/{strategy_id}/performance")
    assert resp.status_code == 200
    data = resp.json()

    assert data["strategy_id"] == str(strategy_id)
    assert data["backtest_count"] == 2

    # Best = highest composite_score (0.9)
    assert data["best"]["composite_score"] == 0.9
    assert data["best"]["sharpe_ratio"] == 2.0
    assert data["best"]["total_pnl"] == 10000.0

    # Latest = most recent completed_at (2025-02-01)
    assert data["latest"]["composite_score"] == 0.7
    assert data["latest"]["sharpe_ratio"] == 1.2


def test_get_performance_no_backtests():
    """GET /strategies/{non_existent}/performance returns 200 with empty results."""
    random_id = uuid.uuid4()
    resp = client.get(f"{PREFIX}/{random_id}/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy_id"] == str(random_id)
    assert data["backtest_count"] == 0
    assert data["best"] is None
    assert data["latest"] is None


def test_get_performance_single_backtest():
    """With one backtest, best == latest."""
    strategy_id = uuid.uuid4()
    db = TestingSessionLocal()
    bt_id = _seed_backtest(
        db,
        strategy_id=strategy_id,
        composite_score=0.85,
        completed_at=datetime(2025, 3, 1),
    )
    db.commit()
    db.close()

    resp = client.get(f"{PREFIX}/{strategy_id}/performance")
    assert resp.status_code == 200
    data = resp.json()

    assert data["backtest_count"] == 1
    assert data["best"]["backtest_id"] == str(bt_id)
    assert data["latest"]["backtest_id"] == str(bt_id)
    assert data["best"]["composite_score"] == 0.85
