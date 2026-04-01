"""Tests for backtest trade/equity persistence via BacktestRepository."""

import uuid
from datetime import datetime, timezone

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


# --- Now import models ---
from poseidon.models.base import Base  # noqa: E402
from poseidon.models.backtest import (  # noqa: E402
    BacktestEquityRecord,
    BacktestRecord,
    BacktestTradeRecord,
)
from poseidon.backtest.portfolio import TradeRecord  # noqa: E402
from poseidon.backtest.repository import BacktestRepository  # noqa: E402
from poseidon.backtest.schemas import BacktestConfig, BacktestResult  # noqa: E402


# --------------- Test DB setup ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(_engine, "connect")
def _register_sqlite_functions(dbapi_conn, connection_record):
    """Register gen_random_uuid() and now() for SQLite compatibility."""
    dbapi_conn.create_function("gen_random_uuid", 0, lambda: str(uuid.uuid4()))
    dbapi_conn.create_function("now", 0, lambda: datetime.now(timezone.utc).isoformat())

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(autouse=True)
def _setup_db():
    """Create all tables before each test and drop after."""
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# --------------- Helpers ---------------

def _make_config() -> BacktestConfig:
    return BacktestConfig(
        strategy_type="rule",
        symbol="BTCUSDT",
        market="crypto_spot",
        interval="1d",
        initial_capital=100_000.0,
    )


def _make_result(backtest_id: uuid.UUID | None = None) -> BacktestResult:
    return BacktestResult(
        backtest_id=backtest_id or uuid.uuid4(),
        config=_make_config(),
        metrics={"sharpe": 1.5, "max_drawdown": -0.12},
        trade_count=2,
        equity_curve_length=10,
        status="completed",
    )


def _make_trades() -> list[TradeRecord]:
    now = datetime.now(timezone.utc)
    return [
        TradeRecord(
            symbol="BTCUSDT",
            action="buy",
            entry_time=now,
            exit_time=None,
            entry_price=50000.0,
            exit_price=None,
            quantity=0.1,
            pnl=None,
            fees=5.0,
            metadata=None,
        ),
        TradeRecord(
            symbol="BTCUSDT",
            action="sell",
            entry_time=now,
            exit_time=now,
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            pnl=100.0,
            fees=5.0,
            metadata={"reason": "take_profit"},
        ),
    ]


def _make_equity_curve() -> list[tuple[datetime, float, float]]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        (base, 100000.0, 0.0),
        (datetime(2026, 1, 2, tzinfo=timezone.utc), 100500.0, 0.0),
        (datetime(2026, 1, 3, tzinfo=timezone.utc), 99800.0, -0.007),
    ]


# --------------- Tests ---------------


def test_save_result_persists_all_tables(db_session):
    """save_result should persist BacktestRecord, trades, and equity."""
    repo = BacktestRepository(db_session)
    config = _make_config()
    result = _make_result()
    trades = _make_trades()
    equity = _make_equity_curve()

    bt_id = repo.save_result(config, result, trades, equity)
    db_session.commit()

    # Verify main record
    record = db_session.query(BacktestRecord).filter_by(id=bt_id).first()
    assert record is not None
    assert record.status == "completed"
    assert record.symbol == "BTCUSDT"

    # Verify trades
    trade_records = db_session.query(BacktestTradeRecord).filter_by(backtest_id=bt_id).all()
    assert len(trade_records) == 2

    # Verify equity
    equity_records = db_session.query(BacktestEquityRecord).filter_by(backtest_id=bt_id).all()
    assert len(equity_records) == 3


def test_get_trades_returns_persisted_trades(db_session):
    """get_trades should return all trade records for a backtest."""
    repo = BacktestRepository(db_session)
    config = _make_config()
    result = _make_result()

    bt_id = repo.save_result(config, result, _make_trades(), _make_equity_curve())
    db_session.commit()

    trades = repo.get_trades(bt_id)
    assert len(trades) == 2
    symbols = {t.symbol for t in trades}
    assert symbols == {"BTCUSDT"}


def test_get_equity_curve_returns_sorted(db_session):
    """get_equity_curve should return equity points sorted by time."""
    repo = BacktestRepository(db_session)
    config = _make_config()
    result = _make_result()

    bt_id = repo.save_result(config, result, _make_trades(), _make_equity_curve())
    db_session.commit()

    curve = repo.get_equity_curve(bt_id)
    assert len(curve) == 3
    times = [pt.time for pt in curve]
    assert times == sorted(times)


def test_save_result_with_strategy_id_and_completed_at(db_session):
    """save_result should set strategy_id and completed_at when provided."""
    repo = BacktestRepository(db_session)
    config = _make_config()
    result = _make_result()
    strategy_id = uuid.uuid4()
    completed = datetime.now(timezone.utc)

    bt_id = repo.save_result(
        config, result, _make_trades(), _make_equity_curve(),
        strategy_id=strategy_id,
        completed_at=completed,
    )
    db_session.commit()

    record = db_session.query(BacktestRecord).filter_by(id=bt_id).first()
    assert record is not None
    assert str(record.strategy_id) == str(strategy_id)
    assert record.completed_at is not None


def test_save_result_without_optional_params(db_session):
    """save_result should work without strategy_id and completed_at."""
    repo = BacktestRepository(db_session)
    config = _make_config()
    result = _make_result()

    bt_id = repo.save_result(config, result, [], [])
    db_session.commit()

    record = db_session.query(BacktestRecord).filter_by(id=bt_id).first()
    assert record is not None
    assert record.strategy_id is None
    assert record.completed_at is None
