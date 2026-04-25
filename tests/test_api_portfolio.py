"""Tests for portfolio API endpoints using remote market data."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from poseidon.api.portfolio import router as portfolio_router
from poseidon.core.database import get_db
from poseidon.models.base import Base
from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.models.trade_log import TradeLogRecord


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(portfolio_router, prefix="/api/portfolio", tags=["portfolio"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db

client = TestClient(_test_app)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


def _seed_holding(
    *,
    symbol: str,
    market: str,
    shares: float,
    entry_price: float,
    side: str = "long",
) -> None:
    db = TestingSessionLocal()
    db.add(
        PortfolioHoldingRecord(
            id=uuid.uuid4(),
            strategy_name="test-portfolio",
            symbol=symbol,
            market=market,
            weight=0.25,
            shares=shares,
            entry_price=entry_price,
            side=side,
            entry_date=datetime(2026, 4, 1, tzinfo=UTC),
            closed=False,
            stop_loss_pct=0.1,
            created_at=datetime(2026, 4, 1, tzinfo=UTC),
            updated_at=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )
    db.commit()
    db.close()


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_get_holdings_uses_remote_ohlcv(mock_from_settings):
    _seed_holding(symbol="2330", market="tw_stock", shares=100.0, entry_price=500.0)

    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = pd.DataFrame(
        {"close": [520.0]},
        index=pd.date_range("2026-04-18", periods=1, tz="UTC", name="time"),
    )
    mock_from_settings.return_value = mock_repo

    resp = client.get("/api/portfolio/holdings")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total_holdings"] == 1
    assert payload["holdings"][0]["current_price"] == 520.0
    assert payload["holdings"][0]["unrealized_pnl"] == 2000.0
    mock_repo.read_ohlcv.assert_called_once_with("2330", "tw_stock", "1d")


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_get_perp_holdings_uses_remote_latest_price(mock_from_settings):
    _seed_holding(symbol="BTCUSDT", market="crypto_perp", shares=0.5, entry_price=30000.0)

    db = TestingSessionLocal()
    db.add(
        TradeLogRecord(
            id=uuid.uuid4(),
            strategy_name="funding",
            symbol="BTCUSDT",
            market="crypto_perp",
            entry_price=30000.0,
            exit_price=30010.0,
            entry_date=datetime(2026, 4, 10, tzinfo=UTC),
            exit_date=datetime(2026, 4, 10, tzinfo=UTC),
            shares=0.5,
            entry_type="funding",
            realized_pnl=-12.5,
            holding_days=0,
            signal_id=None,
            created_at=datetime(2026, 4, 10, tzinfo=UTC),
        )
    )
    db.commit()
    db.close()

    mock_repo = MagicMock()
    mock_repo.read_latest_price.return_value = 31000.0
    mock_from_settings.return_value = mock_repo

    resp = client.get("/api/portfolio/perp-holdings")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total_holdings"] == 1
    assert payload["holdings"][0]["current_price"] == 31000.0
    assert payload["holdings"][0]["cumulative_funding_cost"] == -12.5
    mock_repo.read_latest_price.assert_called_once_with("BTCUSDT")
