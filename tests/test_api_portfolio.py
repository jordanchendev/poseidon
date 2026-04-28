"""Tests for portfolio API endpoints using remote market data."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
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
from poseidon.models.nav_snapshot import NavSnapshotRecord
from poseidon.models.order import OrderRecord
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


# ---------------------------------------------------------------------------
# Phase 87 integration tests
# ---------------------------------------------------------------------------


def _seed_nav(
    *,
    snapshot_date: date,
    total_nav: float,
    market: str = "crypto_perp",
    cash_flow: float = 0.0,
    holdings_value: float = 0.0,
    cash: float | None = None,
    holdings_count: int = 0,
) -> None:
    db = TestingSessionLocal()
    db.add(
        NavSnapshotRecord(
            id=uuid.uuid4(),
            snapshot_date=snapshot_date,
            total_nav=total_nav,
            holdings_value=holdings_value,
            cash=total_nav - holdings_value if cash is None else cash,
            holdings_count=holdings_count,
            cash_flow=Decimal(str(cash_flow)),
            market=market,
            created_at=datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day, tzinfo=UTC),
        )
    )
    db.commit()
    db.close()


def _seed_order(
    *,
    symbol: str,
    market: str,
    status: str,
    reject_reason: dict | None = None,
    created_at: datetime | None = None,
) -> uuid.UUID:
    db = TestingSessionLocal()
    oid = uuid.uuid4()
    db.add(
        OrderRecord(
            id=oid,
            strategy_name="test-strategy",
            symbol=symbol,
            market=market,
            action="buy",
            order_type="market",
            target_weight=0.1,
            quantity=1.0,
            price=None,
            side="long",
            status=status,
            broker_order_id=None,
            broker_mode="paper",
            reject_reason=reject_reason,
            created_at=created_at or datetime(2026, 4, 1, tzinfo=UTC),
            updated_at=created_at or datetime(2026, 4, 1, tzinfo=UTC),
        )
    )
    db.commit()
    db.close()
    return oid


def test_performance_modified_dietz_isolates_cash_flow():
    """TRUTH-01 (D-01): a $9.87M deposit must not inflate total_return_pct.

    Day 1: NAV=1M, cf=0
    Day 2: NAV=11M, cf=+9.87M deposit
    Naive return = (11M - 1M) / 1M = 1000%. Modified Dietz must yield
    a tiny return (the actual investment gain ~= 130K against avg
    capital ~= 5.94M, ~= 2.19%).
    """
    _seed_nav(snapshot_date=date(2026, 4, 27), total_nav=1_000_000.0, cash_flow=0.0)
    _seed_nav(snapshot_date=date(2026, 4, 28), total_nav=11_000_000.0, cash_flow=9_870_000.0)

    resp = client.get("/api/portfolio/performance")
    assert resp.status_code == 200
    payload = resp.json()

    # Sanity: NAV curve preserves cash_flow column (TRUTH-04, D-05).
    assert len(payload["nav_curve"]) == 2
    assert payload["nav_curve"][1]["cash_flow"] == 9_870_000.0

    # Critical: TWR must NOT be ~10. Expected ~= 0.0219.
    assert payload["total_return_pct"] < 0.05, f"cash_flow leaked into return: {payload['total_return_pct']}"
    assert payload["total_return_pct"] > 0.0


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_holdings_excludes_crypto_perp(mock_from_settings):
    """TRUTH-02 (D-09): /holdings allowlist = {tw_stock, us_stock}.

    Seed an ETHUSDT crypto_perp holding alongside a tw_stock holding.
    Only the tw_stock entry should surface; crypto_perp must be excluded.
    """
    _seed_holding(symbol="2330", market="tw_stock", shares=100.0, entry_price=500.0)
    _seed_holding(symbol="ETHUSDT", market="crypto_perp", shares=2.0, entry_price=3000.0)

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
    symbols = {h["symbol"] for h in payload["holdings"]}
    markets = {h["market"] for h in payload["holdings"]}
    assert symbols == {"2330"}
    assert markets <= {"tw_stock", "us_stock"}
    assert "ETHUSDT" not in symbols


def test_orders_returns_structured_reject_reason_dict():
    """TRUTH-03 (D-13/D-17): rejected orders must surface as dict with 4 keys."""
    payload = {
        "check_name": "max_exposure",
        "rule": "Total exposure 110% exceeds 100% limit",
        "shortfall": {"projected_pct": 1.10, "limit_pct": 1.00},
        "details": None,
    }
    _seed_order(symbol="2330", market="tw_stock", status="rejected", reject_reason=payload)

    resp = client.get("/api/portfolio/orders?status=rejected")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    rr = body["orders"][0]["reject_reason"]
    assert isinstance(rr, dict)
    assert set(rr.keys()) == {"check_name", "rule", "shortfall", "details"}
    assert rr["check_name"] == "max_exposure"
    assert rr["shortfall"] == {"projected_pct": 1.10, "limit_pct": 1.00}


def test_orders_legacy_text_wrapped_payload():
    """D-15: legacy free-text reject_reason from migration 034 must surface
    as a structured dict with check_name='legacy', rule='pre_phase87',
    shortfall=None, details=<original text>.

    Migration 034 wraps any pre-existing free-text in this shape so the
    JSONB column is uniform. This test confirms the API still returns it
    cleanly as a 4-key dict.
    """
    legacy_payload = {
        "check_name": "legacy",
        "rule": "pre_phase87",
        "shortfall": None,
        "details": "old free-text rejection note",
    }
    _seed_order(
        symbol="BTCUSDT",
        market="crypto_perp",
        status="rejected",
        reject_reason=legacy_payload,
    )

    resp = client.get("/api/portfolio/orders?status=rejected")
    assert resp.status_code == 200
    body = resp.json()
    rr = body["orders"][0]["reject_reason"]
    assert rr["check_name"] == "legacy"
    assert rr["rule"] == "pre_phase87"
    assert rr["shortfall"] is None
    assert rr["details"] == "old free-text rejection note"
