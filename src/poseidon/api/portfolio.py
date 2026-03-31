"""Portfolio API endpoints for performance tracking, holdings, and orders.

Mounted at /api/portfolio in main.py.
"""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel as PydanticBase
from sqlalchemy import func
from sqlalchemy.orm import Session

from poseidon.models.base import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic response schemas ---


class NavPointResponse(PydanticBase):
    date: str
    total_nav: float
    holdings_value: float
    cash: float
    holdings_count: int


class PerformanceSummaryResponse(PydanticBase):
    nav_curve: list[NavPointResponse]
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float | None
    total_trades: int
    total_realized_pnl: float


class HoldingResponse(PydanticBase):
    symbol: str
    market: str
    weight: float
    shares: float | None
    entry_price: float | None
    entry_date: str | None
    stop_loss_pct: float | None
    current_price: float | None
    unrealized_pnl: float | None
    side: str = "long"


class HoldingsResponse(PydanticBase):
    holdings: list[HoldingResponse]
    total_holdings: int


class OrderResponse(PydanticBase):
    id: str
    symbol: str
    market: str
    action: str
    quantity: float
    status: str
    price: float | None
    broker_mode: str
    reject_reason: str | None
    created_at: str
    side: str = "long"


class OrdersResponse(PydanticBase):
    orders: list[OrderResponse]
    total: int


# --- GET /performance ---


@router.get("/performance", response_model=PerformanceSummaryResponse)
def get_performance(db: Session = Depends(get_db)):
    """Return NAV curve with total return, max drawdown, and Sharpe ratio."""
    from poseidon.models.nav_snapshot import NavSnapshotRecord
    from poseidon.models.trade_log import TradeLogRecord

    # Query NAV snapshots ordered by date
    snapshots = (
        db.query(NavSnapshotRecord)
        .order_by(NavSnapshotRecord.snapshot_date.asc())
        .all()
    )

    # Build NAV curve
    nav_curve = [
        NavPointResponse(
            date=str(s.snapshot_date),
            total_nav=s.total_nav,
            holdings_value=s.holdings_value,
            cash=s.cash,
            holdings_count=s.holdings_count,
        )
        for s in snapshots
    ]

    # Compute total return
    if len(snapshots) >= 2:
        first_nav = snapshots[0].total_nav
        last_nav = snapshots[-1].total_nav
        total_return_pct = ((last_nav / first_nav) - 1) * 100 if first_nav != 0 else 0.0
    elif len(snapshots) == 1:
        total_return_pct = 0.0
    else:
        total_return_pct = 0.0

    # Compute max drawdown (peak-to-trough)
    max_drawdown_pct = 0.0
    peak = 0.0
    for s in snapshots:
        nav = s.total_nav
        if nav > peak:
            peak = nav
        if peak > 0:
            drawdown = (peak - nav) / peak * 100
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

    # Compute Sharpe ratio (annualized from daily returns)
    sharpe_ratio: float | None = None
    if len(snapshots) >= 2:
        navs = [s.total_nav for s in snapshots]
        daily_returns = []
        for i in range(1, len(navs)):
            if navs[i - 1] != 0:
                daily_returns.append(navs[i] / navs[i - 1] - 1)
        if len(daily_returns) >= 1:
            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
            std_ret = math.sqrt(variance)
            if std_ret > 0:
                sharpe_ratio = round((mean_ret / std_ret) * math.sqrt(252), 4)

    # Query trade stats
    trade_stats = db.query(
        func.count(TradeLogRecord.id).label("total_trades"),
        func.coalesce(func.sum(TradeLogRecord.realized_pnl), 0.0).label("total_pnl"),
    ).first()

    total_trades = trade_stats.total_trades if trade_stats else 0
    total_realized_pnl = round(float(trade_stats.total_pnl), 2) if trade_stats else 0.0

    return PerformanceSummaryResponse(
        nav_curve=nav_curve,
        total_return_pct=round(total_return_pct, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        sharpe_ratio=sharpe_ratio,
        total_trades=total_trades,
        total_realized_pnl=total_realized_pnl,
    )


# --- GET /holdings ---


@router.get("/holdings", response_model=HoldingsResponse)
def get_holdings(db: Session = Depends(get_db)):
    """Return current open holdings with unrealized PnL from latest OHLCV close."""
    from poseidon.models.ohlcv import OHLCV
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord

    holdings = (
        db.query(PortfolioHoldingRecord)
        .filter(PortfolioHoldingRecord.closed == False)  # noqa: E712
        .all()
    )

    result = []
    for h in holdings:
        # Fetch latest daily close price
        latest = (
            db.query(OHLCV.close)
            .filter(
                OHLCV.symbol == h.symbol,
                OHLCV.market == h.market,
                OHLCV.interval == "1d",
            )
            .order_by(OHLCV.time.desc())
            .first()
        )

        current_price: float | None = None
        unrealized_pnl: float | None = None

        if latest is not None:
            current_price = float(latest.close)
            if h.entry_price is not None and h.shares is not None:
                direction = 1 if h.side == "long" else -1
                unrealized_pnl = round((current_price - h.entry_price) * h.shares * direction, 2)

        result.append(
            HoldingResponse(
                symbol=h.symbol,
                market=h.market,
                weight=h.weight,
                shares=h.shares,
                entry_price=h.entry_price,
                entry_date=str(h.entry_date) if h.entry_date else None,
                stop_loss_pct=h.stop_loss_pct,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                side=h.side,
            )
        )

    return HoldingsResponse(holdings=result, total_holdings=len(result))


# --- GET /orders ---


@router.get("/orders", response_model=OrdersResponse)
def get_orders(
    status: str | None = Query(None, description="Filter by order status"),
    limit: int = Query(50, ge=1, le=500, description="Max orders to return"),
    db: Session = Depends(get_db),
):
    """Return pending and recent orders with optional status filter."""
    from poseidon.models.order import OrderRecord

    query = db.query(OrderRecord)
    if status is not None:
        query = query.filter(OrderRecord.status == status)
    query = query.order_by(OrderRecord.created_at.desc()).limit(limit)

    orders = query.all()

    result = [
        OrderResponse(
            id=str(o.id),
            symbol=o.symbol,
            market=o.market,
            action=o.action,
            quantity=o.quantity,
            status=o.status,
            price=o.price,
            broker_mode=o.broker_mode,
            reject_reason=o.reject_reason,
            created_at=str(o.created_at),
            side=o.side,
        )
        for o in orders
    ]

    return OrdersResponse(orders=result, total=len(result))
