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

from poseidon.core.database import get_db

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


class PerpHoldingResponse(PydanticBase):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    leverage: int
    liquidation_price: float
    margin_ratio: float
    unrealized_pnl: float
    cumulative_funding_cost: float
    current_price: float | None = None


class PerpHoldingsResponse(PydanticBase):
    holdings: list[PerpHoldingResponse]
    total_holdings: int


# --- GET /performance ---


@router.get("/performance", response_model=PerformanceSummaryResponse)
def get_performance(
    market: str | None = Query(None, description="Filter by market (e.g., 'crypto_perp')"),
    db: Session = Depends(get_db),
):
    """Return NAV curve with total return, max drawdown, and Sharpe ratio.

    When market is provided, returns only NAV snapshots and trades for that market.
    Without market param, returns all markets (backward compatible).
    """
    from poseidon.models.nav_snapshot import NavSnapshotRecord
    from poseidon.models.trade_log import TradeLogRecord

    # Query NAV snapshots ordered by date, optionally filtered by market
    query = db.query(NavSnapshotRecord).order_by(NavSnapshotRecord.snapshot_date.asc())
    if market is not None:
        query = query.filter(NavSnapshotRecord.market == market)
    snapshots = query.all()

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

    # Query trade stats, optionally filtered by market
    trade_query = db.query(
        func.count(TradeLogRecord.id).label("total_trades"),
        func.coalesce(func.sum(TradeLogRecord.realized_pnl), 0.0).label("total_pnl"),
    )
    if market is not None:
        trade_query = trade_query.filter(TradeLogRecord.market == market)
    trade_stats = trade_query.first()

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


# --- GET /perp-holdings ---


@router.get("/perp-holdings", response_model=PerpHoldingsResponse)
def get_perp_holdings(db: Session = Depends(get_db)):
    """Return perp-specific holdings with leverage, liquidation, and margin detail (D-09)."""
    from poseidon.broker.perp_paper_adapter import (
        PerpPaperAdapter,
        PerpPosition,
        calc_liquidation_price,
    )
    from poseidon.core.database import SessionLocal
    from poseidon.models.ohlcv import OHLCV
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord
    from poseidon.models.trade_log import TradeLogRecord

    # Query open perp holdings from DB
    perp_holdings = (
        db.query(PortfolioHoldingRecord)
        .filter(
            PortfolioHoldingRecord.closed == False,  # noqa: E712
            PortfolioHoldingRecord.market == "crypto_perp",
        )
        .all()
    )

    if not perp_holdings:
        return PerpHoldingsResponse(holdings=[], total_holdings=0)

    # Build PerpPaperAdapter and populate positions from DB holdings
    adapter = PerpPaperAdapter(SessionLocal)

    # Reconstruct adapter positions from DB holdings
    for h in perp_holdings:
        leverage = 3  # default, will be overridden by adapter config
        liq_price = calc_liquidation_price(
            h.entry_price or 0.0, leverage, h.side or "long"
        )
        adapter._positions[h.symbol] = PerpPosition(
            symbol=h.symbol,
            side=h.side or "long",
            entry_price=h.entry_price or 0.0,
            quantity=h.shares or 0.0,
            leverage=leverage,
            margin=(h.entry_price or 0.0) * (h.shares or 0.0) / leverage,
            liquidation_price=liq_price,
        )

    # Get latest mark prices from DB (perpetual OHLCV)
    mark_prices: dict[str, float] = {}
    for h in perp_holdings:
        latest = (
            db.query(OHLCV.close)
            .filter(
                OHLCV.symbol == h.symbol,
                OHLCV.market == "crypto_perp",
            )
            .order_by(OHLCV.time.desc())
            .first()
        )
        if latest:
            mark_prices[h.symbol] = float(latest.close)

    # Update mark prices for PnL calculation
    adapter.update_mark_prices(mark_prices)

    # Query cumulative funding from trade logs
    funding_sums = (
        db.query(
            TradeLogRecord.symbol,
            func.coalesce(func.sum(TradeLogRecord.realized_pnl), 0.0).label(
                "total_funding"
            ),
        )
        .filter(
            TradeLogRecord.market == "crypto_perp",
            TradeLogRecord.entry_type == "funding",
        )
        .group_by(TradeLogRecord.symbol)
        .all()
    )
    funding_by_symbol = {row.symbol: float(row.total_funding) for row in funding_sums}

    # Build response from adapter positions
    positions = adapter.query_positions()
    result = []
    for pos in positions:
        result.append(
            PerpHoldingResponse(
                symbol=pos["symbol"],
                side=pos["side"],
                quantity=pos["quantity"],
                entry_price=pos["entryPrice"],
                leverage=pos["leverage"],
                liquidation_price=pos["liquidationPrice"],
                margin_ratio=pos["marginRatio"],
                unrealized_pnl=pos["unrealizedPnl"],
                cumulative_funding_cost=funding_by_symbol.get(pos["symbol"], 0.0),
                current_price=mark_prices.get(pos["symbol"]),
            )
        )

    return PerpHoldingsResponse(holdings=result, total_holdings=len(result))
