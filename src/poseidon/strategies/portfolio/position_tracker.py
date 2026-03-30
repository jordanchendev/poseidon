"""PositionTracker -- DB-backed portfolio position persistence.

Tracks current portfolio holdings in PostgreSQL. On startup, rebuilds
state from DB. On changes, persists immediately. Survives container restarts.

Does NOT import or touch VirtualPortfolio (separate system).
"""

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.strategies.portfolio.schemas import Holding, RebalanceOrder

logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks current portfolio holdings in PostgreSQL.

    On startup, rebuilds state from DB. On changes, persists immediately.
    Survives container restarts (reads from portfolio_holdings table).
    """

    def __init__(self, session_factory):
        """Initialize with a SQLAlchemy session factory (e.g., SessionLocal).

        Args:
            session_factory: Callable that returns a Session (e.g., models.SessionLocal)
        """
        self._session_factory = session_factory
        self._holdings: dict[str, Holding] = {}

    def rebuild_from_db(self) -> None:
        """Load current (non-closed) holdings from DB on startup."""
        session: Session = self._session_factory()
        try:
            records = (
                session.query(PortfolioHoldingRecord)
                .filter(PortfolioHoldingRecord.closed == False)  # noqa: E712
                .all()
            )
            self._holdings = {}
            for r in records:
                self._holdings[r.symbol] = Holding(
                    symbol=r.symbol,
                    market=r.market,
                    weight=r.weight,
                    shares=r.shares,
                    entry_price=r.entry_price,
                    entry_date=r.entry_date,
                    stop_loss_pct=r.stop_loss_pct,
                )
            logger.info("Rebuilt %d holdings from DB", len(self._holdings))
        finally:
            session.close()

    def current_holdings(self) -> dict[str, Holding]:
        """Return current holdings as {symbol: Holding} dict."""
        return dict(self._holdings)

    def apply_orders(
        self, orders: list[RebalanceOrder], strategy_name: str, market: str = "tw_stock"
    ) -> None:
        """Apply executed orders: update in-memory state and persist to DB.

        For "buy": insert new PortfolioHoldingRecord with closed=False.
        For "sell": set existing record's closed=True and close_date=now.
        For "adjust": update weight on existing record.
        """
        session: Session = self._session_factory()
        try:
            now = datetime.now(timezone.utc)
            for order in orders:
                if order.action == "buy":
                    record = PortfolioHoldingRecord(
                        strategy_name=strategy_name,
                        symbol=order.symbol,
                        market=market,
                        weight=order.target_weight,
                        entry_date=now,
                        closed=False,
                    )
                    session.add(record)
                    self._holdings[order.symbol] = Holding(
                        symbol=order.symbol,
                        market=market,
                        weight=order.target_weight,
                        entry_date=now,
                    )
                elif order.action == "sell":
                    existing = (
                        session.query(PortfolioHoldingRecord)
                        .filter(
                            PortfolioHoldingRecord.symbol == order.symbol,
                            PortfolioHoldingRecord.closed == False,  # noqa: E712
                        )
                        .first()
                    )
                    if existing:
                        existing.closed = True
                        existing.close_date = now
                    self._holdings.pop(order.symbol, None)
                elif order.action == "adjust":
                    existing = (
                        session.query(PortfolioHoldingRecord)
                        .filter(
                            PortfolioHoldingRecord.symbol == order.symbol,
                            PortfolioHoldingRecord.closed == False,  # noqa: E712
                        )
                        .first()
                    )
                    if existing:
                        existing.weight = order.target_weight
                        existing.updated_at = now
                    if order.symbol in self._holdings:
                        self._holdings[order.symbol].weight = order.target_weight
            session.commit()
            logger.info("Applied %d orders for strategy %s", len(orders), strategy_name)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
