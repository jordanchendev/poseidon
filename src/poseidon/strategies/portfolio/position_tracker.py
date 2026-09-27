"""PositionTracker -- DB-backed portfolio position persistence.

Tracks current portfolio holdings in PostgreSQL. On startup, rebuilds
state from DB. On changes, persists immediately. Survives container restarts.

Does NOT import or touch VirtualPortfolio (separate system).
"""

import logging
import math
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.models.position_lot import PositionLot
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
                    side=r.side,
                )
            logger.info("Rebuilt %d holdings from DB", len(self._holdings))
        finally:
            session.close()

    def current_holdings(self) -> dict[str, Holding]:
        """Return current holdings as {symbol: Holding} dict."""
        return dict(self._holdings)

    @staticmethod
    def project_lots(session, account_id, prices, account_nav, *, now=None, apply=False):
        """Plan an honest single-pilot compatibility view; caller owns commit."""
        account = session.get(PaperBrokerAccount, account_id)
        marker = f"decision-lots:{account_id}"
        holdings = session.scalars(
            select(PortfolioHoldingRecord)
            .where(PortfolioHoldingRecord.closed.is_(False), PortfolioHoldingRecord.strategy_name == marker)
            .order_by(PortfolioHoldingRecord.id)
            .with_for_update()
        ).all()
        foreign_holding = session.scalar(
            select(PortfolioHoldingRecord.id)
            .where(PortfolioHoldingRecord.closed.is_(False), PortfolioHoldingRecord.strategy_name != marker)
            .limit(1)
        )
        lots = session.scalars(
            select(PositionLot)
            .where(
                PositionLot.open_quantity > 0,
                PositionLot.account_scope == (account.account_scope if account is not None else ""),
                PositionLot.account_generation == (account.account_generation if account is not None else ""),
            )
            .order_by(PositionLot.opened_at, PositionLot.id)
            .with_for_update()
        ).all()
        reasons = []
        if account is None:
            reasons.append("legacy projection account is missing")
        if foreign_holding is not None:
            reasons.append("legacy pre-042/unowned open holdings require explicit resolution")
        if (
            account is not None
            and session.scalar(
                select(PositionLot.id)
                .where(
                    PositionLot.open_quantity > 0,
                    (PositionLot.account_scope != account.account_scope)
                    | (PositionLot.account_generation != account.account_generation),
                )
                .limit(1)
            )
            is not None
        ):
            reasons.append("legacy schema cannot represent multiple account/generation identities")
        grouped = {}
        for lot in lots:
            identity = (lot.market, lot.symbol, lot.instrument, lot.side)
            grouped.setdefault(identity, []).append(lot)
        if len({identity[1] for identity in grouped}) != len(grouped):
            reasons.append("legacy symbol key collapses market/instrument/side identities")
        rows = []
        for (market, symbol, instrument, side), group in sorted(grouped.items()):
            if market != "tw_stock" or instrument != "spot":
                reasons.append("legacy schema cannot losslessly represent non-spot instrument identity")
                continue
            mark = (prices or {}).get((market, symbol, instrument))
            if (
                isinstance(mark, bool)
                or not isinstance(mark, (int, float))
                or not math.isfinite(mark)
                or mark <= 0
                or not isinstance(account_nav, (int, float))
                or isinstance(account_nav, bool)
                or not math.isfinite(account_nav)
                or account_nav <= 0
            ):
                reasons.append("legacy projection requires positive finite marks and NAV")
                continue
            quantity = sum(lot.open_quantity for lot in group)
            entry = sum(lot.open_quantity * lot.cost_basis_json["unit_price"] for lot in group) / quantity
            multipliers = {lot.cost_basis_json["contract_multiplier"] for lot in group}
            if len(multipliers) != 1:
                reasons.append("legacy projection multiplier identities disagree")
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "side": side,
                    "shares": quantity,
                    "entry_price": entry,
                    "entry_date": group[0].opened_at.isoformat(),
                    "weight": abs(mark * quantity * next(iter(multipliers))) / account_nav,
                }
            )
        result = {"status": "unresolved" if reasons else "compatible", "reasons": sorted(set(reasons)), "rows": rows}
        if reasons or not apply:
            return result
        current = {(row.symbol, row.market, row.side): row for row in holdings}
        seen = set()
        current_time = now or datetime.now(UTC)
        for values in rows:
            key = values["symbol"], values["market"], values["side"]
            seen.add(key)
            row = current.get(key)
            if row is None:
                row = PortfolioHoldingRecord(
                    id=uuid.uuid4(),
                    strategy_name=marker,
                    symbol=key[0],
                    market=key[1],
                    side=key[2],
                    closed=False,
                    entry_date=datetime.fromisoformat(values["entry_date"]),
                )
                session.add(row)
            row.shares, row.entry_price, row.weight = values["shares"], values["entry_price"], values["weight"]
            row.updated_at = current_time
        for key, row in current.items():
            if key not in seen:
                row.closed, row.close_date, row.updated_at = True, current_time, current_time
        session.flush()
        return result

    def apply_orders(
        self,
        orders: list[RebalanceOrder],
        strategy_name: str,
        market: str = "tw_stock",
        fill_info: dict[str, tuple[float, float]] | None = None,
        side: str = "long",
    ) -> None:
        """Apply executed orders: update in-memory state and persist to DB.

        Args:
            orders: List of filled RebalanceOrders.
            strategy_name: Strategy that produced the orders.
            market: Market identifier.
            fill_info: {symbol: (shares, entry_price)} from actual fills.
            side: Position side ("long" or "short").

        For "buy": insert new PortfolioHoldingRecord with closed=False.
        For "sell": set existing record's closed=True and close_date=now.
        For "adjust": update weight on existing record.
        """
        fill_info = fill_info or {}
        session: Session = self._session_factory()
        try:
            now = datetime.now(UTC)
            for order in orders:
                shares, entry_price = fill_info.get(order.symbol, (None, None))
                if order.action == "buy":
                    record = PortfolioHoldingRecord(
                        strategy_name=strategy_name,
                        symbol=order.symbol,
                        market=market,
                        weight=order.target_weight,
                        shares=shares,
                        entry_price=entry_price,
                        entry_date=now,
                        closed=False,
                        side=side,
                    )
                    session.add(record)
                    self._holdings[order.symbol] = Holding(
                        symbol=order.symbol,
                        market=market,
                        weight=order.target_weight,
                        shares=shares,
                        entry_price=entry_price,
                        entry_date=now,
                        side=side,
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
