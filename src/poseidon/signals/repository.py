"""Signal repository for DB persistence.

Handles saving and querying SignalRecord objects via SQLAlchemy.
Both passed and rejected signals are persisted.
"""

import uuid
from datetime import datetime

from poseidon.models.signal import SignalRecord
from poseidon.signals.schemas import Signal


class SignalRepository:
    """Persists and queries signal records in PostgreSQL."""

    def __init__(self, db_session):
        self._session = db_session

    def save(self, signal: Signal) -> SignalRecord:
        """Convert a Signal Pydantic model to a SignalRecord ORM model and persist.

        Flushes but does NOT commit -- caller manages the transaction.
        """
        record = SignalRecord(
            id=signal.id,
            strategy_id=signal.strategy_id,
            model_id=signal.model_id,
            symbol=signal.symbol,
            market=signal.market,
            instrument=signal.instrument.value,
            action=signal.action.value,
            confidence=signal.confidence,
            quantity_pct=signal.quantity_pct,
            signal_time=signal.signal_time,
            valid_until=signal.valid_until,
            interval=signal.interval,
            params=signal.params,
            status=signal.status.value,
            reject_reason=signal.reject_reason,
            metadata_=signal.metadata,
            order_type=signal.order_type.value if signal.order_type else None,
            order_price=signal.order_price,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_by_id(self, signal_id: uuid.UUID) -> SignalRecord | None:
        """Query a signal record by primary key."""
        return self._session.get(SignalRecord, signal_id)

    def list_by_market(
        self,
        market: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SignalRecord]:
        """List signals filtered by market, optionally by status.

        Results are ordered by signal_time descending.
        """
        query = self._session.query(SignalRecord).filter(
            SignalRecord.market == market,
        )
        if status is not None:
            query = query.filter(SignalRecord.status == status)
        return query.order_by(SignalRecord.signal_time.desc()).limit(limit).all()

    def count_recent_by_symbol(
        self,
        symbol: str,
        market: str,
        since: datetime,
    ) -> int:
        """Count PASSED signals for a symbol since a given time.

        Used by FrequencyRule to enforce rate limits.
        """
        return (
            self._session.query(SignalRecord)
            .filter(
                SignalRecord.symbol == symbol,
                SignalRecord.market == market,
                SignalRecord.status == "passed",
                SignalRecord.signal_time >= since,
            )
            .count()
        )

    def latest_passed(
        self,
        market: str,
        since: datetime,
        limit: int = 100,
    ) -> list[SignalRecord]:
        """Return the most recent PASSED signals for ``market`` since ``since``.

        Phase 89-02 wires perp_rebalance / portfolio_monthly_rebalance through
        this method so live orders are produced from PASSED signals (closing
        the F8 wiring breach found in Phase 88 — 25 orders, 0 matching signals).

        Args:
            market: Market identifier (e.g. ``"crypto_perp"``, ``"tw_stock"``).
            since: Lower bound on ``signal_time`` (typically now() - 7d).
            limit: Maximum rows returned (default 100, ordered DESC by signal_time).

        Returns:
            List of SignalRecord ordered by signal_time DESC. Excludes
            pending / rejected signals — only ``status == 'passed'``.
        """
        return (
            self._session.query(SignalRecord)
            .filter(
                SignalRecord.market == market,
                SignalRecord.status == "passed",
                SignalRecord.signal_time >= since,
            )
            .order_by(SignalRecord.signal_time.desc())
            .limit(limit)
            .all()
        )
