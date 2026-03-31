"""Funding rate settlement recording for perpetual positions.

Records funding payments as TradeLogRecord entries with entry_type='funding'.
Designed to be called every 8h for each open perp position.

Phase 26: function implementation only.
Phase 27: Celery Beat scheduling.
"""
import logging
from datetime import datetime

from poseidon.models.trade_log import TradeLogRecord

logger = logging.getLogger(__name__)


def record_funding_settlement(
    session_factory,
    symbol: str,
    strategy_name: str,
    funding_amount: float,
    position_quantity: float,
    mark_price: float,
    settlement_time: datetime,
    market: str = "crypto_perp",
) -> TradeLogRecord | None:
    """Record a funding rate payment as a trade log entry.

    Idempotent: checks for existing record at this (symbol, settlement_time)
    before inserting. Duplicate calls for the same settlement period are no-ops.

    Args:
        session_factory: Callable returning SQLAlchemy Session.
        symbol: Perp symbol (e.g. "BTCUSDT").
        strategy_name: Strategy owning this position (e.g. "Crypto Trend 4H").
        funding_amount: Funding payment amount.
            Positive = received (short position when funding is positive).
            Negative = paid (long position when funding is positive).
        position_quantity: Absolute position size at settlement time.
        mark_price: Mark price at settlement time (used for entry_price/exit_price).
        settlement_time: UTC datetime of the funding settlement (00:00, 08:00, or 16:00).
        market: Market identifier (default "crypto_perp").

    Returns:
        Created TradeLogRecord, or None if already recorded (idempotent).
    """
    session = session_factory()
    try:
        # Idempotency check (Research pitfall #4)
        existing = (
            session.query(TradeLogRecord)
            .filter(
                TradeLogRecord.symbol == symbol,
                TradeLogRecord.entry_type == "funding",
                TradeLogRecord.exit_date == settlement_time,
            )
            .first()
        )
        if existing is not None:
            logger.debug(
                "Funding settlement already recorded for %s at %s, skipping",
                symbol,
                settlement_time,
            )
            return None

        record = TradeLogRecord(
            strategy_name=strategy_name,
            symbol=symbol,
            market=market,
            entry_price=mark_price,
            exit_price=mark_price,
            entry_date=settlement_time,
            exit_date=settlement_time,
            shares=position_quantity,
            realized_pnl=funding_amount,
            holding_days=0,
            entry_type="funding",
        )
        session.add(record)
        session.commit()

        logger.info(
            "Recorded funding settlement: %s %s amount=%.6f qty=%.4f",
            symbol,
            "received" if funding_amount >= 0 else "paid",
            abs(funding_amount),
            position_quantity,
        )
        return record
    except Exception:
        session.rollback()
        logger.exception("Failed to record funding settlement for %s", symbol)
        raise
    finally:
        session.close()
