"""Universe snapshot persistence and query functions.

Provides CRUD operations for UniverseSnapshotRecord, enabling
point-in-time universe reconstruction for backtest reproducibility.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from poseidon.data.symbols import SymbolInfo
from poseidon.models.universe_snapshot import UniverseSnapshotRecord

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def save_snapshot(
    db: Session,
    market: str,
    snapshot_time: datetime,
    symbols: list[SymbolInfo],
    source_type: str,
    filter_config: dict | None = None,
) -> UniverseSnapshotRecord:
    """Persist a full universe snapshot (D-10: full snapshot, not diffs).

    Args:
        db: SQLAlchemy session.
        market: Market identifier.
        snapshot_time: Timestamp of this snapshot.
        symbols: Resolved symbol list.
        source_type: Name of the UniverseSource used.
        filter_config: Optional dict describing applied filters.

    Returns:
        The created UniverseSnapshotRecord.
    """
    record = UniverseSnapshotRecord(
        market=market,
        snapshot_time=snapshot_time,
        symbols=[{"id": s.id, "name": s.name, "ccxt_symbol": s.ccxt_symbol} for s in symbols],
        source_type=source_type,
        filter_config=filter_config,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(
        "Saved universe snapshot: market=%s, symbols=%d, time=%s",
        market,
        len(symbols),
        snapshot_time,
    )
    return record


def get_latest_snapshot(db: Session, market: str) -> UniverseSnapshotRecord | None:
    """Get the most recent snapshot for a market."""
    return (
        db.query(UniverseSnapshotRecord)
        .filter(UniverseSnapshotRecord.market == market)
        .order_by(UniverseSnapshotRecord.snapshot_time.desc())
        .first()
    )


def get_snapshot_at(db: Session, market: str, target_date: datetime) -> UniverseSnapshotRecord | None:
    """Get latest snapshot at or before target_date for backtest reproducibility (D-11).

    Args:
        db: SQLAlchemy session.
        market: Market identifier.
        target_date: Point-in-time to query.

    Returns:
        The most recent snapshot with snapshot_time <= target_date, or None.
    """
    return (
        db.query(UniverseSnapshotRecord)
        .filter(
            UniverseSnapshotRecord.market == market,
            UniverseSnapshotRecord.snapshot_time <= target_date,
        )
        .order_by(UniverseSnapshotRecord.snapshot_time.desc())
        .first()
    )


def snapshot_to_symbols(record: UniverseSnapshotRecord) -> list[SymbolInfo]:
    """Deserialize JSONB symbols back to SymbolInfo list."""
    return [
        SymbolInfo(
            id=s["id"],
            name=s["name"],
            ccxt_symbol=s.get("ccxt_symbol"),
        )
        for s in record.symbols
    ]
