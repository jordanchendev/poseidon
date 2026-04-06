"""Base classes for the stateful protection layer.

BaseProtection ABC defines the interface for all protection types.
ProtectionResult is the standard return type from protection checks.
All protection locks persist in PostgreSQL and survive container restarts.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class ProtectionResult:
    """Result of a protection check."""

    locked: bool
    reason: str
    expires_at: datetime | None
    protection_type: str


class BaseProtection(ABC):
    """Abstract base class for stateful protections.

    Each concrete protection implements check() to evaluate whether
    a symbol/market is currently locked, and create_lock() to persist
    a new lock to PostgreSQL.

    Mirrors BaseRule pattern but operates at the protection layer
    (pre-signal-generation) rather than the risk layer (post-signal).
    """

    name: str = ""
    enabled: bool = True

    # Capability metadata (Phase 34 pattern)
    supports_backtest: bool = True
    supports_live: bool = True
    bias_risk: list[str] = []
    stateful: bool = True

    @abstractmethod
    def check(self, symbol: str, market: str, db: Session) -> ProtectionResult:
        """Evaluate whether the symbol/market is currently locked.

        Returns ProtectionResult indicating lock status with reason.
        """
        ...

    def create_lock(
        self,
        symbol: str | None,
        market: str,
        reason: str,
        expires_at: datetime | None,
        db: Session,
    ) -> None:
        """Create a ProtectionLockRecord in the database.

        Args:
            symbol: Symbol to lock, or None for portfolio-level locks.
            market: Market identifier (e.g. "crypto", "tw_stock").
            reason: Human-readable reason for the lock.
            expires_at: When the lock expires, or None for manual-release.
            db: SQLAlchemy session.
        """
        from poseidon.models.protection_lock import ProtectionLockRecord

        lock = ProtectionLockRecord(
            protection_type=self.name,
            symbol=symbol,
            market=market,
            reason=reason,
            expires_at=expires_at,
            active=True,
        )
        db.add(lock)
        db.commit()
        logger.info(
            "Protection lock created: type=%s symbol=%s market=%s expires=%s",
            self.name,
            symbol,
            market,
            expires_at,
        )

    def get_active_lock(
        self,
        symbol: str | None,
        market: str,
        db: Session,
    ) -> object | None:
        """Query for an active lock matching this protection type.

        Returns the ProtectionLockRecord if found, None otherwise.
        Handles both time-expiring and manual-release locks.
        """
        from poseidon.models.protection_lock import ProtectionLockRecord

        now = datetime.now(timezone.utc)
        query = db.query(ProtectionLockRecord).filter(
            ProtectionLockRecord.protection_type == self.name,
            ProtectionLockRecord.market == market,
            ProtectionLockRecord.active.is_(True),
        )

        # Match symbol: None for portfolio-level, specific for per-symbol
        if symbol is None:
            query = query.filter(ProtectionLockRecord.symbol.is_(None))
        else:
            query = query.filter(ProtectionLockRecord.symbol == symbol)

        lock = query.first()
        if lock is None:
            return None

        # Check if time-expiring lock has expired
        if lock.expires_at is not None and lock.expires_at <= now:
            lock.active = False
            lock.released_at = now
            lock.released_by = "auto"
            db.commit()
            logger.info(
                "Protection lock auto-expired: type=%s symbol=%s market=%s",
                self.name,
                symbol,
                market,
            )
            return None

        return lock

    def expire_stale_locks(self, db: Session) -> int:
        """Mark all expired locks as inactive.

        Returns the number of locks expired.
        """
        from poseidon.models.protection_lock import ProtectionLockRecord

        now = datetime.now(timezone.utc)
        stale = (
            db.query(ProtectionLockRecord)
            .filter(
                ProtectionLockRecord.protection_type == self.name,
                ProtectionLockRecord.active.is_(True),
                ProtectionLockRecord.expires_at.isnot(None),
                ProtectionLockRecord.expires_at <= now,
            )
            .all()
        )

        for lock in stale:
            lock.active = False
            lock.released_at = now
            lock.released_by = "auto"

        if stale:
            db.commit()
            logger.info(
                "Expired %d stale %s locks",
                len(stale),
                self.name,
            )

        return len(stale)
