"""Protection manager — orchestrator for all protection checks.

ProtectionManager iterates through registered protections and
aggregates results, including both per-symbol and portfolio-level locks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from poseidon.protections.base import BaseProtection, ProtectionResult
from poseidon.protections.registry import list_protections, get_protection

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poseidon.models.protection_lock import ProtectionLockRecord

logger = logging.getLogger(__name__)


class ProtectionManager:
    """Orchestrator for all protection checks.

    Iterates all configured protections and checks both per-symbol
    and portfolio-level locks.
    """

    def __init__(self, protections: list[BaseProtection] | None = None) -> None:
        self._protections: list[BaseProtection] = protections or []

    @classmethod
    def from_defaults(cls) -> ProtectionManager:
        """Create a manager with all registered protections using default params."""
        protections = []
        for name in list_protections():
            protection_cls = get_protection(name)
            protections.append(protection_cls())
        return cls(protections=protections)

    def check_all(
        self, symbol: str, market: str, db: Session
    ) -> list[ProtectionResult]:
        """Check all protections for a symbol/market.

        Returns list of ProtectionResult where locked=True.
        Checks both per-symbol protections and portfolio-level locks.
        """
        active_locks: list[ProtectionResult] = []

        for protection in self._protections:
            if not protection.enabled:
                continue

            try:
                result = protection.check(symbol, market, db)
                if result.locked:
                    active_locks.append(result)
            except Exception:
                logger.error(
                    "Protection check failed: %s for %s/%s",
                    protection.name,
                    symbol,
                    market,
                    exc_info=True,
                )

        return active_locks

    def is_locked(self, symbol: str, market: str, db: Session) -> bool:
        """Convenience: returns True if any protection lock is active."""
        return len(self.check_all(symbol, market, db)) > 0

    def get_active_locks(
        self, market: str, db: Session, symbol: str | None = None
    ) -> list[ProtectionLockRecord]:
        """Query DB directly for all active locks.

        Args:
            market: Market to query.
            db: SQLAlchemy session.
            symbol: Optional symbol filter. If None, returns all locks for market.

        Returns:
            List of active ProtectionLockRecord objects.
        """
        from poseidon.models.protection_lock import ProtectionLockRecord

        query = db.query(ProtectionLockRecord).filter(
            ProtectionLockRecord.market == market,
            ProtectionLockRecord.active.is_(True),
        )

        if symbol is not None:
            # Return both per-symbol and portfolio-level locks
            from sqlalchemy import or_

            query = query.filter(
                or_(
                    ProtectionLockRecord.symbol == symbol,
                    ProtectionLockRecord.symbol.is_(None),
                )
            )

        return query.all()

    def expire_all_stale(self, db: Session) -> int:
        """Run stale lock expiration across all protections.

        Returns total number of locks expired.
        """
        total = 0
        for protection in self._protections:
            try:
                total += protection.expire_stale_locks(db)
            except Exception:
                logger.error(
                    "Failed to expire stale locks for %s",
                    protection.name,
                    exc_info=True,
                )
        return total
