"""Cooldown protection.

Creates a time-expiring lock after being triggered, preventing
re-entry into a symbol for a configurable number of bars.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from poseidon.protections.base import BaseProtection, ProtectionResult
from poseidon.protections.registry import register_protection

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@register_protection
class CooldownProtection(BaseProtection):
    """Per-symbol cooldown lock after a trade or signal trigger.

    Default: 3 bars of 4h = 12 hours cooldown.
    """

    name = "cooldown"

    def __init__(
        self,
        cooldown_bars: int = 3,
        bar_duration_hours: float = 4.0,
    ) -> None:
        self.cooldown_bars = cooldown_bars
        self.bar_duration_hours = bar_duration_hours

    @property
    def cooldown_hours(self) -> float:
        return self.cooldown_bars * self.bar_duration_hours

    def check(self, symbol: str, market: str, db: Session) -> ProtectionResult:
        """Check if symbol is in cooldown period."""
        lock = self.get_active_lock(symbol, market, db)

        if lock is not None:
            return ProtectionResult(
                locked=True,
                reason=(
                    f"Cooldown active for {symbol} "
                    f"(expires {lock.expires_at.isoformat() if lock.expires_at else 'manual'})"
                ),
                expires_at=lock.expires_at,
                protection_type=self.name,
            )

        return ProtectionResult(
            locked=False,
            reason="",
            expires_at=None,
            protection_type=self.name,
        )

    def trigger_cooldown(self, symbol: str, market: str, db: Session) -> None:
        """Create a cooldown lock for the given symbol.

        Called externally after a trade or signal trigger.
        """
        expires_at = datetime.now(UTC) + timedelta(hours=self.cooldown_hours)
        self.create_lock(
            symbol=symbol,
            market=market,
            reason=f"Cooldown: {self.cooldown_bars} bars x {self.bar_duration_hours}h",
            expires_at=expires_at,
            db=db,
        )
