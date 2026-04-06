"""Max drawdown protection.

Locks a symbol when its drawdown exceeds a configurable threshold.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from poseidon.protections.base import BaseProtection, ProtectionResult
from poseidon.protections.registry import register_protection

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@register_protection
class MaxDrawdownProtection(BaseProtection):
    """Per-symbol lock when drawdown exceeds threshold.

    Default: Lock for 48h after hitting 15% drawdown.
    """

    name = "max_drawdown"

    def __init__(
        self,
        max_drawdown_pct: float = 0.15,
        lock_hours: float = 48.0,
    ) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.lock_hours = lock_hours

    def check(self, symbol: str, market: str, db: Session) -> ProtectionResult:
        """Check if symbol is locked due to max drawdown.

        Queries for existing active lock first. If no lock exists,
        checks current drawdown from portfolio holdings. If drawdown
        exceeds threshold, creates a new lock.
        """
        # Check existing lock first
        lock = self.get_active_lock(symbol, market, db)
        if lock is not None:
            return ProtectionResult(
                locked=True,
                reason=(
                    f"Max drawdown lock for {symbol} "
                    f"(threshold={self.max_drawdown_pct:.0%}, "
                    f"expires {lock.expires_at.isoformat() if lock.expires_at else 'manual'})"
                ),
                expires_at=lock.expires_at,
                protection_type=self.name,
            )

        # Calculate current drawdown from portfolio holdings
        current_dd = self._get_symbol_drawdown(symbol, market, db)
        if current_dd is not None and current_dd >= self.max_drawdown_pct:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=self.lock_hours)
            self.create_lock(
                symbol=symbol,
                market=market,
                reason=(
                    f"Drawdown {current_dd:.2%} >= threshold {self.max_drawdown_pct:.2%}"
                ),
                expires_at=expires_at,
                db=db,
            )
            return ProtectionResult(
                locked=True,
                reason=(
                    f"Max drawdown triggered for {symbol}: "
                    f"{current_dd:.2%} >= {self.max_drawdown_pct:.2%}"
                ),
                expires_at=expires_at,
                protection_type=self.name,
            )

        return ProtectionResult(
            locked=False,
            reason="",
            expires_at=None,
            protection_type=self.name,
        )

    def _get_symbol_drawdown(
        self, symbol: str, market: str, db: Session
    ) -> float | None:
        """Calculate current drawdown for a symbol from portfolio holdings.

        Returns drawdown as a positive fraction (e.g. 0.15 for 15%),
        or None if no position data is available.
        """
        try:
            from poseidon.models.portfolio_holding import PortfolioHoldingRecord

            holding = (
                db.query(PortfolioHoldingRecord)
                .filter(
                    PortfolioHoldingRecord.symbol == symbol,
                    PortfolioHoldingRecord.market == market,
                )
                .first()
            )

            if holding is None:
                return None

            # Drawdown from entry: (entry - current) / entry
            entry_price = getattr(holding, "avg_entry_price", None)
            current_price = getattr(holding, "current_price", None)

            if entry_price and current_price and entry_price > 0:
                drawdown = (entry_price - current_price) / entry_price
                return max(0.0, drawdown)  # Only positive drawdown

        except Exception:
            logger.debug(
                "Could not calculate drawdown for %s/%s, skipping",
                symbol,
                market,
                exc_info=True,
            )

        return None
