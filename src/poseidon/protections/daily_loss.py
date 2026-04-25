"""Daily loss protection.

Portfolio-level lock that halts all trading when daily P&L loss
exceeds a configurable threshold.
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
class DailyLossProtection(BaseProtection):
    """Portfolio-level lock when daily loss exceeds threshold.

    Creates locks with symbol=None (portfolio-level).
    Default: Halt all trading at 3% daily loss.
    """

    name = "daily_loss"

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct

    def check(self, symbol: str, market: str, db: Session) -> ProtectionResult:
        """Check if portfolio is locked due to daily loss.

        This is a portfolio-level check (symbol=None in lock records).
        The symbol parameter is still accepted for API consistency but
        the lock applies to ALL symbols in the market.
        """
        # Check existing portfolio-level lock
        lock = self.get_active_lock(symbol=None, market=market, db=db)
        if lock is not None:
            return ProtectionResult(
                locked=True,
                reason=(
                    f"Daily loss halt active for {market} "
                    f"(expires {lock.expires_at.isoformat() if lock.expires_at else 'manual'})"
                ),
                expires_at=lock.expires_at,
                protection_type=self.name,
            )

        # Calculate today's P&L
        daily_loss_pct = self._get_daily_loss_pct(market, db)
        if daily_loss_pct is not None and daily_loss_pct >= self.max_daily_loss_pct:
            expires_at = self._next_trading_open(market)
            self.create_lock(
                symbol=None,  # Portfolio-level
                market=market,
                reason=(f"Daily loss {daily_loss_pct:.2%} >= threshold {self.max_daily_loss_pct:.2%}"),
                expires_at=expires_at,
                db=db,
            )
            return ProtectionResult(
                locked=True,
                reason=(f"Daily loss triggered for {market}: {daily_loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}"),
                expires_at=expires_at,
                protection_type=self.name,
            )

        return ProtectionResult(
            locked=False,
            reason="",
            expires_at=None,
            protection_type=self.name,
        )

    def _get_daily_loss_pct(self, market: str, db: Session) -> float | None:
        """Calculate today's P&L loss as a positive fraction.

        Returns None if insufficient data.
        """
        try:
            from poseidon.models.nav_snapshot import NavSnapshotRecord

            now = datetime.now(UTC)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # Get most recent NAV before today (yesterday's close)
            prev_nav = (
                db.query(NavSnapshotRecord)
                .filter(
                    NavSnapshotRecord.market == market,
                    NavSnapshotRecord.snapshot_time < today_start,
                )
                .order_by(NavSnapshotRecord.snapshot_time.desc())
                .first()
            )

            # Get most recent NAV today
            current_nav = (
                db.query(NavSnapshotRecord)
                .filter(
                    NavSnapshotRecord.market == market,
                    NavSnapshotRecord.snapshot_time >= today_start,
                )
                .order_by(NavSnapshotRecord.snapshot_time.desc())
                .first()
            )

            if prev_nav is None or current_nav is None:
                return None

            prev_value = getattr(prev_nav, "nav", None) or getattr(prev_nav, "total_value", None)
            curr_value = getattr(current_nav, "nav", None) or getattr(current_nav, "total_value", None)

            if prev_value and curr_value and prev_value > 0:
                loss_pct = (prev_value - curr_value) / prev_value
                return max(0.0, loss_pct)  # Only positive loss

        except Exception:
            logger.debug(
                "Could not calculate daily loss for %s, skipping",
                market,
                exc_info=True,
            )

        return None

    @staticmethod
    def _next_trading_open(market: str) -> datetime:
        """Calculate next trading day open for the market.

        Crypto: Next UTC midnight.
        TW stock: Next weekday 01:30 UTC (09:30 TST).
        """
        now = datetime.now(UTC)

        if market == "tw_stock":
            # Next weekday at 01:30 UTC (09:30 TST)
            next_day = now + timedelta(days=1)
            next_open = next_day.replace(hour=1, minute=30, second=0, microsecond=0)
            # Skip weekends
            while next_open.weekday() >= 5:  # Saturday=5, Sunday=6
                next_open += timedelta(days=1)
            return next_open
        else:
            # Crypto and others: next UTC midnight
            tomorrow = now + timedelta(days=1)
            return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
