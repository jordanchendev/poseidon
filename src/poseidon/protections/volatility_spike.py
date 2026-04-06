"""Volatility spike protection.

Locks a symbol when realized volatility exceeds a configurable
multiple of historical average. Locks are condition-based (no expiry)
and released when volatility drops below threshold.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from poseidon.protections.base import BaseProtection, ProtectionResult
from poseidon.protections.registry import register_protection

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@register_protection
class VolatilitySpikeProtection(BaseProtection):
    """Per-symbol lock when realized vol exceeds N * historical average.

    Default: Lock when realized vol > 2.5x the 30-day average.
    Locks have no expiry — they are released when the condition clears.
    """

    name = "volatility_spike"

    def __init__(
        self,
        vol_std_multiplier: float = 2.5,
        lookback_days: int = 30,
    ) -> None:
        self.vol_std_multiplier = vol_std_multiplier
        self.lookback_days = lookback_days

    def check(self, symbol: str, market: str, db: Session) -> ProtectionResult:
        """Check if symbol is locked due to volatility spike.

        Re-evaluates volatility each check. If vol has dropped below
        threshold, releases existing lock.
        """
        # Check existing lock
        lock = self.get_active_lock(symbol, market, db)

        # Calculate current volatility
        vol_data = self._get_volatility_data(symbol, market, db)

        if vol_data is None:
            # Cannot calculate vol — if lock exists keep it, otherwise pass
            if lock is not None:
                return ProtectionResult(
                    locked=True,
                    reason=f"Volatility spike lock active for {symbol} (data unavailable for re-eval)",
                    expires_at=None,
                    protection_type=self.name,
                )
            return ProtectionResult(
                locked=False,
                reason="",
                expires_at=None,
                protection_type=self.name,
            )

        current_vol, avg_vol = vol_data
        threshold = self.vol_std_multiplier * avg_vol if avg_vol > 0 else float("inf")

        if current_vol >= threshold:
            # Vol still high — create lock if not exists
            if lock is None:
                self.create_lock(
                    symbol=symbol,
                    market=market,
                    reason=(
                        f"Volatility spike: {current_vol:.4f} >= "
                        f"{self.vol_std_multiplier}x avg ({avg_vol:.4f})"
                    ),
                    expires_at=None,  # Condition-based release
                    db=db,
                )
            return ProtectionResult(
                locked=True,
                reason=(
                    f"Volatility spike for {symbol}: "
                    f"{current_vol:.4f} >= {threshold:.4f}"
                ),
                expires_at=None,
                protection_type=self.name,
            )
        else:
            # Vol dropped below threshold — release lock if exists
            if lock is not None:
                now = datetime.now(timezone.utc)
                lock.active = False
                lock.released_at = now
                lock.released_by = "auto"
                db.commit()
                logger.info(
                    "Volatility spike lock released for %s/%s: vol %f < threshold %f",
                    symbol,
                    market,
                    current_vol,
                    threshold,
                )

            return ProtectionResult(
                locked=False,
                reason="",
                expires_at=None,
                protection_type=self.name,
            )

    def _get_volatility_data(
        self, symbol: str, market: str, db: Session
    ) -> tuple[float, float] | None:
        """Calculate realized and average volatility for a symbol.

        Returns (current_vol, avg_vol) as annualized values,
        or None if insufficient data.
        """
        try:
            from poseidon.models.ohlcv import OHLCV

            now = datetime.now(timezone.utc)
            # Get OHLCV data for lookback period (use 2x for avg calculation)
            from datetime import timedelta

            lookback_start = now - timedelta(days=self.lookback_days * 2)

            rows = (
                db.query(OHLCV.close, OHLCV.timestamp)
                .filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timestamp >= lookback_start,
                )
                .order_by(OHLCV.timestamp.asc())
                .all()
            )

            if len(rows) < 10:  # Need minimum data
                return None

            closes = [float(r.close) for r in rows]

            # Calculate log returns
            log_returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0 and closes[i] > 0:
                    log_returns.append(math.log(closes[i] / closes[i - 1]))

            if len(log_returns) < 5:
                return None

            # Split into recent (lookback_days) and historical
            split_point = max(len(log_returns) - self.lookback_days, len(log_returns) // 2)
            recent_returns = log_returns[split_point:]
            historical_returns = log_returns[:split_point]

            if not recent_returns or not historical_returns:
                return None

            # Annualized volatility (assuming daily data, 252 trading days)
            annualization = math.sqrt(252)

            def std(values: list[float]) -> float:
                n = len(values)
                if n < 2:
                    return 0.0
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                return math.sqrt(variance)

            current_vol = std(recent_returns) * annualization
            avg_vol = std(historical_returns) * annualization

            return (current_vol, avg_vol)

        except Exception:
            logger.debug(
                "Could not calculate volatility for %s/%s, skipping",
                symbol,
                market,
                exc_info=True,
            )
            return None
