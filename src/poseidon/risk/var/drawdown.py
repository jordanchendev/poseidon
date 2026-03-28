"""Drawdown monitoring with high-water mark tracking and alert publishing.

Per D-09: Portfolio-level HWM tracking.
Per D-10: Three-tier progressive alerts at configurable thresholds.
Per D-11: Alert events published to Redis Streams poseidon:alerts:risk.
Per D-12: VaR limit breach also publishes alert event.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import redis

logger = logging.getLogger(__name__)


class DrawdownMonitor:
    """Track portfolio equity high-water mark and publish drawdown alerts.

    HWM is persisted in Redis so it survives restarts.  Drawdown alerts
    are published to a Redis Stream with 7-day retention, using the same
    consumer-group pattern as ``SignalDeliveryService``.
    """

    STREAM_KEY = "poseidon:alerts:risk"
    HWM_KEY = "poseidon:risk:hwm:portfolio"
    CONSUMER_GROUP = "default"
    RETENTION_DAYS = 7

    def __init__(
        self,
        redis_client: redis.Redis,
        warning_pct: float = 0.05,
        alert_pct: float = 0.10,
        critical_pct: float = 0.20,
    ) -> None:
        self._redis = redis_client
        # Ordered from lowest to highest severity — iteration keeps last match
        self._thresholds = [
            ("WARNING", warning_pct),
            ("ALERT", alert_pct),
            ("CRITICAL", critical_pct),
        ]
        self._ensure_consumer_group()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, current_equity: float) -> dict | None:
        """Update HWM and check for drawdown threshold breaches.

        Returns the alert dict if an alert was published, ``None`` otherwise.
        Only publishes the **highest-severity** alert to avoid spam.
        """
        raw_hwm = self._redis.get(self.HWM_KEY)
        hwm = float(raw_hwm) if raw_hwm else current_equity

        # Update HWM on new high
        if current_equity > hwm:
            hwm = current_equity

        # Always persist HWM (first call or new high)
        if raw_hwm is None or current_equity > float(raw_hwm):
            self._redis.set(self.HWM_KEY, str(hwm))

        # Guard: avoid division by zero when HWM is 0
        if hwm <= 0:
            return None

        drawdown = (hwm - current_equity) / hwm

        # Find highest-severity threshold breach (last match wins)
        triggered_alert: dict | None = None
        for level, threshold in self._thresholds:
            if drawdown >= threshold:
                triggered_alert = {
                    "event_type": "drawdown_breach",
                    "level": level,
                    "drawdown_pct": round(drawdown, 6),
                    "threshold_pct": threshold,
                    "current_equity": current_equity,
                    "hwm": hwm,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        if triggered_alert is not None:
            self._publish_alert(triggered_alert)

        return triggered_alert

    def publish_var_breach_alert(
        self,
        current_var: float,
        limit: float,
        method: str,
    ) -> dict:
        """Publish VaR limit breach alert per D-12.

        Called by VaR computation tasks when VaR exceeds the configured
        limit.  Uses a different ``event_type`` from drawdown alerts but
        the same Redis Stream.
        """
        alert = {
            "event_type": "var_limit_breach",
            "level": "CRITICAL",
            "current_var": round(current_var, 6),
            "var_limit": limit,
            "var_method": method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._publish_alert(alert)
        return alert

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_consumer_group(self) -> None:
        """Create consumer group on alerts stream if not exists."""
        try:
            self._redis.xgroup_create(
                name=self.STREAM_KEY,
                groupname=self.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _publish_alert(self, alert: dict) -> str:
        """Publish alert to Redis Stream with retention trimming."""
        min_id = self._retention_minid()
        msg_id = self._redis.xadd(
            self.STREAM_KEY,
            {"data": json.dumps(alert)},
            minid=min_id,
            approximate=True,
        )
        logger.warning(
            "Risk alert published: %s level=%s",
            alert["event_type"],
            alert.get("level", "N/A"),
        )
        return msg_id

    def _retention_minid(self) -> str:
        """Calculate MINID for 7-day retention."""
        cutoff_ms = int((time.time() - self.RETENTION_DAYS * 86400) * 1000)
        return f"{cutoff_ms}-0"
