"""VaR limit risk rule.

Rejects signals when portfolio VaR exceeds configurable limit.
Reads cached VaR snapshot from Redis -- never computes inline (per D-01).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import msgpack
import redis

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal, SignalAction

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)

# Snapshot older than this triggers a staleness warning
_STALE_THRESHOLD = timedelta(hours=3)


class VaRLimitRule(BaseRule):
    """Reject signals when portfolio VaR exceeds configurable limit.

    Per D-01: reads cached VaR snapshot from Redis, never computes inline.
    Per D-02: reads poseidon:var:latest:{method} key.
    """

    name = "var_limit"
    supports_live = True

    def __init__(self) -> None:
        self.max_var_pct: float = 0.05
        self.var_method: str = "parametric"
        self._redis: redis.Redis | None = None

    def load_params(self, params: dict) -> None:
        """Load rule parameters from DB-stored config."""
        self.max_var_pct = params.get("max_var_pct", 0.05)
        self.var_method = params.get("var_method", "parametric")

    def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            from poseidon.core.redis import get_redis
            self._redis = get_redis("cache")
        return self._redis

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        """Evaluate whether portfolio VaR is within acceptable limits.

        CLOSE signals always pass (same pattern as PositionLimitRule).
        When no cached snapshot exists, passes by default with a warning.
        """
        # CLOSE signals always pass -- blocking them creates a deadlock
        if signal.action == SignalAction.CLOSE:
            return RuleResult(passed=True, rule_name=self.name)

        key = f"poseidon:var:latest:{self.var_method}"
        raw = self._get_redis().get(key)

        if raw is None:
            logger.warning("No VaR snapshot at %s, passing by default", key)
            return RuleResult(
                passed=True,
                rule_name=self.name,
                reason="No VaR data available, passing by default",
            )

        snapshot = msgpack.unpackb(raw, raw=False)

        # Check for stale snapshot
        computed_at_str = snapshot.get("computed_at")
        if computed_at_str:
            try:
                computed_at = datetime.fromisoformat(computed_at_str)
                if computed_at.tzinfo is None:
                    computed_at = computed_at.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - computed_at
                if age > _STALE_THRESHOLD:
                    logger.warning(
                        "VaR snapshot at %s is stale (age=%s)", key, age
                    )
            except (ValueError, TypeError):
                logger.warning("Could not parse computed_at from snapshot")

        current_var = snapshot["var_95"]

        if current_var > self.max_var_pct:
            return RuleResult(
                passed=False,
                rule_name=self.name,
                reason=(
                    f"Portfolio VaR ({current_var:.4f}) exceeds limit "
                    f"({self.max_var_pct:.4f}) -- portfolio VaR limit breached"
                ),
            )

        return RuleResult(passed=True, rule_name=self.name)
