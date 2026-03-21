"""Signal delivery via Redis Streams.

Passed signals are written to market-specific Redis Streams
(poseidon:signals:{market}) with consumer group support and
7-day retention via MINID trimming.
"""

import json
import time

import redis

from poseidon.signals.schemas import Signal, SignalStatus


class SignalDeliveryService:
    """Delivers passed signals to Redis Streams."""

    STREAM_PREFIX = "poseidon:signals"
    CONSUMER_GROUP = "default"
    RETENTION_DAYS = 7

    def __init__(self, redis_url: str):
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def deliver(self, signal: Signal) -> str | None:
        """Write a passed signal to the appropriate Redis Stream.

        Returns the stream message ID, or None if signal is not PASSED.
        Rejected and pending signals are NOT pushed to Redis.
        """
        if signal.status != SignalStatus.PASSED:
            return None

        stream_key = f"{self.STREAM_PREFIX}:{signal.market}"
        fields: dict[str, str] = {
            "id": str(signal.id),
            "action": signal.action.value,
            "symbol": signal.symbol,
            "market": signal.market,
            "instrument": signal.instrument.value,
            "confidence": str(signal.confidence),
            "signal_time": signal.signal_time.isoformat(),
            "params": json.dumps(signal.params),
            "interval": signal.interval,
        }
        if signal.quantity_pct is not None:
            fields["quantity_pct"] = str(signal.quantity_pct)
        if signal.strategy_id is not None:
            fields["strategy_id"] = str(signal.strategy_id)
        if signal.model_id is not None:
            fields["model_id"] = str(signal.model_id)

        # Ensure consumer group exists before writing
        self._ensure_consumer_group(stream_key)

        # XADD with MINID for approximate 7-day retention
        min_id = self._retention_minid()
        msg_id = self._redis.xadd(stream_key, fields, minid=min_id, approximate=True)
        return msg_id

    def _ensure_consumer_group(self, stream_key: str) -> None:
        """Create consumer group if it doesn't exist.

        Catches BUSYGROUP error (group already exists) and ignores it.
        """
        try:
            self._redis.xgroup_create(
                name=stream_key,
                groupname=self.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _retention_minid(self) -> str:
        """Calculate MINID for 7-day retention (millisecond timestamp)."""
        cutoff_ms = int((time.time() - self.RETENTION_DAYS * 86400) * 1000)
        return f"{cutoff_ms}-0"

    def trim_streams(self, markets: list[str]) -> dict[str, int]:
        """Explicit trim for Celery Beat scheduled cleanup.

        For each market, trims entries older than 7 days using MINID.
        Returns dict of market -> number of removed entries.
        """
        min_id = self._retention_minid()
        results: dict[str, int] = {}
        for market in markets:
            stream_key = f"{self.STREAM_PREFIX}:{market}"
            removed = self._redis.xtrim(stream_key, minid=min_id, approximate=True)
            results[market] = removed
        return results
