"""Redis L1 cache layer for OHLCV data with TTL jitter and single-flight locking."""

import random
import time
from typing import Callable

import msgpack
import pandas as pd
import redis

TTL_CONFIG = {
    "1m": 60,
    "5m": 300,
    "1h": 3600,
    "1d": 3600,
}
TTL_JITTER_MAX = 120


class CacheManager:
    """Cache-aside manager for OHLCV DataFrames stored in Redis.

    Features:
    - Per-interval TTL with random jitter to prevent cache stampede
    - Single-flight locking via SET NX to prevent duplicate API calls
    - Three-layer fallback: Redis cache -> DB -> API provider
    - msgpack serialization for compact DataFrame storage
    """

    KEY_PREFIX = "poseidon:cache"
    LOCK_PREFIX = "poseidon:cache:lock"

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _cache_key(self, symbol: str, interval: str, start: str, end: str) -> str:
        """Build namespaced cache key."""
        return f"{self.KEY_PREFIX}:{symbol}:{interval}:{start}:{end}"

    def _lock_key(self, symbol: str, interval: str, start: str, end: str) -> str:
        """Build single-flight lock key."""
        return f"{self.LOCK_PREFIX}:{symbol}:{interval}:{start}:{end}"

    def _ttl_with_jitter(self, interval: str) -> int:
        """Return base TTL for interval plus random jitter (0-120s)."""
        base = TTL_CONFIG.get(interval, 3600)
        return base + random.randint(0, TTL_JITTER_MAX)

    def _serialize(self, df: pd.DataFrame) -> bytes:
        """Serialize DataFrame to msgpack bytes.

        Stores columns, index (as ISO strings), and values (as nested lists).
        If index is not DatetimeIndex but 'time' column exists, use that.
        """
        if not isinstance(df.index, pd.DatetimeIndex) and "time" in df.columns:
            df = df.set_index("time")
        index_strs = [ts.isoformat() for ts in df.index]
        data = {
            "columns": list(df.columns),
            "index": index_strs,
            "values": df.values.tolist(),
            "tz": str(df.index.tz) if hasattr(df.index, "tz") and df.index.tz is not None else None,
        }
        return msgpack.packb(data, use_bin_type=True)

    def _deserialize(self, raw: bytes) -> pd.DataFrame:
        """Deserialize msgpack bytes back to DataFrame."""
        data = msgpack.unpackb(raw, raw=False)
        df = pd.DataFrame(data["values"], columns=data["columns"])
        tz = data.get("tz")
        idx = pd.to_datetime(data["index"], utc=(tz == "UTC"))
        if tz and tz != "UTC":
            idx = idx.tz_localize("UTC").tz_convert(tz)
        df.index = idx
        return df

    def get(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame | None:
        """Get cached DataFrame. Returns None on cache miss."""
        key = self._cache_key(symbol, interval, start, end)
        raw = self._redis.get(key)
        if raw is None:
            return None
        return self._deserialize(raw)

    def set(self, symbol: str, interval: str, start: str, end: str, df: pd.DataFrame) -> None:
        """Cache a DataFrame with interval-appropriate TTL + jitter."""
        key = self._cache_key(symbol, interval, start, end)
        ttl = self._ttl_with_jitter(interval)
        raw = self._serialize(df)
        self._redis.set(key, raw, ex=ttl)

    def get_or_fetch(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
        db_fetch_fn: Callable[[], pd.DataFrame],
        api_fetch_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Three-layer fallback with single-flight locking.

        1. Try Redis cache
        2. Acquire single-flight lock (SET NX, 30s expiry)
        3. If lock acquired: try DB, then API, cache result, release lock
        4. If lock NOT acquired: poll cache for up to 30s
        """
        # Layer 1: Cache hit
        cached = self.get(symbol, interval, start, end)
        if cached is not None:
            return cached

        lock_key = self._lock_key(symbol, interval, start, end)
        acquired = self._redis.set(lock_key, "1", nx=True, ex=30)

        if acquired:
            try:
                # Layer 2: DB fetch
                db_result = db_fetch_fn()
                if db_result is not None and not db_result.empty:
                    self.set(symbol, interval, start, end, db_result)
                    return db_result

                # Layer 3: API fetch
                api_result = api_fetch_fn()
                if api_result is not None and not api_result.empty:
                    self.set(symbol, interval, start, end, api_result)
                    return api_result

                return pd.DataFrame()
            finally:
                self._redis.delete(lock_key)
        else:
            # Another worker has the lock -- poll cache
            for _ in range(30):
                time.sleep(1)
                cached = self.get(symbol, interval, start, end)
                if cached is not None:
                    return cached
            # Timeout: fall through to own fetch
            db_result = db_fetch_fn()
            if db_result is not None and not db_result.empty:
                self.set(symbol, interval, start, end, db_result)
                return db_result
            api_result = api_fetch_fn()
            if api_result is not None and not api_result.empty:
                self.set(symbol, interval, start, end, api_result)
                return api_result
            return pd.DataFrame()
