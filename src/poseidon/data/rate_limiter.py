"""Distributed rate limiter and circuit breaker for API provider protection.

Uses Redis sorted-set sliding window with Lua atomic scripts for rate limiting,
and Redis-backed state machine for circuit breaker pattern.

Shared across all Celery workers via Redis -- no per-worker state.
"""

import logging
import time
from enum import Enum

import redis

logger = logging.getLogger(__name__)

# ── Lua script for atomic sliding window rate limiting ────────────────
#
# Keys[1] = sorted set key (e.g., poseidon:ratelimit:finmind:3600)
# ARGV[1] = window start timestamp (now - window_seconds)
# ARGV[2] = current timestamp (now)
# ARGV[3] = limit (max allowed requests in window)
# ARGV[4] = TTL for the key (window_seconds + buffer)
# ARGV[5] = unique member value (now + random suffix)
#
# Returns 1 if allowed, 0 if rate limited.

SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local window_start = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local member = ARGV[5]

-- Remove expired entries outside the window
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count current entries in window
local count = redis.call('ZCARD', key)

if count < limit then
    -- Add new entry with current timestamp as score
    redis.call('ZADD', key, now, member)
    -- Set TTL to auto-cleanup
    redis.call('EXPIRE', key, ttl)
    return 1
else
    return 0
end
"""

# ── Provider rate limit configuration ─────────────────────────────────

PROVIDER_LIMITS: dict[str, dict] = {
    "finmind": {"window_seconds": 3600, "limit_key": "ratelimit_finmind_hourly"},
    "yfinance": {"window_seconds": 86400, "limit_key": "ratelimit_yfinance_daily"},
    "ccxt": {"window_seconds": 60, "limit_key": "ratelimit_ccxt_per_minute"},
}


# ── Distributed Rate Limiter ─────────────────────────────────────────


class DistributedRateLimiter:
    """Redis-based distributed sliding window rate limiter.

    Uses Lua script for atomic check-and-increment to prevent race conditions
    across multiple Celery workers.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._script = self._redis.register_script(SLIDING_WINDOW_LUA)
        self._counter = 0  # monotonic counter for unique member values

    def _make_key(
        self,
        provider: str,
        window_seconds: int,
        budget_class: str = "live_ingest",
    ) -> str:
        # Phase 39 D-16/D-17: budget_class isolates bulk backfill traffic from
        # live ingest so a one-shot historical job cannot starve the periodic
        # ingest path's quota.
        return f"poseidon:ratelimit:{provider}:{budget_class}:{window_seconds}"

    def acquire(
        self,
        provider: str,
        window_seconds: int,
        limit: int,
        budget_class: str = "live_ingest",
    ) -> bool:
        """Try to acquire a rate limit slot.

        Returns True if the request is allowed, False if rate limited.
        """
        now = time.time()
        window_start = now - window_seconds
        ttl = window_seconds + 60  # buffer for cleanup
        self._counter += 1
        member = f"{now}:{self._counter}"

        key = self._make_key(provider, window_seconds, budget_class=budget_class)
        result = self._script(
            keys=[key],
            args=[str(window_start), str(now), str(limit), str(ttl), member],
        )
        return bool(result)

    def wait_and_acquire(
        self,
        provider: str,
        window_seconds: int,
        limit: int,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        budget_class: str = "live_ingest",
    ) -> bool:
        """Block until a rate limit slot is available or timeout.

        Returns True if acquired, False if timed out.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.acquire(provider, window_seconds, limit, budget_class=budget_class):
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval, remaining))
        return False

    def get_usage(
        self,
        provider: str,
        window_seconds: int,
        budget_class: str = "live_ingest",
    ) -> int:
        """Get current number of requests in the sliding window."""
        now = time.time()
        window_start = now - window_seconds
        key = self._make_key(provider, window_seconds, budget_class=budget_class)
        # Clean expired entries and count
        self._redis.zremrangebyscore(key, "-inf", str(window_start))
        count = self._redis.zcard(key)
        return count


# ── Circuit Breaker ───────────────────────────────────────────────────


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Redis-backed circuit breaker for API provider protection.

    State transitions:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Provider failing, requests blocked. Auto-recovers after open_timeout via TTL.
    - HALF_OPEN: Not explicitly stored -- when OPEN key expires, state returns CLOSED.

    Uses Redis TTL for automatic OPEN -> CLOSED transition (auto-recovery).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        provider: str,
        failure_threshold: int = 5,
        open_timeout: int = 60,
        failure_window: int = 300,
    ) -> None:
        self._redis = redis_client
        self._provider = provider
        self._failure_threshold = failure_threshold
        self._open_timeout = open_timeout
        self._failure_window = failure_window
        self._state_key = f"poseidon:circuit:{provider}:state"
        self._failures_key = f"poseidon:circuit:{provider}:failures"

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state from Redis."""
        raw = self._redis.get(self._state_key)
        if raw is None:
            return CircuitState.CLOSED
        value = raw.decode() if isinstance(raw, bytes) else raw
        return CircuitState(value)

    def allow_request(self) -> bool:
        """Check if requests are allowed through the circuit breaker."""
        return self.state == CircuitState.CLOSED

    def record_success(self) -> None:
        """Record a successful request -- resets circuit to CLOSED."""
        self._redis.delete(self._state_key)
        self._redis.delete(self._failures_key)
        logger.debug("Circuit breaker %s: success recorded, state CLOSED", self._provider)

    def record_failure(self) -> None:
        """Record a failed request. Opens circuit if threshold exceeded."""
        now = time.time()
        window_start = now - self._failure_window

        # Add failure timestamp to sorted set
        self._redis.zadd(self._failures_key, {str(now): now})

        # Remove old failures outside the window
        self._redis.zremrangebyscore(self._failures_key, "-inf", str(window_start))

        # Count failures in window
        count = self._redis.zcard(self._failures_key)

        if count >= self._failure_threshold:
            # Open the circuit with TTL for auto-recovery
            self._redis.set(self._state_key, CircuitState.OPEN.value, ex=self._open_timeout)
            logger.warning(
                "Circuit breaker %s: OPEN after %d failures (threshold=%d, timeout=%ds)",
                self._provider,
                count,
                self._failure_threshold,
                self._open_timeout,
            )

    def get_health(self) -> dict:
        """Get circuit breaker health status."""
        now = time.time()
        window_start = now - self._failure_window

        # Clean and count
        self._redis.zremrangebyscore(self._failures_key, "-inf", str(window_start))
        failure_count = self._redis.zcard(self._failures_key)

        return {
            "state": self.state.value,
            "failure_count": failure_count,
        }
