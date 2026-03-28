"""Tests for distributed rate limiter and circuit breaker."""

import time

import pytest

from poseidon.data.rate_limiter import (
    PROVIDER_LIMITS,
    CircuitBreaker,
    CircuitState,
    DistributedRateLimiter,
)


# ── Sliding window rate limiter tests ────────────────────────────────


class TestSlidingWindowRateLimiter:
    """Tests for DistributedRateLimiter using Lua-atomic sliding window."""

    def test_sliding_window_allows_within_limit(self, fake_redis):
        limiter = DistributedRateLimiter(fake_redis)
        results = [limiter.acquire("test_provider", window_seconds=60, limit=5) for _ in range(5)]
        assert all(results), "All 5 calls within limit should return True"

    def test_sliding_window_blocks_over_limit(self, fake_redis):
        limiter = DistributedRateLimiter(fake_redis)
        for _ in range(5):
            limiter.acquire("test_provider", window_seconds=60, limit=5)
        result = limiter.acquire("test_provider", window_seconds=60, limit=5)
        assert result is False, "6th call should be blocked"

    def test_sliding_window_lua_atomic(self, fake_redis):
        limiter = DistributedRateLimiter(fake_redis)
        limiter.acquire("myprovider", window_seconds=3600, limit=100)
        # Verify the key pattern used by Lua script
        keys = [k.decode() for k in fake_redis.keys("poseidon:ratelimit:*")]
        assert any("poseidon:ratelimit:myprovider:3600" in k for k in keys)

    def test_get_usage(self, fake_redis):
        limiter = DistributedRateLimiter(fake_redis)
        assert limiter.get_usage("test_provider", window_seconds=60) == 0
        limiter.acquire("test_provider", window_seconds=60, limit=10)
        limiter.acquire("test_provider", window_seconds=60, limit=10)
        limiter.acquire("test_provider", window_seconds=60, limit=10)
        assert limiter.get_usage("test_provider", window_seconds=60) == 3

    def test_wait_and_acquire_blocks(self, fake_redis):
        limiter = DistributedRateLimiter(fake_redis)
        # Fill up the limit
        for _ in range(3):
            limiter.acquire("test_provider", window_seconds=60, limit=3)
        # wait_and_acquire should timeout since no slots will open
        result = limiter.wait_and_acquire(
            "test_provider", window_seconds=60, limit=3, timeout=1.0, poll_interval=0.3
        )
        assert result is False, "Should timeout since window doesn't expire in 1s"


# ── Provider config tests ────────────────────────────────────────────


class TestProviderConfig:
    """Tests for PROVIDER_LIMITS configuration."""

    def test_provider_config_finmind(self):
        cfg = PROVIDER_LIMITS["finmind"]
        assert cfg["window_seconds"] == 3600
        assert cfg["limit_key"] == "ratelimit_finmind_hourly"

    def test_provider_config_yfinance(self):
        cfg = PROVIDER_LIMITS["yfinance"]
        assert cfg["window_seconds"] == 86400
        assert cfg["limit_key"] == "ratelimit_yfinance_daily"

    def test_provider_config_ccxt(self):
        cfg = PROVIDER_LIMITS["ccxt"]
        assert cfg["window_seconds"] == 60
        assert cfg["limit_key"] == "ratelimit_ccxt_per_minute"


# ── Circuit breaker tests ────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for CircuitBreaker with Redis-backed state."""

    def test_circuit_closed_allows(self, fake_redis):
        cb = CircuitBreaker(fake_redis, "test_provider")
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_circuit_opens_after_threshold(self, fake_redis):
        cb = CircuitBreaker(fake_redis, "test_provider", failure_threshold=5)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_circuit_half_open_after_timeout(self, fake_redis):
        cb = CircuitBreaker(fake_redis, "test_provider", failure_threshold=5, open_timeout=1)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Wait for TTL to expire
        time.sleep(1.5)
        # After TTL expires, Redis key is gone -> state returns CLOSED (auto-recovery)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_circuit_success_resets(self, fake_redis):
        cb = CircuitBreaker(fake_redis, "test_provider", failure_threshold=5)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_get_health(self, fake_redis):
        cb = CircuitBreaker(fake_redis, "test_provider")
        health = cb.get_health()
        assert health["state"] == "CLOSED"
        assert health["failure_count"] == 0

    def test_get_health_after_failures(self, fake_redis):
        cb = CircuitBreaker(fake_redis, "test_provider", failure_threshold=5)
        for _ in range(3):
            cb.record_failure()
        health = cb.get_health()
        assert health["failure_count"] == 3


# ── Settings integration tests ───────────────────────────────────────


class TestSettingsRateLimitFields:
    """Tests for rate limit fields on Settings class."""

    def test_settings_has_ratelimit_fields(self):
        from poseidon.core.config import Settings

        s = Settings(
            api_key="test",
            database_url="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379/0",
        )
        assert s.ratelimit_finmind_hourly == 500
        assert s.ratelimit_yfinance_daily == 900
        assert s.ratelimit_ccxt_per_minute == 1200
        assert s.circuit_failure_threshold == 5
        assert s.circuit_open_timeout == 60
        assert s.cache_enabled is True
