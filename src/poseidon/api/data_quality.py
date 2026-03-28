"""Data quality API endpoints -- provider health monitoring."""

import logging

import redis as redis_lib
from fastapi import APIRouter

from poseidon.core.config import settings
from poseidon.data.rate_limiter import (
    CircuitBreaker,
    DistributedRateLimiter,
    PROVIDER_LIMITS,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_redis_client() -> redis_lib.Redis:
    """Create a Redis client for provider health queries."""
    return redis_lib.from_url(settings.redis_url, decode_responses=False)


@router.get("/providers")
def get_provider_health():
    """Get health status for all data providers.

    Returns circuit breaker state, quota usage, and configured limits
    for finmind, yfinance, and ccxt providers.
    """
    redis_client = _get_redis_client()
    rate_limiter = DistributedRateLimiter(redis_client)
    providers = {}

    for provider, cfg in PROVIDER_LIMITS.items():
        window = cfg["window_seconds"]
        limit = getattr(settings, cfg["limit_key"], 0)
        circuit = CircuitBreaker(
            redis_client,
            provider,
            failure_threshold=settings.circuit_failure_threshold,
            open_timeout=settings.circuit_open_timeout,
            failure_window=settings.circuit_failure_window,
        )
        health = circuit.get_health()
        usage = rate_limiter.get_usage(provider, window)
        providers[provider] = {
            "circuit_state": health["state"],
            "failure_count": health["failure_count"],
            "quota_used": usage,
            "quota_limit": limit,
            "window_seconds": window,
        }

    return {"providers": providers}
