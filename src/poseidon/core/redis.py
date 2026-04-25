"""Centralized Redis client factory.

Each Redis purpose maps to a dedicated DB index:
- celery (DB 0): Celery broker/backend + RedBeat
- cache (DB 1): OHLCV cache, VaR snapshots, alert streams
- stream (DB 2): Signal delivery to Thalassa
- ratelimit (DB 3): Rate limiter sorted sets + circuit breaker state
"""

import redis as redis_lib

from poseidon.core.config import settings

_PURPOSE_URL = {
    "celery": lambda: settings.redis_celery_url,
    "cache": lambda: settings.redis_cache_url,
    "stream": lambda: settings.redis_stream_url,
    "ratelimit": lambda: settings.redis_ratelimit_url,
}


def get_redis(purpose: str, *, decode_responses: bool = False) -> redis_lib.Redis:
    """Return a Redis client for the specified purpose.

    Args:
        purpose: One of "celery", "cache", "stream", "ratelimit".
        decode_responses: Pass True for text-mode clients (signal streams).

    Raises:
        ValueError: If purpose is not recognized.
    """
    url_fn = _PURPOSE_URL.get(purpose)
    if url_fn is None:
        raise ValueError(f"Unknown Redis purpose {purpose!r}. Choose from: {sorted(_PURPOSE_URL)}")
    return redis_lib.from_url(url_fn(), decode_responses=decode_responses)
