"""Data quality API endpoints -- provider health and quality scores."""

import logging

import redis as redis_lib
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel as PydanticBase
from sqlalchemy import desc
from sqlalchemy.orm import Session

from poseidon.core.config import settings
from poseidon.data.rate_limiter import (
    CircuitBreaker,
    DistributedRateLimiter,
    PROVIDER_LIMITS,
)
from poseidon.models.base import get_db
from poseidon.models.quality_score import QualityScore

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic response schemas for quality scores (RAPI-05) ---


class QualityScoreResponse(PydanticBase):
    time: str
    symbol: str
    interval: str
    score: float
    completeness: float
    consistency: float
    anomaly_free: float
    timeliness: float


class QualityScoresResponse(PydanticBase):
    scores: list[QualityScoreResponse]


def _get_redis_client() -> redis_lib.Redis:
    """Redis client for rate limiter and circuit breaker state (ratelimit, DB 3)."""
    from poseidon.core.redis import get_redis
    return get_redis("ratelimit")


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


# --- RAPI-05: GET /scores ---


@router.get("/scores", response_model=QualityScoresResponse)
def get_quality_scores(
    symbol: str | None = Query(None, description="Filter by symbol"),
    interval: str | None = Query(None, description="Filter by interval"),
    limit: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Return data quality scores by symbol/interval (RAPI-05, per D-13)."""
    query = db.query(QualityScore).order_by(desc(QualityScore.time))
    if symbol:
        query = query.filter(QualityScore.symbol == symbol)
    if interval:
        query = query.filter(QualityScore.interval == interval)
    rows = query.limit(limit).all()
    scores = [
        QualityScoreResponse(
            time=str(row.time),
            symbol=row.symbol,
            interval=row.interval,
            score=float(row.score),
            completeness=float(row.completeness),
            consistency=float(row.consistency),
            anomaly_free=float(row.anomaly_free),
            timeliness=float(row.timeliness),
        )
        for row in rows
    ]
    return QualityScoresResponse(scores=scores)
