"""Health check endpoint with comprehensive system status.

Reports database, Redis, Celery queue, GPU, and data freshness status.
No authentication required -- used by Docker healthcheck.
"""

from __future__ import annotations

import redis
from fastapi import APIRouter
from sqlalchemy import func, select

from poseidon.core.config import settings
from poseidon.models.base import SessionLocal
from poseidon.models.ohlcv import OHLCV
from poseidon.workers.celery_app import celery_app

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint. No authentication required.

    Reports comprehensive system status including:
    - Database connectivity and data freshness
    - Redis connectivity
    - Celery worker queue lengths
    - GPU availability and memory

    Returns ``{"status": "ok"}`` when all components are healthy,
    ``{"status": "degraded"}`` when any component reports an error.
    """
    components: dict = {}

    # 1. Database check + data freshness
    db = None
    try:
        db = SessionLocal()
        db.execute(select(func.now()))
        components["database"] = "ok"

        # Data freshness: latest OHLCV timestamp
        latest = db.execute(select(func.max(OHLCV.time))).scalar()
        components["data_freshness"] = {
            "latest_ohlcv": latest.isoformat() if latest else None,
        }
    except Exception as e:
        components["database"] = f"error: {e}"
        components["data_freshness"] = {"latest_ohlcv": None}
    finally:
        if db is not None:
            db.close()

    # 2. Redis check
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        components["redis"] = "ok"
    except Exception as e:
        components["redis"] = f"error: {e}"

    # 3. Celery queue lengths
    try:
        inspect = celery_app.control.inspect(timeout=2.0)
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        components["celery"] = {
            "active_tasks": sum(len(v) for v in active.values()),
            "reserved_tasks": sum(len(v) for v in reserved.values()),
        }
    except Exception as e:
        components["celery"] = f"error: {e}"

    # 4. GPU status (best-effort -- torch may not be installed in API container)
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            components["gpu"] = {
                "available": True,
                "free_mb": free // (1024 * 1024),
                "total_mb": total // (1024 * 1024),
            }
        else:
            components["gpu"] = {"available": False}
    except ImportError:
        components["gpu"] = {"available": False, "note": "torch not installed"}

    # Overall status: degraded if any component value is an error string
    errors = [
        k
        for k, v in components.items()
        if isinstance(v, str) and v.startswith("error")
    ]
    overall_status = "degraded" if errors else "ok"

    return {"status": overall_status, "components": components}
