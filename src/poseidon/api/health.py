"""Health check endpoint with comprehensive system status.

Reports database, Redis, Celery queue, GPU, and data freshness status.
No authentication required -- used by Docker healthcheck.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from poseidon.models.base import SessionLocal
from poseidon.models.ohlcv import OHLCV
from poseidon.workers.celery_app import celery_app

router = APIRouter()


@router.get("/health")
async def health(details: bool = Query(False)):
    """Health check endpoint. No authentication required.

    Default mode stays lightweight for Docker liveness checks.
    Pass ``details=true`` to include Celery/GPU inspection.

    Reports system status including:
    - Database connectivity and data freshness
    - Redis connectivity
    - Optional Celery worker queue lengths
    - Optional GPU worker availability

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

    # 2. Redis check (Celery broker, DB 0)
    try:
        from poseidon.core.redis import get_redis
        r = get_redis("celery")
        r.ping()
        components["redis"] = "ok"
    except Exception as e:
        components["redis"] = f"error: {e}"

    if details:
        # 3. Celery queue lengths
        try:
            inspect = celery_app.control.inspect(timeout=1.0)
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
            components["celery"] = {
                "active_tasks": sum(len(v) for v in active.values()),
                "reserved_tasks": sum(len(v) for v in reserved.values()),
            }
        except Exception as e:
            components["celery"] = f"error: {e}"

        # 4. GPU status via Celery worker queue inspection (torch not available in API container)
        try:
            gpu_inspect = celery_app.control.inspect(timeout=1.0)
            active_queues = gpu_inspect.active_queues() or {}
            gpu_workers = [
                worker for worker, queues in active_queues.items()
                if any(q.get("name") == "gpu" for q in queues)
            ]
            if gpu_workers:
                components["gpu"] = {"available": True, "workers": gpu_workers}
            else:
                components["gpu"] = {"available": False, "note": "no GPU workers responding"}
        except Exception as e:
            components["gpu"] = {"available": False, "note": f"inspect failed: {e}"}
    else:
        components["celery"] = {
            "status": "skipped",
            "note": "set details=true to inspect worker queues",
        }
        components["gpu"] = {
            "status": "skipped",
            "note": "set details=true to inspect GPU workers",
        }

    # Overall status: degraded if any component value is an error string
    errors = [
        k
        for k, v in components.items()
        if isinstance(v, str) and v.startswith("error")
    ]
    overall_status = "degraded" if errors else "ok"

    return {"status": overall_status, "components": components}
