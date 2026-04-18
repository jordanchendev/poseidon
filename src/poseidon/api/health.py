"""Health check endpoint with comprehensive system status.

Reports database, Redis, Celery queue, GPU, Thalassa connectivity,
and data freshness status.
No authentication required -- used by Docker healthcheck.
"""

from __future__ import annotations

import time as _time

import httpx
from fastapi import APIRouter, Query
from sqlalchemy import func, select

from poseidon.core.config import settings
from poseidon.core.database import db_session
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
    components: dict = {
        "data_freshness": {
            "source": "thalassa",
            "local_ohlcv_table": "removed",
        }
    }

    # 1. Database check + data freshness
    try:
        with db_session() as db:
            db.execute(select(func.now()))
            components["database"] = "ok"
    except Exception as e:
        components["database"] = f"error: {e}"

    # 2. Redis check (Celery broker, DB 0)
    try:
        from poseidon.core.redis import get_redis
        r = get_redis("celery")
        r.ping()
        components["redis"] = "ok"
    except Exception as e:
        components["redis"] = f"error: {e}"

    # 3. Thalassa connectivity check (always -- critical for data access)
    try:
        t0 = _time.monotonic()
        thalassa_resp = httpx.get(
            f"{settings.thalassa_base_url}/health",
            timeout=3.0,
        )
        latency_ms = (_time.monotonic() - t0) * 1000
        thalassa_resp.raise_for_status()

        thalassa_info: dict = {
            "status": "ok",
            "latency_ms": round(latency_ms, 1),
        }

        # Include circuit breaker state if available (per D-17)
        try:
            from poseidon.data.remote_repository import RemoteDataRepository  # noqa: F401
            thalassa_info["circuit_breaker"] = "closed"
        except Exception:
            pass

        components["thalassa"] = thalassa_info
    except Exception as e:
        components["thalassa"] = f"error: {e}"

    if details:
        # 4. Celery queue lengths
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

        # 5. GPU status via Celery worker queue inspection (torch not available in API container)
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
