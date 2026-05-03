"""RL Order Execution API endpoints (Phase 90 Wave 4b, Plan 90-05b).

Exposes 5 endpoints under the ``/research/rl-execution`` namespace:

* ``POST /run``                — create RLExecutionRun, dispatch Celery task
* ``GET /runs``                — list runs (paginated)
* ``GET /runs/{run_id}``       — detail (status, summary, verdict)
* ``POST /runs/{run_id}/cancel`` — cooperative cancel (mirror BackfillJob pattern)
* ``GET /runs/{run_id}/results`` — stream fills.parquet (only when succeeded)

Task dispatch uses ``celery_app.send_task()`` (NOT direct import of
``qlib_rl_tasks``) to avoid importing qlib in the cp313 API container —
verbatim mirror of ``research_api.py:90-94`` (PATTERNS.md §send_task Dispatch).

Threat-model lifts:
* T-90-02 — auth inherits the existing ``/research/*`` X-API-Key middleware
  via ``main.py`` ``include_router(..., dependencies=secured)``; no new
  per-router auth code.
* T-90-03 — ``RLExecutionRequest`` Pydantic ``Literal`` allowlist + the
  ``_ALLOWED_ALGOS`` defense-in-depth set both reject unknown algo names.
* T-90-04 — ``run_id: UUID`` FastAPI path type validates UUID format BEFORE
  ``parquet_path = AQUARIUM_ROOT / ... / str(run_id) / "fills.parquet"``
  is constructed, so the path cannot escape the results root.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    RLExecutionRequest,
    RLExecutionRunDetailResponse,
    RLExecutionRunListResponse,
    RLExecutionRunResponse,
)
from poseidon.models.base import get_db
from poseidon.models.rl_execution_run import RLExecutionRun
from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app

router = APIRouter()

# Defense-in-depth allowlist (T-90-03). Pydantic Literal already rejects
# unknown algos at request-validation time; this set is a second gate so
# future schema drift cannot silently widen the allowlist.
_ALLOWED_ALGOS: set[str] = {"twap", "vwap", "ppo", "opds"}

# AQUARIUM_ROOT for results path construction (T-90-04). Resolves the same
# way the Celery task does: api/__init__.py is at
# poseidon/src/poseidon/api/rl_execution.py — parents[4] = aquarium/.
AQUARIUM_ROOT = Path(__file__).resolve().parents[4]


@router.post("/run", status_code=202, response_model=RLExecutionRunResponse)
async def create_rl_execution_run(
    request: RLExecutionRequest,
    db: Session = Depends(get_db),
):
    """Create an RL execution run and dispatch to the qlib_queue (EXEC-01).

    Mirrors ``research_api.create_training_run`` 4-step contract:

    1. Defense-in-depth allowlist re-check (T-90-03 layer 2).
    2. Insert pending RLExecutionRun row.
    3. Dispatch Celery task ``poseidon.workers.qlib_tasks.rl_execute`` on
       ``poseidon_qlib`` queue (no direct qlib import in this module).
    4. Return 202 with the row's short response.
    """
    # T-90-03 defense-in-depth (Pydantic Literal already enforced this; we
    # re-check so future schema drift cannot silently widen the allowlist).
    invalid = [a for a in request.algos if a not in _ALLOWED_ALGOS]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown algos: {invalid}")

    run = RLExecutionRun(
        algos=list(request.algos),
        dates=list(request.dates) if request.dates is not None else None,
        mode=request.mode,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # send_task — NOT direct import of qlib_rl_tasks (PATTERNS.md
    # §send_task Dispatch — keeps qlib out of cp313 API container).
    celery_app.send_task(
        "poseidon.workers.qlib_tasks.rl_execute",
        args=[str(run.run_id)],
        queue=POSEIDON_QLIB_QUEUE,
    )

    return RLExecutionRunResponse.model_validate(run)


@router.get("/runs", response_model=RLExecutionRunListResponse)
async def list_rl_execution_runs(
    status: str | None = None,
    verdict: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List RL execution runs with optional filters and pagination."""
    query = db.query(RLExecutionRun)
    if status is not None:
        query = query.filter(RLExecutionRun.status == status)
    if verdict is not None:
        query = query.filter(RLExecutionRun.verdict == verdict)

    total = query.count()
    runs = query.order_by(RLExecutionRun.created_at.desc()).offset(offset).limit(limit).all()

    return RLExecutionRunListResponse(
        runs=[RLExecutionRunResponse.model_validate(r) for r in runs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}", response_model=RLExecutionRunDetailResponse)
async def get_rl_execution_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get full detail for a single RL execution run (D-22)."""
    run = db.query(RLExecutionRun).filter(RLExecutionRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"RL execution run {run_id} not found")
    return RLExecutionRunDetailResponse.model_validate(run)


@router.post("/runs/{run_id}/cancel")
async def cancel_rl_execution_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Cancel a pending or running RL execution run (cooperative cancel).

    The Celery task re-reads ``RLExecutionRun.status`` between every
    (algo, leg) iteration via ``_run_cancelled`` (qlib_rl_tasks.py:75).
    Already-terminal runs (succeeded, failed, cancelled) return 409.
    """
    run = db.query(RLExecutionRun).filter(RLExecutionRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"RL execution run {run_id} not found")
    if run.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel run in status: {run.status}",
        )
    run.status = "cancelled"
    db.commit()
    return {"run_id": str(run.run_id), "status": "cancelled"}


@router.get("/runs/{run_id}/results")
async def get_rl_execution_run_results(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Stream fills.parquet for a succeeded run (D-22, T-90-04 mitigated).

    ``run_id: UUID`` FastAPI path type validates the canonical UUID form
    BEFORE the path is constructed; ``str(run_id)`` is a 36-char hex string
    that cannot escape ``AQUARIUM_ROOT / local_dev/rl-execution/runs/``.
    """
    run = db.query(RLExecutionRun).filter(RLExecutionRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"RL execution run {run_id} not found")
    if run.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Run status is {run.status}, not succeeded — results not available",
        )
    parquet_path = AQUARIUM_ROOT / "local_dev" / "rl-execution" / "runs" / str(run_id) / "fills.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="fills.parquet missing for this run")
    return FileResponse(
        parquet_path,
        media_type="application/octet-stream",
        filename=f"fills_{run_id}.parquet",
    )
