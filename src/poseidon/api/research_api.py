"""Research API endpoints for training run lifecycle (Phase 41).

Exposes POST /train, GET /runs, GET /runs/{run_id}, POST /runs/{run_id}/cancel.
Task dispatch uses ``celery_app.send_task()`` (NOT direct import of qlib_tasks)
to avoid importing qlib/mlflow in the cp313 API container.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    TrainRequest,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    TrainingRunResponse,
)
from poseidon.models.base import get_db
from poseidon.models.training_run import TrainingRun
from poseidon.qlib.allowlist import resolve_handler, resolve_model
from poseidon.workers.celery_app import celery_app

router = APIRouter()


@router.post("/train", status_code=202, response_model=TrainingRunResponse)
async def create_training_run(
    request: TrainRequest,
    db: Session = Depends(get_db),
):
    """Create a training run and dispatch to qlib_queue (RESEARCH-API-01).

    Validates handler_class and model_class against the static allowlist
    (D-08/D-09). Unknown classes are rejected with 422 before any DB write.
    Uses ``send_task`` to dispatch -- the qlib_tasks module is NOT imported
    in this file to avoid pulling qlib/mlflow into the cp313 process.
    """
    # Validate against allowlist (RESEARCH-API-02)
    try:
        resolve_handler(request.handler_class)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        resolve_model(request.model_class)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Create TrainingRun row
    run = TrainingRun(
        handler_class=request.handler_class,
        handler_params=request.handler_params,
        model_class=request.model_class,
        model_params=request.model_params,
        market=request.market,
        symbols=request.symbols,
        interval=request.interval,
        segments=request.segments,
        lookback=request.lookback,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Dispatch to qlib_queue via send_task (no direct import of qlib_tasks)
    celery_app.send_task(
        "poseidon.workers.qlib_tasks.qlib_train",
        args=[str(run.run_id)],
        queue="qlib_queue",
    )

    return TrainingRunResponse.model_validate(run)


@router.get("/runs", response_model=TrainingRunListResponse)
async def list_training_runs(
    status: str | None = None,
    market: str | None = None,
    handler_class: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List training runs with optional filters and pagination (RESEARCH-API-04).

    Query params: status, market, handler_class, limit (1-200, default 50),
    offset (default 0). Results ordered by created_at descending.
    """
    query = db.query(TrainingRun)
    if status is not None:
        query = query.filter(TrainingRun.status == status)
    if market is not None:
        query = query.filter(TrainingRun.market == market)
    if handler_class is not None:
        query = query.filter(TrainingRun.handler_class == handler_class)

    total = query.count()
    runs = query.order_by(TrainingRun.created_at.desc()).offset(offset).limit(limit).all()

    return TrainingRunListResponse(
        runs=[TrainingRunResponse.model_validate(r) for r in runs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}", response_model=TrainingRunDetailResponse)
async def get_training_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get full detail for a single training run (RESEARCH-API-04).

    Returns all fields including metrics, model_version_id, error text.
    Raises 404 if run_id does not exist.
    """
    run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Training run {run_id} not found"
        )
    return TrainingRunDetailResponse.model_validate(run)


@router.post("/runs/{run_id}/cancel")
async def cancel_training_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Cancel a pending or running training run (RESEARCH-API-05).

    Only pending/running runs can be cancelled. Already-terminal runs
    (succeeded, failed, cancelled) return 409.
    """
    run = db.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Training run {run_id} not found"
        )
    if run.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel run in status: {run.status}",
        )
    run.status = "cancelled"
    db.commit()
    return {"run_id": str(run.run_id), "status": "cancelled"}
