"""Factor analysis API endpoints (Phase 47)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    CentralityAnalysisRequest,
    FactorAnalysisRunListResponse,
    FactorAnalysisRunResponse,
    FactorAnalysisTriggerResponse,
    ICAnalysisRequest,
    ShapleyAnalysisRequest,
)
from poseidon.models.base import get_db
from poseidon.models.factor_analysis_run import FactorAnalysisRun
from poseidon.models.model_version import ModelVersion
from poseidon.models.training_run import TrainingRun
from poseidon.workers.celery_app import (
    POSEIDON_CPU_QUEUE,
    POSEIDON_QLIB_QUEUE,
    celery_app,
)

router = APIRouter()


def _serialize_run(run: FactorAnalysisRun) -> FactorAnalysisRunResponse:
    return FactorAnalysisRunResponse(
        id=str(run.id),
        run_type=run.run_type,
        config_json=run.config_json,
        results_json=run.results_json,
        status=run.status,
        market=run.market,
        error=run.error,
        created_at=run.created_at.isoformat(),
        updated_at=run.updated_at.isoformat(),
    )


@router.post("/ic", status_code=202, response_model=FactorAnalysisTriggerResponse)
async def create_ic_analysis_run(
    request: ICAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Create an IC analysis run and dispatch it to the CPU worker."""
    run = FactorAnalysisRun(
        run_type="ic",
        config_json=request.model_dump(),
        status="pending",
        market=request.market,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    celery_app.send_task(
        "poseidon.workers.cpu_tasks.factor_ic_analysis",
        args=[str(run.id)],
        queue=POSEIDON_CPU_QUEUE,
    )
    return FactorAnalysisTriggerResponse(id=str(run.id), status=run.status)


@router.post("/shapley", status_code=202, response_model=FactorAnalysisTriggerResponse)
async def create_shapley_analysis_run(
    request: ShapleyAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Create a SHAP analysis run and dispatch it to qlib-research."""
    model_version = (
        db.query(ModelVersion)
        .filter(ModelVersion.id == UUID(request.model_version_id))
        .first()
    )
    if model_version is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    if not model_version.artifact_path:
        raise HTTPException(
            status_code=409,
            detail="Model version has no artifacts available for SHAP analysis",
        )

    training_run = (
        db.query(TrainingRun)
        .filter(TrainingRun.model_version_id == model_version.id)
        .order_by(TrainingRun.created_at.desc())
        .first()
    )
    if training_run is None:
        raise HTTPException(
            status_code=404,
            detail="No training run found for model version",
        )

    run = FactorAnalysisRun(
        run_type="shapley",
        config_json=request.model_dump(),
        status="pending",
        market=training_run.market,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    celery_app.send_task(
        "poseidon.workers.qlib_tasks.factor_shapley_analysis",
        args=[str(run.id)],
        queue=POSEIDON_QLIB_QUEUE,
    )
    return FactorAnalysisTriggerResponse(id=str(run.id), status=run.status)


@router.post("/centrality", status_code=202, response_model=FactorAnalysisTriggerResponse)
async def create_centrality_analysis_run(
    request: CentralityAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Create a centrality analysis run and dispatch it to the CPU worker."""
    run = FactorAnalysisRun(
        run_type="centrality",
        config_json=request.model_dump(),
        status="pending",
        market=request.market,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    celery_app.send_task(
        "poseidon.workers.cpu_tasks.factor_centrality_analysis",
        args=[str(run.id)],
        queue=POSEIDON_CPU_QUEUE,
    )
    return FactorAnalysisTriggerResponse(id=str(run.id), status=run.status)


@router.get("/runs", response_model=FactorAnalysisRunListResponse)
async def list_factor_analysis_runs(
    run_type: str | None = None,
    market: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List factor analysis runs with optional filters."""
    query = db.query(FactorAnalysisRun)
    if run_type is not None:
        query = query.filter(FactorAnalysisRun.run_type == run_type)
    if market is not None:
        query = query.filter(FactorAnalysisRun.market == market)

    total = query.count()
    runs = (
        query.order_by(FactorAnalysisRun.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return FactorAnalysisRunListResponse(
        runs=[_serialize_run(run) for run in runs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}", response_model=FactorAnalysisRunResponse)
async def get_factor_analysis_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Return full detail for a single factor analysis run."""
    run = db.query(FactorAnalysisRun).filter(FactorAnalysisRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Factor analysis run not found")
    return _serialize_run(run)
