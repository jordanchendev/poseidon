"""Model lifecycle API endpoints.

Provides REST endpoints for model training dispatch, version listing,
lifecycle transitions (shadow/activate), and prediction requests (API-01).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel as PydanticBase
from pydantic import ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.core.schemas import MessageResponse
from poseidon.ml.lifecycle import InvalidTransitionError
from poseidon.ml.manager import ModelManager
from poseidon.models.base import get_db
from poseidon.models.model_version import ModelVersion
from poseidon.workers.gpu_tasks import run_prediction, train_model

router = APIRouter()


# --------------- Pydantic schemas ---------------


class TrainRequest(PydanticBase):
    """Request body for dispatching a model training job."""

    model_name: str
    symbol: str
    market: str
    interval: str = "1d"
    params: dict = {}
    feature_list: list[str] = []
    start_date: str | None = None
    end_date: str | None = None
    label_mode: str = "direction"  # "direction" or "regime"


class ModelVersionResponse(PydanticBase):
    """Response model for a model version."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: int
    status: str
    params: dict
    metrics: dict | None
    feature_list: list
    train_start: datetime | None
    train_end: datetime | None
    artifact_path: str | None
    created_at: datetime
    error_message: str | None


class TransitionRequest(PydanticBase):
    """Optional body for lifecycle transitions (e.g. attach metrics)."""

    metrics: dict | None = None


class PredictRequest(PydanticBase):
    """Request body for model prediction."""

    symbol: str
    market: str
    interval: str = "1d"
    use_latest_data: bool = True


# --------------- Endpoints ---------------


@router.post("/train", response_model=MessageResponse, status_code=202)
async def start_training(
    body: TrainRequest,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Start model training.

    Creates a new ModelVersion in ``training`` status and dispatches
    a Celery task to the GPU queue for the actual training work.
    """
    manager = ModelManager(db)
    mv = manager.create_version(
        name=body.model_name,
        params=body.params,
        feature_list=body.feature_list,
    )

    task = train_model.delay(
        model_name=body.model_name,
        version_id=str(mv.id),
        symbol=body.symbol,
        market=body.market,
        interval=body.interval,
        params={**body.params, "label_mode": body.label_mode},
        feature_list=body.feature_list,
        start_date=body.start_date,
        end_date=body.end_date,
    )

    return MessageResponse(
        message=f"Training dispatched for {body.model_name} v{mv.version}",
        task_id=task.id,
    )


@router.get("", response_model=list[ModelVersionResponse])
async def list_models(
    name: str | None = None,
    db: Session = Depends(get_db),
) -> list[ModelVersionResponse]:
    """List model versions.

    If ``name`` is provided, returns versions for that model name only.
    Otherwise returns the most recent 50 model versions across all models.
    """
    manager = ModelManager(db)

    if name:
        return manager.list_versions(name)

    # All models, newest first, limited to 50
    rows = list(db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(50)).scalars())
    return rows


@router.get("/{version_id}", response_model=ModelVersionResponse)
async def get_model_version(
    version_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    """Get a single model version by ID."""
    manager = ModelManager(db)
    mv = manager.get_version(version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return mv


@router.post("/{version_id}/shadow", response_model=ModelVersionResponse)
async def shadow_model(
    version_id: uuid.UUID,
    body: TransitionRequest | None = None,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    """Transition a model version to shadow mode."""
    manager = ModelManager(db)
    try:
        mv = manager.transition(
            version_id,
            "shadow",
            metrics=body.metrics if body else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return mv


@router.post("/{version_id}/activate", response_model=ModelVersionResponse)
async def activate_model(
    version_id: uuid.UUID,
    body: TransitionRequest | None = None,
    db: Session = Depends(get_db),
) -> ModelVersionResponse:
    """Activate a model version (retires any previously active version of the same model)."""
    manager = ModelManager(db)
    try:
        mv = manager.transition(
            version_id,
            "active",
            metrics=body.metrics if body else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return mv


@router.post("/{version_id}/predict", response_model=MessageResponse, status_code=202)
async def predict(
    version_id: uuid.UUID,
    body: PredictRequest,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Dispatch model prediction to GPU worker.

    Returns 202 Accepted with task_id for async tracking.
    The GPU worker loads the model, runs inference via ModelStrategy,
    filters predictions by confidence threshold, and pipes signals
    through SignalPipeline (risk check -> persist -> deliver).
    """
    manager = ModelManager(db)
    mv = manager.get_version(version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    if mv.status not in ("ready", "shadow", "active"):
        raise HTTPException(
            status_code=400,
            detail=f"Model version status '{mv.status}' cannot run prediction",
        )

    task = run_prediction.delay(
        version_id=str(version_id),
        symbol=body.symbol,
        market=body.market,
        interval=body.interval,
    )
    return MessageResponse(
        message=f"Prediction dispatched for {mv.name} v{mv.version}",
        task_id=task.id,
    )
