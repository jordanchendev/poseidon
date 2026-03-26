"""AutoResearch API endpoints.

Provides endpoints for dispatching autoresearch runs (Celery task returning
202 + task_id), polling task status, stopping a running task, and querying
experiment results.
"""

from __future__ import annotations

from datetime import datetime

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from poseidon.core.schemas import MessageResponse
from poseidon.models.base import get_db
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks import autoresearch_run

router = APIRouter()


# --------------- Pydantic schemas ---------------


class MarketSpecSchema(BaseModel):
    """A single market to search."""

    symbol: str = Field(..., examples=["BTCUSDT"])
    market: str = Field(..., examples=["crypto_spot"])
    interval: str = Field("1h", examples=["1h", "1d"])


class AutoResearchRequest(BaseModel):
    """Request body for dispatching an autoresearch run."""

    markets: list[MarketSpecSchema] = Field(
        ..., min_length=1, examples=[[{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}]]
    )
    n_trials: int = Field(50, ge=5, le=200)
    min_wfe: float = Field(0.50, ge=0.0, le=1.0)
    seed: int = Field(42)


class TaskStatusResponse(BaseModel):
    """Response for task status polling."""

    task_id: str
    status: str
    progress: dict | None = None
    result: dict | None = None
    error: str | None = None


class ExperimentResponse(BaseModel):
    """Response for a single experiment record."""

    id: int
    study_name: str
    market: str
    interval: str
    composite_score: float
    wfe_score: float
    status: str
    config_json: dict
    metrics_json: dict
    created_at: datetime

    model_config = {"from_attributes": True}


# --------------- Endpoints ---------------


@router.post("/run", response_model=MessageResponse, status_code=202)
async def run_autoresearch(request: AutoResearchRequest) -> MessageResponse:
    """Dispatch an autoresearch run to the CPU worker queue.

    Returns 202 with a Celery task_id for status polling via GET /autoresearch/status/{task_id}.
    """
    search_config = {
        "n_trials": request.n_trials,
        "min_wfe": request.min_wfe,
        "seed": request.seed,
    }
    markets = [m.model_dump() for m in request.markets]

    task = autoresearch_run.delay(search_config, markets)
    return MessageResponse(
        message=f"AutoResearch dispatched: {len(markets)} market(s), {request.n_trials} trials",
        task_id=task.id,
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Poll autoresearch task status.

    States: PENDING -> PROGRESS -> SUCCESS / FAILURE
    """
    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(task_id=task_id, status=result.status)

    if result.status == "PROGRESS" and isinstance(result.info, dict):
        response.progress = result.info
    elif result.status == "SUCCESS":
        response.result = result.result
    elif result.status == "FAILURE":
        response.error = str(result.result)

    return response


@router.post("/stop/{task_id}", response_model=MessageResponse)
async def stop_autoresearch(task_id: str) -> MessageResponse:
    """Request graceful stop for a running autoresearch task.

    Sets Redis flag that the runner checks between markets.
    """
    import redis as redis_lib

    redis_client = redis_lib.from_url(celery_app.conf.broker_url)
    redis_client.set(f"autoresearch:stop:{task_id}", "1", ex=3600)
    return MessageResponse(message=f"Stop requested for task {task_id}")


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_experiments(
    study_name: str | None = Query(None, examples=["crypto_spot_BTCUSDT_1h"]),
    status: str | None = Query(None, examples=["passed"]),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[ExperimentResponse]:
    """List experiment results, optionally filtered by study name and status."""
    from poseidon.models.experiment import ExperimentRecord

    query = db.query(ExperimentRecord)
    if study_name:
        query = query.filter(ExperimentRecord.study_name == study_name)
    if status:
        query = query.filter(ExperimentRecord.status == status)
    query = query.order_by(ExperimentRecord.composite_score.desc())
    return query.limit(limit).all()
