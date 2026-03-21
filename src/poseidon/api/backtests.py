"""Backtest API endpoints.

Provides endpoints for dispatching backtests and parameter optimizations
(as Celery tasks returning 202 + task_id) and querying completed results.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel as PydanticBase
from pydantic import ConfigDict
from sqlalchemy.orm import Session

from poseidon.backtest.repository import BacktestRepository
from poseidon.core.schemas import MessageResponse
from poseidon.models.base import get_db
from poseidon.workers.cpu_tasks import run_backtest_task, run_optimization_task

router = APIRouter()


# --------------- Pydantic schemas ---------------


class BacktestRunRequest(PydanticBase):
    """Request body for dispatching a backtest run."""

    strategy_id: uuid.UUID
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 1_000_000.0


class OptimizeRequest(PydanticBase):
    """Request body for dispatching a parameter optimization."""

    strategy_id: uuid.UUID
    param_grid: dict
    method: str = "grid"
    n_trials: int = 50
    target_metric: str = "sharpe_ratio"
    start_date: str | None = None
    end_date: str | None = None


class BacktestResponse(PydanticBase):
    """Response model for a backtest record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    strategy_id: uuid.UUID | None
    strategy_type: str
    symbol: str
    market: str
    interval: str
    config: dict
    metrics: dict | None
    walk_forward: dict | None
    status: str
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None


# --------------- Endpoints ---------------


@router.post("/run", response_model=MessageResponse, status_code=202)
async def run_backtest(request: BacktestRunRequest) -> MessageResponse:
    """Dispatch a backtest to the CPU worker queue.

    Returns 202 with a Celery task_id for status polling.
    """
    task = run_backtest_task.delay(
        str(request.strategy_id),
        request.start_date,
        request.end_date,
        request.initial_capital,
    )
    return MessageResponse(
        message=f"Backtest dispatched for strategy {request.strategy_id}",
        task_id=task.id,
    )


@router.post("/optimize", response_model=MessageResponse, status_code=202)
async def run_optimization(request: OptimizeRequest) -> MessageResponse:
    """Dispatch a parameter optimization to the CPU worker queue.

    Returns 202 with a Celery task_id for status polling.
    """
    task = run_optimization_task.delay(
        str(request.strategy_id),
        request.param_grid,
        request.method,
        request.n_trials,
        request.target_metric,
        request.start_date,
        request.end_date,
    )
    return MessageResponse(
        message=f"Optimization dispatched for strategy {request.strategy_id}",
        task_id=task.id,
    )


@router.get("", response_model=list[BacktestResponse])
async def list_backtests(
    strategy_id: uuid.UUID | None = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[BacktestResponse]:
    """List completed backtests, optionally filtered by strategy_id."""
    repo = BacktestRepository(db)
    return repo.list_backtests(strategy_id=strategy_id, limit=limit)


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(
    backtest_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> BacktestResponse:
    """Get a single backtest result by ID."""
    repo = BacktestRepository(db)
    record = repo.get_by_id(backtest_id)
    if not record:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return record
