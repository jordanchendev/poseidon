"""Backtest API endpoints.

Provides endpoints for dispatching backtests and parameter optimizations
(as Celery tasks returning 202 + task_id) and querying completed results.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel as PydanticBase
from pydantic import ConfigDict, Field
from sqlalchemy.orm import Session

from poseidon.backtest.repository import BacktestRepository
from poseidon.core.schemas import MessageResponse
from poseidon.models.base import get_db
from poseidon.workers.cpu_tasks import run_backtest_task, run_dual_mode_task, run_optimization_task

router = APIRouter()


# --------------- Pydantic schemas ---------------


class BacktestRunRequest(PydanticBase):
    """Request body for dispatching a backtest run."""

    strategy_id: uuid.UUID
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 1_000_000.0
    fill_model: str | None = Field(None, pattern="^(optimistic|pessimistic)$")
    include_funding: bool = False
    sizing_mode: str = Field(
        "fixed_notional",
        pattern="^(fixed_pct|fixed_notional|vol_target|fixed_risk)$",
    )
    sizing_params: dict | None = None


class DualModeRequest(PydanticBase):
    """Request body for dual-mode fill comparison (API-03)."""

    strategy_id: uuid.UUID
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 1_000_000.0
    include_funding: bool = False
    sizing_mode: str = Field(
        "fixed_notional",
        pattern="^(fixed_pct|fixed_notional|vol_target|fixed_risk)$",
    )
    sizing_params: dict | None = None


class DualModeResponse(PydanticBase):
    """Response body wrapping DualModeResult (API-03)."""

    optimistic_metrics: dict
    pessimistic_metrics: dict
    delta_metrics: dict
    is_viable: bool


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


class BatchBacktestQueryRequest(PydanticBase):
    """Request body for batch querying backtests by strategy IDs."""

    strategy_ids: list[uuid.UUID] = Field(min_length=1, max_length=200)
    latest_only: bool = True
    limit: int = Field(200, ge=1, le=1000)


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
        request.sizing_mode,
        request.sizing_params,
        request.fill_model,
        request.include_funding,
    )
    return MessageResponse(
        message=f"Backtest dispatched for strategy {request.strategy_id}",
        task_id=task.id,
    )


@router.post("/dual-mode", response_model=MessageResponse, status_code=202)
async def run_dual_mode(request: DualModeRequest) -> MessageResponse:
    """Dispatch dual-mode fill comparison to CPU worker queue (API-03).

    Runs same strategy with both OPTIMISTIC and PESSIMISTIC fill models.
    Returns 202 with Celery task_id for status polling.
    """
    task = run_dual_mode_task.delay(
        str(request.strategy_id),
        request.start_date,
        request.end_date,
        request.initial_capital,
        request.include_funding,
        request.sizing_mode,
        request.sizing_params,
    )
    return MessageResponse(
        message=f"Dual-mode comparison dispatched for strategy {request.strategy_id}",
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


@router.post("/query", response_model=list[BacktestResponse])
async def query_backtests(
    request: BatchBacktestQueryRequest,
    db: Session = Depends(get_db),
) -> list[BacktestResponse]:
    """Query backtests for multiple strategy IDs in one request."""
    repo = BacktestRepository(db)
    return repo.list_backtests_batch(
        strategy_ids=request.strategy_ids,
        latest_only=request.latest_only,
        limit=request.limit,
    )


class BacktestTradeResponse(PydanticBase):
    """Response model for a single backtest trade."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    backtest_id: uuid.UUID
    symbol: str
    action: str
    entry_time: datetime
    exit_time: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl: float | None
    fees: float


class BacktestEquityPointResponse(PydanticBase):
    """Response model for a single equity curve data point."""

    model_config = ConfigDict(from_attributes=True)

    time: datetime
    equity: float
    drawdown: float


class BacktestEquityResponse(PydanticBase):
    """Response model wrapping the equity curve for a backtest."""

    backtest_id: uuid.UUID
    data: list[BacktestEquityPointResponse]
    count: int


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


@router.get("/{backtest_id}/trades", response_model=list[BacktestTradeResponse])
async def get_backtest_trades(
    backtest_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[BacktestTradeResponse]:
    """Get trade list for a completed backtest."""
    repo = BacktestRepository(db)
    record = repo.get_by_id(backtest_id)
    if not record:
        raise HTTPException(status_code=404, detail="Backtest not found")
    trades = repo.get_trades(backtest_id)
    return trades


@router.get("/{backtest_id}/equity-curve", response_model=BacktestEquityResponse)
async def get_backtest_equity_curve(
    backtest_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> BacktestEquityResponse:
    """Get equity curve time-series for a completed backtest."""
    repo = BacktestRepository(db)
    record = repo.get_by_id(backtest_id)
    if not record:
        raise HTTPException(status_code=404, detail="Backtest not found")
    points = repo.get_equity_curve(backtest_id)
    return BacktestEquityResponse(
        backtest_id=backtest_id,
        data=points,
        count=len(points),
    )
