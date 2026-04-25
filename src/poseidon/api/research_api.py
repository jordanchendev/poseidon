"""Research API endpoints for training run lifecycle + model queries (Phase 41).

Exposes POST /train, GET /runs, GET /runs/{run_id}, POST /runs/{run_id}/cancel,
GET /{model_id}, DELETE /{model_id}, GET /{model_id}/predictions.

Task dispatch uses ``celery_app.send_task()`` (NOT direct import of qlib_tasks)
to avoid importing qlib/mlflow in the cp313 API container.

**Route ordering note:** The ``/{model_id}`` wildcard routes MUST be defined
AFTER ``/runs`` and ``/runs/{run_id}`` to prevent FastAPI from matching
``/runs`` as a ``model_id``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    ModelMetricsResponse,
    PredictionResponse,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    TrainingRunResponse,
    TrainRequest,
)
from poseidon.ml.artifacts import delete_version_artifacts, get_predictions_path
from poseidon.models.base import get_db
from poseidon.models.model_version import ModelVersion
from poseidon.models.training_run import TrainingRun
from poseidon.qlib.allowlist import resolve_handler, resolve_model
from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app

router = APIRouter()


def _parse_period_date(value: str | None) -> datetime | None:
    """Parse a date string from ModelVersion params JSONB to datetime."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


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
        raise HTTPException(status_code=422, detail=str(exc)) from None
    try:
        resolve_model(request.model_class)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

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
        queue=POSEIDON_QLIB_QUEUE,
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
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")
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
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")
    if run.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel run in status: {run.status}",
        )
    run.status = "cancelled"
    db.commit()
    return {"run_id": str(run.run_id), "status": "cancelled"}


# ---------------------------------------------------------------------------
# Model-level endpoints (Plan 04: RESEARCH-API-08, -09, -10)
# ---------------------------------------------------------------------------
# These MUST stay below /runs and /runs/{run_id} to avoid route shadowing.


@router.get("/{model_id}", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_id: UUID,
    db: Session = Depends(get_db),
):
    """Return a ModelVersion with full metrics JSON (per RESEARCH-API-08)."""
    version = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if version is None:
        raise HTTPException(status_code=404, detail=f"Model version {model_id} not found")
    return version


@router.delete("/{model_id}")
async def delete_model(
    model_id: UUID,
    db: Session = Depends(get_db),
):
    """Delete a ModelVersion and its artifact files (per RESEARCH-API-09).

    Best-effort artifact cleanup: if the directory doesn't exist or fails
    to delete, the DB row is still removed. Training runs that reference
    this version have their FK set to NULL to prevent orphaned references.
    """
    version = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if version is None:
        raise HTTPException(status_code=404, detail=f"Model version {model_id} not found")
    # Best-effort artifact cleanup
    if version.artifact_path:
        delete_version_artifacts(version.artifact_path)
    # Unlink any training_runs that reference this model_version
    db.query(TrainingRun).filter(TrainingRun.model_version_id == model_id).update({"model_version_id": None})
    db.delete(version)
    db.commit()
    return {"detail": f"Model version {model_id} deleted"}


@router.get("/{model_id}/predictions", response_model=PredictionResponse)
async def get_predictions(
    model_id: UUID,
    segment: str | None = Query(None, examples=["test", "valid", "train"]),
    start: datetime | None = Query(None, examples=["2024-01-01T00:00:00"]),
    end: datetime | None = Query(None, examples=["2024-12-31T00:00:00"]),
    symbol: str | None = Query(None, examples=["BTCUSDT"]),
    db: Session = Depends(get_db),
):
    """Return predictions with optional range filtering (per D-04, D-05, D-06, PRED-03).

    Query modes (backward compatible per D-04):
    - ``?segment=test`` -- return single segment Parquet (legacy behavior)
    - ``?start=...&end=...`` -- load all segments, concat, filter by date range
    - ``?symbol=...`` -- filter by instrument
    - Combine: ``?start=...&end=...&symbol=...`` for precise queries

    When no segment and no range params provided, defaults to loading test segment
    for backward compatibility.

    Response always includes training period metadata per D-06.
    """
    version = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if version is None:
        raise HTTPException(status_code=404, detail=f"Model version {model_id} not found")
    if not version.artifact_path:
        raise HTTPException(
            status_code=404,
            detail="No artifact path for this model version",
        )

    # Extract training period metadata from params JSONB (per D-06)
    params = version.params or {}
    train_period = params.get("train_period", {})
    valid_period = params.get("valid_period", {})
    test_period = params.get("test_period", {})

    # Determine which segments to load
    if segment is not None:
        # Legacy mode: load single segment
        segments_to_load = [segment]
    elif start is not None or end is not None or symbol is not None:
        # Range query mode: load all available segments (per D-05)
        segments_to_load = []
        for seg in ["train", "valid", "test"]:
            pred_path = get_predictions_path(version.artifact_path, seg)
            if pred_path.exists():
                segments_to_load.append(seg)
        if not segments_to_load:
            raise HTTPException(
                status_code=404,
                detail="No prediction files found for this model version",
            )
    else:
        # Default: test segment for backward compatibility
        segments_to_load = ["test"]
        segment = "test"

    # Load and concatenate Parquet files
    dfs: list[pd.DataFrame] = []
    for seg in segments_to_load:
        pred_path = get_predictions_path(version.artifact_path, seg)
        if not pred_path.exists():
            if segment is not None:
                # Single segment mode: 404 if requested segment missing
                raise HTTPException(
                    status_code=404,
                    detail=f"Predictions not available for segment: {seg}",
                )
            continue  # Range mode: skip missing segments
        dfs.append(pd.read_parquet(pred_path))

    if not dfs:
        raise HTTPException(
            status_code=404,
            detail="No prediction files found",
        )

    df = pd.concat(dfs)

    # Apply date range filter (per D-05)
    if start is not None or end is not None:
        # Reset index to access datetime column for filtering
        idx_names = df.index.names
        df_reset = df.reset_index()
        datetime_col = "datetime" if "datetime" in df_reset.columns else idx_names[0]
        if datetime_col in df_reset.columns:
            df_reset[datetime_col] = pd.to_datetime(df_reset[datetime_col])
            if start is not None:
                df_reset = df_reset[df_reset[datetime_col] >= pd.Timestamp(start)]
            if end is not None:
                df_reset = df_reset[df_reset[datetime_col] <= pd.Timestamp(end)]
        # Re-set index
        df = df_reset.set_index(idx_names) if idx_names[0] is not None else df_reset

    # Apply symbol filter
    if symbol is not None:
        idx_names = df.index.names
        df_reset = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
        instrument_col = "instrument" if "instrument" in df_reset.columns else None
        if instrument_col is not None:
            df_reset = df_reset[df_reset[instrument_col] == symbol]
        df = (
            df_reset.set_index(idx_names)
            if idx_names[0] is not None and isinstance(df.index, pd.MultiIndex)
            else df_reset
        )

    # Convert to JSON records
    df_out = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    for col in df_out.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df_out[col] = df_out[col].astype(str)
    records = df_out.to_dict(orient="records")

    return PredictionResponse(
        model_id=model_id,
        segment=segment,
        count=len(records),
        predictions=records,
        train_start=_parse_period_date(train_period.get("start")),
        train_end=_parse_period_date(train_period.get("end")),
        valid_start=_parse_period_date(valid_period.get("start")),
        valid_end=_parse_period_date(valid_period.get("end")),
        test_start=_parse_period_date(test_period.get("start")),
        test_end=_parse_period_date(test_period.get("end")),
    )
