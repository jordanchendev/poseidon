"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# --- Sentiment ---

class SentimentCreate(BaseModel):
    """Request body for creating a sentiment score."""
    symbol: str = Field(..., min_length=1, max_length=32, examples=["2330"])
    market: str = Field(..., min_length=1, max_length=32, examples=["tw_stock"])
    source_type: str = Field(..., min_length=1, max_length=32, examples=["news"])
    score: float = Field(..., ge=-1.0, le=1.0, examples=[0.75])


class SentimentResponse(BaseModel):
    """Response body for a sentiment score."""
    id: UUID
    symbol: str
    market: str
    source_type: str
    score: float
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Data Fetch ---

class FetchRequest(BaseModel):
    """Request body for triggering a data fetch."""
    market: str = Field(..., examples=["crypto_spot"])
    symbol: str | None = Field(None, examples=["BTCUSDT"])
    interval: str = Field("1d", examples=["1d", "1h"])


class BackfillRequest(BaseModel):
    """Request body for triggering a multi-symbol/interval backfill (Phase 39).

    Per 39-CONTEXT.md D-03: the public API requires explicit ``symbols`` and
    ``intervals`` — silent YAML expansion stays on the dispatcher path.
    """

    market: str = Field(..., examples=["crypto_perp"])
    symbols: list[str] = Field(..., min_length=1, examples=[["BTCUSDT"]])
    intervals: list[str] = Field(..., min_length=1, examples=[["4h"]])
    start: datetime = Field(..., examples=["2024-01-01T00:00:00Z"])
    end: datetime = Field(..., examples=["2024-12-31T00:00:00Z"])


class BackfillStatusResponse(BaseModel):
    """Response body for backfill status list (dashboard compatibility).

    Phase 39 allows ``symbol``/``interval`` to be null for multi-tuple jobs.
    """

    job_id: UUID
    market: str
    symbol: str | None = None
    interval: str | None = None
    status: str
    cursor: dict | None = None
    progress: dict | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BackfillJobDetailResponse(BaseModel):
    """Single-row detail response for GET /api/data/backfill/{job_id} (Phase 39)."""

    job_id: UUID
    market: str
    symbols: list[str] = Field(default_factory=list)
    intervals: list[str] = Field(default_factory=list)
    status: str
    cursor: dict | None = None
    progress: dict | None = None
    error: str | None = None
    requested_by: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}


# --- Data Coverage (Phase 39 plan 39-03) ---


class DataCoverageResponse(BaseModel):
    """Per-tuple data coverage response for GET /api/data/coverage (Phase 39).

    Fields follow 39-CONTEXT.md D-10..D-12 and 39-03-PLAN.md Task 2:

    * ``market`` / ``symbol`` / ``interval`` — the tuple key
    * ``first_ts`` / ``last_ts`` — MIN/MAX(time) from ``data_coverage_mv``
    * ``row_count`` — observed row count (may be null if MV is empty)
    * ``expected_count`` — floor((last-first)/interval) + 1
    * ``gap_count`` — max(0, expected - row_count)
    * ``completeness_pct`` — row_count / expected_count (0..1)
    * ``staleness_seconds`` — seconds since ``last_ts``
    * ``health`` — ``green`` | ``yellow`` | ``red``
    """

    market: str
    symbol: str
    interval: str
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    row_count: int
    expected_count: int
    gap_count: int
    completeness_pct: float
    staleness_seconds: float
    health: str


# --- Phase 40: Data Gaps + Freshness (plan 40-01 D-02..D-04, D-11, D-16) ---


class DataGapResponse(BaseModel):
    """Per-gap response for GET /api/data/gaps (Phase 40 D-02, D-04).

    One row per detected gap window. ``healed_at`` is non-null when a later
    audit run has confirmed the window is now fully populated (D-07). The
    default dashboard query (``?open_only=true``) returns only rows where
    ``healed_at IS NULL``.
    """

    gap_id: UUID
    market: str
    symbol: str
    interval: str
    gap_start: datetime
    gap_end: datetime
    missing_bars: int
    detected_at: datetime
    healed_at: datetime | None = None

    model_config = {"from_attributes": True}


class DataFreshnessResponse(BaseModel):
    """Per-(market, interval) freshness snapshot (Phase 40 D-03, D-11, D-16).

    Returned by the freshness helper the Kairos ``/data-health`` view polls.
    Computed live from ``ingest_state.last_successful_ts`` + the per-tuple
    SLA in ``settings.freshness_sla`` — there is no snapshot table
    (D-16: stateless watchdog).

    * ``expected_lag_seconds`` — SLA pulled from ``settings.freshness_sla``
    * ``observed_lag_seconds`` — ``now() - last_successful_ts`` at read time
    * ``status`` — ``"ok"`` | ``"violation"`` | ``"unknown"`` (``unknown``
      when ``last_successful_ts`` is NULL, i.e. first-time bootstrap)
    """

    market: str
    interval: str
    last_successful_ts: datetime | None = None
    expected_lag_seconds: int
    observed_lag_seconds: float
    status: str


# --- Health ---

class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str


# --- Generic ---

class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    task_id: str | None = None


# --- Training Runs (Phase 41) ---


class TrainRequest(BaseModel):
    """Request body for POST /api/v1/models/train (per D-01, RESEARCH-API-01)."""

    handler_class: str = Field(..., examples=["Alpha158Handler"])
    handler_params: dict = Field(default_factory=dict, examples=[{}])
    model_class: str = Field(..., examples=["LGBModel"])
    model_params: dict = Field(
        default_factory=dict, examples=[{"num_leaves": 128}]
    )
    market: str = Field(..., examples=["crypto_perp"])
    symbols: list[str] = Field(..., min_length=1, examples=[["BTCUSDT"]])
    interval: str = Field(..., examples=["4h"])
    segments: dict = Field(
        default_factory=lambda: {
            "train": ["2023-01-01", "2024-06-30"],
            "valid": ["2024-07-01", "2024-12-31"],
            "test": ["2025-01-01", "2025-06-30"],
        },
        examples=[
            {
                "train": ["2023-01-01", "2024-06-30"],
                "valid": ["2024-07-01", "2024-12-31"],
                "test": ["2025-01-01", "2025-06-30"],
            }
        ],
    )
    lookback: str | None = Field(None, examples=["2y"])


class TrainingRunResponse(BaseModel):
    """Short response for run creation and list items."""

    run_id: UUID
    handler_class: str
    model_class: str
    market: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: dict | None = None

    model_config = {"from_attributes": True}


class TrainingRunDetailResponse(BaseModel):
    """Full detail response for GET /runs/{run_id} (per RESEARCH-API-04)."""

    run_id: UUID
    handler_class: str
    handler_params: dict
    model_class: str
    model_params: dict
    market: str
    symbols: list[str]
    interval: str
    segments: dict
    lookback: str | None = None
    status: str
    metrics: dict | None = None
    model_version_id: UUID | None = None
    mlflow_run_id: str | None = None
    error: str | None = None
    requested_by: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TrainingRunListResponse(BaseModel):
    """Paginated list response for GET /runs (per D-23)."""

    runs: list[TrainingRunResponse]
    total: int
    limit: int
    offset: int


class ModelMetricsResponse(BaseModel):
    """Full metrics response for GET /models/{id} (per RESEARCH-API-08)."""

    id: UUID
    name: str
    version: int
    status: str
    params: dict
    metrics: dict | None = None
    feature_list: list
    train_start: datetime | None = None
    train_end: datetime | None = None
    artifact_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    """Response for GET /models/{id}/predictions with range query (per D-04, D-06, PRED-03).

    Includes training period metadata alongside prediction data so consumers
    know the model's temporal boundary.
    """

    model_id: UUID
    segment: str | None = None  # Set when querying by segment; None for range query
    count: int
    predictions: list[dict]
    # Training period metadata (per D-06)
    train_start: datetime | None = None
    train_end: datetime | None = None
    valid_start: datetime | None = None
    valid_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
