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


# --- Health ---

class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str


# --- Generic ---

class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    task_id: str | None = None
