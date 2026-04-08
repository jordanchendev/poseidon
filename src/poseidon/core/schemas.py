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
    """Request body for triggering a backfill."""
    market: str | None = Field(None, examples=["tw_stock"])
    symbol: str | None = Field(None, examples=["2330"])


class BackfillStatusResponse(BaseModel):
    """Response body for backfill status (Phase 38 D-10: BackfillJob shape)."""
    job_id: UUID
    market: str
    symbol: str
    interval: str
    status: str
    cursor: dict | None = None
    progress: dict | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

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
