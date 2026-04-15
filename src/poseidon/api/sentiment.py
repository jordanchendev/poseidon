"""Sentiment score API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from poseidon.core.schemas import SentimentCreate, SentimentResponse
from poseidon.data.repository import DataRepository
from poseidon.models.base import get_db

router = APIRouter()


@router.post("", response_model=SentimentResponse, status_code=201)
async def create_sentiment(
    body: SentimentCreate,
    db: Session = Depends(get_db),
):
    """Receive a sentiment score and persist it.

    The score must be between -1.0 (most negative) and 1.0 (most positive).
    """
    repo = DataRepository(db)
    row = repo.write_sentiment(
        symbol=body.symbol,
        market=body.market,
        source_type=body.source_type,
        score=body.score,
    )
    db.commit()
    db.refresh(row)
    return row


@router.get("", response_model=list[SentimentResponse])
async def list_sentiment(
    symbol: str = Query(..., min_length=1, description="Symbol to query sentiment for"),
    market: str | None = Query(None, description="Optional market filter"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
    db: Session = Depends(get_db),
):
    """Get sentiment scores for a symbol, most recent first."""
    repo = DataRepository(db)
    rows = repo.read_sentiment(symbol=symbol, market=market, limit=limit)
    return rows
