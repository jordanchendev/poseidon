"""Data management API endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel as PydanticBase
from sqlalchemy import text
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    BackfillJobDetailResponse,
    BackfillRequest,
    BackfillStatusResponse,
    DataCoverageResponse,
    DataFreshnessResponse,
    DataGapResponse,
    FetchRequest,
    MessageResponse,
)
from poseidon.data import coverage as coverage_helpers
from poseidon.data.backfill_jobs import (
    cancel_backfill_job,
    create_backfill_job,
    get_backfill_job,
)
from poseidon.models.backfill import BackfillJob
from poseidon.models.base import get_db
from poseidon.workers.cpu_tasks import fetch_market_data

router = APIRouter()


# --- OHLCV response schemas ---


class OHLCVPoint(PydanticBase):
    time: str  # ISO format string
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVResponse(PydanticBase):
    data: list[OHLCVPoint]
    symbol: str
    market: str
    interval: str
    count: int


# --- Funding rate response schemas ---


class FundingRatePoint(PydanticBase):
    time: str
    symbol: str
    funding_rate: float
    mark_price: float | None
    index_price: float | None


class FundingRateResponse(PydanticBase):
    data: list[FundingRatePoint]
    count: int


@router.post("/fetch", response_model=MessageResponse, status_code=202)
async def trigger_fetch(request: FetchRequest):
    """Trigger data fetch for a market.

    Dispatches a Celery task to fetch latest data for all symbols in the market.
    """
    task = fetch_market_data.delay(request.market, request.interval, request.symbol)
    label = f"{request.market}/{request.interval}"
    if request.symbol:
        label += f" (symbol={request.symbol})"
    return MessageResponse(message=f"Fetch task dispatched for {label}", task_id=task.id)


@router.post("/backfill", status_code=202)
async def trigger_backfill_endpoint(
    request: BackfillRequest,
    db: Session = Depends(get_db),
):
    """Trigger a historical data backfill (Phase 39 D-01..D-05).

    Creates a durable ``BackfillJob`` row in Postgres and enqueues the
    existing ``backfill_chunk`` task on the dedicated ``backfill`` queue.
    Clients poll ``GET /api/data/backfill/{job_id}`` for progress.
    """
    # Deferred import: avoids loading workers (and Celery) at module import
    # time when only the API test suite wants this router.
    from poseidon.workers.backfill_tasks import backfill_chunk

    job = create_backfill_job(db, request, requested_by="api")
    backfill_chunk.delay(str(job.job_id))
    return {"job_id": str(job.job_id), "status": job.status}


@router.get("/backfill/status", response_model=list[BackfillStatusResponse])
async def get_backfill_status(
    market: str | None = None,
    db: Session = Depends(get_db),
):
    """Return all BackfillJob rows, optionally filtered by market.

    Registered before ``/backfill/{job_id}`` so FastAPI does not try to parse
    the literal ``status`` segment as a UUID path parameter.
    """
    query = db.query(BackfillJob)
    if market:
        query = query.filter(BackfillJob.market == market)
    rows = (
        query.order_by(
            BackfillJob.market,
            BackfillJob.symbol,
            BackfillJob.created_at.desc(),
        )
        .all()
    )
    return rows


@router.get("/backfill/{job_id}", response_model=BackfillJobDetailResponse)
async def get_backfill_job_endpoint(
    job_id: UUID,
    db: Session = Depends(get_db),
) -> BackfillJobDetailResponse:
    """Return a durable BackfillJob row by ``job_id`` (Phase 39 D-04)."""
    job = get_backfill_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"BackfillJob {job_id} not found")
    return BackfillJobDetailResponse.model_validate(job)


@router.post("/backfill/{job_id}/cancel", response_model=BackfillJobDetailResponse)
async def cancel_backfill_job_endpoint(
    job_id: UUID,
    db: Session = Depends(get_db),
) -> BackfillJobDetailResponse:
    """Cooperatively cancel a BackfillJob (Phase 39 D-05).

    Flips status to ``cancelled`` and preserves ``cursor``/``progress`` so
    operators can see where the worker stopped. Already-terminal jobs are
    returned unchanged (idempotent cancel).
    """
    job = cancel_backfill_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"BackfillJob {job_id} not found")
    return BackfillJobDetailResponse.model_validate(job)


# --- GET /coverage: per-tuple data coverage (Phase 39 plan 39-03 D-10..D-11) ---


@router.get("/coverage", response_model=list[DataCoverageResponse])
async def get_data_coverage(
    market: str | None = Query(None, description="Filter by market, e.g. crypto_perp"),
    symbol: str | None = Query(None, description="Filter by ticker symbol"),
    interval: str | None = Query(None, description="Filter by candle interval"),
    db: Session = Depends(get_db),
) -> list[DataCoverageResponse]:
    """Return per-(market, symbol, interval) coverage from ``data_coverage_mv``.

    The materialized view is refreshed hourly and immediately after every
    successful backfill completion (Phase 39 D-12), so this endpoint is a
    cheap read-only surface operators can hit to answer "what history do
    we actually have?" without writing SQL.

    ``expected_count``, ``gap_count``, ``completeness_pct``, and ``health``
    are computed at request time from ``first_ts`` / ``last_ts`` /
    ``row_count`` / ``staleness_seconds`` via
    :mod:`poseidon.data.coverage` helpers.
    """
    # Build a filtered query against the MV. We use raw SQL because the MV
    # is not an ORM-mapped model (it has no primary key in SQLAlchemy terms).
    where_clauses = []
    params: dict[str, str] = {}
    if market is not None:
        where_clauses.append("market = :market")
        params["market"] = market
    if symbol is not None:
        where_clauses.append("symbol = :symbol")
        params["symbol"] = symbol
    if interval is not None:
        where_clauses.append("interval = :interval")
        params["interval"] = interval
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    stmt = text(
        f"""
        SELECT
            market,
            symbol,
            interval,
            first_ts,
            last_ts,
            row_count,
            staleness_seconds
        FROM data_coverage_mv
        {where_sql}
        ORDER BY market, symbol, interval
        """
    )
    rows = db.execute(stmt, params).fetchall()

    def _parse_ts(value):
        # Postgres returns a datetime; SQLite returns an ISO string. Normalize.
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    response: list[DataCoverageResponse] = []
    for row in rows:
        first_ts = _parse_ts(row.first_ts)
        last_ts = _parse_ts(row.last_ts)
        row_count = int(row.row_count or 0)
        staleness_seconds = float(row.staleness_seconds or 0.0)

        expected_count = coverage_helpers.compute_expected_count(
            first_ts, last_ts, row.interval
        )
        gap_count = coverage_helpers.compute_gap_count(row_count, expected_count)
        completeness_pct = (
            row_count / expected_count if expected_count > 0 else 0.0
        )
        health = coverage_helpers.compute_health(
            completeness_pct=completeness_pct,
            staleness_seconds=staleness_seconds,
            interval=row.interval,
            gap_count=gap_count,
        )
        response.append(
            DataCoverageResponse(
                market=row.market,
                symbol=row.symbol,
                interval=row.interval,
                first_ts=first_ts,
                last_ts=last_ts,
                row_count=row_count,
                expected_count=expected_count,
                gap_count=gap_count,
                completeness_pct=completeness_pct,
                staleness_seconds=staleness_seconds,
                health=health,
            )
        )
    return response


# --- GET /gaps: per-window data gaps (Phase 40 plan 40-02 D-01..D-02) ---


@router.get("/gaps", response_model=list[DataGapResponse])
async def get_data_gaps(
    market: str | None = Query(None, description="Filter by market, e.g. crypto_perp"),
    symbol: str | None = Query(None, description="Filter by ticker symbol"),
    interval: str | None = Query(None, description="Filter by candle interval"),
    open_only: bool = Query(
        True,
        description="When true (default), only return rows where healed_at IS NULL",
    ),
    db: Session = Depends(get_db),
) -> list[DataGapResponse]:
    """Return detected gap windows from ``data_gaps`` (Phase 40 D-04..D-07).

    Populated daily by the ``data_gap_audit`` Celery task. Each row is
    a contiguous missing time window per ``(market, symbol, interval)``
    identified via a ``LAG(time)`` window function with a 1.5x tolerance
    threshold. ``healed_at`` is non-null when a later audit confirmed
    the window is fully populated.

    Default ``open_only=true`` matches the operator triage flow: most
    callers want "what is currently broken?", not the historical
    record. Pass ``open_only=false`` for full history.
    """
    from poseidon.models.data_gap import DataGap

    query = db.query(DataGap)
    if market is not None:
        query = query.filter(DataGap.market == market)
    if symbol is not None:
        query = query.filter(DataGap.symbol == symbol)
    if interval is not None:
        query = query.filter(DataGap.interval == interval)
    if open_only:
        query = query.filter(DataGap.healed_at.is_(None))
    rows = query.order_by(
        DataGap.market, DataGap.symbol, DataGap.interval, DataGap.gap_start
    ).all()
    return [DataGapResponse.model_validate(row) for row in rows]


# --- GET /freshness: per-(market, interval) freshness snapshot (Phase 40 D-03/D-16) ---


@router.get("/freshness", response_model=list[DataFreshnessResponse])
async def get_data_freshness(
    market: str | None = Query(None, description="Filter by market, e.g. crypto_perp"),
    db: Session = Depends(get_db),
) -> list[DataFreshnessResponse]:
    """Return per-(market, interval) freshness vs SLA from ingest_state.

    Stateless reader (Phase 40 D-16): no snapshot table. Operators (and
    the new Kairos /data-health view) call this endpoint to see the
    same data the watchdog evaluates every 15 min, just on-demand
    rather than on a beat schedule.

    Tuples with no SLA entry in ``settings.freshness_sla`` are silently
    skipped -- operators only care about tracked markets.
    """
    from datetime import datetime, timezone

    from poseidon.core.config import settings
    from poseidon.data.freshness import evaluate_freshness

    records = evaluate_freshness(
        db,
        now=datetime.now(timezone.utc),
        sla_dict=settings.freshness_sla,
    )
    if market is not None:
        records = [r for r in records if r.market == market]
    return [
        DataFreshnessResponse(
            market=r.market,
            interval=r.interval,
            last_successful_ts=r.last_successful_ts,
            expected_lag_seconds=r.expected_lag_seconds,
            observed_lag_seconds=r.observed_lag_seconds,
            status=r.status,
        )
        for r in records
    ]


# --- GET /ohlcv: OHLCV candlestick data (API-01) ---


@router.get("/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str = Query(..., description="Ticker symbol, e.g. BTCUSDT"),
    market: str = Query(..., description="Market name, e.g. crypto_perp"),
    interval: str = Query("1d", description="Candle interval, e.g. 1d, 4h, 1h"),
    start: datetime | None = Query(None, description="Start date (ISO format)"),
    end: datetime | None = Query(None, description="End date (ISO format)"),
    db: Session = Depends(get_db),
) -> OHLCVResponse:
    """Return OHLCV candlestick data for a given symbol/market/interval/date range."""
    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db, symbol, market, interval, start=start, end=end)
    if df.empty:
        return OHLCVResponse(data=[], symbol=symbol, market=market, interval=interval, count=0)
    records = df.reset_index()
    data = [
        OHLCVPoint(
            time=row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in records.iterrows()
    ]
    return OHLCVResponse(data=data, symbol=symbol, market=market, interval=interval, count=len(data))


# --- GET /funding-rates: Funding rate history (API-06) ---


@router.get("/funding-rates", response_model=FundingRateResponse)
async def get_funding_rates(
    symbol: str = Query(..., description="Perp symbol, e.g. BTC/USDT:USDT"),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> FundingRateResponse:
    """Return funding rate history for a perpetual contract symbol."""
    from poseidon.models.funding_rate import FundingRateRecord

    query = db.query(FundingRateRecord).filter(FundingRateRecord.symbol == symbol)
    if start is not None:
        query = query.filter(FundingRateRecord.time >= start)
    if end is not None:
        query = query.filter(FundingRateRecord.time <= end)
    rows = query.order_by(FundingRateRecord.time.desc()).limit(limit).all()
    data = [
        FundingRatePoint(
            time=r.time.isoformat() if hasattr(r.time, "isoformat") else str(r.time),
            symbol=r.symbol,
            funding_rate=r.funding_rate,
            mark_price=r.mark_price,
            index_price=r.index_price,
        )
        for r in rows
    ]
    return FundingRateResponse(data=data, count=len(data))
