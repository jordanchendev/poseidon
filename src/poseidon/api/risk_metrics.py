"""Risk metrics API endpoints (RAPI-01, RAPI-02, RAPI-03, RAPI-04).

Mounted at /api/risk in main.py. Separate from /api/risk-rules (CRUD for risk rules).
"""

from __future__ import annotations

import json
import logging

import msgpack
import redis as redis_lib
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel as PydanticBase

from poseidon.core.config import settings
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic response schemas ---


class VaRSnapshotResponse(PydanticBase):
    method: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    portfolio_value: float
    holding_period: int
    as_of: str | None = None
    computed_at: str | None = None


class VaRResponse(PydanticBase):
    snapshots: list[VaRSnapshotResponse]


class ExposureItem(PydanticBase):
    category: str  # market name or "total"
    value: float


class ExposureResponse(PydanticBase):
    exposures: list[ExposureItem]
    total: float


class StressTestRequest(PydanticBase):
    scenario_name: str
    custom_shocks: dict[str, float] | None = None


class StressTestTriggerResponse(PydanticBase):
    task_id: str


class StressTestStatusResponse(PydanticBase):
    status: str  # "pending" | "completed" | "failed"
    result: dict | None = None
    error: str | None = None


class AlertItem(PydanticBase):
    id: str
    event_type: str
    data: dict


class AlertsResponse(PydanticBase):
    alerts: list[AlertItem]
    total: int


def _get_redis_client() -> redis_lib.Redis:
    return redis_lib.from_url(settings.redis_url, decode_responses=False)


# --- RAPI-01: GET /var ---


@router.get("/var", response_model=VaRResponse)
def get_var():
    """Return current VaR snapshots from all methods (per D-13)."""
    r = _get_redis_client()
    methods = ["parametric", "historical", "cornish_fisher", "monte_carlo"]
    snapshots = []
    for method in methods:
        raw = r.get(f"poseidon:var:latest:{method}")
        if raw:
            data = msgpack.unpackb(raw, raw=False)
            snapshots.append(VaRSnapshotResponse(**data))
    return VaRResponse(snapshots=snapshots)


# --- RAPI-02: GET /exposure ---


@router.get("/exposure", response_model=ExposureResponse)
def get_exposure():
    """Return portfolio exposure breakdown by market and total (per D-13).

    Rebuilds VirtualPortfolio from DB signals, then aggregates
    quantity_pct by market.
    """
    from poseidon.models.base import SessionLocal
    from poseidon.risk.portfolio import VirtualPortfolio

    portfolio = VirtualPortfolio()
    db = SessionLocal()
    try:
        portfolio.rebuild_from_db(db)
    finally:
        db.close()

    # Aggregate by market
    market_exposure: dict[str, float] = {}
    for key, pos in portfolio.positions.items():
        market = pos.market
        market_exposure[market] = market_exposure.get(market, 0.0) + pos.quantity_pct

    exposures = [
        ExposureItem(category=market, value=round(value, 6))
        for market, value in sorted(market_exposure.items())
    ]
    total = round(portfolio.total_exposure(), 6)
    return ExposureResponse(exposures=exposures, total=total)


# --- RAPI-03: POST /stress-test/run + GET /stress-test/{task_id} ---


@router.post("/stress-test/run", response_model=StressTestTriggerResponse)
def trigger_stress_test(body: StressTestRequest):
    """Trigger async stress test (per D-12). Returns task_id for polling."""
    from poseidon.workers.cpu_tasks import run_stress_test

    task = run_stress_test.delay(body.scenario_name, body.custom_shocks)
    return StressTestTriggerResponse(task_id=task.id)


@router.get("/stress-test/{task_id}", response_model=StressTestStatusResponse)
def get_stress_test_result(task_id: str):
    """Poll stress test result by task_id (per D-12)."""
    result = celery_app.AsyncResult(task_id)
    if result.state == "PENDING":
        return StressTestStatusResponse(status="pending")
    elif result.state == "SUCCESS":
        return StressTestStatusResponse(status="completed", result=result.result)
    elif result.state == "FAILURE":
        return StressTestStatusResponse(status="failed", error=str(result.result))
    else:
        return StressTestStatusResponse(status=result.state.lower())


# --- RAPI-04: GET /alerts ---


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return risk alert history from Redis Stream with limit/offset pagination (per D-14).

    Reads from poseidon:alerts:risk stream using XREVRANGE (newest first).
    Offset implemented by skipping first ``offset`` entries from the reversed stream.

    Note: DrawdownMonitor stores alerts as {"data": json.dumps(alert)}, so
    we parse the JSON ``data`` field to reconstruct the alert structure.
    """
    STREAM_KEY = "poseidon:alerts:risk"
    r = _get_redis_client()
    # Read more than needed to support offset
    fetch_count = limit + offset
    raw_entries = r.xrevrange(STREAM_KEY, count=fetch_count)
    # Skip offset entries, take limit
    entries = raw_entries[offset : offset + limit]
    alerts = []
    for msg_id, fields in entries:
        msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        # DrawdownMonitor wraps alert as {"data": json.dumps(alert_dict)}
        raw_data = fields.get(b"data") or fields.get("data")
        if raw_data:
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode()
            try:
                parsed = json.loads(raw_data)
                event_type = parsed.pop("event_type", "unknown")
                alerts.append(AlertItem(id=msg_id_str, event_type=event_type, data=parsed))
            except (json.JSONDecodeError, AttributeError):
                # Fallback: treat raw fields as flat dict
                decoded = {
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in fields.items()
                }
                event_type = decoded.pop("event_type", "unknown")
                alerts.append(AlertItem(id=msg_id_str, event_type=event_type, data=decoded))
        else:
            # No data field -- decode raw fields
            decoded = {
                k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                for k, v in fields.items()
            }
            event_type = decoded.pop("event_type", "unknown")
            alerts.append(AlertItem(id=msg_id_str, event_type=event_type, data=decoded))

    # total: count of entries fetched (may be less than full stream)
    total_in_stream = r.xlen(STREAM_KEY)
    return AlertsResponse(alerts=alerts, total=total_in_stream)
