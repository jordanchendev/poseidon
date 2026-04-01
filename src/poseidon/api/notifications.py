"""Unified notifications endpoint -- aggregates alerts from multiple Redis streams.

Mounted at /api/notifications in main.py (per D-09: separate from risk_metrics router).
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query
from pydantic import BaseModel as PydanticBase

from poseidon.api.risk_metrics import _get_redis_client

logger = logging.getLogger(__name__)

router = APIRouter()


class NotificationItem(PydanticBase):
    id: str
    source: str  # "risk", "data_quality", "autoresearch"
    event_type: str
    data: dict
    timestamp: str | None = None


class NotificationsResponse(PydanticBase):
    notifications: list[NotificationItem]
    total: int


@router.get("", response_model=NotificationsResponse)
def get_notifications(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source: str | None = Query(None, description="Filter by source: risk, data_quality, autoresearch"),
):
    """Return unified notification feed from Redis streams.

    Aggregates alerts from:
    - poseidon:alerts:risk (risk alerts -- drawdown, VaR breach)
    - poseidon:alerts:data_quality (data quality events)
    - poseidon:alerts:autoresearch (autoresearch completions)
    """
    r = _get_redis_client()
    all_notifications: list[NotificationItem] = []

    streams = {
        "risk": "poseidon:alerts:risk",
        "data_quality": "poseidon:alerts:data_quality",
        "autoresearch": "poseidon:alerts:autoresearch",
    }

    if source:
        streams = {k: v for k, v in streams.items() if k == source}

    for src_name, stream_key in streams.items():
        try:
            fetch_count = limit + offset
            raw_entries = r.xrevrange(stream_key, count=fetch_count)
            for msg_id, fields in raw_entries:
                msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                raw_data = fields.get(b"data") or fields.get("data")
                if raw_data:
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode()
                    try:
                        parsed = json.loads(raw_data)
                        event_type = parsed.pop("event_type", "unknown")
                        timestamp = parsed.pop("timestamp", None)
                        all_notifications.append(
                            NotificationItem(
                                id=msg_id_str,
                                source=src_name,
                                event_type=event_type,
                                data=parsed,
                                timestamp=timestamp,
                            )
                        )
                    except (json.JSONDecodeError, AttributeError):
                        pass
        except Exception:
            logger.debug("Stream %s not found or error reading", stream_key)
            continue

    all_notifications.sort(key=lambda n: n.id, reverse=True)
    paginated = all_notifications[offset : offset + limit]
    return NotificationsResponse(notifications=paginated, total=len(all_notifications))
