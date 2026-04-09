"""Freshness watchdog helpers -- Phase 40 D-10..D-16.

Pure helpers separated from the Celery task wrapper so unit tests can
drive them without booting Celery. The HC.io ping is fire-and-forget
by design (D-14): a network outage on Healthchecks.io must NEVER
poison the watchdog task itself.

See:
- .planning/phases/40-data-health-observability/40-CONTEXT.md
- poseidon/src/poseidon/core/config.py for freshness_sla / healthchecks_freshness_url
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from poseidon.models.ingest_state import IngestState

logger = logging.getLogger(__name__)

HC_PING_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class FreshnessRecord:
    market: str
    interval: str
    last_successful_ts: datetime | None
    expected_lag_seconds: int
    observed_lag_seconds: float
    status: str  # "ok" | "violation" | "unknown"


def lookup_sla(
    sla_dict: dict[str, int], market: str, interval: str
) -> int | None:
    """Return the SLA seconds for ``(market, interval)`` or ``None``.

    SLA dict keys are colon-joined ``"market:interval"`` because
    pydantic-settings cannot round-trip tuple keys via env vars
    (Phase 40 D-10).
    """
    return sla_dict.get(f"{market}:{interval}")


def compute_freshness_status(
    *,
    now: datetime,
    last_successful_ts: datetime | None,
    expected_lag_seconds: int,
) -> tuple[str, float]:
    """Return ``(status, observed_lag_seconds)`` for one tuple.

    - ``unknown`` when ``last_successful_ts is None`` (ingest never ran or
      tuple is bootstrapping)
    - ``violation`` when observed lag exceeds ``expected_lag_seconds``
    - ``ok`` otherwise

    Pure function -- no DB, no I/O.
    """
    if last_successful_ts is None:
        return ("unknown", 0.0)
    # Normalize timezone: SQLite returns naive datetimes, Postgres returns
    # tz-aware. Ensure both sides are comparable.
    ts = last_successful_ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    observed = (now - ts).total_seconds()
    if observed < 0:
        observed = 0.0
    if observed > expected_lag_seconds:
        return ("violation", observed)
    return ("ok", observed)


def evaluate_freshness(
    session: Session,
    *,
    now: datetime,
    sla_dict: dict[str, int],
) -> list[FreshnessRecord]:
    """Walk every ``ingest_state`` row, compute status against SLA dict.

    Rows whose ``(market, interval)`` has no matching SLA entry are
    silently skipped (Phase 40 D-15: no spam for tuples we do not
    track). Returns one ``FreshnessRecord`` per matched row.

    The function collapses to ``(market, interval)`` granularity by
    picking the WORST (most stale) symbol per group. Operators care
    about whether ANY symbol in a market is stale, not which one.
    """
    rows = (
        session.query(IngestState)
        .order_by(IngestState.market, IngestState.interval, IngestState.symbol)
        .all()
    )

    # Collapse to (market, interval) -> minimum last_successful_ts so the
    # watchdog reports the WORST tuple per (market, interval).
    worst: dict[tuple[str, str], IngestState] = {}
    for r in rows:
        key = (r.market, r.interval)
        existing = worst.get(key)
        if existing is None:
            worst[key] = r
            continue
        existing_ts = existing.last_successful_ts
        new_ts = r.last_successful_ts
        if existing_ts is None or (
            new_ts is not None and new_ts < existing_ts
        ):
            worst[key] = r

    records: list[FreshnessRecord] = []
    for (market, interval), row in sorted(worst.items()):
        sla_seconds = lookup_sla(sla_dict, market, interval)
        if sla_seconds is None:
            continue
        status, observed = compute_freshness_status(
            now=now,
            last_successful_ts=row.last_successful_ts,
            expected_lag_seconds=sla_seconds,
        )
        records.append(
            FreshnessRecord(
                market=market,
                interval=interval,
                last_successful_ts=row.last_successful_ts,
                expected_lag_seconds=sla_seconds,
                observed_lag_seconds=observed,
                status=status,
            )
        )
    return records


def ping_healthchecks(url: str, *, success: bool) -> bool:
    """Fire-and-forget HC.io ping. Returns True iff a request was sent.

    Phase 40 D-12..D-14:
    - empty url = local/dev no-op (returns False)
    - success=True  -> GET ``url``           (heartbeat)
    - success=False -> GET ``url + "/fail"`` (violation alert)
    - 3s timeout, all exceptions swallowed and logged

    We deliberately do NOT raise on HC.io errors -- the watchdog must
    keep emitting signals even if the alerting layer is temporarily
    unreachable.
    """
    if not url:
        return False

    target = url if success else url.rstrip("/") + "/fail"
    try:
        import requests

        requests.get(target, timeout=HC_PING_TIMEOUT_SECONDS)
        return True
    except Exception as exc:
        logger.warning(
            "ping_healthchecks failed for %s: %s",
            target,
            exc,
        )
        return False
