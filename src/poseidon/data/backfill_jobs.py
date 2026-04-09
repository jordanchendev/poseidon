"""Durable BackfillJob helpers — Phase 39 backfill-api-coverage.

These helpers centralize row create/get/cancel semantics so the FastAPI
layer never talks to the ORM directly and all writes go through a single
place that the worker and dispatcher paths can reuse.

Contract per .planning/phases/39-backfill-api-coverage/39-CONTEXT.md:
- D-04: API reads status from Postgres, not Celery result backend
- D-05: cancellation is cooperative — flip status, preserve cursor/progress
- D-09: restart survival source of truth is the row itself
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from poseidon.core.schemas import BackfillRequest
from poseidon.models.backfill import BackfillJob


_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def create_backfill_job(
    session: Session,
    request: BackfillRequest,
    requested_by: str = "api",
) -> BackfillJob:
    """Persist a new BackfillJob row for a multi-symbol/interval API request.

    The cursor is seeded so ``backfill_chunk`` can resume from
    ``cursor.next_ts``. ``progress`` records the requested scope so operators
    polling the API can see exactly what was asked for even before the worker
    has touched the row.
    """
    symbols = list(request.symbols)
    intervals = list(request.intervals)
    start_iso = request.start.isoformat()
    end_iso = request.end.isoformat()

    cursor = {
        "symbol_idx": 0,
        "interval_idx": 0,
        "next_ts": start_iso,
        "end_ts": end_iso,
    }
    progress = {
        "requested_symbols": symbols,
        "requested_intervals": intervals,
        "rows_written": 0,
        "chunks_done": 0,
        "chunks_total": None,
    }

    job = BackfillJob(
        status="pending",
        market=request.market,
        symbol=None,
        interval=None,
        symbols=symbols,
        intervals=intervals,
        requested_by=requested_by,
        cursor=cursor,
        progress=progress,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_backfill_job(session: Session, job_id: UUID) -> BackfillJob | None:
    """Return a BackfillJob row by ``job_id`` or ``None`` if it does not exist."""
    return session.query(BackfillJob).filter(BackfillJob.job_id == job_id).one_or_none()


def cancel_backfill_job(session: Session, job_id: UUID) -> BackfillJob | None:
    """Mark a job cancelled unless it is already terminal.

    Preserves ``cursor`` and ``progress`` so operators can see where the
    worker stopped (D-05). Stamps ``finished_at`` iff the cancellation
    actually transitioned the row.
    """
    job = get_backfill_job(session, job_id)
    if job is None:
        return None
    if job.status in _TERMINAL_STATUSES:
        return job
    job.status = "cancelled"
    job.finished_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(job)
    return job
