"""BackfillJob chunk Celery task — DATA-FOUND-04/05 per Phase 38 D-08..D-13.

Idempotent, resumable, and registered on the dedicated ``backfill`` queue so
one-shot backfills never starve periodic ingest. NOT on the Beat schedule
(D-13) — Phase 39 adds the REST dispatcher.

Crash-safety model (D-08/D-09):

    fetch → validate → upsert_ohlcv (commits)
                           │
                           ▼
                   advance BackfillJob.cursor (commits)

A SIGKILL between the two commits replays the previous batch on restart.
``upsert_ohlcv`` uses ``ON CONFLICT pk_ohlcv DO UPDATE`` so the replay is
idempotent — zero duplicate rows, zero gaps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from poseidon.data.fetchers import get_fetcher
from poseidon.data.storage import upsert_ohlcv
from poseidon.data.symbols import get_market_config, load_symbols
from poseidon.data.validation import validate_ohlcv
from poseidon.models.backfill import BackfillJob
from poseidon.models.base import SessionLocal
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Chunk one batch = ~1000 candles, matching the cursor-mode ingest helper.
CHUNK_CANDLES = 1000
# Soft cap so a single task invocation never monopolises a backfill worker.
# Celery ack_late=True + retry will re-schedule the same job_id to continue.
MAX_CHUNKS_PER_CALL = 50

_INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _run_backfill_chunk(session, job_id: str) -> dict:
    """Core loop — separated from the Celery task wrapper so unit tests can
    drive it with an injected session without going through broker."""
    job = session.query(BackfillJob).filter_by(job_id=UUID(job_id)).one()
    if job.status in ("succeeded", "cancelled"):
        return {"job_id": job_id, "status": job.status, "noop": True}

    job.status = "running"
    session.commit()

    cursor = dict(job.cursor or {})
    start = _parse_ts(cursor.get("next_ts")) or _parse_ts(cursor.get("start_ts"))
    end = _parse_ts(cursor.get("end_ts")) or datetime.now(timezone.utc)
    if start is None:
        raise ValueError(f"backfill_chunk: job {job_id} has no start cursor")

    try:
        config = load_symbols()
        market_cfg = get_market_config(job.market, config)
    except Exception:  # noqa: BLE001
        market_cfg = None
    instrument = market_cfg.instrument if market_cfg else "spot"
    interval_sec = _INTERVAL_SECONDS.get(job.interval, 3600)
    chunk_td = timedelta(seconds=interval_sec * CHUNK_CANDLES)

    chunks_done = int(cursor.get("completed_chunks", 0))
    total_chunks = int(
        cursor.get("total_chunks")
        or max(1, int((end - start) / chunk_td) + 1)
    )

    fetcher = get_fetcher(job.market)
    start_ts_original = cursor.get("start_ts") or start.isoformat()
    chunks_this_call = 0

    while start < end and chunks_this_call < MAX_CHUNKS_PER_CALL:
        batch_end = min(start + chunk_td, end)
        df = fetcher.fetch_ohlcv(
            job.symbol,
            job.interval,
            start.strftime("%Y-%m-%d"),
            batch_end.strftime("%Y-%m-%d"),
        )

        if df is not None and not df.empty:
            vresult = validate_ohlcv(df, job.market)
            if vresult.has_critical:
                job.status = "failed"
                job.error = "critical validation failure"
                session.commit()
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "chunks_done": chunks_done,
                }
            # --- CRITICAL SECTION (D-08) -------------------------------
            upsert_ohlcv(
                session, df, job.symbol, job.market, instrument, job.interval
            )
            # upsert_ohlcv commits. SIGKILL here is safe: replay hits
            # ON CONFLICT pk_ohlcv and dedupes (D-09).
            # ----------------------------------------------------------

        chunks_done += 1
        chunks_this_call += 1
        # Advance cursor strictly AFTER the upsert commit.
        job.cursor = {
            "start_ts": start_ts_original,
            "next_ts": batch_end.isoformat(),
            "end_ts": end.isoformat(),
            "completed_chunks": chunks_done,
            "total_chunks": total_chunks,
        }
        session.commit()
        start = batch_end

    if start >= end:
        job.status = "succeeded"
        session.commit()

    return {
        "job_id": job_id,
        "status": job.status,
        "chunks_done": chunks_done,
        "total_chunks": total_chunks,
    }


@celery_app.task(
    name="poseidon.workers.backfill_tasks.backfill_chunk",
    queue="backfill",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def backfill_chunk(self, job_id: str) -> dict:
    """Idempotent, resumable backfill chunk processor.

    Reads the ``BackfillJob`` row by ``job_id``, resumes from
    ``cursor.next_ts``, writes via ``upsert_ohlcv`` (idempotent on
    ``pk_ohlcv``), and advances the cursor AFTER the write commits (D-08).
    """
    session = SessionLocal()
    try:
        return _run_backfill_chunk(session, job_id)
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        job = session.query(BackfillJob).filter_by(job_id=UUID(job_id)).first()
        if job is not None:
            job.status = "failed"
            job.error = str(exc)[:2000]
            session.commit()
        logger.exception("backfill_chunk failed for job_id=%s", job_id)
        raise self.retry(exc=exc)
    finally:
        session.close()
