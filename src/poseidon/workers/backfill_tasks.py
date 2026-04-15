"""BackfillJob chunk Celery task — DATA-FOUND-04/05 + Phase 39 request-level execution.

Idempotent, resumable, and registered on the dedicated ``backfill`` queue so
one-shot backfills never starve periodic ingest. NOT on the Beat schedule
(D-13) — Phase 39 adds the REST dispatcher and the public API.

Crash-safety model (Phase 38 D-08/D-09):

    fetch → validate → upsert_ohlcv (commits)
                           │
                           ▼
                   advance BackfillJob.cursor (commits)

A SIGKILL between the two commits replays the previous batch on restart.
``upsert_ohlcv`` uses ``ON CONFLICT pk_ohlcv DO UPDATE`` so the replay is
idempotent — zero duplicate rows, zero gaps.

Phase 39-02 semantics (bulk_backfill), layered on top of the Phase 38 substrate:

* ``_run_backfill_chunk`` walks ``progress.requested_symbols`` ×
  ``progress.requested_intervals`` driven by ``cursor.symbol_idx`` /
  ``cursor.interval_idx`` so a single API job can cover multiple tuples while
  the chunk-level crash-safety and resumability properties stay intact.
* Before each chunk the currently-active ``(symbol, interval)`` is mirrored
  into ``job.symbol`` / ``job.interval`` for the v6.0 dashboard's
  ``/backfill/status`` list endpoint.
* Rate-limit acquires use ``budget_class="bulk_backfill"`` so a one-shot
  historical job cannot starve the live ingest quota (Phase 39 D-16/D-17).
* Between chunk iterations the worker re-reads ``job.status`` and exits
  cleanly (``noop=True``) when the operator has flipped it to ``cancelled``
  via ``POST /api/data/backfill/{job_id}/cancel`` (Phase 39 D-05).
* When ``MAX_CHUNKS_PER_CALL`` is reached but unfinished tuples remain, the
  task re-enqueues itself on queue ``backfill`` via
  ``backfill_chunk.delay(job_id)`` instead of (incorrectly) marking the job
  succeeded (Phase 39 research Recommendation 2).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import redis as redis_lib

from poseidon.core.config import settings
from poseidon.data import coverage as coverage_helpers
from poseidon.data.fetchers import get_fetcher
from poseidon.data.rate_limiter import (
    PROVIDER_LIMITS,
    CircuitBreaker,
    DistributedRateLimiter,
)
from poseidon.data.repository import DataRepository
from poseidon.data.symbols import get_market_config, load_symbols
from poseidon.data.validation import validate_ohlcv
from poseidon.core.database import db_session
from poseidon.models.backfill import BackfillJob
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Chunk one batch = ~1000 candles, matching the cursor-mode ingest helper.
CHUNK_CANDLES = 1000
# Soft cap so a single task invocation never monopolises a backfill worker.
# When unfinished work remains the task re-enqueues itself.
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

# Market -> provider mapping for rate limiting and circuit breaker.
# Mirrors cpu_tasks.MARKET_TO_PROVIDER intentionally: the two worker pools
# share the same provider quota keyed on ``budget_class`` so backfill traffic
# lands in its own Redis bucket while live ingest keeps live_ingest isolation.
MARKET_TO_PROVIDER = {
    "tw_stock": "finmind",
    "tw_futures": "finmind",
    "us_stock": "yfinance",
    "crypto_spot": "ccxt",
    "crypto_perp": "ccxt",
}

# Phase 39 D-16/D-17: explicit backfill-budget settings keys per provider.
_BACKFILL_LIMIT_KEYS = {
    "finmind": "ratelimit_finmind_backfill_hourly",
    "yfinance": "ratelimit_yfinance_backfill_daily",
    "ccxt": "ratelimit_ccxt_backfill_per_minute",
}

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _get_redis_client() -> redis_lib.Redis:
    """Create a Redis client for rate limiter and circuit breaker (DB 3).

    Exposed as a module-level function so unit tests can monkeypatch it and
    avoid touching a real Redis instance.
    """
    from poseidon.core.redis import get_redis
    return get_redis("ratelimit")


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _resolve_request_scope(job: BackfillJob) -> tuple[list[str], list[str]]:
    """Extract the requested (symbols, intervals) lists from the job.

    Phase 39 API jobs store them in ``progress.requested_symbols`` /
    ``progress.requested_intervals``. Legacy Phase 38 dispatcher jobs only
    have ``job.symbol`` / ``job.interval``; fall back to those so the
    executor still works for single-tuple rows.
    """
    progress = dict(job.progress or {})
    requested_symbols = list(progress.get("requested_symbols") or [])
    requested_intervals = list(progress.get("requested_intervals") or [])

    if not requested_symbols:
        if job.symbols:
            requested_symbols = list(job.symbols)
        elif job.symbol:
            requested_symbols = [job.symbol]
    if not requested_intervals:
        if job.intervals:
            requested_intervals = list(job.intervals)
        elif job.interval:
            requested_intervals = [job.interval]

    if not requested_symbols or not requested_intervals:
        raise ValueError(
            f"backfill_chunk: job {job.job_id} has no requested symbols/intervals"
        )
    return requested_symbols, requested_intervals


def _acquire_bulk_backfill_slot(rate_limiter, circuit, provider: str) -> bool:
    """Gate a single fetcher call on the bulk_backfill Redis bucket.

    Returns True if the fetch is allowed to proceed. Returns False when the
    circuit is open or the rate limit wait times out; the caller should
    then skip this tuple (the next call site re-checks the cancel flag and
    the loop can retry on the next invocation).
    """
    if not circuit.allow_request():
        logger.warning("Circuit open for %s, skipping bulk_backfill fetch", provider)
        return False
    provider_cfg = PROVIDER_LIMITS.get(provider, {})
    window = provider_cfg.get("window_seconds", 3600)
    limit_key = _BACKFILL_LIMIT_KEYS.get(provider, "ratelimit_finmind_backfill_hourly")
    limit = getattr(settings, limit_key, 100)
    acquired = rate_limiter.wait_and_acquire(
        provider, window, limit, timeout=30, budget_class="bulk_backfill"
    )
    if not acquired:
        logger.warning(
            "bulk_backfill rate limit timeout for %s (limit_key=%s limit=%d)",
            provider,
            limit_key,
            limit,
        )
    return acquired


def _job_cancelled(session, job_id: UUID) -> bool:
    """Re-read the row just to check the cancel flag.

    A fresh read is required because the ORM session cache could otherwise
    hold a stale ``job.status`` from before the API cancel committed.
    """
    # Expire the existing row so the next access hits the DB.
    current = session.query(BackfillJob).filter_by(job_id=job_id).one()
    session.refresh(current)
    return current.status == "cancelled"


def _run_backfill_chunk(session, job_id: str) -> dict:
    """Core loop — separated from the Celery task wrapper so unit tests can
    drive it with an injected session without going through broker.

    Walks the request-level scope stored in ``progress`` via ``cursor``
    indices, cooperatively cancels between chunks, and re-enqueues itself
    on queue ``backfill`` when ``MAX_CHUNKS_PER_CALL`` is reached but
    unfinished tuples remain. See module docstring for full semantics.
    """
    job_uuid = UUID(job_id)
    repo = DataRepository(session)
    job = session.query(BackfillJob).filter_by(job_id=job_uuid).one()
    if job.status in _TERMINAL_STATUSES:
        return {"job_id": job_id, "status": job.status, "noop": True}

    job.status = "running"
    if job.started_at is None:
        job.started_at = datetime.now(timezone.utc)
    session.commit()

    requested_symbols, requested_intervals = _resolve_request_scope(job)

    cursor = dict(job.cursor or {})
    symbol_idx = int(cursor.get("symbol_idx", 0) or 0)
    interval_idx = int(cursor.get("interval_idx", 0) or 0)
    request_start_iso = cursor.get("start_ts") or cursor.get("next_ts")
    request_end_iso = cursor.get("end_ts")
    request_start = _parse_ts(request_start_iso)
    request_end = _parse_ts(request_end_iso) or datetime.now(timezone.utc)
    if request_start is None:
        raise ValueError(f"backfill_chunk: job {job_id} has no start cursor")
    current_tuple_start = _parse_ts(cursor.get("next_ts")) or request_start

    # Load market config once; fall back to spot instrument if symbols.yaml
    # is unavailable (fresh API symbols not yet committed to yaml).
    try:
        config = load_symbols()
        market_cfg = get_market_config(job.market, config)
    except Exception:  # noqa: BLE001
        market_cfg = None
    instrument = market_cfg.instrument if market_cfg else "spot"

    # Phase 39 D-16/D-17: set up the bulk_backfill rate limiter + circuit.
    # Instantiate through module attributes so unit tests can monkeypatch.
    redis_client = _get_redis_client()
    circuit = CircuitBreaker(
        redis_client,
        MARKET_TO_PROVIDER.get(job.market, "unknown"),
        failure_threshold=settings.circuit_failure_threshold,
        open_timeout=settings.circuit_open_timeout,
        failure_window=settings.circuit_failure_window,
    )
    rate_limiter = DistributedRateLimiter(redis_client)
    provider = MARKET_TO_PROVIDER.get(job.market, "unknown")

    fetcher = get_fetcher(job.market)
    chunks_this_call = 0
    rows_written_total = int((job.progress or {}).get("rows_written", 0) or 0)
    chunks_done_total = int((job.progress or {}).get("chunks_done", 0) or 0)

    # Outer loop: tuples (symbol_idx, interval_idx) in row-major order.
    while symbol_idx < len(requested_symbols):
        current_symbol = requested_symbols[symbol_idx]
        while interval_idx < len(requested_intervals):
            current_interval = requested_intervals[interval_idx]
            interval_sec = _INTERVAL_SECONDS.get(current_interval, 3600)
            chunk_td = timedelta(seconds=interval_sec * CHUNK_CANDLES)

            # Mirror the current tuple into the legacy columns so
            # /backfill/status and operator tooling can see what the worker
            # is actively processing.
            if job.symbol != current_symbol or job.interval != current_interval:
                job.symbol = current_symbol
                job.interval = current_interval
                session.commit()

            # Inner loop: walk ``current_tuple_start`` → ``request_end`` in
            # chunks, honouring MAX_CHUNKS_PER_CALL and cooperative cancel.
            while current_tuple_start < request_end and chunks_this_call < MAX_CHUNKS_PER_CALL:
                if _job_cancelled(session, job_uuid):
                    logger.info(
                        "backfill_chunk: job %s observed cancelled flag; exiting cleanly",
                        job_id,
                    )
                    return {"job_id": job_id, "status": "cancelled", "noop": True}

                if not _acquire_bulk_backfill_slot(rate_limiter, circuit, provider):
                    # Back off: leave the cursor where it is so the next
                    # scheduled invocation can retry this tuple. Return
                    # without marking succeeded.
                    _persist_cursor(
                        session,
                        job,
                        symbol_idx=symbol_idx,
                        interval_idx=interval_idx,
                        next_ts=current_tuple_start,
                        request_start_iso=request_start_iso or request_start.isoformat(),
                        request_end_iso=request_end_iso or request_end.isoformat(),
                        rows_written=rows_written_total,
                        chunks_done=chunks_done_total,
                    )
                    backfill_chunk.delay(job_id)
                    return {
                        "job_id": job_id,
                        "status": job.status,
                        "rescheduled": True,
                        "reason": "rate_limited",
                    }

                batch_end = min(current_tuple_start + chunk_td, request_end)
                try:
                    df = fetcher.fetch_ohlcv(
                        current_symbol,
                        current_interval,
                        current_tuple_start.strftime("%Y-%m-%d"),
                        batch_end.strftime("%Y-%m-%d"),
                    )
                    circuit.record_success()
                except Exception:  # noqa: BLE001
                    circuit.record_failure()
                    raise

                if df is not None and not df.empty:
                    vresult = validate_ohlcv(df, job.market)
                    if vresult.has_critical:
                        job.status = "failed"
                        job.error = (
                            f"critical validation failure on {current_symbol}/{current_interval}"
                        )
                        job.finished_at = datetime.now(timezone.utc)
                        session.commit()
                        return {
                            "job_id": job_id,
                            "status": "failed",
                            "chunks_done": chunks_done_total,
                        }
                    # --- CRITICAL SECTION (Phase 38 D-08) ------------------
                    count = repo.upsert_ohlcv(
                        df,
                        current_symbol,
                        job.market,
                        instrument,
                        current_interval,
                    )
                    session.commit()
                    # Explicit commit: DataRepository.upsert_ohlcv does NOT
                    # auto-commit (D-10). SIGKILL here is safe: replay
                    # hits ON CONFLICT pk_ohlcv and dedupes (D-09).
                    # ------------------------------------------------------
                    rows_written_total += int(count or 0)

                chunks_done_total += 1
                chunks_this_call += 1
                current_tuple_start = batch_end
                # Advance cursor strictly AFTER the upsert commit.
                _persist_cursor(
                    session,
                    job,
                    symbol_idx=symbol_idx,
                    interval_idx=interval_idx,
                    next_ts=current_tuple_start,
                    request_start_iso=request_start_iso or request_start.isoformat(),
                    request_end_iso=request_end_iso or request_end.isoformat(),
                    rows_written=rows_written_total,
                    chunks_done=chunks_done_total,
                )

            if current_tuple_start >= request_end:
                # Current (symbol, interval) tuple is complete; move to the
                # next interval and reset the window.
                interval_idx += 1
                current_tuple_start = request_start
                _persist_cursor(
                    session,
                    job,
                    symbol_idx=symbol_idx,
                    interval_idx=interval_idx,
                    next_ts=current_tuple_start,
                    request_start_iso=request_start_iso or request_start.isoformat(),
                    request_end_iso=request_end_iso or request_end.isoformat(),
                    rows_written=rows_written_total,
                    chunks_done=chunks_done_total,
                )
                continue

            # We broke out of the inner loop without completing the tuple,
            # which means MAX_CHUNKS_PER_CALL was hit. Re-enqueue and exit.
            backfill_chunk.delay(job_id)
            return {
                "job_id": job_id,
                "status": job.status,
                "rescheduled": True,
                "chunks_done": chunks_done_total,
            }

        # All intervals for the current symbol are complete; advance symbol
        # index and reset interval cursor + window for the next symbol.
        symbol_idx += 1
        interval_idx = 0
        current_tuple_start = request_start
        _persist_cursor(
            session,
            job,
            symbol_idx=symbol_idx,
            interval_idx=interval_idx,
            next_ts=current_tuple_start,
            request_start_iso=request_start_iso or request_start.isoformat(),
            request_end_iso=request_end_iso or request_end.isoformat(),
            rows_written=rows_written_total,
            chunks_done=chunks_done_total,
        )

    # All tuples complete — mark succeeded + finished_at.
    job.status = "succeeded"
    job.finished_at = datetime.now(timezone.utc)
    session.commit()

    # Phase 39 plan 39-03 Task 3 / D-12: immediately refresh the coverage
    # materialized view once a manual backfill completes so operators see
    # fresh coverage rows on the dashboard/API without waiting for the
    # hourly beat tick. Delegated to the dedicated ``backfill`` queue via
    # ``coverage_view_refresh.delay()`` to avoid blocking the current
    # worker slot on the REFRESH.
    try:
        coverage_view_refresh.delay()
    except Exception:  # noqa: BLE001
        # Never fail a successful job because the refresh enqueue hiccuped;
        # the hourly beat task will still pick the view back up.
        logger.exception(
            "backfill_chunk: failed to enqueue coverage_view_refresh for job %s",
            job_id,
        )

    return {
        "job_id": job_id,
        "status": "succeeded",
        "chunks_done": chunks_done_total,
        "rows_written": rows_written_total,
    }


def _persist_cursor(
    session,
    job: BackfillJob,
    *,
    symbol_idx: int,
    interval_idx: int,
    next_ts: datetime,
    request_start_iso: str,
    request_end_iso: str,
    rows_written: int,
    chunks_done: int,
) -> None:
    """Persist the current cursor + progress snapshot.

    This helper is the one place that writes the cursor so the shape stays
    consistent across the cold-start, mid-chunk, and tuple-boundary writes.

    Phase 38 backward-compat: ``completed_chunks`` / ``total_chunks`` keys
    are retained in the cursor so the legacy single-tuple dispatcher path
    and the existing Phase 38 test suite continue to read meaningful
    progress values.
    """
    previous = dict(job.cursor or {})
    cursor_update = {
        "symbol_idx": symbol_idx,
        "interval_idx": interval_idx,
        "next_ts": next_ts.isoformat(),
        "start_ts": request_start_iso,
        "end_ts": request_end_iso,
        "completed_chunks": chunks_done,
    }
    if "total_chunks" in previous:
        cursor_update["total_chunks"] = previous["total_chunks"]
    job.cursor = cursor_update
    progress = dict(job.progress or {})
    progress["rows_written"] = rows_written
    progress["chunks_done"] = chunks_done
    job.progress = progress
    session.commit()


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
    ``pk_ohlcv``), and advances the cursor AFTER the write commits
    (Phase 38 D-08).
    """
    with db_session() as session:
        try:
            return _run_backfill_chunk(session, job_id)
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            job = session.query(BackfillJob).filter_by(job_id=UUID(job_id)).first()
            if job is not None:
                job.status = "failed"
                job.error = str(exc)[:2000]
                job.finished_at = datetime.now(timezone.utc)
                session.commit()
            logger.exception("backfill_chunk failed for job_id=%s", job_id)
            raise self.retry(exc=exc)


@celery_app.task(
    name="poseidon.workers.backfill_tasks.coverage_view_refresh",
    queue="backfill",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def coverage_view_refresh(self) -> dict:
    """Refresh the ``data_coverage_mv`` materialized view (Phase 39 D-12).

    Scheduled hourly on celery beat and also triggered via
    ``coverage_view_refresh.delay()`` from the success path of
    :func:`backfill_chunk` so operators see fresh coverage immediately
    after a manual backfill completes.

    Runs on the dedicated ``backfill`` queue so it never contends with
    live-ingest or model-training workers.
    """
    with db_session() as session:
        try:
            coverage_helpers.refresh_data_coverage_mv(session)
            return {"status": "refreshed"}
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.exception("coverage_view_refresh failed: %s", exc)
            raise self.retry(exc=exc)
