"""Phase 39-02 worker isolation + cooperative cancel + self-reschedule tests.

Covers:

- Task 2: backfill_chunk cooperative cancel, tuple iteration, and
  self-reschedule on queue ``backfill``.
- Task 3: dedicated backfill-worker service in docker-compose.yml with
  the exact queue flag and concurrency the runbook expects.

These tests deliberately run against in-memory SQLite + stubbed fetcher /
rate limiter / Celery dispatch so they stay in the unit-test fast path and
do NOT require stormtrooper's TimescaleDB. The Phase 38 integration suite
(``tests/integration/test_backfill_kill_resume.py``) still exercises the
real Postgres path.

Stormtrooper smoke command operators/executors must run after deployment:

    docker compose exec api celery -A poseidon.workers.celery_app inspect active_queues
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID as _UUID, uuid4

import pandas as pd
import pytest

# --- SQLite compatibility shims for Postgres-only types --------------------
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.models.base import Base  # noqa: E402
from poseidon.models.backfill import BackfillJob  # noqa: E402
from poseidon.workers import backfill_tasks  # noqa: E402
from poseidon.workers.celery_app import celery_app  # noqa: E402


# --- Shared fixtures -------------------------------------------------------


@pytest.fixture()
def sqlite_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


class _ValidationOK:
    has_critical = False
    warning_count = 0
    checks: list = []


class _NonEmptyFetcher:
    """Returns a single-row frame per call so the loop can always advance."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def fetch_ohlcv(self, symbol, interval, start_s, end_s):
        self.calls.append((symbol, interval, start_s, end_s))
        return pd.DataFrame(
            {
                "time": [pd.Timestamp(start_s, tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1.0],
            }
        )


@pytest.fixture()
def stub_backfill_env(monkeypatch):
    """Patch all external dependencies of ``_run_backfill_chunk`` so it can
    run against SQLite without touching the real Postgres hypertable,
    network, or Redis.
    """
    fetcher = _NonEmptyFetcher()
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _ValidationOK())

    # upsert_ohlcv hits real TimescaleDB ON CONFLICT SQL that SQLite can't
    # parse, so stub it to a no-op that returns the row count.
    def _noop_upsert(session, df, symbol, market, instrument, interval):
        return len(df)

    monkeypatch.setattr(backfill_tasks, "upsert_ohlcv", _noop_upsert)

    class _Cfg:
        instrument = "spot"

    monkeypatch.setattr(backfill_tasks, "load_symbols", lambda: object())
    monkeypatch.setattr(backfill_tasks, "get_market_config", lambda m, c: _Cfg())
    return fetcher


def _make_multi_tuple_job(
    session,
    *,
    symbols=("BTCUSDT", "ETHUSDT"),
    intervals=("4h",),
    hours=8,
) -> BackfillJob:
    """Insert a request-level job with multiple symbols/intervals.

    Uses a short window (hours) so the test runs in milliseconds.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=hours)
    end = now - timedelta(hours=1)
    job = BackfillJob(
        job_id=uuid4(),
        status="pending",
        market="crypto_perp",
        symbol=None,
        interval=None,
        symbols=list(symbols),
        intervals=list(intervals),
        requested_by="api",
        cursor={
            "symbol_idx": 0,
            "interval_idx": 0,
            "next_ts": start.isoformat(),
            "end_ts": end.isoformat(),
        },
        progress={
            "requested_symbols": list(symbols),
            "requested_intervals": list(intervals),
            "rows_written": 0,
            "chunks_done": 0,
            "chunks_total": None,
        },
    )
    session.add(job)
    session.commit()
    return job


# --- Task 2 tests ----------------------------------------------------------


def test_backfill_chunk_iterates_tuples_from_progress(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """``backfill_chunk`` must walk ``progress.requested_symbols`` x
    ``progress.requested_intervals`` rather than relying on the legacy
    singular ``job.symbol`` / ``job.interval`` columns which are NULL for
    Phase 39 API jobs."""
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1000)
    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT", "ETHUSDT"), intervals=("4h",), hours=8
    )

    result = backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))

    # Every requested (symbol, interval) tuple must have been serviced.
    serviced_symbols = {call[0] for call in stub_backfill_env.calls}
    assert serviced_symbols == {"BTCUSDT", "ETHUSDT"}, (
        f"all requested symbols should have been fetched, got {serviced_symbols}"
    )
    assert result["status"] == "succeeded"

    reloaded = sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
    assert reloaded.status == "succeeded"
    assert reloaded.finished_at is not None


def test_backfill_chunk_writes_current_tuple_into_job_columns(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """Before servicing each chunk, the currently-active ``(symbol, interval)``
    must be written back into ``job.symbol`` / ``job.interval`` so operator
    tooling and the compatibility ``/backfill/status`` list endpoint can
    display the tuple currently being worked on."""
    # Force a single chunk per call so we can inspect the intermediate state.
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1)
    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT",), intervals=("1h",), hours=4
    )

    backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))
    reloaded = sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
    assert reloaded.symbol == "BTCUSDT"
    assert reloaded.interval == "1h"


def test_backfill_chunk_uses_bulk_backfill_budget_class(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """The backfill path must acquire rate-limit slots with
    ``budget_class='bulk_backfill'`` so a one-shot historical job cannot
    starve the periodic live-ingest quota."""
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1000)

    captured_budget_classes: list[str] = []

    class _StubRateLimiter:
        def __init__(self, *args, **kwargs):
            pass

        def wait_and_acquire(self, provider, window, limit, timeout=30, budget_class="live_ingest"):
            captured_budget_classes.append(budget_class)
            return True

    class _StubCircuit:
        def __init__(self, *args, **kwargs):
            pass

        def allow_request(self):
            return True

        def record_success(self):
            return None

        def record_failure(self):
            return None

    monkeypatch.setattr(backfill_tasks, "DistributedRateLimiter", _StubRateLimiter)
    monkeypatch.setattr(backfill_tasks, "CircuitBreaker", _StubCircuit)
    # The stubs do not need a real redis client; give it a sentinel.
    monkeypatch.setattr(backfill_tasks, "_get_redis_client", lambda: object())

    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT",), intervals=("1h",), hours=4
    )

    backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))
    assert captured_budget_classes, "rate limiter must be invoked on backfill path"
    assert all(bc == "bulk_backfill" for bc in captured_budget_classes), (
        f"all backfill rate-limit acquires should be bulk_backfill, got {captured_budget_classes}"
    )


def test_backfill_chunk_cooperatively_cancels_between_chunks(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """A cancelled job must stop making forward progress on the next chunk
    iteration without raising / retrying. The cursor and progress rows
    should be preserved so operators can see where the worker stopped."""
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1000)

    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT", "ETHUSDT"), intervals=("4h",), hours=16
    )

    # Flip to cancelled before running so the first iteration observes it.
    job.status = "cancelled"
    sqlite_session.commit()

    result = backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))
    assert result["status"] == "cancelled"
    assert result.get("noop") is True

    # The fetcher must not have been invoked: cooperative cancel exits clean.
    assert stub_backfill_env.calls == [], (
        "cancelled job must not trigger any fetcher calls"
    )
    reloaded = sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
    assert reloaded.status == "cancelled"


def test_backfill_chunk_cancelled_midway_stops_before_next_tuple(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """Cancellation observed mid-job (between chunk iterations) must stop
    the loop. Simulated by flipping status to 'cancelled' from inside the
    fetcher stub on the first call."""
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1000)

    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), intervals=("4h",), hours=8
    )

    real_fetcher = stub_backfill_env

    def _cancel_after_first_call(*args, **kwargs):
        df = _NonEmptyFetcher.fetch_ohlcv(real_fetcher, *args, **kwargs)
        if len(real_fetcher.calls) == 1:
            # Flip cancellation AFTER the first chunk commits but BEFORE
            # the loop re-enters. This requires a fresh read in the worker.
            sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).update(
                {BackfillJob.status: "cancelled"}
            )
            sqlite_session.commit()
        return df

    real_fetcher.fetch_ohlcv = _cancel_after_first_call  # type: ignore[method-assign]

    result = backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))
    # Worker should leave the job in cancelled state and NOT iterate the
    # rest of the tuples.
    assert result["status"] == "cancelled"
    reloaded = sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
    assert reloaded.status == "cancelled"


def test_backfill_chunk_self_reschedules_when_cap_hit(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """When ``MAX_CHUNKS_PER_CALL`` is reached but unfinished tuples remain,
    the worker must enqueue ``backfill_chunk.delay(job_id)`` on queue
    ``backfill`` again instead of (incorrectly) marking the job succeeded.
    """
    # One chunk per call so we hit the cap immediately.
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1)

    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), intervals=("4h",), hours=8
    )

    reenqueues: list[tuple] = []

    def fake_delay(*args, **kwargs):
        reenqueues.append(args)

        class _R:
            id = "stub-resched-id"

        return _R()

    monkeypatch.setattr(backfill_tasks.backfill_chunk, "delay", fake_delay)

    result = backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))

    assert reenqueues, "worker must re-enqueue itself when unfinished work remains"
    assert reenqueues[0][0] == str(job.job_id)

    reloaded = sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
    assert reloaded.status != "succeeded", (
        "unfinished job must not be incorrectly marked succeeded"
    )
    assert reloaded.finished_at is None
    assert result.get("rescheduled") is True


def test_backfill_chunk_final_tuple_marks_succeeded_and_sets_finished_at(
    sqlite_session, stub_backfill_env, monkeypatch
):
    """When the final (symbol, interval) tuple completes its window, the
    worker must mark ``status='succeeded'`` and stamp ``finished_at``."""
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 1000)

    job = _make_multi_tuple_job(
        sqlite_session, symbols=("BTCUSDT",), intervals=("1h",), hours=2
    )

    backfill_tasks._run_backfill_chunk(sqlite_session, str(job.job_id))
    reloaded = sqlite_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
    assert reloaded.status == "succeeded"
    assert reloaded.finished_at is not None


# --- Queue routing regression guard ---------------------------------------


def test_celery_task_routes_preserve_backfill_and_cpu_isolation():
    """Live cpu_tasks must stay on queue ``cpu`` and backfill_tasks must stay
    on queue ``backfill``. Regression guard for D-07."""
    routes = celery_app.conf.task_routes
    assert routes["poseidon.workers.cpu_tasks.*"]["queue"] == "cpu"
    assert routes["poseidon.workers.backfill_tasks.*"]["queue"] == "backfill"
    # GPU is a separate queue and must not be conflated with backfill.
    assert routes["poseidon.workers.gpu_tasks.*"]["queue"] == "gpu"


# --- Task 3 docker-compose verification ------------------------------------


_COMPOSE_PATH = (
    Path(__file__).resolve().parents[2] / "docker-compose.yml"
)


def test_docker_compose_has_dedicated_backfill_worker_service():
    """docker-compose.yml must define a dedicated ``backfill-worker`` service
    that subscribes ONLY to the ``backfill`` queue with concurrency 1, so the
    queue routing declared in celery_app.py actually has a real consumer."""
    body = _COMPOSE_PATH.read_text()
    assert "\n  backfill-worker:" in body, (
        "docker-compose.yml must define a top-level 'backfill-worker:' service"
    )
    assert "worker -Q backfill -c 1 --loglevel=info" in body, (
        "backfill-worker must subscribe to the 'backfill' queue with -c 1"
    )


def test_docker_compose_preserves_cpu_worker_service():
    """The cpu-worker must remain unchanged so live ingest keeps working.
    Regression guard against a plan that accidentally rewires cpu-worker
    onto the backfill queue."""
    body = _COMPOSE_PATH.read_text()
    assert "\n  cpu-worker:" in body
    # cpu-worker must NOT be subscribing to the backfill queue.
    cpu_block_start = body.index("\n  cpu-worker:")
    next_service_idx = body.find("\n  ", cpu_block_start + 1)
    cpu_block = (
        body[cpu_block_start:next_service_idx] if next_service_idx != -1 else body[cpu_block_start:]
    )
    assert "-Q cpu" in cpu_block, "cpu-worker must still subscribe to queue 'cpu'"
    assert "-Q backfill" not in cpu_block, (
        "cpu-worker must NOT subscribe to the 'backfill' queue"
    )
