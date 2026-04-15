"""Unit tests for backfill_chunk idempotency + cursor-after-commit (DATA-FOUND-05).

Runs inside the qlib-research container against real TimescaleDB via the
``db_session`` fixture.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import text

from poseidon.models.backfill import BackfillJob
from poseidon.workers import backfill_tasks

pytestmark = pytest.mark.phase38


# --- Helpers ---------------------------------------------------------------


class _StubFetcher:
    """Stub that yields pre-computed OHLCV frames per call.

    The real fetcher signature takes ``%Y-%m-%d`` strings which lose sub-day
    granularity, so the stub is driven by an explicit iterable of frames
    produced from the BackfillJob window by ``_build_chunks_for_job``.
    """

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def fetch_ohlcv(self, symbol, interval, start_s, end_s):
        if self.calls >= len(self.frames):
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        df = self.frames[self.calls]
        self.calls += 1
        return df


def _build_chunks_for_job(job, interval_sec=3600, candles_per_chunk=1000):
    """Replicate the loop's chunk walk to produce one DataFrame per chunk."""
    start = datetime.fromisoformat(job.cursor["start_ts"])
    end = datetime.fromisoformat(job.cursor["end_ts"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    chunk_td = timedelta(seconds=interval_sec * candles_per_chunk)
    frames = []
    cur = start
    while cur < end:
        batch_end = min(cur + chunk_td, end)
        times = []
        t = cur
        while t < batch_end:
            times.append(t)
            t = t + timedelta(seconds=interval_sec)
        frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "open": [100.0] * len(times),
                    "high": [101.0] * len(times),
                    "low": [99.0] * len(times),
                    "close": [100.5] * len(times),
                    "volume": [10.0] * len(times),
                }
            )
        )
        cur = batch_end
    return frames


class _ValidationOK:
    has_critical = False
    warning_count = 0
    checks = []


def _patch_job_stubs(monkeypatch, job):
    frames = _build_chunks_for_job(job)
    # Multiply by 3 so replays find more frames on each invocation.
    fetcher = _StubFetcher(frames * 3)
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _ValidationOK())

    class _Cfg:
        instrument = "spot"

    monkeypatch.setattr(backfill_tasks, "load_symbols", lambda: object())
    monkeypatch.setattr(backfill_tasks, "get_market_config", lambda market, cfg: _Cfg())
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 50)
    return fetcher


@pytest.fixture
def backfill_job(db_session):
    """Create a short-horizon BackfillJob, yield it, then delete rows written."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=3)
    end = now - timedelta(hours=1)
    symbol = f"TEST_{uuid.uuid4().hex[:8]}"
    market = "crypto_spot"
    interval = "1h"
    job = BackfillJob(
        job_id=uuid.uuid4(),
        status="pending",
        market=market,
        symbol=symbol,
        interval=interval,
        cursor={
            "start_ts": start.isoformat(),
            "next_ts": start.isoformat(),
            "end_ts": end.isoformat(),
            "completed_chunks": 0,
            "total_chunks": 2,
        },
    )
    db_session.add(job)
    db_session.commit()
    yield job
    # cleanup
    db_session.execute(
        text(
            "DELETE FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"
        ),
        {"s": symbol, "m": market, "i": interval},
    )
    db_session.query(BackfillJob).filter_by(job_id=job.job_id).delete()
    db_session.commit()


# --- Tests ------------------------------------------------------------------


def test_onconflict_dedupe(db_session, monkeypatch, backfill_job):
    """Running backfill_chunk twice over the same window produces the same row
    count (ON CONFLICT pk_ohlcv dedupes)."""
    _patch_job_stubs(monkeypatch, backfill_job)
    result1 = backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))
    assert result1["status"] == "succeeded"

    count1 = db_session.execute(
        text(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"
        ),
        {"s": backfill_job.symbol, "m": backfill_job.market, "i": backfill_job.interval},
    ).scalar()

    # Reset cursor to replay the whole job a second time.
    job = db_session.query(BackfillJob).filter_by(job_id=backfill_job.job_id).one()
    cursor = dict(job.cursor)
    cursor["next_ts"] = cursor["start_ts"]
    cursor["completed_chunks"] = 0
    job.cursor = cursor
    job.status = "pending"
    db_session.commit()

    result2 = backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))
    assert result2["status"] == "succeeded"

    count2 = db_session.execute(
        text(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"
        ),
        {"s": backfill_job.symbol, "m": backfill_job.market, "i": backfill_job.interval},
    ).scalar()

    assert count1 > 0, "first run should write at least one row"
    assert count2 == count1, f"replay must not duplicate rows (got {count2} vs {count1})"


def test_cursor_advances_after_upsert(db_session, monkeypatch, backfill_job):
    """Cursor must advance strictly AFTER upsert_ohlcv commits. If the upsert
    raises, the cursor must remain at its pre-call position."""
    _patch_job_stubs(monkeypatch, backfill_job)

    # Upsert raises BEFORE cursor advance — cursor should stay unchanged.
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash before cursor advance")

    monkeypatch.setattr(backfill_tasks, "upsert_ohlcv", _boom)

    original_next_ts = backfill_job.cursor["next_ts"]

    with pytest.raises(RuntimeError):
        backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))

    db_session.rollback()
    job = db_session.query(BackfillJob).filter_by(job_id=backfill_job.job_id).one()
    assert job.cursor["next_ts"] == original_next_ts, (
        "cursor.next_ts must NOT advance when upsert raises (D-08)"
    )
    assert job.cursor["completed_chunks"] == 0


# --- Branch-coverage targeted tests ---------------------------------------


def test_parse_ts_helpers():
    assert backfill_tasks._parse_ts(None) is None
    # Naive iso string gets stamped UTC.
    parsed = backfill_tasks._parse_ts("2025-01-01T00:00:00")
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_noop_on_terminal_status(db_session, backfill_job):
    """Jobs already succeeded/cancelled short-circuit without touching
    fetcher, upsert, or cursor."""
    backfill_job.status = "succeeded"
    db_session.commit()
    result = backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))
    assert result == {
        "job_id": str(backfill_job.job_id),
        "status": "succeeded",
        "noop": True,
    }


def test_missing_cursor_raises(db_session, backfill_job):
    """A BackfillJob with no start cursor must raise ValueError (loud
    failure; worker wrapper marks the job failed)."""
    backfill_job.cursor = {}  # no start_ts / next_ts
    db_session.commit()
    with pytest.raises(ValueError, match="no start cursor"):
        backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))


def test_load_symbols_failure_falls_back_to_spot(db_session, monkeypatch, backfill_job):
    """If load_symbols raises (e.g. symbols.yaml missing for a symbol
    Phase 39 created on the fly), the loop must fall back to instrument
    ``spot`` rather than crashing the job."""
    def _boom():
        raise RuntimeError("no symbols.yaml in this env")
    monkeypatch.setattr(backfill_tasks, "load_symbols", _boom)
    # Still need fetcher + validator + frames.
    frames = _build_chunks_for_job(backfill_job)
    fetcher = _StubFetcher(frames)
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _ValidationOK())
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 50)

    result = backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))
    assert result["status"] == "succeeded"


def test_critical_validation_marks_failed(db_session, monkeypatch, backfill_job):
    """Critical validation failure must mark the job failed and return
    without touching upsert/cursor."""
    frames = _build_chunks_for_job(backfill_job)
    fetcher = _StubFetcher(frames)
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)

    class _Critical:
        has_critical = True
        warning_count = 0
        checks = []
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _Critical())

    class _Cfg:
        instrument = "spot"
    monkeypatch.setattr(backfill_tasks, "load_symbols", lambda: object())
    monkeypatch.setattr(backfill_tasks, "get_market_config", lambda m, c: _Cfg())

    # Upsert MUST NOT be called on critical validation.
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("upsert_ohlcv called despite critical validation")
    monkeypatch.setattr(backfill_tasks, "upsert_ohlcv", _should_not_be_called)

    result = backfill_tasks._run_backfill_chunk(db_session, str(backfill_job.job_id))
    assert result["status"] == "failed"

    db_session.rollback()
    job = db_session.query(BackfillJob).filter_by(job_id=backfill_job.job_id).one()
    assert job.status == "failed"
    assert "critical" in (job.error or "")


def test_backfill_chunk_wrapper_marks_failed_and_retries(monkeypatch, backfill_job):
    """The Celery task wrapper must rollback, persist ``status=failed`` +
    error, and invoke ``self.retry``. Simulate by calling ``run`` directly
    on a fake task instance."""
    # Patch db_session to produce a real session so the wrapper touches DB.
    import os
    from contextlib import contextmanager
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    real_dsn = (
        os.environ.get("POSEIDON_REAL_DATABASE_URL")
        or "postgresql://poseidon:poseidon@host.docker.internal:5433/poseidon"
    )
    engine = create_engine(real_dsn, future=True)
    LiveSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    @contextmanager
    def mock_db_session():
        session = LiveSession()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(backfill_tasks, "db_session", mock_db_session)

    # Force _run_backfill_chunk to raise.
    def _boom(session, job_id):
        raise RuntimeError("boom-from-wrapper-test")
    monkeypatch.setattr(backfill_tasks, "_run_backfill_chunk", _boom)

    # Stub retry so it re-raises rather than scheduling a real Celery retry.
    def _retry(self, exc=None, **kwargs):
        raise exc
    monkeypatch.setattr(
        type(backfill_tasks.backfill_chunk), "retry", _retry, raising=False
    )

    with pytest.raises(RuntimeError, match="boom-from-wrapper-test"):
        backfill_tasks.backfill_chunk.run(str(backfill_job.job_id))

    # Verify the job was marked failed.
    sess = LiveSession()
    try:
        job = sess.query(BackfillJob).filter_by(job_id=backfill_job.job_id).one()
        assert job.status == "failed"
        assert "boom-from-wrapper-test" in (job.error or "")
    finally:
        sess.close()
        engine.dispose()
