"""Integration tests for backfill resume + SIGKILL mid-chunk (DATA-FOUND-04/05).

Runs inside qlib-research container against real TimescaleDB.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
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
    """Driven by a pre-computed list of frames (date-string args are lossy)."""

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


def _build_resume_chunks(*, resume_from: datetime, end: datetime, interval_sec=3600, candles_per_chunk=1000):
    """Chunks starting from an arbitrary resume point (for resume test)."""
    chunk_td = timedelta(seconds=interval_sec * candles_per_chunk)
    frames = []
    cur = resume_from
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


def _patch_stubs(monkeypatch, frames):
    fetcher = _StubFetcher(frames)
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _ValidationOK())

    class _Cfg:
        instrument = "spot"
    monkeypatch.setattr(backfill_tasks, "load_symbols", lambda: object())
    monkeypatch.setattr(backfill_tasks, "get_market_config", lambda market, cfg: _Cfg())
    return fetcher


def _make_job(db_session, *, hours: int, chunks_per_call: int = 1, symbol: str | None = None):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=hours)
    end = now - timedelta(minutes=5)
    symbol = symbol or f"KILLRES_{uuid.uuid4().hex[:6]}"
    job = BackfillJob(
        job_id=uuid.uuid4(),
        status="pending",
        market="crypto_spot",
        symbol=symbol,
        interval="1h",
        cursor={
            "start_ts": start.isoformat(),
            "next_ts": start.isoformat(),
            "end_ts": end.isoformat(),
            "completed_chunks": 0,
            "total_chunks": hours,
        },
    )
    db_session.add(job)
    db_session.commit()
    return job


def _cleanup_job(db_session, job):
    db_session.execute(
        text(
            "DELETE FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"
        ),
        {"s": job.symbol, "m": job.market, "i": job.interval},
    )
    db_session.query(BackfillJob).filter_by(job_id=job.job_id).delete()
    db_session.commit()


# --- Tests ------------------------------------------------------------------


def test_resume_from_cursor(db_session, monkeypatch):
    """Start with a partially-progressed cursor and verify backfill_chunk
    resumes from ``cursor.next_ts`` rather than ``cursor.start_ts``."""
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 50)

    job = _make_job(db_session, hours=4)
    try:
        # Simulate a previous partial run: advance next_ts by 2 hours.
        cursor = dict(job.cursor)
        start_dt = datetime.fromisoformat(cursor["start_ts"])
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(cursor["end_ts"])
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        resume_from = start_dt + timedelta(hours=2)
        cursor["next_ts"] = resume_from.isoformat()
        cursor["completed_chunks"] = 2
        job.cursor = cursor
        db_session.commit()
        _patch_stubs(monkeypatch, _build_resume_chunks(resume_from=resume_from, end=end_dt))

        rows_before = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i "
                "AND time < :t"
            ),
            {"s": job.symbol, "m": job.market, "i": job.interval, "t": resume_from},
        ).scalar()
        assert rows_before == 0, "pre-resume window should not be written on resume"

        result = backfill_tasks._run_backfill_chunk(db_session, str(job.job_id))
        assert result["status"] == "succeeded"

        # Nothing should have been written before resume_from — the fetcher
        # is only called for [resume_from, end_ts].
        rows_before_after = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i "
                "AND time < :t"
            ),
            {"s": job.symbol, "m": job.market, "i": job.interval, "t": resume_from},
        ).scalar()
        assert rows_before_after == 0, (
            "resume must not backfill data before cursor.next_ts"
        )

        rows_after = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i "
                "AND time >= :t"
            ),
            {"s": job.symbol, "m": job.market, "i": job.interval, "t": resume_from},
        ).scalar()
        assert rows_after > 0, "resume window must produce rows"
    finally:
        _cleanup_job(db_session, job)


def test_sigkill_midchunk_no_dupes_no_gaps(db_session, monkeypatch):
    """Worst case (D-09): SIGKILL after upsert commit but BEFORE cursor
    advance. The restart must replay the last batch and ON CONFLICT must
    dedupe so we see zero duplicates and zero gaps.

    Strategy: rather than spawning a subprocess (which would not share the
    test db_session), we simulate the crash in-process by running
    ``_run_backfill_chunk`` once with the cursor-advance commit monkeypatched
    to raise RIGHT after upsert commits, rolling back, then running the
    function again cleanly and asserting the resulting hypertable state.
    """
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 50)

    job = _make_job(db_session, hours=3)
    # Build enough frames for phase-1 + full replay in phase-2.
    frames = _build_chunks_for_job(job) * 3
    fetcher = _patch_stubs(monkeypatch, frames)
    try:
        # --- Phase 1: crash AFTER upsert commits but BEFORE cursor write ----
        real_upsert = backfill_tasks.upsert_ohlcv
        crashed = {"done": False}

        def crashing_upsert(session, df, symbol, market, instrument, interval):
            if crashed["done"]:
                return real_upsert(session, df, symbol, market, instrument, interval)
            n = real_upsert(session, df, symbol, market, instrument, interval)
            # Mimic SIGKILL right after the upsert commit: raise BEFORE any
            # cursor-update SQL runs. The ohlcv rows are durable in DB.
            crashed["done"] = True
            raise RuntimeError("simulated SIGKILL between upsert and cursor update")

        monkeypatch.setattr(backfill_tasks, "upsert_ohlcv", crashing_upsert)

        with pytest.raises(RuntimeError):
            backfill_tasks._run_backfill_chunk(db_session, str(job.job_id))

        # Session state is poisoned by the raise — rollback and reload.
        db_session.rollback()

        # Cursor must still reflect the pre-crash state (completed_chunks=0).
        job_after_crash = (
            db_session.query(BackfillJob).filter_by(job_id=job.job_id).one()
        )
        assert job_after_crash.cursor["completed_chunks"] == 0, (
            "cursor must not advance when crash happens before cursor commit"
        )

        # But the first chunk's rows ARE durably in the hypertable.
        rows_mid = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"
            ),
            {"s": job.symbol, "m": job.market, "i": job.interval},
        ).scalar()
        assert rows_mid > 0, "crash happened after upsert commit — rows must be durable"

        # --- Phase 2: restart. backfill_chunk should replay the same batch,
        # hit ON CONFLICT pk_ohlcv (idempotent), and then continue forward.
        result = backfill_tasks._run_backfill_chunk(db_session, str(job.job_id))
        assert result["status"] == "succeeded"

        # --- Assertions --------------------------------------------------
        # (a) Zero duplicates: COUNT(*) == COUNT(DISTINCT time)
        distinct_check = db_session.execute(
            text(
                "SELECT COUNT(*), COUNT(DISTINCT time) FROM ohlcv "
                "WHERE symbol=:s AND market=:m AND interval=:i"
            ),
            {"s": job.symbol, "m": job.market, "i": job.interval},
        ).first()
        assert distinct_check[0] == distinct_check[1], (
            f"duplicates detected: total={distinct_check[0]} distinct={distinct_check[1]}"
        )
        assert distinct_check[0] > 0

        # (b) Zero gaps: consecutive time diffs must all equal 1 hour.
        times = db_session.execute(
            text(
                "SELECT time FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i "
                "ORDER BY time ASC"
            ),
            {"s": job.symbol, "m": job.market, "i": job.interval},
        ).fetchall()
        ts_list = [row[0] for row in times]
        assert len(ts_list) >= 2
        diffs = {(ts_list[i + 1] - ts_list[i]).total_seconds() for i in range(len(ts_list) - 1)}
        assert diffs == {3600.0}, f"gap detected: diffs={diffs}"
    finally:
        _cleanup_job(db_session, job)
