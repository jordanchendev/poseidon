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


def _stub_frame(start: datetime, batch_end: datetime, interval_sec: int = 3600):
    """Build a realistic OHLCV frame covering [start, batch_end)."""
    times = []
    t = start
    while t < batch_end:
        times.append(t)
        t = t + timedelta(seconds=interval_sec)
    return pd.DataFrame(
        {
            "time": times,
            "open": [100.0] * len(times),
            "high": [101.0] * len(times),
            "low": [99.0] * len(times),
            "close": [100.5] * len(times),
            "volume": [10.0] * len(times),
        }
    )


class _StubFetcher:
    def __init__(self, interval_sec=3600):
        self.interval_sec = interval_sec
        self.calls = 0

    def fetch_ohlcv(self, symbol, interval, start_s, end_s):
        self.calls += 1
        start = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)
        return _stub_frame(start, end, self.interval_sec)


class _ValidationOK:
    has_critical = False
    warning_count = 0
    checks = []


@pytest.fixture
def stub_fetcher_and_validate(monkeypatch):
    """Patch fetcher + validator + load_symbols so _run_backfill_chunk is
    fully driven by stubs."""
    fetcher = _StubFetcher()
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _ValidationOK())

    class _Cfg:
        instrument = "spot"

    monkeypatch.setattr(backfill_tasks, "load_symbols", lambda: object())
    monkeypatch.setattr(backfill_tasks, "get_market_config", lambda market, cfg: _Cfg())
    # Hard-cap chunks per call so short-horizon jobs exit after 1-2 iters.
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 5)
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


def test_onconflict_dedupe(db_session, stub_fetcher_and_validate, backfill_job):
    """Running backfill_chunk twice over the same window produces the same row
    count (ON CONFLICT pk_ohlcv dedupes)."""
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
    # Replace fetcher + validator with passing stubs.
    fetcher = _StubFetcher()
    monkeypatch.setattr(backfill_tasks, "get_fetcher", lambda market: fetcher)
    monkeypatch.setattr(backfill_tasks, "validate_ohlcv", lambda df, market: _ValidationOK())

    class _Cfg:
        instrument = "spot"
    monkeypatch.setattr(backfill_tasks, "load_symbols", lambda: object())
    monkeypatch.setattr(backfill_tasks, "get_market_config", lambda market, cfg: _Cfg())
    monkeypatch.setattr(backfill_tasks, "MAX_CHUNKS_PER_CALL", 5)

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
