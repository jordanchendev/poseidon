"""Unit tests for IngestCursorService (DATA-FOUND-01/02) and cursor-mode chunked loop.

Implemented in plan 38-02. Runs against the live poseidon DB inside the
cpu-worker container on stormtrooper.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.phase38


def _unique_key(prefix: str) -> tuple[str, str, str]:
    """Return a (symbol, market, interval) tuple unique to this test run."""
    tag = uuid.uuid4().hex[:10]
    return (f"TEST{prefix}{tag}", "crypto_perp", "1h")


def _cleanup(db_session, symbol, market, interval):
    db_session.execute(
        text("DELETE FROM ingest_state WHERE symbol=:s AND market=:m AND interval=:i"),
        {"s": symbol, "m": market, "i": interval},
    )
    db_session.execute(
        text("DELETE FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"),
        {"s": symbol, "m": market, "i": interval},
    )
    db_session.commit()


def test_bootstrap_from_hypertable(db_session, hypertable_seed_ohlcv):
    from poseidon.data.ingest_cursor import IngestCursorService
    from poseidon.models.ingest_state import IngestState

    symbol, market, interval = _unique_key("BOOT")
    try:
        max_ts = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        hypertable_seed_ohlcv(
            db_session,
            symbol=symbol,
            market=market,
            interval=interval,
            timestamps=[
                max_ts - timedelta(hours=2),
                max_ts - timedelta(hours=1),
                max_ts,
            ],
        )

        svc = IngestCursorService()
        result = svc.get_or_bootstrap(db_session, symbol, market, interval)
        assert result == max_ts

        row = (
            db_session.query(IngestState)
            .filter_by(symbol=symbol, market=market, interval=interval)
            .one()
        )
        assert row.first_backfill_done is True
        assert row.last_successful_ts == max_ts
    finally:
        _cleanup(db_session, symbol, market, interval)


def test_bootstrap_empty_hypertable(db_session):
    from poseidon.data.ingest_cursor import IngestCursorService
    from poseidon.models.ingest_state import IngestState

    symbol, market, interval = _unique_key("EMPTY")
    try:
        svc = IngestCursorService()
        result = svc.get_or_bootstrap(db_session, symbol, market, interval)
        assert result is None

        row = (
            db_session.query(IngestState)
            .filter_by(symbol=symbol, market=market, interval=interval)
            .one()
        )
        assert row.first_backfill_done is False
        assert row.last_successful_ts is None
    finally:
        _cleanup(db_session, symbol, market, interval)


def test_advance_after_commit(db_session, ingest_state_seed):
    from poseidon.data.ingest_cursor import IngestCursorService
    from poseidon.models.ingest_state import IngestState

    symbol, market, interval = _unique_key("ADV")
    try:
        ingest_state_seed(
            db_session,
            symbol=symbol,
            market=market,
            interval=interval,
            last_successful_ts=None,
            last_error="prior error",
            first_backfill_done=False,
        )

        svc = IngestCursorService()
        new_ts = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        svc.advance(db_session, symbol, market, interval, new_ts)

        row = (
            db_session.query(IngestState)
            .filter_by(symbol=symbol, market=market, interval=interval)
            .one()
        )
        db_session.refresh(row)
        assert row.last_successful_ts == new_ts
        assert row.last_error is None
        assert row.last_attempt_ts is not None
    finally:
        _cleanup(db_session, symbol, market, interval)


def test_bootstrap_idempotent(db_session):
    """Calling get_or_bootstrap twice must not raise UniqueViolation (Pitfall 1)."""
    from poseidon.data.ingest_cursor import IngestCursorService

    symbol, market, interval = _unique_key("IDEM")
    try:
        svc = IngestCursorService()
        first = svc.get_or_bootstrap(db_session, symbol, market, interval)
        second = svc.get_or_bootstrap(db_session, symbol, market, interval)
        assert first == second

        # Simulate a racing worker: pre-existing row, then another bootstrap
        svc.get_or_bootstrap(db_session, symbol, market, interval)
    finally:
        _cleanup(db_session, symbol, market, interval)


def test_chunked_loop_bounded(db_session, ingest_state_seed, monkeypatch):
    """Cursor-mode gap=30d / interval=1h chunks into ~1000-candle batches and
    respects MAX_CHUNKS_PER_TICK (hard bound)."""
    from unittest.mock import MagicMock

    import pandas as pd

    from poseidon.data.symbols import SymbolInfo
    from poseidon.workers import cpu_tasks

    symbol, market, interval = _unique_key("CHUNK")
    try:
        now = datetime.now(timezone.utc)
        ingest_state_seed(
            db_session,
            symbol=symbol,
            market=market,
            interval=interval,
            last_successful_ts=now - timedelta(days=30),
            first_backfill_done=True,
        )

        calls: list[tuple[str, str]] = []

        class StubFetcher:
            def fetch_ohlcv(self, sym, itv, start, end):
                calls.append((start, end))
                ts = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return pd.DataFrame(
                    [{
                        "time": ts, "open": 1.0, "high": 1.0,
                        "low": 1.0, "close": 1.0, "volume": 1.0,
                    }]
                )

        monkeypatch.setattr(cpu_tasks, "get_fetcher", lambda m: StubFetcher())
        monkeypatch.setattr(cpu_tasks, "upsert_ohlcv", lambda *a, **kw: 1)
        monkeypatch.setattr(
            cpu_tasks,
            "validate_ohlcv",
            lambda df, m: type("V", (), {"has_critical": False, "warning_count": 0, "checks": []})(),
        )
        monkeypatch.setattr(cpu_tasks, "_get_redis_client", lambda: MagicMock())
        monkeypatch.setattr(
            cpu_tasks,
            "CircuitBreaker",
            lambda *a, **kw: type(
                "C", (),
                {"allow_request": lambda self: True,
                 "record_success": lambda self: None,
                 "record_failure": lambda self: None},
            )(),
        )
        monkeypatch.setattr(
            cpu_tasks,
            "DistributedRateLimiter",
            lambda *a, **kw: type(
                "R", (),
                {"wait_and_acquire": lambda self, *a, **kw: True},
            )(),
        )
        # Stub db_session to return our live db_session so the helper sees
        # the seeded cursor row without opening a separate connection.
        from contextlib import contextmanager

        @contextmanager
        def _mock_db_session():
            yield _SessionProxy(db_session)

        monkeypatch.setattr(cpu_tasks, "db_session", _mock_db_session)

        # Shrink the per-tick bound so the test finishes fast
        monkeypatch.setattr(cpu_tasks, "MAX_CHUNKS_PER_TICK", 3)

        sym_info = SymbolInfo(id=symbol, name=symbol)
        market_cfg = type("MC", (), {"instrument": "perp"})()
        result = cpu_tasks._fetch_market_data_cursor(
            market, interval, [sym_info], market_cfg
        )

        assert result["mode"] == "cursor"
        assert 1 <= len(calls) <= 3
    finally:
        _cleanup(db_session, symbol, market, interval)


class _SessionProxy:
    """Wrap an existing session so SessionLocal() returns it but .close() is a no-op."""
    def __init__(self, real):
        self._real = real
    def __getattr__(self, name):
        return getattr(self._real, name)
    def close(self):
        pass
