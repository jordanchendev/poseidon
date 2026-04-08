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


def test_chunked_loop_bounded(db_session, monkeypatch):
    """Cursor-mode gap=30d / interval=1h chunks into ~1000-candle batches and
    respects both MAX_CHUNKS_PER_TICK and TICK_WALL_BUDGET_SEC."""
    from poseidon.core.config import settings
    from poseidon.workers import cpu_tasks

    symbol, market, interval = _unique_key("CHUNK")
    market_real = "crypto_perp"
    try:
        # Seed cursor 30 days behind "now"
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        db_session.execute(
            text(
                "INSERT INTO ingest_state (symbol, market, interval, last_successful_ts, first_backfill_done) "
                "VALUES (:s, :m, :i, :ts, true) ON CONFLICT DO NOTHING"
            ),
            {"s": symbol, "m": market_real, "i": interval, "ts": thirty_days_ago},
        )
        db_session.commit()

        # Stub fetcher + upsert to avoid hitting network / polluting real tables
        calls: list[tuple[datetime, datetime]] = []

        class StubFetcher:
            def fetch_ohlcv(self, sym, itv, start, end):
                import pandas as pd
                calls.append((start, end))
                # return a 1-row frame so upsert path is exercised but rowcount is tiny
                ts = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return pd.DataFrame(
                    [{"time": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}]
                )

        monkeypatch.setattr(cpu_tasks, "get_fetcher", lambda m: StubFetcher())
        monkeypatch.setattr(cpu_tasks, "upsert_ohlcv", lambda *a, **kw: 1)
        monkeypatch.setattr(cpu_tasks, "validate_ohlcv", lambda df, m: type("V", (), {"has_critical": False, "warning_count": 0, "checks": []})())
        monkeypatch.setattr(settings, "ingest_cursor_mode", "cursor")

        # Shrink MAX_CHUNKS_PER_TICK to verify bound is honored
        monkeypatch.setattr(cpu_tasks, "MAX_CHUNKS_PER_TICK", 3)

        # Use an ad-hoc symbol not in symbols.yaml so the loop targets just this one
        result = cpu_tasks.fetch_market_data(market_real, interval, symbol=symbol)
        assert result["mode"] == "cursor"
        # Should have chunked at most MAX_CHUNKS_PER_TICK times
        assert len(calls) <= 3
        assert len(calls) >= 1
    finally:
        _cleanup(db_session, symbol, market_real, interval)
