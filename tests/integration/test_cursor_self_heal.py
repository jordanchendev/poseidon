"""Integration test: long-gap recovery via cursor-mode ingest (DATA-FOUND-01).

Implemented in plan 38-02. Runs against the real poseidon DB inside the
qlib-research container on stormtrooper.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import text

pytestmark = pytest.mark.phase38


def _unique_key():
    tag = uuid.uuid4().hex[:10]
    return (f"SELFHEAL{tag}", "crypto_perp", "1h")


def _cleanup(session, symbol, market, interval):
    session.execute(
        text("DELETE FROM ingest_state WHERE symbol=:s AND market=:m AND interval=:i"),
        {"s": symbol, "m": market, "i": interval},
    )
    session.execute(
        text("DELETE FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i"),
        {"s": symbol, "m": market, "i": interval},
    )
    session.commit()


class _SessionProxy:
    def __init__(self, real):
        self._real = real
    def __getattr__(self, name):
        return getattr(self._real, name)
    def close(self):
        pass


def test_long_gap_recovery(db_session, hypertable_seed_ohlcv, ingest_state_seed, monkeypatch):
    """Seed ohlcv up to T-30d + ingest_state cursor at T-30d. Run cursor-mode
    fetch repeatedly — the gap closes with the cursor advancing after each
    batch commit."""
    from poseidon.data.ingest_cursor import IngestCursorService
    from poseidon.data.symbols import SymbolInfo
    from poseidon.models.ingest_state import IngestState
    from poseidon.workers import cpu_tasks

    symbol, market, interval = _unique_key()
    try:
        now = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
        cursor_start = now - timedelta(days=30)

        hypertable_seed_ohlcv(
            db_session,
            symbol=symbol,
            market=market,
            interval=interval,
            instrument="perp",
            timestamps=[cursor_start - timedelta(hours=1), cursor_start],
        )
        ingest_state_seed(
            db_session,
            symbol=symbol,
            market=market,
            interval=interval,
            last_successful_ts=cursor_start,
            first_backfill_done=True,
        )

        # Stub fetcher: return a single-row frame per chunk (real upsert writes
        # it; no network I/O). The helper advances after each commit, so the
        # cursor should monotonically march toward `now`.
        def _make_df(start_str, _end_str):
            ts = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return pd.DataFrame(
                [{
                    "time": ts, "open": 1.0, "high": 1.0,
                    "low": 1.0, "close": 1.0, "volume": 1.0,
                }]
            )

        class StubFetcher:
            def fetch_ohlcv(self, sym, itv, start, end):
                return _make_df(start, end)

        monkeypatch.setattr(cpu_tasks, "get_fetcher", lambda m: StubFetcher())
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
        monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _SessionProxy(db_session))

        sym_info = SymbolInfo(id=symbol, name=symbol)
        market_cfg = type("MC", (), {"instrument": "perp"})()

        # Run multiple ticks until the cursor reaches "now". A 1000-candle
        # 1h chunk is ~41 days, so typically a single tick closes the 30-day
        # gap, but we iterate defensively.
        svc = IngestCursorService()
        for _ in range(10):
            cpu_tasks._fetch_market_data_cursor(market, interval, [sym_info], market_cfg)
            cursor_now = svc.get_or_bootstrap(db_session, symbol, market, interval)
            if cursor_now is not None and cursor_now >= now - timedelta(hours=2):
                break

        final = (
            db_session.query(IngestState)
            .filter_by(symbol=symbol, market=market, interval=interval)
            .one()
        )
        assert final.last_successful_ts is not None
        assert final.last_successful_ts >= now - timedelta(hours=2), (
            f"cursor did not reach now: got {final.last_successful_ts} vs now={now}"
        )
        assert final.last_error is None

        # Zero-duplicate invariant: count rows in ohlcv — each successful chunk
        # upserts a single row with a distinct timestamp, and ON CONFLICT dedupes.
        dup_count = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ("
                "  SELECT time FROM ohlcv WHERE symbol=:s AND market=:m AND interval=:i "
                "  GROUP BY time HAVING COUNT(*) > 1"
                ") d"
            ),
            {"s": symbol, "m": market, "i": interval},
        ).scalar()
        assert dup_count == 0
    finally:
        _cleanup(db_session, symbol, market, interval)
