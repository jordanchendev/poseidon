"""Tests for historical_backfill_dispatcher (Phase 39 plan 39-04 / BACKFILL-05).

The dispatcher walks every (market, symbol, interval) tuple in symbols.yaml,
checks ingest_state.first_backfill_done, and creates a BackfillJob row via
poseidon.data.backfill_jobs.create_backfill_job for any tuple that has not
been backfilled yet — with duplicate suppression so that a queued or running
job covering the same tuple short-circuits a second dispatch.

Default lookback windows (Phase 39 RESEARCH Recommendation 5):

    1d:  730 days   4h:  730 days   1h:  180 days
    30m: 90 days    15m: 60 days    5m:  30 days

Tested behaviours:

* tuples with first_backfill_done=true are skipped
* queued/running jobs already covering a tuple suppress duplicate dispatch
* default lookback for crypto_perp 4h is exactly now - 730 days
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility shims for Postgres-only types -------------------
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.models.base import Base  # noqa: E402
from poseidon.models.backfill import BackfillJob  # noqa: E402
from poseidon.models.ingest_state import IngestState  # noqa: E402
from poseidon.data.symbols import (  # noqa: E402
    MarketConfig,
    SymbolConfig,
    SymbolInfo,
)


# --- Test DB wiring --------------------------------------------------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(autouse=True)
def _setup_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture
def session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def fake_symbols_config():
    """Minimal SymbolConfig with one crypto_perp tuple and one crypto_spot tuple."""
    return SymbolConfig(
        markets={
            "crypto_perp": MarketConfig(
                instrument="perpetual",
                intervals=["4h"],
                symbols=[
                    SymbolInfo(id="BTCUSDT", name="Bitcoin Perp", ccxt_symbol="BTC/USDT:USDT"),
                ],
            ),
            "crypto_spot": MarketConfig(
                instrument="spot",
                intervals=["1d"],
                symbols=[
                    SymbolInfo(id="ETHUSDT", name="Ethereum", ccxt_symbol="ETH/USDT"),
                ],
            ),
        }
    )


@pytest.fixture
def patch_dispatcher(monkeypatch, fake_symbols_config):
    """Replace SessionLocal + load_symbols inside the dispatcher module.

    The dispatcher is a Celery task that builds its own SessionLocal; the
    test wants it to talk to the SQLite in-memory engine instead.
    """
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(cpu_tasks, "load_symbols", lambda: fake_symbols_config)
    return cpu_tasks


# --- Tests -----------------------------------------------------------------


def test_dispatcher_creates_backfill_jobs_for_missing_tuples(session, patch_dispatcher):
    """Tuples with no ingest_state row receive a fresh BackfillJob."""
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    historical_backfill_dispatcher()

    rows = session.query(BackfillJob).order_by(BackfillJob.market).all()
    assert len(rows) == 2

    perp = next(r for r in rows if r.market == "crypto_perp")
    assert perp.symbols == ["BTCUSDT"]
    assert perp.intervals == ["4h"]
    assert perp.requested_by == "dispatcher"
    assert perp.status == "pending"


def test_dispatcher_skips_first_backfill_done_tuples(session, patch_dispatcher):
    """Tuples with first_backfill_done=true must NOT receive a new job."""
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    session.add(
        IngestState(
            symbol="BTCUSDT",
            market="crypto_perp",
            interval="4h",
            last_successful_ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
            first_backfill_done=True,
        )
    )
    session.commit()

    historical_backfill_dispatcher()

    perp_jobs = (
        session.query(BackfillJob).filter(BackfillJob.market == "crypto_perp").all()
    )
    assert perp_jobs == [], "first_backfill_done=true tuple should be skipped"

    # The other tuple (crypto_spot ETHUSDT 1d) is still missing → one job.
    spot_jobs = (
        session.query(BackfillJob).filter(BackfillJob.market == "crypto_spot").all()
    )
    assert len(spot_jobs) == 1


def test_dispatcher_skips_existing_running_job(session, patch_dispatcher):
    """A running job covering a tuple must suppress duplicate dispatch."""
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    # Pre-existing running job that already covers crypto_perp BTCUSDT 4h.
    existing = BackfillJob(
        status="running",
        market="crypto_perp",
        symbol=None,
        interval=None,
        symbols=["BTCUSDT"],
        intervals=["4h"],
        requested_by="api",
        cursor={
            "symbol_idx": 0,
            "interval_idx": 0,
            "next_ts": "2024-01-01T00:00:00+00:00",
            "end_ts": "2026-01-01T00:00:00+00:00",
        },
        progress={
            "requested_symbols": ["BTCUSDT"],
            "requested_intervals": ["4h"],
            "rows_written": 0,
            "chunks_done": 0,
            "chunks_total": None,
        },
    )
    session.add(existing)
    session.commit()

    historical_backfill_dispatcher()

    perp_jobs = (
        session.query(BackfillJob).filter(BackfillJob.market == "crypto_perp").all()
    )
    # The running job is still there, but no NEW dispatcher row was added.
    assert len(perp_jobs) == 1
    assert perp_jobs[0].requested_by == "api"


def test_dispatcher_skips_existing_pending_job(session, patch_dispatcher):
    """A pending (queued) job covering a tuple must suppress duplicate dispatch."""
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    pending = BackfillJob(
        status="pending",
        market="crypto_perp",
        symbol=None,
        interval=None,
        symbols=["BTCUSDT"],
        intervals=["4h"],
        requested_by="dispatcher",
        cursor={
            "symbol_idx": 0,
            "interval_idx": 0,
            "next_ts": "2024-01-01T00:00:00+00:00",
            "end_ts": "2026-01-01T00:00:00+00:00",
        },
        progress={
            "requested_symbols": ["BTCUSDT"],
            "requested_intervals": ["4h"],
            "rows_written": 0,
            "chunks_done": 0,
            "chunks_total": None,
        },
    )
    session.add(pending)
    session.commit()

    historical_backfill_dispatcher()

    perp_jobs = (
        session.query(BackfillJob).filter(BackfillJob.market == "crypto_perp").all()
    )
    assert len(perp_jobs) == 1, "queued covering job should suppress duplicate dispatch"


def test_dispatcher_does_not_skip_when_existing_job_is_terminal(session, patch_dispatcher):
    """A succeeded/failed/cancelled job is NOT a covering job — dispatch a new one.

    Without this rule, a single failed first-time backfill would prevent the
    dispatcher from ever retrying the tuple.
    """
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    failed = BackfillJob(
        status="failed",
        market="crypto_perp",
        symbol=None,
        interval=None,
        symbols=["BTCUSDT"],
        intervals=["4h"],
        requested_by="dispatcher",
        cursor=None,
        progress=None,
        error="boom",
    )
    session.add(failed)
    session.commit()

    historical_backfill_dispatcher()

    perp_jobs = (
        session.query(BackfillJob)
        .filter(BackfillJob.market == "crypto_perp")
        .order_by(BackfillJob.created_at)
        .all()
    )
    # Original failed row + a new dispatcher row.
    assert len(perp_jobs) == 2
    assert perp_jobs[1].status == "pending"
    assert perp_jobs[1].requested_by == "dispatcher"


def test_dispatcher_default_lookback_for_crypto_perp_4h(session, patch_dispatcher):
    """crypto_perp 4h default start must be exactly now - 730 days.

    Tolerance: 60 seconds, since the dispatcher reads its own ``now``.
    """
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    before = datetime.now(timezone.utc)
    historical_backfill_dispatcher()
    after = datetime.now(timezone.utc)

    perp = (
        session.query(BackfillJob).filter(BackfillJob.market == "crypto_perp").one()
    )
    cursor = perp.cursor or {}
    next_ts = datetime.fromisoformat(cursor["next_ts"])

    expected_lower = before - timedelta(days=730) - timedelta(seconds=60)
    expected_upper = after - timedelta(days=730) + timedelta(seconds=60)
    assert expected_lower <= next_ts <= expected_upper, (
        f"crypto_perp 4h start ({next_ts}) outside [{expected_lower}, {expected_upper}]"
    )


def test_dispatcher_default_lookback_per_interval(session, patch_dispatcher, monkeypatch):
    """All six default lookback windows are honoured per interval."""
    from poseidon.workers import cpu_tasks
    from poseidon.workers.cpu_tasks import historical_backfill_dispatcher

    cfg = SymbolConfig(
        markets={
            "crypto_spot": MarketConfig(
                instrument="spot",
                intervals=["1d", "4h", "1h", "30m", "15m", "5m"],
                symbols=[
                    SymbolInfo(id="BTCUSDT", name="Bitcoin", ccxt_symbol="BTC/USDT"),
                ],
            ),
        }
    )
    monkeypatch.setattr(cpu_tasks, "load_symbols", lambda: cfg)

    expected_days = {
        "1d": 730,
        "4h": 730,
        "1h": 180,
        "30m": 90,
        "15m": 60,
        "5m": 30,
    }

    before = datetime.now(timezone.utc)
    historical_backfill_dispatcher()
    after = datetime.now(timezone.utc)

    rows = session.query(BackfillJob).all()
    assert len(rows) == 6, f"expected one job per interval, got {len(rows)}"

    by_interval: dict[str, BackfillJob] = {}
    for r in rows:
        # dispatcher creates one job per (market, symbol, interval) tuple
        assert r.intervals and len(r.intervals) == 1
        by_interval[r.intervals[0]] = r

    for interval, days in expected_days.items():
        job = by_interval[interval]
        next_ts = datetime.fromisoformat((job.cursor or {})["next_ts"])
        lower = before - timedelta(days=days) - timedelta(seconds=60)
        upper = after - timedelta(days=days) + timedelta(seconds=60)
        assert lower <= next_ts <= upper, (
            f"{interval} default lookback wrong: {next_ts} not in [{lower}, {upper}]"
        )
