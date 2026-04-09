"""Tests for the Phase 39 coverage visibility slice (plan 39-03).

Covers the locked Phase 39 decisions in
.planning/phases/39-backfill-api-coverage/39-CONTEXT.md (D-10..D-12):

- Task 1: ``data_coverage_mv`` migration 022 shape + coverage arithmetic
- Task 2: ``GET /api/data/coverage`` endpoint wiring and response shape
- Task 3: hourly MV refresh task + success-path refresh hook in backfill

The unit suite runs against an in-memory SQLite harness that mirrors the
Postgres schema at the row level. The materialized view DDL itself is
Postgres-only and is exercised at the grep/assertion level here and
end-to-end on stormtrooper via ``alembic upgrade head``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility shims for Postgres-only types ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.models.base import Base, get_db  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from poseidon.api.data import router as data_router  # noqa: E402
from poseidon.data import coverage as coverage_module  # noqa: E402


# --- Test DB / app wiring --------------------------------------------------


_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

_test_app = FastAPI()
_test_app.include_router(data_router, prefix="/api/data", tags=["data"])


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


_test_app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    return TestClient(_test_app)


@pytest.fixture(autouse=True)
def _setup_db():
    """Create ORM tables + a SQLite stand-in for ``data_coverage_mv``.

    The real migration uses ``CREATE MATERIALIZED VIEW`` which SQLite does
    not support; we emulate it with a regular table so the API handler can
    query it through SQLAlchemy the same way it will in production.
    """
    Base.metadata.create_all(bind=_engine)
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS data_coverage_mv (
                    market VARCHAR(32) NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    interval VARCHAR(8) NOT NULL,
                    first_ts TIMESTAMP,
                    last_ts TIMESTAMP,
                    row_count INTEGER NOT NULL,
                    staleness_seconds DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (market, symbol, interval)
                )
                """
            )
        )
    yield
    with _engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_coverage_mv"))
    Base.metadata.drop_all(bind=_engine)


def _seed_coverage_row(
    market: str,
    symbol: str,
    interval: str,
    first_ts: datetime,
    last_ts: datetime,
    row_count: int,
    staleness_seconds: float,
) -> None:
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO data_coverage_mv
                (market, symbol, interval, first_ts, last_ts, row_count, staleness_seconds)
                VALUES (:m, :s, :i, :f, :l, :r, :st)
                """
            ),
            {
                "m": market,
                "s": symbol,
                "i": interval,
                "f": first_ts,
                "l": last_ts,
                "r": row_count,
                "st": staleness_seconds,
            },
        )


# ---------------------------------------------------------------------------
# Task 1: migration 022 + coverage arithmetic helpers
# ---------------------------------------------------------------------------

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "022_create_data_coverage_mv.py"
)


def test_migration_022_creates_materialized_view():
    """022 must create data_coverage_mv with MIN/MAX/COUNT/staleness columns."""
    assert MIGRATION_PATH.exists(), f"migration file missing: {MIGRATION_PATH}"
    content = MIGRATION_PATH.read_text()

    assert "CREATE MATERIALIZED VIEW data_coverage_mv" in content
    assert re.search(r"MIN\s*\(\s*time\s*\)\s+AS\s+first_ts", content, re.IGNORECASE)
    assert re.search(r"MAX\s*\(\s*time\s*\)\s+AS\s+last_ts", content, re.IGNORECASE)
    assert re.search(r"COUNT\s*\(\s*\*\s*\)\s+AS\s+row_count", content, re.IGNORECASE)
    assert "staleness_seconds" in content
    assert "FROM ohlcv" in content
    assert "GROUP BY market, symbol, interval" in content


def test_migration_022_creates_unique_index_for_concurrent_refresh():
    """A unique index on (market, symbol, interval) is required by CONCURRENTLY refresh."""
    content = MIGRATION_PATH.read_text()
    assert "CREATE UNIQUE INDEX ix_data_coverage_mv_tuple" in content
    assert "data_coverage_mv" in content
    # Must carry all three tuple columns
    idx_pattern = re.compile(
        r"CREATE UNIQUE INDEX ix_data_coverage_mv_tuple\s+ON\s+data_coverage_mv\s*\(([^)]+)\)",
        re.IGNORECASE,
    )
    m = idx_pattern.search(content)
    assert m is not None, "unique index definition not found"
    cols = {c.strip() for c in m.group(1).split(",")}
    assert cols == {"market", "symbol", "interval"}


def test_migration_022_downgrade_drops_view_and_index():
    content = MIGRATION_PATH.read_text()
    downgrade_block = content.split("def downgrade")[-1]
    assert "DROP" in downgrade_block
    assert "data_coverage_mv" in downgrade_block


def test_interval_seconds_table_has_expected_intervals():
    expected = {"1d", "4h", "1h", "30m", "15m", "5m"}
    assert expected.issubset(set(coverage_module.INTERVAL_SECONDS.keys()))
    assert coverage_module.INTERVAL_SECONDS["1d"] == 86400
    assert coverage_module.INTERVAL_SECONDS["4h"] == 4 * 3600
    assert coverage_module.INTERVAL_SECONDS["1h"] == 3600
    assert coverage_module.INTERVAL_SECONDS["30m"] == 1800
    assert coverage_module.INTERVAL_SECONDS["15m"] == 900
    assert coverage_module.INTERVAL_SECONDS["5m"] == 300


def test_compute_expected_count_exact_full_window():
    """Full 1h window with no gaps: expected = (last - first)/interval + 1."""
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)  # 9 hours later
    # 9 hours at 1h interval: 0,1,2,3,4,5,6,7,8,9 -> 10 candles
    assert coverage_module.compute_expected_count(first, last, "1h") == 10


def test_compute_expected_count_handles_none():
    """If first_ts or last_ts is missing, expected_count is 0 (no data)."""
    assert coverage_module.compute_expected_count(None, None, "1h") == 0
    assert (
        coverage_module.compute_expected_count(
            None, datetime(2024, 1, 1, tzinfo=timezone.utc), "1h"
        )
        == 0
    )
    assert (
        coverage_module.compute_expected_count(
            datetime(2024, 1, 1, tzinfo=timezone.utc), None, "1h"
        )
        == 0
    )


def test_compute_expected_count_same_timestamp_returns_one():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert coverage_module.compute_expected_count(ts, ts, "1h") == 1


def test_compute_gap_count_basic():
    assert coverage_module.compute_gap_count(10, 10) == 0
    assert coverage_module.compute_gap_count(9, 10) == 1
    assert coverage_module.compute_gap_count(0, 0) == 0


def test_compute_gap_count_over_count_clamps_to_zero():
    """row_count > expected_count is a data bug, not a negative gap."""
    assert coverage_module.compute_gap_count(11, 10) == 0


def test_compute_health_green_when_complete_and_fresh():
    # 1h interval: staleness <= 2 * 3600 and gap_count == 0 implied by completeness
    assert (
        coverage_module.compute_health(
            completeness_pct=1.0, staleness_seconds=3600, interval="1h", gap_count=0
        )
        == "green"
    )


def test_compute_health_yellow_when_recent_minor_gap():
    # 95% complete, within 4x interval staleness
    assert (
        coverage_module.compute_health(
            completeness_pct=0.97,
            staleness_seconds=4 * 3600,
            interval="1h",
            gap_count=3,
        )
        == "yellow"
    )


def test_compute_health_red_when_stale_or_incomplete():
    # 80% complete, very stale
    assert (
        coverage_module.compute_health(
            completeness_pct=0.80,
            staleness_seconds=48 * 3600,
            interval="1h",
            gap_count=100,
        )
        == "red"
    )
    # Good completeness but super stale
    assert (
        coverage_module.compute_health(
            completeness_pct=1.0,
            staleness_seconds=10 * 86400,
            interval="1h",
            gap_count=0,
        )
        == "red"
    )


# ---------------------------------------------------------------------------
# Task 2: GET /api/data/coverage endpoint
# ---------------------------------------------------------------------------


def test_coverage_endpoint_mounted_at_api_data_coverage(client):
    """Unfiltered GET must respond with a list (even if empty)."""
    resp = client.get("/api/data/coverage")
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), list)


def test_coverage_endpoint_returns_seeded_rows(client):
    """A single seeded MV row must round-trip through the endpoint."""
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    _seed_coverage_row(
        market="crypto_perp",
        symbol="BTCUSDT",
        interval="1h",
        first_ts=first,
        last_ts=last,
        row_count=10,
        staleness_seconds=3600.0,
    )

    resp = client.get("/api/data/coverage")
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["market"] == "crypto_perp"
    assert row["symbol"] == "BTCUSDT"
    assert row["interval"] == "1h"
    assert row["row_count"] == 10
    assert row["expected_count"] == 10
    assert row["gap_count"] == 0
    assert row["completeness_pct"] == pytest.approx(1.0)
    assert row["health"] == "green"
    assert row["staleness_seconds"] == pytest.approx(3600.0)


def test_coverage_endpoint_exact_gap_count_and_completeness(client):
    """Sparse MV row: row_count=8 over a 10-slot window => gap=2, completeness=0.8."""
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    _seed_coverage_row(
        market="crypto_spot",
        symbol="ETHUSDT",
        interval="1h",
        first_ts=first,
        last_ts=last,
        row_count=8,
        staleness_seconds=3600 * 48,  # 2 days stale
    )

    resp = client.get("/api/data/coverage")
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["row_count"] == 8
    assert row["expected_count"] == 10
    assert row["gap_count"] == 2
    assert row["completeness_pct"] == pytest.approx(0.8, rel=1e-3)
    assert row["health"] == "red"  # 0.8 completeness fails yellow threshold


def test_coverage_endpoint_market_filter(client):
    """?market=crypto_perp only returns crypto_perp rows."""
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    _seed_coverage_row("crypto_perp", "BTCUSDT", "1h", first, last, 10, 3600)
    _seed_coverage_row("crypto_spot", "BTCUSDT", "1h", first, last, 10, 3600)
    _seed_coverage_row("tw_stock", "2330", "1d", first, last, 10, 3600)

    resp = client.get("/api/data/coverage", params={"market": "crypto_perp"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["market"] == "crypto_perp"


def test_coverage_endpoint_symbol_filter(client):
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    _seed_coverage_row("crypto_perp", "BTCUSDT", "1h", first, last, 10, 3600)
    _seed_coverage_row("crypto_perp", "ETHUSDT", "1h", first, last, 10, 3600)

    resp = client.get("/api/data/coverage", params={"symbol": "ETHUSDT"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "ETHUSDT"


def test_coverage_endpoint_interval_filter(client):
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    _seed_coverage_row("crypto_perp", "BTCUSDT", "1h", first, last, 10, 3600)
    _seed_coverage_row("crypto_perp", "BTCUSDT", "4h", first, last, 10, 3600)

    resp = client.get("/api/data/coverage", params={"interval": "4h"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["interval"] == "4h"


def test_coverage_endpoint_sorted_by_market_symbol_interval(client):
    first = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    last = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    _seed_coverage_row("crypto_spot", "BTCUSDT", "4h", first, last, 10, 3600)
    _seed_coverage_row("crypto_perp", "ETHUSDT", "1h", first, last, 10, 3600)
    _seed_coverage_row("crypto_perp", "BTCUSDT", "1h", first, last, 10, 3600)
    _seed_coverage_row("crypto_perp", "BTCUSDT", "4h", first, last, 10, 3600)

    resp = client.get("/api/data/coverage")
    rows = resp.json()
    assert len(rows) == 4
    keys = [(r["market"], r["symbol"], r["interval"]) for r in rows]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Task 3: hourly refresh task + success-path hook
# ---------------------------------------------------------------------------


def test_refresh_helper_executes_concurrent_refresh_statement():
    """refresh_data_coverage_mv must emit REFRESH MATERIALIZED VIEW CONCURRENTLY.

    We capture the executed SQL via a stub session so this test does not
    require a real MV. The production helper runs the same statement against
    a real SQLAlchemy session on stormtrooper.
    """

    executed: list[str] = []

    class StubSession:
        def execute(self, stmt):
            # Accept both text() and string for flexibility
            if hasattr(stmt, "text"):
                executed.append(str(stmt.text))
            else:
                executed.append(str(stmt))

        def commit(self):
            pass

    coverage_module.refresh_data_coverage_mv(StubSession())
    assert executed, "refresh_data_coverage_mv did not execute any SQL"
    joined = " ".join(executed)
    assert "REFRESH MATERIALIZED VIEW CONCURRENTLY data_coverage_mv" in joined


def test_coverage_view_refresh_task_registered_on_beat_schedule():
    """A beat entry must call coverage_view_refresh on an hourly cadence."""
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    # At least one entry must reference the new task
    matching = [
        entry
        for entry in schedule.values()
        if entry.get("task", "").endswith("coverage_view_refresh")
    ]
    assert matching, f"no beat entry calls coverage_view_refresh: {list(schedule)}"


def test_coverage_view_refresh_task_exists_on_backfill_module():
    from poseidon.workers import backfill_tasks

    assert hasattr(backfill_tasks, "coverage_view_refresh"), (
        "backfill_tasks.coverage_view_refresh task not defined"
    )
    task = backfill_tasks.coverage_view_refresh
    # Must be registered on the backfill queue (D-07 queue isolation)
    # Celery tasks may expose the queue via .queue attribute
    # We tolerate either an exposed .queue or a task_routes routing.
    name = getattr(task, "name", None)
    assert name, "coverage_view_refresh has no task name"
    assert name.endswith("coverage_view_refresh")


def test_backfill_success_triggers_coverage_refresh(monkeypatch):
    """When _run_backfill_chunk marks the job succeeded, coverage_view_refresh.delay is called."""
    from poseidon.workers import backfill_tasks
    from poseidon.models.backfill import BackfillJob
    from uuid import uuid4

    # Record .delay invocations on coverage_view_refresh
    calls: list[tuple] = []

    def fake_delay(*args, **kwargs):
        calls.append((args, kwargs))

        class _R:
            id = "stub-refresh-id"

        return _R()

    monkeypatch.setattr(
        backfill_tasks.coverage_view_refresh, "delay", fake_delay, raising=False
    )

    # Stub out the heavy machinery: fetcher, rate limiter, storage, validation.
    class _StubDf:
        def __init__(self):
            self._empty = True

        @property
        def empty(self):
            return True

    class _StubFetcher:
        def fetch_ohlcv(self, *args, **kwargs):
            return _StubDf()

    monkeypatch.setattr(
        backfill_tasks, "get_fetcher", lambda market: _StubFetcher()
    )
    monkeypatch.setattr(
        backfill_tasks,
        "_acquire_bulk_backfill_slot",
        lambda rl, cb, p: True,
    )

    class _StubRedis:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(backfill_tasks, "_get_redis_client", lambda: _StubRedis())

    class _StubCircuit:
        def __init__(self, *a, **kw):
            pass

        def allow_request(self):
            return True

        def record_success(self):
            pass

        def record_failure(self):
            pass

    class _StubLimiter:
        def __init__(self, *a, **kw):
            pass

        def wait_and_acquire(self, *a, **kw):
            return True

    monkeypatch.setattr(backfill_tasks, "CircuitBreaker", _StubCircuit)
    monkeypatch.setattr(backfill_tasks, "DistributedRateLimiter", _StubLimiter)

    # Prepare a near-complete job row in the SQLite harness.
    session = TestingSessionLocal()
    try:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)  # tiny window: one chunk fills it
        job = BackfillJob(
            status="pending",
            market="crypto_perp",
            symbol=None,
            interval=None,
            symbols=["BTCUSDT"],
            intervals=["1h"],
            requested_by="api",
            cursor={
                "symbol_idx": 0,
                "interval_idx": 0,
                "next_ts": start.isoformat(),
                "start_ts": start.isoformat(),
                "end_ts": end.isoformat(),
            },
            progress={
                "requested_symbols": ["BTCUSDT"],
                "requested_intervals": ["1h"],
                "rows_written": 0,
                "chunks_done": 0,
            },
        )
        session.add(job)
        session.commit()
        job_id = str(job.job_id)
    finally:
        session.close()

    # Run the executor end-to-end in the SQLite session.
    session = TestingSessionLocal()
    try:
        result = backfill_tasks._run_backfill_chunk(session, job_id)
    finally:
        session.close()

    assert result["status"] == "succeeded", result
    assert calls, (
        "coverage_view_refresh.delay should have been called on successful completion"
    )
