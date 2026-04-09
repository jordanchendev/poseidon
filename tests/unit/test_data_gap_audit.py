"""Tests for the Phase 40 data-gap-audit pillar (plan 40-02 Task 1).

Covers the locked decisions in
.planning/phases/40-data-health-observability/40-CONTEXT.md (D-04..D-09):

- detect_gaps_for_tuple: SQL LAG(time) window function with 1.5x tolerance
- upsert_gaps: idempotent ON CONFLICT DO NOTHING via unique index
- heal_resolved_gaps: lifecycle transition when gap is fully populated
- data_gap_audit Celery task: iterates data_coverage_mv, writes data_gaps

The unit suite runs against an in-memory SQLite harness. SQLite 3.25+
supports window functions (LAG) which Python ships natively.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


from poseidon.models.base import Base  # noqa: E402

# --- Test DB wiring ---

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(autouse=True)
def _setup_db():
    """Create ORM tables + SQLite stand-ins for ohlcv and data_coverage_mv."""
    Base.metadata.create_all(bind=_engine)
    with _engine.begin() as conn:
        # Stand-in for ohlcv (simplified — only columns the gap detection needs)
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    market VARCHAR(32) NOT NULL,
                    instrument VARCHAR(32) NOT NULL DEFAULT '',
                    interval VARCHAR(8) NOT NULL,
                    open NUMERIC NOT NULL DEFAULT 0,
                    high NUMERIC NOT NULL DEFAULT 0,
                    low NUMERIC NOT NULL DEFAULT 0,
                    close NUMERIC NOT NULL DEFAULT 0,
                    volume NUMERIC NOT NULL DEFAULT 0,
                    PRIMARY KEY (time, symbol, market, interval)
                )
                """
            )
        )
        # Stand-in for data_coverage_mv
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS data_coverage_mv (
                    market VARCHAR(32) NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    interval VARCHAR(8) NOT NULL,
                    first_ts TIMESTAMP,
                    last_ts TIMESTAMP,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    staleness_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                    PRIMARY KEY (market, symbol, interval)
                )
                """
            )
        )
    yield
    with _engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data_coverage_mv"))
        conn.execute(text("DROP TABLE IF EXISTS ohlcv"))
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _seed_ohlcv(
    conn,
    market: str,
    symbol: str,
    interval: str,
    times: list[datetime],
) -> None:
    """Insert ohlcv rows at the given timestamps."""
    for t in times:
        conn.execute(
            text(
                """
                INSERT INTO ohlcv (time, symbol, market, instrument, interval, open, high, low, close, volume)
                VALUES (:t, :s, :m, '', :i, 100, 110, 90, 105, 1000)
                """
            ),
            {"t": t, "s": symbol, "m": market, "i": interval},
        )


def _seed_coverage_mv(conn, market: str, symbol: str, interval: str) -> None:
    """Insert a data_coverage_mv row for the given tuple."""
    conn.execute(
        text(
            """
            INSERT INTO data_coverage_mv (market, symbol, interval, first_ts, last_ts, row_count, staleness_seconds)
            VALUES (:m, :s, :i, '2024-01-01', '2024-01-02', 10, 0)
            """
        ),
        {"m": market, "s": symbol, "i": interval},
    )


# ---------------------------------------------------------------------------
# Test 1: detect_gaps_for_tuple finds gap when LAG(time) > 1.5x tolerance
# ---------------------------------------------------------------------------


def test_detect_gaps_finds_gap_when_delta_exceeds_threshold(db_session):
    """4h bars at t=0h, 4h, 8h, 20h: the 12h jump from 8h->20h is 3x the
    expected 4h interval, which crosses the 1.5x tolerance => one gap with
    gap_start=8h, gap_end=20h, missing_bars=2."""
    from poseidon.data.gaps import detect_gaps_for_tuple

    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [
        base,
        base + timedelta(hours=4),
        base + timedelta(hours=8),
        base + timedelta(hours=20),  # 12h gap from 8h
    ]
    with _engine.begin() as conn:
        _seed_ohlcv(conn, "crypto_perp", "BTCUSDT", "4h", times)

    gaps = detect_gaps_for_tuple(
        db_session, market="crypto_perp", symbol="BTCUSDT", interval="4h"
    )
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.market == "crypto_perp"
    assert gap.symbol == "BTCUSDT"
    assert gap.interval == "4h"
    assert gap.gap_start == base + timedelta(hours=8)
    assert gap.gap_end == base + timedelta(hours=20)
    assert gap.missing_bars == 2  # floor(12h / 4h) - 1 = 2


# ---------------------------------------------------------------------------
# Test 2: contiguous data returns empty gap list
# ---------------------------------------------------------------------------


def test_detect_gaps_returns_empty_for_contiguous_data(db_session):
    """Rows at every 4h mark => no gaps."""
    from poseidon.data.gaps import detect_gaps_for_tuple

    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(hours=4 * i) for i in range(7)]  # 0,4,8,12,16,20,24
    with _engine.begin() as conn:
        _seed_ohlcv(conn, "crypto_perp", "BTCUSDT", "4h", times)

    gaps = detect_gaps_for_tuple(
        db_session, market="crypto_perp", symbol="BTCUSDT", interval="4h"
    )
    assert gaps == []


# ---------------------------------------------------------------------------
# Test 3: tiny drift below 1.5x tolerance produces no gap
# ---------------------------------------------------------------------------


def test_detect_gaps_tolerates_minor_drift_below_threshold(db_session):
    """4h + 5min drift is still within 1.5x tolerance (6h threshold), no gap."""
    from poseidon.data.gaps import detect_gaps_for_tuple

    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [
        base,
        base + timedelta(hours=4, minutes=5),  # 4h05m: below 6h threshold
    ]
    with _engine.begin() as conn:
        _seed_ohlcv(conn, "crypto_perp", "BTCUSDT", "4h", times)

    gaps = detect_gaps_for_tuple(
        db_session, market="crypto_perp", symbol="BTCUSDT", interval="4h"
    )
    assert gaps == []


# ---------------------------------------------------------------------------
# Test 4: data_gap_audit is idempotent (second run inserts zero new rows)
# ---------------------------------------------------------------------------


def test_data_gap_audit_idempotent_on_rerun(db_session, monkeypatch):
    """Running data_gap_audit twice on the same fixture inserts zero
    additional rows thanks to ON CONFLICT DO NOTHING."""
    from poseidon.data.gaps import detect_gaps_for_tuple, upsert_gaps

    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [base, base + timedelta(hours=4), base + timedelta(hours=20)]
    with _engine.begin() as conn:
        _seed_ohlcv(conn, "crypto_perp", "BTCUSDT", "4h", times)
        _seed_coverage_mv(conn, "crypto_perp", "BTCUSDT", "4h")

    # First run: detect and upsert
    gaps = detect_gaps_for_tuple(
        db_session, market="crypto_perp", symbol="BTCUSDT", interval="4h"
    )
    assert len(gaps) > 0
    inserted_first = upsert_gaps(db_session, gaps)
    assert inserted_first > 0

    # Second run on same data: should insert zero new rows
    gaps2 = detect_gaps_for_tuple(
        db_session, market="crypto_perp", symbol="BTCUSDT", interval="4h"
    )
    inserted_second = upsert_gaps(db_session, gaps2)
    assert inserted_second == 0


# ---------------------------------------------------------------------------
# Test 5: heal_resolved_gaps sets healed_at when bars are now present
# ---------------------------------------------------------------------------


def test_heal_resolved_gaps_sets_healed_at(db_session):
    """After filling the previously-missing bars, heal_resolved_gaps stamps
    healed_at on the gap row."""
    from poseidon.data.gaps import (
        detect_gaps_for_tuple,
        heal_resolved_gaps,
        upsert_gaps,
    )

    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Initially: bars at 0h, 4h, 20h (gap from 4h to 20h)
    initial_times = [base, base + timedelta(hours=4), base + timedelta(hours=20)]
    with _engine.begin() as conn:
        _seed_ohlcv(conn, "crypto_perp", "BTCUSDT", "4h", initial_times)

    gaps = detect_gaps_for_tuple(
        db_session, market="crypto_perp", symbol="BTCUSDT", interval="4h"
    )
    assert len(gaps) == 1
    upsert_gaps(db_session, gaps)

    # Now fill the gap: add bars at 8h, 12h, 16h (the 3 missing bars between 4h and 20h)
    fill_times = [
        base + timedelta(hours=8),
        base + timedelta(hours=12),
        base + timedelta(hours=16),
    ]
    with _engine.begin() as conn:
        _seed_ohlcv(conn, "crypto_perp", "BTCUSDT", "4h", fill_times)

    healed = heal_resolved_gaps(db_session)
    assert healed == 1

    # Verify the gap row now has healed_at set
    from poseidon.models.data_gap import DataGap

    gap_row = db_session.query(DataGap).first()
    assert gap_row is not None
    assert gap_row.healed_at is not None


# ---------------------------------------------------------------------------
# Test 6: data_gap_audit is registered as a Celery task
# ---------------------------------------------------------------------------


def test_data_gap_audit_celery_task_registered():
    """data_gap_audit must be registered as poseidon.workers.cpu_tasks.data_gap_audit."""
    from poseidon.workers.celery_app import celery_app

    task_names = list(celery_app.tasks.keys())
    assert "poseidon.workers.cpu_tasks.data_gap_audit" in task_names
