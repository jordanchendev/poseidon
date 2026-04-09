"""Tests for the Phase 40 freshness watchdog (plan 40-03 Tasks 1 + 2).

Task 1 tests: pure helper functions in poseidon.data.freshness
  - compute_freshness_status (ok / violation / unknown)
  - lookup_sla (hit + miss)
  - evaluate_freshness (aggregate, SLA filter, worst-tuple collapse)
  - ping_monitor (no-op on empty URL, success path, fail path, swallow exceptions)

Task 2 tests: ingest_freshness_watchdog Celery task + beat schedule
  - Task return dict shape
  - Uptime Kuma ping on all-ok / any-violation / empty URL
  - Exception swallow
  - Beat schedule entry

Uses the same in-memory SQLite harness + @compiles shim pattern established
in test_data_coverage_api.py (Phase 39) and test_data_gap_audit.py (Phase 40-02).

See .planning/phases/40-data-health-observability/40-CONTEXT.md D-10..D-16.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
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
from poseidon.models.ingest_state import IngestState  # noqa: E402

# --- Test DB wiring ---

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=_engine
)


@pytest.fixture(autouse=True)
def _setup_db():
    """Create ORM tables fresh for every test."""
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# Shared SLA dict matching the locked D-11 defaults for crypto_perp:4h
TEST_SLA = {"crypto_perp:4h": 18000}
NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ==========================================================================
# Task 1 Tests: pure helpers in poseidon.data.freshness
# ==========================================================================


class TestComputeFreshnessStatus:
    """Tests 1-3: pure status classification."""

    def test_ok_within_sla(self):
        from poseidon.data.freshness import compute_freshness_status

        status, observed = compute_freshness_status(
            now=NOW,
            last_successful_ts=NOW - timedelta(seconds=7200),
            expected_lag_seconds=18000,
        )
        assert status == "ok"
        assert abs(observed - 7200.0) < 1.0

    def test_violation_over_sla(self):
        from poseidon.data.freshness import compute_freshness_status

        status, observed = compute_freshness_status(
            now=NOW,
            last_successful_ts=NOW - timedelta(seconds=20000),
            expected_lag_seconds=18000,
        )
        assert status == "violation"
        assert abs(observed - 20000.0) < 1.0

    def test_unknown_when_none(self):
        from poseidon.data.freshness import compute_freshness_status

        status, observed = compute_freshness_status(
            now=NOW,
            last_successful_ts=None,
            expected_lag_seconds=18000,
        )
        assert status == "unknown"
        assert observed == 0.0


class TestLookupSla:
    """Test 4: SLA dict lookup."""

    def test_hit(self):
        from poseidon.data.freshness import lookup_sla

        assert (
            lookup_sla(
                {"crypto_perp:4h": 18000}, market="crypto_perp", interval="4h"
            )
            == 18000
        )

    def test_miss(self):
        from poseidon.data.freshness import lookup_sla

        assert (
            lookup_sla(
                {"crypto_perp:4h": 18000},
                market="unknown_market",
                interval="1d",
            )
            is None
        )


class TestEvaluateFreshness:
    """Test 5: evaluate_freshness with IngestState rows."""

    def test_evaluate_freshness_aggregates_worst_and_filters_sla(self, db):
        from poseidon.data.freshness import evaluate_freshness

        # Seed three rows
        # 1) crypto_perp BTCUSDT 4h — ok (7200s old)
        db.add(
            IngestState(
                symbol="BTCUSDT",
                market="crypto_perp",
                interval="4h",
                last_successful_ts=NOW - timedelta(seconds=7200),
                first_backfill_done=True,
            )
        )
        # 2) crypto_perp ETHUSDT 4h — violation (20000s old) — WORST for (crypto_perp, 4h)
        db.add(
            IngestState(
                symbol="ETHUSDT",
                market="crypto_perp",
                interval="4h",
                last_successful_ts=NOW - timedelta(seconds=20000),
                first_backfill_done=True,
            )
        )
        # 3) unknown_market FOO 1d — NOT in SLA → silently skipped
        db.add(
            IngestState(
                symbol="FOO",
                market="unknown_market",
                interval="1d",
                last_successful_ts=NOW - timedelta(seconds=100),
                first_backfill_done=True,
            )
        )
        db.commit()

        records = evaluate_freshness(db, now=NOW, sla_dict=TEST_SLA)

        # Only one result: (crypto_perp, 4h) — the WORST (ETHUSDT)
        assert len(records) == 1
        r = records[0]
        assert r.market == "crypto_perp"
        assert r.interval == "4h"
        assert r.status == "violation"
        assert r.observed_lag_seconds >= 19999
        assert r.expected_lag_seconds == 18000


class TestPingMonitor:
    """Tests 6-9: fire-and-forget Uptime Kuma ping."""

    def test_noop_on_empty_url(self):
        from poseidon.data.freshness import ping_monitor

        assert ping_monitor("", success=True) is False

    def test_success_ping(self, monkeypatch):
        from poseidon.data import freshness as freshness_mod

        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params, timeout))

        monkeypatch.setattr("requests.get", fake_get)
        import importlib

        importlib.reload(freshness_mod)

        result = freshness_mod.ping_monitor(
            "http://kuma.local/api/push/tok", success=True
        )
        assert result is True
        assert len(calls) == 1
        assert calls[0][0] == "http://kuma.local/api/push/tok"
        assert calls[0][1]["status"] == "up"
        assert calls[0][2] == 3.0

    def test_fail_ping(self, monkeypatch):
        from poseidon.data import freshness as freshness_mod

        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params, timeout))

        monkeypatch.setattr("requests.get", fake_get)
        import importlib

        importlib.reload(freshness_mod)

        result = freshness_mod.ping_monitor(
            "http://kuma.local/api/push/tok", success=False, message="stale"
        )
        assert result is True
        assert len(calls) == 1
        assert calls[0][0] == "http://kuma.local/api/push/tok"
        assert calls[0][1]["status"] == "down"
        assert calls[0][1]["msg"] == "stale"
        assert calls[0][2] == 3.0

    def test_swallows_exceptions(self, monkeypatch, caplog):
        import requests as requests_mod
        from poseidon.data import freshness as freshness_mod

        def fake_get(url, params=None, timeout=None):
            raise requests_mod.ConnectionError("network down")

        monkeypatch.setattr("requests.get", fake_get)
        import importlib

        importlib.reload(freshness_mod)

        with caplog.at_level(logging.WARNING):
            result = freshness_mod.ping_monitor(
                "http://kuma.local/api/push/tok", success=True
            )
        assert result is False
        assert any(
            "ping_monitor failed" in r.message for r in caplog.records
        )


# ==========================================================================
# Task 2 Tests: ingest_freshness_watchdog Celery task + beat schedule
# ==========================================================================


class TestIngestFreshnessWatchdogTask:
    """Tests for the Celery task wrapper."""

    def _seed_ok_rows(self, db):
        """Seed rows that are all within SLA."""
        real_now = datetime.now(timezone.utc)
        db.add(
            IngestState(
                symbol="BTCUSDT",
                market="crypto_perp",
                interval="4h",
                last_successful_ts=real_now - timedelta(seconds=3600),
                first_backfill_done=True,
            )
        )
        db.commit()

    def _seed_violation_rows(self, db):
        """Seed rows with at least one SLA violation."""
        real_now = datetime.now(timezone.utc)
        db.add(
            IngestState(
                symbol="BTCUSDT",
                market="crypto_perp",
                interval="4h",
                last_successful_ts=real_now - timedelta(seconds=20000),
                first_backfill_done=True,
            )
        )
        db.commit()

    def test_all_ok_pings_success(self, monkeypatch, db):
        """With all tuples within SLA, watchdog pings success."""
        self._seed_ok_rows(db)

        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params, timeout))

        monkeypatch.setattr("requests.get", fake_get)

        # Monkeypatch SessionLocal to use our test session
        monkeypatch.setattr(
            "poseidon.workers.cpu_tasks.SessionLocal",
            lambda: TestingSessionLocal(),
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.freshness_sla",
            TEST_SLA,
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.uptime_kuma_push_url",
            "http://kuma.local/api/push/test-token",
        )

        from poseidon.workers.cpu_tasks import ingest_freshness_watchdog

        result = ingest_freshness_watchdog()

        assert result["checked"] >= 1
        assert result["violations"] == 0
        assert result["monitor_pinged"] == "success"
        assert len(calls) == 1
        assert calls[0][0] == "http://kuma.local/api/push/test-token"
        assert calls[0][1]["status"] == "up"

    def test_violation_pings_fail(self, monkeypatch, db):
        """With any violation, watchdog pings status=down."""
        self._seed_violation_rows(db)

        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params, timeout))

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr(
            "poseidon.workers.cpu_tasks.SessionLocal",
            lambda: TestingSessionLocal(),
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.freshness_sla",
            TEST_SLA,
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.uptime_kuma_push_url",
            "http://kuma.local/api/push/test-token",
        )

        from poseidon.workers.cpu_tasks import ingest_freshness_watchdog

        result = ingest_freshness_watchdog()

        assert result["violations"] >= 1
        assert result["monitor_pinged"] == "fail"
        assert len(calls) == 1
        assert calls[0][0] == "http://kuma.local/api/push/test-token"
        assert calls[0][1]["status"] == "down"

    def test_empty_url_skips_ping(self, monkeypatch, db):
        """With empty UPTIME_KUMA_PUSH_URL, no HTTP call is made."""
        self._seed_ok_rows(db)

        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params, timeout))

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr(
            "poseidon.workers.cpu_tasks.SessionLocal",
            lambda: TestingSessionLocal(),
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.freshness_sla",
            TEST_SLA,
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.uptime_kuma_push_url",
            "",
        )

        from poseidon.workers.cpu_tasks import ingest_freshness_watchdog

        result = ingest_freshness_watchdog()

        assert result["monitor_pinged"] == "skipped"
        assert len(calls) == 0  # no HTTP call

    def test_hcio_exception_does_not_raise(self, monkeypatch, db):
        """Uptime Kuma exception is swallowed — task still returns its summary."""
        self._seed_ok_rows(db)

        import requests as requests_mod

        def fake_get(url, params=None, timeout=None):
            raise requests_mod.ConnectionError("boom")

        monkeypatch.setattr("requests.get", fake_get)
        monkeypatch.setattr(
            "poseidon.workers.cpu_tasks.SessionLocal",
            lambda: TestingSessionLocal(),
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.freshness_sla",
            TEST_SLA,
        )
        monkeypatch.setattr(
            "poseidon.core.config.settings.uptime_kuma_push_url",
            "http://kuma.local/api/push/test-token",
        )

        from poseidon.workers.cpu_tasks import ingest_freshness_watchdog

        # Must NOT raise — task returns summary dict
        result = ingest_freshness_watchdog()
        assert "checked" in result
        assert "monitor_pinged" in result
        # The ping returned False but the task still reported success intent
        assert result["monitor_pinged"] == "success"


class TestBeatScheduleEntry:
    """Test 6: beat schedule registration."""

    def test_ingest_freshness_watchdog_registered(self):
        from celery.schedules import crontab

        from poseidon.workers.celery_app import celery_app

        schedule = celery_app.conf.beat_schedule
        assert "ingest-freshness-watchdog" in schedule

        entry = schedule["ingest-freshness-watchdog"]
        assert (
            entry["task"]
            == "poseidon.workers.cpu_tasks.ingest_freshness_watchdog"
        )
        assert str(entry["schedule"]) == str(crontab(minute="*/15"))
