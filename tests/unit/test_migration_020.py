"""Migration 020 schema assertions (Phase 38 plan 38-01 Task 2).

Runs against the live test database (assumes ``alembic upgrade head`` has been
applied). Verifies:
- ingest_state table shape (DATA-FOUND-02)
- backfill_jobs table shape (DATA-FOUND-04)
- backfill_progress table has been dropped (D-10)
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect

from poseidon.core.config import settings

pytestmark = pytest.mark.phase38


@pytest.fixture(scope="module")
def inspector():
    engine = create_engine(settings.database_url)
    insp = inspect(engine)
    yield insp
    engine.dispose()


def _col_map(inspector, table: str) -> dict[str, dict]:
    return {c["name"]: c for c in inspector.get_columns(table)}


def test_ingest_state_shape(inspector):
    assert "ingest_state" in inspector.get_table_names(), (
        "ingest_state table missing — migration 020 not applied"
    )

    cols = _col_map(inspector, "ingest_state")
    expected = {
        "symbol",
        "market",
        "interval",
        "last_successful_ts",
        "last_attempt_ts",
        "last_error",
        "first_backfill_done",
        "created_at",
        "updated_at",
    }
    assert expected.issubset(cols.keys()), f"missing cols: {expected - cols.keys()}"

    # Primary key must be composite (symbol, market, interval)
    pk = inspector.get_pk_constraint("ingest_state")
    assert set(pk["constrained_columns"]) == {"symbol", "market", "interval"}, (
        f"ingest_state PK should be (symbol, market, interval), got {pk}"
    )

    # last_successful_ts nullable (cursor may be unset on first deploy)
    assert cols["last_successful_ts"]["nullable"] is True
    # first_backfill_done is non-null with default false
    assert cols["first_backfill_done"]["nullable"] is False


def test_backfill_jobs_shape(inspector):
    assert "backfill_jobs" in inspector.get_table_names(), (
        "backfill_jobs table missing — migration 020 not applied"
    )

    cols = _col_map(inspector, "backfill_jobs")
    expected = {
        "job_id",
        "status",
        "market",
        "symbol",
        "interval",
        "cursor",
        "progress",
        "error",
        "created_at",
        "updated_at",
    }
    assert expected.issubset(cols.keys()), f"missing cols: {expected - cols.keys()}"

    pk = inspector.get_pk_constraint("backfill_jobs")
    assert pk["constrained_columns"] == ["job_id"], (
        f"backfill_jobs PK should be job_id, got {pk}"
    )

    # status CHECK constraint must exist
    checks = inspector.get_check_constraints("backfill_jobs")
    check_names = {c["name"] for c in checks}
    assert "ck_backfill_jobs_status" in check_names, (
        f"status CHECK constraint missing; got {check_names}"
    )

    # status index for dispatcher scans
    index_names = {ix["name"] for ix in inspector.get_indexes("backfill_jobs")}
    assert "ix_backfill_jobs_status" in index_names


def test_backfill_progress_dropped(inspector):
    assert "backfill_progress" not in inspector.get_table_names(), (
        "backfill_progress must be dropped per Phase 38 D-10"
    )
