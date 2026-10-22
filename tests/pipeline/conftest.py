"""Fixtures for the strategy-pipeline tests.

PostgreSQL-backed tests run only when ``POSEIDON_PIPELINE_TEST_DATABASE_URL``
names a throwaway database; the schema comes from the real ``pipeline``
Alembic stream (``p001``), so triggers and partial indexes are the deployed ones.
"""

from __future__ import annotations

import copy
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

PIPELINE_TEST_DB_ENV = "POSEIDON_PIPELINE_TEST_DATABASE_URL"
REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1_PLAN = ["spec_generate", "spec_validate", "backtest"]

TX_SPEC = {
    "name": "tx-rsi-reversion",
    "description": "hand-written v1 TX rule spec for the Step-1 skeleton",
    "symbol": "TX",
    "market": "tw_futures",
    "interval": "5m",
    "rules": [
        {
            "condition": {
                "all": [
                    {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                    {"type": "price_crosses", "indicator": "sma", "params": {"period": 20}, "direction": "up"},
                ]
            },
            "action": "long",
            "quantity_pct": 0.5,
        },
        {
            "condition": {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
            "action": "close",
        },
    ],
}


def tx_spec(**overrides) -> dict:
    spec = copy.deepcopy(TX_SPEC)
    spec.update(overrides)
    return spec


class Clock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 10, 22, 2, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> datetime:
        self.now += timedelta(**delta)
        return self.now


def pipeline_alembic_config(url: str):
    """The ``[pipeline]`` stream without ini logging (fileConfig would reset test loggers)."""

    from alembic.config import Config

    config = Config(ini_section="pipeline")
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("version_locations", str(REPO_ROOT / "alembic" / "pipeline_versions"))
    config.set_main_option("version_table", "alembic_version_pipeline")
    config.set_main_option("target_metadata", "poseidon.pipeline.models:PipelineBase")
    config.set_main_option("sqlalchemy.url", url)
    return config


@pytest.fixture(scope="session")
def pipeline_engine():
    url = os.environ.get(PIPELINE_TEST_DB_ENV)
    if not url:
        pytest.skip(f"{PIPELINE_TEST_DB_ENV} is not set (needs a throwaway PostgreSQL)")
    from alembic import command
    from poseidon.core.config import settings

    original = settings.database_url
    settings.database_url = url  # env.py always migrates settings.database_url
    try:
        command.upgrade(pipeline_alembic_config(url), "head")
    finally:
        settings.database_url = original
    engine = create_engine(url, future=True, pool_size=10, max_overflow=10)
    yield engine
    engine.dispose()


@pytest.fixture
def pipeline_sessions(pipeline_engine):
    with pipeline_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE pipeline_events, pipeline_trials, stage_runs, pipeline_runs RESTART IDENTITY CASCADE")
        )
    return sessionmaker(bind=pipeline_engine, autoflush=False, future=True)


@pytest.fixture
def clock():
    return Clock()


class Harness:
    """Direct-call stand-in for Celery: ``enqueue`` appends, ``drain`` advances."""

    def __init__(self, sessions, artifacts_root, clock, registry=None, heartbeat_seconds=0.05):
        from poseidon.pipeline.dispatcher import PipelineDispatcher

        self.sessions = sessions
        self.clock = clock
        self.queue: list[uuid.UUID] = []
        self.dispatcher = PipelineDispatcher(
            sessions,
            registry=registry,
            enqueue=self.queue.append,
            artifacts_root=artifacts_root,
            heartbeat_seconds=heartbeat_seconds,
            clock=clock,
            code_version="test-sha",
        )

    def create(self, *, spec=TX_SPEC, stage_plan=STEP1_PLAN, key=None, actor="human:researcher", **body):
        from poseidon.pipeline.service import PipelineService

        payload = {"idea_text": "TX RSI reversion", "source": "api", "spec_json": spec, "stage_plan": stage_plan}
        payload.update(body)
        with self.sessions() as session:
            response, _ = PipelineService(session, registry=self.dispatcher.registry, clock=self.clock).create(
                payload, actor_id=actor, idempotency_key=key or uuid.uuid4().hex
            )
            session.commit()
        run_id = uuid.UUID(response["id"])
        self.queue.append(run_id)
        return run_id

    def drain(self, limit: int = 50) -> list[dict]:
        results = []
        while self.queue:
            if len(results) >= limit:
                raise AssertionError("dispatcher did not settle")
            results.append(self.dispatcher.advance(self.queue.pop(0)))
        return results

    def run(self, run_id):
        from poseidon.pipeline.models import PipelineRun

        with self.sessions() as session:
            run = session.get(PipelineRun, run_id)
            session.expunge(run)
            return run

    def attempts(self, run_id):
        from poseidon.pipeline.models import StageRun

        with self.sessions() as session:
            rows = session.query(StageRun).filter_by(run_id=run_id).order_by(StageRun.seq, StageRun.attempt).all()
            for row in rows:
                session.expunge(row)
            return rows

    def events(self, run_id):
        from poseidon.pipeline.models import PipelineEvent

        with self.sessions() as session:
            rows = (
                session.query(PipelineEvent)
                .filter_by(run_id=run_id)
                .order_by(PipelineEvent.created_at, PipelineEvent.id)
                .all()
            )
            return [row.event_type for row in rows]


@pytest.fixture
def harness(pipeline_sessions, tmp_path, clock):
    return Harness(pipeline_sessions, tmp_path / "artifacts", clock)
