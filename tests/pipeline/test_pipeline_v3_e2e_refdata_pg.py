"""M1 wiring end to end on the real refdata: the acceptance spec created through the API, driven by the dispatcher.

Runs only with ``PV2_PAPER_REFDATA`` = a refdata root holding ``twse_market_calendar`` (a read-only copy of
the shared store after the daily job) and the throwaway PostgreSQL. The run is created at the real
time (data window 2001-01-02 .. today, exclusive); the dispatcher clock is 2026-10-24 13:30Z, so the
paper anchors on the first trading day after it, after the data.

* ``natural`` -- the run as the runner would drive it: every stage until the first verdict that is not
  a pass. Whatever the gates decide is the result (reported, not asserted).
* ``observed`` -- the same spec in a second family with :class:`~tests.pipeline.v3_wiring_support.Observe`
  around every stage that can fail: a fail is recorded on the attempt (``observed_verdict``) and the run
  continues, so every stage up to ``shadow_paper`` (waiting) runs on real data with real inputs.

``POSEIDON_V3_E2E_OUT`` keeps both traces as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api import pipelines
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.pipeline.plan import V3_DAILY_STAGE_PLAN, V3_GATE_STAGES
from poseidon.pipeline.refdata.datasets import MARKET_CALENDAR
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.v3_wiring_support import Observe, trace, v3_registry

ROOT = os.environ.get("PV2_PAPER_REFDATA")
OUT = os.environ.get("POSEIDON_V3_E2E_OUT")
NOW = datetime(2026, 10, 24, 13, 30, tzinfo=UTC)
pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(not ROOT or not RefdataStore(ROOT).exists(MARKET_CALENDAR), reason="PV2_PAPER_REFDATA not set"),
]
KEY = "e2e-researcher-secret"
OBSERVED = ("backtest", "optimize", *V3_GATE_STAGES)


@pytest.fixture
def api(pipeline_sessions, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "pipeline_refdata_root", ROOT)
    monkeypatch.setattr(
        settings,
        "api_principals_json",
        json.dumps(
            {
                hashlib.sha256(KEY.encode()).hexdigest(): {
                    "actor_id": "strategy-bridge",
                    "roles": ["researcher"],
                    "account_scopes": [],
                }
            }
        ),
    )
    clock = Clock(NOW)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", clock, registry=v3_registry())
    app = FastAPI()
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")

    def db():
        session = pipeline_sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[pipelines.get_pipeline_enqueuer] = lambda: harness.queue.append
    client = TestClient(app)
    client.harness = harness
    return client


def create(api, idea: str) -> uuid.UUID:
    response = api.post(
        "/api/v1/pipelines",
        json={"idea_text": idea, "source": "api", "spec_json": tmf_dynamic_leverage_3x()},
        headers={"X-API-Key": KEY, "Idempotency-Key": uuid.uuid4().hex},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert (body["market"], body["symbol"], body["current_stage"]) == ("tw_futures", "TMF", "spec_validate")
    assert body["config_json"]["version"] == 6 and body["config_json"]["strategy_class"] == "daily_equity"
    return uuid.UUID(body["id"])


def keep(name: str, harness, run_id) -> dict:
    run = harness.run(run_id)
    record = {
        "run": {"status": run.status, "current_stage": run.current_stage, "reason": run.reason},
        "config": {key: run.config_json[key] for key in ("data", "refdata", "strategy_class", "registry_version")},
        "trace": trace(harness, run_id),
    }
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


@pytest.mark.usefixtures("pipeline_engine")
def test_acceptance_spec_end_to_end_on_real_refdata(api):
    harness = api.harness

    natural = create(api, "3x TMF, 200-day MA band to 0.5x, vol cap (acceptance, natural)")
    harness.drain(limit=40)
    record = keep("natural", harness, natural)
    rows = record["trace"]
    assert rows[0]["stage"] == "spec_validate" and rows[0]["verdict"] == "pass", rows
    assert all(row["verdict"] != "error" and row["summary"] for row in rows), rows  # wiring: no stage errored
    last = rows[-1]
    assert last["verdict"] == "fail" or (last["stage"], last["status"]) == ("shadow_paper", "waiting"), last

    for name in OBSERVED:  # second family, every failing verdict recorded and passed through
        harness.dispatcher.registry[name] = Observe(harness.dispatcher.registry[name])
    observed = create(api, "3x TMF, 200-day MA band to 0.5x, vol cap (acceptance, observed)")
    harness.drain(limit=40)
    record = keep("observed", harness, observed)
    rows = record["trace"]
    run = harness.run(observed)
    assert (run.status, run.current_stage) == ("waiting", "shadow_paper"), rows
    assert [row["stage"] for row in rows] == list(V3_DAILY_STAGE_PLAN[:-1])
    assert all(row["summary"] and row["verdict"] != "error" for row in rows), rows
    paper = rows[-1]
    assert paper["status"] == "waiting" and "paper starts" in paper["summary"]
