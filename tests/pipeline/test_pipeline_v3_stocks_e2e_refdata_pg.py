"""Step 8 (M3) acceptance (2) on the real refdata: the hand-written 00631L / 0050 rotation spec created
through ``PipelineService`` and driven by the dispatcher through every v3 stage on PostgreSQL.

Runs only with ``PV2_PAPER_REFDATA`` = a refdata root holding ``thalassa_tw_stock_daily`` (a read-only copy
of the shared store after the stock dataset build) and the throwaway PostgreSQL. The run is created at
2026-10-24 13:30Z (data window from 00631L's first day, end exclusive 2026-10-24).

* ``natural`` -- as the runner drives it: every stage until the first verdict that is not a pass; the
  verdict is the result (reported, not asserted).
* ``observed`` -- :class:`Observe` around every gate so every stage up to ``shadow_paper`` (waiting for
  its first trading day) runs on the real stock data.

``POSEIDON_V3_STOCKS_OUT`` keeps both traces as JSON.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from poseidon.pipeline.plan import V3_DAILY_STAGE_PLAN, V3_GATE_STAGES
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050
from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.daily_paper_support import AsOf
from tests.pipeline.v3_wiring_support import Observe, create_v3, trace, v3_registry

ROOT = os.environ.get("PV2_PAPER_REFDATA")
OUT = os.environ.get("POSEIDON_V3_STOCKS_OUT")
NOW = datetime(2026, 10, 24, 13, 30, tzinfo=UTC)
STORE_CUTOFF = "2026-10-22"  # the store as it was on the morning of NOW, whenever the copy was taken
pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(not ROOT or not RefdataStore(ROOT).exists(STOCKS), reason="PV2_PAPER_REFDATA lacks stock bars"),
]
OBSERVED = ("backtest", "optimize", *V3_GATE_STAGES)


def keep(name: str, harness, run_id) -> list[dict]:
    rows = trace(harness, run_id)
    if OUT:
        run = harness.run(run_id)
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(
            json.dumps(
                {
                    "run": {"status": run.status, "stage": run.current_stage, "reason": run.reason},
                    "config": {key: run.config_json[key] for key in ("data", "refdata", "strategy_class")},
                    "trace": rows,
                },
                indent=2,
                default=str,
            )
        )
    return rows


def report(harness, run_id, stage) -> dict:
    row = next(r for r in harness.attempts(run_id) if r.stage == stage)
    return (row.artifact_json or {}).get("report") or {}


@pytest.mark.usefixtures("pipeline_engine")
def test_etf_rotation_end_to_end_on_real_refdata(pipeline_sessions, tmp_path):
    store = AsOf(RefdataStore(ROOT), STORE_CUTOFF)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(NOW), registry=v3_registry(store))

    natural = create_v3(harness, etf_rotation_00631l_0050(), store=store, idea="00631L / 0050 rotation (natural)")
    run = harness.run(natural)
    assert (run.market, run.symbol) == ("multi", "PORTFOLIO")
    assert {STOCKS, ACTIONS} <= set(run.config_json["refdata"]["datasets"])
    assert run.config_json["data"]["start"] >= "2014-10-31"
    harness.drain(limit=40)
    rows = keep("natural", harness, natural)
    assert rows[0]["stage"] == "spec_validate" and rows[0]["verdict"] == "pass", rows
    assert set(report(harness, natural, "spec_validate")["feasibility"]["prices"]) == {"0050", "00631L"}
    assert all(row["verdict"] != "error" and row["summary"] for row in rows), rows
    last = rows[-1]
    assert last["verdict"] == "fail" or (last["stage"], last["status"]) == ("shadow_paper", "waiting"), last

    for name in OBSERVED:
        harness.dispatcher.registry[name] = Observe(harness.dispatcher.registry[name])
    observed = create_v3(harness, etf_rotation_00631l_0050(), store=store, idea="00631L / 0050 rotation (observed)")
    harness.drain(limit=40)
    rows = keep("observed", harness, observed)
    run = harness.run(observed)
    assert (run.status, run.current_stage) == ("waiting", "shadow_paper"), rows
    assert [row["stage"] for row in rows] == list(V3_DAILY_STAGE_PLAN[:-1])
    assert all(row["summary"] and row["verdict"] != "error" for row in rows), rows
    assert report(harness, observed, "benchmark")["inputs"]["underlying"] == "0050 total return"
    paper = rows[-1]
    assert paper["status"] == "waiting" and "paper starts" in paper["summary"]
