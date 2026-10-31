"""``rebalance.monthly`` through every v3 stage on PostgreSQL: run #16's 00631L / 0050 rotation with the
monthly check its idea asked for ("每月檢查一次"), hand-written (no LLM).

* ``synthetic`` -- the planted store of ``test_pipeline_v3_stocks_pg`` (ETFs priced off a TAIEX with MA-band
  regimes), :class:`Observe` around every gate, so the daily shadow paper books 2026-09-01..10-21 and
  replays it.
* ``real`` -- the same spec on the real refdata copy (``PV2_PAPER_REFDATA``), created 2026-10-27 13:30Z,
  every gate observed, up to the shadow paper waiting for its first day.

Both check what the component promises at the run level: every rebalance fill of the backtest (and of
the paper) lands on the first session of a month, the open after the month's last session, except the
initial entry at the first open of the window.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd
import pytest

from poseidon.pipeline.plan import V3_DAILY_STAGE_PLAN, V3_GATE_STAGES
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.test_pipeline_v3_stocks_pg import PAPER_CLOCK_START, SEED, add_etfs
from tests.pipeline.v3_wiring_support import Observe, build_long_store, create_v3, trace, v3_registry

pytestmark = pytest.mark.postgresql
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "spec_v3_run16_rotation.json").read_text())
OBSERVED = ("backtest", "optimize", *V3_GATE_STAGES)
REAL_ROOT = os.environ.get("PV2_PAPER_REFDATA")
REAL_NOW = datetime(2026, 10, 27, 13, 30, tzinfo=UTC)


def monthly_rotation(signal: str = "index:taiex_tr") -> dict:
    doc = json.loads(json.dumps(FIXTURE["template"]))
    doc["name"] = "taiex_tr_regime_switch_0050_00631l_monthly"
    doc["allocation"]["params"]["signal"] = signal
    for sleeve in doc["sleeves"]:
        sleeve["rebalance"] = {"component": "rebalance.monthly", "params": {"day": "last_trading_day"}}
    return doc


def stage_row(harness, run_id, stage):
    return next(r for r in harness.attempts(run_id) if r.stage == stage)


def stage_frame(harness, run_id, stage: str, name: str) -> pd.DataFrame:
    directory = Path(url2pathname(urlparse(stage_row(harness, run_id, stage).artifact_uri).path))
    return pd.read_csv(directory / name)


def first_sessions(sessions) -> set[pd.Timestamp]:
    days = pd.Series(pd.DatetimeIndex(pd.to_datetime(sessions)).unique().sort_values())
    return set(days.groupby(days.dt.to_period("M")).min())


def assert_monthly_fills(fills: pd.DataFrame, sessions, entry: pd.Timestamp) -> int:
    traded = sorted({pd.Timestamp(d) for d in fills.loc[fills["reason"] == "rebalance", "date"]})
    allowed = first_sessions(sessions) | {entry}
    assert traded and set(traded) <= allowed, sorted(set(traded) - allowed)[:5]
    return len(traded)


def drive(harness, store, idea: str):
    for name in OBSERVED:
        if not isinstance(harness.dispatcher.registry[name], Observe):
            harness.dispatcher.registry[name] = Observe(harness.dispatcher.registry[name])
    run_id = create_v3(harness, monthly_rotation(), store=store, idea=idea)
    harness.drain(limit=40)
    return run_id, trace(harness, run_id)


def assert_backtest_is_monthly(harness, run_id) -> None:
    report = (stage_row(harness, run_id, "backtest").artifact_json or {})["report"]
    positions = stage_frame(harness, run_id, "backtest", "positions_daily.csv")
    fills = stage_frame(harness, run_id, "backtest", "fills.csv")
    sessions = pd.to_datetime(positions["date"])
    entry = sorted(sessions.unique())[1]  # decided at the window's first close, filled at the next open
    assert assert_monthly_fills(fills, sessions, entry) >= 3
    assert report["metrics"]["forced_liquidations"] == 0


@pytest.mark.usefixtures("pipeline_engine")
def test_monthly_rotation_runs_every_stage_and_the_paper_trades_only_after_month_ends(pipeline_sessions, tmp_path):
    store = build_long_store(tmp_path / "refdata", kind="ma_band", seed=SEED)
    add_etfs(store)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(PAPER_CLOCK_START), registry=v3_registry(store))
    run_id, rows = drive(harness, store, "00631L / 0050 rotation, checked monthly (synthetic)")
    run = harness.run(run_id)
    assert (run.status, run.current_stage) == ("waiting", "awaiting_release"), rows[-1]["summary"]
    assert [row["stage"] for row in rows] == list(V3_DAILY_STAGE_PLAN)
    assert all(row["summary"] and row["verdict"] != "error" for row in rows), rows
    assert "每月只在當月最後一個交易日收盤後檢查一次" in json.dumps(
        stage_row(harness, run_id, "spec_faithfulness").artifact_json, ensure_ascii=False
    )
    assert_backtest_is_monthly(harness, run_id)

    paper = (stage_row(harness, run_id, "shadow_paper").artifact_json or {})["report"]
    assert paper["mode"] == "daily_target" and paper["status"] == "pass" and paper["consistency"]["ratio"] == 1.0
    days = stage_frame(harness, run_id, "shadow_paper", "paper_days.csv")
    paper_fills = stage_frame(harness, run_id, "shadow_paper", "paper_fills.csv")
    booked = set(pd.to_datetime(paper_fills.loc[paper_fills["reason"] == "rebalance", "trading_day"]))
    assert booked <= {pd.Timestamp("2026-10-01")}, booked  # the only month end inside the paper is 09-30
    assert len(days) >= 30


@pytest.mark.skipif(
    not REAL_ROOT or not RefdataStore(REAL_ROOT).exists(STOCKS), reason="PV2_PAPER_REFDATA lacks stock bars"
)
@pytest.mark.usefixtures("pipeline_engine")
def test_monthly_rotation_on_the_real_refdata(pipeline_sessions, tmp_path):
    store = RefdataStore(REAL_ROOT)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(REAL_NOW), registry=v3_registry(store))
    run_id, rows = drive(harness, store, "00631L / 0050 rotation, checked monthly (real refdata)")
    run = harness.run(run_id)
    assert (run.status, run.current_stage) == ("waiting", "shadow_paper"), rows[-1]["summary"]
    assert [row["stage"] for row in rows] == list(V3_DAILY_STAGE_PLAN[:-1])
    assert all(row["summary"] and row["verdict"] != "error" for row in rows), rows
    assert run.config_json["data"]["start"] >= "2014-10-31"
    assert_backtest_is_monthly(harness, run_id)
