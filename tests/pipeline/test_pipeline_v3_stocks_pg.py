"""Step 8 (M3) acceptance on synthetic refdata: the 00631L / 0050 rotation spec through every v3 stage.

``build_long_store``'s planted MA-band market (TX, TAIEX) plus two ETFs priced from the same index:
0050 at index / 100 from 2003-06-30, 00631L as a 2x daily-leveraged path from 2014-10-31 (the real
first days), a 0050 1 -> 4 split on 2025-06-18 and a 00631L 1 -> 4 split on 2026-03-31 booked in
``tw_corporate_actions``. The spec is hand-written (``v3.examples.etf_rotation_00631l_0050``, no LLM).

* ``natural`` -- as the runner drives it: every stage until the first verdict that is not a pass;
  whatever the gates decide is reported, not asserted.
* ``observed`` -- :class:`Observe` around every gate: each verdict is recorded and the run continues,
  so every stage up to the daily shadow paper runs on stock inputs and the paper books its days.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.plan import V3_DAILY_STAGE_PLAN, V3_GATE_STAGES
from poseidon.pipeline.refdata.datasets import TAIEX
from poseidon.pipeline.refdata.tw_corporate_actions import ACTION_COLUMNS
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_stocks import COLUMNS
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050
from tests.pipeline.conftest import Clock, Harness
from tests.pipeline.daily_paper_support import _by_month
from tests.pipeline.v3_wiring_support import Observe, build_long_store, create_v3, trace, v3_registry

pytestmark = pytest.mark.postgresql
OUT = os.environ.get("POSEIDON_V3_STOCKS_OUT")
SEED = 5
PAPER_CLOCK_START = datetime(2026, 9, 1, 2, 0, tzinfo=UTC)  # 10:00 Taipei: the paper starts that day
SPLITS = {"0050": ("2025-06-18", 4.0), "00631L": ("2026-03-31", 4.0)}
FIRST = {"0050": "2003-06-30", "00631L": "2014-10-31"}
OBSERVED = ("backtest", "optimize", *V3_GATE_STAGES)


def add_etfs(store) -> None:
    index = store.read(TAIEX)
    days = pd.DatetimeIndex(index["trade_date"])
    close = index["close"].to_numpy(float)
    ratio = {c: index[c].to_numpy(float) / close for c in ("open", "high", "low")}
    daily = close[1:] / close[:-1] - 1.0
    lev = np.concatenate([[1.0], np.cumprod(1.0 + 2.0 * daily)]) * 20.0
    rows, actions = [], []
    for symbol, level in (("0050", close / 100.0), ("00631L", lev)):
        split_day, ratio_split = SPLITS[symbol]
        raw = np.where(days >= pd.Timestamp(split_day), level / ratio_split, level)
        frame = pd.DataFrame({"trade_date": days, "symbol": symbol, "close": raw, "volume": 1000.0})
        for column in ("open", "high", "low"):
            frame[column] = raw * ratio[column]
        rows.append(frame.loc[days >= pd.Timestamp(FIRST[symbol]), COLUMNS])
        prev = float(level[days.get_loc(pd.Timestamp(split_day)) - 1])
        action = dict.fromkeys(ACTION_COLUMNS)
        action.update(
            date=pd.Timestamp(split_day),
            market="twse",
            symbol=symbol,
            action_type="split",
            prev_close=prev,
            reference_price=prev / ratio_split,
            factor=1.0 / ratio_split,
            source="test",
        )
        actions.append(action)
    for dataset, frame, column in (
        (STOCKS, pd.concat(rows, ignore_index=True), "trade_date"),
        (ACTIONS, pd.DataFrame(actions), "date"),
    ):
        store.commit(
            dataset,
            schema_version=1,
            date_column=column,
            partitions=_by_month(frame, column),
            sources=[],
            code_version="t",
            full=True,
        )


def keep(name: str, harness, run_id) -> list[dict]:
    rows = trace(harness, run_id)
    if OUT:
        run = harness.run(run_id)
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(
            json.dumps(
                {"run": {"status": run.status, "stage": run.current_stage, "reason": run.reason}, "trace": rows},
                indent=2,
                default=str,
            )
        )
    return rows


def report(harness, run_id, stage) -> dict:
    row = next(r for r in harness.attempts(run_id) if r.stage == stage)
    return (row.artifact_json or {}).get("report") or {}


@pytest.mark.usefixtures("pipeline_engine")
def test_etf_rotation_runs_every_stage_on_stock_inputs(pipeline_sessions, tmp_path):
    store = build_long_store(tmp_path / "refdata", kind="ma_band", seed=SEED)
    add_etfs(store)
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(PAPER_CLOCK_START), registry=v3_registry(store))

    natural = create_v3(harness, etf_rotation_00631l_0050(), store=store, idea="00631L / 0050 rotation (natural)")
    run = harness.run(natural)
    assert (run.market, run.symbol) == ("multi", "PORTFOLIO")
    assert set(run.config_json["refdata"]["datasets"]) >= {STOCKS, ACTIONS, TAIEX}
    assert run.config_json["data"]["start"] >= "2014-10-31"  # 00631L's first day moves the window
    harness.drain(limit=40)
    rows = keep("natural", harness, natural)
    assert rows[0]["stage"] == "spec_validate" and rows[0]["verdict"] == "pass", rows
    validated = report(harness, natural, "spec_validate")
    assert set(validated["feasibility"]["prices"]) == {"0050", "00631L"}
    assert all(row["verdict"] != "error" and row["summary"] for row in rows), rows
    last = rows[-1]
    assert last["verdict"] == "fail" or (last["stage"], last["status"]) == ("shadow_paper", "waiting"), last

    for name in OBSERVED:  # second family, every failing verdict recorded and passed through
        harness.dispatcher.registry[name] = Observe(harness.dispatcher.registry[name])
    observed = create_v3(harness, etf_rotation_00631l_0050(), store=store, idea="00631L / 0050 rotation (observed)")
    harness.drain(limit=40)
    rows = keep("observed", harness, observed)
    run = harness.run(observed)
    # the paper starts 2026-09-01 and the data runs to 10-21: it books its days at the first wake-up
    assert (run.status, run.current_stage) == ("waiting", "awaiting_release"), rows
    assert [row["stage"] for row in rows] == list(run.config_json["stage_plan"]) == list(V3_DAILY_STAGE_PLAN)
    assert all(row["summary"] and row["verdict"] != "error" for row in rows), rows
    backtest = report(harness, observed, "backtest")
    assert backtest["liquidation"]["verdict"] == "pass"  # paid in full: nothing to liquidate
    benchmark = report(harness, observed, "benchmark")
    assert benchmark["inputs"]["underlying"] == "0050 total return"
    regime = report(harness, observed, "regime_split")
    assert regime["inputs"]["underlying"]["labelled_days"] > 250
    timing = report(harness, observed, "execution_timing")["gate_report"]
    assert set(timing["scenarios"]) == {"signal_close", "signal_lag_1d"}
    paper = report(harness, observed, "shadow_paper")
    assert paper["mode"] == "daily_target" and paper["cutoff"] == "2026-10-21" and paper["status"] == "pass"
    assert paper["engine"]["instruments"]["lev"] == {"product": "00631L", "symbol": "00631L", "roll_days_before": 0}
    assert paper["trading_days"] >= 30 and paper["consistency"]["ratio"] == 1.0
    assert paper["paper_start_day"] == "2026-09-01" and paper["protection"]["tripped"] is None
