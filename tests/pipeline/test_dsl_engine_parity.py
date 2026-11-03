"""DSL v2 on the target-position engine (design §10 step 7): trade-for-trade parity with the v1 runner.

The run #3 / #6 specs are the optimize best specs stored in the live DB (``fixtures/dsl_parity_runs.json``);
``scripts/pv2_dsl_parity.py`` checks them on their full frozen windows, these tests on bars that ship
with the repo.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
from poseidon.pipeline import backtesting
from poseidon.pipeline.backtesting import BacktestFailed, dsl_engine, run_backtest, stressed_cost_model
from poseidon.pipeline.dsl_parity import compare_trades, run_parity
from poseidon.pipeline.plan import DEFAULT_BACKTEST
from poseidon.pipeline.stage_api import StageStopped
from poseidon.pipeline.stages import BacktestStage
from poseidon.strategies.dsl.v2.examples import tx_atr_squeeze_breakout, tx_orb30_day
from tests.dsl_v2_support import random_walk_bars, real_tx_fixture

FIXTURES = Path(__file__).parent / "fixtures"
RUNS = {run["run_no"]: run for run in json.loads((FIXTURES / "dsl_parity_runs.json").read_text())}
FAR = {"209912": pd.Timestamp("2099-12-16")}
# The real-bar fixture (2026-09-01..09-12) holds no settlement; this calendar puts a switch on Wed 09-09.
SWITCH_0909 = {"A": pd.Timestamp("2026-09-09"), "B": pd.Timestamp("2099-12-16")}
CONFIG = copy.deepcopy(DEFAULT_BACKTEST)


def _llm_v21() -> dict:
    return json.loads((Path(__file__).parents[1] / "fixtures" / "llm_v21_atr_contraction_spec.json").read_text())


DOCS = {
    "orb30_day": tx_orb30_day,
    "atr_squeeze": tx_atr_squeeze_breakout,
    "llm_v21_trail": _llm_v21,
    "run3_best": lambda: copy.deepcopy(RUNS[3]["best_spec"]),
    "run6_best": lambda: copy.deepcopy(RUNS[6]["best_spec"]),
}


def _factory(name: str):
    return DslV2StrategyFactory(spec=DOCS[name]()).build


def _walk(seed: int) -> pd.DataFrame:
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-05", periods=30)]
    return random_walk_bars(dates, seed=seed, vol=8.0)


@pytest.mark.parametrize("name", sorted(DOCS))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_golden_spec_trades_identically_on_both_engines(name, seed):
    report = run_parity(_factory(name), _walk(seed), CONFIG, expiries=FAR)
    assert report.mismatches == () and report.summary()["trades_csv_identical"]
    assert report.trades_v1 == report.trades_target
    assert report.max_money_diff_twd <= 1e-6 and report.daily_max_abs_diff_twd <= 1e-6


@pytest.mark.parametrize("name", sorted(DOCS))
def test_cost_stress_and_a_warm_up_cutoff_keep_parity(name):
    bars = _walk(4)
    stressed = {"cost_model": stressed_cost_model("tw_futures_pipeline", 2.0), "stop_slippage_ticks": 1.0}
    assert run_parity(_factory(name), bars, CONFIG, expiries=FAR, **stressed).ok
    assert run_parity(_factory(name), bars, CONFIG, expiries=FAR, trade_from=bars.index[len(bars) // 2]).ok


@pytest.mark.parametrize("name", sorted(DOCS))
def test_on_real_bars_the_settlement_roll_is_the_only_divergence(name):
    bars = real_tx_fixture()
    report = run_parity(_factory(name), bars, CONFIG, expiries=SWITCH_0909)
    assert report.trades_v1 == report.trades_target
    held_across = [line for line in report.mismatches if "held across a settlement switch" in line]
    daily = [line for line in report.mismatches if line.startswith("daily")]
    assert set(report.mismatches) == set(held_across) | set(daily)
    assert bool(report.roll_crossing) == bool(report.mismatches)
    assert run_parity(_factory(name), bars, CONFIG, expiries=FAR).ok


def test_a_trade_across_the_switch_pays_the_roll_and_skips_the_gap():
    doc = copy.deepcopy(RUNS[6]["best_spec"])  # long at the 13:40 close, held 172 bars
    rule = doc["rules"][0]
    rule["condition"].update(start="13:00", end="13:00")
    rule["risk"]["max_hold_bars"] = 12  # exits at 14:00 local: always crosses 13:25 -> 13:30
    bars = real_tx_fixture()
    report = run_parity(DslV2StrategyFactory(spec=doc).build, bars, CONFIG, expiries=SWITCH_0909)
    assert not report.ok and len(report.roll_crossing) == 1
    v1 = {t["trade_no"]: t for t in report.v1.trades}
    target = {t["trade_no"]: t for t in report.target.trades}
    (crossing,) = report.roll_crossing
    a, b = v1[crossing], target[crossing]
    assert (a["entry_time"], a["exit_time"], a["entry_price"], a["exit_price"]) == (
        b["entry_time"],
        b["exit_time"],
        b["entry_price"],
        b["exit_price"],
    )
    switch = bars.index.get_loc(pd.Timestamp("2026-09-09 13:30", tz="Asia/Taipei").tz_convert("UTC"))
    gap = bars["open"].iloc[switch] - bars["close"].iloc[switch - 1]
    roll_cost = 2 * (200.0 + 50.0) + 0.00002 * 200 * (
        bars["close"].iloc[switch - 1] - 1 + bars["open"].iloc[switch] + 1
    )
    assert b["gross_pnl_twd"] == pytest.approx(a["gross_pnl_twd"] - gap * 200 - 2 * 200.0)
    assert b["pnl_twd"] == pytest.approx(a["pnl_twd"] - gap * 200 - roll_cost)
    others = [line for line in report.mismatches if f"trade {crossing} " not in line and not line.startswith("daily")]
    assert others == []


def _trades() -> list[dict]:
    return run_backtest(_factory("orb30_day")(), _walk(1), CONFIG).trades


def _swap_entry_times(trades: list[dict]) -> None:
    trades[0]["entry_time"], trades[1]["entry_time"] = trades[1]["entry_time"], trades[0]["entry_time"]


@pytest.mark.parametrize(
    ("seed_bad", "fragment"),
    [
        (lambda t: t[3].update(entry_price=t[3]["entry_price"] + 1), "trade 4 entry_price"),
        (lambda t: t[2].update(pnl_twd=t[2]["pnl_twd"] + 2.0), "trade 3 pnl_twd"),
        (lambda t: t[4].update(exit_reason="seeded"), "trade 5 exit_reason"),
        (lambda t: t.pop(), "trade count"),
        (_swap_entry_times, "trade 1 entry_time"),
    ],
)
def test_the_comparison_catches_each_seeded_mismatch(seed_bad, fragment):
    good = _trades()
    assert len(good) > 5 and compare_trades(good, _trades())[0] == []
    bad = copy.deepcopy(good)
    seed_bad(bad)
    problems, _ = compare_trades(good, bad)
    assert any(fragment in line for line in problems), problems


def test_a_money_difference_within_one_twd_is_tolerated():
    good = _trades()
    bad = copy.deepcopy(good)
    bad[0]["pnl_twd"] += 0.5
    problems, worst = compare_trades(good, bad)
    assert problems == [] and worst == pytest.approx(0.5)


def test_the_engine_is_v1_unless_a_config_selects_target_v1():
    assert "engine" not in DEFAULT_BACKTEST  # every config frozen so far: v1, hashes unchanged
    assert dsl_engine(CONFIG) == "v1" and dsl_engine({**CONFIG, "engine": "target_v1"}) == "target_v1"
    with pytest.raises(ValueError, match="unknown backtest engine"):
        dsl_engine({**CONFIG, "engine": "v2"})


def test_a_cost_model_without_a_per_contract_form_is_refused():
    with pytest.raises(BacktestFailed, match="not a per-contract futures model"):
        run_backtest(
            _factory("orb30_day")(),
            _walk(1),
            {**CONFIG, "cost_model": "tw_futures", "engine": "target_v1"},
            expiries=FAR,
        )


def _stage_pipeline(tmp_path, engine: str | None):
    from tests.pipeline.test_pipeline_step2_unit import Pipeline, _config, _validated, tx_spec

    config = _config()
    if engine is not None:
        config["backtest"]["engine"] = engine
    pipeline = Pipeline(tmp_path / (engine or "v1"), tx_spec(), config=config)
    pipeline.seed(_validated(tx_spec()).report)
    return pipeline


def test_a_backtest_stage_selecting_target_v1_writes_the_same_trades(tmp_path, monkeypatch):
    monkeypatch.setattr(backtesting, "settlement_expiries", lambda product: FAR)
    outputs = {}
    for engine in (None, "target_v1"):
        pipeline = _stage_pipeline(tmp_path, engine)
        result, _ = pipeline.run(BacktestStage(pipeline.source))
        outputs[engine] = result
    v1, target = outputs[None], outputs["target_v1"]
    assert target.files["trades.csv"] == v1.files["trades.csv"]
    assert pd.read_csv(io.BytesIO(target.files["trades.csv"]))["pnl_twd"].size == v1.report["metrics"]["n_trades"] > 0
    assert target.report["metrics"]["total_pnl_twd"] == pytest.approx(v1.report["metrics"]["total_pnl_twd"])


def test_a_stop_request_inside_a_target_engine_backtest_raises_stage_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(backtesting, "settlement_expiries", lambda product: FAR)
    pipeline = _stage_pipeline(tmp_path, "target_v1")
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 50

    with pytest.raises(StageStopped):
        pipeline.run(BacktestStage(pipeline.source), should_stop=should_stop)
