"""Daily-class controls (design 2026-10-24 §7): do the daily gates discriminate on real engine output?

Each control runs the target-position engine on synthetic TMF-like bars (12 years, one contract)
and judges the result with every daily gate under the registered ``daily_equity`` thresholds
(tests/pipeline/daily_gate_support.py):

* planted edge -- a persistent public signal (AR(1), coefficient 0.95: its sign lasts ~10 days);
  the next day's open-to-close move carries 0.15% in its direction. A strategy trading the signal
  at leverage 1-3 must pass every gate, ``execution_timing`` included (one day late keeps ~80% of
  the edge);
* one-day edge (timing artefact) -- the same 0.10% but on an i.i.d. signal: the edge exists only
  for a decision on the same day's signal, so the ``signal_lag_1d`` scenario loses it and
  ``execution_timing`` must fail;
* overnight-gap edge (timing artefact) -- the signal predicts the gap close(t) -> open(t+1); a
  strategy filled at the signal close takes it, ``next_open`` cannot: ``execution_timing`` must fail;
* random exposure -- the one-day-edge market and grid, direction from a seeded coin: must fail a gate;
* static leverage -- always long at the grid's leverage (2-4x) on the one-day-edge market: must fail
  ``benchmark`` (alpha HAC t >= 1.65 since the owner decision of 2026-10-24 afternoon);
* forced liquidation / ruin -- the one-day edge at ~4x (4-5x) with one train day whose low is
  24.3% (26%) under the previous close: must fail ``liquidation`` (code ``forced_liquidation`` /
  ``ruin``).

A few seconds each (one engine run on ~2,500 days is ~0.3 s). The suite runs seeds 1-3; the 20-seed
sweep is ``python -m tests.pipeline.daily_gate_sweep``. Over seeds 1-20 (2026-10-24, after the owner
decisions of that afternoon, not part of the suite): the persistent planted edge passed every gate 19/20
at 0.15% (seed 12 failed ``holdout``) and 16/20 at 0.10% (``wfe`` 2, 3, 10, ``regime_split`` 3,
``holdout`` 12; ``execution_timing`` passed all 20, one day late kept 67-104% of the Sharpe); the one-day
edge and the overnight-gap edge failed ``execution_timing`` 20/20; static leverage failed ``benchmark``
20/20 at t >= 1.65 (largest t 1.33; at the earlier t >= 1.0, seeds 4, 8 and 19 would have passed it);
random exposure failed a gate 20/20.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from poseidon.pipeline.plan import class_settings
from tests.pipeline.daily_gate_support import coin, failed_gates, grid, judge, synthetic_market

N_DAYS = 3100
EDGE = 0.001  # the one-day, gap, random-exposure and static-leverage markets
PLANTED_EDGE = 0.0015  # the persistent planted edge (0.10% passes 16/20 seeds, see above)
PERSISTENCE = 0.95
SEEDS = [1, 2, 3]  # the first three, not picked
OUT = os.environ.get("POSEIDON_DAILY_GATE_CONTROLS_OUT")


def evidence(name: str, judged: dict) -> dict:
    record = {
        "control": name,
        "best_params": judged["best_params"],
        "n_trials": judged["n_trials"],
        "wfe_layout": judged["wfe_layout"],
        "train_days": judged["train"].n_days,
        "holdout_days": judged["holdout"].n_days,
        "failed": failed_gates(judged),
        "gates": {
            gate: None
            if result is None
            else {"verdict": result.verdict.value, "code": result.reason_code, "headline": result.headline}
            for gate, result in judged["results"].items()
        },
        "benchmark_alpha_t": judged["results"]["benchmark"].report.get("alpha_t_hac"),
        "execution_timing": judged["results"]["execution_timing"].report,
    }
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def planted_edge(seed: int, edge: float = PLANTED_EDGE) -> dict:
    market = synthetic_market(N_DAYS, seed=seed, edge=edge, persistence=PERSISTENCE)
    return judge(market, market.signal, grid())


def one_day_edge(seed: int) -> dict:
    market = synthetic_market(N_DAYS, seed=seed, edge=EDGE)
    return judge(market, market.signal, grid())


def overnight_gap_edge(seed: int) -> dict:
    market = synthetic_market(N_DAYS, seed=seed, gap_edge=EDGE)
    return judge(market, market.signal, grid(), execution="signal_close")


def random_exposure(seed: int) -> dict:
    market = synthetic_market(N_DAYS, seed=seed, edge=EDGE)
    return judge(market, coin(seed, market.dates), grid())


def static_leverage(seed: int) -> dict:
    market = synthetic_market(N_DAYS, seed=seed, edge=EDGE)
    return judge(market, dict.fromkeys(market.dates, 1.0), grid((2.0, 4.0, 1.0)))


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_daily_edge_passes_every_gate(seed):
    judged = planted_edge(seed)
    record = evidence(f"planted_edge_seed{seed}", judged)
    assert record["failed"] == [], record
    assert judged["wfe_layout"] == {"windows": 5, "is_days": 1460, "oos_days": 365}  # the long-history layout
    assert record["holdout_days"] >= class_settings("daily_equity")["thresholds"]["holdout_min_days"]
    assert set(record["execution_timing"]["scenarios"]) == {"signal_close", "signal_lag_1d"}


@pytest.mark.parametrize("seed", SEEDS)
def test_one_day_edge_fails_execution_timing(seed):
    judged = one_day_edge(seed)
    record = evidence(f"one_day_edge_seed{seed}", judged)
    assert "execution_timing" in record["failed"], record
    lag = record["execution_timing"]["scenarios"]["signal_lag_1d"]
    assert not lag["passed"] and record["execution_timing"]["scenarios"]["signal_close"]["passed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_overnight_gap_edge_fails_execution_timing(seed):
    judged = overnight_gap_edge(seed)
    record = evidence(f"overnight_gap_edge_seed{seed}", judged)
    assert "execution_timing" in record["failed"], record
    assert not record["execution_timing"]["scenarios"]["next_open"]["passed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_exposure_fails(seed):
    record = evidence(f"random_exposure_seed{seed}", random_exposure(seed))
    assert record["failed"], record


@pytest.mark.parametrize("seed", SEEDS)
def test_static_leverage_fails_benchmark(seed):
    judged = static_leverage(seed)
    record = evidence(f"static_leverage_seed{seed}", judged)
    result = judged["results"]["benchmark"]
    assert not result.passed and result.reason_code in ("alpha_not_positive", "alpha_not_significant"), record
    assert result.report["beta"] > 1.5  # it is leveraged beta, nothing else
    assert result.report["alpha_t_min"] == 1.65
    assert judged["results"]["liquidation"].passed


def test_forced_liquidation_in_train_fails():
    # ~4x long into a day whose low is 24.3% under the previous close: the risk ratio at the low
    # (equity / initial margin ~0.13-0.19) is under 0.25 while equity stays positive -> forced
    # liquidation at the low, no ruin.
    market = synthetic_market(N_DAYS, seed=3, edge=EDGE, crash_day=900, crash_low=0.243)
    judged = judge(market, market.signal, grid((4.0, 4.02, 0.01)))
    record = evidence("forced_liquidation", judged)
    result = judged["results"]["liquidation"]
    assert not result.passed and result.reason_code == "forced_liquidation", record
    assert result.report["ruined"] is None and result.report["forced_liquidations"][0]["date"] == str(market.dates[900])


def test_ruin_in_train_fails():
    # 4-5x long into a 26% low: equity goes below zero at the low -> liquidated, then ruin at the close.
    market = synthetic_market(N_DAYS, seed=3, edge=EDGE, crash_day=900, crash_low=0.26)
    judged = judge(market, market.signal, grid((4.0, 5.0, 0.5)))
    record = evidence("ruin", judged)
    result = judged["results"]["liquidation"]
    assert not result.passed and result.reason_code == "ruin", record
    assert result.report["ruined"] == str(market.dates[900])
