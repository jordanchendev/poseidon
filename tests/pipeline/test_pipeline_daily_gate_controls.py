"""Daily-class controls (design 2026-10-24 §7): do the daily gates discriminate on real engine output?

Each control runs the target-position engine on synthetic TMF-like bars (12 years, one contract)
and judges the result with every daily gate under the registered ``daily_equity`` thresholds
(tests/pipeline/daily_gate_support.py):

* planted edge -- the next day's open-to-close move carries 0.10% in the direction of a public
  signal; a strategy trading the signal at leverage 1-3 must pass every gate;
* random exposure -- the same market and grid, direction from a seeded coin: must fail a gate;
* static leverage -- always long at the grid's leverage (2-4x): must fail ``benchmark``;
* forced liquidation / ruin -- the planted edge at ~4x (4-5x) with one train day whose low is
  24.3% (26%) under the previous close: must fail ``liquidation`` (code ``forced_liquidation`` /
  ``ruin``).

A few seconds each (one engine run on ~2,500 days is ~0.3 s). Over seeds 1-11 (2026-10-24, not
part of the suite): the planted edge (daily Sharpe ~0.09) passed 10/11 -- seed 11 failed ``wfe``
(0.28): one 365-day OOS window at 3x carries ~+-60% noise in annualised return. At ``EDGE=0.0015``
(seeds 1-10) random exposure was caught 10/10 (``dsr`` every time) and static leverage failed
``benchmark`` 9/10 -- alpha HAC t is noise around ~0 (+-1), so ``t >= 1.0`` lets leveraged beta through
now and then (seed 4: t 1.17, then failed ``holdout`` alpha).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from poseidon.pipeline.plan import class_settings
from tests.pipeline.daily_gate_support import coin, failed_gates, grid, judge, synthetic_market

N_DAYS = 3100
EDGE = 0.001
SEEDS = [1, 2, 3]  # the first three, not picked
OUT = os.environ.get("POSEIDON_DAILY_GATE_CONTROLS_OUT")


def _evidence(name: str, judged: dict) -> dict:
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
        "execution_sensitivity": judged["execution_sensitivity"],
    }
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_daily_edge_passes_every_gate(seed):
    market = synthetic_market(N_DAYS, seed=seed, edge=EDGE)
    judged = judge(market, market.signal, grid())
    record = _evidence(f"planted_edge_seed{seed}", judged)
    assert record["failed"] == [], record
    assert judged["wfe_layout"] == {"windows": 5, "is_days": 1460, "oos_days": 365}  # the long-history layout
    assert record["holdout_days"] >= class_settings("daily_equity")["thresholds"]["holdout_min_days"]
    report = judged["execution_sensitivity"]
    assert report["report_only"] and report["alternative_execution"] == "signal_close"


@pytest.mark.parametrize("seed", SEEDS)
def test_random_exposure_fails(seed):
    market = synthetic_market(N_DAYS, seed=seed, edge=EDGE)
    judged = judge(market, coin(seed, market.dates), grid())
    record = _evidence(f"random_exposure_seed{seed}", judged)
    assert record["failed"], record


@pytest.mark.parametrize("seed", SEEDS)
def test_static_leverage_fails_benchmark(seed):
    market = synthetic_market(N_DAYS, seed=seed, edge=EDGE)
    always_long = dict.fromkeys(market.dates, 1.0)
    judged = judge(market, always_long, grid((2.0, 4.0, 1.0)))
    record = _evidence(f"static_leverage_seed{seed}", judged)
    result = judged["results"]["benchmark"]
    assert not result.passed and result.reason_code in ("alpha_not_positive", "alpha_not_significant"), record
    assert result.report["beta"] > 1.5  # it is leveraged beta, nothing else
    assert judged["results"]["liquidation"].passed


def test_forced_liquidation_in_train_fails():
    # ~4x long into a day whose low is 24.3% under the previous close: the risk ratio at the low
    # (equity / initial margin ~0.13-0.19) is under 0.25 while equity stays positive -> forced
    # liquidation at the low, no ruin.
    market = synthetic_market(N_DAYS, seed=3, edge=EDGE, crash_day=900, crash_low=0.243)
    judged = judge(market, market.signal, grid((4.0, 4.02, 0.01)))
    record = _evidence("forced_liquidation", judged)
    result = judged["results"]["liquidation"]
    assert not result.passed and result.reason_code == "forced_liquidation", record
    assert result.report["ruined"] is None and result.report["forced_liquidations"][0]["date"] == str(market.dates[900])


def test_ruin_in_train_fails():
    # 4-5x long into a 26% low: equity goes below zero at the low -> liquidated, then ruin at the close.
    market = synthetic_market(N_DAYS, seed=3, edge=EDGE, crash_day=900, crash_low=0.26)
    judged = judge(market, market.signal, grid((4.0, 5.0, 0.5)))
    record = _evidence("ruin", judged)
    result = judged["results"]["liquidation"]
    assert not result.passed and result.reason_code == "ruin", record
    assert result.report["ruined"] == str(market.dates[900])
