"""The holdout gate's benchmark condition under ``holdout_benchmark_rule`` (owner decision 2026-10-29).

Runs frozen without the rule require holdout alpha > 0 and never build the matched run. With it, the holdout passes
that condition on alpha > 0 or on the benchmark gate's risk route evaluated on the holdout run, with the holdout's
own block length; when the risk route does not apply, alpha alone decides. Synthetic series cover the rule; the
daily stage runs the real engine on the synthetic long-history refdata of ``v3_wiring_support``.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poseidon.decision_loop.manifest import canonical_json
from poseidon.pipeline.daily_stages import DailyHoldoutStage, DailyWindow
from poseidon.pipeline.family import MemoryFamily
from poseidon.pipeline.gates import benchmark, holdout
from poseidon.pipeline.gates.daily import DailyResult
from poseidon.pipeline.plan import DAILY_GATES, DAILY_THRESHOLDS, class_settings
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.v3.config import daily_data_config
from tests.pipeline.v3_wiring_support import build_long_store, planted_edge_spec, static_leverage_spec

# No config freezes the holdout rule (2026-10-29: alpha > 0 alone passes ~38% of the nulls at holdout length, so
# alpha or risk cannot stay within 4.5%; tests/pipeline/holdout_risk_calibration.py). RULE: the daily thresholds plus
# the one setting whose risk route alone stayed within 3.0% on every null and length (21-day blocks, 99%).
ALPHA_ONLY = dict(DAILY_THRESHOLDS)
RULE = {
    **ALPHA_ONLY,
    holdout.RULE_KEY: benchmark.RULE_ALPHA_OR_RISK,
    holdout.BLOCK_KEY: 21,
    holdout.CONFIDENCE_KEY: 0.99,
}
TERMINAL = json.loads((Path(__file__).parent / "fixtures" / "v3_terminal_runs.json").read_text())["runs"]


def days(n: int, start: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, periods=n)]


def daily(returns, dates) -> DailyResult:
    equity = 1_000_000.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    zeros = (0.0,) * len(dates)
    return DailyResult(tuple(dates), tuple(float(e) for e in equity), 1_000_000.0, zeros, zeros, zeros)


def market(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A 1x underlying alternating 250-day bull and 100-day bear phases; ``bear`` flags the bear days."""

    rng = np.random.default_rng(seed)
    bear = (np.arange(n) % 350) >= 250
    return np.where(bear, rng.normal(-0.003, 0.02, n), rng.normal(0.0008, 0.01, n)), bear


class Matched:
    """The beta-matched run of the synthetic holdout: ``leverage`` x the underlying, ``cost`` a day; records calls."""

    def __init__(self, dates, x, cost: float = 0.0):
        self.dates, self.x, self.cost, self.calls = dates, x, cost, []

    def __call__(self, leverage: float):
        self.calls.append(leverage)
        return daily(leverage * self.x - self.cost, self.dates)


def never(_leverage):
    raise AssertionError("the matched run is built only under holdout_benchmark_rule")


def inputs(strategy, x, *, matched=None, basis="return", train_mean=0.001, alpha=None) -> holdout.HoldoutDailyInputs:
    """A train run, ``strategy`` as the holdout on ``x``'s days and its regression on ``x`` as the benchmark;
    ``alpha`` replaces the regression's alpha (a holdout whose alpha route fails, whatever the series)."""

    held_days = days(len(x), "2024-01-01")
    train = daily(np.random.default_rng(1).normal(train_mean, 0.01, 1000), days(1000, "2020-01-01"))
    held = daily(strategy, held_days)
    stats = benchmark.regress_daily(held, dict(zip(held_days, x.tolist(), strict=True)), basis=basis, lags=5)
    if alpha is not None:
        stats["alpha_per_day"] = alpha
    return holdout.HoldoutDailyInputs(train, held, basis, stats, matched=matched)


def check(result, verdict: str, code: str):
    assert (result.verdict, result.reason_code) == (Verdict(verdict), code), (result.reason, result.report)
    canonical_json(result.report)
    return result


def timed(n: int = 1050, seed: int = 4):
    """2x the underlying in bull phases, 0.5x in bear phases: a clearly smaller drawdown than the beta-matched run."""

    x, bear = market(n, seed)
    return np.where(bear, 0.5, 2.0) * x, x


def test_no_config_freezes_the_holdout_rule():
    for strategy_class in ("daily_equity", "daily_fixed", "intraday_fixed"):
        assert holdout.RULE_KEY not in class_settings(strategy_class)["thresholds"]
    for run in TERMINAL.values():  # every judged run keeps holdout alpha alone
        assert holdout.RULE_KEY not in json.dumps(run["config_json"])


def test_without_the_rule_the_holdout_judges_alpha_alone():
    strategy, x = timed()
    with_hook = holdout.evaluate_daily(inputs(strategy, x, matched=never, alpha=-1e-6), ALPHA_ONLY)
    without = holdout.evaluate_daily(inputs(strategy, x, alpha=-1e-6), ALPHA_ONLY)
    assert with_hook == without
    check(with_hook, "fail", "holdout_alpha_not_positive")
    assert not {"rule", "risk_route", "passed_routes"} & set(with_hook.report)


def test_a_drawdown_cutting_holdout_passes_by_the_risk_route_with_the_holdout_block():
    strategy, x = timed()
    matched = Matched(days(len(x), "2024-01-01"), x)
    result = check(holdout.evaluate_daily(inputs(strategy, x, matched=matched, alpha=-1e-6), RULE), "pass", "pass")
    report = result.report
    assert report["alpha_route"] == "fail"
    assert report["rule"] == benchmark.RULE_ALPHA_OR_RISK and report["passed_routes"] == ["risk"]
    risk = report["risk_route"]
    assert matched.calls == [risk["matched_leverage"]] == [round(report["benchmark"]["beta"], 2)]
    assert risk["status"] == "pass" and risk["calmar_diff_lower"] > 0
    assert (risk["block_days"], risk["confidence"]) == (21, 0.99)
    assert risk["strategy"]["max_drawdown"] < risk["matched"]["max_drawdown"]
    assert "Calmar" in result.headline and "passed by the risk route" in result.reason
    again = holdout.evaluate_daily(inputs(strategy, x, matched=Matched(matched.dates, x), alpha=-1e-6), RULE)
    assert again == result  # seeded bootstrap: the same verdict and numbers on a re-run


def test_static_leverage_on_the_holdout_fails_both_routes():
    x = np.random.default_rng(4).normal(0.0008, 0.01, 1050)
    jitter = np.random.default_rng(1).normal(0.0, 1e-5, len(x))
    matched = Matched(days(len(x), "2024-01-01"), x, cost=0.00002)
    result = holdout.evaluate_daily(inputs(2.0 * x - 0.00002 + jitter, x, matched=matched), RULE)
    check(result, "fail", "holdout_alpha_not_positive")
    risk = result.report["risk_route"]
    assert risk["status"] == "fail" and risk["matched_leverage"] == 2.0 and risk["calmar_diff_lower"] <= 0
    assert result.report["passed_routes"] == [] and "risk route: Calmar not above" in result.reason


def test_when_the_risk_route_does_not_apply_alpha_alone_decides_and_the_report_says_why():
    x, _bear = market(1050, 4)
    noise = np.random.default_rng(2).normal(0.0, 0.001, len(x))
    low_beta = holdout.evaluate_daily(inputs(0.1 * x + 0.0002 + noise, x, matched=never), RULE)
    check(low_beta, "pass", "pass")
    assert low_beta.report["passed_routes"] == ["alpha"]
    assert low_beta.report["risk_route"]["status"] == "not_applicable"
    assert "beta 0.10 < 0.25" in low_beta.report["risk_route"]["reason"]

    def liquidated(_leverage):
        return benchmark.NotApplicable("the 1.5x matched run hit forced_liquidation on 2024-03-04")

    strategy, x = timed()
    lost = holdout.evaluate_daily(inputs(strategy, x, matched=liquidated, alpha=-1e-6), RULE)
    check(lost, "fail", "holdout_alpha_not_positive")
    assert lost.report["risk_route"]["reason"] == "the 1.5x matched run hit forced_liquidation on 2024-03-04"
    assert "risk route n/a (the 1.5x matched run" in lost.headline

    short = 2 * RULE[holdout.BLOCK_KEY] - 1  # too few days for two blocks (below holdout_min_days as well)
    bull = np.random.default_rng(3).normal(0.0008, 0.01, short)
    any_length = {k: v for k, v in RULE.items() if k != "holdout_min_days"}
    few = holdout.evaluate_daily(
        inputs(bull + 0.0005, bull, matched=Matched(days(short, "2024-01-01"), bull)), any_length
    )
    check(few, "pass", "pass")
    assert (
        few.report["risk_route"]["status"] == "not_applicable" and "on at least" in few.report["risk_route"]["reason"]
    )

    x, _bear = market(1050, 4)
    twd = check(holdout.evaluate_daily(inputs(0.1 * x + 0.0002, x, matched=never, basis="twd"), RULE), "pass", "pass")
    assert twd.report["risk_route"]["status"] == "not_applicable"  # daily_fixed: judged in TWD


def test_the_rule_needs_the_benchmark_rule_its_block_and_a_known_value():
    strategy, x = timed()
    alone = {k: v for k, v in RULE.items() if k != benchmark.RULE_KEY}
    check(holdout.evaluate_daily(inputs(strategy, x, matched=never), alone), "fail", "invalid_input")
    unknown = {**RULE, holdout.RULE_KEY: "calmar_only"}
    result = check(holdout.evaluate_daily(inputs(strategy, x, matched=never), unknown), "fail", "invalid_input")
    assert "calmar_only" in result.reason
    no_block = {k: v for k, v in RULE.items() if k != holdout.BLOCK_KEY}
    matched = Matched(days(len(x), "2024-01-01"), x)
    result = check(holdout.evaluate_daily(inputs(strategy, x, matched=matched), no_block), "fail", "invalid_input")
    assert holdout.BLOCK_KEY in result.reason and holdout.RULE_KEY in result.reason


def test_the_holdout_decides_its_earlier_checks_before_building_the_matched_run():
    strategy, x = timed()
    weak = inputs(strategy, x, matched=never, train_mean=-0.001)
    check(holdout.evaluate_daily(weak, RULE), "fail", "train_not_positive")


# ----------------------------------------------------------------------------- the daily stage on the engine


AS_OF = date(2026, 10, 22)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    return build_long_store(tmp_path_factory.mktemp("planted"), kind="ma_band", seed=5)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return build_long_store(tmp_path_factory.mktemp("plain"), kind="drift", seed=5)


def judge(store, spec: dict, params: dict, thresholds: dict = RULE):
    config = {
        "strategy_class": "daily_equity",
        "thresholds": copy.deepcopy(thresholds),
        "gates": copy.deepcopy(DAILY_GATES),
        "data": daily_data_config(as_of=AS_OF),
    }
    upstream = {
        "spec_validate": SimpleNamespace(report={"template": spec}),
        "optimize": SimpleNamespace(report={"best": {"params": params}}),
    }
    ctx = SimpleNamespace(
        config=config,
        upstream=upstream,
        family=MemoryFamily(run_id="r"),
        trials=None,
        run_id="r",
        attempt=1,
        stage_run_id=None,
    )
    return DailyHoldoutStage(refdata=lambda: store).evaluate(ctx), config


def test_the_stage_matches_the_holdout_beta_over_the_holdout_days(planted):
    result, config = judge(planted, planted_edge_spec(), {"ma_days": 200, "band": 0.02})
    gate = result.report["gate_report"]
    risk = gate["risk_route"]
    assert gate["rule"] == benchmark.RULE_ALPHA_OR_RISK and risk["status"] in ("pass", "fail")
    assert risk["matched_leverage"] == round(gate["benchmark"]["beta"], 2) and "matched_costs" in risk
    assert risk["block_days"] == RULE[holdout.BLOCK_KEY]
    window = DailyWindow.from_config(config)
    assert gate["holdout_start"] >= window.holdout_start.isoformat()  # fresh capital on the holdout days only


def test_the_stage_without_the_rule_reports_as_before(plain):
    with_rule, _ = judge(plain, static_leverage_spec(), {"lev": 2.5})
    gate = with_rule.report["gate_report"]
    assert gate["risk_route"]["status"] == "fail" and gate["risk_route"]["matched_leverage"] == pytest.approx(
        2.5, abs=0.1
    )
    without, _ = judge(plain, static_leverage_spec(), {"lev": 2.5}, ALPHA_ONLY)
    assert not {"rule", "risk_route", "passed_routes"} & set(without.report["gate_report"])
