"""The benchmark gate's beta-matched risk route (owner decision 2026-10-28; ``thresholds.benchmark_rule``).

Runs frozen without the rule judge alpha alone and never build the matched run. With it, the gate passes on
alpha (as before) or on Calmar against the 1x underlying held at the strategy's measured beta. Synthetic series
cover the rule; the daily stage runs the real engine on the synthetic long-history refdata of
``v3_wiring_support`` (a planted MA-band regime, and a structure-free market for static leverage).
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

from poseidon.core.canonical import canonical_json
from poseidon.pipeline.daily_stages import DailyBenchmarkStage, beta_matched_run, beta_matched_spec, underlying_daily
from poseidon.pipeline.gates import benchmark
from poseidon.pipeline.gates.daily import DailyResult, underlying_returns
from poseidon.pipeline.plan import DAILY_GATES, DAILY_THRESHOLDS, class_settings
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.v3.config import daily_data_config
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050, tmf_dynamic_leverage_3x
from tests.pipeline.v3_wiring_support import build_long_store, planted_edge_spec, static_leverage_spec

RISK = dict(DAILY_THRESHOLDS)
ALPHA_ONLY = {k: v for k, v in RISK.items() if k not in (benchmark.RULE_KEY, *benchmark.RISK_KEYS)}
TERMINAL = json.loads((Path(__file__).parent / "fixtures" / "v3_terminal_runs.json").read_text())["runs"]


def days(n: int) -> list[date]:
    return [d.date() for d in pd.bdate_range("2005-01-03", periods=n)]


def daily(returns, dates) -> DailyResult:
    equity = 1_000_000.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    zeros = (0.0,) * len(dates)
    return DailyResult(tuple(dates), tuple(float(e) for e in equity), 1_000_000.0, zeros, zeros, zeros)


def market(n: int = 3000, seed: int = 4) -> tuple[list[date], np.ndarray, np.ndarray]:
    """A 1x underlying alternating 250-day bull and 100-day bear phases; ``bear`` flags the bear days."""

    rng = np.random.default_rng(seed)
    bear = (np.arange(n) % 350) >= 250
    x = np.where(bear, rng.normal(-0.003, 0.02, n), rng.normal(0.0008, 0.01, n))
    return days(n), x, bear


def inputs(strategy, dates, x, *, matched=None, basis="return") -> benchmark.BenchmarkDailyInputs:
    underlying = dict(zip(dates, x.tolist(), strict=True))
    return benchmark.BenchmarkDailyInputs(daily(strategy, dates), underlying, basis, 5, matched=matched)


class Matched:
    """The beta-matched run of the synthetic market: ``leverage`` x the underlying, ``cost`` a day; records calls."""

    def __init__(self, dates, x, cost: float = 0.0):
        self.dates, self.x, self.cost, self.calls = dates, x, cost, []

    def __call__(self, leverage: float):
        self.calls.append(leverage)
        return daily(leverage * self.x - self.cost, self.dates)


def never(_leverage):
    raise AssertionError("the matched run is built only under benchmark_rule")


def test_new_daily_configs_freeze_the_rule_and_the_dsl_class_does_not():
    thresholds = class_settings("daily_equity")["thresholds"]
    assert thresholds[benchmark.RULE_KEY] == benchmark.RULE_ALPHA_OR_RISK
    assert all(key in thresholds for key in benchmark.RISK_KEYS)
    assert benchmark.RULE_KEY not in class_settings("intraday_fixed")["thresholds"]
    for run in TERMINAL.values():  # runs judged before the rule existed keep the alpha-only rule
        assert benchmark.RULE_KEY not in json.dumps(run["config_json"])


def test_without_the_rule_the_gate_judges_alpha_alone():
    dates, x, _bear = market()
    timed = x + 0.002
    with_hook = benchmark.evaluate_daily(inputs(timed, dates, x, matched=never), ALPHA_ONLY)
    without = benchmark.evaluate_daily(inputs(timed, dates, x), ALPHA_ONLY)
    assert with_hook == without and with_hook.passed
    assert "risk_route" not in with_hook.report and "rule" not in with_hook.report


def test_a_drawdown_cutting_strategy_passes_by_the_risk_route_alone():
    dates, x, bear = market()
    strategy = np.where(bear, 0.5, 2.0) * x
    matched = Matched(dates, x)
    strict_alpha = {**RISK, "benchmark_alpha_t_min": 99.0}
    result = benchmark.evaluate_daily(inputs(strategy, dates, x, matched=matched), strict_alpha)
    report = result.report
    assert result.passed and report["passed_routes"] == ["risk"] and report["alpha_route"] == "fail"
    risk = report["risk_route"]
    assert matched.calls == [risk["matched_leverage"]] == [round(report["beta"], 2)]
    assert risk["status"] == "pass" and risk["calmar_diff_lower"] > 0
    assert risk["strategy"]["max_drawdown"] < risk["matched"]["max_drawdown"]
    assert all(risk[side][key] is not None for side in ("strategy", "matched") for key in ("cagr", "calmar"))
    assert "Calmar" in result.headline and result.reason == "passed by the risk route"
    canonical_json(report)
    again = benchmark.evaluate_daily(inputs(strategy, dates, x, matched=Matched(dates, x)), strict_alpha)
    assert again == result  # seeded bootstrap: the same verdict and numbers on a re-run


def test_static_leverage_fails_on_alpha_and_the_risk_route_does_not_apply():
    dates, x, _bear = market()
    jitter = np.random.default_rng(1).normal(0.0, 1e-5, len(x))
    result = benchmark.evaluate_daily(
        inputs(2.0 * x - 0.00002 + jitter, dates, x, matched=Matched(dates, x, cost=0.00002)), RISK
    )
    assert (result.verdict, result.reason_code) == (Verdict.FAIL, "alpha_not_positive")
    risk = result.report["risk_route"]
    assert risk["status"] == "not_applicable" and risk["matched_leverage"] == 2.0
    assert result.report["passed_routes"] == [] and "risk route: tracking" in result.reason


def test_a_strategy_within_rounding_of_its_matched_run_is_not_judged_by_the_risk_route():
    dates, x, bear = market()
    near = np.where(bear, 0.95, 0.97) * x  # a 0.02x gap: what one more TMF contract on 1M TWD does on some days
    strict_alpha = {**RISK, "benchmark_alpha_t_min": 99.0}
    frozen_before = {k: v for k, v in strict_alpha.items() if k != "benchmark_risk_min_tracking"}  # no floor
    unfloored = benchmark.evaluate_daily(inputs(near, dates, x, matched=Matched(dates, x)), frozen_before)
    assert unfloored.report["risk_route"]["status"] == "pass"  # the same-sign edge clears a lower bound > 0
    result = benchmark.evaluate_daily(inputs(near, dates, x, matched=Matched(dates, x)), strict_alpha)
    risk = result.report["risk_route"]
    assert result.verdict == Verdict.FAIL and result.report["passed_routes"] == []
    assert risk["status"] == "not_applicable" and risk["tracking"] < RISK["benchmark_risk_min_tracking"]
    assert "matched run up to contract rounding" in risk["reason"] and "calmar_diff_lower" not in risk
    timed = benchmark.evaluate_daily(inputs(np.where(bear, 0.5, 2.0) * x, dates, x, matched=Matched(dates, x)), RISK)
    assert timed.report["risk_route"]["tracking"] > 0.5 and timed.report["risk_route"]["status"] == "pass"


def test_the_risk_route_does_not_apply_without_long_exposure_a_holdable_match_or_an_equity_basis():
    dates, x, _bear = market()
    flat = 0.1 * x + np.random.default_rng(2).normal(0.0002, 0.001, len(x))
    low = benchmark.evaluate_daily(inputs(flat, dates, x, matched=never), RISK)
    assert (
        low.report["risk_route"]["status"] == "not_applicable" and "beta 0.10 <" in low.report["risk_route"]["reason"]
    )

    def cannot(_leverage):
        return benchmark.NotApplicable("0050 is paid in full: 1.54x cannot be held")

    timed = 1.5 * x + 0.002
    held = benchmark.evaluate_daily(inputs(timed, dates, x, matched=cannot), RISK)
    assert held.passed and held.report["passed_routes"] == ["alpha"]
    assert held.report["risk_route"]["reason"] == "0050 is paid in full: 1.54x cannot be held"
    assert "risk route n/a (0050 is paid in full" in held.headline
    twd = benchmark.evaluate_daily(inputs(timed, dates, x, matched=never, basis="twd"), RISK)
    assert twd.report["risk_route"]["status"] == "not_applicable"


def test_a_rule_without_its_settings_or_an_unknown_rule_is_invalid():
    dates, x, _bear = market()
    unknown = benchmark.evaluate_daily(inputs(x, dates, x), {**RISK, benchmark.RULE_KEY: "calmar_only"})
    assert unknown.reason_code == "invalid_input" and "calmar_only" in unknown.reason
    partial = {k: v for k, v in RISK.items() if k != "benchmark_risk_block_days"}
    missing = benchmark.evaluate_daily(inputs(2.0 * x, dates, x, matched=Matched(dates, x)), partial)
    assert missing.reason_code == "invalid_input" and "benchmark_risk_block_days" in missing.reason


def test_the_matched_spec_is_the_first_sleeves_future_or_0050_at_a_holdable_leverage():
    template = tmf_dynamic_leverage_3x()
    spec = beta_matched_spec(template, "TX", 1.11)
    assert spec["instruments"] == {"core": template["instruments"]["core"]}  # TMF with its roll
    sleeve = spec["sleeves"][0]
    assert sleeve["leverage"] == [{"component": "leverage.base", "params": {"value": 1.11}}]
    assert sleeve["rebalance"]["component"] == "rebalance.daily" and not sleeve["hedges"]
    assert spec["execution"] == template["execution"] and spec["capital"] == template["capital"]
    over = beta_matched_spec(template, "TX", 5.5)
    assert isinstance(over, benchmark.NotApplicable) and "cannot be held" in over.reason
    rotation = etf_rotation_00631l_0050()
    stock = beta_matched_spec(rotation, "0050", 1.54)
    assert stock == benchmark.NotApplicable("0050 is paid in full: 1.54x cannot be held")
    assert beta_matched_spec(rotation, "0050", 0.8)["instruments"] == {"base": {"symbol": "0050"}}


# ----------------------------------------------------------------------------- the daily stage on the engine


AS_OF = date(2026, 10, 22)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    return build_long_store(tmp_path_factory.mktemp("planted"), kind="ma_band", seed=5)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return build_long_store(tmp_path_factory.mktemp("plain"), kind="drift", seed=5)


def judge(store, spec: dict, params: dict, thresholds: dict = RISK):
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
    stage = DailyBenchmarkStage(refdata=lambda: store)
    ctx = SimpleNamespace(config=config, upstream=upstream)
    return stage, ctx, stage.evaluate(ctx)


def test_the_stage_matches_the_measured_beta_on_the_strategys_own_contract(planted):
    stage, ctx, result = judge(planted, planted_edge_spec(), {"ma_days": 200, "band": 0.02})
    gate = result.report["gate_report"]
    risk = gate["risk_route"]
    assert risk["matched_leverage"] == round(gate["beta"], 2) and risk["status"] in ("pass", "fail")
    assert gate["rule"] == benchmark.RULE_ALPHA_OR_RISK and "matched_costs" in risk
    runner = stage.spec_runner(ctx, "train")
    matched = beta_matched_run(runner, "TX", risk["matched_leverage"])
    underlying = underlying_returns(underlying_daily(runner.store, "TX", through=runner.end)["close"])
    held = benchmark.regress_daily(matched, underlying, basis="return", lags=5)
    assert held["beta"] == pytest.approx(risk["matched_leverage"], abs=0.05)  # it holds the measured beta
    assert matched.costs.total_twd > 0 and not matched.event_days(["forced_liquidation", "ruin"])


def test_a_matched_run_of_too_few_contracts_is_not_judged_by_the_risk_route(planted):
    coarse = planted_edge_spec()
    coarse["instruments"]["core"]["symbol"] = "TX"  # 20 TMF each: 1M TWD holds one or two near the measured beta
    params = {"ma_days": 200, "band": 0.02}
    _stage, _ctx, result = judge(planted, coarse, params)
    risk = result.report["gate_report"]["risk_route"]
    assert risk["status"] == "not_applicable" and risk["tracking"] is None
    assert f"fewer than {RISK[benchmark.MIN_CONTRACTS_KEY]}: contract rounding" in risk["reason"]
    before = {k: v for k, v in RISK.items() if k != benchmark.MIN_CONTRACTS_KEY}  # a config frozen before the key
    _stage, _ctx, judged = judge(planted, coarse, params, before)
    assert judged.report["gate_report"]["risk_route"]["status"] in ("pass", "fail")
    _stage, _ctx, fine = judge(planted, planted_edge_spec(), params)  # TMF: tens of contracts, judged as before
    assert fine.report["gate_report"]["risk_route"]["status"] in ("pass", "fail")


def test_the_stage_passes_the_planted_regime_and_fails_static_leverage(planted, plain):
    _stage, _ctx, timed = judge(planted, planted_edge_spec(), {"ma_days": 200, "band": 0.02})
    assert timed.verdict == Verdict.PASS and "risk" in timed.report["gate_report"]["passed_routes"]
    _stage, _ctx, static = judge(plain, static_leverage_spec(), {"lev": 2.5})
    gate = static.report["gate_report"]
    assert static.verdict == Verdict.FAIL and gate["passed_routes"] == []
    risk = gate["risk_route"]
    assert risk["status"] == "not_applicable" and risk["matched_leverage"] == pytest.approx(2.5, abs=0.1)
    assert risk["tracking"] < RISK["benchmark_risk_min_tracking"]
