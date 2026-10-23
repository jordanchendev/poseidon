"""Step-3 validation gates: pure verdicts on synthetic backtest artifacts."""

from __future__ import annotations

import itertools
import math
import types
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from poseidon.decision_loop.manifest import canonical_json
from poseidon.pipeline.gates import (
    GATES,
    cost_stress,
    dsr,
    holdout,
    param_stability,
    regime_split,
    summarize,
    summary_line,
    wfe,
)
from poseidon.pipeline.gates.base import CostBreakdown, ResultSummary, Trade
from poseidon.pipeline.plan import default_config
from poseidon.pipeline.state import Verdict

NAN = float("nan")
T = default_config(stage_plan=None, as_of=date(2026, 10, 22))["thresholds"]


def check(result, verdict, code):
    assert (result.verdict, result.reason_code) == (Verdict(verdict), code), (result.reason, result.report)
    canonical_json(result.report)  # reports must be finite JSON (artifact_json)
    return result


def with_thresholds(**overrides):
    return {**T, **overrides}


def test_registry_and_frozen_thresholds():
    assert list(GATES) == ["wfe", "holdout", "cost_stress", "dsr", "param_stability", "regime_split", "benchmark"]
    expected = {
        "wfe_min": 0.5,
        "holdout_sharpe_decay_max": 0.5,
        "holdout_min_trades": 20,
        "holdout_se_z": 1.96,
        "benchmark_alpha_t_min": 1.0,
        "holdout_alpha_positive": True,
        "cost_stress_multiplier": 2.0,
        "dsr_min": 0.95,
        "dsr_min_obs": 30,
        "param_neighbourhood_profitable_min": 0.6,
        "param_min_neighbours": 2,
        "regime_min_trades_per_segment": 5,
        "regime_min_segments": 3,
        "regime_profitable_segments_min": 3,
        "regime_trend_sma_days": 60,
        "regime_atr_days": 20,
        "regime_vol_median_days": 250,
    }
    assert {key: T[key] for key in expected} == expected


# ------------------------------------------------------------------------ wfe


def window(index, is_ret, oos_ret, oos_pnl=1_000.0, **kw):
    return wfe.WindowMetrics(index, is_ret, oos_ret, kw.get("is_pnl", 5_000.0), oos_pnl, 30, kw.get("oos_trades", 8))


def test_wfe_pass_reports_sorted_per_window_table():
    result = check(wfe.evaluate(wfe.WfeInputs([window(1, 0.3, 0.2), window(0, 0.1, 0.1)]), T), "pass", "pass")
    assert result.report["wfe"] == pytest.approx(0.75)
    assert [row["index"] for row in result.report["windows"]] == [0, 1]
    assert result.report["windows"][1]["wfe"] == pytest.approx(2 / 3)
    assert result.report["aggregate_oos_total_pnl_twd"] == 2_000.0
    assert summary_line(result) == "wfe ✅ WFE 0.75"


def test_wfe_boundary_exactly_at_min_passes():
    check(wfe.evaluate(wfe.WfeInputs([window(0, 0.2, 0.1)]), T), "pass", "pass")
    check(wfe.evaluate(wfe.WfeInputs([window(0, 0.2, 0.0999)]), T), "fail", "wfe_below_min")


def test_wfe_uses_compute_wfe_semantics_for_non_positive_is():
    result = check(wfe.evaluate(wfe.WfeInputs([window(0, -0.1, 0.3)]), T), "fail", "wfe_below_min")
    assert result.report["wfe"] == 0.0 and "not positive" in result.reason


def test_wfe_requires_positive_aggregate_oos_pnl():
    windows = [window(0, 0.2, 0.2, oos_pnl=500.0), window(1, 0.2, 0.2, oos_pnl=-500.0)]
    check(wfe.evaluate(wfe.WfeInputs(windows), T), "fail", "oos_not_profitable")  # sum == 0 is not > 0


@pytest.mark.parametrize("windows", [[], [window(0, NAN, 0.1)], [window(0, 0.2, 0.1, oos_pnl=math.inf)]])
def test_wfe_empty_or_nan_inputs_fail_closed(windows):
    check(wfe.evaluate(wfe.WfeInputs(windows), T), "fail", "invalid_input")


# -------------------------------------------------------------------- holdout


def summary(pnl=10_000.0, sharpe=1.0, trades=40, costs=None):
    return ResultSummary(pnl, sharpe, trades, costs or CostBreakdown())


# Criterion (b): holdout_sharpe >= train_sharpe - z * sqrt((1 + holdout_sharpe^2 / 2) / T).
# DECAY_ONLY: the thresholds of a run frozen before ``holdout_se_z`` existed (so also before the alpha keys).
NEWER_KEYS = ("holdout_se_z", "holdout_alpha_positive", "benchmark_alpha_t_min")
DECAY_ONLY = {key: value for key, value in T.items() if key not in NEWER_KEYS}


# A positive holdout alpha over buy-and-hold (new runs require one: holdout_alpha_positive).
ALPHA_OK = {"alpha_twd_per_day": 500.0, "alpha_t_hac": 2.0, "beta": 0.3}


def inputs(train, held, n_obs=348, benchmark=ALPHA_OK):
    return holdout.HoldoutInputs(train, held, n_obs, benchmark)


def test_holdout_pass_and_trade_boundary():
    result = check(holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=1.5, trades=20)), T), "pass", "pass")
    assert result.report["sharpe_decay"] == pytest.approx(0.25)


def test_holdout_decay_boundary_exactly_at_max_passes():
    check(holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=1.0)), T), "pass", "pass")
    check(holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=0.98)), T), "fail", "sharpe_decay")


def test_holdout_decay_headline_matches_telegram_format():
    result = check(
        holdout.evaluate(inputs(summary(sharpe=1.0), summary(sharpe=0.38)), DECAY_ONLY), "fail", "sharpe_decay"
    )
    assert summary_line(result) == "holdout ❌ Sharpe decay 62%"


@pytest.mark.parametrize("train_sharpe", [0.0, -0.4])
def test_holdout_non_positive_train_sharpe_fails(train_sharpe):
    result = holdout.evaluate(inputs(summary(sharpe=train_sharpe), summary(sharpe=0.5)), T)
    assert check(result, "fail", "train_not_positive").report["sharpe_decay"] is None


def test_holdout_insufficient_trades():
    check(holdout.evaluate(inputs(summary(), summary(trades=19)), T), "fail", "insufficient_trades")


def test_holdout_requires_positive_pnl():
    check(holdout.evaluate(inputs(summary(), summary(pnl=0.0)), T), "fail", "holdout_not_profitable")
    relaxed = with_thresholds(holdout_pnl_positive=False)
    check(holdout.evaluate(inputs(summary(), summary(pnl=-1.0)), relaxed), "pass", "pass")


@pytest.mark.parametrize(
    ("train", "held"),
    [
        (summary(sharpe=NAN), summary()),
        (summary(), summary(sharpe=NAN)),
        (summary(), summary(pnl=NAN)),
        (summary(), summary(trades=None)),
    ],
)
def test_holdout_nan_inputs_fail_closed(train, held):
    check(holdout.evaluate(inputs(train, held), T), "fail", "invalid_input")


def held_out(train_sharpe, holdout_sharpe, n_obs=348, thresholds=None, **held):
    pair = inputs(summary(sharpe=train_sharpe), summary(sharpe=holdout_sharpe, **held), n_obs)
    return holdout.evaluate(pair, T if thresholds is None else thresholds)


def test_holdout_sharpe_se_is_lo_2002():
    assert holdout.sharpe_se(0.0, 100) == pytest.approx(0.1)
    assert holdout.sharpe_se(0.032, 348) == pytest.approx(math.sqrt((1 + 0.032**2 / 2) / 348))


def test_holdout_decay_pass_reports_both_criteria():
    # Large Sharpe: decay passes while the gap is far beyond z * SE.
    result = check(held_out(2.0, 1.5), "pass", "pass")
    assert result.report["passed_criterion"] == "decay"
    assert summary_line(result) == "holdout ✅ Sharpe decay 25%, alpha +500/day (t 2.00), beta 0.30"
    se = result.report["criteria"]["se"]
    assert se["enabled"] and not se["passed"] and se["n_obs"] == 348 and se["z"] == 1.96
    # Small Sharpe: both pass; decay is reported as the deciding criterion.
    both = check(held_out(0.2, 0.15), "pass", "pass")
    assert both.report["passed_criterion"] == "decay" and both.report["criteria"]["se"]["passed"]
    assert result.report["criteria"]["decay"] == {"value": pytest.approx(0.25), "max": 0.5, "passed": True}


def test_holdout_decay_fail_but_within_se_passes():
    # Weak-edge sweep drift 6 seed 11: train 0.134, holdout 0.032 over 348 trading days.
    result = check(held_out(0.134, 0.032), "pass", "pass")
    se = result.report["criteria"]["se"]
    assert result.report["passed_criterion"] == "se" and not result.report["criteria"]["decay"]["passed"]
    assert se["se"] == pytest.approx(0.053619, abs=1e-6) and se["delta"] == pytest.approx(0.102)
    assert se["allowed_delta"] == pytest.approx(1.96 * se["se"]) and se["margin"] == pytest.approx(0.003094, abs=1e-6)
    assert summary_line(result) == (
        "holdout ✅ decay 76% but within 1.96·SE (Δ 0.102 ≤ 0.105), alpha +500/day (t 2.00), beta 0.30"
    )


def test_holdout_both_criteria_fail():
    result = check(held_out(1.0, 0.38), "fail", "sharpe_decay")
    assert result.report["passed_criterion"] is None and result.report["criteria"]["se"]["passed"] is False
    assert summary_line(result) == "holdout ❌ Sharpe decay 62%, beyond 1.96·SE (Δ 0.620 > 0.109)"


@pytest.mark.parametrize(("train_sharpe", "holdout_sharpe"), [(2.0, 1.5), (0.134, 0.032)])
def test_holdout_non_positive_pnl_fails_whichever_criterion_passes(train_sharpe, holdout_sharpe):
    check(held_out(train_sharpe, holdout_sharpe, pnl=0.0), "fail", "holdout_not_profitable")
    check(held_out(train_sharpe, holdout_sharpe, pnl=-5.0), "fail", "holdout_not_profitable")


def test_holdout_se_boundary_exactly_at_z_se_passes():
    # holdout Sharpe 0 over T=100 -> SE = 0.1; z = 2 -> allowed delta 0.2 exactly.
    z2 = with_thresholds(holdout_se_z=2.0)
    result = check(held_out(0.2, 0.0, n_obs=100, thresholds=z2), "pass", "pass")
    assert result.report["criteria"]["se"]["margin"] == 0.0 and result.report["passed_criterion"] == "se"
    check(held_out(0.2 + 1e-9, 0.0, n_obs=100, thresholds=z2), "fail", "sharpe_decay")


def test_holdout_small_t_widens_se():
    wide = check(held_out(1.0, 0.2, n_obs=4), "pass", "pass").report["criteria"]["se"]
    assert wide["se"] == pytest.approx(math.sqrt(1.02 / 4)) and wide["allowed_delta"] > 0.8
    check(held_out(1.0, 0.2, n_obs=348), "fail", "sharpe_decay")
    # Fewer than 2 observations has no Sharpe; criterion (b) fails closed.
    check(held_out(1.0, 0.2, n_obs=1), "fail", "invalid_input")


def test_holdout_without_se_z_keeps_decay_only_behaviour():
    result = check(held_out(0.134, 0.032, thresholds=DECAY_ONLY), "fail", "sharpe_decay")
    assert summary_line(result) == "holdout ❌ Sharpe decay 76%"
    assert result.report["criteria"]["se"] == {
        "enabled": False,
        "passed": None,
        "note": "holdout_se_z not frozen in this run",
    }
    # n_obs is not needed for decay-only runs (the inputs of runs frozen before the key).
    check(holdout.evaluate(holdout.HoldoutInputs(summary(sharpe=2.0), summary(sharpe=1.5)), DECAY_ONLY), "pass", "pass")


# Runs frozen before CONFIG_VERSION 5 (e.g. run #6) have neither holdout_alpha_positive nor holdout_se_z.
NO_ALPHA = DECAY_ONLY


def test_holdout_requires_positive_alpha_last():
    negative = {"alpha_twd_per_day": -1.0, "alpha_t_hac": -0.1, "beta": 0.9}
    result = check(
        holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=1.5), benchmark=negative), T),
        "fail",
        "holdout_alpha_not_positive",
    )
    assert result.report["alpha_positive_required"] and result.report["benchmark"] == negative
    assert summary_line(result) == "holdout ❌ holdout alpha -1/day (t -0.10), beta 0.90"
    zero = {**negative, "alpha_twd_per_day": 0.0}
    check(
        holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=1.5), benchmark=zero), T),
        "fail",
        "holdout_alpha_not_positive",
    )
    # the earlier criteria still decide first
    check(
        holdout.evaluate(inputs(summary(), summary(pnl=0.0), benchmark=negative), T), "fail", "holdout_not_profitable"
    )
    check(holdout.evaluate(inputs(summary(), summary(trades=3), benchmark=negative), T), "fail", "insufficient_trades")


@pytest.mark.parametrize("benchmark", [None, {"alpha_twd_per_day": None, "alpha_t_hac": None, "beta": None}])
def test_holdout_alpha_required_but_missing_fails_closed(benchmark):
    check(holdout.evaluate(inputs(summary(), summary(), benchmark=benchmark), T), "fail", "invalid_input")


def test_holdout_without_alpha_key_reports_alpha_but_does_not_gate():
    negative = {"alpha_twd_per_day": -2_000.0, "alpha_t_hac": -1.5, "beta": 1.1}
    result = check(
        holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=1.5), benchmark=negative), NO_ALPHA), "pass", "pass"
    )
    assert result.report["alpha_positive_required"] is False and result.report["benchmark"] == negative
    assert summary_line(result) == "holdout ✅ Sharpe decay 25%, alpha -2,000/day (t -1.50), beta 1.10"
    legacy = check(
        holdout.evaluate(inputs(summary(sharpe=2.0), summary(sharpe=1.5), benchmark=None), NO_ALPHA), "pass", "pass"
    )
    assert legacy.report["benchmark"] is None and summary_line(legacy) == "holdout ✅ Sharpe decay 25%"


@pytest.mark.parametrize(
    ("n_obs", "thresholds"),
    [
        (None, T),
        (NAN, T),
        (math.inf, T),
        (348, with_thresholds(holdout_se_z=NAN)),
        (348, with_thresholds(holdout_se_z=-1.0)),
    ],
)
def test_holdout_se_criterion_nan_inputs_fail_closed(n_obs, thresholds):
    check(held_out(0.134, 0.032, n_obs=n_obs, thresholds=thresholds), "fail", "invalid_input")


# ---------------------------------------------------------------- cost_stress

BASE_COSTS = CostBreakdown(commission_twd=1_000.0, tax_twd=500.0, slippage_twd=1_500.0)
STRESSED_COSTS = CostBreakdown(commission_twd=2_000.0, tax_twd=1_000.0, slippage_twd=3_000.0)


def stress(base_pnl=10_000.0, stressed_pnl=4_000.0, multiplier=2.0, stressed_costs=STRESSED_COSTS):
    return cost_stress.CostStressInputs(
        summary(base_pnl, costs=BASE_COSTS), summary(stressed_pnl, costs=stressed_costs), multiplier
    )


def test_cost_stress_pass_reports_breakdown_and_erosion():
    result = check(cost_stress.evaluate(stress(), T), "pass", "pass")
    assert result.report["pnl_erosion_pct"] == pytest.approx(60.0)
    assert result.report["stressed_costs"]["total_twd"] == 6_000.0
    assert result.report["base_costs"] == {
        "commission_twd": 1000.0,
        "tax_twd": 500.0,
        "slippage_twd": 1500.0,
        "total_twd": 3000.0,
    }
    assert summary_line(result) == "cost_stress ✅ costs x2 PnL 4,000 (erosion 60%)"


@pytest.mark.parametrize("stressed_pnl", [0.0, -2_000.0])
def test_cost_stress_boundary_zero_and_loss_fail(stressed_pnl):
    check(cost_stress.evaluate(stress(stressed_pnl=stressed_pnl), T), "fail", "stressed_not_profitable")
    check(cost_stress.evaluate(stress(stressed_pnl=0.01), T), "pass", "pass")


def test_cost_stress_rejects_a_run_built_with_another_multiplier():
    check(cost_stress.evaluate(stress(multiplier=1.5), T), "fail", "multiplier_mismatch")


def test_cost_stress_zero_base_pnl_has_no_erosion():
    result = check(cost_stress.evaluate(stress(base_pnl=0.0, stressed_pnl=-10.0), T), "fail", "stressed_not_profitable")
    assert result.report["pnl_erosion_pct"] is None


@pytest.mark.parametrize(
    "inputs",
    [stress(stressed_pnl=NAN), stress(multiplier=NAN), stress(stressed_costs=CostBreakdown(slippage_twd=math.inf))],
)
def test_cost_stress_nan_inputs_fail_closed(inputs):
    check(cost_stress.evaluate(inputs, T), "fail", "invalid_input")


# ------------------------------------------------------------------------ dsr

# Bailey & Lopez de Prado (2014), section "A numerical example": annualized
# SR 2.5 over T = 1250 daily obs, skew -3, (non-excess) kurtosis 10, N = 100
# trials whose annualized Sharpe variance is 0.5. Per-period values:
#   SR = 2.5/sqrt(250) = 0.158113883, V = 0.5/250 = 0.002
#   SR0 = sqrt(V)*((1-g)*Phi^-1(0.99) + g*Phi^-1(1 - 1/(100e))) = 0.113172002
#   denominator = 1 + 3*SR + (10-1)/4*SR^2 = 1.530591649
#   z = (SR - SR0)*sqrt(1249)/sqrt(denominator) = 1.283816037 -> DSR = 0.900396834
# The paper reports SR0 ~ 0.1132 and DSR ~ 0.9004.
REF_SR = 2.5 / math.sqrt(250)
REF_V = 0.5 / 250


def trials_with_variance(n: int, variance: float) -> list[float]:
    """n (even) Sharpes, half at +a and half at -a, with sample variance ``variance``."""

    a = math.sqrt(variance * (n - 1) / n)
    return [a] * (n // 2) + [-a] * (n // 2)


def test_dsr_matches_the_paper_reference_example():
    assert dsr.expected_max_sharpe(100, REF_V) == pytest.approx(0.113172002, abs=1e-9)
    assert dsr.deflated_sharpe(REF_SR, 1250, -3.0, 10.0, 0.113172002) == pytest.approx(0.900396834, abs=1e-8)
    inputs = dsr.DsrInputs(REF_SR, 1250, -3.0, 10.0, trials_with_variance(100, REF_V))
    result = check(dsr.evaluate(inputs, T), "fail", "dsr_below_min")
    assert result.report["trial_sharpe_variance"] == pytest.approx(REF_V)
    assert result.report["sr0"] == pytest.approx(0.113172002, abs=1e-9)
    assert result.report["dsr"] == pytest.approx(0.900396834, abs=1e-8)
    assert summary_line(result) == "dsr ❌ DSR 0.90 (N=100)"


def test_dsr_kurtosis_is_non_excess():
    sr, n_obs = 0.1, 500
    psr_normal = dsr.deflated_sharpe(sr, n_obs, 0.0, 3.0, 0.0)
    assert psr_normal == pytest.approx(
        0.5 * math.erfc(-(sr * math.sqrt(n_obs - 1) / math.sqrt(1 + sr**2 / 2)) / math.sqrt(2))
    )
    # Passing excess kurtosis (0.0 for normal returns) is caught, not silently used.
    check(dsr.evaluate(dsr.DsrInputs(sr, n_obs, 0.0, 0.0, [sr]), T), "fail", "invalid_moments")


def test_dsr_single_trial_is_psr_and_can_pass():
    result = check(dsr.evaluate(dsr.DsrInputs(0.1, 1250, 0.0, 3.0, [0.1]), T), "pass", "pass")
    assert result.report["sr0"] == 0.0
    assert result.report["dsr"] == pytest.approx(dsr.deflated_sharpe(0.1, 1250, 0.0, 3.0, 0.0))


def test_dsr_more_trials_means_lower_dsr():
    values = [
        dsr.deflated_sharpe(0.12, 1250, -0.5, 5.0, dsr.expected_max_sharpe(n, 0.001)) for n in (1, 2, 5, 10, 100, 1000)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(values)), values


def test_dsr_errored_trials_count_toward_n():
    finite = trials_with_variance(10, 0.0005)
    clean = dsr.evaluate(dsr.DsrInputs(0.12, 1250, 0.0, 3.0, finite), T)
    with_errors = dsr.evaluate(dsr.DsrInputs(0.12, 1250, 0.0, 3.0, finite + [None] * 30 + [NAN]), T)
    assert with_errors.report["n_trials"] == 41 and with_errors.report["n_trials_with_sharpe"] == 10
    assert with_errors.report["trial_sharpe_variance"] == clean.report["trial_sharpe_variance"]
    assert with_errors.report["dsr"] < clean.report["dsr"]


def test_dsr_ledger_rows_keep_every_state():
    rows = [
        types.SimpleNamespace(state=state, sharpe=sharpe)
        for state, sharpe in [("complete", 0.1), ("fail", None), ("pruned", 0.02)]
    ]
    assert dsr.trial_sharpes_from_ledger(rows) == [0.1, None, 0.02]


def test_dsr_boundary_exactly_at_min_passes():
    inputs = dsr.DsrInputs(0.1, 400, 0.0, 3.0, trials_with_variance(4, 0.0004))
    value = dsr.evaluate(inputs, T).report["dsr"]
    check(dsr.evaluate(inputs, with_thresholds(dsr_min=value)), "pass", "pass")
    check(dsr.evaluate(inputs, with_thresholds(dsr_min=math.nextafter(value, 1.0))), "fail", "dsr_below_min")


def test_dsr_observation_guard():
    check(dsr.evaluate(dsr.DsrInputs(0.5, 29, 0.0, 3.0, [0.5]), T), "fail", "insufficient_observations")
    check(dsr.evaluate(dsr.DsrInputs(0.5, 30, 0.0, 3.0, [0.5]), T), "pass", "pass")


def test_dsr_denominator_guard():
    # skew 2, SR 1, kurtosis 5 (= 1 + skew^2): 1 - 2 + 4/4 = 0
    check(dsr.evaluate(dsr.DsrInputs(1.0, 100, 2.0, 5.0, [1.0]), T), "fail", "invalid_denominator")
    with pytest.raises(ValueError):
        dsr.deflated_sharpe(1.0, 100, 2.0, 5.0, 0.0)


@pytest.mark.parametrize(
    ("inputs", "code"),
    [
        (dsr.DsrInputs(NAN, 1250, 0.0, 3.0, [0.1]), "invalid_input"),
        (dsr.DsrInputs(0.1, 1250, NAN, 3.0, [0.1]), "invalid_input"),
        (dsr.DsrInputs(0.1, 1250, 0.0, 3.0, []), "no_trials"),
        (dsr.DsrInputs(0.1, 1250, 0.0, 3.0, [0.1, None, NAN]), "insufficient_trial_sharpes"),
    ],
)
def test_dsr_nan_or_empty_inputs_fail_closed(inputs, code):
    check(dsr.evaluate(inputs, T), "fail", code)


# ------------------------------------------------------------ param_stability

GRID = {"fast": [5, 10, 15, 20], "slow": [40, 60], "stop": [1.0, 1.5, 2.0]}
BEST = {"fast": 5, "slow": 60, "stop": 1.5}


def test_one_step_neighbours_marks_out_of_bounds():
    neighbours = param_stability.one_step_neighbours(BEST, GRID)
    assert neighbours == [
        ("fast", -1, None),
        ("fast", 1, {"fast": 10, "slow": 60, "stop": 1.5}),
        ("slow", -1, {"fast": 5, "slow": 40, "stop": 1.5}),
        ("slow", 1, None),
        ("stop", -1, {"fast": 5, "slow": 60, "stop": 1.0}),
        ("stop", 1, {"fast": 5, "slow": 60, "stop": 2.0}),
    ]
    with pytest.raises(ValueError):
        param_stability.one_step_neighbours({**BEST, "fast": 7}, GRID)


def stability(*pnls, best_pnl=50_000.0):
    neighbours = [
        param_stability.NeighbourResult(name, step, params or {}, None if params is None else pnl)
        for (name, step, params), pnl in zip(param_stability.one_step_neighbours(BEST, GRID), pnls, strict=True)
    ]
    return param_stability.ParamStabilityInputs(BEST, best_pnl, neighbours)


def test_param_stability_share_boundary():
    # fast-1 and slow+1 are out of bounds: None regardless of the value given
    result = check(param_stability.evaluate(stability(0, 900, 800, 0, 0.0, -100), T), "fail", "unstable_neighbourhood")
    assert (result.report["evaluated"], result.report["profitable"], result.report["excluded"]) == (4, 2, 2)
    six = {**GRID, "fast": [0, 5, 10]}  # every slot in bounds except slow+1 -> 5 evaluated
    neighbours = [
        param_stability.NeighbourResult(name, step, params or {}, None if params is None else pnl)
        for (name, step, params), pnl in zip(
            param_stability.one_step_neighbours(BEST, six), [100, 200, 300, 0, -1, -2], strict=True
        )
    ]
    passed = check(
        param_stability.evaluate(param_stability.ParamStabilityInputs(BEST, 1.0, neighbours), T), "pass", "pass"
    )
    assert passed.report["profitable_share"] == pytest.approx(0.6)
    assert summary_line(passed) == "param_stability ✅ 3/5 neighbours profitable (60%)"


def test_param_stability_two_evaluated_neighbours_is_enough():
    check(param_stability.evaluate(stability(0, 10, 10, 0, None, None), T), "pass", "pass")


@pytest.mark.parametrize("pnls", [(0, 10, None, 0, None, None), (0, None, None, 0, None, None)])
def test_param_stability_insufficient_neighbours(pnls):
    check(param_stability.evaluate(stability(*pnls), T), "fail", "insufficient_neighbours")


def test_param_stability_empty_and_nan():
    check(
        param_stability.evaluate(param_stability.ParamStabilityInputs(BEST, 1.0, []), T),
        "fail",
        "insufficient_neighbours",
    )
    check(param_stability.evaluate(stability(0, NAN, 10, 0, 10, 10), T), "fail", "invalid_input")
    check(param_stability.evaluate(stability(0, 10, 10, 0, 10, 10, best_pnl=NAN), T), "fail", "invalid_input")


# --------------------------------------------------------------- regime_split


def daily_frame(n=400, seed=7, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    close = 17_000 + np.cumsum(rng.normal(0, 120, n))
    spread = np.abs(rng.normal(0, 80, n)) * np.where(np.arange(n) > n * 0.7, 3.0, 1.0)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": close - rng.normal(0, 30, n), "high": close + spread, "low": close - spread, "close": close},
        index=index,
    )


def test_regime_states_warm_up_and_known_regimes():
    states = regime_split.regime_states(daily_frame())
    first = next(i for i, value in enumerate(states) if value is not None)
    assert first == 269  # TR needs a prior close (1) + ATR 20 (-> 20) + 250-bar median (-> 269); SMA 60 earlier
    assert set(states.iloc[first:]) <= set(regime_split.BUCKETS)
    up = pd.DataFrame({"close": np.linspace(100, 200, 300)}, index=pd.bdate_range("2024-01-01", periods=300))
    up["high"], up["low"] = up.close + 1, up.close - 1
    assert regime_split.regime_states(up, median_days=20).iloc[-1].startswith("bull")
    assert regime_split.regime_states(up.iloc[::-1].set_axis(up.index), median_days=20).iloc[-1].startswith("bear")


def test_regime_states_have_no_look_ahead():
    frame = daily_frame()
    full = regime_split.regime_states(frame)
    for cut in (270, 300, 333, 399):
        truncated = regime_split.regime_states(frame.iloc[:cut])
        pd.testing.assert_series_equal(truncated, full.iloc[:cut])


def test_regime_label_uses_only_prior_completed_days():
    frame = daily_frame()
    entry_day = frame.index[350]
    entry = datetime.combine(entry_day.date(), datetime.min.time()).replace(hour=9)
    states = regime_split.regime_states(frame).to_dict()
    dates = sorted(states)
    label = regime_split.label_for_entry(entry, dates, states)
    assert label == states[frame.index[349].date()]
    shocked = frame.copy()
    factor = 0.3 if label.startswith("bull") else 3.0  # flip the trend from the entry day on
    shocked.loc[entry_day:, ["open", "high", "low", "close"]] *= factor
    shocked_states = regime_split.regime_states(shocked).to_dict()
    assert shocked_states[entry_day.date()] != label  # the shock is visible on the entry day itself
    assert regime_split.label_for_entry(entry, dates, shocked_states) == label


def test_regime_entry_date_is_taipei_and_strictly_after_the_state_bar():
    states = {date(2026, 1, 5): "bull_low_vol", date(2026, 1, 6): "bear_high_vol"}
    dates = sorted(states)
    late_utc = datetime(2026, 1, 5, 20, 0, tzinfo=UTC)  # 04:00 on 2026-01-06 in Taipei
    assert regime_split.label_for_entry(late_utc, dates, states) == "bull_low_vol"
    assert regime_split.label_for_entry(datetime(2026, 1, 6, 13, 0), dates, states) == "bull_low_vol"
    assert regime_split.label_for_entry(datetime(2026, 1, 5, 9, 0), dates, states) is None  # no prior bar


STATE_DAYS = {
    date(2025, 3, 3): "bull_high_vol",
    date(2025, 3, 10): "bull_low_vol",
    date(2025, 3, 17): "bear_high_vol",
    date(2025, 3, 24): "bear_low_vol",
}


def regime_trades(plan):
    """plan: {bucket: [pnl, ...]} -> trades entering the day after the bucket's state bar."""

    by_bucket = {label: day for day, label in STATE_DAYS.items()}
    trades = []
    for bucket, pnls in plan.items():
        entry = datetime.combine(by_bucket[bucket] + timedelta(days=1), datetime.min.time()).replace(hour=9, minute=30)
        trades += [Trade(entry, entry + timedelta(hours=2), "long", pnl, 100.0) for pnl in pnls]
    return trades


def regime(plan):
    return regime_split.RegimeInputs(regime_trades(plan), STATE_DAYS)


def test_regime_pass_three_profitable_one_loser():
    plan = {"bull_high_vol": [100] * 5, "bull_low_vol": [50] * 6, "bear_high_vol": [10] * 5, "bear_low_vol": [-100] * 7}
    result = check(regime_split.evaluate(regime(plan), T), "pass", "pass")
    assert result.report["buckets"]["bear_low_vol"] == {
        "trades": 7,
        "pnl_twd": -700.0,
        "evaluated": True,
        "profitable": False,
    }
    assert summary_line(result) == "regime_split ✅ 3/4 regimes profitable"


def test_regime_fail_when_profits_concentrate():
    plan = {"bull_high_vol": [100] * 5, "bull_low_vol": [50] * 6, "bear_high_vol": [-10] * 5, "bear_low_vol": [0] * 7}
    check(regime_split.evaluate(regime(plan), T), "fail", "regime_concentration")  # a 0-pnl bucket is not profitable


def test_regime_bucket_trade_boundary_and_coverage():
    four = {"bull_high_vol": [100] * 5, "bull_low_vol": [50] * 5, "bear_high_vol": [10] * 4, "bear_low_vol": [10] * 4}
    result = check(regime_split.evaluate(regime(four), T), "fail", "insufficient_coverage")
    assert result.report["evaluated_buckets"] == 2
    five = {**four, "bear_high_vol": [10] * 5}
    check(regime_split.evaluate(regime(five), T), "pass", "pass")  # exactly 3 evaluated, all profitable


def test_regime_empty_unlabelled_and_nan():
    check(regime_split.evaluate(regime_split.RegimeInputs([], STATE_DAYS), T), "fail", "insufficient_coverage")
    early = datetime(2025, 1, 2, 9, 0)
    result = check(
        regime_split.evaluate(regime_split.RegimeInputs([Trade(early, early, "short", 1.0)] * 9, STATE_DAYS), T),
        "fail",
        "insufficient_coverage",
    )
    assert result.report["unlabelled_trades"] == 9
    nan_plan = {"bull_high_vol": [100] * 4 + [NAN]}
    check(regime_split.evaluate(regime(nan_plan), T), "fail", "invalid_input")


def test_regime_threshold_windows_come_from_the_run_config():
    assert regime_split.threshold_windows(T) == {"sma_days": 60, "atr_days": 20, "median_days": 250}


# ------------------------------------------------------------------ summarize


def test_summarize_one_line_per_gate():
    results = [
        wfe.evaluate(wfe.WfeInputs([window(0, 0.2, 0.15)]), T),
        holdout.evaluate(inputs(summary(sharpe=1.0), summary(sharpe=0.38)), T),
        dsr.evaluate(dsr.DsrInputs(NAN, 1, 0.0, 3.0, []), T),
    ]
    assert summarize(results) == [
        "wfe ✅ WFE 0.75",
        "holdout ❌ Sharpe decay 62%, beyond 1.96·SE (Δ 0.620 > 0.109)",
        "dsr ❌ invalid input",
    ]
