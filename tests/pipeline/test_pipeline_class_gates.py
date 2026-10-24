"""Strategy Pipeline v2 class-keyed gates (design 2026-10-24 §7): the threshold table and the daily rules."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline.gates import (
    DAILY_GATES,
    GATES,
    GATES_BY_CLASS,
    benchmark,
    cost_stress,
    dsr,
    execution_timing,
    holdout,
    liquidation,
    param_stability,
    regime_split,
    wfe,
)
from poseidon.pipeline.gates.daily import (
    DailyResult,
    EngineEvent,
    RerunRequest,
    basis_for,
    choose_walk_forward,
    from_engine,
    moments,
    walk_forward_windows,
)
from poseidon.pipeline.optimization import ParamGrid
from poseidon.pipeline.plan import (
    CONFIG_VERSION,
    DEFAULT_GATES,
    DEFAULT_GATES_BY_CLASS,
    DEFAULT_THRESHOLDS,
    DEFAULT_THRESHOLDS_BY_CLASS,
    class_settings,
    default_config,
)
from poseidon.pipeline.state import Verdict
from poseidon.strategies.dsl.v2.schema import ParamSpec

D = DEFAULT_THRESHOLDS_BY_CLASS["daily_equity"]

# Today's (CONFIG_VERSION 5) thresholds, written out: the DSL class must keep them exactly.
TODAY_THRESHOLDS = {
    "wfe_min": 0.5,
    "holdout_sharpe_decay_max": 0.5,
    "holdout_pnl_positive": True,
    "holdout_min_trades": 20,
    "holdout_se_z": 1.96,
    "benchmark_alpha_t_min": 1.0,
    "holdout_alpha_positive": True,
    "cost_stress_multiplier": 2.0,
    "cost_stress_pnl_positive": True,
    "dsr_min": 0.95,
    "dsr_min_obs": 30,
    "param_neighbourhood_profitable_min": 0.6,
    "param_min_neighbours": 2,
    "regime_profitable_segments_min": 3,
    "regime_min_segments": 3,
    "regime_min_trades_per_segment": 5,
    "regime_trend_sma_days": 60,
    "regime_atr_days": 20,
    "regime_vol_median_days": 250,
    "paper_min_trading_days": 20,
    "paper_min_trades": 10,
    "paper_replay_match_min": 0.9,
    "paper_replay_tolerance_ticks": 1,
}
# content_sha256(default_config(stage_plan=None, as_of=2026-10-22)) at fd2294d (before this branch).
DEFAULT_CONFIG_SHA_AT_BASE = "f515e57f7ad0cabc7746fd96a2144810ba3bdb818c9112f6840e8e6b6a47fb25"


def check(result, verdict, code):
    assert (result.verdict, result.reason_code) == (Verdict(verdict), code), (result.reason, result.report)
    canonical_json(result.report)
    return result


def days(n: int, start: date = date(2020, 1, 1)) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, periods=n)]


def daily(returns, *, start: date = date(2020, 1, 1), initial: float = 1_000_000.0, **kw) -> DailyResult:
    """A DailyResult whose daily returns are ``returns`` (costs 0 unless given)."""

    dates = kw.pop("dates", None) or days(len(returns), start)
    equity = initial * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    zeros = (0.0,) * len(dates)
    return DailyResult(
        dates=tuple(dates),
        equity=tuple(float(e) for e in equity),
        initial_equity=initial,
        commission=kw.pop("commission", zeros),
        tax=kw.pop("tax", zeros),
        slippage=kw.pop("slippage", zeros),
        **kw,
    )


def noise(n: int, mean: float, sd: float = 0.01, seed: int = 0) -> np.ndarray:
    values = np.random.default_rng(seed).normal(0.0, sd, n)
    return values - values.mean() + mean  # exact mean


# ----------------------------------------------------------------------------- threshold table


def test_dsl_class_thresholds_equal_todays_values():
    assert DEFAULT_THRESHOLDS_BY_CLASS["intraday_fixed"] is DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS == TODAY_THRESHOLDS
    assert DEFAULT_GATES_BY_CLASS["intraday_fixed"] is DEFAULT_GATES
    settings = class_settings("intraday_fixed")
    config = default_config(stage_plan=None, as_of=date(2026, 10, 22))
    assert settings == {"thresholds": config["thresholds"], "gates": config["gates"]}
    assert GATES_BY_CLASS["intraday_fixed"] is GATES
    assert list(GATES) == ["wfe", "holdout", "cost_stress", "dsr", "param_stability", "regime_split", "benchmark"]
    assert all(GATES[name].__name__ == "evaluate" for name in GATES)


def test_default_config_is_byte_identical_to_the_base():
    config = default_config(stage_plan=None, as_of=date(2026, 10, 22))
    assert config["version"] == CONFIG_VERSION == 5
    assert content_sha256(config) == DEFAULT_CONFIG_SHA_AT_BASE


def test_daily_thresholds_are_the_registered_section_7_numbers():
    assert DEFAULT_THRESHOLDS_BY_CLASS["daily_fixed"] is D
    assert D["holdout_min_days"] == 250
    assert D["regime_min_days_per_segment"] == 60
    assert D["dsr_min_obs"] == 250 and D["dsr_min"] == 0.95
    # owner decisions 2026-10-24 afternoon: daily alpha t >= 1.65 (one-sided 5 %), blocking execution timing
    assert D["benchmark_alpha_t_min"] == 1.65 and D["holdout_alpha_positive"] is True
    assert D["execution_timing_pnl_positive"] is True and D["execution_timing_sharpe_ratio_min"] == 0.5
    assert D["paper_target_match_min"] == 0.95 and D["paper_min_trading_days"] == 20
    assert D["liquidation_fails"] is True
    # trade counts have no meaning for a daily class: those keys are absent (criterion off)
    assert not {"holdout_min_trades", "regime_min_trades_per_segment", "paper_min_trades"} & set(D)
    # everything else carries over unchanged
    for key in ("wfe_min", "holdout_sharpe_decay_max", "holdout_se_z", "cost_stress_multiplier"):
        assert D[key] == DEFAULT_THRESHOLDS[key]
    layouts = DEFAULT_GATES_BY_CLASS["daily_equity"]["wfe"]["layouts"]
    assert layouts == [
        {"windows": 5, "is_days": 1460, "oos_days": 365},
        {"windows": 4, "is_days": 730, "oos_days": 182},
    ]
    assert set(DAILY_GATES) == set(GATES) | {"liquidation", "execution_timing"}
    for execution, scenarios in DEFAULT_GATES_BY_CLASS["daily_equity"]["execution_timing"]["scenarios"].items():
        assert scenarios == list(execution_timing.scenarios_for({}, base_execution=execution))


def test_class_settings_copies_and_refuses_unregistered_classes():
    first = class_settings("daily_equity")
    first["thresholds"]["holdout_min_days"] = 1
    first["gates"]["wfe"]["layouts"].clear()
    assert class_settings("daily_equity")["thresholds"]["holdout_min_days"] == 250
    assert DEFAULT_GATES_BY_CLASS["daily_equity"]["wfe"]["layouts"]
    with pytest.raises(KeyError, match="intraday_equity"):
        class_settings("intraday_equity")
    assert (basis_for("daily_equity"), basis_for("daily_fixed")) == ("return", "twd")
    with pytest.raises(KeyError):
        basis_for("intraday_fixed")


def test_the_acceptance_spec_class_and_the_dsl_class_have_entries():
    from poseidon.pipeline.v3.dsl_bridge import v2_to_v3
    from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
    from poseidon.pipeline.v3.spec import strategy_class
    from poseidon.pipeline.v3.validation import validate_spec_v3
    from poseidon.strategies.dsl.v2.examples import tx_orb30_day

    assert validate_spec_v3(tmf_dynamic_leverage_3x()).strategy_class == "daily_equity"
    assert strategy_class(v2_to_v3(tx_orb30_day())) == "intraday_fixed"
    assert set(DEFAULT_THRESHOLDS_BY_CLASS) == {"intraday_fixed", "daily_equity", "daily_fixed"}


# ----------------------------------------------------------------------------- DailyResult


def test_daily_result_returns_windows_and_costs():
    result = daily(
        [0.01, -0.02, 0.03, 0.0],
        commission=(10.0, 0.0, 5.0, 0.0),
        tax=(1.0, 0.0, 1.0, 0.0),
        slippage=(2.0, 0.0, 2.0, 0.0),
        fill_days=frozenset(days(4)[i] for i in (0, 2)),
        events=(EngineEvent(days(4)[1], "margin_call"),),
    )
    np.testing.assert_allclose(result.returns, [0.01, -0.02, 0.03, 0.0])
    assert result.exposure_changes == 2 and result.costs.total_twd == 21.0
    part = result.window(days(4)[1], days(4)[3])
    assert part.dates == tuple(days(4)[1:3]) and part.initial_equity == pytest.approx(1_010_000.0)
    np.testing.assert_allclose(part.returns, [-0.02, 0.03])
    assert part.exposure_changes == 1 and part.costs.commission_twd == 5.0 and len(part.events) == 1
    assert result.window(date(2030, 1, 1)).n_days == 0


def test_moments_match_backtesting_daily_stats():
    from poseidon.pipeline.backtesting import daily_stats

    values = noise(300, 0.0007, seed=4) + np.random.default_rng(5).standard_t(4, 300) * 0.002
    ours, theirs = moments(values), daily_stats(pd.Series(values), 252)
    for key in ("n_obs", "sharpe_daily", "sharpe_annualised", "skew", "kurtosis"):
        assert ours[key] == pytest.approx(theirs[key], rel=1e-12), key
    assert moments([0.01, 0.01, 0.01])["sharpe_daily"] is None  # zero variance


def test_from_engine_maps_the_engine_frames():
    from poseidon.pipeline.engine import EngineConfig, run
    from tests.pipeline.engine.support import Scripted, flat_bars, tmf

    closes = [18000, 18100, 18050, 18200, 18150, 18300]
    result = run(Scripted({0: 5, 3: 0}), tmf(flat_bars(closes)), EngineConfig(initial_capital=1_000_000.0))
    mapped = from_engine(result)
    assert mapped.dates == tuple(pd.Timestamp(d).date() for d in result.daily_pnl["date"])
    np.testing.assert_allclose(mapped.equity, result.daily_pnl["equity"])
    np.testing.assert_allclose(mapped.returns, result.daily_pnl["daily_return"])
    assert mapped.initial_equity == 1_000_000.0
    assert mapped.costs.commission_twd == pytest.approx(result.fills["commission"].sum())
    assert mapped.costs.tax_twd == pytest.approx(result.fills["tax"].sum())
    assert mapped.exposure_changes == result.fills["date"].nunique() == 2
    assert mapped.ruined is None


# ----------------------------------------------------------------------------- holdout


def holdout_inputs(train_mean=0.001, holdout_mean=0.001, n_holdout=250, benchmark_alpha=0.0005, sd=0.01):
    train = daily(noise(1000, train_mean, sd, seed=1))
    held = daily(noise(n_holdout, holdout_mean, sd, seed=2), start=date(2024, 1, 1))
    stats = (
        None
        if benchmark_alpha is None
        else {"alpha_per_day": benchmark_alpha, "alpha_t_hac": 1.5, "beta": 0.2, "basis": "return"}
    )
    return holdout.HoldoutDailyInputs(train, held, "return", stats)


def test_holdout_daily_needs_250_trading_days():
    check(holdout.evaluate_daily(holdout_inputs(n_holdout=249), D), "fail", "insufficient_days")
    result = check(holdout.evaluate_daily(holdout_inputs(n_holdout=250), D), "pass", "pass")
    assert result.report["holdout_days"] == 250 and result.report["passed_criterion"] == "decay"
    assert "250 days" in result.headline and "%/day" in result.headline
    # key missing (not frozen) -> the day minimum is off
    relaxed = {k: v for k, v in D.items() if k != "holdout_min_days"}
    check(holdout.evaluate_daily(holdout_inputs(n_holdout=100), relaxed), "pass", "pass")


def test_holdout_daily_sharpe_criteria_use_the_day_count():
    # Sharpe 0.1/day train, 0.04 holdout: decay 60% > 50%; SE = sqrt((1 + .04^2/2)/T)
    weak = holdout_inputs(train_mean=0.001, holdout_mean=0.0004, n_holdout=250)
    result = check(holdout.evaluate_daily(weak, D), "pass", "pass")  # 0.06 <= 1.96 * 0.0633
    assert result.report["passed_criterion"] == "se" and result.report["criteria"]["se"]["n_obs"] == 250
    long = holdout_inputs(train_mean=0.001, holdout_mean=0.0004, n_holdout=2500)
    check(holdout.evaluate_daily(long, D), "fail", "sharpe_decay")  # SE shrinks with T
    check(holdout.evaluate_daily(holdout_inputs(train_mean=-0.0001), D), "fail", "train_not_positive")


def test_holdout_daily_return_and_alpha():
    check(
        holdout.evaluate_daily(holdout_inputs(train_mean=0.0001, holdout_mean=-0.0001), D),
        "fail",
        "holdout_not_profitable",
    )
    check(holdout.evaluate_daily(holdout_inputs(benchmark_alpha=-1e-6), D), "fail", "holdout_alpha_not_positive")
    check(holdout.evaluate_daily(holdout_inputs(benchmark_alpha=None), D), "fail", "invalid_input")
    gappy = holdout_inputs()
    gappy.benchmark["missing_underlying_days"] = 3
    check(holdout.evaluate_daily(gappy, D), "fail", "invalid_input")


# ----------------------------------------------------------------------------- regime split


def regime_case(bucket_days: dict[str, int], bucket_mean: dict[str, float]):
    """Days in blocks per bucket; ``states`` labels the day *before* each block with its bucket."""

    returns, labels = [], []
    for name, n in bucket_days.items():
        returns += list(noise(n, bucket_mean[name], 0.005, seed=len(labels)))
        labels += [name] * n
    dates = days(len(returns) + 1)
    states = {dates[i]: labels[i] for i in range(len(labels))}  # state at close of day i labels day i+1
    result = daily(returns, dates=dates[1:])
    return regime_split.RegimeDailyInputs(result, "return", states)


def test_regime_daily_counts_days_per_bucket():
    names = regime_split.BUCKETS
    case = regime_case(dict.fromkeys(names, 60), dict(zip(names, (0.001, 0.001, 0.001, -0.001), strict=True)))
    result = check(regime_split.evaluate_daily(case, D), "pass", "pass")
    assert [result.report["buckets"][n]["days"] for n in names] == [60, 60, 60, 60]
    assert result.report["profitable_buckets"] == 3 and result.report["unlabelled_days"] == 0
    short = regime_case({**dict.fromkeys(names, 60), names[2]: 59, names[3]: 59}, dict.fromkeys(names, 0.001))
    check(regime_split.evaluate_daily(short, D), "fail", "insufficient_coverage")
    two_losers = regime_case(dict.fromkeys(names, 80), dict(zip(names, (0.001, 0.001, -0.001, -0.001), strict=True)))
    check(regime_split.evaluate_daily(two_losers, D), "fail", "regime_concentration")


def test_regime_daily_labels_a_day_with_the_previous_close_state():
    dates = days(3)
    result = daily([0.01, 0.01], dates=dates[1:])
    states = {dates[0]: "bull_high_vol", dates[1]: "bear_low_vol", dates[2]: "bear_high_vol"}
    report = regime_split.evaluate_daily(regime_split.RegimeDailyInputs(result, "return", states), D).report
    assert report["buckets"]["bull_high_vol"]["days"] == 1 and report["buckets"]["bear_low_vol"]["days"] == 1
    assert report["buckets"]["bear_high_vol"]["days"] == 0


# ----------------------------------------------------------------------------- benchmark


def underlying(n=600, seed=9):
    dates = days(n)
    values = np.random.default_rng(seed).normal(0.0004, 0.012, n)
    return dates, dict(zip(dates, values, strict=True))


def test_benchmark_daily_static_leverage_has_no_alpha():
    dates, x = underlying()
    cost = 0.00002  # a daily cost drag; tiny noise stands in for integer rounding
    jitter = noise(len(dates), 0.0, 1e-5, seed=3)
    static = daily([3.0 * x[d] - cost + e for d, e in zip(dates, jitter, strict=True)], dates=dates)
    result = check(
        benchmark.evaluate_daily(benchmark.BenchmarkDailyInputs(static, x, "return", 5), D),
        "fail",
        "alpha_not_positive",
    )
    assert result.report["beta"] == pytest.approx(3.0, abs=1e-3)
    assert result.report["alpha_per_day"] == pytest.approx(-cost, abs=3e-6)
    assert result.report["unit"] == "daily return" and "alpha_twd_per_day" not in result.report


def test_benchmark_daily_timing_alpha_passes_and_missing_days_are_invalid():
    dates, x = underlying()
    timed = daily([x[d] + 0.002 + 0.002 * np.sin(i) for i, d in enumerate(dates)], dates=dates)
    result = check(benchmark.evaluate_daily(benchmark.BenchmarkDailyInputs(timed, x, "return", 5), D), "pass", "pass")
    assert result.report["alpha_t_hac"] >= 1.0
    partial = {d: v for d, v in x.items() if d != dates[10]}
    bad = check(
        benchmark.evaluate_daily(benchmark.BenchmarkDailyInputs(timed, partial, "return", 5), D),
        "fail",
        "invalid_input",
    )
    assert bad.report["first_missing_underlying_day"] == dates[10].isoformat()
    report_only = {k: v for k, v in D.items() if k != "benchmark_alpha_t_min"}
    jitter = noise(len(dates), 0.0, 1e-5, seed=4)
    static = daily([2.0 * x[d] - 0.0001 + e for d, e in zip(dates, jitter, strict=True)], dates=dates)
    passed = check(
        benchmark.evaluate_daily(benchmark.BenchmarkDailyInputs(static, x, "return", 5), report_only), "pass", "pass"
    )
    assert "report only" in passed.headline


# ----------------------------------------------------------------------------- liquidation


def test_liquidation_fails_forced_liquidation_and_ruin_reports_margin_calls():
    base = noise(30, 0.001)
    dates = days(30)
    forced = daily(base, events=(EngineEvent(dates[5], "forced_liquidation", "intraday"),))
    check(liquidation.evaluate(forced, D), "fail", "forced_liquidation")
    ruined = daily(base, ruined=dates[29], events=(EngineEvent(dates[29], "ruin"),))
    check(liquidation.evaluate(ruined, D), "fail", "ruin")
    called = check(liquidation.evaluate(daily(base, events=(EngineEvent(dates[3], "margin_call"),)), D), "pass", "pass")
    assert called.report["margin_call_count"] == 1 and "1 margin calls" in called.headline
    off = {k: v for k, v in D.items() if k != "liquidation_fails"}
    assert "report only" in check(liquidation.evaluate(forced, off), "pass", "pass").headline


# ----------------------------------------------------------------------------- cost stress / execution hook


class RecordingRerun:
    def __init__(self, result_for):
        self.requests: list[RerunRequest] = []
        self.result_for = result_for

    def __call__(self, request: RerunRequest) -> DailyResult:
        self.requests.append(request)
        return self.result_for(request)


def test_cost_stress_daily_reruns_with_the_frozen_multiplier():
    base = daily(noise(300, 0.0005), commission=(100.0,) * 300, fill_days=frozenset(days(300)[::5]))
    stressed_ok = daily(noise(300, 0.0003), commission=(200.0,) * 300)
    rerun = RecordingRerun(lambda request: stressed_ok)
    result = check(cost_stress.run_daily(rerun, {"lev": 3.0}, base, "return", D), "pass", "pass")
    assert rerun.requests == [RerunRequest(params={"lev": 3.0}, cost_multiplier=2.0)]
    assert result.report["base_trade_count"] == 60 and result.report["trade_count_unit"] == "days with a fill"
    losing = RecordingRerun(lambda request: daily(noise(300, -0.0001)))
    check(cost_stress.run_daily(losing, {"lev": 3.0}, base, "return", D), "fail", "stressed_not_profitable")
    mismatch = cost_stress.CostStressDailyInputs(base, stressed_ok, 1.5, "return")
    check(cost_stress.evaluate_daily(mismatch, D), "fail", "multiplier_mismatch")


def test_execution_timing_scenarios_follow_the_spec_execution():
    assert execution_timing.scenarios_for({"lev": 3.0}, base_execution="next_open") == {
        "signal_close": RerunRequest(params={"lev": 3.0}, execution="signal_close"),
        "signal_lag_1d": RerunRequest(params={"lev": 3.0}, signal_lag_extra=1),
    }
    assert set(execution_timing.scenarios_for({}, base_execution="signal_close")) == {"next_open", "signal_lag_1d"}
    with pytest.raises(ValueError):
        RerunRequest(params={}, signal_lag_extra=-1)


def _timing(base, scenarios, thresholds=D, execution="next_open"):
    inputs = execution_timing.ExecutionTimingInputs(base, scenarios, "return", execution)
    return execution_timing.evaluate_daily(inputs, thresholds)


def test_execution_timing_passes_when_every_scenario_keeps_pnl_and_half_the_sharpe():
    base = daily(noise(300, 0.0010))  # the same seed throughout: equal std, Sharpe ratio = ratio of means
    close_fill = daily(noise(300, 0.0008))  # 80 %
    lagged = daily(noise(300, 0.00055))  # 55 %
    rerun = RecordingRerun(lambda request: close_fill if request.execution else lagged)
    result = check(
        execution_timing.run_daily(rerun, {"lev": 3.0}, base, base_execution="next_open", basis="return", thresholds=D),
        "pass",
        "pass",
    )
    assert rerun.requests == list(execution_timing.scenarios_for({"lev": 3.0}, base_execution="next_open").values())
    lag = result.report["scenarios"]["signal_lag_1d"]
    assert lag["sharpe_vs_base"] == pytest.approx(0.55) and lag["passed"]
    assert lag["delta_total_pnl_twd"] == pytest.approx(lagged.total_pnl_twd - base.total_pnl_twd)
    assert result.report["sharpe_floor"] == pytest.approx(0.5 * result.report["base"]["sharpe_daily"])
    assert "signal_lag_1d Sharpe 55% of base" in result.headline


def test_execution_timing_fails_a_sharpe_collapse_or_a_loss_in_any_scenario():
    base = daily(noise(300, 0.0010))
    fine = daily(noise(300, 0.0008))
    collapsed = daily(noise(300, 0.00045))  # profitable, 45 % of the base Sharpe
    failed = check(
        _timing(base, {"signal_close": fine, "signal_lag_1d": collapsed}), "fail", "scenario_sharpe_below_ratio"
    )
    assert (
        not failed.report["scenarios"]["signal_lag_1d"]["passed"]
        and failed.report["scenarios"]["signal_close"]["passed"]
    )
    assert "signal_lag_1d Sharpe" in failed.reason
    losing = daily(noise(300, -0.0002))
    check(_timing(base, {"signal_close": losing, "signal_lag_1d": fine}), "fail", "scenario_not_profitable")
    # base on signal_close: the scenarios are next_open and the lag
    check(_timing(base, {"next_open": fine, "signal_lag_1d": fine}, execution="signal_close"), "pass", "pass")


def test_execution_timing_refuses_other_scenario_sets_and_is_report_only_without_thresholds():
    base, fine = daily(noise(300, 0.0010)), daily(noise(300, 0.0008))
    check(_timing(base, {"signal_close": fine}), "fail", "invalid_input")
    check(_timing(base, {"next_open": fine, "signal_lag_1d": fine}), "fail", "invalid_input")
    flat = daily([0.0] * 300)
    check(_timing(flat, {"signal_close": fine, "signal_lag_1d": fine}), "fail", "invalid_input")  # no base Sharpe
    old = {k: v for k, v in D.items() if not k.startswith("execution_timing_")}
    losing = daily(noise(300, -0.0002))
    result = check(_timing(base, {"signal_close": losing, "signal_lag_1d": losing}, old), "pass", "pass")
    assert "report only" in result.headline and result.report["criterion_enabled"] is False
    pnl_only = {**old, "execution_timing_pnl_positive": True}
    check(_timing(base, {"signal_close": fine, "signal_lag_1d": daily(noise(300, 0.0001))}, pnl_only), "pass", "pass")


# ----------------------------------------------------------------------------- dsr / param_stability / wfe


def test_dsr_daily_needs_250_daily_returns_and_uses_the_ledger_unchanged():
    trials = [0.08, 0.02, -0.01, 0.05]
    check(
        dsr.evaluate(dsr.inputs_daily(daily(noise(249, 0.001)), "return", trials), D),
        "fail",
        "insufficient_observations",
    )
    result = check(dsr.evaluate(dsr.inputs_daily(daily(noise(1500, 0.001)), "return", trials), D), "pass", "pass")
    assert result.report["n_obs"] == 1500 and result.report["n_trials"] == 4
    flat = daily([0.0] * 300)
    check(dsr.evaluate(dsr.inputs_daily(flat, "return", trials), D), "fail", "invalid_input")


def test_param_stability_daily_on_v3_param_grid():
    grid = ParamGrid(
        {
            "lev": ParamSpec(type="float", low=1.0, high=3.0, step=0.5, default=2.0),
            "ma_days": ParamSpec(type="int", low=100, high=200, step=50, default=200),
        }
    )
    rows = grid.neighbours({"lev": 2.0, "ma_days": 200}, 1)
    outcomes = {(1.5, 200): 0.0005, (2.5, 200): -0.0005, (2.0, 150): 0.0003}

    def run(params):
        key = (params["lev"], params["ma_days"])
        return None if key == (2.5, 200) else daily(noise(300, outcomes[key]))

    neighbours, failed = param_stability.neighbours_daily(rows, run)
    assert failed == [{"param": "lev", "direction": 1}]
    assert [n.total_pnl_twd is None for n in neighbours] == [False, False, False, True]  # ma_days 250 off-grid
    result = check(
        param_stability.evaluate(
            param_stability.ParamStabilityInputs({"lev": 2.0, "ma_days": 200}, 1.0, neighbours), D
        ),
        "pass",
        "pass",
    )
    assert result.report["profitable"] == 2 and result.report["evaluated"] == 3


def test_walk_forward_layout_by_days():
    end = date(2020, 1, 1)
    windows = walk_forward_windows(end, windows=5, is_days=1460, oos_days=365)
    assert windows[-1]["oos_end"] == end and windows[0]["is_start"] == end - timedelta(days=5 * 365 + 1460)
    assert all(w["oos_end"] - w["oos_start"] == timedelta(days=365) for w in windows)
    layouts = DEFAULT_GATES_BY_CLASS["daily_equity"]["wfe"]["layouts"]
    assert choose_walk_forward(end - timedelta(days=3285), end, layouts)["layout"]["windows"] == 5
    assert choose_walk_forward(end - timedelta(days=3284), end, layouts)["layout"]["windows"] == 4
    assert choose_walk_forward(end - timedelta(days=1458), end, layouts)["layout"]["windows"] == 4
    assert choose_walk_forward(end - timedelta(days=1457), end, layouts) is None


def test_wfe_window_metrics_daily_are_linear():
    in_sample = daily([0.001] * 504)
    oos = daily([0.00051] * 252, start=date(2022, 1, 1))
    metrics = wfe.window_metrics_daily(0, in_sample, oos, "return")
    assert metrics.is_annualized_return == pytest.approx(0.252)
    assert metrics.oos_annualized_return == pytest.approx(0.12852)
    check(wfe.evaluate(wfe.WfeInputs([metrics]), D), "pass", "pass")  # WFE 0.51
    worse = wfe.window_metrics_daily(0, in_sample, daily([0.00049] * 252, start=date(2022, 1, 1)), "return")
    check(wfe.evaluate(wfe.WfeInputs([worse]), D), "fail", "wfe_below_min")
    fixed = wfe.window_metrics_daily(0, in_sample, oos, "twd")
    assert fixed.is_annualized_return == pytest.approx(in_sample.total_pnl_twd / 1e6 * 252 / 504)
    assert metrics.is_trade_count == 0
