"""The benchmark gate (alpha over TX buy-and-hold): regression, verdicts, stage wiring."""

from __future__ import annotations

import copy
import math
import uuid
from datetime import date

import numpy as np
import pandas as pd
import pytest

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.pipeline.gate_stages import (
    BENCHMARK_CONFIG_MISSING,
    BenchmarkStage,
    align_daily,
    buy_and_hold_daily,
    daily_bars,
    time_in_market,
)
from poseidon.pipeline.gates import benchmark, summary_line
from poseidon.pipeline.gates.base import Trade
from poseidon.pipeline.plan import default_config
from poseidon.pipeline.stages import StageContext, UpstreamResult
from poseidon.pipeline.state import Verdict
from tests.pipeline.conftest import SyntheticTxSource

T = default_config(stage_plan=None, as_of=date(2026, 10, 22))["thresholds"]
REPORT_ONLY = {key: value for key, value in T.items() if key != "benchmark_alpha_t_min"}


def check(result, verdict, code):
    assert (result.verdict, result.reason_code) == (Verdict(verdict), code), (result.reason, result.report)
    canonical_json(result.report)
    return result


def series(n=600, alpha=300.0, beta=0.5, noise=20_000.0, seed=3, drift=500.0, exact=True):
    """Daily strategy PnL = alpha + beta * buy-and-hold + noise; ``exact``: OLS recovers alpha/beta exactly."""

    rng = np.random.default_rng(seed)
    x = rng.normal(drift, 60_000.0, n)
    e = rng.normal(0.0, noise, n)
    if exact:  # residual orthogonal to [1, x]
        design = np.column_stack([np.ones(n), x])
        e = e - design @ np.linalg.lstsq(design, e, rcond=None)[0]
    return alpha + beta * x + e, x


def inputs(y, x, lags=5, tim=None):
    return benchmark.BenchmarkInputs(y, x, lags, 252.0, tim)


# ------------------------------------------------------------------ regression


def test_regress_recovers_a_planted_alpha_and_beta():
    y, x = series(n=5_000, alpha=1_000.0, beta=0.4, noise=5_000.0, exact=False)
    stats = benchmark.regress(y, x, lags=5)
    assert stats["beta"] == pytest.approx(0.4, abs=0.005)
    assert stats["alpha_twd_per_day"] == pytest.approx(1_000.0, abs=200.0)
    assert stats["alpha_annualised_twd"] == pytest.approx(252 * stats["alpha_twd_per_day"])
    assert stats["alpha_t_hac"] == pytest.approx(stats["alpha_twd_per_day"] / stats["alpha_se"])
    assert stats["alpha_total_pnl_twd"] == pytest.approx(stats["alpha_twd_per_day"] * len(y))
    assert stats["beta_scaled_benchmark_pnl_twd"] == pytest.approx(stats["beta"] * x.sum())
    assert stats["strategy_total_pnl_twd"] == pytest.approx(y.sum())


def test_hac_with_zero_lags_is_hc0():
    y, x = series(n=200)
    stats = benchmark.regress(y, x, lags=0)
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.solve(design.T @ design, design.T @ y)
    resid = y - design @ coef
    bread = np.linalg.inv(design.T @ design)
    hc0 = bread @ (design.T * resid**2) @ design @ bread
    assert stats["alpha_se"] == pytest.approx(math.sqrt(hc0[0, 0]))


def test_hac_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    y, x = series(n=400)
    y = y + np.convolve(np.random.default_rng(9).normal(0, 8_000.0, 404), np.ones(5), "valid")  # autocorrelated
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    stats = benchmark.regress(y, x, lags=5)
    assert stats["alpha_se"] == pytest.approx(fit.bse[0], rel=1e-9)
    assert stats["alpha_t_hac"] == pytest.approx(fit.tvalues[0], rel=1e-9)
    assert stats["beta_t_hac"] == pytest.approx(fit.tvalues[1], rel=1e-9)


def test_regress_is_undefined_on_flat_or_tiny_inputs():
    assert benchmark.regress([1.0, 2.0], [1.0, 3.0], lags=5)["alpha_t_hac"] is None
    assert benchmark.regress([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], lags=5)["beta"] is None
    assert benchmark.regress([1.0, float("nan"), 3.0], [1.0, 2.0, 4.0], lags=5)["alpha_twd_per_day"] is None


# ------------------------------------------------------------------ verdicts


def test_significant_positive_alpha_passes():
    y, x = series(alpha=3_000.0)
    result = check(benchmark.evaluate(inputs(y, x, tim=0.4), T), "pass", "pass")
    report = result.report
    assert report["alpha_t_hac"] >= 1.0 and report["criterion_enabled"] and report["alpha_t_min"] == 1.0
    assert report["time_in_market"] == 0.4 and report["hac_lags"] == 5
    a, t, b = report["alpha_twd_per_day"], report["alpha_t_hac"], report["beta"]
    assert summary_line(result) == f"benchmark ✅ alpha {a:+,.0f}/day (t {t:.2f}), beta {b:.2f}"


def test_pure_beta_fails():
    # beta 0.6 of buy-and-hold minus costs: a positive mean daily PnL but no alpha
    y, x = series(alpha=-150.0, beta=0.6, drift=2_000.0)
    assert y.mean() > 0
    result = check(benchmark.evaluate(inputs(y, x), T), "fail", "alpha_not_positive")
    assert result.report["alpha_twd_per_day"] == pytest.approx(-150.0)


def test_positive_but_insignificant_alpha_fails():
    y, x = series(alpha=150.0, noise=60_000.0, seed=4)
    stats = benchmark.regress(y, x, lags=5)
    assert stats["alpha_twd_per_day"] == pytest.approx(150.0) and 0 < stats["alpha_t_hac"] < 1.0
    check(benchmark.evaluate(inputs(y, x), T), "fail", "alpha_not_significant")


def test_t_boundary_is_inclusive():
    y, x = series(alpha=3_000.0)
    t = benchmark.regress(y, x, lags=5)["alpha_t_hac"]
    check(benchmark.evaluate(inputs(y, x), {**T, "benchmark_alpha_t_min": t}), "pass", "pass")
    check(benchmark.evaluate(inputs(y, x), {**T, "benchmark_alpha_t_min": t + 1e-9}), "fail", "alpha_not_significant")


def test_missing_threshold_means_report_only():
    y, x = series(alpha=-5_000.0)
    result = check(benchmark.evaluate(inputs(y, x), REPORT_ONLY), "pass", "pass")
    assert result.report["criterion_enabled"] is False and result.report["alpha_t_min"] is None
    assert summary_line(result).endswith("(report only)")


@pytest.mark.parametrize(
    ("y", "x", "thresholds"),
    [
        ([1.0, 2.0], [3.0, 1.0], T),
        ([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], T),
        ([1.0, math.inf, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], T),
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], {**T, "benchmark_alpha_t_min": float("nan")}),
    ],
)
def test_undefined_inputs_fail_closed(y, x, thresholds):
    check(benchmark.evaluate(inputs(y, x), thresholds), "fail", "invalid_input")


# ------------------------------------------------------------------ series helpers


def _frame():
    source = SyntheticTxSource()
    start = pd.Timestamp("2026-10-01", tz="Asia/Taipei").tz_convert("UTC")
    return source.fetch("TX", "tw_futures", "5m", start, start + pd.Timedelta(days=12))


def test_buy_and_hold_is_close_to_close_per_trading_day():
    frame = _frame()
    bh = buy_and_hold_daily(frame, point_value=200.0, contracts=1)
    daily = daily_bars(frame)
    assert list(bh.index) == list(daily.index)
    assert bh.sum() == pytest.approx((daily["close"].iloc[-1] - daily["open"].iloc[0]) * 200.0)
    assert bh.iloc[1] == pytest.approx((daily["close"].iloc[1] - daily["close"].iloc[0]) * 200.0)
    assert buy_and_hold_daily(frame, point_value=200.0, contracts=2).sum() == pytest.approx(2 * bh.sum())


def test_align_and_time_in_market():
    days = [date(2026, 10, d) for d in (1, 2, 5)]
    y, x, info = align_daily(pd.Series([1.0, 2.0, 3.0], index=days), pd.Series([4.0, 5.0], index=days[1:]))
    assert list(y.index) == days[1:] == list(x.index) and info["strategy_only_days"] == 1
    index = pd.date_range("2026-10-01 01:00", periods=10, freq="5min", tz="UTC")
    trade = Trade(index[2].to_pydatetime(), index[5].to_pydatetime(), "long", 1.0)
    assert time_in_market([trade], index) == pytest.approx(0.3)
    assert time_in_market([], index) == 0.0 and time_in_market([trade], index[:0]) is None


# ------------------------------------------------------------------ stage


def _ctx(config, tmp_path, report):
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="x",
        input_spec_json=None,
        config=config,
        config_sha256=content_sha256(config),
        stage="benchmark",
        attempt=1,
        upstream={"optimize": UpstreamResult("optimize", 1, "0" * 64, "1" * 64, report)},
        artifacts_root=tmp_path,
    )


def test_benchmark_stage_is_unsupported_for_runs_frozen_before_version_5(tmp_path):
    config = default_config(stage_plan=None, as_of=date(2026, 10, 22))
    old = copy.deepcopy(config)
    old["version"] = 4
    old["gates"].pop("benchmark")
    report = {"best_spec": {"dsl_version": 2}, "best": {"params": {}}}
    result = BenchmarkStage(SyntheticTxSource()).run(_ctx(old, tmp_path, report))
    assert result.verdict == Verdict.UNSUPPORTED and result.reason == BENCHMARK_CONFIG_MISSING
