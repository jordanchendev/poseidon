"""Gate 7 -- alpha over TX buy-and-hold (train part).

Daily strategy PnL is regressed on daily TX buy-and-hold PnL over the same
trading days (both TWD, the run's contract count, mark-to-market per TAIFEX
trading day: night session -> next day session, see
``backtesting.trading_days``)::

    strategy_t = alpha + beta * buy_and_hold_t + e_t        (OLS)

``alpha`` is TWD per trading day; its standard error is Newey-West (1987)
HAC: Bartlett kernel, ``hac_lags`` lags, no small-sample factor -- equal to
statsmodels ``OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})``. A strategy that only rides the
market's drift has ``alpha <= 0`` once costs are paid, whatever its Sharpe.

Pass iff ``alpha > 0`` and ``alpha_t >= benchmark_alpha_t_min``. Runs frozen
without ``benchmark_alpha_t_min`` have the criterion off: the numbers are
reported and the gate passes ("report only"). :func:`regress` is shared with
the holdout gate (``holdout_alpha_positive``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from poseidon.pipeline.gates.base import (
    INVALID_INPUT,
    GateResult,
    failed,
    finite_or_none,
    invalid,
    is_finite,
    passed,
)
from poseidon.pipeline.gates.daily import Basis, DailyResult, align

GATE = "benchmark"
METHOD = "OLS strategy_daily = alpha + beta * buy_and_hold_daily; Newey-West HAC (Bartlett kernel, no df correction)"
MIN_OBS = 3


@dataclass(frozen=True)
class BenchmarkInputs:
    strategy_daily: Sequence[float]  # TWD per trading day, aligned with ``benchmark_daily``
    benchmark_daily: Sequence[float]
    hac_lags: int
    annualisation_factor: float = 252.0
    time_in_market: float | None = None  # share of the evaluated 5m bars with a position open


def newey_west(design: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West covariance of OLS coefficients (Bartlett weights 1 - l/(lags+1))."""

    n = design.shape[0]
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    bread = np.linalg.inv(design.T @ design)
    return bread @ meat @ bread


def regress(strategy: Sequence[float], benchmark: Sequence[float], *, lags: int, annualisation: float = 252.0) -> dict:
    """Alpha/beta of ``strategy`` on ``benchmark`` with HAC t-stats; non-finite figures are ``None``."""

    y = np.asarray(strategy, dtype=float)
    x = np.asarray(benchmark, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("strategy and benchmark series must be 1-d and aligned")
    n = len(y)
    out = {
        "n_obs": n,
        "hac_lags": int(lags),
        "method": METHOD,
        "alpha_twd_per_day": None,
        "alpha_annualised_twd": None,
        "alpha_se": None,
        "alpha_t_hac": None,
        "beta": None,
        "beta_t_hac": None,
        "r_squared": None,
        "strategy_total_pnl_twd": finite_or_none(float(y.sum())) if n else None,
        "benchmark_total_pnl_twd": finite_or_none(float(x.sum())) if n else None,
        "beta_scaled_benchmark_pnl_twd": None,
        "alpha_total_pnl_twd": None,
    }
    if n < MIN_OBS or not (np.isfinite(y).all() and np.isfinite(x).all()) or not x.std() > 0:
        return out
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    cov = newey_west(design, residuals, int(lags))
    alpha, beta = float(coef[0]), float(coef[1])
    alpha_se, beta_se = (math.sqrt(v) if v > 0 else float("nan") for v in np.diag(cov))
    total = float(((y - y.mean()) ** 2).sum())
    out.update(
        {
            "alpha_twd_per_day": finite_or_none(alpha),
            "alpha_annualised_twd": finite_or_none(alpha * annualisation),
            "alpha_se": finite_or_none(alpha_se),
            "alpha_t_hac": finite_or_none(alpha / alpha_se),
            "beta": finite_or_none(beta),
            "beta_t_hac": finite_or_none(beta / beta_se),
            "r_squared": finite_or_none(1.0 - float((residuals**2).sum()) / total) if total > 0 else None,
            "beta_scaled_benchmark_pnl_twd": finite_or_none(beta * float(x.sum())),
            "alpha_total_pnl_twd": finite_or_none(float(y.sum()) - beta * float(x.sum())),
        }
    )
    return out


def phrase(stats: Mapping) -> str:
    """``"alpha +1,234/day (t 2.31), beta 0.42"`` -- the Telegram/Kairos fragment."""

    alpha, t, beta = stats.get("alpha_twd_per_day"), stats.get("alpha_t_hac"), stats.get("beta")
    if not (is_finite(alpha) and is_finite(t) and is_finite(beta)):
        return "alpha n/a"
    return f"alpha {alpha:+,.0f}/day (t {t:.2f}), beta {beta:.2f}"


def evaluate(inputs: BenchmarkInputs, thresholds: Mapping) -> GateResult:
    t_min = thresholds.get("benchmark_alpha_t_min")  # missing: frozen before this gate -> report only
    if t_min is not None and not is_finite(t_min):
        return invalid(GATE, f"benchmark_alpha_t_min is not a finite number ({t_min!r})")
    if not is_finite(inputs.hac_lags) or int(inputs.hac_lags) < 0:
        return invalid(GATE, f"hac_lags must be a non-negative integer ({inputs.hac_lags!r})")
    stats = regress(
        inputs.strategy_daily,
        inputs.benchmark_daily,
        lags=int(inputs.hac_lags),
        annualisation=float(inputs.annualisation_factor),
    )
    report = {
        **stats,
        "time_in_market": finite_or_none(inputs.time_in_market),
        "alpha_t_min": float(t_min) if t_min is not None else None,
        "criterion_enabled": t_min is not None,
    }
    bad = [name for name in ("alpha_twd_per_day", "alpha_t_hac", "beta") if stats[name] is None]
    if bad:
        return invalid(
            GATE,
            f"regression undefined on {stats['n_obs']} aligned days (non-finite {', '.join(bad)})",
            report,
        )
    alpha, t = stats["alpha_twd_per_day"], stats["alpha_t_hac"]
    headline = phrase(stats)
    if t_min is None:
        return passed(GATE, f"{headline} (report only)", report, reason="benchmark_alpha_t_min not frozen in this run")
    if not alpha > 0:
        return failed(
            GATE,
            "alpha_not_positive",
            f"alpha {alpha:+,.0f} TWD/day <= 0 over TX buy-and-hold (beta {stats['beta']:.2f}, t {t:.2f})",
            headline,
            report,
        )
    if t < float(t_min):
        return failed(
            GATE,
            "alpha_not_significant",
            f"alpha HAC t {t:.2f} < {float(t_min):g} (alpha {alpha:+,.0f} TWD/day, beta {stats['beta']:.2f})",
            headline,
            report,
        )
    return passed(GATE, headline, report)


# --- daily classes (design 2026-10-24 §7, §12 #8) ---------------------------------------------
#
# The strategy's daily metric (daily return for ``daily_equity``, TWD PnL for ``daily_fixed``) is
# regressed on the 1x underlying's daily *return* over the same trading days (TX family: the
# back-adjusted continuous TX). Static leverage L of the underlying is y = L * x - costs: alpha
# ~ -costs, so it fails, which is the point -- only timing / vol control earns alpha. Same
# criteria as the intraday gate: alpha > 0 and HAC t >= ``benchmark_alpha_t_min`` (missing:
# report only). Any train day without an underlying return is an input error.

DAILY_METHOD = (
    "OLS strategy_daily = alpha + beta * underlying_1x_daily_return; Newey-West HAC (Bartlett kernel, no df correction)"
)
_NEUTRAL_KEYS = {
    "alpha_twd_per_day": "alpha_per_day",
    "alpha_annualised_twd": "alpha_annualised",
    "strategy_total_pnl_twd": "strategy_sum",
    "benchmark_total_pnl_twd": "underlying_return_sum",
    "beta_scaled_benchmark_pnl_twd": "beta_scaled_underlying_sum",
    "alpha_total_pnl_twd": "alpha_sum",
}


@dataclass(frozen=True)
class NotApplicable:
    """The beta-matched comparator cannot be built or held (the risk route is then not applicable)."""

    reason: str


# leverage -> the train backtest of the 1x underlying held at that constant leverage, rebalanced daily, on the
# strategy's own instrument, capital, execution and cost model over the same window; or why it cannot be held.
MatchedRun = Callable[[float], "DailyResult | NotApplicable"]


@dataclass(frozen=True)
class BenchmarkDailyInputs:
    result: DailyResult  # the train backtest
    underlying: Mapping[date, float]  # 1x underlying simple daily return per trading day
    basis: Basis
    hac_lags: int
    annualisation_factor: float = 252.0
    matched: MatchedRun | None = None  # read only under RULE_ALPHA_OR_RISK


def regress_daily(
    result: DailyResult, underlying: Mapping[date, float], *, basis: Basis, lags: int, annualisation: float = 252.0
) -> dict:
    """:func:`regress` of the daily metric on the underlying's daily return, with unit-neutral keys."""

    y, x, missing = align(result, underlying, basis)
    stats = regress(y, x, lags=lags, annualisation=annualisation)
    out = {_NEUTRAL_KEYS.get(key, key): value for key, value in stats.items()}
    out.update(
        method=DAILY_METHOD,
        basis=basis,
        unit="daily return" if basis == "return" else "TWD per day",
        missing_underlying_days=len(missing),
        first_missing_underlying_day=missing[0].isoformat() if missing else None,
    )
    return out


def phrase_daily(stats: Mapping) -> str:
    """``"alpha +0.031%/day (t 2.10), beta 2.95"`` (return basis) or the TWD phrase."""

    alpha, t, beta = stats.get("alpha_per_day"), stats.get("alpha_t_hac"), stats.get("beta")
    if not (is_finite(alpha) and is_finite(t) and is_finite(beta)):
        return "alpha n/a"
    if stats.get("basis") == "return":
        return f"alpha {alpha * 100:+.3f}%/day (t {t:.2f}), beta {beta:.2f}"
    return f"alpha {alpha:+,.0f}/day (t {t:.2f}), beta {beta:.2f}"


def evaluate_daily(inputs: BenchmarkDailyInputs, thresholds: Mapping) -> GateResult:
    rule = thresholds.get(RULE_KEY)  # missing: the alpha route alone (every run frozen before 2026-10-28)
    if rule is not None and rule != RULE_ALPHA_OR_RISK:
        return invalid(GATE, f"unknown {RULE_KEY} {rule!r}")
    alpha = _alpha_route(inputs, thresholds)
    if rule is None or alpha.reason_code == INVALID_INPUT or thresholds.get("benchmark_alpha_t_min") is None:
        return alpha
    risk = risk_route(inputs.result, inputs.basis, alpha.report["beta"], inputs.matched, thresholds)
    if risk["status"] == "invalid":
        return invalid(GATE, f"risk route: {risk['reason']}", {**alpha.report, "rule": rule, "risk_route": risk})
    routes = [name for name, ok in (("alpha", alpha.passed), ("risk", risk["status"] == "pass")) if ok]
    report = {**alpha.report, "rule": rule, "alpha_route": "pass" if alpha.passed else "fail", "risk_route": risk}
    report["passed_routes"] = routes
    headline = f"{alpha.headline}; {phrase_risk(risk)}"
    if routes:
        return passed(GATE, headline, report, reason=f"passed by the {' and '.join(routes)} route")
    return failed(GATE, alpha.reason_code, f"{alpha.reason}; risk route: {risk['reason']}", headline, report)


def _alpha_route(inputs: BenchmarkDailyInputs, thresholds: Mapping) -> GateResult:
    t_min = thresholds.get("benchmark_alpha_t_min")  # missing -> report only
    if t_min is not None and not is_finite(t_min):
        return invalid(GATE, f"benchmark_alpha_t_min is not a finite number ({t_min!r})")
    if not is_finite(inputs.hac_lags) or int(inputs.hac_lags) < 0:
        return invalid(GATE, f"hac_lags must be a non-negative integer ({inputs.hac_lags!r})")
    stats = regress_daily(
        inputs.result,
        inputs.underlying,
        basis=inputs.basis,
        lags=int(inputs.hac_lags),
        annualisation=float(inputs.annualisation_factor),
    )
    report = {
        **stats,
        "alpha_t_min": float(t_min) if t_min is not None else None,
        "criterion_enabled": t_min is not None,
    }
    if stats["missing_underlying_days"]:
        return invalid(
            GATE,
            f"{stats['missing_underlying_days']} train days have no underlying return "
            f"(first {stats['first_missing_underlying_day']})",
            report,
        )
    bad = [name for name in ("alpha_per_day", "alpha_t_hac", "beta") if stats[name] is None]
    if bad:
        return invalid(
            GATE, f"regression undefined on {stats['n_obs']} aligned days (non-finite {', '.join(bad)})", report
        )
    alpha, t = stats["alpha_per_day"], stats["alpha_t_hac"]
    headline = phrase_daily(stats)
    if t_min is None:
        return passed(GATE, f"{headline} (report only)", report, reason="benchmark_alpha_t_min not frozen in this run")
    if not alpha > 0:
        return failed(
            GATE,
            "alpha_not_positive",
            f"{headline}: alpha <= 0 over the 1x underlying",
            headline,
            report,
        )
    if t < float(t_min):
        return failed(GATE, "alpha_not_significant", f"{headline}: HAC t < {float(t_min):g}", headline, report)
    return passed(GATE, headline, report)


# --- the beta-matched risk route (owner decision 2026-10-28; runs frozen with ``benchmark_rule``) -----------
#
# Leveraged-index timing (cut leverage below a moving average, a vol cap, a put) mostly buys a smaller drawdown,
# which the regression cannot see: its beta is the average exposure and its alpha only the extra return. The
# risk route compares the strategy with the beta-matched, no-timing version of itself -- the 1x underlying at a
# constant leverage equal to the measured beta, rebalanced daily, same instrument, capital, execution and costs,
# same train days -- on Calmar (CAGR / max drawdown). It passes iff the one-sided lower confidence bound of
# Calmar(strategy) - Calmar(matched) is above 0 under a paired circular block bootstrap of the daily returns
# (the same blocks of days for both). Block length, resamples and confidence are frozen in ``thresholds``; they
# were set on null strategies (constant leverage, random regime switching on TX), not on real runs:
# tests/pipeline/benchmark_risk_calibration.py.
#
# The route does not apply when the strategy is its matched run up to contract rounding: tracking = std(strategy
# - matched) / std(matched), daily returns, below ``benchmark_risk_min_tracking``. A constant-leverage strategy and
# its matched run then differ only by whole-contract sizing, a fixed exposure gap whose Calmar edge has the same
# sign in every resample, so the lower bound lands just above 0 on rounding alone (TMF, 1M TWD: up to 23 % of
# constant-leverage windows at train length; tests/pipeline/holdout_risk_calibration.py, ``frozen_train``).

RULE_KEY = "benchmark_rule"
RULE_ALPHA_OR_RISK = "alpha_or_beta_matched_risk"
RISK_KEYS = (
    "benchmark_risk_block_days",
    "benchmark_risk_resamples",
    "benchmark_risk_confidence",
    "benchmark_risk_min_beta",
    "benchmark_risk_min_tracking",  # missing: no tracking floor (a config frozen before 2026-10-29)
)
# The stage's rule, not the gate's (``daily_gates.beta_matched_run``): a futures matched run holding fewer contracts
# is not applicable. Missing: no such rule (a config frozen before 2026-10-29).
MIN_CONTRACTS_KEY = "benchmark_risk_min_contracts"
RISK_SEED = 0
RISK_CHUNK = 256  # resamples per vectorised batch


@dataclass(frozen=True)
class RiskKeys:
    """The ``thresholds`` keys one risk route reads (the train route's by default)."""

    rule: str = RULE_KEY
    block_days: str = RISK_KEYS[0]
    resamples: str = RISK_KEYS[1]
    confidence: str = RISK_KEYS[2]
    min_beta: str = RISK_KEYS[3]
    min_tracking: str = RISK_KEYS[4]


RISK_METHOD = (
    "Calmar(strategy) - Calmar(1x underlying at constant leverage = beta rounded to 0.01, rebalanced daily, same "
    "instrument / capital / execution / costs / days); one-sided lower bound of a paired circular block bootstrap"
)


def calmar_rows(returns: np.ndarray, annualisation: float = 252.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CAGR, max drawdown (a positive fraction, from the starting equity) and Calmar of each row of daily returns.

    A path that loses everything has CAGR -1; a path without a drawdown has Calmar +inf (or nan at CAGR 0).
    """

    growth = np.cumprod(1.0 + returns, axis=-1)
    peak = np.maximum.accumulate(np.maximum(growth, 1.0), axis=-1)
    drawdown = np.max(1.0 - growth / peak, axis=-1)
    final = growth[..., -1]
    years = returns.shape[-1] / annualisation
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cagr = np.where(final > 0, np.abs(final) ** (1.0 / years) - 1.0, -1.0)
        calmar = cagr / drawdown
    return cagr, drawdown, calmar


def block_indices(rng: np.random.Generator, n: int, block: int, resamples: int) -> np.ndarray:
    """``resamples`` circular block bootstrap index rows of length ``n``: blocks of ``block`` consecutive days
    (wrapping past the last day) from uniform random starts."""

    blocks = -(-n // block)
    starts = rng.integers(0, n, size=(resamples, blocks))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(resamples, -1)[:, :n]


def calmar_draws(
    strategy: np.ndarray,
    matched: np.ndarray,
    *,
    block_days: int,
    resamples: int,
    seed: int = RISK_SEED,
    annualisation: float = 252.0,
) -> np.ndarray:
    """Calmar(strategy) - Calmar(matched) on ``resamples`` paired circular block bootstrap paths (the same days
    for both); an undefined difference counts as -inf, against the strategy."""

    rng = np.random.default_rng(seed)
    draws = []
    for done in range(0, resamples, RISK_CHUNK):
        rows = block_indices(rng, len(strategy), block_days, min(RISK_CHUNK, resamples - done))
        draws.append(calmar_rows(strategy[rows], annualisation)[2] - calmar_rows(matched[rows], annualisation)[2])
    return np.nan_to_num(np.concatenate(draws), nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def lower_bound(draws: np.ndarray, confidence: float) -> float:
    """The one-sided lower bound: the order statistic below which at most ``1 - confidence`` of the draws lie."""

    return float(np.quantile(draws, 1.0 - confidence, method="lower"))


def calmar_difference(
    strategy: np.ndarray, matched: np.ndarray, *, block_days: int, resamples: int, confidence: float
) -> dict:
    """Calmar of both paths, their difference and its one-sided lower bound at ``confidence``."""

    s, m = np.asarray(strategy, dtype=float), np.asarray(matched, dtype=float)
    (s_cagr, s_dd, s_calmar), (m_cagr, m_dd, m_calmar) = (map(float, calmar_rows(path)) for path in (s, m))
    draws = calmar_draws(s, m, block_days=block_days, resamples=resamples)
    return {
        "strategy": {
            "cagr": finite_or_none(s_cagr),
            "max_drawdown": finite_or_none(s_dd),
            "calmar": finite_or_none(s_calmar),
        },
        "matched": {
            "cagr": finite_or_none(m_cagr),
            "max_drawdown": finite_or_none(m_dd),
            "calmar": finite_or_none(m_calmar),
        },
        "calmar_diff": finite_or_none(s_calmar - m_calmar),
        "calmar_diff_lower": finite_or_none(lower_bound(draws, confidence)),
        "resamples_below_zero": float(np.mean(draws <= 0.0)),
    }


def risk_route(
    result: DailyResult,
    basis: Basis,
    beta: float,
    matched: MatchedRun | None,
    thresholds: Mapping,
    keys: RiskKeys = RiskKeys(),
) -> dict:
    """The risk route's report on ``result`` (the train run; the holdout run reads its own ``keys``):
    ``status`` pass | fail | not_applicable | invalid, ``reason`` and the numbers. ``beta``: the regression's
    beta on the same days."""

    names = (keys.block_days, keys.resamples, keys.confidence, keys.min_beta)
    missing = [key for key in names if not is_finite(thresholds.get(key))]
    if missing:
        return {"status": "invalid", "reason": f"{keys.rule} is frozen without {', '.join(missing)}"}
    min_tracking = thresholds.get(keys.min_tracking)
    if min_tracking is not None and not is_finite(min_tracking):
        return {"status": "invalid", "reason": f"{keys.min_tracking} is not a finite number ({min_tracking!r})"}
    block, resamples = int(thresholds[keys.block_days]), int(thresholds[keys.resamples])
    confidence, min_beta = float(thresholds[keys.confidence]), float(thresholds[keys.min_beta])
    out = {
        "status": "not_applicable",
        "reason": "",
        "metric": "calmar",
        "method": RISK_METHOD,
        "block_days": block,
        "resamples": resamples,
        "confidence": confidence,
        "min_beta": min_beta,
        "min_tracking": min_tracking,
        "matched_leverage": None,
        "tracking": None,
    }

    def not_applicable(reason: str) -> dict:
        return {**out, "reason": reason}

    if basis != "return":
        return not_applicable("defined for equity-sized (daily_equity) runs; this run is judged in TWD")
    if not beta >= min_beta:
        return not_applicable(f"beta {beta:.2f} < {min_beta:g}: no long exposure to match")
    leverage = round(float(beta), 2)
    out["matched_leverage"] = leverage
    if matched is None:
        return {**out, "status": "invalid", "reason": "the stage supplied no beta-matched run"}
    run = matched(leverage)
    if isinstance(run, NotApplicable):
        return not_applicable(run.reason)
    if run.dates != result.dates:
        return {**out, "status": "invalid", "reason": "the beta-matched run does not cover the strategy's days"}
    s, m = result.returns, run.returns
    if not (np.isfinite(s).all() and np.isfinite(m).all()) or len(s) < 2 * block:
        return not_applicable(f"needs finite daily returns on at least {2 * block} days")
    with np.errstate(divide="ignore", invalid="ignore"):
        tracking = float(np.std(s - m) / np.std(m))
    out["tracking"] = finite_or_none(tracking)
    if min_tracking is not None and not tracking >= min_tracking:
        return not_applicable(
            f"tracking {tracking:.3f} < {min_tracking:g}: the {leverage:g}x matched run up to contract rounding"
        )
    stats = calmar_difference(s, m, block_days=block, resamples=resamples, confidence=confidence)
    out.update(stats, matched_costs=run.costs.as_report(), matched_margin_calls=len(run.event_days(["margin_call"])))
    lower = stats["calmar_diff_lower"]
    level = f"{confidence * 100:g}%"  # 95%, 97.5%
    if lower is not None and lower > 0:
        return {
            **out,
            "status": "pass",
            "reason": f"Calmar beats the {leverage:g}x matched run ({level} lower bound > 0)",
        }
    return {**out, "status": "fail", "reason": f"Calmar not above the {leverage:g}x matched run at {level}"}


def phrase_risk(risk: Mapping) -> str:
    """``"Calmar 0.31 vs 0.12 at 1.11x (95% lower +0.04)"``, or why the route did not apply."""

    if risk["status"] == "not_applicable":
        return f"risk route n/a ({risk['reason']})"
    s, m, lower = risk["strategy"]["calmar"], risk["matched"]["calmar"], risk["calmar_diff_lower"]
    if not (is_finite(s) and is_finite(m) and is_finite(lower)):
        return f"risk route at {risk['matched_leverage']:g}x: Calmar n/a"
    return (
        f"Calmar {s:.2f} vs {m:.2f} at {risk['matched_leverage']:g}x ({risk['confidence'] * 100:g}% lower {lower:+.2f})"
    )
