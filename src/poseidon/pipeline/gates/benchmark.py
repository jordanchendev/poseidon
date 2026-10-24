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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from poseidon.pipeline.gates.base import GateResult, failed, finite_or_none, invalid, is_finite, passed
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
class BenchmarkDailyInputs:
    result: DailyResult  # the train backtest
    underlying: Mapping[date, float]  # 1x underlying simple daily return per trading day
    basis: Basis
    hac_lags: int
    annualisation_factor: float = 252.0


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
