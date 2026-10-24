"""Gate 2 -- locked holdout decay.

``decay = 1 - holdout_sharpe / train_sharpe`` (Sharpe per trading day). A
non-positive train Sharpe has no meaningful decay and fails
``train_not_positive``. Checks, in order: train Sharpe > 0, holdout trades >=
``holdout_min_trades``, the Sharpe criterion, holdout ``total_pnl_twd`` > 0
(when ``holdout_pnl_positive``).

The Sharpe criterion passes when EITHER

* (a) ``decay <= holdout_sharpe_decay_max``, OR
* (b) the holdout Sharpe is not statistically worse than train:
  ``holdout_sharpe >= train_sharpe - z * SE`` with
  ``SE = sqrt((1 + holdout_sharpe**2 / 2) / T)`` (Lo 2002, i.i.d.), ``T`` the
  number of holdout trading-day observations (``n_obs``) and ``z =
  holdout_se_z``.

With a small train Sharpe the decay ratio is dominated by the holdout Sharpe's
sampling noise, which (b) accounts for. Runs frozen before ``holdout_se_z``
existed have no such threshold: (b) is off and only (a) applies.

Alpha over TX buy-and-hold (``benchmark.regress`` on the holdout days, passed
in as ``benchmark``) is reported whenever the stage supplies it; with
``holdout_alpha_positive`` the gate also requires holdout alpha > 0 (checked
last). Runs frozen without that key have it off. The split
itself is enforced upstream by ``backtest.holdout.HoldoutConfig``; this gate
only judges the two results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from poseidon.pipeline.gates.base import (
    GateResult,
    ResultSummary,
    failed,
    finite_or_none,
    invalid,
    is_finite,
    non_finite,
    passed,
    pct,
)
from poseidon.pipeline.gates.benchmark import phrase as benchmark_phrase
from poseidon.pipeline.gates.benchmark import phrase_daily
from poseidon.pipeline.gates.daily import Basis, DailyResult, sharpe

GATE = "holdout"
SE_FORMULA = "sqrt((1 + holdout_sharpe^2 / 2) / T_holdout)  (Lo 2002, i.i.d.)"


@dataclass(frozen=True)
class HoldoutInputs:
    train: ResultSummary
    holdout: ResultSummary
    # T: trading-day observations behind ``holdout.sharpe`` (the backtest's ``n_obs``).
    holdout_n_obs: int | None = None
    # ``benchmark.regress`` of the holdout daily PnL on TX buy-and-hold (None: not computed).
    benchmark: Mapping | None = None


def sharpe_se(sharpe: float, n_obs: int) -> float:
    """Standard error of a per-period Sharpe estimate (Lo 2002, i.i.d. returns)."""

    return math.sqrt((1.0 + sharpe * sharpe / 2.0) / n_obs)


def evaluate(inputs: HoldoutInputs, thresholds: Mapping) -> GateResult:
    decay_max = float(thresholds["holdout_sharpe_decay_max"])
    min_trades = int(thresholds["holdout_min_trades"])
    require_pnl = bool(thresholds["holdout_pnl_positive"])
    z = thresholds.get("holdout_se_z")  # missing: frozen before criterion (b) existed -> off
    require_alpha = bool(thresholds.get("holdout_alpha_positive", False))  # missing -> off
    stats = dict(inputs.benchmark) if inputs.benchmark is not None else None
    alpha = (stats or {}).get("alpha_twd_per_day")
    train, holdout = inputs.train, inputs.holdout
    named = [
        ("train.sharpe", train.sharpe),
        ("holdout.sharpe", holdout.sharpe),
        ("holdout.total_pnl_twd", holdout.total_pnl_twd),
        ("holdout.trade_count", holdout.trade_count),
    ]
    if z is not None:
        named += [("thresholds.holdout_se_z", z), ("holdout.n_obs", inputs.holdout_n_obs)]
    if require_alpha:
        named += [("benchmark.alpha_twd_per_day", alpha)]
    bad = non_finite(named)
    if bad:
        return invalid(GATE, "non-finite inputs: " + ", ".join(bad))
    if z is not None and (z < 0 or inputs.holdout_n_obs < 2):
        return invalid(
            GATE, f"criterion (b) needs holdout_se_z >= 0 and n_obs >= 2 (z={z}, n_obs={inputs.holdout_n_obs})"
        )

    report = {
        "train_sharpe": float(train.sharpe),
        "holdout_sharpe": float(holdout.sharpe),
        "train_total_pnl_twd": finite_or_none(train.total_pnl_twd),
        "holdout_total_pnl_twd": float(holdout.total_pnl_twd),
        "train_trade_count": train.trade_count,
        "holdout_trade_count": int(holdout.trade_count),
        "holdout_n_obs": int(inputs.holdout_n_obs) if is_finite(inputs.holdout_n_obs) else None,
        "sharpe_decay": None,
        "sharpe_decay_max": decay_max,
        "min_trades": min_trades,
        "pnl_positive_required": require_pnl,
        "criteria": None,
        "passed_criterion": None,
        "alpha_positive_required": require_alpha,
        "benchmark": stats,
    }
    if train.sharpe <= 0:
        return failed(
            GATE,
            "train_not_positive",
            f"train Sharpe {train.sharpe:.3f} <= 0; decay is undefined",
            f"train Sharpe {train.sharpe:.2f}",
            report,
        )
    decay = 1.0 - holdout.sharpe / train.sharpe
    decay_ok = decay <= decay_max
    report["sharpe_decay"] = decay
    criteria = {"decay": {"value": decay, "max": decay_max, "passed": decay_ok}}
    se_ok = False
    if z is None:
        criteria["se"] = {"enabled": False, "passed": None, "note": "holdout_se_z not frozen in this run"}
    else:
        z = float(z)
        se = sharpe_se(float(holdout.sharpe), int(inputs.holdout_n_obs))
        delta = float(train.sharpe) - float(holdout.sharpe)
        allowed = z * se
        se_ok = delta <= allowed  # == holdout_sharpe >= train_sharpe - z * SE
        criteria["se"] = {
            "enabled": True,
            "z": z,
            "se": se,
            "n_obs": int(inputs.holdout_n_obs),
            "delta": delta,
            "allowed_delta": allowed,
            "margin": allowed - delta,
            "passed": se_ok,
            "formula": SE_FORMULA,
        }
    report["criteria"] = criteria
    report["passed_criterion"] = "decay" if decay_ok else ("se" if se_ok else None)
    if holdout.trade_count < min_trades:
        return failed(
            GATE,
            "insufficient_trades",
            f"holdout has {holdout.trade_count} trades < {min_trades}",
            f"{holdout.trade_count} holdout trades",
            report,
        )
    se_phrase = ""
    if z is not None:
        c = criteria["se"]
        sign = "≤" if se_ok else ">"
        se_phrase = f"{z:g}·SE (Δ {c['delta']:.3f} {sign} {c['allowed_delta']:.3f})"
    if not (decay_ok or se_ok):
        reason = f"Sharpe decay {decay:.1%} > {decay_max:.0%}"
        headline = f"Sharpe decay {pct(decay)}"
        if z is not None:
            reason += f" and holdout Sharpe {holdout.sharpe:.3f} < train {train.sharpe:.3f} - {se_phrase}"
            headline += f", beyond {se_phrase}"
        return failed(GATE, "sharpe_decay", reason, headline, report)
    if require_pnl and not holdout.total_pnl_twd > 0:
        return failed(
            GATE,
            "holdout_not_profitable",
            f"holdout pnl {holdout.total_pnl_twd:,.0f} TWD <= 0",
            f"holdout PnL {holdout.total_pnl_twd:,.0f}",
            report,
        )
    alpha_phrase = f", {benchmark_phrase(stats)}" if stats is not None else ""
    if require_alpha and not alpha > 0:
        return failed(
            GATE,
            "holdout_alpha_not_positive",
            f"holdout alpha {alpha:+,.0f} TWD/day <= 0 over TX buy-and-hold ({benchmark_phrase(stats)})",
            f"holdout {benchmark_phrase(stats)}",
            report,
        )
    if decay_ok:
        return passed(GATE, f"Sharpe decay {pct(decay)}{alpha_phrase}", report)
    return passed(
        GATE,
        f"decay {pct(decay)} but within {se_phrase}{alpha_phrase}",
        report,
        reason=f"Sharpe decay {decay:.1%} > {decay_max:.0%}, but holdout Sharpe is within {se_phrase} of train",
    )


# --- daily classes (design 2026-10-24 §7) -----------------------------------------------------
#
# Same Sharpe criterion as above on the daily metric (daily return for ``daily_equity``, TWD PnL
# for ``daily_fixed``), with T = the holdout's trading days. Checks, in order: train Sharpe > 0,
# holdout days >= ``holdout_min_days`` (missing: off), decay or SE criterion, holdout total > 0
# (``holdout_pnl_positive``: final equity > the equity at the holdout start), holdout alpha over
# the 1x underlying > 0 (``holdout_alpha_positive``; ``benchmark.regress_daily`` on the holdout).


@dataclass(frozen=True)
class HoldoutDailyInputs:
    train: DailyResult
    holdout: DailyResult
    basis: Basis
    benchmark: Mapping | None = None  # benchmark.regress_daily on the holdout days (None: not computed)


def evaluate_daily(inputs: HoldoutDailyInputs, thresholds: Mapping) -> GateResult:
    decay_max = float(thresholds["holdout_sharpe_decay_max"])
    require_pnl = bool(thresholds["holdout_pnl_positive"])
    min_days = thresholds.get("holdout_min_days")  # missing -> off
    z = thresholds.get("holdout_se_z")  # missing -> criterion (b) off
    require_alpha = bool(thresholds.get("holdout_alpha_positive", False))
    stats = dict(inputs.benchmark) if inputs.benchmark is not None else None
    alpha = (stats or {}).get("alpha_per_day")
    train_sharpe, holdout_sharpe = sharpe(inputs.train, inputs.basis), sharpe(inputs.holdout, inputs.basis)
    n_days = inputs.holdout.n_days
    named = [
        ("train.sharpe", train_sharpe),
        ("holdout.sharpe", holdout_sharpe),
        ("holdout.total_pnl_twd", inputs.holdout.total_pnl_twd),
    ]
    if z is not None:
        named.append(("thresholds.holdout_se_z", z))
    if min_days is not None:
        named.append(("thresholds.holdout_min_days", min_days))
    if require_alpha:
        named.append(("benchmark.alpha_per_day", alpha))
    bad = non_finite(named)
    if bad:
        return invalid(GATE, "non-finite inputs: " + ", ".join(bad))
    if z is not None and z < 0:
        return invalid(GATE, f"holdout_se_z must be >= 0 (z={z})")
    if require_alpha and stats.get("missing_underlying_days"):
        return invalid(
            GATE, f"{stats['missing_underlying_days']} holdout days have no underlying return", {"benchmark": stats}
        )

    report = {
        "basis": inputs.basis,
        "train_sharpe": float(train_sharpe),
        "holdout_sharpe": float(holdout_sharpe),
        "train_days": inputs.train.n_days,
        "holdout_days": n_days,
        "holdout_start": inputs.holdout.dates[0].isoformat(),
        "train_total_return": finite_or_none(inputs.train.total_return),
        "holdout_total_return": finite_or_none(inputs.holdout.total_return),
        "holdout_total_pnl_twd": float(inputs.holdout.total_pnl_twd),
        "holdout_exposure_changes": inputs.holdout.exposure_changes,
        "sharpe_decay": None,
        "sharpe_decay_max": decay_max,
        "min_days": int(min_days) if min_days is not None else None,
        "pnl_positive_required": require_pnl,
        "criteria": None,
        "passed_criterion": None,
        "alpha_positive_required": require_alpha,
        "benchmark": stats,
    }
    if train_sharpe <= 0:
        return failed(
            GATE,
            "train_not_positive",
            f"train Sharpe {train_sharpe:.3f} <= 0; decay is undefined",
            f"train Sharpe {train_sharpe:.2f}",
            report,
        )
    decay = 1.0 - holdout_sharpe / train_sharpe
    decay_ok = decay <= decay_max
    criteria = {"decay": {"value": decay, "max": decay_max, "passed": decay_ok}}
    se_ok = False
    if z is None:
        criteria["se"] = {"enabled": False, "passed": None, "note": "holdout_se_z not frozen in this run"}
    else:
        se = sharpe_se(float(holdout_sharpe), n_days)
        delta = float(train_sharpe) - float(holdout_sharpe)
        allowed = float(z) * se
        se_ok = delta <= allowed
        criteria["se"] = {
            "enabled": True,
            "z": float(z),
            "se": se,
            "n_obs": n_days,
            "delta": delta,
            "allowed_delta": allowed,
            "margin": allowed - delta,
            "passed": se_ok,
            "formula": SE_FORMULA,
        }
    report.update(
        sharpe_decay=decay, criteria=criteria, passed_criterion="decay" if decay_ok else ("se" if se_ok else None)
    )
    if min_days is not None and n_days < int(min_days):
        return failed(
            GATE,
            "insufficient_days",
            f"holdout has {n_days} trading days < {int(min_days)}",
            f"{n_days} holdout days",
            report,
        )
    se_phrase = ""
    if z is not None:
        c = criteria["se"]
        se_phrase = f"{float(z):g}·SE (Δ {c['delta']:.3f} {'≤' if se_ok else '>'} {c['allowed_delta']:.3f})"
    if not (decay_ok or se_ok):
        reason = f"Sharpe decay {decay:.1%} > {decay_max:.0%}"
        headline = f"Sharpe decay {pct(decay)}"
        if z is not None:
            reason += f" and holdout Sharpe {holdout_sharpe:.3f} < train {train_sharpe:.3f} - {se_phrase}"
            headline += f", beyond {se_phrase}"
        return failed(GATE, "sharpe_decay", reason, headline, report)
    if require_pnl and not inputs.holdout.total_pnl_twd > 0:
        return failed(
            GATE,
            "holdout_not_profitable",
            f"holdout return {inputs.holdout.total_return:+.2%} ({inputs.holdout.total_pnl_twd:,.0f} TWD) <= 0",
            f"holdout return {inputs.holdout.total_return:+.1%}",
            report,
        )
    alpha_phrase = f", {phrase_daily(stats)}" if stats is not None else ""
    if require_alpha and not alpha > 0:
        return failed(
            GATE,
            "holdout_alpha_not_positive",
            f"holdout {phrase_daily(stats)}: alpha <= 0 over the 1x underlying",
            f"holdout {phrase_daily(stats)}",
            report,
        )
    if decay_ok:
        return passed(GATE, f"Sharpe decay {pct(decay)}, {n_days} days{alpha_phrase}", report)
    return passed(
        GATE,
        f"decay {pct(decay)} but within {se_phrase}, {n_days} days{alpha_phrase}",
        report,
        reason=f"Sharpe decay {decay:.1%} > {decay_max:.0%}, but holdout Sharpe is within {se_phrase} of train",
    )
