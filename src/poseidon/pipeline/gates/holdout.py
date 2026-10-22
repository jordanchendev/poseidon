"""Gate 2 -- locked holdout decay.

``decay = 1 - holdout_sharpe / train_sharpe``. A non-positive train Sharpe has
no meaningful decay and fails ``train_not_positive``. Checks, in order:
train Sharpe > 0, holdout trades >= ``holdout_min_trades``, decay <=
``holdout_sharpe_decay_max``, holdout ``total_pnl_twd`` > 0 (when
``holdout_pnl_positive``). The split itself is enforced upstream by
``backtest.holdout.HoldoutConfig``; this gate only judges the two results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from poseidon.pipeline.gates.base import (
    GateResult,
    ResultSummary,
    failed,
    finite_or_none,
    invalid,
    non_finite,
    passed,
    pct,
)

GATE = "holdout"


@dataclass(frozen=True)
class HoldoutInputs:
    train: ResultSummary
    holdout: ResultSummary


def evaluate(inputs: HoldoutInputs, thresholds: Mapping) -> GateResult:
    decay_max = float(thresholds["holdout_sharpe_decay_max"])
    min_trades = int(thresholds["holdout_min_trades"])
    require_pnl = bool(thresholds["holdout_pnl_positive"])
    train, holdout = inputs.train, inputs.holdout
    bad = non_finite(
        [
            ("train.sharpe", train.sharpe),
            ("holdout.sharpe", holdout.sharpe),
            ("holdout.total_pnl_twd", holdout.total_pnl_twd),
            ("holdout.trade_count", holdout.trade_count),
        ]
    )
    if bad:
        return invalid(GATE, "non-finite inputs: " + ", ".join(bad))

    report = {
        "train_sharpe": float(train.sharpe),
        "holdout_sharpe": float(holdout.sharpe),
        "train_total_pnl_twd": finite_or_none(train.total_pnl_twd),
        "holdout_total_pnl_twd": float(holdout.total_pnl_twd),
        "train_trade_count": train.trade_count,
        "holdout_trade_count": int(holdout.trade_count),
        "sharpe_decay": None,
        "sharpe_decay_max": decay_max,
        "min_trades": min_trades,
        "pnl_positive_required": require_pnl,
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
    report["sharpe_decay"] = decay
    if holdout.trade_count < min_trades:
        return failed(
            GATE,
            "insufficient_trades",
            f"holdout has {holdout.trade_count} trades < {min_trades}",
            f"{holdout.trade_count} holdout trades",
            report,
        )
    if decay > decay_max:
        return failed(
            GATE, "sharpe_decay", f"Sharpe decay {decay:.1%} > {decay_max:.0%}", f"Sharpe decay {pct(decay)}", report
        )
    if require_pnl and not holdout.total_pnl_twd > 0:
        return failed(
            GATE,
            "holdout_not_profitable",
            f"holdout pnl {holdout.total_pnl_twd:,.0f} TWD <= 0",
            f"holdout PnL {holdout.total_pnl_twd:,.0f}",
            report,
        )
    return passed(GATE, f"Sharpe decay {pct(decay)}", report)
