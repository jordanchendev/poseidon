"""Gate 1 -- walk-forward efficiency.

WFE follows ``backtest.walk_forward`` exactly: the mean IS annualized return
and the mean OOS annualized return across windows go through ``compute_wfe``
(0.0 when the IS mean is not positive). Pass iff WFE >= ``wfe_min`` and the
summed OOS ``total_pnl_twd`` across windows is > 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from poseidon.backtest.walk_forward import compute_wfe
from poseidon.pipeline.gates.base import GateResult, failed, finite_or_none, invalid, non_finite, passed

GATE = "wfe"


@dataclass(frozen=True)
class WindowMetrics:
    index: int
    is_annualized_return: float
    oos_annualized_return: float
    is_total_pnl_twd: float
    oos_total_pnl_twd: float
    is_trade_count: int
    oos_trade_count: int


@dataclass(frozen=True)
class WfeInputs:
    windows: Sequence[WindowMetrics]


def _window_row(window: WindowMetrics) -> dict:
    return {
        "index": window.index,
        "is_annualized_return": finite_or_none(window.is_annualized_return),
        "oos_annualized_return": finite_or_none(window.oos_annualized_return),
        "wfe": finite_or_none(compute_wfe(window.is_annualized_return, window.oos_annualized_return))
        if not non_finite([("is", window.is_annualized_return), ("oos", window.oos_annualized_return)])
        else None,
        "is_total_pnl_twd": finite_or_none(window.is_total_pnl_twd),
        "oos_total_pnl_twd": finite_or_none(window.oos_total_pnl_twd),
        "is_trade_count": window.is_trade_count,
        "oos_trade_count": window.oos_trade_count,
    }


def evaluate(inputs: WfeInputs, thresholds: Mapping) -> GateResult:
    wfe_min = float(thresholds["wfe_min"])
    windows = sorted(inputs.windows, key=lambda window: window.index)
    if not windows:
        return invalid(GATE, "no walk-forward windows", {"windows": [], "wfe_min": wfe_min})
    bad = [
        f"window {window.index}.{name}"
        for window in windows
        for name in non_finite(
            [
                ("is_annualized_return", window.is_annualized_return),
                ("oos_annualized_return", window.oos_annualized_return),
                ("oos_total_pnl_twd", window.oos_total_pnl_twd),
            ]
        )
    ]
    if bad:
        return invalid(
            GATE, "non-finite window metrics: " + ", ".join(bad), {"windows": [_window_row(w) for w in windows]}
        )

    mean_is = sum(window.is_annualized_return for window in windows) / len(windows)
    mean_oos = sum(window.oos_annualized_return for window in windows) / len(windows)
    wfe = compute_wfe(mean_is, mean_oos)
    oos_pnl = sum(window.oos_total_pnl_twd for window in windows)
    report = {
        "wfe": wfe,
        "wfe_min": wfe_min,
        "mean_is_annualized_return": mean_is,
        "mean_oos_annualized_return": mean_oos,
        "aggregate_oos_total_pnl_twd": oos_pnl,
        "oos_trade_count": sum(window.oos_trade_count for window in windows),
        "windows": [_window_row(window) for window in windows],
    }
    headline = f"WFE {wfe:.2f}"
    if wfe < wfe_min:
        detail = " (mean IS return not positive)" if mean_is <= 0 else ""
        return failed(
            GATE, "wfe_below_min", f"WFE {wfe:.3f} < {wfe_min}{detail}", f"WFE {wfe:.2f} < {wfe_min:g}", report
        )
    if not oos_pnl > 0:
        return failed(
            GATE, "oos_not_profitable", f"aggregate OOS pnl {oos_pnl:,.0f} TWD <= 0", f"OOS PnL {oos_pnl:,.0f}", report
        )
    return passed(GATE, headline, report)
