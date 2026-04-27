"""Phase 85 — pure helper formulas for Optuna+WFE verdict_inputs.

These four helpers exist because the existing ``backtest/metrics.py`` and
``backtest/walk_forward.py`` do NOT match D-16 / D-17 / D-19 / Pitfall 6
(see RESEARCH.md gaps 1, 2, 3, 4, 6 + Pitfall 9 for serialization).

Centralized here so Phase 86 can audit ONE file for verdict math. All
helpers are pure (no I/O, no global state) and return raw values — no
PASS/FAIL booleans (those belong to Phase 86).

Public surface:

* ``compute_max_consecutive_losses(trades)`` — D-19 longest losing streak
  over closed trades only (open positions skipped).
* ``wfe_degradation_excluding_is_negative(per_window)`` — D-16 mean of
  ``oos_sharpe / is_sharpe`` over windows where IS Sharpe > 0; ``None``
  when every window has IS Sharpe ≤ 0 (verdict undefined).
* ``oos_aggregate_sharpe_trade_weighted(per_window)`` — D-17 trade-count
  weighted mean OOS Sharpe; safe when total trades == 0.
* ``to_jsonable(obj)`` — Pitfall 9 numpy/pandas serialization helper for
  ``json.dumps(payload, default=to_jsonable)``.
* ``BARS_PER_YEAR_1M`` — Pitfall 6 constant (525_600 1m bars per year).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

#: 1m bars per calendar year — Pitfall 6. Pass to ``compute_metrics(bars_per_year=...)``.
BARS_PER_YEAR_1M: int = 525_600


def compute_max_consecutive_losses(trades: Iterable) -> int:
    """D-19: longest run of consecutive negative-PnL closed trades.

    Open positions (``exit_time is None``) are SKIPPED — they neither
    extend nor break the streak. Zero-PnL closed trades break the streak
    (treated as "not a loss"). Empty input returns 0.

    Accepts both dataclass-like trade records (``.exit_time``, ``.pnl``)
    and dict-shaped records (``{"exit_time": ..., "pnl": ...}``) so the
    Phase 85 driver can pass whichever representation is convenient.
    """
    best = 0
    streak = 0
    for trade in trades:
        if isinstance(trade, dict):
            exit_time = trade.get("exit_time")
            pnl = trade.get("pnl", 0.0)
        else:
            exit_time = getattr(trade, "exit_time", None)
            pnl = getattr(trade, "pnl", 0.0)
        if exit_time is None:
            # D-19: open position — neither extends nor breaks the streak.
            continue
        if pnl is None:
            # Defensive: treat missing pnl on a closed trade as "not a loss".
            streak = 0
            continue
        if pnl < 0:
            streak += 1
            if streak > best:
                best = streak
        else:
            streak = 0
    return best


def wfe_degradation_excluding_is_negative(per_window) -> float | None:
    """D-16: mean of ``oos_sharpe / is_sharpe`` over windows with IS > 0.

    Returns ``None`` if every window has ``is_sharpe <= 0`` (verdict
    undefined — distinct from 0.0 which would silently signal "perfect
    degradation").

    Each ``per_window`` element must expose ``.is_metrics`` and
    ``.oos_metrics`` (dataclass-like) or be a dict with those keys.
    Within each metrics dict the only required key is ``sharpe_ratio``.
    """
    ratios: list[float] = []
    for window in per_window:
        is_metrics = (
            window.is_metrics if hasattr(window, "is_metrics") else window["is_metrics"]
        )
        oos_metrics = (
            window.oos_metrics if hasattr(window, "oos_metrics") else window["oos_metrics"]
        )
        is_sharpe = float(is_metrics.get("sharpe_ratio", 0.0))
        oos_sharpe = float(oos_metrics.get("sharpe_ratio", 0.0))
        if is_sharpe > 0:
            ratios.append(oos_sharpe / is_sharpe)
    if not ratios:
        return None
    return float(sum(ratios) / len(ratios))


def oos_aggregate_sharpe_trade_weighted(per_window) -> float:
    """D-17: trade-count-weighted mean OOS Sharpe across windows.

    Returns 0.0 if total OOS trades == 0 (no ``ZeroDivisionError``).
    Zero-trade windows contribute zero to BOTH numerator and denominator
    so a freak ``sharpe=∞`` from an empty window cannot inflate the
    aggregate.

    Each window's ``oos_metrics`` must contain ``sharpe_ratio`` and a
    ``trades`` count (alias ``trade_count`` is also accepted because the
    Poseidon ``compute_metrics`` output uses ``trade_count``).
    """
    numerator = 0.0
    denominator = 0
    for window in per_window:
        oos_metrics = (
            window.oos_metrics if hasattr(window, "oos_metrics") else window["oos_metrics"]
        )
        sharpe = float(oos_metrics.get("sharpe_ratio", 0.0))
        # Poseidon's compute_metrics returns "trade_count"; D-17 spec uses
        # "trades". Accept either so the driver can pass raw window output.
        if "trades" in oos_metrics:
            trades = int(oos_metrics["trades"])
        else:
            trades = int(oos_metrics.get("trade_count", 0))
        if trades <= 0:
            continue
        numerator += sharpe * trades
        denominator += trades
    if denominator == 0:
        return 0.0
    return numerator / denominator


def to_jsonable(obj):
    """JSON serializer for numpy / pandas scalars + timestamps + arrays.

    Pass as ``json.dumps(payload, default=to_jsonable)``. Pitfall 9.

    NaN floats are converted to ``None`` because JSON has no NaN literal —
    Phase 86 verdict logic must never see the string ``"NaN"`` in
    artifact files. Inf floats are likewise converted to ``None``.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if hasattr(obj, "isoformat"):  # datetime.datetime / datetime.date
        return obj.isoformat()
    if hasattr(obj, "to_dict"):  # pydantic / dataclass-like
        return obj.to_dict()
    raise TypeError(
        f"phase85_metrics.to_jsonable: unsupported {type(obj)!r}"
    )


__all__ = [
    "BARS_PER_YEAR_1M",
    "compute_max_consecutive_losses",
    "wfe_degradation_excluding_is_negative",
    "oos_aggregate_sharpe_trade_weighted",
    "to_jsonable",
]
