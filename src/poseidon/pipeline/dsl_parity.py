"""Trade-for-trade parity of a DSL v2 strategy on the v1 runner and the target-position engine (step 7).

:func:`run_parity` backtests one strategy twice through :func:`~poseidon.pipeline.backtesting.run_backtest`
(``backtest.engine`` ``v1`` and ``target_v1``) on the same bars and costs, and :func:`compare_trades`
lines up the two ``trades.csv`` row lists: every trade must have the same side, rule, entry and exit
time, prices, quantity and exit reason, and gross PnL, fees and PnL within ``TOLERANCE_TWD``. A v1
trade held across a settlement-day switch (entry and exit in different R1 contract months) cannot
match: the target engine pays the roll and does not book the gap between the months, so those are
listed apart as ``roll_crossing`` and still fail the check. ``scripts/pv2_dsl_parity.py`` runs it on a
pipeline run's frozen window.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from poseidon.pipeline.backtesting import TARGET_ENGINE, V1_ENGINE, BacktestOutcome, run_backtest, trades_csv

TOLERANCE_TWD = 1.0
EXACT_COLUMNS = (
    "trade_no",
    "side",
    "rule_id",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "exit_reason",
)
MONEY_COLUMNS = ("gross_pnl_twd", "fees_twd", "pnl_twd")


@dataclass(frozen=True)
class ParityReport:
    trades_v1: int
    trades_target: int
    mismatches: tuple[str, ...]
    roll_crossing: tuple[int, ...]  # trade_no of v1 trades held across a settlement switch
    max_money_diff_twd: float  # over trades present in both
    daily_max_abs_diff_twd: float  # daily pnl_twd, NaN when the trading days differ
    seconds: dict[str, float]
    v1: BacktestOutcome
    target: BacktestOutcome

    @property
    def ok(self) -> bool:
        return not self.mismatches and math.isfinite(self.daily_max_abs_diff_twd)

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "trades_v1": self.trades_v1,
            "trades_target": self.trades_target,
            "mismatches": len(self.mismatches),
            "first_mismatches": list(self.mismatches[:20]),
            "roll_crossing_trades": list(self.roll_crossing),
            "max_money_diff_twd": self.max_money_diff_twd,
            "daily_max_abs_diff_twd": self.daily_max_abs_diff_twd,
            "trades_csv_identical": trades_csv(self.v1.trades) == trades_csv(self.target.trades),
            "seconds": self.seconds,
            "total_pnl_twd": {"v1": self.v1.metrics["total_pnl_twd"], "target": self.target.metrics["total_pnl_twd"]},
        }


def compare_trades(
    v1: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    roll_crossing: frozenset[int] = frozenset(),
    tolerance: float = TOLERANCE_TWD,
) -> tuple[list[str], float]:
    """Mismatch lines (empty = identical) and the largest money difference over paired trades."""

    problems = []
    if len(v1) != len(target):
        problems.append(f"trade count: v1 {len(v1)}, target {len(target)}")
    worst = 0.0
    for a, b in zip(v1, target, strict=False):
        tag = " (held across a settlement switch)" if a["trade_no"] in roll_crossing else ""
        for column in EXACT_COLUMNS:
            if a[column] != b[column]:
                problems.append(f"trade {a['trade_no']} {column}: v1 {a[column]!r}, target {b[column]!r}{tag}")
        for column in MONEY_COLUMNS:
            diff = abs(float(a[column]) - float(b[column]))
            worst = max(worst, diff)
            if not diff <= tolerance:
                problems.append(f"trade {a['trade_no']} {column}: v1 {a[column]!r}, target {b[column]!r}{tag}")
    return problems, worst


def crossing_trades(trades: Sequence[Mapping[str, Any]], expiries: Mapping[str, Any]) -> frozenset[int]:
    """``trade_no`` of closed trades whose entry and exit bars belong to different R1 contracts."""

    from poseidon.pipeline.engine.intraday import r1_contracts

    closed = [t for t in trades if t["exit_time"]]
    if not closed:
        return frozenset()
    stamps = pd.DatetimeIndex([t["entry_time"] for t in closed] + [t["exit_time"] for t in closed])
    contracts = r1_contracts(stamps.tz_convert("UTC"), expiries)
    n = len(closed)
    return frozenset(t["trade_no"] for t, a, b in zip(closed, contracts[:n], contracts[n:], strict=True) if a != b)


def run_parity(
    build_strategy: Callable[[], Any],
    bars: pd.DataFrame,
    backtest_config: Mapping[str, Any],
    *,
    expiries: Mapping[str, Any],
    tolerance: float = TOLERANCE_TWD,
    **backtest_kwargs: Any,
) -> ParityReport:
    """Backtest a fresh strategy from ``build_strategy`` on both engines; ``backtest_kwargs`` go to both."""

    base = {key: copy.deepcopy(value) for key, value in backtest_config.items() if key != "engine"}
    started = time.monotonic()
    v1 = run_backtest(build_strategy(), bars, {**base, "engine": V1_ENGINE}, **backtest_kwargs)
    middle = time.monotonic()
    target = run_backtest(
        build_strategy(), bars, {**base, "engine": TARGET_ENGINE}, expiries=expiries, **backtest_kwargs
    )
    seconds = {"v1": round(middle - started, 1), "target": round(time.monotonic() - middle, 1)}
    crossing = crossing_trades(v1.trades, expiries)
    problems, worst = compare_trades(v1.trades, target.trades, roll_crossing=crossing, tolerance=tolerance)
    if v1.daily.index.equals(target.daily.index):
        daily_diff = float((v1.daily["pnl_twd"] - target.daily["pnl_twd"]).abs().max()) if len(v1.daily) else 0.0
        if not daily_diff <= tolerance:
            problems.append(f"daily pnl_twd differs by up to {daily_diff:,.2f} TWD")
    else:
        daily_diff = math.nan
        problems.append(f"daily rows: v1 {len(v1.daily)} trading days, target {len(target.daily)}")
    return ParityReport(
        trades_v1=len(v1.trades),
        trades_target=len(target.trades),
        mismatches=tuple(problems),
        roll_crossing=tuple(sorted(crossing)),
        max_money_diff_twd=worst,
        daily_max_abs_diff_twd=daily_diff,
        seconds=seconds,
        v1=v1,
        target=target,
    )
