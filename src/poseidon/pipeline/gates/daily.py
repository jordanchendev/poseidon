"""Engine outputs reduced to what the daily-class gates judge (design 2026-10-24 §7).

A daily gate never sees trades. It sees one :class:`DailyResult` per engine run (or a date
window of one): the equity at each trading day's close, the costs of that day's fills, the
engine events, and the ruin date. The metric series follows the frozen ``strategy_class``:

* ``daily_equity`` -> basis ``"return"``: daily return = pnl / previous equity;
* ``daily_fixed``  -> basis ``"twd"``: daily PnL in TWD (the v1 convention).

Sharpe, skew and kurtosis use the very formulas of ``backtesting.daily_stats`` (Sharpe per
trading day, std ddof=1, biased skew, non-excess biased kurtosis), so a trial ledger row built
from :func:`moments` is comparable with the DSR's selected Sharpe. "Days" are rows of the
engine's ``daily_pnl`` (trading days); walk-forward windows are measured in calendar days, as
the intraday ``wfe`` stage does.

Re-runs (``cost_stress``, ``execution_sensitivity``) go through a :data:`Rerun` callable the
stage supplies; the gates only say what they need re-run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from poseidon.pipeline.gates.base import CostBreakdown

Basis = Literal["return", "twd"]
Execution = Literal["next_open", "signal_close"]
BASIS_BY_CLASS: dict[str, Basis] = {"daily_equity": "return", "daily_fixed": "twd"}
TRADING_DAYS_PER_YEAR = 252.0
LIQUIDATION_EVENTS = ("forced_liquidation", "ruin")


def basis_for(strategy_class: str) -> Basis:
    try:
        return BASIS_BY_CLASS[strategy_class]
    except KeyError:
        raise KeyError(f"{strategy_class!r} is not a daily strategy class") from None


@dataclass(frozen=True)
class EngineEvent:
    day: date
    kind: str  # forced_liquidation | ruin | margin_call | roll | clip_leverage | clip_margin | ...
    note: str = ""


@dataclass(frozen=True)
class DailyResult:
    """One engine run over consecutive trading days.

    ``equity[i]`` is the equity at the close of ``dates[i]``; ``initial_equity`` the equity before
    ``dates[0]`` (the initial capital for a whole run). ``commission`` / ``tax`` / ``slippage`` are
    the TWD costs of the fills on each day and ``fill_days`` the days with at least one fill (any
    reason): the daily replacement for a trade count. ``ruined`` is the day equity reached 0.
    """

    dates: tuple[date, ...]
    equity: tuple[float, ...]
    initial_equity: float
    commission: tuple[float, ...]
    tax: tuple[float, ...]
    slippage: tuple[float, ...]
    fill_days: frozenset[date] = frozenset()
    events: tuple[EngineEvent, ...] = ()
    ruined: date | None = None

    def __post_init__(self) -> None:
        n = len(self.dates)
        if not (len(self.equity) == len(self.commission) == len(self.tax) == len(self.slippage) == n):
            raise ValueError("dates, equity and the cost columns must be aligned")
        if any(b <= a for a, b in zip(self.dates, self.dates[1:], strict=False)):
            raise ValueError("dates must be strictly increasing")

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def previous_equity(self) -> np.ndarray:
        return np.concatenate([[self.initial_equity], np.asarray(self.equity[:-1], dtype=float)])

    @property
    def pnl_twd(self) -> np.ndarray:
        return np.asarray(self.equity, dtype=float) - self.previous_equity

    @property
    def returns(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.pnl_twd / self.previous_equity

    def metric(self, basis: Basis) -> np.ndarray:
        if basis == "return":
            return self.returns
        if basis == "twd":
            return self.pnl_twd
        raise ValueError(f"unknown basis {basis!r}")

    @property
    def final_equity(self) -> float:
        return float(self.equity[-1]) if self.equity else float(self.initial_equity)

    @property
    def total_pnl_twd(self) -> float:
        return self.final_equity - float(self.initial_equity)

    @property
    def total_return(self) -> float:
        return self.final_equity / float(self.initial_equity) - 1.0

    @property
    def exposure_changes(self) -> int:
        return sum(1 for day in self.dates if day in self.fill_days)

    @property
    def costs(self) -> CostBreakdown:
        return CostBreakdown(
            commission_twd=float(sum(self.commission)),
            tax_twd=float(sum(self.tax)),
            slippage_twd=float(sum(self.slippage)),
        )

    def window(self, start: date | None = None, end: date | None = None) -> DailyResult:
        """Days in ``[start, end)``; the equity before the first kept day becomes ``initial_equity``."""

        keep = [i for i, day in enumerate(self.dates) if (start is None or day >= start) and (end is None or day < end)]
        if not keep:
            return DailyResult((), (), self.final_equity, (), (), (), frozenset(), (), None)
        first, last = keep[0], keep[-1] + 1
        initial = self.initial_equity if first == 0 else self.equity[first - 1]
        days = self.dates[first:last]
        lo, hi = days[0], days[-1]
        return DailyResult(
            dates=days,
            equity=self.equity[first:last],
            initial_equity=float(initial),
            commission=self.commission[first:last],
            tax=self.tax[first:last],
            slippage=self.slippage[first:last],
            fill_days=frozenset(day for day in self.fill_days if lo <= day <= hi),
            events=tuple(event for event in self.events if lo <= event.day <= hi),
            ruined=self.ruined if self.ruined is not None and lo <= self.ruined <= hi else None,
        )

    def event_days(self, kinds: Iterable[str]) -> list[dict]:
        wanted = set(kinds)
        return [{"date": e.day.isoformat(), "kind": e.kind, "note": e.note} for e in self.events if e.kind in wanted]


def _day(value) -> date:
    return pd.Timestamp(value).date()


def from_engine(result) -> DailyResult:
    """Map a ``pipeline.engine.EngineResult`` (§5.8 frames) onto :class:`DailyResult`.

    Reads ``daily_pnl[date, equity]``, ``fills[date, commission, tax, slippage_twd]``,
    ``events[date, kind, note]``, ``ruined`` and ``config.initial_capital`` (the first
    ``daily_pnl`` row's pnl is measured from it).
    """

    daily = result.daily_pnl
    dates = tuple(_day(value) for value in daily["date"])
    per_day = {day: [0.0, 0.0, 0.0] for day in dates}
    fill_days: set[date] = set()
    for row in result.fills.itertuples(index=False):
        day = _day(row.date)
        fill_days.add(day)
        if day in per_day:
            costs = per_day[day]
            costs[0] += float(row.commission)
            costs[1] += float(row.tax)
            costs[2] += float(row.slippage_twd)
    events = tuple(
        EngineEvent(_day(row.date), str(row.kind), "" if pd.isna(row.note) else str(row.note))
        for row in result.events.itertuples(index=False)
    )
    return DailyResult(
        dates=dates,
        equity=tuple(float(value) for value in daily["equity"]),
        initial_equity=float(result.config.initial_capital),
        commission=tuple(per_day[day][0] for day in dates),
        tax=tuple(per_day[day][1] for day in dates),
        slippage=tuple(per_day[day][2] for day in dates),
        fill_days=frozenset(fill_days),
        events=events,
        ruined=_day(result.ruined) if result.ruined is not None else None,
    )


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def moments(values: Sequence[float], annualisation: float = TRADING_DAYS_PER_YEAR) -> dict:
    """``backtesting.daily_stats`` on any daily series: n_obs, sharpe_daily, skew, kurtosis (non-excess).

    The keys are the ones ``trials.make_record`` reads, so a daily trial goes into the ledger unchanged.
    """

    array = np.asarray(values, dtype=float)
    n_obs = len(array)
    stats = {
        "n_obs": n_obs,
        "mean_daily": _finite(array.mean()) if n_obs else None,
        "std_daily": None,
        "sharpe_daily": None,
        "sharpe_annualised": None,
        "skew": None,
        "kurtosis": None,
    }
    if n_obs < 2 or not np.isfinite(array).all():
        return stats
    std = float(array.std(ddof=1))
    stats["std_daily"] = _finite(std)
    if not std > 0:
        return stats
    from scipy.stats import kurtosis, skew

    sharpe = float(array.mean()) / std
    stats["sharpe_daily"] = _finite(sharpe)
    stats["sharpe_annualised"] = _finite(sharpe * math.sqrt(annualisation))
    stats["skew"] = _finite(skew(array, bias=True))
    stats["kurtosis"] = _finite(kurtosis(array, fisher=False, bias=True))
    return stats


def sharpe(result: DailyResult, basis: Basis) -> float | None:
    return moments(result.metric(basis))["sharpe_daily"]


def annualised_return(result: DailyResult, basis: Basis) -> float | None:
    """Mean daily metric x 252, the linear rule of ``gate_stages.annualised_return``: return basis
    mean daily return x 252; TWD basis pnl / initial equity x 252 / days.

    Linear on purpose: WFE is a ratio of IS and OOS means, and a compounded rate (CAGR) is convex
    in the daily mean, so at 3x leverage it shrinks the OOS/IS ratio of an unchanged edge.
    """

    n = result.n_days
    if not n or not result.initial_equity > 0:
        return None
    if basis == "twd":
        return result.total_pnl_twd / result.initial_equity * TRADING_DAYS_PER_YEAR / n
    return float(np.mean(result.returns)) * TRADING_DAYS_PER_YEAR


def underlying_returns(close: pd.Series) -> dict[date, float]:
    """Simple daily returns of the 1x underlying from its (back-adjusted) close, keyed by trading day."""

    series = close.sort_index().astype(float)
    returns = series / series.shift(1) - 1.0
    return {_day(index): float(value) for index, value in returns.iloc[1:].items()}


def align(result: DailyResult, benchmark: Mapping[date, float], basis: Basis) -> tuple[list, list, list[date]]:
    """``(strategy, benchmark, missing)``: the metric and the underlying return on the result's days;
    ``missing`` lists result days the underlying has no return for (the caller fails on any)."""

    metric = result.metric(basis)
    y, x, missing = [], [], []
    for day, value in zip(result.dates, metric, strict=True):
        if day in benchmark:
            y.append(float(value))
            x.append(float(benchmark[day]))
        else:
            missing.append(day)
    return y, x, missing


def walk_forward_windows(train_end: date, *, windows: int, is_days: int, oos_days: int) -> list[dict]:
    """Rolling IS/OOS windows in calendar days, anchored at ``train_end`` (exclusive), most recent last.

    Window ``i``: IS ``[is_start, oos_start)``, OOS ``[oos_start, oos_end)``.
    """

    out = []
    for index in range(windows):
        oos_end = train_end - timedelta(days=oos_days * (windows - 1 - index))
        oos_start = oos_end - timedelta(days=oos_days)
        out.append(
            {
                "index": index,
                "is_start": oos_start - timedelta(days=is_days),
                "oos_start": oos_start,
                "oos_end": oos_end,
            }
        )
    return out


def choose_walk_forward(first_day: date, train_end: date, layouts: Sequence[Mapping]) -> dict | None:
    """The first frozen layout whose earliest IS window starts on or after ``first_day``, with its windows."""

    for layout in layouts:
        windows = walk_forward_windows(
            train_end, windows=int(layout["windows"]), is_days=int(layout["is_days"]), oos_days=int(layout["oos_days"])
        )
        if windows[0]["is_start"] >= first_day:
            return {"layout": dict(layout), "windows": windows}
    return None


@dataclass(frozen=True)
class RerunRequest:
    """What a daily gate needs re-run: the same spec over the same window with ``params``.

    ``cost_multiplier`` scales every cost component of every instrument (commission, tax, slippage);
    ``execution`` replaces the spec's execution component (None: keep it).
    """

    params: Mapping[str, Any]
    cost_multiplier: float = 1.0
    execution: Execution | None = None


Rerun = Callable[[RerunRequest], DailyResult]
