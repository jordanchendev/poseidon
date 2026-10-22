"""DSL v2 condition evaluation over an incremental numpy column cache.

The evaluator only reads rows ``<= row`` of the frame it is handed, so it is
causal by construction. A rule whose operands (or session columns) are NaN at
the bars it needs does not fire at all, whatever combinator wraps them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from poseidon.strategies.dsl.v2.schema import (
    AllNode,
    AnyNode,
    AtrOffset,
    AtrStop,
    AtrTarget,
    AtrTrailing,
    CompareCondition,
    ConstOperand,
    CrossCondition,
    IndicatorOperand,
    NoneNode,
    OpeningRangeOperand,
    PointsOffset,
    PriceOperand,
    SessionElapsedCondition,
    SessionLevelOperand,
    StrategyV2,
    TimeWindowCondition,
)
from poseidon.strategies.dsl.v2.validate import walk_model

DEFAULT_BAR_LABEL = "open"
SESSION_ID = "s_id"
SESSION_KIND = "s_kind"
SESSION_TOD = "s_tod"
SESSION_MIN_OPEN = "s_min_open"
SESSION_MIN_CLOSE = "s_min_close"
SESSION_KIND_CODE = {"day": (1.0,), "night": (2.0,), "any": (1.0, 2.0)}


def operand_column(op) -> str | None:
    """Feature column an operand reads (None for a constant)."""
    if isinstance(op, PriceOperand):
        return op.price
    if isinstance(op, IndicatorOperand):
        return f"{op.ind}_{op.period}"
    if isinstance(op, OpeningRangeOperand):
        return f"s_or_{op.session[3:]}_{op.minutes}"
    if isinstance(op, SessionLevelOperand):
        return f"s_{op.session}"
    if isinstance(op, ConstOperand):
        return None
    raise TypeError(f"not an operand: {op!r}")


def atr_column(period: int) -> str:
    return f"atr_{period}"


def derive_feature_specs(spec: StrategyV2, bar_label: str = DEFAULT_BAR_LABEL) -> list[tuple[str, dict]]:
    """FeatureEngine specs needed by every operand, offset and risk block of ``spec``."""
    session_params = {"interval_minutes": spec.interval_minutes, "bar_label": bar_label}
    specs: list[tuple[str, dict]] = [("tx_session", dict(session_params))]

    def add(name: str, params: dict) -> None:
        if (name, params) not in specs:
            specs.append((name, params))

    for _path, node in walk_model(spec):
        if isinstance(node, IndicatorOperand):
            add(node.ind, {"period": node.period})
        elif isinstance(node, OpeningRangeOperand):
            add("opening_range", {"minutes": node.minutes, **session_params})
        elif isinstance(node, SessionLevelOperand):
            add("session_levels", dict(session_params))
        elif isinstance(node, AtrOffset | AtrStop | AtrTarget | AtrTrailing):
            add("atr", {"period": node.atr_period})
    return specs


class ColumnCache:
    """Incremental float64 copies of selected columns.

    ``sync(frame)`` appends the newest row when ``frame`` extends the cached
    prefix by one bar (the backtest case, O(1) per bar) and rebuilds from the
    frame otherwise (live evaluation, a new run). It never reads past the
    frame it is given.
    """

    def __init__(self, columns: list[str]):
        self.columns = list(dict.fromkeys(columns))
        self._data: dict[str, np.ndarray] = {}
        self._len = 0
        self._first = None
        self._last = None
        self._positions = np.array([], dtype=np.intp)
        self.index: pd.Index = pd.Index([])

    def __len__(self) -> int:
        return self._len

    def _rebuild(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"feature columns missing for DSL v2 evaluation: {missing}")
        n = len(frame)
        cap = max(16, n * 2)
        self._data = {}
        for col in self.columns:
            arr = np.full(cap, np.nan)
            arr[:n] = frame[col].to_numpy(dtype=float, na_value=np.nan)
            self._data[col] = arr
        self._positions = np.array([frame.columns.get_loc(c) for c in self.columns], dtype=np.intp)
        self._len = n

    @staticmethod
    def _keys(index: pd.Index):
        """Cheap comparable view of the index (int64 nanoseconds for datetimes)."""
        return index.asi8 if isinstance(index, pd.DatetimeIndex) else index

    def sync(self, frame: pd.DataFrame) -> int:
        """Bring the cache in line with ``frame``; return the last row index.

        ``self.index`` is the index of the frame last synced (rows beyond it
        are never read).
        """
        n = len(frame)
        self.index = frame.index
        if n == 0:
            self._len = 0
            return -1
        keys = self._keys(frame.index)
        first, last = keys[0], keys[n - 1]
        if self._len and first == self._first and n == self._len and last == self._last:
            return n - 1
        if self._len and first == self._first and n == self._len + 1 and keys[n - 2] == self._last:
            row = n - 1
            # One row read for all cached columns (iat per column is ~10x slower).
            values = frame.iloc[row].to_numpy()[self._positions]
            for col, value in zip(self.columns, values, strict=True):
                arr = self._data[col]
                if row >= len(arr):
                    grown = np.full(len(arr) * 2, np.nan)
                    grown[: len(arr)] = arr
                    self._data[col] = arr = grown
                arr[row] = np.nan if value is None or value is pd.NA else float(value)
            self._len = n
        else:
            self._rebuild(frame)
        self._first, self._last = first, last
        return n - 1

    def get(self, col: str, row: int) -> float:
        if row < 0 or row >= self._len:
            return math.nan
        return float(self._data[col][row])


@dataclass(frozen=True)
class _Dependency:
    column: str
    lag: int


class CompiledCondition:
    """A rule condition plus the (column, lag) cells it needs to be non-NaN."""

    def __init__(self, condition):
        self.condition = condition
        deps: list[_Dependency] = [_Dependency(SESSION_ID, 0)]
        self._collect(condition, deps)
        self.dependencies = tuple(dict.fromkeys(deps))

    @property
    def columns(self) -> list[str]:
        return [d.column for d in self.dependencies]

    def _collect(self, node, deps: list[_Dependency]) -> None:
        if isinstance(node, AllNode | AnyNode | NoneNode):
            for child in _children(node):
                self._collect(child, deps)
            return
        if isinstance(node, CrossCondition | CompareCondition):
            lags = (0, 1) if isinstance(node, CrossCondition) else (0,)
            cols = [operand_column(node.a), operand_column(node.b)]
            if isinstance(node.offset, AtrOffset):
                cols.append(atr_column(node.offset.atr_period))
            for col in cols:
                if col is not None:
                    for lag in lags:
                        deps.append(_Dependency(col, lag))
            return
        if isinstance(node, TimeWindowCondition):
            deps.extend([_Dependency(SESSION_KIND, 0), _Dependency(SESSION_TOD, 0)])
            return
        if isinstance(node, SessionElapsedCondition):
            deps.append(_Dependency(SESSION_MIN_OPEN, 0))
            return
        raise TypeError(f"unknown condition node {node!r}")

    def evaluate(self, cache: ColumnCache, row: int) -> bool:
        for dep in self.dependencies:
            if math.isnan(cache.get(dep.column, row - dep.lag)):
                return False
        return _eval(self.condition, cache, row)


def _children(node) -> list:
    if isinstance(node, AllNode):
        return node.all
    if isinstance(node, AnyNode):
        return node.any
    return node.none


def operand_value(op, cache: ColumnCache, row: int) -> float:
    if isinstance(op, ConstOperand):
        return float(op.const)
    return cache.get(operand_column(op), row)


def offset_value(offset, cache: ColumnCache, row: int) -> float:
    if offset is None:
        return 0.0
    if isinstance(offset, PointsOffset):
        return float(offset.points)
    return float(offset.atr_mult) * cache.get(atr_column(offset.atr_period), row)


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _eval(node, cache: ColumnCache, row: int) -> bool:
    if isinstance(node, AllNode):
        return all(_eval(c, cache, row) for c in node.all)
    if isinstance(node, AnyNode):
        return any(_eval(c, cache, row) for c in node.any)
    if isinstance(node, NoneNode):
        return not any(_eval(c, cache, row) for c in node.none)
    if isinstance(node, CrossCondition):
        sign = 1.0 if node.direction == "above" else -1.0
        a0, a1 = operand_value(node.a, cache, row - 1), operand_value(node.a, cache, row)
        b0 = operand_value(node.b, cache, row - 1) + sign * offset_value(node.offset, cache, row - 1)
        b1 = operand_value(node.b, cache, row) + sign * offset_value(node.offset, cache, row)
        if node.direction == "above":
            return a0 <= b0 and a1 > b1
        return a0 >= b0 and a1 < b1
    if isinstance(node, CompareCondition):
        a = operand_value(node.a, cache, row)
        sign = 1.0 if node.op in (">", ">=") else -1.0
        b = operand_value(node.b, cache, row) + sign * offset_value(node.offset, cache, row)
        if node.op == ">":
            return a > b
        if node.op == ">=":
            return a >= b
        if node.op == "<":
            return a < b
        return a <= b
    if isinstance(node, TimeWindowCondition):
        if cache.get(SESSION_KIND, row) not in SESSION_KIND_CODE[node.session]:
            return False
        tod = cache.get(SESSION_TOD, row)
        start, end = _minutes(node.start), _minutes(node.end)
        if start <= end:
            return start <= tod <= end
        return tod >= start or tod <= end
    if isinstance(node, SessionElapsedCondition):
        elapsed = cache.get(SESSION_MIN_OPEN, row)
        if node.min_minutes is not None and elapsed < node.min_minutes:
            return False
        return not (node.max_minutes is not None and elapsed > node.max_minutes)
    raise TypeError(f"unknown condition node {node!r}")
