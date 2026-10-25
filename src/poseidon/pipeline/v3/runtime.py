"""Runtime of a compiled spec v3 (design 2026-10-24 §3.3, §5.1): what a component sees, and the
``TargetStrategy`` that runs the components of every sleeve at each decision close.

Per decision, per sleeve: direction ``d`` -> leverage chain ``L`` (each link gets the previous
link's output, the first gets 0.0) -> allocation weight ``w`` -> ``sizing.units(equity, d, L, w,
price = today's close of the held contract)`` -> ``rebalance.should_trade`` decides whether that
target becomes an order (left out of the ``Decision`` = keep the position). The engine then
clips by margin and leverage caps and fills per ``execution``.

Components only ever get a :class:`SleeveView` (arrays up to the decision day, read-only) and
their own state, so no registry component can read the future. A series ref resolves to an
instrument's series (``self:adjusted``, ``core:close``) or to an index aux series
(``index:taiex``, ``index:taiex_tr``: the aux of that name, column ``close``). :class:`Conventions`
are the knobs a spec cannot set: the levtx parity run and the execution-timing gate's re-runs
(``gates.execution_timing``: other fill, signal one day late) use them; a pipeline run uses the
defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import BookView, Decision, MarketView
from poseidon.pipeline.v3.instruments import INDEX_ALIAS, Instrument
from poseidon.pipeline.v3.registry import Component, StrictParams

# spec series field -> engine SeriesView field
SERIES_FIELD = {"open": "open", "high": "high", "low": "low", "close": "close", "adjusted": "adjusted_close"}


@dataclass(frozen=True)
class Conventions:
    """Run conventions outside the spec. Defaults = the spec as written (§12 defaults).

    ``execution`` overrides the spec's execution component (``signal_close`` = fill at the
    decision close: the levtx convention and an execution-timing scenario). ``signal_lag`` drops
    that many latest rows from every signal series: the decision at close t reads signals up to
    t - lag (levtx decides from index data up to t-1: lag 1; the execution-timing gate's "signal
    one day late" scenario is ``RerunRequest.signal_lag_extra = 1``).
    """

    execution: Literal["next_open", "signal_close"] | None = None
    signal_lag: int = 0

    def __post_init__(self) -> None:
        if self.signal_lag < 0:
            raise ValueError("signal_lag must be >= 0")


class Series(NamedTuple):
    dates: np.ndarray  # datetime64[ns], read-only
    values: np.ndarray  # float, read-only


class SleeveView:
    """One sleeve's window on the market at a decision close: series refs resolve to arrays
    ending at (or, with ``signal_lag``, before) the decision day. ``book`` is the engine's read-only
    book at that close (``direction.dsl_rules`` reads its position and fills from it)."""

    def __init__(self, market: MarketView, alias: str, conventions: Conventions, book: BookView | None = None) -> None:
        self._market = market
        self.alias = alias
        self._conventions = conventions
        self.book = book

    @property
    def date(self) -> pd.Timestamp:
        return self._market.date

    def series(self, ref: str) -> Series:
        alias, name = ref.split(":")
        if alias == INDEX_ALIAS:
            view = self._market.aux(name)
            dates, values = view.dates, view["close"]
        else:
            view = self._market[self.alias if alias == "self" else alias]
            dates, values = view.dates, getattr(view, SERIES_FIELD[name])
        lag = self._conventions.signal_lag
        if lag:
            keep = max(len(values) - lag, 0)
            dates, values = dates[:keep], values[:keep]
        return Series(dates, values)


@dataclass(frozen=True)
class RebalanceContext:
    date: pd.Timestamp
    rolled_today: bool  # the held contract changed at this close
    target_leverage: float  # d x L x weight (signed)
    actual_leverage: float  # current units x close x point value / equity (signed)


@dataclass(frozen=True)
class RollCalendar:
    trading_days: pd.DatetimeIndex
    expiries: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class Bound:
    """A registered component with its parsed parameters."""

    component: Any  # a registry Component subclass instance
    params: StrictParams


@dataclass(frozen=True)
class SleevePlan:
    id: str
    alias: str  # instrument alias = engine instrument name
    instrument: Instrument
    direction: Bound
    leverage: tuple[Bound, ...]
    sizing: Bound
    rebalance: Bound


class SpecStrategy:
    """A compiled spec v3 as an engine ``TargetStrategy``. Build a fresh one per run.

    ``trace`` gets one row per sleeve per decision (direction, every leverage link, target
    units, whether the rebalance rule let it through) for reports; ``record_trace=False`` skips it
    (intraday runs decide at every bar). A direction's protective orders go into
    ``Decision.protective``.
    """

    def __init__(
        self,
        sleeves: tuple[SleevePlan, ...],
        allocation: Bound,
        conventions: Conventions,
        *,
        record_trace: bool = True,
    ) -> None:
        self.sleeves = sleeves
        self.allocation = allocation
        self.conventions = conventions
        self.record_trace = record_trace
        self.states: dict[tuple[str, str], Any] = {}
        for sleeve in sleeves:
            for slot, bound in self._bound(sleeve):
                self.states[(sleeve.id, slot)] = bound.component.init(bound.params)
        self.states[("", "allocation")] = allocation.component.init(allocation.params)
        self.trace: list[dict[str, Any]] = []

    @staticmethod
    def _bound(sleeve: SleevePlan) -> list[tuple[str, Bound]]:
        links = [(f"leverage.{index}", link) for index, link in enumerate(sleeve.leverage)]
        return [("direction", sleeve.direction), *links, ("sizing", sleeve.sizing), ("rebalance", sleeve.rebalance)]

    def state(self, sleeve_id: str, slot: str) -> Any:
        return self.states[(sleeve_id, slot)]

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        signals, protective = {}, {}
        for sleeve in self.sleeves:
            sleeve_view = SleeveView(view, sleeve.alias, self.conventions, book)
            d = sleeve.direction
            direction = d.component.step(d.params, self.states[(sleeve.id, "direction")], sleeve_view)
            protect = d.component.protect(d.params, self.states[(sleeve.id, "direction")])
            if protect is not None:
                protective[sleeve.alias] = protect
            leverage, links = 0.0, []
            for index, link in enumerate(sleeve.leverage):
                state = self.states[(sleeve.id, f"leverage.{index}")]
                leverage = link.component.step(link.params, state, sleeve_view, leverage)
                links.append(leverage)
            signals[sleeve.id] = (direction, leverage, links)
        a = self.allocation
        ids = tuple(sleeve.id for sleeve in self.sleeves)
        weights = a.component.weights(a.params, self.states[("", "allocation")], view, ids)

        targets, target_leverage = {}, {}
        for sleeve in self.sleeves:
            direction, leverage, links = signals[sleeve.id]
            weight = weights[sleeve.id]
            series = view[sleeve.alias]
            price = float(series.close[-1])
            s = sleeve.sizing
            units = s.component.units(
                s.params,
                equity=book.equity,
                direction=direction,
                leverage=leverage,
                weight=weight,
                price=price,
                instrument=sleeve.instrument,
            )
            current = book.positions[sleeve.alias]
            notional = sleeve.instrument.notional_twd(price)
            ctx = RebalanceContext(
                date=view.date,
                rolled_today=len(series) > 1 and series.contract[-1] != series.contract[-2],
                target_leverage=direction * leverage * weight,
                actual_leverage=current * notional / book.equity if book.equity > 0 else math.inf,
            )
            r = sleeve.rebalance
            trade = r.component.should_trade(
                r.params,
                self.states[(sleeve.id, "rebalance")],
                current_units=current,
                target_units=units,
                ctx=ctx,
            )
            if trade:
                targets[sleeve.alias] = units
            target_leverage[sleeve.alias] = ctx.target_leverage
            if not self.record_trace:
                continue
            self.trace.append(
                {
                    "date": view.date,
                    "sleeve": sleeve.id,
                    "direction": direction,
                    "links": tuple(links),
                    "target_leverage": ctx.target_leverage,
                    "target_units": units,
                    "current_units": current,
                    "traded": trade,
                }
            )
        return Decision(targets, target_leverage, protective)


StrategyFactory = Callable[[], SpecStrategy]


def parse_bound(component: Component, params: Mapping[str, Any] | None) -> Bound:
    return Bound(component, component.parse(params or {}))
