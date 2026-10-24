"""Runtime of a compiled spec v3 (design 2026-10-24 §3.3, §5.1): what a component sees, and the
``TargetStrategy`` that runs the components of every sleeve at each decision close.

Per decision, per sleeve: direction ``d`` -> leverage chain ``L`` (each link gets the previous
link's output, the first gets 0.0) -> allocation weight ``w`` -> ``sizing.units(equity, d, L, w,
price = today's close of the held contract)`` -> ``rebalance.should_trade`` decides whether that
target becomes an order (left out of the ``Decision`` = keep the position). The engine then
clips by margin and leverage caps and fills per ``execution``.

Components only ever get a :class:`SleeveView` (arrays up to the decision day, read-only) and
their own state, so no registry component can read the future. :class:`Conventions` are the
knobs a spec cannot set: the levtx parity run and the fill-timing sensitivity (§12 #1) use them;
a pipeline run uses the defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import BookView, Decision, MarketView
from poseidon.pipeline.v3.instruments import Instrument
from poseidon.pipeline.v3.registry import Component, StrictParams

# spec series field -> engine SeriesView field
SERIES_FIELD = {"open": "open", "high": "high", "low": "low", "close": "close", "adjusted": "adjusted_close"}


@dataclass(frozen=True)
class Conventions:
    """Run conventions outside the spec. Defaults = the spec as written (§12 defaults).

    ``execution`` overrides the spec's execution component (``signal_close`` = fill at the
    decision close: the levtx convention and the §12 #1 sensitivity run). ``signal_aux`` maps a
    resolved series ref (``"core:adjusted"``; ``self`` is resolved to the sleeve's alias first)
    to an aux series whose ``close`` replaces it, e.g. ``{"core:adjusted": "taiex"}`` signals
    on the TAIEX price index as levtx does. ``signal_lag`` drops that many latest rows from
    every signal series (levtx decides at close t from index data up to t-1: lag 1).
    """

    execution: Literal["next_open", "signal_close"] | None = None
    signal_aux: Mapping[str, str] = field(default_factory=dict)
    signal_lag: int = 0

    def __post_init__(self) -> None:
        if self.signal_lag < 0:
            raise ValueError("signal_lag must be >= 0")


class Series(NamedTuple):
    dates: np.ndarray  # datetime64[ns], read-only
    values: np.ndarray  # float, read-only


class SleeveView:
    """One sleeve's window on the market at a decision close: series refs resolve to arrays
    ending at (or, with ``signal_lag``, before) the decision day."""

    def __init__(self, market: MarketView, alias: str, conventions: Conventions) -> None:
        self._market = market
        self.alias = alias
        self._conventions = conventions

    @property
    def date(self) -> pd.Timestamp:
        return self._market.date

    def series(self, ref: str) -> Series:
        alias, name = ref.split(":")
        key = f"{self.alias if alias == 'self' else alias}:{name}"
        aux = self._conventions.signal_aux.get(key)
        if aux is not None:
            view = self._market.aux(aux)
            dates, values = view.dates, view["close"]
        else:
            view = self._market[key.split(":")[0]]
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
    units, whether the rebalance rule let it through) for reports.
    """

    def __init__(self, sleeves: tuple[SleevePlan, ...], allocation: Bound, conventions: Conventions) -> None:
        self.sleeves = sleeves
        self.allocation = allocation
        self.conventions = conventions
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
        signals = {}
        for sleeve in self.sleeves:
            sleeve_view = SleeveView(view, sleeve.alias, self.conventions)
            d = sleeve.direction
            direction = d.component.step(d.params, self.states[(sleeve.id, "direction")], sleeve_view)
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
        return Decision(targets, target_leverage)


StrategyFactory = Callable[[], SpecStrategy]


def parse_bound(component: Component, params: Mapping[str, Any] | None) -> Bound:
    return Bound(component, component.parse(params or {}))
