"""What a strategy sees: append-only market buffers and a read-only book snapshot.

The engine writes bar t into the buffers only when the clock reaches t, so the memory behind a
view (``arr.base`` included) holds nothing past the decision day: no look-ahead is reachable,
not merely discouraged. Arrays handed out are read-only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

SERIES_FIELDS = ("open", "high", "low", "close", "adjusted_close")


def _frozen(arr: np.ndarray) -> np.ndarray:
    view = arr.view()
    view.flags.writeable = False
    return view


class _Buffer:
    """Preallocated columns filled one row at a time; rows past ``n`` are never written yet."""

    def __init__(self, capacity: int, fields: tuple[str, ...]) -> None:
        self.n = 0
        self.dates = np.full(capacity, np.datetime64("NaT"), dtype="datetime64[ns]")
        self.columns = {f: np.full(capacity, np.nan) for f in fields}

    def append(self, date: pd.Timestamp, values: Mapping[str, float]) -> None:
        self.dates[self.n] = date.to_datetime64()
        for field, value in values.items():
            self.columns[field][self.n] = value
        self.n += 1


class SeriesBuffer(_Buffer):
    """Continuous series of one futures instrument: the contract held after each close."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, SERIES_FIELDS)
        self.contract = np.empty(capacity, dtype=object)

    def append_bar(self, date: pd.Timestamp, contract: str, values: Mapping[str, float]) -> None:
        self.contract[self.n] = contract
        self.append(date, values)


@dataclass(frozen=True)
class SeriesView:
    """One instrument up to and including the decision day.

    ``open/high/low/close`` are raw prices of the contract held after that day's close (on a roll
    day: the new contract). ``adjusted_close`` is a causal, ratio-adjusted continuous price: it
    compounds each day's close-to-close return on the contract held over that day, starting at
    the first close. Its level drifts from the raw price by the cumulative roll ratios, but every
    scale-free indicator (MA bands, returns, volatility) equals the ratio back-adjusted series,
    and appending a bar never rewrites history.
    """

    _buffer: SeriesBuffer
    _n: int

    def __len__(self) -> int:
        return self._n

    @property
    def dates(self) -> np.ndarray:
        return _frozen(self._buffer.dates[: self._n])

    @property
    def contract(self) -> np.ndarray:
        return _frozen(self._buffer.contract[: self._n])

    def __getattr__(self, field: str) -> np.ndarray:
        if field not in SERIES_FIELDS:
            raise AttributeError(field)
        return _frozen(self._buffer.columns[field][: self._n])


@dataclass(frozen=True)
class AuxView:
    """An auxiliary series: rows dated on or before the decision day."""

    _buffer: _Buffer
    _n: int

    def __len__(self) -> int:
        return self._n

    @property
    def dates(self) -> np.ndarray:
        return _frozen(self._buffer.dates[: self._n])

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._buffer.columns)

    def __getitem__(self, column: str) -> np.ndarray:
        return _frozen(self._buffer.columns[column][: self._n])


@dataclass(frozen=True)
class MarketView:
    """The market at a decision close. ``calendar`` is the run's whole trading calendar: reference data
    known ex ante (sessions that traded, then the official schedule), which the truncation audit keeps
    whole too; it holds no prices. Empty on the intraday engine."""

    date: pd.Timestamp
    _series: Mapping[str, SeriesView]
    _aux: Mapping[str, AuxView]
    calendar: pd.DatetimeIndex = dataclasses.field(default_factory=lambda: pd.DatetimeIndex([]))

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(self._series)

    def __getitem__(self, instrument: str) -> SeriesView:
        return self._series[instrument]

    def aux(self, name: str) -> AuxView:
        return self._aux[name]


@dataclass(frozen=True)
class Fill:
    date: pd.Timestamp
    session: str  # "open" | "intraday" | "close"
    instrument: str
    contract: str
    units: int  # signed
    price: float  # after slippage
    ref_price: float  # before slippage
    tax: float
    commission: float
    slippage_twd: float
    reason: str  # rebalance | roll | margin_call | forced_liquidation


@dataclass(frozen=True)
class Protect:
    """Protective orders on one position, checked intrabar by the intraday engine (design §3.4).

    ``stop`` fills at the stop (gap-aware: at the open when the bar opens through it), ``take_profit``
    at its price; the stop wins when one bar touches both. Neither slips unless the run sets
    ``IntradayConfig.stop_slippage_ticks`` (stops only).
    """

    stop: float | None = None
    take_profit: float | None = None


@dataclass(frozen=True)
class BookView:
    """Portfolio state at the decision point (after settlement, any roll, any forced action).

    Intraday engine: ``fills_today`` holds every fill since the previous decision (that decision's
    own fills at the bar close included) and ``protective`` the orders resting on each position.
    """

    date: pd.Timestamp
    initial_capital: float
    cash: float
    equity: float
    positions: Mapping[str, int]
    contracts: Mapping[str, str | None]
    margin_required: float  # initial margin on current positions at today's close
    maintenance_required: float
    fills_today: tuple[Fill, ...]
    protective: Mapping[str, Protect] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))
        object.__setattr__(self, "protective", MappingProxyType(dict(self.protective)))
