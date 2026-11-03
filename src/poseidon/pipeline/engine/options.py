"""Bought options in the daily engine (design 2026-10-24 §5.7): chain, valuation, fills, expiry, hedges.

Buyer only. A hedge is an engine overlay on one futures instrument: at each decision the engine
sizes it from that instrument's final target units and picks the series itself, so the hedge
always covers the position the engine lets through (after margin and leverage caps).

Prices
    Every option price is a TAIFEX daily settlement price (deep out-of-the-money series often
    do not trade, so the last trade is unreliable). Valuation: each held series is marked at its
    settlement every day it has one, and keeps its last mark on a day without one. Fills: the
    settlement of the fill day plus slippage against the order, ``max(slippage_ticks x tick,
    slippage_pct x premium)``, a sale never below 0. The fill day follows the futures:
    ``next_open`` -> the next trading day's settlement, ``signal_close`` -> the decision day's.

Selection (protective put)
    Expiry: the eligible monthly expiries, those with more than ``roll_days`` trading days left
    after the decision day, sorted; ``tenor`` 0 takes the first, 1 the second. Strike: target =
    underlying close x (1 - moneyness) (calls: x (1 + moneyness)); the listed strike of that
    expiry and side with a settlement that day nearest to it, the lower one on a tie. A strike
    more than ``max_strike_gap`` (relative) from the target, or no series at all, is
    ``hedge_unavailable``: no new series that day, a held one is kept.

Holding
    One series per hedge. It is kept while it has more than ``roll_days`` trading days left and
    re-sized on it when the target units change; at ``roll_days`` or fewer it is sold and the
    newly selected series bought (a roll). With ``roll_with_tenor`` it is kept only while it is
    still the ``tenor``-th eligible expiry instead: a ``next`` put rolls once the front month has
    ``roll_days`` or fewer days left (0: on the front's expiry day), so it is always the second
    expiry. Units = floor(cover_ratio x long underlying units x
    underlying point value / option point value): one TXO put (50 TWD/pt) covers five TMF
    (10 TWD/pt). A flat or short underlying holds no hedge.

Take-profit
    With ``take_profit`` set, a held series whose settlement on the decision day is at least
    ``take_profit`` x its cost (the average fill price of the units held, slippage included) is
    sold and the newly selected series bought, on the same fill as a roll (fill reason
    ``take_profit``). It is checked on the daily settlement, the only price the chain has. It
    takes precedence over a roll due the same day (same replacement, reason ``take_profit``); a
    series cash-settled at today's close is gone before the decision, so expiry wins over a
    trigger on the same day. If the selection returns the held series, or none, the held series
    is kept.

Expiry
    A series still held on its final settlement day (the expiry day from 2008-12, the next
    trading day before) is cash-settled at the final settlement price: intrinsic value x point
    value x units into cash, fill reason ``expiry``. A settlement in the money pays tax on that
    value and the commission; one that expires worthless costs nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from poseidon.pipeline.v3.instruments import Instrument

OPTION_QUOTE_COLUMNS = ("date", "contract_month", "strike", "cp", "settlement")
HEDGE_UNAVAILABLE = "hedge_unavailable"
MAX_STRIKE_GAP = 0.05  # §5.7: a strike further than this from the target means no hedge that day
MAX_UNAVAILABLE_SHARE = 0.05  # §5.7: more hedge-unavailable days than this share of hedged days is an error

CallPut = Literal["C", "P"]


@dataclass(frozen=True)
class OptionSpec:
    """One option contract family. ``tick_ladder``: ``((premium below, tick), ...)`` ascending; the
    last tick applies above the last bound."""

    symbol: str
    point_value: float  # TWD per premium point per contract
    commission_per_unit: float  # TWD per contract per side
    tax_rate: float  # fraction of premium value per side (and of the cash-settlement value)
    tick_ladder: tuple[tuple[float, float], ...]
    slippage_ticks: float = 1.0
    slippage_pct: float = 0.05

    def __post_init__(self) -> None:
        if self.point_value <= 0 or not self.tick_ladder:
            raise ValueError(f"{self.symbol}: point_value must be positive and the tick ladder non-empty")
        if min(self.commission_per_unit, self.tax_rate, self.slippage_ticks, self.slippage_pct) < 0:
            raise ValueError(f"{self.symbol}: costs must be non-negative")

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> OptionSpec:
        cost = instrument.cost_model
        if instrument.kind != "option":
            raise ValueError(f"{instrument.symbol} is a {instrument.kind}, not an option")
        return cls(
            symbol=instrument.symbol,
            point_value=instrument.point_value_twd * instrument.lot,
            commission_per_unit=cost.commission_per_contract,
            tax_rate=cost.tax_rate,
            tick_ladder=instrument.tick_ladder,
            slippage_ticks=cost.slippage_ticks,
            slippage_pct=cost.slippage_pct,
        )

    def tick(self, premium: float) -> float:
        for below, tick in self.tick_ladder:
            if premium < below:
                return tick
        return self.tick_ladder[-1][1]

    def fill_price(self, units: int, settlement: float) -> float:
        slip = max(self.slippage_ticks * self.tick(settlement), self.slippage_pct * settlement)
        return settlement + slip if units > 0 else max(settlement - slip, 0.0)

    def leg_fees(self, units: int, price: float) -> tuple[float, float]:
        """(tax, commission) in TWD for one leg of ``units`` contracts at premium ``price``."""

        return self.tax_rate * abs(units) * price * self.point_value, abs(units) * self.commission_per_unit


@dataclass(frozen=True)
class OptionSeries:
    contract_month: str
    strike: float
    cp: CallPut
    expiry: pd.Timestamp  # last trading day

    @property
    def key(self) -> str:
        return f"{self.contract_month}{self.cp}{self.strike:g}"

    def intrinsic(self, underlying: float) -> float:
        return max(self.strike - underlying, 0.0) if self.cp == "P" else max(underlying - self.strike, 0.0)


@dataclass(frozen=True)
class OptionInput:
    """One option chain: daily settlement quotes per series plus the ex-ante expiry calendar.

    ``quotes`` columns ``OPTION_QUOTE_COLUMNS`` (day session; a row without a positive settlement
    is no quote). ``expiries``: contract month -> last trading day, ex-ante reference data that
    truncation never cuts. ``final_settlement``: contract month -> (final settlement day, price),
    market data known from its day on, so truncation drops later ones.
    """

    spec: OptionSpec
    quotes: pd.DataFrame
    expiries: Mapping[str, Any]
    final_settlement: Mapping[str, tuple[Any, float]]

    def truncated(self, cut: Any) -> OptionInput:
        cut = pd.Timestamp(cut)
        return replace(
            self,
            quotes=self.quotes[pd.to_datetime(self.quotes["date"]).dt.normalize() <= cut],
            final_settlement={k: v for k, v in self.final_settlement.items() if pd.Timestamp(v[0]) <= cut},
        )


@dataclass(frozen=True)
class HedgePolicy:
    """A bought-option overlay on futures instrument ``underlying`` (see the module docstring)."""

    name: str  # instrument name of the hedge in fills / positions / events
    underlying: str
    chain: str
    moneyness: float  # fraction
    roll_days: int
    cover_ratio: float
    tenor: int = 0  # 0 = front eligible expiry, 1 = the one after
    cp: CallPut = "P"
    max_strike_gap: float = MAX_STRIKE_GAP
    take_profit: float | None = None  # sell at this multiple of the cost (see the module docstring)
    roll_with_tenor: bool = False  # roll once the held series is no longer the tenor-th eligible expiry

    def __post_init__(self) -> None:
        if not 0 <= self.moneyness < 1 or self.roll_days < 0 or self.cover_ratio <= 0 or self.tenor < 0:
            raise ValueError(f"hedge {self.name}: invalid policy {self}")
        if self.take_profit is not None and not self.take_profit > 1:
            raise ValueError(f"hedge {self.name}: take_profit must be above 1, got {self.take_profit}")

    def units(self, underlying_units: int, underlying_point_value: float, option_point_value: float) -> int:
        if underlying_units <= 0:
            return 0
        raw = self.cover_ratio * underlying_units * underlying_point_value / option_point_value
        return math.floor(raw + 1e-9 * max(1.0, raw))

    def target_strike(self, underlying: float) -> float:
        return underlying * (1 - self.moneyness) if self.cp == "P" else underlying * (1 + self.moneyness)


class OptionChain:
    """Per-day quote lookups of one ``OptionInput`` over the engine's dates."""

    def __init__(self, name: str, option: OptionInput, calendar: pd.DatetimeIndex) -> None:
        missing = [c for c in OPTION_QUOTE_COLUMNS if c not in option.quotes.columns]
        if missing:
            raise ValueError(f"{name}: quotes missing columns {missing}")
        self.spec = option.spec
        self.calendar = calendar
        self.expiries = {str(k): pd.Timestamp(v).normalize() for k, v in option.expiries.items()}
        self.final = {str(k): (pd.Timestamp(d).normalize(), float(p)) for k, (d, p) in option.final_settlement.items()}
        df = option.quotes.loc[:, list(OPTION_QUOTE_COLUMNS)].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["contract_month"] = df["contract_month"].astype(str)
        df["settlement"] = pd.to_numeric(df["settlement"], errors="coerce")
        df = df[np.isfinite(df["settlement"]) & (df["settlement"] > 0)]
        unknown = sorted(set(df["contract_month"]) - set(self.expiries))
        if unknown:
            raise ValueError(f"{name}: no expiry for contract months {unknown[:5]}")
        if df.duplicated(["date", "contract_month", "strike", "cp"]).any():
            raise ValueError(f"{name}: duplicate quotes")
        df = df.sort_values(["date", "contract_month", "cp", "strike"])
        # date -> (contract_month, cp) -> (strikes ascending, settlements)
        self.days: dict[pd.Timestamp, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]] = {}
        for (day, month, cp), rows in df.groupby(["date", "contract_month", "cp"], sort=False):
            self.days.setdefault(day, {})[(month, cp)] = (rows["strike"].to_numpy(float), rows["settlement"].to_numpy())
        self.months = sorted(self.expiries, key=lambda m: (self.expiries[m], m))

    def settlement(self, day: pd.Timestamp, series: OptionSeries) -> float:
        strikes, settles = self.days.get(day, {}).get((series.contract_month, series.cp), (None, None))
        if strikes is None:
            return math.nan
        j = int(np.searchsorted(strikes, series.strike))
        return float(settles[j]) if j < len(strikes) and strikes[j] == series.strike else math.nan

    def days_left(self, day: pd.Timestamp, expiry: pd.Timestamp) -> int:
        """Trading days from ``day`` to ``expiry`` on the engine calendar (0 on the expiry day)."""

        return int(self.calendar.searchsorted(expiry) - self.calendar.searchsorted(day))

    def eligible(self, day: pd.Timestamp, policy: HedgePolicy) -> list[str]:
        """Contract months with more than ``policy.roll_days`` trading days left after ``day``, by expiry."""

        return [m for m in self.months if self.days_left(day, self.expiries[m]) > policy.roll_days]

    def roll_due(self, day: pd.Timestamp, policy: HedgePolicy, held: OptionSeries) -> bool:
        if policy.roll_with_tenor:
            eligible = self.eligible(day, policy)
            return len(eligible) <= policy.tenor or eligible[policy.tenor] != held.contract_month
        return self.days_left(day, held.expiry) <= policy.roll_days

    def select(self, day: pd.Timestamp, policy: HedgePolicy, underlying: float) -> tuple[OptionSeries | None, str]:
        """The series ``policy`` buys on ``day``, or (None, why)."""

        eligible = self.eligible(day, policy)
        if len(eligible) <= policy.tenor:
            return None, f"no expiry with more than {policy.roll_days} trading days left"
        month = eligible[policy.tenor]
        target = policy.target_strike(underlying)
        strikes, _ = self.days.get(day, {}).get((month, policy.cp), (np.empty(0), None))
        if not len(strikes):
            return None, f"{month}{policy.cp}: no series with a settlement price"
        strike = float(min(strikes, key=lambda k: (abs(k - target), k)))
        gap = abs(strike - target) / target
        if gap > policy.max_strike_gap:
            return None, (
                f"{month}{policy.cp}: nearest strike {strike:g} is {gap:.1%} from the target {target:,.0f} "
                f"(> {policy.max_strike_gap:.0%})"
            )
        return OptionSeries(month, strike, policy.cp, self.expiries[month]), ""


class HedgeOrder(NamedTuple):
    series: OptionSeries | None  # None: sell what is held
    units: int
    take_profit: bool = False


@dataclass
class HedgeBook:
    """What one hedge holds, and the order waiting for its fill day."""

    policy: HedgePolicy
    series: OptionSeries | None = None
    units: int = 0
    mark: float = math.nan
    cost: float = math.nan  # average fill price of the units held
    target_units: int | None = None  # today's decision, for positions_daily
    pending: HedgeOrder | None = None

    @property
    def take_profit_hit(self) -> bool:
        multiple = self.policy.take_profit
        return multiple is not None and self.units > 0 and self.mark >= multiple * self.cost


def hedge_coverage(positions_daily: pd.DataFrame, events: pd.DataFrame, names: list[str]) -> dict[str, dict]:
    """Per hedge: days it wanted a position, days with ``hedge_unavailable``, the share (§5.7)."""

    out = {}
    for name in names:
        rows = positions_daily[positions_daily["instrument"] == name]
        wanted = int((rows["target_units"].fillna(0) > 0).sum())
        unavailable = events[(events["kind"] == HEDGE_UNAVAILABLE) & (events["instrument"] == name)]
        days = int(unavailable["date"].nunique())
        out[name] = {
            "hedged_days": wanted,
            "unavailable_days": days,
            "unavailable_share": days / wanted if wanted else 0.0,
            "unavailable_dates": [pd.Timestamp(d).date().isoformat() for d in sorted(unavailable["date"].unique())],
        }
    return out
