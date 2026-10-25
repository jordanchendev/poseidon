"""Contract specs the engine needs, the instrument-kind table, and the per-leg cost arithmetic.

The engine does not own an instrument table: ``ContractSpec.from_instrument`` builds one from a
row of ``pipeline.v3.instruments.INSTRUMENTS`` (its ``CostModel``, margin ratios, listing and
proxy). Sizing is not here either: the v3 sizing components are the one formula, the engine only
clips what a strategy asks for.

How the engine books an instrument is a row of :data:`KIND_POLICIES`, keyed by ``kind``, never a
branch on the symbol: futures settle daily (variation margin into cash, mark = close) and carry a
notional-ratio margin; stocks are paid in full from cash, never settle (mark stays 0, so the
position's value is units x price) and cannot go short. A new kind (options) is a new row plus the
hooks it needs.

Per-leg fees follow ``BacktestPortfolio._entry_fees``: futures pay the tax on the notional of
every leg plus a fixed commission per contract per side; stocks pay a commission rate on the
notional of both legs and the transaction tax on sells only. Slippage is not a fee: it is inside
the fill price (``slippage_ticks`` x the tick at that price, against the order), so it reaches
equity through mark-to-market, and ``Fill.slippage_twd`` reports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poseidon.pipeline.v3.instruments import Instrument

ROUNDING_EPS = 1e-9  # keeps exact ratios such as 3e6 / 2e5 from flooring to 14 on float noise

# (price upper bound, exclusive; tick) steps of a price-tiered tick, last bound inf.
TickLadder = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class KindPolicy:
    """How the ledger treats one instrument kind."""

    daily_settlement: bool  # mark to each close, variation margin into cash (mark = close)
    margined: bool  # notional-ratio initial / maintenance margin; else paid in full from cash
    short: bool  # negative units allowed
    rolls: bool  # contract-month table with a roll schedule (else one column, held forever)


KIND_POLICIES: dict[str, KindPolicy] = {
    "future": KindPolicy(daily_settlement=True, margined=True, short=True, rolls=True),
    "stock": KindPolicy(daily_settlement=False, margined=False, short=False, rolls=False),
}


@dataclass(frozen=True)
class ContractSpec:
    """One tradable instrument. Margin ratios are fractions of notional (price x point value)."""

    symbol: str
    point_value: float  # TWD per price point per unit (contract, or board lot / share for stocks)
    tick_size: float  # price points (the smallest tick when ``tick_ladder`` is set)
    commission_per_unit: float  # TWD per unit per side
    tax_rate: float  # fraction of notional per taxed leg
    slippage_ticks: float = 1.0  # per side, against the order
    initial_margin_ratio: float = 0.07
    maintenance_margin_ratio: float = 0.054
    listed_since: date | None = None  # earlier rows are simulated and flagged ``proxy``
    price_proxy: str | None = None  # symbol whose prices stand in for this one (TMF priced from TX)
    kind: str = "future"  # row of KIND_POLICIES
    commission_rate: float = 0.0  # fraction of notional per side (stocks: 0.1425 %)
    tax_on_buy: bool = True  # futures tax both legs; the stock transaction tax is on sells only
    tick_ladder: TickLadder | None = None  # price-tiered tick (TW stocks / ETFs); None = ``tick_size``

    def __post_init__(self) -> None:
        if self.kind not in KIND_POLICIES:
            raise ValueError(f"{self.symbol}: unknown instrument kind {self.kind!r}")
        if self.point_value <= 0 or self.tick_size <= 0:
            raise ValueError(f"{self.symbol}: point_value and tick_size must be positive")
        if min(self.commission_per_unit, self.commission_rate, self.tax_rate, self.slippage_ticks) < 0:
            raise ValueError(f"{self.symbol}: costs must be non-negative")
        if self.policy.margined:
            if not 0 < self.maintenance_margin_ratio <= self.initial_margin_ratio:
                raise ValueError(f"{self.symbol}: need 0 < maintenance_margin_ratio <= initial_margin_ratio")
        elif self.initial_margin_ratio or self.maintenance_margin_ratio:
            raise ValueError(f"{self.symbol}: a {self.kind} is paid in full; margin ratios must be 0")
        if self.tick_ladder is not None and (not self.tick_ladder or self.tick_ladder[-1][0] != math.inf):
            raise ValueError(f"{self.symbol}: tick_ladder must end with an inf bound")

    @property
    def policy(self) -> KindPolicy:
        return KIND_POLICIES[self.kind]

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> ContractSpec:
        """The engine's view of an instrument-table row. One engine unit is ``instrument.lot`` shares or
        one contract, so an odd-lot sleeve passes a row with ``lot = 1``."""

        cost = instrument.cost_model
        if instrument.kind not in KIND_POLICIES:
            raise ValueError(f"{instrument.symbol}: the daily engine has no ledger for {instrument.kind}")
        if instrument.kind == "future" and not cost.tax_on_entry:
            raise ValueError(f"{cost.market}: the engine charges futures tax on every leg; use a tax_on_entry model")
        if cost.sell_commission_rate != cost.buy_commission_rate:
            raise ValueError(f"{cost.market}: the engine has one commission rate for both sides")
        return cls(
            symbol=instrument.symbol,
            point_value=instrument.point_value_twd * instrument.lot,
            tick_size=instrument.tick_size,
            commission_per_unit=cost.commission_per_contract * instrument.lot,
            tax_rate=cost.tax_rate,
            slippage_ticks=cost.slippage_ticks,
            initial_margin_ratio=instrument.initial_margin_ratio,
            maintenance_margin_ratio=instrument.maintenance_margin_ratio,
            listed_since=instrument.listed_since,
            price_proxy=instrument.price_proxy,
            kind=instrument.kind,
            commission_rate=cost.buy_commission_rate,
            tax_on_buy=cost.tax_on_entry,
            tick_ladder=instrument.tick_ladder,
        )

    def tick(self, price: float) -> float:
        if self.tick_ladder is None:
            return self.tick_size
        return next(tick for bound, tick in self.tick_ladder if price < bound)

    def notional(self, units: int, price: float) -> float:
        return abs(units) * price * self.point_value

    def fill_price(self, units: int, price: float) -> float:
        """Price after slippage for an order of signed ``units`` (buy pays up, sell gives up)."""

        return price + math.copysign(self.slippage_ticks * self.tick(price), units)

    def leg_fees(self, units: int, fill_price: float) -> tuple[float, float]:
        """(tax, commission) in TWD for one leg of ``units`` at ``fill_price``."""

        notional = self.notional(units, fill_price)
        taxed = units < 0 or self.tax_on_buy
        tax = self.tax_rate * notional if taxed else 0.0
        return tax, abs(units) * self.commission_per_unit + self.commission_rate * notional
