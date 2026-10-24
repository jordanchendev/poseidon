"""Contract specs the engine needs, and the per-leg cost and sizing arithmetic built on them.

The engine does not own an instrument table: ``ContractSpec.from_instrument`` builds one from a
row of ``pipeline.v3.instruments.INSTRUMENTS`` (its ``CostModel``, margin ratios, listing and
proxy). Sizing is not here either: ``sizing.target_leverage`` in the v3 registry is the one
formula, the engine only clips what a strategy asks for.

Per-leg fees follow ``BacktestPortfolio._entry_fees`` with ``tax_on_entry``: futures tax on the
notional of every leg (entry and exit alike) plus a fixed commission per contract per side.
Slippage is not a fee: it is inside the fill price (``slippage_ticks`` x ``tick_size`` against the
order), so it reaches equity through mark-to-market, and ``Fill.slippage_twd`` reports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poseidon.pipeline.v3.instruments import Instrument

ROUNDING_EPS = 1e-9  # keeps exact ratios such as 3e6 / 2e5 from flooring to 14 on float noise


@dataclass(frozen=True)
class ContractSpec:
    """One tradable instrument. Margin ratios are fractions of notional (price x point value)."""

    symbol: str
    point_value: float  # TWD per price point per contract
    tick_size: float  # price points
    commission_per_unit: float  # TWD per contract per side
    tax_rate: float  # fraction of notional per side
    slippage_ticks: float = 1.0  # per side, against the order
    initial_margin_ratio: float = 0.07
    maintenance_margin_ratio: float = 0.054
    listed_since: date | None = None  # earlier rows are simulated and flagged ``proxy``
    price_proxy: str | None = None  # symbol whose prices stand in for this one (TMF priced from TX)

    def __post_init__(self) -> None:
        if self.point_value <= 0 or self.tick_size <= 0:
            raise ValueError(f"{self.symbol}: point_value and tick_size must be positive")
        if self.commission_per_unit < 0 or self.tax_rate < 0 or self.slippage_ticks < 0:
            raise ValueError(f"{self.symbol}: costs must be non-negative")
        if not 0 < self.maintenance_margin_ratio <= self.initial_margin_ratio:
            raise ValueError(f"{self.symbol}: need 0 < maintenance_margin_ratio <= initial_margin_ratio")

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> ContractSpec:
        """The engine's view of an instrument-table row (futures only in Phase 1)."""

        cost = instrument.cost_model
        if instrument.kind != "future":
            raise ValueError(f"{instrument.symbol}: the daily engine trades futures only, not {instrument.kind}")
        if not cost.tax_on_entry:
            raise ValueError(f"{cost.market}: the engine charges futures tax on every leg; use a tax_on_entry model")
        return cls(
            symbol=instrument.symbol,
            point_value=instrument.point_value_twd * instrument.lot,
            tick_size=instrument.tick_size,
            commission_per_unit=cost.commission_per_contract,
            tax_rate=cost.tax_rate,
            slippage_ticks=cost.slippage_ticks,
            initial_margin_ratio=instrument.initial_margin_ratio,
            maintenance_margin_ratio=instrument.maintenance_margin_ratio,
            listed_since=instrument.listed_since,
            price_proxy=instrument.price_proxy,
        )

    def notional(self, units: int, price: float) -> float:
        return abs(units) * price * self.point_value

    def fill_price(self, units: int, price: float) -> float:
        """Price after slippage for an order of signed ``units`` (buy pays up, sell gives up)."""

        return price + math.copysign(self.slippage_ticks * self.tick_size, units)

    def leg_fees(self, units: int, fill_price: float) -> tuple[float, float]:
        """(tax, commission) in TWD for one leg of ``units`` contracts at ``fill_price``."""

        return self.tax_rate * self.notional(units, fill_price), abs(units) * self.commission_per_unit
