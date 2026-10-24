"""Contract specs the engine needs, and the per-leg cost and sizing arithmetic built on them.

The engine does not own an instrument table (that lives with the v3 registry); callers pass one
``ContractSpec`` per instrument. ``ContractSpec.from_cost_model`` adapts an existing
``backtest.cost_model.CostModel`` (for example ``tw_futures_pipeline`` for TX).

Per-leg fees follow ``BacktestPortfolio._entry_fees`` with ``tax_on_entry``: futures tax on the
notional of every leg (entry and exit alike) plus a fixed commission per contract per side.
Slippage is not a fee: it is inside the fill price (``slippage_ticks`` x ``tick_size`` against the
order), so it reaches equity through mark-to-market, and ``Fill.slippage_twd`` reports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

from poseidon.backtest.cost_model import CostModel

Rounding = Literal["floor", "nearest"]
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
    def from_cost_model(cls, symbol: str, cost: CostModel, **margin_and_listing) -> ContractSpec:
        """Reuse a futures ``CostModel`` (tax on both legs, per-contract commission, tick slippage)."""

        if cost.tick_size <= 0:
            raise ValueError(f"{cost.market}: a futures cost model needs tick_size > 0")
        if not cost.tax_on_entry:
            raise ValueError(f"{cost.market}: the engine charges futures tax on every leg; use a tax_on_entry model")
        return cls(
            symbol=symbol,
            point_value=cost.point_value,
            tick_size=cost.tick_size,
            commission_per_unit=cost.commission_per_contract,
            tax_rate=cost.tax_rate,
            slippage_ticks=cost.slippage_ticks,
            **margin_and_listing,
        )

    def notional(self, units: int, price: float) -> float:
        return abs(units) * price * self.point_value

    def fill_price(self, units: int, price: float) -> float:
        """Price after slippage for an order of signed ``units`` (buy pays up, sell gives up)."""

        return price + math.copysign(self.slippage_ticks * self.tick_size, units)

    def leg_fees(self, units: int, fill_price: float) -> tuple[float, float]:
        """(tax, commission) in TWD for one leg of ``units`` contracts at ``fill_price``."""

        return self.tax_rate * self.notional(units, fill_price), abs(units) * self.commission_per_unit


def target_leverage_units(
    equity: float,
    leverage: float,
    price: float,
    point_value: float,
    *,
    direction: int = 1,
    weight: float = 1.0,
    lot: int = 1,
    rounding: Rounding = "floor",
) -> int:
    """§5.3 ``sizing.target_leverage``: units = round(equity x d x L x w / (price x point_value x lot)) x lot.

    ``floor`` rounds the magnitude down (toward zero) so the position never exceeds the target
    leverage, long or short. ``nearest`` uses Python's round-half-to-even.
    """

    if price <= 0 or point_value <= 0 or lot <= 0:
        raise ValueError("price, point_value and lot must be positive")
    if equity <= 0:
        return 0
    raw = equity * direction * leverage * weight / (price * point_value * lot)
    if rounding == "floor":
        lots = math.trunc(raw + math.copysign(ROUNDING_EPS * max(1.0, abs(raw)), raw))
    elif rounding == "nearest":
        lots = round(raw)
    else:
        raise ValueError(f"unknown rounding {rounding!r}")
    return int(lots) * lot
