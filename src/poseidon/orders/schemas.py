"""Order and fill dataclasses for the broker/order subsystem."""

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from poseidon.orders.state_machine import OrderStatus


@dataclass
class Order:
    """A trading order to be submitted to a broker adapter."""

    symbol: str
    market: str  # "tw_stock"
    action: str  # "buy" | "sell"
    order_type: str  # "market" | "limit"
    target_weight: float
    quantity: float  # shares (rounded to lot_size, or fractional for perps)
    strategy_name: str
    broker_mode: str  # "paper" | "live"
    side: str = "long"  # "long" | "short"
    price: float | None = None  # limit price, None for market
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str | None = None
    # TRUTH-03 (D-13/D-17): structured 4-key dict, not free text.
    reject_reason: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: _uuid.uuid4().hex)


@dataclass
class Fill:
    """A single fill event for an order."""

    order_id: str
    fill_price: float
    fill_quantity: float
    fill_time: datetime
    broker_fill_id: str | None = None


@dataclass
class OrderResult:
    """Aggregated result of placing an order."""

    order: Order
    fills: list[Fill] = field(default_factory=list)
    success: bool = False


@dataclass
class RiskCheckResult:
    """Result of a pre-order risk check.

    TRUTH-03 (D-13/D-14, RES-Q2): extended with check_name and shortfall so
    the OrderManager can build a structured reject_reason without inspecting
    the human-readable `reason` string.
    """

    passed: bool
    reason: str = ""
    check_name: str | None = None
    shortfall: dict[str, Any] | None = None
