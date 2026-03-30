"""Order and fill dataclasses for the broker/order subsystem."""

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime

from poseidon.orders.state_machine import OrderStatus


@dataclass
class Order:
    """A trading order to be submitted to a broker adapter."""

    symbol: str
    market: str  # "tw_stock"
    action: str  # "buy" | "sell"
    order_type: str  # "market" | "limit"
    target_weight: float
    quantity: int  # shares (rounded to lot_size)
    strategy_name: str
    broker_mode: str  # "paper" | "live"
    price: float | None = None  # limit price, None for market
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str | None = None
    reject_reason: str | None = None
    id: str = field(default_factory=lambda: _uuid.uuid4().hex)


@dataclass
class Fill:
    """A single fill event for an order."""

    order_id: str
    fill_price: float
    fill_quantity: int
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
    """Result of a pre-order risk check."""

    passed: bool
    reason: str = ""
