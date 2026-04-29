"""Order and fill dataclasses for the broker/order subsystem."""

import uuid
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

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
    # Phase 89-01 (D-04, F8 wiring fix): FK to signals.id when this Order
    # was triggered by an upstream PASSED signal. NULL for portfolio-level
    # rebalances, protective close-outs, and other non-signal-driven flows
    # (those carry order_origin tags instead — see Plan 89-02).
    signal_id: uuid.UUID | None = None
    # Phase 89-02 (W4 audit whitelist): tags how this order was triggered so
    # the mini-audit can distinguish "missing wiring" (origin=signal +
    # signal_id IS NULL → breach) from "by-design protective close"
    # (origin=stop_loss/liquidation + signal_id IS NULL → legitimate Cat-B).
    order_origin: Literal["signal", "stop_loss", "liquidation", "manual"] = "signal"
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
