"""Orders package -- order schemas, state machine, risk checker, and manager."""

from poseidon.orders.schemas import Fill, Order, OrderResult, RiskCheckResult
from poseidon.orders.state_machine import (
    VALID_TRANSITIONS,
    OrderStatus,
    transition_order,
)

__all__ = [
    "VALID_TRANSITIONS",
    "Fill",
    "Order",
    "OrderManager",
    "OrderResult",
    "OrderRiskChecker",
    "OrderStatus",
    "RiskCheckResult",
    "transition_order",
    "weight_to_shares",
]


def __getattr__(name: str):
    """Lazy imports for OrderManager/OrderRiskChecker to avoid circular import."""
    if name == "OrderManager":
        from poseidon.orders.manager import OrderManager

        return OrderManager
    if name == "weight_to_shares":
        from poseidon.orders.manager import weight_to_shares

        return weight_to_shares
    if name == "OrderRiskChecker":
        from poseidon.orders.risk_checker import OrderRiskChecker

        return OrderRiskChecker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
