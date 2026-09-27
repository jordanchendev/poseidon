"""Order status enum and state transition validation."""

from enum import StrEnum


class OrderStatus(StrEnum):
    PENDING = "pending"
    PENDING_SUBMIT = "pending_submit"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.PENDING_SUBMIT, OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.PENDING_SUBMIT: {OrderStatus.RECONCILIATION_REQUIRED},
    OrderStatus.RECONCILIATION_REQUIRED: {
        OrderStatus.SUBMITTED,
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.RECONCILIATION_REQUIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.RECONCILIATION_REQUIRED,
    },
    OrderStatus.FILLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.CANCELLED: set(),
}


def transition_order(current: OrderStatus, new: OrderStatus) -> OrderStatus:
    """Validate and execute an order status transition.

    Raises ValueError if the transition is not allowed.
    """
    if new not in VALID_TRANSITIONS[current]:
        raise ValueError(f"Invalid transition: {current} -> {new}")
    return new
