"""Order reservations shared by ordinary and protective execution and by reconciliation.

A pending order reserves quantity and cash until its fills are projected; ``outstanding_reservation`` is what
it still holds. ``ExecutionConflictError`` is the execution layer's fail-closed error."""

import math
from decimal import Decimal

from sqlalchemy import select

from poseidon.models.order_fill import OrderFillRecord
from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS

ACTIVE_RESERVATION_STATUSES = frozenset({"pending_submit", "reconciliation_required", "submitted", "partially_filled"})


PROTECTIVE_ORIGINS = DURABLE_PROTECTIVE_ORIGINS


class ExecutionConflictError(RuntimeError):
    """The frozen decision cannot be materialized without changing its meaning."""


def _finite_positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ExecutionConflictError(f"{field} must be finite and positive")
    return float(value)


def outstanding_reservation(session, order, currency, *, lock_fills=True):
    original = Decimal(str(_finite_positive(order.reserved_quantity, "reserved_quantity")))
    statement = select(OrderFillRecord).where(OrderFillRecord.order_id == order.id)
    if lock_fills:
        statement = statement.with_for_update()
    fills = session.scalars(statement).all()
    applied = Decimal("0")
    total = Decimal("0")
    pending_projection = False
    for fill in fills:
        if fill.projection_status not in {"projection_pending", "applied"}:
            raise ExecutionConflictError("pending reservation has invalid fill projection status")
        quantity = Decimal(str(_finite_positive(fill.fill_quantity, "reserved fill quantity")))
        total += quantity
        if fill.projection_status == "applied":
            applied += quantity
        else:
            pending_projection = True
    if total > original:
        raise ExecutionConflictError("reserved fills exceed original reservation")
    cash = order.reserved_cash_json
    if (
        not isinstance(cash, dict)
        or cash.get("currency") != currency
        or isinstance(cash.get("amount"), bool)
        or not isinstance(cash.get("amount"), (int, float))
        or not math.isfinite(cash["amount"])
        or cash["amount"] < 0
    ):
        raise ExecutionConflictError("pending reservation has invalid reserved cash")
    if not isinstance(order.intent_json, dict) or not isinstance(order.intent_json.get("frozen_intent"), dict):
        raise ExecutionConflictError("pending reservation has invalid frozen action")
    action = order.intent_json["frozen_intent"].get("action")
    if not isinstance(action, str) or action not in {"enter", "add", "reduce", "exit"}:
        raise ExecutionConflictError("pending reservation has invalid frozen action")
    if action in {"reduce", "exit"} and cash["amount"] != 0:
        raise ExecutionConflictError("reserved cash disagrees with frozen action")
    released_terminal = (
        order.reservation_status == "released"
        and order.status in {"filled", "rejected", "cancelled"}
        and not pending_projection
    )
    outstanding = Decimal("0") if released_terminal else original - applied
    return outstanding, Decimal(str(cash["amount"])) * outstanding / original
