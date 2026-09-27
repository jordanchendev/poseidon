"""One account-locked, replay-safe FIFO fill projection."""

import math
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from poseidon.decision_loop.manifest import content_sha256, iso_time
from poseidon.decision_loop.reconciliation import ReconciliationConflictError, ReconciliationService, _utc
from poseidon.models.fill_allocation import FillAllocation
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.position_lot import PositionLot
from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS

IDENTITY_FIELDS = ("account_scope", "account_generation", "market", "symbol", "instrument", "side")


class FillProjectionConflictError(ReconciliationConflictError):
    """The fill cannot be projected without guessing or changing economics."""


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FillProjectionConflictError("projection economics must be finite numbers")
    return Decimal(str(value))


class FillProjectionService:
    """Plan/validate all changes, then flush in the caller's transaction."""

    def __init__(self, session):
        self.session = session

    def _economics(self, order, fill, account):
        if (
            account is None
            or not isinstance(order.intent_json, dict)
            or not isinstance(order.intent_json.get("frozen_intent"), dict)
            or not isinstance(order.intent_json.get("economics"), dict)
        ):
            raise FillProjectionConflictError("fill account or persisted intent is invalid")
        if (
            order.order_origin not in {"decision", *DURABLE_PROTECTIVE_ORIGINS}
            or order.broker_mode != "paper"
            or not order.client_order_ref
            or not order.broker_order_id
            or not fill.broker_fill_id
            or order.submit_attempted_at is None
            or fill.order_id != order.id
            or order.side not in {"long", "short"}
            or any(not getattr(order, field) for field in IDENTITY_FIELDS)
        ):
            raise FillProjectionConflictError("fill lacks canonical decision/client/order identity")
        try:
            ReconciliationService(self.session)._validate_durable_intent(order)
        except ReconciliationConflictError as error:
            raise FillProjectionConflictError(str(error)) from error
        action = order.intent_json["frozen_intent"]["action"]
        opening = action in {"enter", "add"}
        expected_action = "buy" if (order.side == "long") == opening else "sell"
        if action not in {"enter", "add", "reduce", "exit"} or order.action != expected_action:
            raise FillProjectionConflictError("fill action and position side disagree")
        quantity, price = _number(fill.fill_quantity), _number(fill.fill_price)
        if quantity <= 0 or price <= 0:
            raise FillProjectionConflictError("fill quantity and price must be positive")
        multiplier = _number(order.intent_json["economics"].get("contract_multiplier"))
        if multiplier <= 0:
            raise FillProjectionConflictError("contract multiplier must be positive")
        rules = order.intent_json["economics"].get("sizing_rules", {})
        if order.market == "crypto_perp":
            if not isinstance(rules, dict):
                raise FillProjectionConflictError("persisted sizing rules are invalid")
            if (rules.get("margin_semantics"), rules.get("funding_semantics")) != ("full_notional", "excluded"):
                raise FillProjectionConflictError("unsupported paper collateral semantics")
            if _number(rules.get("contract_multiplier")) != multiplier:
                raise FillProjectionConflictError("frozen multiplier changed")
        elif order.market != "tw_stock" or multiplier != 1:
            raise FillProjectionConflictError("unsupported position market or multiplier")
        if fill.fill_time is None:
            raise FillProjectionConflictError("fill time is missing")
        digest = content_sha256(
            {
                "fill_id": str(fill.id),
                "order_id": str(order.id),
                "decision_id": str(order.decision_id),
                "identity": {field: getattr(order, field) for field in IDENTITY_FIELDS},
                "client_order_ref": order.client_order_ref,
                "broker_order_id": order.broker_order_id,
                "broker_fill_id": fill.broker_fill_id,
                "intent_sha256": order.intent_sha256,
                "quantity": float(quantity),
                "price": float(price),
                "fill_time": iso_time(_utc(fill.fill_time), "fill_time"),
            }
        )
        cost = {
            "projection_sha256": digest,
            "unit_price": float(price),
            "contract_multiplier": float(multiplier),
            "currency": account.currency,
            "original_cost": float(quantity * price * multiplier),
        }
        if not math.isfinite(cost["original_cost"]):
            raise FillProjectionConflictError("fill cost must be finite")
        return opening, quantity, cost

    def _validate_lot(self, lot, identity, account):
        if lot is None or any(getattr(lot, field) != value for field, value in identity.items()):
            raise FillProjectionConflictError("lot identity changed")
        opening_fill = self.session.get(OrderFillRecord, lot.opening_fill_id)
        opening_order = None if opening_fill is None else self.session.get(OrderRecord, opening_fill.order_id)
        if opening_order is None or opening_fill.projection_status != "applied":
            raise FillProjectionConflictError("lot lacks an applied opening fill")
        opening, quantity, cost = self._economics(opening_order, opening_fill, account)
        if (
            not opening
            or lot.opening_decision_id != opening_order.decision_id
            or _number(lot.original_quantity) != quantity
            or lot.cost_basis_json != cost
            or _utc(lot.opened_at) != _utc(opening_fill.fill_time)
            or any(getattr(opening_order, field) != value for field, value in identity.items())
        ):
            raise FillProjectionConflictError("opening lot identity or economics changed")
        opened, reserved = _number(lot.open_quantity), _number(lot.reserved_close_quantity)
        if opened < 0 or opened > quantity or reserved < 0 or reserved > opened:
            raise FillProjectionConflictError("lot quantity or reservation is invalid")
        allocations = self.session.scalars(
            select(FillAllocation).where(FillAllocation.position_lot_id == lot.id).with_for_update()
        ).all()
        amounts = [_number(row.quantity) for row in allocations]
        if any(amount <= 0 for amount in amounts) or opened != quantity - sum(amounts, Decimal(0)):
            raise FillProjectionConflictError("lot balance disagrees with closing allocations")

    def _reservation_need(self, identity, applying_fill=None, release=False):
        reserved_orders = self.session.scalars(
            select(OrderRecord)
            .filter_by(**identity, reservation_status="reserved")
            .order_by(OrderRecord.created_at, OrderRecord.id)
            .with_for_update()
        ).all()
        required = Decimal(0)
        for sibling in reserved_orders:
            if not isinstance(sibling.intent_json, dict) or not isinstance(
                sibling.intent_json.get("frozen_intent"), dict
            ):
                raise FillProjectionConflictError("sibling persisted intent is invalid")
            action = sibling.intent_json["frozen_intent"].get("action")
            if not isinstance(action, str) or action not in {"enter", "add", "reduce", "exit"}:
                raise FillProjectionConflictError("sibling intent action is invalid")
            if (applying_fill is not None and sibling.id == applying_fill.order_id and release) or action in {
                "enter",
                "add",
            }:
                continue
            sibling_fills = self.session.scalars(
                select(OrderFillRecord).where(OrderFillRecord.order_id == sibling.id).with_for_update()
            ).all()
            if any(
                row.projection_status not in {"projection_pending", "applied"} or _number(row.fill_quantity) <= 0
                for row in sibling_fills
            ):
                raise FillProjectionConflictError("sibling fill has invalid economics or projection status")
            applied = sum(
                (
                    _number(row.fill_quantity)
                    for row in sibling_fills
                    if row.projection_status == "applied" or (applying_fill is not None and row.id == applying_fill.id)
                ),
                Decimal(0),
            )
            outstanding = _number(sibling.reserved_quantity) - applied
            if outstanding < 0:
                raise FillProjectionConflictError("applied close fills exceed their durable reservation")
            required += outstanding
        return required

    @staticmethod
    def _allocation_cost(fill_cost, lot, quantity):
        entry_price = _number(lot.cost_basis_json["unit_price"])
        multiplier = _number(lot.cost_basis_json["contract_multiplier"])
        if multiplier != _number(fill_cost["contract_multiplier"]):
            raise FillProjectionConflictError("closing fill and opening lot multiplier disagree")
        return {
            "projection_sha256": fill_cost["projection_sha256"],
            "entry_price": float(entry_price),
            "contract_multiplier": float(multiplier),
            "currency": fill_cost["currency"],
            "entry_cost": float(quantity * entry_price * multiplier),
        }

    def apply(self, fill_id):
        probe = self.session.get(OrderFillRecord, fill_id)
        if probe is None:
            raise FillProjectionConflictError("persisted fill does not exist")
        order_id = probe.order_id
        try:
            order = ReconciliationService(self.session)._locked_order_and_account(order_id)
        except ReconciliationConflictError as error:
            raise FillProjectionConflictError(str(error)) from error
        fill = self.session.scalar(
            select(OrderFillRecord)
            .where(OrderFillRecord.id == fill_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if fill is None or fill.order_id != order_id or fill.projection_status not in {"projection_pending", "applied"}:
            raise FillProjectionConflictError("fill identity or projection status changed")
        identity = {field: getattr(order, field) for field in IDENTITY_FIELDS}
        account = self.session.scalar(
            select(PaperBrokerAccount).where(
                PaperBrokerAccount.account_scope == order.account_scope,
                PaperBrokerAccount.account_generation == order.account_generation,
            )
        )
        opening, quantity, cost = self._economics(order, fill, account)
        order_fills = self.session.scalars(
            select(OrderFillRecord)
            .where(OrderFillRecord.order_id == order.id)
            .order_by(OrderFillRecord.id)
            .with_for_update()
        ).all()
        if any(row.projection_status not in {"projection_pending", "applied"} for row in order_fills) or sum(
            (_number(row.fill_quantity) for row in order_fills), Decimal(0)
        ) > _number(order.quantity):
            raise FillProjectionConflictError("order fill quantity or projection state is invalid")
        lots = self.session.scalars(
            select(PositionLot)
            .filter_by(**identity)
            .order_by(PositionLot.opened_at, PositionLot.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        own_lot = self.session.scalar(
            select(PositionLot).where(PositionLot.opening_fill_id == fill.id).with_for_update()
        )
        allocations = self.session.scalars(
            select(FillAllocation)
            .where(FillAllocation.closing_fill_id == fill.id)
            .join(PositionLot)
            .order_by(PositionLot.opened_at, PositionLot.id)
            .with_for_update()
        ).all()
        if fill.projection_status == "applied":
            required = self._reservation_need(identity)
            for lot in lots:
                expected = min(_number(lot.open_quantity), required)
                if _number(lot.reserved_close_quantity) != expected:
                    raise FillProjectionConflictError("lot reservation disagrees with durable close orders")
                required -= expected
            if required:
                raise FillProjectionConflictError("close reservations exceed remaining inventory")
            if opening:
                if own_lot is None or allocations:
                    raise FillProjectionConflictError("applied opening projection is missing or conflicting")
                self._validate_lot(own_lot, identity, account)
                return self._response(fill, [own_lot], [])
            if own_lot is not None or not allocations:
                raise FillProjectionConflictError("applied closing projection is missing or conflicting")
            replay_lots = []
            for allocation in allocations:
                lot = self.session.get(PositionLot, allocation.position_lot_id)
                self._validate_lot(lot, identity, account)
                amount = _number(allocation.quantity)
                if (
                    amount <= 0
                    or allocation.closing_decision_id != order.decision_id
                    or allocation.realized_cost_json != self._allocation_cost(cost, lot, amount)
                ):
                    raise FillProjectionConflictError("applied allocation economics changed")
                replay_lots.append(lot)
            if sum((_number(row.quantity) for row in allocations), Decimal(0)) != quantity:
                raise FillProjectionConflictError("applied allocation quantity changed")
            return self._response(fill, replay_lots, allocations)
        if own_lot is not None or allocations:
            raise FillProjectionConflictError("pending fill already has a conflicting projection")
        for lot in lots:
            self._validate_lot(lot, identity, account)
        changes = {lot.id: _number(lot.open_quantity) for lot in lots}
        projected_lots, planned_allocations = [], []
        if opening:
            lot = PositionLot(
                id=uuid.uuid4(),
                **identity,
                opening_fill_id=fill.id,
                opening_decision_id=order.decision_id,
                original_quantity=float(quantity),
                open_quantity=float(quantity),
                reserved_close_quantity=0,
                cost_basis_json=cost,
                opened_at=_utc(fill.fill_time),
            )
            lots.append(lot)
            lots.sort(key=lambda row: (_utc(row.opened_at), row.id))
            changes[lot.id] = quantity
            projected_lots.append(lot)
        else:
            pending_fills = self.session.execute(
                select(OrderFillRecord, OrderRecord)
                .join(OrderRecord, OrderRecord.id == OrderFillRecord.order_id)
                .where(
                    *(getattr(OrderRecord, field) == value for field, value in identity.items()),
                    OrderFillRecord.projection_status == "projection_pending",
                    OrderFillRecord.fill_time <= fill.fill_time,
                    OrderFillRecord.id != fill.id,
                )
                .with_for_update()
            ).all()
            for earlier_fill, earlier_order in pending_fills:
                earlier_opening, _, _ = self._economics(earlier_order, earlier_fill, account)
                if earlier_opening:
                    raise FillProjectionConflictError("earlier opening fill requires projection before FIFO close")
            remaining = quantity
            for lot in lots:
                amount = min(remaining, changes[lot.id])
                if amount <= 0:
                    continue
                changes[lot.id] -= amount
                remaining -= amount
                planned_allocations.append(
                    FillAllocation(
                        id=uuid.uuid4(),
                        closing_fill_id=fill.id,
                        position_lot_id=lot.id,
                        closing_decision_id=order.decision_id,
                        quantity=float(amount),
                        realized_cost_json=self._allocation_cost(cost, lot, amount),
                    )
                )
                projected_lots.append(lot)
                if not remaining:
                    break
            if remaining:
                raise FillProjectionConflictError("closing fill exceeds open FIFO inventory")

        pending = any(row.id != fill.id and row.projection_status == "projection_pending" for row in order_fills)
        release = order.status in {"filled", "rejected", "cancelled"} and not pending
        required = self._reservation_need(identity, fill, release)
        reservations = {}
        for lot in lots:
            reservations[lot.id] = min(changes[lot.id], required)
            required -= reservations[lot.id]
        if required:
            raise FillProjectionConflictError("sibling close reservations exceed remaining inventory")

        # All validation precedes mutation: failures cannot leave half a FIFO close.
        now = datetime.now(UTC)
        for lot in lots:
            if lot in projected_lots and opening:
                self.session.add(lot)
            if lot.open_quantity != float(changes[lot.id]) or lot.reserved_close_quantity != float(
                reservations[lot.id]
            ):
                lot.open_quantity = float(changes[lot.id])
                lot.reserved_close_quantity = float(reservations[lot.id])
                lot.updated_at = now
        self.session.add_all(planned_allocations)
        fill.projection_status = "applied"
        if release:
            order.reservation_status = "released"
        order.updated_at = now
        self.session.flush()
        return self._response(fill, projected_lots, planned_allocations)

    @staticmethod
    def _response(fill, lots, allocations):
        return {
            "fill_id": str(fill.id),
            "projection_status": "applied",
            "lot_ids": [str(lot.id) for lot in lots],
            "allocation_ids": [str(row.id) for row in allocations],
        }


def apply_fill_projection(session_factory, fill_id):
    """Commit one fill projection, retaining an observable reconciliation gap on failure."""
    try:
        with session_factory() as session, session.begin():
            return FillProjectionService(session).apply(fill_id)
    except FillProjectionConflictError as error:
        try:
            with session_factory() as session, session.begin():
                fill = session.get(OrderFillRecord, fill_id)
                if fill is not None:
                    reconciliation = ReconciliationService(session)
                    try:
                        order = reconciliation._locked_order_and_account(fill.order_id)
                    except ReconciliationConflictError:
                        # A missing account cannot prevent recording an order-only gap.
                        order = reconciliation._locked_order(fill.order_id)
                    order.reconciliation_status = "required"
                    order.updated_at = datetime.now(UTC)
        except ReconciliationConflictError as marker_error:
            error.add_note(f"reconciliation marker unavailable: {marker_error}")
        raise
