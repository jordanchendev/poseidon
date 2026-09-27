"""Short transaction boundaries for decision-order submission and recovery."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from poseidon.broker.base import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCapabilityError,
    BrokerFillSnapshot,
    BrokerOrderSnapshot,
)
from poseidon.decision_loop.execution import DecisionExecutionService, ExecutionConflictError
from poseidon.decision_loop.manifest import timestamp
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.position_lot import PositionLot
from poseidon.orders.schemas import Order
from poseidon.orders.state_machine import OrderStatus, transition_order


class ReconciliationConflictError(RuntimeError):
    """Broker and internal state cannot be reconciled without guessing."""


@dataclass(frozen=True)
class PreparedAttempt:
    order: Order
    should_submit: bool


def _number(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReconciliationConflictError("broker economics must be finite")
    return Decimal(str(value))


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _order_dto(record: OrderRecord) -> Order:
    return Order(
        id=str(record.id),
        strategy_name=record.strategy_name,
        symbol=record.symbol,
        market=record.market,
        action=record.action,
        order_type=record.order_type,
        target_weight=record.target_weight,
        quantity=record.quantity,
        price=record.price,
        side=record.side,
        status=record.status,
        broker_order_id=record.broker_order_id,
        broker_mode=record.broker_mode,
        reject_reason=record.reject_reason,
        signal_id=record.signal_id,
        order_origin=record.order_origin,
        decision_id=record.decision_id,
        account_scope=record.account_scope,
        account_generation=record.account_generation,
        execution_key=record.execution_key,
        client_order_ref=record.client_order_ref,
        instrument=record.instrument,
        intent_json=record.intent_json,
        intent_sha256=record.intent_sha256,
        reserved_cash_json=record.reserved_cash_json,
        reserved_quantity=record.reserved_quantity,
        reservation_status=record.reservation_status,
        reconciliation_status=record.reconciliation_status,
        submit_attempted_at=record.submit_attempted_at,
        protective_context_json=record.protective_context_json,
    )


def _normalized_status(status: str) -> str:
    aliases = {
        "accepted": "submitted",
        "new": "submitted",
        "open": "submitted",
        "pending": "submitted",
        "submitted": "submitted",
        "partial": "partially_filled",
        "partially_filled": "partially_filled",
        "filled": "filled",
        "rejected": "rejected",
        "cancelled": "cancelled",
        "canceled": "cancelled",
    }
    try:
        return aliases[status.lower()]
    except (AttributeError, KeyError) as error:
        raise ReconciliationConflictError(f"unsupported broker order status: {status!r}") from error


def _difference(field, internal, broker):
    return {"field": field, "internal": internal, "broker": broker}


class ReconciliationService:
    """Mutate reconciliation state in a caller-owned transaction; flush only."""

    def __init__(self, session):
        self.session = session

    def _locked_order(self, order_id) -> OrderRecord:
        order = self.session.scalar(
            select(OrderRecord)
            .where(OrderRecord.id == order_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if order is None:
            raise ReconciliationConflictError("decision order does not exist")
        return order

    def _validate_durable_intent(self, order: OrderRecord) -> None:
        decision = self.session.get(DecisionRecord, order.decision_id)
        if decision is None:
            raise ReconciliationConflictError("durable order intent changed after materialization")
        execution = DecisionExecutionService(self.session)
        try:
            _, policy, intents = execution._policy_and_intents(decision)
            reconciliation = policy.reconciliation
            replay = execution._replay(decision, intents, reconciliation.account_generation)
        except ExecutionConflictError as error:
            raise ReconciliationConflictError("durable order intent changed after materialization") from error
        if replay is None or str(order.id) not in replay["order_ids"]:
            raise ReconciliationConflictError("durable order intent changed after materialization")

    def _locked_order_and_account(self, order_id) -> OrderRecord:
        probe = self.session.get(OrderRecord, order_id)
        if probe is None or not probe.account_scope or not probe.account_generation:
            raise ReconciliationConflictError("decision order has incomplete account identity")
        identity = (probe.account_scope, probe.account_generation)
        DecisionExecutionService(self.session)._advisory_lock(*identity)
        account = self.session.scalar(
            select(PaperBrokerAccount)
            .where(
                PaperBrokerAccount.account_scope == identity[0],
                PaperBrokerAccount.account_generation == identity[1],
            )
            .with_for_update()
        )
        if account is None:
            raise ReconciliationConflictError("approved paper account generation does not exist")
        order = self._locked_order(order_id)
        if (order.account_scope, order.account_generation) != identity:
            raise ReconciliationConflictError("decision order account identity changed while locking")
        return order

    def prepare_attempt(self, order_id, adapter: BrokerAdapter, *, now=None) -> PreparedAttempt:
        """Lock and mark one never-attempted intent before any adapter I/O."""
        order = self._locked_order(order_id)
        self._validate_durable_intent(order)
        dto = _order_dto(order)
        try:
            adapter._require_decision_submission(dto, order.client_order_ref)
        except BrokerCapabilityError as error:
            raise ReconciliationConflictError(str(error)) from error
        if order.submit_attempted_at is not None:
            return PreparedAttempt(dto, False)
        if order.status != "pending_submit" or order.reconciliation_status != "pending":
            raise ReconciliationConflictError("only a pending never-attempted intent may be submitted")
        attempted_at = timestamp(now if now is not None else datetime.now(UTC), "now")
        order.submit_attempted_at = attempted_at
        order.status = "reconciliation_required"
        order.reconciliation_status = "required"
        order.updated_at = attempted_at
        self.session.flush()
        return PreparedAttempt(_order_dto(order), True)

    def mark_lookup_unknown(self, order_id, *, now=None):
        """Persist missing broker lookup without resetting a legal terminal status."""
        order = self._locked_order(order_id)
        if order.submit_attempted_at is None:
            raise ReconciliationConflictError("never-attempted order cannot have an unknown lookup")
        if order.reconciliation_status != "required":
            order.reconciliation_status = "required"
            order.updated_at = timestamp(now if now is not None else datetime.now(UTC), "now")
        self.session.flush()
        return order.status

    def _release_terminal_reservation(self, order, pending_projection, imported_at):
        if order.status not in {"filled", "rejected", "cancelled"} or pending_projection:
            return False
        if order.reservation_status == "released":
            return False
        action = (order.intent_json or {}).get("frozen_intent", {}).get("action")
        order.reservation_status = "released"
        if action in {"reduce", "exit"}:
            self.session.flush()
            reserved_orders = self.session.scalars(
                select(OrderRecord)
                .where(
                    OrderRecord.account_scope == order.account_scope,
                    OrderRecord.account_generation == order.account_generation,
                    OrderRecord.market == order.market,
                    OrderRecord.symbol == order.symbol,
                    OrderRecord.instrument == order.instrument,
                    OrderRecord.side == order.side,
                    OrderRecord.reservation_status == "reserved",
                )
                .order_by(OrderRecord.created_at, OrderRecord.id)
                .with_for_update()
            ).all()
            required = Decimal("0")
            for reserved_order in reserved_orders:
                reserved_action = (reserved_order.intent_json or {}).get("frozen_intent", {}).get("action")
                if reserved_action not in {"reduce", "exit"}:
                    continue
                reservation_fills = self.session.scalars(
                    select(OrderFillRecord).where(OrderFillRecord.order_id == reserved_order.id).with_for_update()
                ).all()
                if any(fill.projection_status not in {"projection_pending", "applied"} for fill in reservation_fills):
                    raise ReconciliationConflictError("internal fill has an invalid projection status")
                applied = sum(
                    (_number(fill.fill_quantity) for fill in reservation_fills if fill.projection_status == "applied"),
                    start=Decimal("0"),
                )
                outstanding = _number(reserved_order.reserved_quantity) - applied
                if outstanding < 0:
                    raise ReconciliationConflictError("applied close fills exceed the durable reservation")
                required += outstanding

            lots = self.session.scalars(
                select(PositionLot)
                .where(
                    PositionLot.account_scope == order.account_scope,
                    PositionLot.account_generation == order.account_generation,
                    PositionLot.market == order.market,
                    PositionLot.symbol == order.symbol,
                    PositionLot.instrument == order.instrument,
                    PositionLot.side == order.side,
                )
                .order_by(PositionLot.opened_at, PositionLot.id)
                .with_for_update()
            ).all()
            remaining = required
            for lot in lots:
                rebuilt = min(_number(lot.open_quantity), remaining)
                if _number(lot.reserved_close_quantity) != rebuilt:
                    lot.reserved_close_quantity = float(rebuilt)
                    lot.updated_at = imported_at
                remaining -= rebuilt
            if remaining:
                raise ReconciliationConflictError("close reservation demand exceeds open position lots")
        return True

    @staticmethod
    def _validate_snapshot(order: OrderRecord, snapshot: BrokerOrderSnapshot) -> None:
        expected = (
            order.client_order_ref,
            order.account_scope,
            order.account_generation,
            order.market,
            order.symbol,
            order.instrument,
            order.action,
            order.side,
            order.order_type,
            _number(order.quantity),
        )
        actual = (
            snapshot.client_order_ref,
            snapshot.account_scope,
            snapshot.account_generation,
            snapshot.market,
            snapshot.symbol,
            snapshot.instrument,
            snapshot.action,
            snapshot.side,
            snapshot.order_type,
            _number(snapshot.quantity),
        )
        if expected != actual:
            raise ReconciliationConflictError("broker order identity or economics changed")
        if order.broker_order_id is not None and order.broker_order_id != snapshot.broker_order_id:
            raise ReconciliationConflictError("client reference resolved to a different broker order")

    @staticmethod
    def _validate_fill(order: OrderRecord, snapshot: BrokerOrderSnapshot, fill: BrokerFillSnapshot) -> None:
        if (
            fill.broker_order_id,
            fill.account_scope,
            fill.account_generation,
            fill.market,
            fill.symbol,
            fill.instrument,
            fill.side,
        ) != (
            snapshot.broker_order_id,
            order.account_scope,
            order.account_generation,
            order.market,
            order.symbol,
            order.instrument,
            order.side,
        ):
            raise ReconciliationConflictError("broker fill identity changed")
        if not fill.broker_fill_id or _number(fill.fill_price) < 0 or _number(fill.fill_quantity) <= 0:
            raise ReconciliationConflictError("broker fill economics are invalid")

    def import_broker_state(self, order_id, snapshot, fills, *, now=None):
        """Import one normalized broker result and exact-replay-safe fills."""
        if not isinstance(snapshot, BrokerOrderSnapshot):
            raise ReconciliationConflictError("adapter did not return a normalized broker order")
        order = self._locked_order_and_account(order_id)
        imported_at = timestamp(now if now is not None else datetime.now(UTC), "now")
        before = (order.broker_order_id, order.status, order.reconciliation_status, order.reservation_status)
        self._validate_snapshot(order, snapshot)
        normalized = _normalized_status(snapshot.status)
        if order.status != normalized:
            try:
                transition_order(OrderStatus(order.status), OrderStatus(normalized))
            except ValueError as error:
                raise ReconciliationConflictError(str(error)) from error

        broker_fills = list(fills)
        stored_fill_ids = set(
            self.session.scalars(
                select(OrderFillRecord.broker_fill_id).where(OrderFillRecord.order_id == order.id)
            ).all()
        )
        seen = set()
        total = Decimal("0")
        total_notional = Decimal("0")
        new_fill = False
        pending_projection = False
        for fill in broker_fills:
            if not isinstance(fill, BrokerFillSnapshot):
                raise ReconciliationConflictError("adapter did not return normalized broker fills")
            self._validate_fill(order, snapshot, fill)
            if fill.broker_fill_id in seen:
                raise ReconciliationConflictError("adapter returned a duplicate fill identity")
            seen.add(fill.broker_fill_id)
            total += _number(fill.fill_quantity)
            total_notional += _number(fill.fill_price) * _number(fill.fill_quantity)
            existing = self.session.scalar(
                select(OrderFillRecord).where(
                    OrderFillRecord.order_id == order.id,
                    OrderFillRecord.broker_fill_id == fill.broker_fill_id,
                )
            )
            economics = (_number(fill.fill_price), _number(fill.fill_quantity), _utc(fill.fill_time))
            if existing is not None:
                stored = (_number(existing.fill_price), _number(existing.fill_quantity), _utc(existing.fill_time))
                if stored != economics:
                    raise ReconciliationConflictError("fill identity changed economics")
                if existing.projection_status not in {"projection_pending", "applied"}:
                    raise ReconciliationConflictError("internal fill has an invalid projection status")
                pending_projection = pending_projection or existing.projection_status == "projection_pending"
                continue
            new_fill = True
            pending_projection = True
            self.session.add(
                OrderFillRecord(
                    id=uuid.uuid4(),
                    order_id=order.id,
                    fill_price=fill.fill_price,
                    fill_quantity=fill.fill_quantity,
                    fill_time=_utc(fill.fill_time),
                    broker_fill_id=fill.broker_fill_id,
                    projection_status="projection_pending",
                    created_at=imported_at,
                )
            )
        if not stored_fill_ids.issubset(seen):
            raise ReconciliationConflictError("broker replay omitted an imported fill")
        if total > _number(order.quantity):
            raise ReconciliationConflictError("broker fills exceed the durable order quantity")
        if total and snapshot.price is not None and _number(float(total_notional / total)) != _number(snapshot.price):
            raise ReconciliationConflictError("broker snapshot price disagrees with fill economics")
        if normalized == "filled" and total != _number(order.quantity):
            raise ReconciliationConflictError("filled broker order lacks exact fill quantity")
        if normalized == "partially_filled" and (total <= 0 or total >= _number(order.quantity)):
            raise ReconciliationConflictError("partial broker order has invalid fill quantity")
        if normalized in {"submitted", "rejected"} and total:
            raise ReconciliationConflictError(f"{normalized} broker order cannot contain fills")

        order.broker_order_id = snapshot.broker_order_id
        order.status = normalized
        order.reconciliation_status = "required" if normalized in {"submitted", "partially_filled"} else "resolved"
        reservation_changed = self._release_terminal_reservation(order, pending_projection, imported_at)
        after = (order.broker_order_id, order.status, order.reconciliation_status, order.reservation_status)
        if new_fill or reservation_changed or before != after:
            order.updated_at = imported_at
        self.session.flush()
        return {
            "order_id": str(order.id),
            "broker_order_id": order.broker_order_id,
            "status": order.status,
            "operation": "import_broker_state",
            "fill_ids": sorted(seen),
        }

    def compare_execution_state(
        self,
        order_id,
        snapshot: BrokerOrderSnapshot,
        fills: list[BrokerFillSnapshot],
        account: BrokerAccountSnapshot,
    ) -> list[dict]:
        """Return observable drift without mutating either truth surface."""
        order = self.session.get(OrderRecord, order_id)
        if order is None:
            raise ReconciliationConflictError("decision order does not exist")
        differences = []
        pairs = {
            "order.account_scope": (order.account_scope, snapshot.account_scope),
            "order.account_generation": (order.account_generation, snapshot.account_generation),
            "order.market": (order.market, snapshot.market),
            "order.symbol": (order.symbol, snapshot.symbol),
            "order.instrument": (order.instrument, snapshot.instrument),
            "order.side": (order.side, snapshot.side),
            "order.action": (order.action, snapshot.action),
            "order.quantity": (_number(order.quantity), _number(snapshot.quantity)),
        }
        economics = (order.intent_json or {}).get("economics", {})
        if "price" in economics:
            pairs["order.price"] = (_number(order.price), _number(economics["price"]))
        for field, (internal, broker) in pairs.items():
            if internal != broker:
                differences.append(_difference(field, str(internal), str(broker)))

        internal_fills = {
            fill.broker_fill_id: fill
            for fill in self.session.scalars(select(OrderFillRecord).where(OrderFillRecord.order_id == order.id)).all()
        }
        broker_fills = {fill.broker_fill_id: fill for fill in fills}
        for fill_id in sorted(set(internal_fills) | set(broker_fills)):
            internal = internal_fills.get(fill_id)
            broker = broker_fills.get(fill_id)
            if internal is None or broker is None:
                differences.append(_difference(f"fill.{fill_id}", bool(internal), bool(broker)))
                continue
            for name in ("fill_price", "fill_quantity"):
                internal_value = _number(getattr(internal, name))
                broker_value = _number(getattr(broker, name))
                if internal_value != broker_value:
                    differences.append(_difference(f"fill.{fill_id}.{name}", str(internal_value), str(broker_value)))

        action = (order.intent_json or {}).get("frozen_intent", {}).get("action")
        if action in {"enter", "add"}:
            multiplier = _number(economics.get("contract_multiplier", 1.0))
            frozen_debit = (
                _number(economics.get("price")) * _number(economics.get("materialized_quantity")) * multiplier
            )
            reserved = _number((order.reserved_cash_json or {}).get("amount"))
            if reserved != frozen_debit:
                differences.append(_difference("cash.reserved_amount", str(reserved), str(frozen_debit)))
        if (account.account_scope, account.account_generation) != (order.account_scope, order.account_generation):
            differences.append(
                _difference(
                    "cash.account_identity",
                    f"{order.account_scope}/{order.account_generation}",
                    f"{account.account_scope}/{account.account_generation}",
                )
            )
        return differences


def submit_or_reconcile_order(session_factory, order_id, adapter: BrokerAdapter, *, now=None):
    """Commit the attempt, perform one broker operation, then import separately."""
    with session_factory() as session, session.begin():
        prepared = ReconciliationService(session).prepare_attempt(order_id, adapter, now=now)

    order = prepared.order
    if prepared.should_submit:
        snapshot = adapter.place_decision_order(order, client_order_ref=order.client_order_ref)
    else:
        snapshot = adapter.find_order_by_client_ref(
            order.client_order_ref,
            account_scope=order.account_scope,
            account_generation=order.account_generation,
        )
    if snapshot is None:
        with session_factory() as session, session.begin():
            status = ReconciliationService(session).mark_lookup_unknown(order_id, now=now)
        return {
            "order_id": str(order_id),
            "broker_order_id": None,
            "status": status,
            "reconciliation_status": "required",
            "operation": "reconcile_order",
            "fill_ids": [],
        }
    if not isinstance(snapshot, BrokerOrderSnapshot):
        raise ReconciliationConflictError("adapter did not return a normalized broker order")
    fills = adapter.query_fills(
        snapshot.broker_order_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )
    with session_factory() as session, session.begin():
        return ReconciliationService(session).import_broker_state(order_id, snapshot, fills, now=now)
