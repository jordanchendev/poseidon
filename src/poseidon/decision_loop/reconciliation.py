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
from poseidon.decision_loop.execution import (
    PROTECTIVE_ORIGINS,
    DecisionExecutionService,
    ExecutionConflictError,
    ProtectiveExecutionService,
    internal_state_watermark,
)
from poseidon.decision_loop.manifest import canonical_json, content_sha256, iso_time, timestamp
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion
from poseidon.orders.schemas import Order
from poseidon.orders.state_machine import OrderStatus, transition_order


class ReconciliationConflictError(RuntimeError):
    """Broker and internal state cannot be reconciled without guessing."""


def paper_liquidation_nav(snapshot, policy, prices, ledger):
    """Value independent inventory at current marks, including short collateral."""
    if not isinstance(snapshot.positions, (tuple, list)):
        raise ValueError("paper snapshot requires complete positions")

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("paper valuation requires finite numeric economics")
        return Decimal(str(value))

    inventory = {}
    multipliers = {}
    for order, fill in ledger:
        identity = (order.market, order.symbol, order.instrument, order.side)
        if order.side not in {"long", "short"} or order.action not in {"buy", "sell"}:
            raise ValueError("paper ledger has invalid side or action")
        multiplier = Decimal(1)
        if order.market == "crypto_perp":
            rule = policy.perp_instrument_rules.get(order.instrument)
            if rule is None or rule.margin_semantics != "full_notional" or rule.funding_semantics != "excluded":
                raise ValueError("unsupported paper collateral semantics")
            multiplier = Decimal(str(rule.contract_multiplier))
            if not multiplier.is_finite() or multiplier <= 0:
                raise ValueError("paper multiplier must be positive and finite")
        elif order.market != "tw_stock" or order.instrument != "spot":
            raise ValueError("unsupported paper position identity")
        multipliers[identity] = multiplier
        quantity, cost = inventory.get(identity, (Decimal(0), Decimal(0)))
        amount, price = number(fill.fill_quantity), number(fill.fill_price)
        if not amount.is_finite() or amount <= 0 or not price.is_finite() or price <= 0:
            raise ValueError("paper ledger has invalid fill economics")
        if order.action == ("buy" if order.side == "long" else "sell"):
            quantity, cost = quantity + amount, cost + price * amount
        else:
            if amount > quantity or quantity <= 0:
                raise ValueError("paper ledger contains negative inventory")
            cost, quantity = cost - cost / quantity * amount, quantity - amount
        inventory[identity] = quantity, cost
    positions = {position.identity: number(position.quantity) for position in snapshot.positions}
    if len(positions) != len(snapshot.positions) or any(
        not value.is_finite() or value <= 0 for value in positions.values()
    ):
        raise ValueError("paper snapshot has duplicate or invalid inventory")
    if {key: value[0] for key, value in inventory.items() if value[0]} != positions:
        raise ValueError("paper snapshot and inventory ledger disagree")
    nav = number(snapshot.cash)
    if not nav.is_finite():
        raise ValueError("paper cash must be finite")
    for (market, symbol, instrument, side), (quantity, cost) in inventory.items():
        if not quantity:
            continue
        mark = number(prices.get((market, symbol, instrument)))
        if not mark.is_finite() or mark <= 0:
            raise ValueError("materialization requires a positive finite mark")
        multiplier = multipliers[(market, symbol, instrument, side)]
        nav += (mark * quantity if side == "long" else 2 * cost - mark * quantity) * multiplier
    if not nav.is_finite() or nav <= 0 or not math.isfinite(float(nav)):
        raise ValueError("materialization requires positive finite NAV")
    return float(nav)


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


def _position_key(scope, generation, market, symbol, instrument, side):
    return canonical_json([scope, generation, market, symbol, instrument, side])


def _multiplier(policy, market, instrument):
    if market == "tw_stock" and instrument == "spot":
        return Decimal(1)
    rule = policy.perp_instrument_rules.get(instrument) if market == "crypto_perp" else None
    if rule is None or rule.margin_semantics != "full_notional" or rule.funding_semantics != "excluded":
        raise ReconciliationConflictError("missing supported owner collateral semantics")
    value = _number(rule.contract_multiplier)
    if value <= 0:
        raise ReconciliationConflictError("invalid owner contract multiplier")
    return value


def _account_differences(internal, broker, policy):
    differences = []

    def compare(path, left, right, tolerance):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                compare(f"{path}.{key}", left.get(key), right.get(key), tolerance)
        elif left != right:
            numeric = (
                isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
            )
            within = numeric and abs(_number(left) - _number(right)) <= _number(tolerance)
            differences.append({**_difference(path, left, right), "within_tolerance": bool(within)})

    for category, tolerance in (
        ("orders", 0),
        ("fills", policy.fill_tolerance),
        ("positions", policy.position_tolerance),
        ("cash", policy.cash_tolerance),
        ("reservations", 0),
    ):
        compare(category, internal[category], broker[category], tolerance)
    return differences


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

    def _validate_durable_intent(self, order: OrderRecord, *, lock_fills=True) -> None:
        if order.order_origin in PROTECTIVE_ORIGINS:
            try:
                ProtectiveExecutionService(self.session).validate_order(order, lock_fills=lock_fills)
            except ExecutionConflictError as error:
                raise ReconciliationConflictError("durable order intent changed after materialization") from error
            return
        if order.order_origin != "decision":
            raise ReconciliationConflictError("durable order intent changed after materialization")
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
        # Fail fast on corrupted identity without taking a row lock in the
        # opposite order; the authoritative validation repeats under locks.
        probe = self.session.get(OrderRecord, order_id)
        if probe is None:
            raise ReconciliationConflictError("decision order does not exist")
        self._validate_durable_intent(probe, lock_fills=False)
        order = self._locked_order_and_account(order_id)
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
            legacy_required = Decimal("0")
            legacy_source_ids = set()
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
                context = reserved_order.protective_context_json
                if isinstance(context, dict) and context.get("legacy_exception") is True:
                    try:
                        legacy_source_ids.update(uuid.UUID(value) for value in context["source_holding_ids"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ReconciliationConflictError("legacy protective source provenance is invalid") from error
                    legacy_required += outstanding
                else:
                    required += outstanding

            if legacy_required:
                legacy_holdings = self.session.scalars(
                    select(PortfolioHoldingRecord)
                    .where(PortfolioHoldingRecord.id.in_(legacy_source_ids))
                    .with_for_update()
                ).all()
                available = sum(
                    (_number(row.shares) for row in legacy_holdings if row.shares is not None),
                    Decimal("0"),
                )
                if len(legacy_holdings) != len(legacy_source_ids) or available < legacy_required:
                    raise ReconciliationConflictError("legacy close reservation demand exceeds open holdings")

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

    def reconcile_account(self, account_id, snapshot, *, now=None, prices=None):
        """Compare independent ledgers and flush one immutable account result."""
        probe = self.session.get(PaperBrokerAccount, account_id)
        if probe is None:
            raise ReconciliationConflictError("paper account does not exist")
        scope, generation = probe.account_scope, probe.account_generation
        execution = DecisionExecutionService(self.session)
        execution._advisory_lock(scope, generation)
        account = self.session.scalar(
            select(PaperBrokerAccount)
            .where(PaperBrokerAccount.id == account_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        as_of = timestamp(now if now is not None else datetime.now(UTC), "now")
        reasons, invariants = [], []
        decisions = self.session.scalars(
            select(DecisionRecord)
            .where(DecisionRecord.account_scope == scope)
            .order_by(DecisionRecord.created_at.desc(), DecisionRecord.id.desc())
            .with_for_update()
        ).all()
        owner, policy, frozen = None, None, None
        for decision in decisions:
            version = self.session.get(StrategyVersion, decision.strategy_version_id)
            candidate = (
                None
                if version is None or not isinstance(version.policy_json, dict)
                else version.policy_json.get("reconciliation")
            )
            if (
                candidate is None
                or not isinstance(candidate, dict)
                or candidate.get("account_generation") == generation
            ):
                owner, frozen = decision, candidate
                try:
                    _, full_policy, _ = execution._policy_and_intents(decision)
                    policy = full_policy.reconciliation
                    if policy is None:
                        raise ExecutionConflictError("missing owner reconciliation policy")
                except (ExecutionConflictError, ValueError) as error:
                    reasons.append(f"owner policy unresolved: {error}")
                break
        if owner is None:
            reasons.append("owner policy is missing for account generation")
        policy_sha = owner.policy_sha256 if owner is not None else "0" * 64
        broker_orders = self.session.scalars(
            select(PaperBrokerOrder)
            .where(PaperBrokerOrder.account_scope == scope, PaperBrokerOrder.account_generation == generation)
            .order_by(PaperBrokerOrder.state_version, PaperBrokerOrder.id)
            .with_for_update()
        ).all()
        all_broker_orders = broker_orders
        try:
            from poseidon.broker.paper_adapter import _validated_attributions, _validated_baselines

            control_ids = {
                row.id
                for row in [
                    *_validated_baselines(self.session, scope, generation),
                    *_validated_attributions(self.session, scope, generation),
                ]
            }
        except BrokerCapabilityError as error:
            control_ids = set()
            reasons.append(f"broker control ledger unresolved: {error}")
        broker_orders = [row for row in all_broker_orders if row.id not in control_ids]
        broker_fills = self.session.scalars(
            select(PaperBrokerFill)
            .where(PaperBrokerFill.account_scope == scope, PaperBrokerFill.account_generation == generation)
            .order_by(PaperBrokerFill.state_version, PaperBrokerFill.id)
            .with_for_update()
        ).all()
        movements = self.session.scalars(
            select(PaperCashMovement)
            .where(PaperCashMovement.account_scope == scope, PaperCashMovement.account_generation == generation)
            .order_by(PaperCashMovement.state_version, PaperCashMovement.id)
            .with_for_update()
        ).all()
        orders = self.session.scalars(
            select(OrderRecord)
            .where(OrderRecord.account_scope == scope, OrderRecord.account_generation == generation)
            .order_by(OrderRecord.id)
            .with_for_update()
        ).all()
        fills = self.session.scalars(
            select(OrderFillRecord)
            .join(OrderRecord)
            .where(OrderRecord.account_scope == scope, OrderRecord.account_generation == generation)
            .order_by(OrderFillRecord.fill_time, OrderFillRecord.id)
            .with_for_update()
        ).all()
        lots = self.session.scalars(
            select(PositionLot)
            .where(PositionLot.account_scope == scope, PositionLot.account_generation == generation)
            .order_by(PositionLot.opened_at, PositionLot.id)
            .with_for_update()
        ).all()
        broker = {"orders": {}, "fills": {}, "positions": {}, "cash": {}, "reservations": {}}
        internal = {"orders": {}, "fills": {}, "positions": {}, "cash": {}, "reservations": {}}
        broker_by_id = {row.id: row for row in broker_orders}
        internal_by_id = {row.id: row for row in orders}

        def order_value(row):
            if any(
                not getattr(row, field)
                for field in ("client_order_ref", "broker_order_id", "market", "symbol", "instrument", "side")
            ):
                raise ReconciliationConflictError("order canonical identity is incomplete")
            quantity = _number(row.quantity)
            if quantity <= 0:
                raise ReconciliationConflictError("order quantity must be positive")
            return {
                "broker_order_id": row.broker_order_id,
                "market": row.market,
                "symbol": row.symbol,
                "instrument": row.instrument,
                "side": row.side,
                "action": row.action,
                "order_type": row.order_type,
                "quantity": float(quantity),
                "price": None if row.price is None else float(_number(row.price)),
                "status": _normalized_status(row.status),
            }

        def fill_value(row, order):
            if not row.broker_fill_id or _number(row.fill_quantity) <= 0 or _number(row.fill_price) <= 0:
                raise ReconciliationConflictError("fill canonical identity/economics are incomplete")
            return {
                "broker_order_id": order.broker_order_id,
                "market": order.market,
                "symbol": order.symbol,
                "instrument": order.instrument,
                "side": order.side,
                "quantity": float(_number(row.fill_quantity)),
                "price": float(_number(row.fill_price)),
                "fill_time": iso_time(_utc(row.fill_time), "fill_time"),
            }

        for rows, output in ((broker_orders, broker), (orders, internal)):
            for row in rows:
                try:
                    if row.client_order_ref in output["orders"]:
                        raise ReconciliationConflictError("duplicate canonical client reference")
                    output["orders"][row.client_order_ref] = order_value(row)
                except ReconciliationConflictError as error:
                    reasons.append(f"{('broker' if output is broker else 'internal')} order unresolved: {error}")
        for rows, output, order_map in ((broker_fills, broker, broker_by_id), (fills, internal, internal_by_id)):
            for row in rows:
                order = order_map.get(row.paper_broker_order_id if output is broker else row.order_id)
                try:
                    if order is None:
                        raise ReconciliationConflictError("fill order identity is missing")
                    key = canonical_json([order.client_order_ref, row.broker_fill_id])
                    if key in output["fills"]:
                        raise ReconciliationConflictError("duplicate canonical fill identity")
                    output["fills"][key] = fill_value(row, order)
                    if output is internal and row.projection_status != "applied":
                        reasons.append(f"internal fill {row.id} projection is {row.projection_status}")
                except ReconciliationConflictError as error:
                    reasons.append(f"fill unresolved: {error}")
        for order in orders:
            order_fills = [fill for fill in fills if fill.order_id == order.id]
            quantity = sum((_number(fill.fill_quantity) for fill in order_fills), Decimal(0))
            if quantity and order.client_order_ref in internal["orders"]:
                internal["orders"][order.client_order_ref]["price"] = float(
                    sum((_number(fill.fill_price) * _number(fill.fill_quantity) for fill in order_fills), Decimal(0))
                    / quantity
                )
        try:
            cash = {account.currency: _number(account.opening_cash)}
            for movement in movements:
                cash[movement.currency] = cash.get(movement.currency, Decimal(0)) + _number(movement.amount)
            broker["cash"] = {key: float(value) for key, value in sorted(cash.items())}
            if any(
                row.state_version > account.state_version for row in [*all_broker_orders, *broker_fills, *movements]
            ):
                reasons.append("broker ledger exceeds account watermark")
            if not isinstance(snapshot, BrokerAccountSnapshot):
                raise ReconciliationConflictError("adapter account snapshot is missing required fields")
            if (snapshot.account_scope, snapshot.account_generation, snapshot.currency, snapshot.state_version) != (
                scope,
                generation,
                account.currency,
                account.state_version,
            ):
                raise ReconciliationConflictError("broker snapshot identity/watermark changed")
            if _utc(snapshot.as_of) > as_of or (
                policy is not None
                and (as_of - _utc(snapshot.as_of)).total_seconds() > policy.max_reconciliation_age_seconds
            ):
                raise ReconciliationConflictError("broker snapshot is future or stale")
            if broker["cash"] != {snapshot.currency: float(_number(snapshot.cash))}:
                raise ReconciliationConflictError("broker snapshot and complete cash ledger disagree")
            for position in snapshot.positions:
                key = _position_key(
                    scope, generation, position.market, position.symbol, position.instrument, position.side
                )
                if key in broker["positions"] or _number(position.quantity) <= 0:
                    raise ReconciliationConflictError("broker snapshot inventory is invalid")
                broker["positions"][key] = float(_number(position.quantity))
        except (ReconciliationConflictError, TypeError, AttributeError) as error:
            reasons.append(f"broker account unresolved: {error}")
        opening = None
        currency = None
        if isinstance(frozen, dict):
            try:
                opening, currency = _number(frozen.get("opening_cash")), frozen.get("currency")
                if opening < 0 or not isinstance(currency, str) or not currency:
                    raise ReconciliationConflictError("owner opening terms are invalid")
                internal["cash"] = {currency: float(opening)}
            except ReconciliationConflictError as error:
                reasons.append(f"internal opening terms unresolved: {error}")
        for lot in lots:
            try:
                amount = _number(lot.open_quantity)
                if amount < 0:
                    raise ReconciliationConflictError("negative lot inventory")
                key = _position_key(scope, generation, lot.market, lot.symbol, lot.instrument, lot.side)
                if amount:
                    internal["positions"][key] = internal["positions"].get(key, 0.0) + float(amount)
                from poseidon.positions.lots import FillProjectionService

                FillProjectionService(self.session)._validate_lot(
                    lot,
                    {
                        field: getattr(lot, field)
                        for field in ("account_scope", "account_generation", "market", "symbol", "instrument", "side")
                    },
                    account,
                )
            except ReconciliationConflictError as error:
                invariants.append(
                    {
                        **_difference(f"projections.lot.{lot.id}", str(error), "valid applied provenance"),
                        "within_tolerance": False,
                    }
                )
        from poseidon.positions.lots import FillProjectionService

        for fill in fills:
            if fill.projection_status == "applied":
                try:
                    FillProjectionService(self.session).apply(fill.id)
                except ReconciliationConflictError as error:
                    invariants.append(
                        {
                            **_difference(
                                f"projections.fill.{fill.id}", str(error), "valid applied provenance/reservations"
                            ),
                            "within_tolerance": False,
                        }
                    )
        if policy is not None:
            try:
                internal_cash = opening
                short_groups = {}
                legacy_short_consumption = {}
                for fill in fills:
                    order = internal_by_id[fill.order_id]
                    identity = order.market, order.symbol, order.instrument, order.side
                    amount, price = _number(fill.fill_quantity), _number(fill.fill_price)
                    if amount <= 0 or price <= 0:
                        raise ReconciliationConflictError("internal cash fill economics are invalid")
                    multiplier = _multiplier(policy, order.market, order.instrument)
                    is_open = order.action == ("buy" if order.side == "long" else "sell")
                    if order.side == "long":
                        internal_cash += (-1 if is_open else 1) * price * amount * multiplier
                    else:
                        context = order.protective_context_json
                        legacy_close = (
                            not is_open and isinstance(context, dict) and context.get("legacy_exception") is True
                        )
                        if legacy_close:
                            self._validate_durable_intent(order)
                            remaining = amount
                            entry_notional = Decimal("0")
                            for holding_id in context["source_holding_ids"]:
                                key = order.id, holding_id
                                planned = _number(context["source_holding_quantities"][holding_id])
                                used = legacy_short_consumption.get(key, Decimal("0"))
                                available = planned - used
                                consumed = min(available, remaining)
                                legacy_short_consumption[key] = used + consumed
                                entry_notional += consumed * _number(
                                    context["source_holding_risk"][holding_id]["entry_price"]
                                )
                                remaining -= consumed
                            if remaining:
                                raise ReconciliationConflictError(
                                    "internal legacy short fills exceed frozen attribution"
                                )
                            internal_cash += (Decimal("2") * entry_notional - price * amount) * multiplier
                            continue
                        short_groups.setdefault((identity, _utc(fill.fill_time)), []).append((is_open, amount, price))
                inventory = {}
                for (identity, _), group in sorted(short_groups.items()):
                    if len({row[0] for row in group}) != 1:
                        raise ReconciliationConflictError("ambiguous same-time opening/closing short fills")
                    amount = sum((row[1] for row in group), Decimal(0))
                    notional = sum((row[1] * row[2] for row in group), Decimal(0))
                    quantity, cost = inventory.get(identity, (Decimal(0), Decimal(0)))
                    multiplier = _multiplier(policy, identity[0], identity[2])
                    if group[0][0]:
                        internal_cash -= notional * multiplier
                        quantity, cost = quantity + amount, cost + notional
                    else:
                        if quantity <= 0 or amount > quantity:
                            raise ReconciliationConflictError("internal short cash ledger contains over-close")
                        internal_cash += (2 * cost / quantity * amount - notional) * multiplier
                        cost, quantity = cost - cost / quantity * amount, quantity - amount
                    inventory[identity] = quantity, cost
                internal["cash"] = {currency: float(internal_cash)}
                for order in orders:
                    if order.reservation_status not in {"reserved", "released"}:
                        raise ReconciliationConflictError("internal reservation state is unknown")
                    if order.reservation_status == "reserved":
                        quantity, cash = execution._outstanding_reservation(order, currency)
                        if quantity:
                            internal["reservations"][order.client_order_ref] = {
                                "quantity": float(quantity),
                                "cash": float(cash),
                            }
                for order in broker_orders:
                    if _normalized_status(order.status) in {"submitted", "partially_filled"}:
                        amount = _number(order.quantity) - sum(
                            (
                                _number(row.fill_quantity)
                                for row in broker_fills
                                if row.paper_broker_order_id == order.id
                            ),
                            Decimal(0),
                        )
                        if amount < 0:
                            raise ReconciliationConflictError("broker fills exceed accepted quantity")
                        if amount:
                            cash = (
                                amount * _number(order.price) * _multiplier(policy, order.market, order.instrument)
                                if order.action == ("buy" if order.side == "long" else "sell")
                                else Decimal(0)
                            )
                            broker["reservations"][order.client_order_ref] = {
                                "quantity": float(amount),
                                "cash": float(cash),
                            }
            except (ReconciliationConflictError, ExecutionConflictError, KeyError, TypeError) as error:
                reasons.append(f"internal cash/reservations unresolved: {error}")
        differences = invariants + ([] if policy is None else _account_differences(internal, broker, policy))
        economic_mismatch = any(not row["within_tolerance"] for row in differences)
        nav = None
        if policy is not None and isinstance(snapshot, BrokerAccountSnapshot):
            try:
                ledger = [(broker_by_id[row.paper_broker_order_id], row) for row in broker_fills]
                nav = (
                    paper_liquidation_nav(snapshot, policy, prices or {}, ledger)
                    if snapshot.positions or (ledger and not control_ids)
                    else float(_number(snapshot.cash))
                )
            except (ValueError, KeyError, ReconciliationConflictError) as error:
                reasons.append(f"valuation unresolved: {error}")
        from poseidon.strategies.portfolio.position_tracker import PositionTracker

        projection = {"status": "not_applied", "reasons": [], "rows": []}
        if not economic_mismatch and not reasons:
            projection = PositionTracker.project_lots(self.session, account_id, prices, nav, now=as_of)
            reasons.extend(projection["reasons"])
        internal["projection"] = {
            "prices": [
                [
                    *key,
                    value
                    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                    else {"invalid": repr(value)},
                ]
                for key, value in sorted((prices or {}).items())
            ],
            "account_nav": nav,
            "result": projection,
        }
        status = "unresolved" if reasons else ("mismatch" if economic_mismatch else "matched")
        # A known economic disagreement remains mismatch; pending projections are recoverable unresolved.
        if economic_mismatch and not any(
            any(
                kind in reason
                for kind in (
                    "projection is",
                    "ambiguous",
                    "broker account unresolved",
                    "order unresolved",
                    "fill unresolved",
                    "owner policy unresolved",
                )
            )
            for reason in reasons
        ):
            status = "mismatch" if policy is not None else "unresolved"
        broker_hash = content_sha256(broker)
        watermark = internal_state_watermark(self.session, scope, generation)
        identity = {
            "account_scope": scope,
            "account_generation": generation,
            "as_of": as_of,
            "broker_state_watermark": f"broker:{account.state_version}",
            "internal_state_watermark": watermark,
            "broker_snapshot_sha256": broker_hash,
            "policy_sha256": policy_sha,
        }
        existing = self.session.scalar(select(AccountReconciliation).filter_by(**identity))
        difference = {
            "differences": differences,
            "unresolved": sorted(set(reasons)),
            "owner_policy": frozen,
            "max_reconciliation_age_seconds": None if policy is None else policy.max_reconciliation_age_seconds,
        }
        if existing is not None:
            if (
                existing.internal_snapshot_json != internal
                or existing.difference_json != difference
                or existing.status != status
            ):
                raise ReconciliationConflictError("reconciliation replay changed snapshot/projection content")
            return {"reconciliation_id": str(existing.id), "status": existing.status}
        row = AccountReconciliation(
            id=uuid.uuid4(),
            **identity,
            broker_snapshot_json=broker,
            internal_snapshot_json=internal,
            difference_json=difference,
            status=status,
        )
        self.session.add(row)
        if status == "matched":
            PositionTracker.project_lots(self.session, account_id, prices, nav, now=as_of, apply=True)
            for decision in decisions:
                if decision.status != "execution_claimed":
                    continue
                try:
                    _, _, intents = execution._policy_and_intents(decision)
                    canonical = execution._replay(decision, intents, generation)
                except ExecutionConflictError:
                    continue
                canonical_ids = (
                    set() if canonical is None else {uuid.UUID(order_id) for order_id in canonical["order_ids"]}
                )
                decision_orders = [order for order in orders if order.id in canonical_ids]
                if (
                    canonical is None
                    or len(canonical_ids) != len(intents)
                    or len(decision_orders) != len(intents)
                    or any(
                        order.status not in {"filled", "rejected", "cancelled"}
                        or order.reservation_status != "released"
                        for order in decision_orders
                    )
                    or any(
                        fill.projection_status != "applied"
                        for fill in fills
                        if fill.order_id in {order.id for order in decision_orders}
                    )
                ):
                    continue
                revision = decision.revision
                decision.status, decision.revision = "executed", revision + 1
                self.session.add(
                    DecisionEvent(
                        decision_id=decision.id,
                        event_type="executed",
                        actor_id="system:account-reconciliation",
                        expected_revision=revision,
                        payload_json={"reconciliation_id": str(row.id)},
                    )
                )
        self.session.flush()
        return {"reconciliation_id": str(row.id), "status": status}

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


def reconcile_account(session_factory, account_id, adapter, *, now=None, prices=None):
    """Perform adapter I/O with no open DB session, then commit a short comparison."""
    persisted_id = uuid.UUID(str(account_id))
    with session_factory() as session:
        account = session.get(PaperBrokerAccount, persisted_id)
        if account is None:
            raise ReconciliationConflictError("paper account does not exist")
        scope, generation = account.account_scope, account.account_generation
    snapshot = adapter.query_account_snapshot(scope, generation)
    with session_factory() as session, session.begin():
        return ReconciliationService(session).reconcile_account(persisted_id, snapshot, now=now, prices=prices)


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
