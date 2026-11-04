"""Materialize approved decisions as durable broker-independent order intents."""

import json
import uuid
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import Float, func, select

from poseidon.core.authz import AuthPrincipal
from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.errors import ValidationError
from poseidon.core.locks import lock_account
from poseidon.core.timeutil import as_utc, stored_iso_time, stored_timestamp, timestamp
from poseidon.decision_loop.decisions import (
    DecisionService,
    _validate_order_intents,
    _validate_selection,
    require_active_pointer,
)
from poseidon.decision_loop.evaluation import verify_complete_run
from poseidon.decision_loop.policy import DecisionPolicy
from poseidon.decision_loop.protective_execution import ProtectiveExecutionService
from poseidon.decision_loop.reservations import (
    ACTIVE_RESERVATION_STATUSES,
    PROTECTIVE_ORIGINS,
    ExecutionConflictError,
    _finite_positive,
    outstanding_reservation,
)
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.fill_allocation import FillAllocation
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion


def _floor_step(value, step):
    decimal_value = Decimal(str(value))
    decimal_step = Decimal(str(step))
    return (decimal_value / decimal_step).to_integral_value(rounding=ROUND_FLOOR) * decimal_step


def internal_state_watermark(session, account_scope, account_generation):
    """Fingerprint complete internal economic state, including timestamp-neutral corruption."""
    orders = session.scalars(
        select(OrderRecord)
        .where(OrderRecord.account_scope == account_scope, OrderRecord.account_generation == account_generation)
        .order_by(OrderRecord.id)
    ).all()
    order_ids = [row.id for row in orders]
    fills = session.scalars(
        select(OrderFillRecord).where(OrderFillRecord.order_id.in_(order_ids)).order_by(OrderFillRecord.id)
    ).all()
    lots = session.scalars(
        select(PositionLot)
        .where(PositionLot.account_scope == account_scope, PositionLot.account_generation == account_generation)
        .order_by(PositionLot.id)
    ).all()
    allocations = session.scalars(
        select(FillAllocation)
        .where(FillAllocation.position_lot_id.in_([row.id for row in lots]))
        .order_by(FillAllocation.id)
    ).all()
    if not orders and not fills and not lots and not allocations:
        return "internal:0"

    def normalized(row):
        values = {}
        for column in row.__table__.columns:
            value = getattr(row, column.key)
            if isinstance(value, datetime):
                value = stored_iso_time(value, column.key)
            elif isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(column.type, Float) and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
            values[column.key] = value
        return values

    return "internal:" + content_sha256(
        [[normalized(row) for row in rows] for rows in (orders, fills, lots, allocations)]
    )


class DecisionExecutionService:
    """Validate and flush one complete intent set; the caller owns the transaction."""

    def __init__(self, session):
        self.session = session

    def _response(self, decision, orders):
        return {
            "decision_id": str(decision.id),
            "status": "pending_submit",
            "order_ids": [str(order.id) for order in orders],
            "client_order_refs": [order.client_order_ref for order in orders],
        }

    def replay(self, decision, intents, account_generation):
        orders = self.session.scalars(
            select(OrderRecord)
            .where(
                OrderRecord.decision_id == decision.id,
                OrderRecord.execution_key == decision.execution_key,
            )
            .order_by(OrderRecord.client_order_ref)
        ).all()
        if not orders:
            return None
        expected_refs = [f"DL-{decision.execution_key.hex}-{index:04d}" for index in range(len(intents))]
        if len(orders) != len(intents) or [order.client_order_ref for order in orders] != expected_refs:
            raise ExecutionConflictError("stored order intent set is incomplete or colliding")
        for order, intent in zip(orders, intents, strict=True):
            stored = order.intent_json
            frozen = intent.model_dump(mode="json")
            if (
                not isinstance(stored, dict)
                or stored.get("frozen_intent") != frozen
                or order.intent_sha256 != content_sha256(stored)
            ):
                raise ExecutionConflictError("stored order intent content changed")
            economics = stored.get("economics")
            if not isinstance(economics, dict):
                raise ExecutionConflictError("stored order executable content changed")
            try:
                stored_price = _finite_positive(economics.get("price"), "stored price")
                stored_quantity = _finite_positive(
                    economics.get("materialized_quantity"),
                    "stored materialized quantity",
                )
            except ExecutionConflictError as error:
                raise ExecutionConflictError("stored order executable content changed") from error
            broker_action = "buy" if (intent.side == "long") == (intent.action in {"enter", "add"}) else "sell"
            if (
                order.decision_id != decision.id
                or order.execution_key != decision.execution_key
                or order.account_scope != decision.account_scope
                or order.account_generation != account_generation
                or order.market != intent.market
                or order.symbol != intent.symbol
                or order.instrument != intent.instrument
                or order.side != intent.side
                or order.action != broker_action
                or order.order_type != intent.order_type
                or Decimal(str(order.target_weight)) != Decimal(str(intent.target_weight))
                or order.price is None
                or Decimal(str(order.price)) != Decimal(str(stored_price))
                or Decimal(str(order.quantity)) != Decimal(str(stored_quantity))
                or order.broker_mode != "paper"
                or order.order_origin != "decision"
                or order.signal_id is not None
            ):
                raise ExecutionConflictError("stored order executable content changed")
        return self._response(decision, orders)

    def policy_and_intents(self, decision):
        version = self.session.get(StrategyVersion, decision.strategy_version_id)
        if version is None:
            raise ExecutionConflictError("decision strategy version does not exist")
        try:
            version.verify_content()
            policy = DecisionPolicy.model_validate(version.policy_json)
            _, _, _, snapshots = verify_complete_run(self.session, decision.evaluation_run_id)
            final_action = decision.final_json.get("final_action")
            selected = _validate_selection(snapshots, decision.final_json.get("selected_evaluation_ids"), final_action)
            intents = _validate_order_intents(selected, decision.final_json, policy)
        except (TypeError, ValueError, ValidationError, PydanticValidationError) as error:
            raise ExecutionConflictError(f"frozen order intent is invalid: {error}") from error
        if not intents:
            raise ExecutionConflictError("claimed decision requires executable order intents")
        if content_sha256(version.policy_json) != decision.policy_sha256:
            raise ExecutionConflictError("decision policy changed")
        if policy.account_scope != decision.account_scope or policy.reconciliation is None:
            raise ExecutionConflictError("owner account authorization and reconciliation policy are required")
        intents = sorted(
            intents,
            key=lambda intent: (
                decision.account_scope,
                policy.reconciliation.account_generation,
                intent.market,
                intent.symbol,
                intent.instrument,
                intent.side,
                str(intent.evaluation_snapshot_id),
            ),
        )
        return version, policy, intents

    def _lock_account(self, decision, policy):
        reconciliation = policy.reconciliation
        lock_account(self.session, decision.account_scope, reconciliation.account_generation)
        account = (
            self.session.query(PaperBrokerAccount)
            .filter_by(
                account_scope=decision.account_scope,
                account_generation=reconciliation.account_generation,
            )
            .with_for_update()
            .populate_existing()
            .one_or_none()
        )
        if account is None:
            raise ExecutionConflictError("approved paper account generation does not exist")
        if account.currency != reconciliation.currency or Decimal(str(account.opening_cash)) != Decimal(
            str(reconciliation.opening_cash)
        ):
            raise ExecutionConflictError("paper account does not match owner-frozen opening terms")
        return account

    def preflight_approved(self, decision, *, principal, market, account_generation, adapter, now=None):
        """Validate the ordinary cutover gate before a caller claims the decision."""
        from poseidon.broker.base import BrokerCapabilities

        principal.require_role("decision-worker")
        principal.require_account_scope(decision.account_scope)
        require_active_pointer(
            self.session,
            decision,
            lock=True,
            error_type=ExecutionConflictError,
        )
        capabilities = adapter.capabilities
        if not isinstance(capabilities, BrokerCapabilities) or not capabilities.supports_reconciliation:
            raise ExecutionConflictError("paper adapter lacks full reconciliation capability")
        _, policy, intents = self.policy_and_intents(decision)
        if policy.market != market or policy.reconciliation.account_generation != account_generation:
            raise ExecutionConflictError("approved execution identity changed")
        if any(intent.market != market for intent in intents):
            raise ExecutionConflictError("approved intent market changed")
        account = self._lock_account(decision, policy)
        decision = self.session.scalar(
            select(DecisionRecord)
            .where(DecisionRecord.id == decision.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        require_active_pointer(
            self.session,
            decision,
            lock=False,
            error_type=ExecutionConflictError,
        )
        current_time = now if now is not None else datetime.now(UTC)
        DecisionService(self.session).validate_execution(decision.id, decision.revision, current_time)
        self._require_reconciled(decision, policy.reconciliation, account, current_time)
        return policy

    def _risk_failures(self, decision, policy, intents):
        failures = decision.risk_snapshot_json.get("hard_failures")
        if not isinstance(failures, list):
            raise ExecutionConflictError("current hard risk snapshot is invalid")
        current = list(failures)
        for intent in intents:
            if intent.target_weight > policy.hard_limits.max_position_weight:
                current.append(f"{intent.symbol}:position_limit")
        return current

    def _record_risk_block(self, decision, failures, principal):
        expected_revision = decision.revision
        decision.status = "risk_blocked"
        decision.revision += 1
        response = {"decision_id": str(decision.id), "status": "risk_blocked", "order_ids": [], "failures": failures}
        self.session.add(
            DecisionEvent(
                decision_id=decision.id,
                event_type="risk_blocked",
                actor_id=principal.actor_id,
                expected_revision=expected_revision,
                payload_json={"response": response},
            )
        )
        self.session.flush()
        return response

    def _require_reconciled(self, decision, reconciliation_policy, account, now):
        latest = self.session.scalar(
            select(AccountReconciliation)
            .where(
                AccountReconciliation.account_scope == decision.account_scope,
                AccountReconciliation.account_generation == reconciliation_policy.account_generation,
            )
            .order_by(
                AccountReconciliation.as_of.desc(),
                AccountReconciliation.created_at.desc(),
                AccountReconciliation.id.desc(),
            )
            .limit(1)
        )
        if latest is None or latest.status != "matched":
            raise ExecutionConflictError("account reconciliation is unresolved")
        if latest.policy_sha256 != decision.policy_sha256:
            raise ExecutionConflictError("account reconciliation policy changed")
        as_of = as_utc(latest.as_of)
        if as_of > now or (now - as_of).total_seconds() > reconciliation_policy.max_reconciliation_age_seconds:
            raise ExecutionConflictError("account reconciliation is stale")
        if latest.broker_state_watermark != f"broker:{account.state_version}":
            raise ExecutionConflictError("account reconciliation broker watermark changed")
        if latest.internal_state_watermark != internal_state_watermark(
            self.session,
            decision.account_scope,
            reconciliation_policy.account_generation,
        ):
            raise ExecutionConflictError("account reconciliation internal watermark changed")
        account_identity = (
            decision.account_scope,
            reconciliation_policy.account_generation,
        )
        order_fill_state = (
            select(OrderFillRecord.id)
            .join(OrderRecord, OrderRecord.id == OrderFillRecord.order_id)
            .where(
                OrderRecord.account_scope == account_identity[0],
                OrderRecord.account_generation == account_identity[1],
                (OrderFillRecord.created_at > latest.as_of)
                | (OrderFillRecord.projection_status == "projection_pending"),
            )
        )
        newer_state = (
            select(OrderRecord.id).where(
                OrderRecord.account_scope == account_identity[0],
                OrderRecord.account_generation == account_identity[1],
                OrderRecord.updated_at > latest.as_of,
            ),
            order_fill_state,
            select(PositionLot.id).where(
                PositionLot.account_scope == account_identity[0],
                PositionLot.account_generation == account_identity[1],
                PositionLot.updated_at > latest.as_of,
            ),
            select(PaperBrokerOrder.id).where(
                PaperBrokerOrder.account_scope == account_identity[0],
                PaperBrokerOrder.account_generation == account_identity[1],
                PaperBrokerOrder.updated_at > latest.as_of,
            ),
            select(PaperBrokerFill.id).where(
                PaperBrokerFill.account_scope == account_identity[0],
                PaperBrokerFill.account_generation == account_identity[1],
                PaperBrokerFill.created_at > latest.as_of,
            ),
            select(PaperCashMovement.id).where(
                PaperCashMovement.account_scope == account_identity[0],
                PaperCashMovement.account_generation == account_identity[1],
                PaperCashMovement.created_at > latest.as_of,
            ),
        )
        if as_utc(account.updated_at) > as_of or any(
            self.session.scalar(statement.limit(1)) is not None for statement in newer_state
        ):
            raise ExecutionConflictError("account reconciliation predates outstanding state")

    def pending_orders(self, decision, account_generation):
        return self.session.scalars(
            select(OrderRecord).where(
                OrderRecord.account_scope == decision.account_scope,
                OrderRecord.account_generation == account_generation,
                OrderRecord.reservation_status == "reserved",
                OrderRecord.status.in_(ACTIVE_RESERVATION_STATUSES),
            )
        ).all()

    def _position_lots(self, decision, generation, intent):
        return self.session.scalars(
            select(PositionLot)
            .where(
                PositionLot.account_scope == decision.account_scope,
                PositionLot.account_generation == generation,
                PositionLot.market == intent.market,
                PositionLot.symbol == intent.symbol,
                PositionLot.instrument == intent.instrument,
                PositionLot.side == intent.side,
            )
            .order_by(PositionLot.opened_at, PositionLot.id)
            .with_for_update()
        ).all()

    @staticmethod
    def _identity(order):
        return (order.market, order.symbol, order.instrument, order.side)

    def _projected_risk_failures(self, decision, reconciliation, pending, specs, prices, nav, policy, reservations):
        quantities = {}
        lots = self.session.scalars(
            select(PositionLot).where(
                PositionLot.account_scope == decision.account_scope,
                PositionLot.account_generation == reconciliation.account_generation,
            )
        ).all()
        for lot in lots:
            identity = (lot.market, lot.symbol, lot.instrument, lot.side)
            quantities[identity] = quantities.get(identity, Decimal("0")) + Decimal(str(lot.open_quantity))
        for order in pending:
            identity = self._identity(order)
            action = (order.intent_json or {}).get("frozen_intent", {}).get("action")
            if action not in {"enter", "add", "reduce", "exit"}:
                raise ExecutionConflictError("pending reservation has invalid frozen action")
            direction = Decimal("1") if action in {"enter", "add"} else Decimal("-1")
            quantities[identity] = quantities.get(identity, Decimal("0")) + direction * reservations[order.id][0]
        for _, intent, quantity, _, _, _, _ in specs:
            identity = (intent.market, intent.symbol, intent.instrument, intent.side)
            direction = Decimal("1") if intent.action in {"enter", "add"} else Decimal("-1")
            quantities[identity] = quantities.get(identity, Decimal("0")) + direction * quantity

        gross_notional = Decimal("0")
        failures = []
        for (market, symbol, instrument, _side), quantity in quantities.items():
            if quantity <= 0:
                continue
            price_identity = (market, symbol, instrument)
            if price_identity not in prices:
                raise ExecutionConflictError(f"missing current price for {price_identity}")
            price = _finite_positive(prices[price_identity], f"price for {price_identity}")
            multiplier = Decimal("1")
            if market == "crypto_perp":
                rule = reconciliation.perp_instrument_rules.get(instrument)
                if rule is None:
                    raise ExecutionConflictError("owner-frozen perp sizing rules are required")
                multiplier = Decimal(str(rule.contract_multiplier))
            notional = quantity * Decimal(str(price)) * multiplier
            weight = notional / Decimal(str(nav))
            if weight > Decimal(str(policy.hard_limits.max_position_weight)):
                failures.append(f"{symbol}:position_limit")
            gross_notional += notional
        if gross_notional / Decimal(str(nav)) > Decimal(str(policy.hard_limits.max_gross_exposure)):
            failures.append("max_gross_exposure")
        return failures

    def materialize(
        self,
        decision_id,
        *,
        principal: AuthPrincipal,
        account_nav,
        prices,
        now=None,
        expected_account_state_version=None,
    ):
        principal.require_role("decision-worker")
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        nav = _finite_positive(account_nav, "account_nav")
        nav_decimal = Decimal(str(nav))
        decision = self.session.get(DecisionRecord, decision_id)
        if decision is None:
            raise ExecutionConflictError("decision does not exist")
        principal.require_account_scope(decision.account_scope)
        require_active_pointer(
            self.session,
            decision,
            lock=True,
            error_type=ExecutionConflictError,
        )
        _, policy, intents = self.policy_and_intents(decision)
        reconciliation = policy.reconciliation
        account = self._lock_account(decision, policy)
        if expected_account_state_version is not None and account.state_version != expected_account_state_version:
            raise ValueError("paper account changed during materialization")
        decision = (
            self.session.query(DecisionRecord).filter_by(id=decision.id).with_for_update().populate_existing().one()
        )
        require_active_pointer(
            self.session,
            decision,
            lock=False,
            error_type=ExecutionConflictError,
        )
        _, locked_policy, locked_intents = self.policy_and_intents(decision)
        if locked_policy.reconciliation.account_generation != reconciliation.account_generation:
            raise ExecutionConflictError("owner account generation changed while locking")
        policy, intents, reconciliation = locked_policy, locked_intents, locked_policy.reconciliation

        replay = self.replay(decision, intents, reconciliation.account_generation)
        if replay is not None:
            return replay
        if decision.status == "risk_blocked":
            return {"decision_id": str(decision.id), "status": "risk_blocked", "order_ids": []}
        if decision.status != "execution_claimed" or decision.execution_key is None:
            raise ExecutionConflictError("decision is not execution_claimed")
        if current_time >= stored_timestamp(decision.valid_until, "valid_until"):
            raise ExecutionConflictError("decision has expired")

        failures = self._risk_failures(decision, policy, intents)
        if failures:
            return self._record_risk_block(decision, failures, principal)
        if any(intent.action in {"enter", "add"} for intent in intents):
            self._require_reconciled(decision, reconciliation, account, current_time)

        pending = self.pending_orders(decision, reconciliation.account_generation)
        reservations = {
            order.id: outstanding_reservation(self.session, order, reconciliation.currency) for order in pending
        }
        lot_pending = []
        protective = ProtectiveExecutionService(self.session)
        for order in pending:
            if order.order_origin in PROTECTIVE_ORIGINS:
                protective.validate_order(order, lock_fills=False)
            context = order.protective_context_json
            if (
                order.decision_id is None
                and order.order_origin in PROTECTIVE_ORIGINS
                and isinstance(context, dict)
                and context.get("legacy_exception") is True
            ):
                continue
            lot_pending.append(order)
        cash_movements = self.session.scalar(
            select(func.coalesce(func.sum(PaperCashMovement.amount), 0.0)).where(
                PaperCashMovement.account_scope == decision.account_scope,
                PaperCashMovement.account_generation == reconciliation.account_generation,
                PaperCashMovement.currency == reconciliation.currency,
            )
        )
        reserved_cash = sum(
            (reservations[order.id][1] for order in pending),
            start=Decimal("0"),
        )
        available_cash = Decimal(str(account.opening_cash)) + Decimal(str(cash_movements)) - reserved_cash

        specs = []
        planned_cash = Decimal("0")
        planned_quantity = {}
        for sequence, intent in enumerate(intents):
            identity = (intent.market, intent.symbol, intent.instrument, intent.side)
            price_identity = (intent.market, intent.symbol, intent.instrument)
            if price_identity not in prices:
                raise ExecutionConflictError(f"missing current price for {price_identity}")
            price = _finite_positive(prices[price_identity], f"price for {price_identity}")
            price_decimal = Decimal(str(price))
            multiplier = Decimal("1")
            step = Decimal("1")
            sizing_rules = {"quantity_rounding": "whole_share_floor"}
            if intent.market == "crypto_perp":
                rule = reconciliation.perp_instrument_rules.get(intent.instrument)
                if rule is None:
                    raise ExecutionConflictError("owner-frozen perp sizing rules are required")
                multiplier = Decimal(str(rule.contract_multiplier))
                step = Decimal(str(rule.quantity_step))
                sizing_rules = rule.model_dump(mode="json")
            target_quantity = Decimal(str(intent.target_weight)) * nav_decimal / (price_decimal * multiplier)
            lots = self._position_lots(decision, reconciliation.account_generation, intent)
            settled = sum((Decimal(str(lot.open_quantity)) for lot in lots), start=Decimal("0"))
            lot_reserved_close = sum(
                (Decimal(str(lot.reserved_close_quantity)) for lot in lots),
                start=Decimal("0"),
            )
            pending_increase = sum(
                (
                    reservations[order.id][0]
                    for order in lot_pending
                    if self._identity(order) == identity
                    and (order.intent_json or {}).get("frozen_intent", {}).get("action") in {"enter", "add"}
                ),
                start=Decimal("0"),
            ) + planned_quantity.get((identity, "increase"), Decimal("0"))
            pending_close = sum(
                (
                    reservations[order.id][0]
                    for order in lot_pending
                    if self._identity(order) == identity
                    and (order.intent_json or {}).get("frozen_intent", {}).get("action") in {"reduce", "exit"}
                ),
                start=Decimal("0"),
            ) + planned_quantity.get((identity, "close"), Decimal("0"))
            effective = settled + pending_increase - pending_close
            delta = target_quantity - effective
            if intent.action in {"enter", "add"} and delta <= 0:
                raise ExecutionConflictError("enter/add requires a positive quantity delta")
            if intent.action in {"reduce", "exit"} and delta >= 0:
                raise ExecutionConflictError("reduce/exit requires a negative quantity delta")
            quantity = _floor_step(abs(delta), step)
            if quantity <= 0:
                raise ExecutionConflictError("materialized order requires positive quantity")
            if intent.action in {"reduce", "exit"} and quantity > settled - max(pending_close, lot_reserved_close):
                raise ExecutionConflictError("close quantity exceeds settled unreserved quantity")
            cash_amount = quantity * price_decimal * multiplier if intent.action in {"enter", "add"} else Decimal("0")
            if planned_cash + cash_amount > available_cash:
                raise ExecutionConflictError("pending intents exceed available paper cash")
            planned_cash += cash_amount
            reservation_kind = "increase" if intent.action in {"enter", "add"} else "close"
            planned_quantity[(identity, reservation_kind)] = (
                planned_quantity.get(
                    (identity, reservation_kind),
                    Decimal("0"),
                )
                + quantity
            )
            frozen = intent.model_dump(mode="json")
            economics = {
                "account_nav": nav,
                "price": price,
                "target_quantity": float(target_quantity),
                "settled_quantity": float(settled),
                "pending_increase_quantity": float(pending_increase),
                "pending_close_quantity": float(pending_close),
                "delta_quantity": float(delta),
                "materialized_quantity": float(quantity),
                "contract_multiplier": float(multiplier),
                "sizing_rules": sizing_rules,
            }
            stored_intent = json.loads(canonical_json({"frozen_intent": frozen, "economics": economics}))
            specs.append((sequence, intent, quantity, cash_amount, stored_intent, lots, reservation_kind))

        projected_failures = self._projected_risk_failures(
            decision,
            reconciliation,
            lot_pending,
            specs,
            prices,
            nav,
            policy,
            reservations,
        )
        if projected_failures:
            return self._record_risk_block(decision, projected_failures, principal)

        for _, _, quantity, _, _, lots, reservation_kind in specs:
            if reservation_kind != "close":
                continue
            remaining = quantity
            for lot in lots:
                current_reserved = Decimal(str(lot.reserved_close_quantity))
                reservable = Decimal(str(lot.open_quantity)) - current_reserved
                reserved = min(remaining, reservable)
                lot.reserved_close_quantity = float(current_reserved + reserved)
                remaining -= reserved
                if remaining <= 0:
                    break
            if remaining > 0:
                raise ExecutionConflictError("close quantity reservation is incomplete")

        orders = []
        for sequence, intent, quantity, cash_amount, stored_intent, _, _ in specs:
            broker_action = "buy" if (intent.side == "long") == (intent.action in {"enter", "add"}) else "sell"
            order = OrderRecord(
                id=uuid.uuid4(),
                strategy_name=f"decision-loop:{decision.strategy_version_id}",
                symbol=intent.symbol,
                market=intent.market,
                action=broker_action,
                order_type=intent.order_type,
                target_weight=intent.target_weight,
                quantity=float(quantity),
                price=stored_intent["economics"]["price"],
                side=intent.side,
                status="pending_submit",
                broker_mode="paper",
                order_origin="decision",
                decision_id=decision.id,
                account_scope=decision.account_scope,
                account_generation=reconciliation.account_generation,
                execution_key=decision.execution_key,
                client_order_ref=f"DL-{decision.execution_key.hex}-{sequence:04d}",
                instrument=intent.instrument,
                intent_json=stored_intent,
                intent_sha256=content_sha256(stored_intent),
                reserved_cash_json={"currency": reconciliation.currency, "amount": float(cash_amount)},
                reserved_quantity=float(quantity),
                reservation_status="reserved",
                reconciliation_status="pending",
                created_at=current_time,
                updated_at=current_time,
            )
            self.session.add(order)
            orders.append(order)
        self.session.flush()
        return self._response(decision, orders)
