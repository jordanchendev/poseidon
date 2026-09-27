"""Materialize approved decisions as durable broker-independent order intents."""

import json
import math
import uuid
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import Float, func, select, text

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import (
    DecisionPolicy,
    _stored_time,
    _validate_order_intents,
    _validate_selection,
)
from poseidon.decision_loop.evaluation import verify_complete_run
from poseidon.decision_loop.manifest import ValidationError, canonical_json, content_sha256, iso_time, timestamp
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
from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion
from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS

ACTIVE_RESERVATION_STATUSES = frozenset({"pending_submit", "reconciliation_required", "submitted", "partially_filled"})
PROTECTIVE_ORIGINS = DURABLE_PROTECTIVE_ORIGINS


class ExecutionConflictError(RuntimeError):
    """The frozen decision cannot be materialized without changing its meaning."""


def _finite_positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ExecutionConflictError(f"{field} must be finite and positive")
    return float(value)


def _aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


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
                value = iso_time(_aware(value), column.key)
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

    def _advisory_lock(self, account_scope, account_generation):
        if self.session.get_bind().dialect.name != "postgresql":
            return
        lock_key = int(
            content_sha256({"account_scope": account_scope, "account_generation": account_generation})[:16], 16
        )
        if lock_key >= 2**63:
            lock_key -= 2**64
        self.session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})

    def _response(self, decision, orders):
        return {
            "decision_id": str(decision.id),
            "status": "pending_submit",
            "order_ids": [str(order.id) for order in orders],
            "client_order_refs": [order.client_order_ref for order in orders],
        }

    def _replay(self, decision, intents, account_generation):
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

    def _policy_and_intents(self, decision):
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
        self._advisory_lock(decision.account_scope, reconciliation.account_generation)
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
        from poseidon.decision_loop.decisions import DecisionService

        principal.require_role("decision-worker")
        principal.require_account_scope(decision.account_scope)
        capabilities = adapter.capabilities
        if not isinstance(capabilities, BrokerCapabilities) or not capabilities.supports_reconciliation:
            raise ExecutionConflictError("paper adapter lacks full reconciliation capability")
        _, policy, intents = self._policy_and_intents(decision)
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
        as_of = _aware(latest.as_of)
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
        if _aware(account.updated_at) > as_of or any(
            self.session.scalar(statement.limit(1)) is not None for statement in newer_state
        ):
            raise ExecutionConflictError("account reconciliation predates outstanding state")

    def _pending_orders(self, decision, account_generation):
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

    def _outstanding_reservation(self, order, currency, *, lock_fills=True):
        original = Decimal(str(_finite_positive(order.reserved_quantity, "reserved_quantity")))
        statement = select(OrderFillRecord).where(OrderFillRecord.order_id == order.id)
        if lock_fills:
            statement = statement.with_for_update()
        fills = self.session.scalars(statement).all()
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
    ):
        principal.require_role("decision-worker")
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        nav = _finite_positive(account_nav, "account_nav")
        nav_decimal = Decimal(str(nav))
        decision = self.session.get(DecisionRecord, decision_id)
        if decision is None:
            raise ExecutionConflictError("decision does not exist")
        principal.require_account_scope(decision.account_scope)
        _, policy, intents = self._policy_and_intents(decision)
        reconciliation = policy.reconciliation
        account = self._lock_account(decision, policy)
        decision = (
            self.session.query(DecisionRecord).filter_by(id=decision.id).with_for_update().populate_existing().one()
        )
        _, locked_policy, locked_intents = self._policy_and_intents(decision)
        if locked_policy.reconciliation.account_generation != reconciliation.account_generation:
            raise ExecutionConflictError("owner account generation changed while locking")
        policy, intents, reconciliation = locked_policy, locked_intents, locked_policy.reconciliation

        replay = self._replay(decision, intents, reconciliation.account_generation)
        if replay is not None:
            return replay
        if decision.status == "risk_blocked":
            return {"decision_id": str(decision.id), "status": "risk_blocked", "order_ids": []}
        if decision.status != "execution_claimed" or decision.execution_key is None:
            raise ExecutionConflictError("decision is not execution_claimed")
        if current_time >= _stored_time(decision.valid_until, "valid_until"):
            raise ExecutionConflictError("decision has expired")

        failures = self._risk_failures(decision, policy, intents)
        if failures:
            return self._record_risk_block(decision, failures, principal)
        if any(intent.action in {"enter", "add"} for intent in intents):
            self._require_reconciled(decision, reconciliation, account, current_time)

        pending = self._pending_orders(decision, reconciliation.account_generation)
        reservations = {order.id: self._outstanding_reservation(order, reconciliation.currency) for order in pending}
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


class ProtectiveExecutionService(DecisionExecutionService):
    """Materialize one account-locked, reduction-only protective intent."""

    @staticmethod
    def _response_for(order):
        return {
            "status": order.status,
            "execution_key": str(order.execution_key),
            "order_ids": [str(order.id)],
            "client_order_refs": [order.client_order_ref],
        }

    def _policy(self, decision_id, account_scope, account_generation, market):
        decision = self.session.get(DecisionRecord, decision_id)
        version = None if decision is None else self.session.get(StrategyVersion, decision.strategy_version_id)
        try:
            if decision is None or version is None:
                raise ValueError("missing decision provenance")
            version.verify_content()
            policy = DecisionPolicy.model_validate(version.policy_json)
        except (TypeError, ValueError, PydanticValidationError) as error:
            raise ExecutionConflictError("protective source decision policy is invalid") from error
        reconciliation = policy.reconciliation
        if (
            decision.account_scope != account_scope
            or policy.account_scope != account_scope
            or policy.market != market
            or reconciliation is None
            or reconciliation.account_generation != account_generation
            or not policy.protective_exit.allowed_without_approval
            or decision.policy_sha256 != content_sha256(version.policy_json)
        ):
            raise ExecutionConflictError("protective source decision is outside the approved paper scope")
        return decision, policy

    @staticmethod
    def _dedupe_input(order_origin, context):
        result = {
            "account_scope": context["account_scope"],
            "account_generation": context["account_generation"],
            "origin": order_origin,
            "trigger_generation": context["trigger_generation"],
        }
        source = "source_holding_ids" if context.get("legacy_exception") is True else "source_lot_ids"
        result[source] = context[source]
        if context.get("legacy_exception") is True:
            result["legacy_context_sha256"] = context["legacy_context_sha256"]
        return result

    def validate_order(self, order, *, lock_fills=True):
        """Revalidate a stored protective intent before attempt or projection."""
        if order.order_origin not in PROTECTIVE_ORIGINS:
            raise ExecutionConflictError("protective order origin is not exact-allowlisted")
        if order.broker_mode != "paper" or order.signal_id is not None:
            raise ExecutionConflictError("protective order must be paper and non-signal")
        intent = order.intent_json
        context = order.protective_context_json
        legacy = isinstance(context, dict) and context.get("legacy_exception") is True
        expected_context = {
            "account_scope",
            "account_generation",
            "identity",
            "origin",
            "trigger_generation",
            "source_lot_ids",
            "source_decision_ids",
            "dedupe_sha256",
        }
        if legacy:
            expected_context |= {
                "legacy_exception",
                "source_holding_ids",
                "source_holding_quantities",
                "source_holding_risk",
                "legacy_context_sha256",
            }
        if (
            not isinstance(intent, dict)
            or order.intent_sha256 != content_sha256(intent)
            or not isinstance(intent.get("frozen_intent"), dict)
            or not isinstance(intent.get("economics"), dict)
            or not isinstance(context, dict)
            or set(context) != expected_context
        ):
            raise ExecutionConflictError("protective durable content is invalid")
        frozen = intent["frozen_intent"]
        identity = {
            "market": order.market,
            "symbol": order.symbol,
            "instrument": order.instrument,
            "side": order.side,
        }
        if (
            frozen
            != {
                **identity,
                "action": frozen.get("action"),
                "target_weight": 0.0,
                "order_type": "market",
            }
            or frozen["action"] not in {"reduce", "exit"}
            or context["identity"] != identity
            or context["account_scope"] != order.account_scope
            or context["account_generation"] != order.account_generation
            or context["origin"] != order.order_origin
            or not isinstance(context["trigger_generation"], str)
            or not context["trigger_generation"]
            or context["trigger_generation"] != context["trigger_generation"].strip()
        ):
            raise ExecutionConflictError("protective intent is not reduction-only or changed identity")
        digest = content_sha256(self._dedupe_input(order.order_origin, context))
        expected_key = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:execution:{digest}")
        expected_id = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:order:{digest}")
        if (
            context["dedupe_sha256"] != digest
            or order.execution_key != expected_key
            or order.id != expected_id
            or order.client_order_ref != f"PX-{digest}"
        ):
            raise ExecutionConflictError("protective deterministic identity changed")
        if legacy:
            self._validate_legacy_order(order, context, lock_fills=lock_fills)
            return
        try:
            source_lot_ids = [uuid.UUID(value) for value in context["source_lot_ids"]]
            source_decision_ids = [uuid.UUID(value) for value in context["source_decision_ids"]]
        except (TypeError, ValueError) as error:
            raise ExecutionConflictError("protective source provenance is invalid") from error
        if (
            not source_lot_ids
            or len(set(source_lot_ids)) != len(source_lot_ids)
            or not source_decision_ids
            or context["source_decision_ids"] != sorted(context["source_decision_ids"])
            or order.decision_id != min(source_decision_ids, key=str)
        ):
            raise ExecutionConflictError("protective source provenance is invalid")
        lots = self.session.scalars(
            select(PositionLot)
            .where(PositionLot.id.in_(source_lot_ids))
            .order_by(PositionLot.opened_at, PositionLot.id)
        ).all()
        if [lot.id for lot in lots] != source_lot_ids:
            raise ExecutionConflictError("protective source lot set changed")
        if (
            any(
                (
                    lot.account_scope,
                    lot.account_generation,
                    lot.market,
                    lot.symbol,
                    lot.instrument,
                    lot.side,
                )
                != (
                    order.account_scope,
                    order.account_generation,
                    order.market,
                    order.symbol,
                    order.instrument,
                    order.side,
                )
                for lot in lots
            )
            or sorted({str(lot.opening_decision_id) for lot in lots}) != context["source_decision_ids"]
        ):
            raise ExecutionConflictError("protective source lot ownership changed")
        policies = [
            self._policy(decision_id, order.account_scope, order.account_generation, order.market)
            for decision_id in source_decision_ids
        ]
        currency = policies[0][1].reconciliation.currency
        if any(policy.reconciliation.currency != currency for _, policy in policies):
            raise ExecutionConflictError("protective source policies disagree")
        quantity = _finite_positive(order.quantity, "protective quantity")
        economics = intent["economics"]
        if (
            order.reserved_quantity != quantity
            or economics.get("materialized_quantity") != quantity
            or economics.get("price") != order.price
            or order.action != ("sell" if order.side == "long" else "buy")
            or order.target_weight != 0
            or order.order_type != "market"
        ):
            raise ExecutionConflictError("protective executable economics changed")
        outstanding, cash = self._outstanding_reservation(order, currency, lock_fills=lock_fills)
        if cash != 0 or sum((Decimal(str(lot.reserved_close_quantity)) for lot in lots), Decimal(0)) < outstanding:
            raise ExecutionConflictError("protective close reservation changed")
        event = self.session.scalar(
            select(DecisionEvent).where(
                DecisionEvent.decision_id == order.decision_id,
                DecisionEvent.event_type == f"protective_{digest[:13]}",
            )
        )
        expected_event_payload = {
            "kind": "protective_exit_requested",
            "order_id": str(order.id),
            "execution_key": str(order.execution_key),
            "client_order_ref": order.client_order_ref,
            "intent_sha256": order.intent_sha256,
            "protective_context_sha256": digest,
            "origin": order.order_origin,
            "trigger_generation": context["trigger_generation"],
            "current_price": order.price,
            "identity": context["identity"],
            "action": frozen["action"],
            "quantity": order.quantity,
            "source_lot_ids": context["source_lot_ids"],
            "source_decision_ids": context["source_decision_ids"],
        }
        if event is None or event.payload_json != expected_event_payload:
            raise ExecutionConflictError("protective audit event is missing or changed")

    def _validate_legacy_order(self, order, context, *, lock_fills):
        try:
            source_ids = [uuid.UUID(value) for value in context["source_holding_ids"]]
        except (TypeError, ValueError) as error:
            raise ExecutionConflictError("protective legacy source provenance is invalid") from error
        quantities = context["source_holding_quantities"]
        risk = context["source_holding_risk"]
        legacy_context_sha256 = content_sha256({"source_holding_quantities": quantities, "source_holding_risk": risk})
        if (
            not source_ids
            or len(set(source_ids)) != len(source_ids)
            or context["source_holding_ids"] != sorted(context["source_holding_ids"])
            or context["source_lot_ids"] != []
            or context["source_decision_ids"] != []
            or order.decision_id is not None
            or not isinstance(quantities, dict)
            or set(quantities) != set(context["source_holding_ids"])
            or not isinstance(risk, dict)
            or set(risk) != set(context["source_holding_ids"])
            or context["legacy_context_sha256"] != legacy_context_sha256
            or order.intent_json["economics"].get("legacy_context_sha256") != legacy_context_sha256
        ):
            raise ExecutionConflictError("protective legacy source provenance is invalid")
        account = self.session.scalar(
            select(PaperBrokerAccount).where(
                PaperBrokerAccount.account_scope == order.account_scope,
                PaperBrokerAccount.account_generation == order.account_generation,
            )
        )
        holdings = self.session.scalars(
            select(PortfolioHoldingRecord)
            .where(PortfolioHoldingRecord.id.in_(source_ids))
            .order_by(PortfolioHoldingRecord.id)
        ).all()
        if account is None or account.currency != "TWD" or [row.id for row in holdings] != source_ids:
            raise ExecutionConflictError("protective legacy source set changed")
        original = Decimal("0")
        remaining = Decimal("0")
        for holding in holdings:
            amount = _finite_positive(quantities[str(holding.id)], "protective legacy source quantity")
            frozen_risk = risk[str(holding.id)]
            if not isinstance(frozen_risk, dict) or set(frozen_risk) != {"entry_price", "stop_loss_pct"}:
                raise ExecutionConflictError("protective legacy source risk is invalid")
            entry_price = _finite_positive(frozen_risk["entry_price"], "protective legacy entry price")
            stop_loss_pct = _finite_positive(frozen_risk["stop_loss_pct"], "protective legacy stop loss")
            shares = Decimal(str(holding.shares)) if holding.shares is not None else Decimal("-1")
            if (
                holding.strategy_name.startswith("decision-lots:")
                or (holding.market, holding.symbol, holding.side) != (order.market, order.symbol, order.side)
                or shares < 0
                or shares > Decimal(str(amount))
                or holding.closed != (shares == 0)
                or entry_price <= 0
                or stop_loss_pct >= 1
            ):
                raise ExecutionConflictError("protective legacy source ownership changed")
            original += Decimal(str(amount))
            remaining += shares
        quantity = _finite_positive(order.quantity, "protective quantity")
        economics = order.intent_json["economics"]
        if (
            order.market != "tw_stock"
            or order.instrument != "spot"
            or order.reserved_quantity != quantity
            or economics.get("materialized_quantity") != quantity
            or economics.get("price") != order.price
            or order.action != ("sell" if order.side == "long" else "buy")
            or original != Decimal(str(quantity))
        ):
            raise ExecutionConflictError("protective legacy executable economics changed")
        outstanding, cash = self._outstanding_reservation(order, account.currency, lock_fills=lock_fills)
        if cash != 0 or remaining < outstanding:
            raise ExecutionConflictError("protective legacy close reservation changed")

    def _materialize_legacy_holdings(
        self,
        *,
        account,
        account_scope,
        account_generation,
        market,
        symbol,
        instrument,
        side,
        origin,
        trigger_generation,
        current_price,
        action,
        quantity,
        source_holding_ids,
        current_time,
    ):
        if market != "tw_stock" or instrument != "spot" or account.currency != "TWD":
            raise ExecutionConflictError("legacy protective exception is limited to the approved TW paper account")
        try:
            source_ids = sorted({uuid.UUID(str(value)) for value in source_holding_ids}, key=str)
        except (TypeError, ValueError) as error:
            raise ExecutionConflictError("protective legacy source provenance is invalid") from error
        if not source_ids or len(source_ids) != len(source_holding_ids):
            raise ExecutionConflictError("protective legacy source provenance is invalid")
        holdings = self.session.scalars(
            select(PortfolioHoldingRecord)
            .where(PortfolioHoldingRecord.id.in_(source_ids))
            .order_by(PortfolioHoldingRecord.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        if [row.id for row in holdings] != source_ids or any(
            row.closed
            or row.strategy_name.startswith("decision-lots:")
            or (row.market, row.symbol, row.side) != (market, symbol, side)
            or row.shares is None
            or not math.isfinite(row.shares)
            or row.shares <= 0
            or row.entry_price is None
            or not math.isfinite(row.entry_price)
            or row.entry_price <= 0
            or row.stop_loss_pct is None
            or not math.isfinite(row.stop_loss_pct)
            or row.stop_loss_pct <= 0
            or row.stop_loss_pct >= 1
            for row in holdings
        ):
            raise ExecutionConflictError("protective legacy source ownership changed")
        requested_ids = {str(value) for value in source_ids}
        replay_candidates = self.session.scalars(
            select(OrderRecord).where(
                OrderRecord.account_scope == account_scope,
                OrderRecord.account_generation == account_generation,
                OrderRecord.order_origin == origin,
            )
        ).all()
        for candidate in replay_candidates:
            candidate_context = candidate.protective_context_json
            if (
                isinstance(candidate_context, dict)
                and candidate_context.get("legacy_exception") is True
                and candidate_context.get("trigger_generation") == trigger_generation
                and candidate_context.get("source_holding_ids") == sorted(requested_ids)
            ):
                self.validate_order(candidate)
                return self._response_for(candidate)
        reserved_by_id = {value: Decimal("0") for value in requested_ids}
        siblings = self.session.scalars(
            select(OrderRecord).where(
                OrderRecord.account_scope == account_scope,
                OrderRecord.account_generation == account_generation,
                OrderRecord.market == market,
                OrderRecord.symbol == symbol,
                OrderRecord.instrument == instrument,
                OrderRecord.side == side,
                OrderRecord.order_origin.in_(PROTECTIVE_ORIGINS),
                OrderRecord.reservation_status == "reserved",
            )
        ).all()
        for sibling in siblings:
            context = sibling.protective_context_json
            if (
                isinstance(context, dict)
                and context.get("legacy_exception") is True
                and requested_ids.intersection(context.get("source_holding_ids", ()))
            ):
                self.validate_order(sibling)
                outstanding, _cash = self._outstanding_reservation(sibling, account.currency)
                sibling_quantities = context["source_holding_quantities"]
                filled = Decimal(str(sibling.reserved_quantity)) - outstanding
                for holding_id in context["source_holding_ids"]:
                    planned = Decimal(str(sibling_quantities[holding_id]))
                    consumed = min(planned, filled)
                    filled -= consumed
                    if holding_id in reserved_by_id:
                        reserved_by_id[holding_id] += planned - consumed
                if filled:
                    raise ExecutionConflictError("legacy protective fills exceed frozen source attribution")
        available_holdings = []
        available_by_id = {}
        for row in holdings:
            available = Decimal(str(row.shares)) - reserved_by_id[str(row.id)]
            if available < 0:
                raise ExecutionConflictError("legacy protective reservations exceed source inventory")
            if available:
                available_holdings.append(row)
                available_by_id[str(row.id)] = available
        available = sum(available_by_id.values(), Decimal("0"))
        requested = available if quantity is None else Decimal(str(_finite_positive(quantity, "protective quantity")))
        if quantity is not None and requested != available:
            raise ExecutionConflictError("legacy protective exception requires a full available close")
        close_quantity = available
        if close_quantity <= 0:
            return {"status": "already_reserved", "execution_key": None, "order_ids": [], "client_order_refs": []}
        source_holding_ids = sorted(available_by_id)
        source_holding_quantities = {
            holding_id: float(available_by_id[holding_id]) for holding_id in source_holding_ids
        }
        source_holding_risk = {
            str(row.id): {"entry_price": row.entry_price, "stop_loss_pct": row.stop_loss_pct}
            for row in available_holdings
        }
        legacy_context_sha256 = content_sha256(
            {
                "source_holding_quantities": source_holding_quantities,
                "source_holding_risk": source_holding_risk,
            }
        )
        dedupe_input = {
            "account_scope": account_scope,
            "account_generation": account_generation,
            "origin": origin,
            "source_holding_ids": source_holding_ids,
            "trigger_generation": trigger_generation,
            "legacy_context_sha256": legacy_context_sha256,
        }
        digest = content_sha256(dedupe_input)
        order_id = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:order:{digest}")
        existing = self.session.get(OrderRecord, order_id)
        if existing is not None:
            self.validate_order(existing)
            return self._response_for(existing)
        execution_key = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:execution:{digest}")
        frozen = {
            "market": market,
            "symbol": symbol,
            "instrument": instrument,
            "side": side,
            "action": action,
            "target_weight": 0.0,
            "order_type": "market",
        }
        intent = json.loads(
            canonical_json(
                {
                    "frozen_intent": frozen,
                    "economics": {
                        "price": current_price,
                        "materialized_quantity": float(close_quantity),
                        "contract_multiplier": 1.0,
                        "sizing_rules": {"quantity_rounding": "whole_share_floor"},
                        "legacy_context_sha256": legacy_context_sha256,
                    },
                }
            )
        )
        context = json.loads(
            canonical_json(
                {
                    **dedupe_input,
                    "identity": {"market": market, "symbol": symbol, "instrument": instrument, "side": side},
                    "source_lot_ids": [],
                    "source_decision_ids": [],
                    "source_holding_quantities": source_holding_quantities,
                    "source_holding_risk": source_holding_risk,
                    "legacy_exception": True,
                    "dedupe_sha256": digest,
                }
            )
        )
        order = OrderRecord(
            id=order_id,
            strategy_name=f"protective:{origin}",
            symbol=symbol,
            market=market,
            action="sell" if side == "long" else "buy",
            order_type="market",
            target_weight=0,
            quantity=float(close_quantity),
            price=current_price,
            side=side,
            status="pending_submit",
            broker_mode="paper",
            order_origin=origin,
            decision_id=None,
            account_scope=account_scope,
            account_generation=account_generation,
            execution_key=execution_key,
            client_order_ref=f"PX-{digest}",
            instrument=instrument,
            intent_json=intent,
            intent_sha256=content_sha256(intent),
            reserved_cash_json={"currency": account.currency, "amount": 0.0},
            reserved_quantity=float(close_quantity),
            reservation_status="reserved",
            reconciliation_status="pending",
            protective_context_json=context,
            created_at=current_time,
            updated_at=current_time,
        )
        self.session.add(order)
        self.session.flush()
        return self._response_for(order)

    def materialize(
        self,
        *,
        account_scope,
        account_generation,
        market,
        symbol,
        instrument,
        side,
        origin,
        trigger_generation,
        price,
        principal: AuthPrincipal,
        action="exit",
        quantity=None,
        source_holding_ids=None,
        allow_legacy_holdings=False,
        now=None,
    ):
        principal.require_role("decision-worker")
        principal.require_account_scope(account_scope)
        if origin not in PROTECTIVE_ORIGINS:
            raise ExecutionConflictError("protective origin is not exact-allowlisted")
        if action not in {"reduce", "exit"}:
            raise ExecutionConflictError("protective intent must be reduction-only")
        if (
            not isinstance(trigger_generation, str)
            or not trigger_generation
            or trigger_generation != trigger_generation.strip()
        ):
            raise ExecutionConflictError("protective trigger generation must be non-empty trimmed text")
        if market not in {"tw_stock", "crypto_perp"} or side not in {"long", "short"}:
            raise ExecutionConflictError("protective identity is invalid")
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        current_price = _finite_positive(price, "protective price")
        self._advisory_lock(account_scope, account_generation)
        account = self.session.scalar(
            select(PaperBrokerAccount)
            .where(
                PaperBrokerAccount.account_scope == account_scope,
                PaperBrokerAccount.account_generation == account_generation,
            )
            .with_for_update()
        )
        if account is None:
            raise ExecutionConflictError("approved paper account generation does not exist")
        if source_holding_ids is not None:
            if not allow_legacy_holdings:
                raise ExecutionConflictError("legacy protective source requires explicit authorization")
            return self._materialize_legacy_holdings(
                account=account,
                account_scope=account_scope,
                account_generation=account_generation,
                market=market,
                symbol=symbol,
                instrument=instrument,
                side=side,
                origin=origin,
                trigger_generation=trigger_generation,
                current_price=current_price,
                action=action,
                quantity=quantity,
                source_holding_ids=source_holding_ids,
                current_time=current_time,
            )
        if allow_legacy_holdings:
            raise ExecutionConflictError("legacy protective source IDs are required")
        active_source_lot_ids = set()
        for existing in self.session.scalars(
            select(OrderRecord).where(
                OrderRecord.account_scope == account_scope,
                OrderRecord.account_generation == account_generation,
                OrderRecord.market == market,
                OrderRecord.symbol == symbol,
                OrderRecord.instrument == instrument,
                OrderRecord.side == side,
                OrderRecord.order_origin == origin,
                OrderRecord.reservation_status == "reserved",
                OrderRecord.status.in_(ACTIVE_RESERVATION_STATUSES),
            )
        ):
            context = existing.protective_context_json
            if isinstance(context, dict) and context.get("trigger_generation") == trigger_generation:
                try:
                    active_source_lot_ids.update(uuid.UUID(value) for value in context["source_lot_ids"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ExecutionConflictError("protective source provenance is invalid") from error
        lots = self.session.scalars(
            select(PositionLot)
            .where(
                PositionLot.account_scope == account_scope,
                PositionLot.account_generation == account_generation,
                PositionLot.market == market,
                PositionLot.symbol == symbol,
                PositionLot.instrument == instrument,
                PositionLot.side == side,
                (PositionLot.open_quantity > 0) | (PositionLot.id.in_(active_source_lot_ids)),
            )
            .order_by(PositionLot.opened_at, PositionLot.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).all()
        if not lots:
            return {"status": "no_open_lots", "execution_key": None, "order_ids": [], "client_order_refs": []}
        source_decision_ids = sorted({str(lot.opening_decision_id) for lot in lots})
        policies = [
            self._policy(uuid.UUID(decision_id), account_scope, account_generation, market)
            for decision_id in source_decision_ids
        ]
        currency = policies[0][1].reconciliation.currency
        if account.currency != currency or any(policy.reconciliation.currency != currency for _, policy in policies):
            raise ExecutionConflictError("protective account terms disagree with source policy")
        source_lot_ids = [str(lot.id) for lot in lots]
        dedupe_input = {
            "account_scope": account_scope,
            "account_generation": account_generation,
            "origin": origin,
            "source_lot_ids": source_lot_ids,
            "trigger_generation": trigger_generation,
        }
        digest = content_sha256(dedupe_input)
        execution_key = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:execution:{digest}")
        order_id = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:order:{digest}")
        existing = self.session.get(OrderRecord, order_id)
        if existing is not None:
            self.validate_order(existing)
            return self._response_for(existing)
        available = sum(
            (Decimal(str(lot.open_quantity)) - Decimal(str(lot.reserved_close_quantity)) for lot in lots),
            Decimal(0),
        )
        requested = available if quantity is None else Decimal(str(_finite_positive(quantity, "protective quantity")))
        close_quantity = min(available, requested)
        if close_quantity <= 0:
            return {"status": "already_reserved", "execution_key": None, "order_ids": [], "client_order_refs": []}
        anchor = min((uuid.UUID(value) for value in source_decision_ids), key=str)
        policy = next(policy for decision, policy in policies if decision.id == anchor)
        multiplier = Decimal("1")
        sizing_rules = {"quantity_rounding": "whole_share_floor"}
        if market == "crypto_perp":
            rule = policy.reconciliation.perp_instrument_rules.get(instrument)
            if rule is None:
                raise ExecutionConflictError("owner-frozen perp sizing rules are required")
            multiplier = Decimal(str(rule.contract_multiplier))
            sizing_rules = rule.model_dump(mode="json")
        frozen = {
            "market": market,
            "symbol": symbol,
            "instrument": instrument,
            "side": side,
            "action": action,
            "target_weight": 0.0,
            "order_type": "market",
        }
        intent = json.loads(
            canonical_json(
                {
                    "frozen_intent": frozen,
                    "economics": {
                        "price": current_price,
                        "materialized_quantity": float(close_quantity),
                        "contract_multiplier": float(multiplier),
                        "sizing_rules": sizing_rules,
                    },
                }
            )
        )
        context = json.loads(
            canonical_json(
                {
                    **dedupe_input,
                    "identity": {"market": market, "symbol": symbol, "instrument": instrument, "side": side},
                    "source_decision_ids": source_decision_ids,
                    "dedupe_sha256": digest,
                }
            )
        )
        remaining = close_quantity
        for lot in lots:
            reservable = Decimal(str(lot.open_quantity)) - Decimal(str(lot.reserved_close_quantity))
            reserved = min(remaining, reservable)
            lot.reserved_close_quantity = float(Decimal(str(lot.reserved_close_quantity)) + reserved)
            remaining -= reserved
            if remaining <= 0:
                break
        if remaining:
            raise ExecutionConflictError("protective close reservation is incomplete")
        order = OrderRecord(
            id=order_id,
            strategy_name=f"protective:{origin}",
            symbol=symbol,
            market=market,
            action="sell" if side == "long" else "buy",
            order_type="market",
            target_weight=0,
            quantity=float(close_quantity),
            price=current_price,
            side=side,
            status="pending_submit",
            broker_mode="paper",
            order_origin=origin,
            decision_id=anchor,
            account_scope=account_scope,
            account_generation=account_generation,
            execution_key=execution_key,
            client_order_ref=f"PX-{digest}",
            instrument=instrument,
            intent_json=intent,
            intent_sha256=content_sha256(intent),
            reserved_cash_json={"currency": currency, "amount": 0.0},
            reserved_quantity=float(close_quantity),
            reservation_status="reserved",
            reconciliation_status="pending",
            protective_context_json=context,
            created_at=current_time,
            updated_at=current_time,
        )
        anchor_decision = next(decision for decision, _ in policies if decision.id == anchor)
        self.session.add(order)
        self.session.add(
            DecisionEvent(
                decision_id=anchor,
                event_type=f"protective_{digest[:13]}",
                actor_id=principal.actor_id,
                expected_revision=anchor_decision.revision,
                payload_json={
                    "kind": "protective_exit_requested",
                    "order_id": str(order_id),
                    "execution_key": str(execution_key),
                    "client_order_ref": order.client_order_ref,
                    "intent_sha256": order.intent_sha256,
                    "protective_context_sha256": digest,
                    "origin": origin,
                    "trigger_generation": trigger_generation,
                    "current_price": current_price,
                    "identity": context["identity"],
                    "action": action,
                    "quantity": float(close_quantity),
                    "source_lot_ids": source_lot_ids,
                    "source_decision_ids": source_decision_ids,
                },
            )
        )
        self.session.flush()
        return self._response_for(order)
