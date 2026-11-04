"""Protective (stop-loss) execution: validate and materialize account-locked, reduction-only exit orders."""

import json
import math
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from poseidon.core.authz import AuthPrincipal
from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.locks import lock_account
from poseidon.core.timeutil import timestamp
from poseidon.decision_loop.policy import DecisionPolicy
from poseidon.decision_loop.reservations import (
    ACTIVE_RESERVATION_STATUSES,
    PROTECTIVE_ORIGINS,
    ExecutionConflictError,
    _finite_positive,
    outstanding_reservation,
)
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion
from poseidon.orders.protective_context import ProtectiveContext, ProtectiveContextError


class ProtectiveExecutionService:
    """Materialize one account-locked, reduction-only protective intent."""

    def __init__(self, session):
        self.session = session

    @staticmethod
    def _response_for(order):
        return {
            "status": order.status,
            "execution_key": str(order.execution_key),
            "order_ids": [str(order.id)],
            "client_order_refs": [order.client_order_ref],
        }

    def source_decision_policy(self, decision_id, account_scope, account_generation, market):
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
        if "predecessor_order_ids" in context:
            result["predecessor_order_ids"] = context["predecessor_order_ids"]
            result["remaining_source_quantities"] = context["remaining_source_quantities"]
        return result

    @staticmethod
    def _generation_complete(order, fills):
        return (
            order.status in {"filled", "rejected", "cancelled"}
            and order.reservation_status == "released"
            and all(fill.projection_status == "applied" for fill in fills)
        )

    def _generation_replay(
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
        source_field,
        source_ids,
    ):
        completed = []
        incomplete = []
        candidates = self.session.scalars(
            select(OrderRecord)
            .where(
                OrderRecord.account_scope == account_scope,
                OrderRecord.account_generation == account_generation,
                OrderRecord.market == market,
                OrderRecord.symbol == symbol,
                OrderRecord.instrument == instrument,
                OrderRecord.side == side,
                OrderRecord.order_origin == origin,
            )
            .order_by(OrderRecord.id)
        ).all()
        for candidate in candidates:
            context = candidate.protective_context_json
            self.validate_order(candidate)
            if context.get("trigger_generation") != trigger_generation:
                continue
            if context.get(source_field) != source_ids:
                continue
            fills = self.session.scalars(
                select(OrderFillRecord).where(OrderFillRecord.order_id == candidate.id).with_for_update()
            ).all()
            (completed if self._generation_complete(candidate, fills) else incomplete).append(candidate)
        if len(incomplete) > 1:
            raise ExecutionConflictError("protective generation has multiple incomplete orders")
        if not completed:
            return (incomplete[0] if incomplete else None), []
        completed_ids = {str(order.id) for order in completed}
        parent_by_child = {}
        parent_ids = []
        for order in completed:
            predecessors = order.protective_context_json.get("predecessor_order_ids", [])
            if not predecessors:
                continue
            predecessor_id = predecessors[0]
            if predecessor_id not in completed_ids:
                raise ExecutionConflictError("protective generation topology is invalid")
            parent_by_child[str(order.id)] = predecessor_id
            parent_ids.append(predecessor_id)
        heads = completed_ids - set(parent_ids)
        roots = completed_ids - set(parent_by_child)
        if len(parent_ids) != len(set(parent_ids)) or len(heads) != 1 or len(roots) != 1:
            raise ExecutionConflictError("protective generation topology is invalid")
        head = heads.pop()
        visited = set()
        current = head
        while current in parent_by_child and current not in visited:
            visited.add(current)
            current = parent_by_child[current]
        visited.add(current)
        if visited != completed_ids:
            raise ExecutionConflictError("protective generation topology is invalid")
        if incomplete:
            predecessors = incomplete[0].protective_context_json.get("predecessor_order_ids", [])
            if predecessors != [head]:
                raise ExecutionConflictError("protective generation topology is invalid")
        return (incomplete[0] if incomplete else None), [head]

    def validate_order(self, order, *, lock_fills=True):
        """Revalidate a stored protective intent before attempt or projection."""
        if order.order_origin not in PROTECTIVE_ORIGINS:
            raise ExecutionConflictError("protective order origin is not exact-allowlisted")
        if order.broker_mode != "paper" or order.signal_id is not None:
            raise ExecutionConflictError("protective order must be paper and non-signal")
        intent = order.intent_json
        if (
            not isinstance(intent, dict)
            or order.intent_sha256 != content_sha256(intent)
            or not isinstance(intent.get("frozen_intent"), dict)
            or not isinstance(intent.get("economics"), dict)
        ):
            raise ExecutionConflictError("protective durable content is invalid")
        context = order.protective_context_json
        try:
            parsed = ProtectiveContext.parse(context)
        except ProtectiveContextError as error:
            if error.generation:
                raise ExecutionConflictError("protective generation provenance is invalid") from error
            raise ExecutionConflictError("protective durable content is invalid") from error
        legacy = parsed.legacy
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
        source_field = "source_holding_ids" if legacy else "source_lot_ids"
        if parsed.successor:
            predecessor_id = uuid.UUID(parsed.predecessor_order_ids[0])
            if order.id == predecessor_id:
                raise ExecutionConflictError("protective generation provenance is invalid")
            predecessor_orders = self.session.scalars(select(OrderRecord).where(OrderRecord.id == predecessor_id)).all()
            if [row.id for row in predecessor_orders] != [predecessor_id]:
                raise ExecutionConflictError("protective generation predecessor set changed")
            for predecessor in predecessor_orders:
                predecessor_context = predecessor.protective_context_json
                if (
                    (
                        predecessor.account_scope,
                        predecessor.account_generation,
                        predecessor.market,
                        predecessor.symbol,
                        predecessor.instrument,
                        predecessor.side,
                        predecessor.order_origin,
                    )
                    != (
                        order.account_scope,
                        order.account_generation,
                        order.market,
                        order.symbol,
                        order.instrument,
                        order.side,
                        order.order_origin,
                    )
                    or not isinstance(predecessor_context, dict)
                    or predecessor_context.get("trigger_generation") != context["trigger_generation"]
                    or predecessor_context.get(source_field) != context[source_field]
                ):
                    raise ExecutionConflictError("protective generation predecessor identity changed")
                self.validate_order(predecessor, lock_fills=lock_fills)
                statement = select(OrderFillRecord).where(OrderFillRecord.order_id == predecessor.id)
                if lock_fills:
                    statement = statement.with_for_update()
                fills = self.session.scalars(statement).all()
                if not self._generation_complete(predecessor, fills):
                    raise ExecutionConflictError("protective generation predecessor is not complete")
        if legacy:
            self._validate_legacy_order(order, context, lock_fills=lock_fills)
            return
        source_lot_ids = [uuid.UUID(value) for value in parsed.source_lot_ids]
        source_decision_ids = [uuid.UUID(value) for value in parsed.source_decision_ids]
        if order.decision_id != min(source_decision_ids, key=str):
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
            self.source_decision_policy(decision_id, order.account_scope, order.account_generation, order.market)
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
        outstanding, cash = outstanding_reservation(self.session, order, currency, lock_fills=lock_fills)
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
        if parsed.successor:
            expected_event_payload.update(
                predecessor_order_ids=context["predecessor_order_ids"],
                remaining_source_quantities=context["remaining_source_quantities"],
            )
        if event is None or event.payload_json != expected_event_payload:
            raise ExecutionConflictError("protective audit event is missing or changed")

    def _validate_legacy_order(self, order, context, *, lock_fills):
        source_ids = [uuid.UUID(value) for value in context["source_holding_ids"]]
        quantities = context["source_holding_quantities"]
        risk = context["source_holding_risk"]
        legacy_context_sha256 = content_sha256({"source_holding_quantities": quantities, "source_holding_risk": risk})
        if (
            order.decision_id is not None
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
            amount = float(quantities[str(holding.id)])
            frozen_risk = risk[str(holding.id)]
            entry_price = float(frozen_risk["entry_price"])
            stop_loss_pct = float(frozen_risk["stop_loss_pct"])
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
        outstanding, cash = outstanding_reservation(self.session, order, account.currency, lock_fills=lock_fills)
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
        requested_source_ids = sorted(requested_ids)
        replay, predecessor_order_ids = self._generation_replay(
            account_scope=account_scope,
            account_generation=account_generation,
            market=market,
            symbol=symbol,
            instrument=instrument,
            side=side,
            origin=origin,
            trigger_generation=trigger_generation,
            source_field="source_holding_ids",
            source_ids=requested_source_ids,
        )
        if replay is not None:
            return self._response_for(replay)
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
                outstanding, _cash = outstanding_reservation(self.session, sibling, account.currency)
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
        if source_holding_ids != requested_source_ids:
            predecessor_order_ids = []
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
        generation_context = (
            {
                "predecessor_order_ids": predecessor_order_ids,
                "remaining_source_quantities": source_holding_quantities,
            }
            if predecessor_order_ids
            else {}
        )
        dedupe_input = {
            "account_scope": account_scope,
            "account_generation": account_generation,
            "origin": origin,
            "source_holding_ids": source_holding_ids,
            "trigger_generation": trigger_generation,
            "legacy_context_sha256": legacy_context_sha256,
            **generation_context,
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
        current_time = None if now is None else timestamp(now, "now")
        current_price = _finite_positive(price, "protective price")
        lock_account(self.session, account_scope, account_generation)
        if current_time is None:
            current_time = timestamp(datetime.now(UTC), "now")
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
            self.source_decision_policy(uuid.UUID(decision_id), account_scope, account_generation, market)
            for decision_id in source_decision_ids
        ]
        currency = policies[0][1].reconciliation.currency
        if account.currency != currency or any(policy.reconciliation.currency != currency for _, policy in policies):
            raise ExecutionConflictError("protective account terms disagree with source policy")
        source_lot_ids = [str(lot.id) for lot in lots]
        replay, predecessor_order_ids = self._generation_replay(
            account_scope=account_scope,
            account_generation=account_generation,
            market=market,
            symbol=symbol,
            instrument=instrument,
            side=side,
            origin=origin,
            trigger_generation=trigger_generation,
            source_field="source_lot_ids",
            source_ids=source_lot_ids,
        )
        if replay is not None:
            return self._response_for(replay)
        available_by_lot = {
            str(lot.id): Decimal(str(lot.open_quantity)) - Decimal(str(lot.reserved_close_quantity)) for lot in lots
        }
        generation_context = (
            {
                "predecessor_order_ids": predecessor_order_ids,
                "remaining_source_quantities": {lot_id: float(available_by_lot[lot_id]) for lot_id in source_lot_ids},
            }
            if predecessor_order_ids
            else {}
        )
        dedupe_input = {
            "account_scope": account_scope,
            "account_generation": account_generation,
            "origin": origin,
            "source_lot_ids": source_lot_ids,
            "trigger_generation": trigger_generation,
            **generation_context,
        }
        digest = content_sha256(dedupe_input)
        execution_key = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:execution:{digest}")
        order_id = uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:order:{digest}")
        existing = self.session.get(OrderRecord, order_id)
        if existing is not None:
            self.validate_order(existing)
            return self._response_for(existing)
        available = sum(available_by_lot.values(), Decimal(0))
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
                    **generation_context,
                },
            )
        )
        self.session.flush()
        return self._response_for(order)
