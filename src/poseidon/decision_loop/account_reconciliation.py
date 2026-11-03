"""Account reconciliation: compare a paper account with its broker snapshot and record the differences."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from poseidon.broker.base import (
    BrokerAccountSnapshot,
    BrokerCapabilityError,
)
from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.locks import lock_account
from poseidon.core.timeutil import as_utc, iso_time, timestamp
from poseidon.decision_loop.execution import DecisionExecutionService, internal_state_watermark
from poseidon.decision_loop.policy import DecisionPolicy
from poseidon.decision_loop.reconciliation import (
    ReconciliationConflictError,
    ReconciliationService,
    _difference,
    _normalized_status,
    _number,
)
from poseidon.decision_loop.reservations import ExecutionConflictError, outstanding_reservation
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion


def paper_liquidation_nav(snapshot, policy, prices, ledger, *, position_economics=None):
    """Value independent inventory at current marks, including short collateral."""
    if not isinstance(snapshot.positions, (tuple, list)):
        raise ValueError("paper snapshot requires complete positions")

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)) or not math.isfinite(value):
            raise ValueError("paper valuation requires finite numeric economics")
        return Decimal(str(value))

    inventory = {}
    multipliers = {}

    def multiplier_for(identity):
        market, _symbol, instrument, side = identity
        if side not in {"long", "short"}:
            raise ValueError("paper ledger has invalid side or action")
        multiplier = Decimal(1)
        if market == "crypto_perp":
            rule = policy.perp_instrument_rules.get(instrument)
            if rule is None or rule.margin_semantics != "full_notional" or rule.funding_semantics != "excluded":
                raise ValueError("unsupported paper collateral semantics")
            multiplier = Decimal(str(rule.contract_multiplier))
            if not multiplier.is_finite() or multiplier <= 0:
                raise ValueError("paper multiplier must be positive and finite")
        elif market != "tw_stock" or instrument != "spot":
            raise ValueError("unsupported paper position identity")
        return multiplier

    if position_economics is not None:
        if not isinstance(position_economics, dict):
            raise ValueError("paper position economics are invalid")
        for identity, economics in position_economics.items():
            if not isinstance(identity, tuple) or len(identity) != 4 or not isinstance(economics, tuple):
                raise ValueError("paper position economics are invalid")
            quantity, cost = map(number, economics)
            if quantity < 0 or cost < 0 or (quantity == 0) != (cost == 0):
                raise ValueError("paper position economics are invalid")
            inventory[identity] = quantity, cost
            multipliers[identity] = multiplier_for(identity)
    else:
        for order, fill in ledger:
            identity = (order.market, order.symbol, order.instrument, order.side)
            if order.action not in {"buy", "sell"}:
                raise ValueError("paper ledger has invalid side or action")
            multiplier = multiplier_for(identity)
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


class AccountReconciliationService:
    """Compare one paper account with its broker snapshot and record the result; the caller owns the transaction."""

    def __init__(self, session):
        self.session = session

    def _active_version_owner_policy(self, scope, generation, reasons):
        """Owner policy before the first decision exists (Phase 101 shadow soak).

        Shadow mode produces evaluation runs only, so no ``DecisionRecord`` can name the owner.
        The owner is then the frozen policy of the released ``StrategyVersion`` that authorizes
        this exact account scope and generation. Several active versions must agree on one
        policy digest; anything else stays unresolved, never guessed.
        """
        rows = self.session.execute(
            select(StrategyVersion)
            .join(StrategyRecord, StrategyRecord.active_version_id == StrategyVersion.id)
            .order_by(StrategyVersion.id)
        ).all()
        candidates = {}
        for (version,) in rows:
            policy_json = version.policy_json if isinstance(version.policy_json, dict) else {}
            frozen = policy_json.get("reconciliation")
            if (
                policy_json.get("account_scope") != scope
                or not isinstance(frozen, dict)
                or frozen.get("account_generation") != generation
            ):
                continue
            candidates.setdefault(content_sha256(policy_json), version)
        if not candidates:
            reasons.append("owner policy is missing for account generation")
            return None
        if len(candidates) > 1:
            reasons.append("owner policy is ambiguous across active strategy versions")
            return None
        policy_sha, version = next(iter(candidates.items()))
        try:
            version.verify_content()
            full_policy = DecisionPolicy.model_validate(version.policy_json)
        except (TypeError, ValueError, PydanticValidationError) as error:
            reasons.append(f"owner policy unresolved: {error}")
            return None
        if full_policy.reconciliation is None:
            reasons.append("owner policy unresolved: missing owner reconciliation policy")
            return None
        return version.policy_json["reconciliation"], full_policy.reconciliation, policy_sha

    def reconcile_account(self, account_id, snapshot, *, now=None, prices=None):
        """Compare independent ledgers and flush one immutable account result."""
        probe = self.session.get(PaperBrokerAccount, account_id)
        if probe is None:
            raise ReconciliationConflictError("paper account does not exist")
        scope, generation = probe.account_scope, probe.account_generation
        execution = DecisionExecutionService(self.session)
        lock_account(self.session, scope, generation)
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
        policy_sha = owner.policy_sha256 if owner is not None else "0" * 64
        if owner is None:
            resolved = self._active_version_owner_policy(scope, generation, reasons)
            if resolved is not None:
                frozen, policy, policy_sha = resolved
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
                "fill_time": iso_time(as_utc(row.fill_time), "fill_time"),
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
            if as_utc(snapshot.as_of) > as_of or (
                policy is not None
                and (as_of - as_utc(snapshot.as_of)).total_seconds() > policy.max_reconciliation_age_seconds
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
                            ReconciliationService(self.session)._validate_durable_intent(order)
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
                        short_groups.setdefault((identity, as_utc(fill.fill_time)), []).append((is_open, amount, price))
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
                        quantity, cash = outstanding_reservation(self.session, order, currency)
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
                from poseidon.broker.paper_adapter import validated_position_economics

                ledger = [(broker_by_id[row.paper_broker_order_id], row) for row in broker_fills]
                position_economics = validated_position_economics(
                    self.session,
                    scope,
                    generation,
                    snapshot.state_version,
                )
                nav = paper_liquidation_nav(
                    snapshot,
                    policy,
                    prices or {},
                    ledger,
                    position_economics=position_economics,
                )
            except (BrokerCapabilityError, ValueError, KeyError, ReconciliationConflictError) as error:
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
        return AccountReconciliationService(session).reconcile_account(persisted_id, snapshot, now=now, prices=prices)
