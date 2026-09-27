"""Paper trading broker adapter with durable decision-order recovery."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select

from poseidon.broker.base import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCapabilities,
    BrokerCapabilityError,
    BrokerFillSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
)
from poseidon.decision_loop.manifest import content_sha256
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS, Fill, Order

PAPER_RECONCILIATION_CAPABILITIES = BrokerCapabilities(
    stable_client_reference=True,
    find_order_by_client_reference=True,
    query_order=True,
    query_fills=True,
    query_account_snapshot=True,
)
LEGACY_BASELINE_ACTION = "position_base"
LEGACY_BASELINE_STATUS = "position_baseline"
LEGACY_BASELINE_REF_PREFIX = "LEGACY-BASE-"
LEGACY_EXECUTION_PREFIX = "PAPER-LEGACY"
LEGACY_ATTRIBUTION_ACTION = "position_alloc"
LEGACY_ATTRIBUTION_STATUS = "position_attribution"
LEGACY_ATTRIBUTION_REF_PREFIX = "LEGACY-ALLOC-"
LEGACY_ATTRIBUTION_BROKER_PREFIX = "PAPER-ALLOC"


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _order_snapshot(record: PaperBrokerOrder) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        broker_order_id=record.broker_order_id,
        client_order_ref=record.client_order_ref,
        account_scope=record.account_scope,
        account_generation=record.account_generation,
        market=record.market,
        symbol=record.symbol,
        instrument=record.instrument,
        action=record.action,
        side=record.side,
        order_type=record.order_type,
        quantity=record.quantity,
        price=record.price,
        status=record.status,
        accepted_at=_utc(record.accepted_at),
        state_version=record.state_version,
    )


def _fill_snapshot(record: PaperBrokerFill, broker_order_id: str) -> BrokerFillSnapshot:
    return BrokerFillSnapshot(
        broker_order_id=broker_order_id,
        broker_fill_id=record.broker_fill_id,
        account_scope=record.account_scope,
        account_generation=record.account_generation,
        market=record.market,
        symbol=record.symbol,
        instrument=record.instrument,
        side=record.side,
        fill_price=record.fill_price,
        fill_quantity=record.fill_quantity,
        fill_time=_utc(record.fill_time),
        state_version=record.state_version,
    )


def _validate_owner_account(account: PaperBrokerAccount | None) -> PaperBrokerAccount:
    if account is None:
        raise BrokerCapabilityError("owner-frozen paper account generation is missing")
    if not account.currency.strip() or not math.isfinite(account.opening_cash) or account.opening_cash < 0:
        raise BrokerCapabilityError("owner-frozen opening cash and currency are invalid")
    return account


def _intent_action(order: Order) -> str:
    action = (order.intent_json or {}).get("frozen_intent", {}).get("action")
    if action not in {"enter", "add", "reduce", "exit"}:
        raise BrokerCapabilityError("decision order requires a frozen intent action")
    return action


def _broker_order_id(prefix: str, order: Order, client_order_ref: str) -> str:
    digest = content_sha256({"client_order_ref": client_order_ref, "intent_sha256": order.intent_sha256})
    return f"{prefix}-{digest[:40]}"


def _same_order(record: PaperBrokerOrder, order: Order, expected_broker_order_id: str) -> bool:
    return (
        record.broker_order_id,
        record.market,
        record.symbol,
        record.instrument,
        record.action,
        record.side,
        record.order_type,
        record.quantity,
    ) == (
        expected_broker_order_id,
        order.market,
        order.symbol,
        order.instrument,
        order.action,
        order.side,
        order.order_type,
        order.quantity,
    )


def _existing_replay(session_factory, order: Order, client_order_ref: str, broker_prefix: str):
    session = session_factory()
    try:
        record = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == order.account_scope,
                PaperBrokerOrder.account_generation == order.account_generation,
                PaperBrokerOrder.client_order_ref == client_order_ref,
            )
        )
        if record is None:
            return None
        expected_id = _broker_order_id(broker_prefix, order, client_order_ref)
        if not _same_order(record, order, expected_id):
            raise BrokerCapabilityError("client reference replay changed the frozen intent fingerprint")
        return _order_snapshot(record)
    finally:
        session.close()


def _legacy_context(order: Order):
    context = order.protective_context_json
    return context if isinstance(context, dict) and context.get("legacy_exception") is True else None


def _baseline_broker_id(row: PaperBrokerOrder) -> str:
    digest = content_sha256(
        {
            "account_scope": row.account_scope,
            "account_generation": row.account_generation,
            "client_order_ref": row.client_order_ref,
            "market": row.market,
            "symbol": row.symbol,
            "instrument": row.instrument,
            "side": row.side,
            "quantity": row.quantity,
            "price": row.price,
        }
    )
    return f"PAPER-BASE-{digest[:40]}"


def _control_broker_id(prefix: str, row: PaperBrokerOrder) -> str:
    digest = content_sha256(
        {
            "account_scope": row.account_scope,
            "account_generation": row.account_generation,
            "client_order_ref": row.client_order_ref,
            "market": row.market,
            "symbol": row.symbol,
            "instrument": row.instrument,
            "action": row.action,
            "side": row.side,
            "order_type": row.order_type,
            "quantity": row.quantity,
            "price": row.price,
            "status": row.status,
            "state_version": row.state_version,
        }
    )
    return f"{prefix}-{digest[:40]}"


def _baseline_holding_id(row: PaperBrokerOrder) -> uuid.UUID:
    try:
        value = row.client_order_ref.removeprefix(LEGACY_BASELINE_REF_PREFIX)
        holding_id = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise BrokerCapabilityError("legacy position baseline identity is invalid") from error
    if not row.client_order_ref.startswith(LEGACY_BASELINE_REF_PREFIX) or str(holding_id) != value:
        raise BrokerCapabilityError("legacy position baseline identity is invalid")
    return holding_id


def _attribution_identity(row: PaperBrokerOrder) -> tuple[uuid.UUID, str]:
    try:
        value = row.client_order_ref.removeprefix(LEGACY_ATTRIBUTION_REF_PREFIX)
        holding_value, execution_ref = value.split(":", 1)
        holding_id = uuid.UUID(holding_value)
    except (AttributeError, ValueError) as error:
        raise BrokerCapabilityError("legacy close attribution identity is invalid") from error
    if (
        not row.client_order_ref.startswith(LEGACY_ATTRIBUTION_REF_PREFIX)
        or str(holding_id) != holding_value
        or len(execution_ref) != 67
        or not execution_ref.startswith("PX-")
        or any(character not in "0123456789abcdef" for character in execution_ref[3:])
    ):
        raise BrokerCapabilityError("legacy close attribution identity is invalid")
    return holding_id, execution_ref


def _validated_baselines(
    session, account_scope, account_generation, *, market=None, symbol=None, instrument=None, side=None
):
    statement = select(PaperBrokerOrder).where(
        PaperBrokerOrder.account_scope == account_scope,
        PaperBrokerOrder.account_generation == account_generation,
        PaperBrokerOrder.action == LEGACY_BASELINE_ACTION,
    )
    for field, value in (("market", market), ("symbol", symbol), ("instrument", instrument), ("side", side)):
        if value is not None:
            statement = statement.where(getattr(PaperBrokerOrder, field) == value)
    rows = session.scalars(statement.order_by(PaperBrokerOrder.state_version, PaperBrokerOrder.client_order_ref)).all()
    for row in rows:
        _baseline_holding_id(row)
        if (
            row.status != LEGACY_BASELINE_STATUS
            or row.order_type != "market"
            or row.broker_order_id != _baseline_broker_id(row)
            or not math.isfinite(row.quantity)
            or row.quantity <= 0
            or row.price is None
            or not math.isfinite(row.price)
            or row.price <= 0
            or row.state_version <= 0
            or session.scalar(
                select(PaperBrokerFill.id).where(PaperBrokerFill.paper_broker_order_id == row.id).limit(1)
            )
            is not None
        ):
            raise BrokerCapabilityError("legacy position baseline is invalid")
    return rows


def _validated_attributions(
    session, account_scope, account_generation, *, market=None, symbol=None, instrument=None, side=None
):
    statement = select(PaperBrokerOrder).where(
        PaperBrokerOrder.account_scope == account_scope,
        PaperBrokerOrder.account_generation == account_generation,
        PaperBrokerOrder.action == LEGACY_ATTRIBUTION_ACTION,
    )
    for field, value in (("market", market), ("symbol", symbol), ("instrument", instrument), ("side", side)):
        if value is not None:
            statement = statement.where(getattr(PaperBrokerOrder, field) == value)
    rows = session.scalars(statement.order_by(PaperBrokerOrder.state_version, PaperBrokerOrder.client_order_ref)).all()
    baselines = {
        _baseline_holding_id(row): row
        for row in _validated_baselines(
            session,
            account_scope,
            account_generation,
            market=market,
            symbol=symbol,
            instrument=instrument,
            side=side,
        )
    }
    execution_statement = select(PaperBrokerOrder).where(
        PaperBrokerOrder.account_scope == account_scope,
        PaperBrokerOrder.account_generation == account_generation,
        PaperBrokerOrder.broker_order_id.like(f"{LEGACY_EXECUTION_PREFIX}-%"),
    )
    for field, value in (("market", market), ("symbol", symbol), ("instrument", instrument), ("side", side)):
        if value is not None:
            execution_statement = execution_statement.where(getattr(PaperBrokerOrder, field) == value)
    executions = {row.client_order_ref: row for row in session.scalars(execution_statement).all()}
    for execution in executions.values():
        expected_action = "sell" if execution.side == "long" else "buy"
        if (
            len(execution.client_order_ref) != 67
            or not execution.client_order_ref.startswith("PX-")
            or any(character not in "0123456789abcdef" for character in execution.client_order_ref[3:])
            or len(execution.broker_order_id) != len(f"{LEGACY_EXECUTION_PREFIX}-") + 40
            or not execution.broker_order_id.startswith(f"{LEGACY_EXECUTION_PREFIX}-")
            or any(
                character not in "0123456789abcdef"
                for character in execution.broker_order_id[len(f"{LEGACY_EXECUTION_PREFIX}-") :]
            )
            or execution.action != expected_action
            or execution.order_type != "market"
            or execution.status not in {"filled", "cancelled"}
        ):
            raise BrokerCapabilityError("legacy close execution is invalid")
    groups: dict[str, list[PaperBrokerOrder]] = {}
    for row in rows:
        holding_id, execution_ref = _attribution_identity(row)
        execution = executions.get(execution_ref)
        baseline = baselines.get(holding_id)
        if (
            row.status != LEGACY_ATTRIBUTION_STATUS
            or row.order_type != "market"
            or row.broker_order_id != _control_broker_id(LEGACY_ATTRIBUTION_BROKER_PREFIX, row)
            or execution is None
            or baseline is None
            or row.state_version != execution.state_version
            or _utc(row.accepted_at) != _utc(execution.accepted_at)
            or any(
                getattr(row, field) != getattr(execution, field)
                for field in ("account_scope", "account_generation", "market", "symbol", "instrument", "side")
            )
            or not math.isfinite(row.quantity)
            or row.quantity <= 0
            or row.price is None
            or not math.isfinite(row.price)
            or row.price <= 0
            or Decimal(str(row.price)) != Decimal(str(baseline.price))
            or session.scalar(
                select(PaperBrokerFill.id).where(PaperBrokerFill.paper_broker_order_id == row.id).limit(1)
            )
            is not None
        ):
            raise BrokerCapabilityError("legacy close attribution is invalid")
        groups.setdefault(execution_ref, []).append(row)
    for execution_ref, group in groups.items():
        execution = executions[execution_ref]
        holding_ids = [_attribution_identity(row)[0] for row in group]
        filled = session.scalar(
            select(func.coalesce(func.sum(PaperBrokerFill.fill_quantity), 0.0)).where(
                PaperBrokerFill.paper_broker_order_id == execution.id
            )
        )
        if (
            len(holding_ids) != len(set(holding_ids))
            or sum((Decimal(str(row.quantity)) for row in group), Decimal("0")) != Decimal(str(execution.quantity))
            or Decimal(str(filled)) > Decimal(str(execution.quantity))
        ):
            raise BrokerCapabilityError("legacy close attribution group is invalid")
    if set(groups) != set(executions):
        raise BrokerCapabilityError("legacy close execution lacks exact attribution")
    return rows


def _attributed_consumption(session, attributions, *, exclude_execution_ref=None):
    usage: dict[uuid.UUID, Decimal] = {}
    groups: dict[str, list[PaperBrokerOrder]] = {}
    for row in attributions:
        _holding_id, execution_ref = _attribution_identity(row)
        if execution_ref != exclude_execution_ref:
            groups.setdefault(execution_ref, []).append(row)
    for execution_ref, group in groups.items():
        execution = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == group[0].account_scope,
                PaperBrokerOrder.account_generation == group[0].account_generation,
                PaperBrokerOrder.client_order_ref == execution_ref,
            )
        )
        remaining = Decimal(
            str(
                session.scalar(
                    select(func.coalesce(func.sum(PaperBrokerFill.fill_quantity), 0.0)).where(
                        PaperBrokerFill.paper_broker_order_id == execution.id
                    )
                )
            )
        )
        for row in sorted(group, key=lambda item: item.client_order_ref):
            holding_id, _execution_ref = _attribution_identity(row)
            consumed = min(Decimal(str(row.quantity)), remaining)
            usage[holding_id] = usage.get(holding_id, Decimal("0")) + consumed
            remaining -= consumed
        if remaining:
            raise BrokerCapabilityError("legacy close attribution is smaller than execution fills")
    return usage


def _ensure_legacy_baselines(
    session,
    account,
    order: Order,
    client_order_ref: str,
    *,
    allow_create: bool,
) -> None:
    context = _legacy_context(order)
    if context is None:
        return
    expected_context = {
        "account_scope",
        "account_generation",
        "identity",
        "origin",
        "trigger_generation",
        "source_lot_ids",
        "source_decision_ids",
        "source_holding_ids",
        "source_holding_quantities",
        "source_holding_risk",
        "legacy_context_sha256",
        "legacy_exception",
        "dedupe_sha256",
    }
    source_ids = context.get("source_holding_ids")
    quantities = context.get("source_holding_quantities")
    risks = context.get("source_holding_risk")
    try:
        valid_ids = (
            isinstance(source_ids, list)
            and bool(source_ids)
            and source_ids == sorted(source_ids)
            and len(source_ids) == len(set(source_ids))
            and all(str(uuid.UUID(value)) == value for value in source_ids)
        )
        valid_maps = (
            isinstance(quantities, dict)
            and set(quantities) == set(source_ids or ())
            and isinstance(risks, dict)
            and set(risks) == set(source_ids or ())
        )
        total = Decimal("0")
        if valid_maps:
            for holding_id in source_ids:
                quantity = quantities[holding_id]
                risk = risks[holding_id]
                if (
                    isinstance(quantity, bool)
                    or not isinstance(quantity, (int, float))
                    or not math.isfinite(quantity)
                    or quantity <= 0
                    or not isinstance(risk, dict)
                    or set(risk) != {"entry_price", "stop_loss_pct"}
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                        for value in risk.values()
                    )
                ):
                    valid_maps = False
                    break
                total += Decimal(str(quantity))
    except (TypeError, ValueError):
        valid_ids = valid_maps = False
        total = Decimal("0")
    try:
        legacy_context_sha256 = content_sha256({"source_holding_quantities": quantities, "source_holding_risk": risks})
        dedupe_sha256 = content_sha256(
            {
                "account_scope": context.get("account_scope"),
                "account_generation": context.get("account_generation"),
                "origin": order.order_origin,
                "source_holding_ids": source_ids,
                "trigger_generation": context.get("trigger_generation"),
                "legacy_context_sha256": legacy_context_sha256,
            }
        )
    except (TypeError, ValueError) as error:
        raise BrokerCapabilityError("legacy protective context is not canonical") from error
    if (
        set(context) != expected_context
        or not valid_ids
        or not valid_maps
        or context["source_lot_ids"] != []
        or context["source_decision_ids"] != []
        or Decimal(str(order.quantity)) != total
        or context["legacy_context_sha256"] != legacy_context_sha256
        or (order.intent_json or {}).get("economics", {}).get("legacy_context_sha256") != legacy_context_sha256
        or context["dedupe_sha256"] != dedupe_sha256
    ):
        raise BrokerCapabilityError("legacy protective context requires exact positive full-close sources")
    refs = {holding_id: f"{LEGACY_BASELINE_REF_PREFIX}{holding_id}" for holding_id in context["source_holding_ids"]}
    baselines = _validated_baselines(
        session,
        order.account_scope,
        order.account_generation,
        market=order.market,
        symbol=order.symbol,
        instrument=order.instrument,
        side=order.side,
    )
    by_ref = {row.client_order_ref: row for row in baselines}
    missing = [holding_id for holding_id, ref in refs.items() if ref not in by_ref]
    if missing:
        if not allow_create:
            raise BrokerCapabilityError("legacy position baseline is missing on replay")
        state_version = account.state_version + 1
        accepted_at = datetime.now(UTC)
        for holding_id in missing:
            risk = context["source_holding_risk"][holding_id]
            row = PaperBrokerOrder(
                account_scope=order.account_scope,
                account_generation=order.account_generation,
                client_order_ref=refs[holding_id],
                broker_order_id="pending",
                market=order.market,
                symbol=order.symbol,
                instrument=order.instrument,
                action=LEGACY_BASELINE_ACTION,
                side=order.side,
                order_type="market",
                quantity=context["source_holding_quantities"][holding_id],
                price=risk["entry_price"],
                status=LEGACY_BASELINE_STATUS,
                state_version=state_version,
                accepted_at=accepted_at,
            )
            row.broker_order_id = _baseline_broker_id(row)
            session.add(row)
        account.state_version = state_version
        account.updated_at = accepted_at
        session.flush()
        baselines = _validated_baselines(
            session,
            order.account_scope,
            order.account_generation,
            market=order.market,
            symbol=order.symbol,
            instrument=order.instrument,
            side=order.side,
        )
        by_ref = {row.client_order_ref: row for row in baselines}
    attributions = _validated_attributions(
        session,
        order.account_scope,
        order.account_generation,
        market=order.market,
        symbol=order.symbol,
        instrument=order.instrument,
        side=order.side,
    )
    consumed = _attributed_consumption(session, attributions, exclude_execution_ref=client_order_ref)
    for holding_id, ref in refs.items():
        row = by_ref.get(ref)
        risk = context["source_holding_risk"][holding_id]
        holding_uuid = uuid.UUID(holding_id)
        if (
            row is None
            or Decimal(str(context["source_holding_quantities"][holding_id]))
            != Decimal(str(row.quantity)) - consumed.get(holding_uuid, Decimal("0"))
            or Decimal(str(risk["entry_price"])) != Decimal(str(row.price))
        ):
            raise BrokerCapabilityError("legacy position baseline changed")
    current = [row for row in attributions if _attribution_identity(row)[1] == client_order_ref]
    if not allow_create:
        planned = {
            str(_attribution_identity(row)[0]): (Decimal(str(row.quantity)), Decimal(str(row.price))) for row in current
        }
        expected = {
            holding_id: (
                Decimal(str(context["source_holding_quantities"][holding_id])),
                Decimal(str(context["source_holding_risk"][holding_id]["entry_price"])),
            )
            for holding_id in source_ids
        }
        if planned != expected:
            raise BrokerCapabilityError("legacy close attribution changed on replay")


def _position_state_for_identity(
    session,
    account_scope,
    account_generation,
    market,
    symbol,
    instrument,
    side,
    *,
    state_version=None,
) -> tuple[Decimal, Decimal]:
    baselines = _validated_baselines(
        session,
        account_scope,
        account_generation,
        market=market,
        symbol=symbol,
        instrument=instrument,
        side=side,
    )
    attributions = _validated_attributions(
        session,
        account_scope,
        account_generation,
        market=market,
        symbol=symbol,
        instrument=instrument,
        side=side,
    )
    if state_version is not None and any(row.state_version > state_version for row in [*baselines, *attributions]):
        raise BrokerCapabilityError("legacy position control ledger exceeds account watermark")
    attribution_groups: dict[str, list[PaperBrokerOrder]] = {}
    attribution_remaining: dict[uuid.UUID, Decimal] = {}
    baseline_prices = {_baseline_holding_id(row): Decimal(str(row.price)) for row in baselines}
    holding_remaining: dict[uuid.UUID, Decimal] = {}
    for row in attributions:
        holding_id, execution_ref = _attribution_identity(row)
        attribution_groups.setdefault(execution_ref, []).append(row)
        attribution_remaining[row.id] = Decimal(str(row.quantity))
    rows = session.execute(
        select(PaperBrokerOrder, PaperBrokerFill)
        .join(PaperBrokerFill, PaperBrokerFill.paper_broker_order_id == PaperBrokerOrder.id)
        .where(
            PaperBrokerOrder.account_scope == account_scope,
            PaperBrokerOrder.account_generation == account_generation,
            PaperBrokerOrder.market == market,
            PaperBrokerOrder.symbol == symbol,
            PaperBrokerOrder.instrument == instrument,
            PaperBrokerOrder.side == side,
            *(() if state_version is None else (PaperBrokerFill.state_version <= state_version,)),
        )
        .order_by(PaperBrokerFill.state_version, PaperBrokerFill.id)
    ).all()
    events = [(row.state_version, 0, row.client_order_ref, row, None) for row in baselines]
    events.extend((fill.state_version, 1, str(fill.id), broker_order, fill) for broker_order, fill in rows)
    quantity = Decimal("0")
    entry_cost = Decimal("0")
    opening_action = "buy" if side == "long" else "sell"
    for _version, priority, _key, broker_order, fill in sorted(events, key=lambda event: event[:3]):
        if priority == 0:
            holding_id = _baseline_holding_id(broker_order)
            baseline_quantity = Decimal(str(broker_order.quantity))
            if holding_id in holding_remaining:
                raise BrokerCapabilityError("independent paper broker ledger contains a duplicate baseline")
            holding_remaining[holding_id] = baseline_quantity
            quantity += baseline_quantity
            entry_cost += baseline_quantity * Decimal(str(broker_order.price))
            continue
        fill_quantity = Decimal(str(fill.fill_quantity))
        group = attribution_groups.get(broker_order.client_order_ref)
        if group is not None:
            remaining_fill = fill_quantity
            attributed_cost = Decimal("0")
            for attribution in sorted(group, key=lambda item: item.client_order_ref):
                holding_id, _execution_ref = _attribution_identity(attribution)
                available = attribution_remaining[attribution.id]
                consumed = min(available, remaining_fill)
                if consumed > holding_remaining.get(holding_id, Decimal("0")):
                    raise BrokerCapabilityError("legacy close exceeds its attributed holding")
                attribution_remaining[attribution.id] -= consumed
                holding_remaining[holding_id] -= consumed
                attributed_cost += baseline_prices[holding_id] * consumed
                remaining_fill -= consumed
            if remaining_fill:
                raise BrokerCapabilityError("legacy close exceeds its frozen attribution")
            if fill_quantity > quantity:
                raise BrokerCapabilityError("independent paper broker ledger contains a negative position")
            quantity -= fill_quantity
            entry_cost -= attributed_cost
            continue
        if broker_order.action == opening_action:
            quantity += fill_quantity
            entry_cost += Decimal(str(fill.fill_price)) * fill_quantity
            continue
        if fill_quantity > quantity:
            raise BrokerCapabilityError("independent paper broker ledger contains a negative position")
        average_entry = entry_cost / quantity
        entry_cost -= average_entry * fill_quantity
        quantity -= fill_quantity
    return quantity, entry_cost


def _position_state(session, order: Order) -> tuple[Decimal, Decimal]:
    return _position_state_for_identity(
        session,
        order.account_scope,
        order.account_generation,
        order.market,
        order.symbol,
        order.instrument,
        order.side,
    )


def _validate_action(order: Order, intent_action: str) -> None:
    opening = intent_action in {"enter", "add"}
    expected = ("buy" if order.side == "long" else "sell") if opening else ("sell" if order.side == "long" else "buy")
    if order.action != expected:
        raise BrokerCapabilityError("frozen intent action, broker action, and side disagree")


def _accept_decision_order(
    session_factory,
    order: Order,
    client_order_ref: str,
    fill_price: float,
    *,
    broker_prefix: str,
    contract_multiplier: float = 1.0,
) -> BrokerOrderSnapshot:
    """Commit one independent paper acceptance before returning to the caller."""
    if not math.isfinite(fill_price) or fill_price <= 0:
        raise BrokerCapabilityError("paper fill price must be positive and finite")
    if not math.isfinite(order.quantity) or order.quantity <= 0:
        raise BrokerCapabilityError("paper order quantity must be positive and finite")
    intent_action = _intent_action(order)
    _validate_action(order, intent_action)
    expected_broker_order_id = _broker_order_id(broker_prefix, order, client_order_ref)
    session = session_factory()
    try:
        account = _validate_owner_account(
            session.scalar(
                select(PaperBrokerAccount)
                .where(
                    PaperBrokerAccount.account_scope == order.account_scope,
                    PaperBrokerAccount.account_generation == order.account_generation,
                )
                .with_for_update()
            )
        )
        existing = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == order.account_scope,
                PaperBrokerOrder.account_generation == order.account_generation,
                PaperBrokerOrder.client_order_ref == client_order_ref,
            )
        )
        _ensure_legacy_baselines(
            session,
            account,
            order,
            client_order_ref,
            allow_create=existing is None,
        )
        if existing is not None:
            if not _same_order(existing, order, expected_broker_order_id):
                raise BrokerCapabilityError("client reference replay changed the frozen intent fingerprint")
            return _order_snapshot(existing)

        current_cash = Decimal(str(account.opening_cash)) + sum(
            (
                Decimal(str(amount))
                for amount in session.scalars(
                    select(PaperCashMovement.amount).where(
                        PaperCashMovement.account_scope == order.account_scope,
                        PaperCashMovement.account_generation == order.account_generation,
                        PaperCashMovement.currency == account.currency,
                    )
                ).all()
            ),
            start=Decimal("0"),
        )
        position_quantity, entry_cost = _position_state(session, order)
        quantity = Decimal(str(order.quantity))
        price = Decimal(str(fill_price))
        multiplier = Decimal(str(contract_multiplier))
        notional = price * quantity * multiplier
        if intent_action in {"enter", "add"}:
            cash_delta = -notional
        else:
            if quantity > position_quantity:
                raise BrokerCapabilityError("close exceeds independent broker inventory")
            if order.side == "long":
                cash_delta = notional
            else:
                legacy_context = _legacy_context(order)
                if legacy_context is None:
                    average_entry = entry_cost / position_quantity
                else:
                    attributed_cost = sum(
                        (
                            Decimal(str(legacy_context["source_holding_quantities"][holding_id]))
                            * Decimal(str(legacy_context["source_holding_risk"][holding_id]["entry_price"]))
                            for holding_id in legacy_context["source_holding_ids"]
                        ),
                        Decimal("0"),
                    )
                    average_entry = attributed_cost / quantity
                cash_delta = (Decimal("2") * average_entry - price) * quantity * multiplier
        if current_cash + cash_delta < 0:
            raise BrokerCapabilityError("paper order exceeds durable account cash")

        accepted_at = datetime.now(UTC)
        state_version = account.state_version + 1
        broker_order_id = expected_broker_order_id
        broker_order = PaperBrokerOrder(
            account_scope=order.account_scope,
            account_generation=order.account_generation,
            client_order_ref=client_order_ref,
            broker_order_id=broker_order_id,
            market=order.market,
            symbol=order.symbol,
            instrument=order.instrument,
            action=order.action,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=fill_price,
            status="filled",
            state_version=state_version,
            accepted_at=accepted_at,
        )
        session.add(broker_order)
        session.flush()
        session.add(
            PaperBrokerFill(
                paper_broker_order_id=broker_order.id,
                account_scope=order.account_scope,
                account_generation=order.account_generation,
                broker_fill_id=f"{broker_order_id}-fill",
                market=order.market,
                symbol=order.symbol,
                instrument=order.instrument,
                side=order.side,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                fill_time=accepted_at,
                state_version=state_version,
            )
        )
        legacy_context = _legacy_context(order)
        if legacy_context is not None:
            for holding_id in legacy_context["source_holding_ids"]:
                attribution = PaperBrokerOrder(
                    account_scope=order.account_scope,
                    account_generation=order.account_generation,
                    client_order_ref=f"{LEGACY_ATTRIBUTION_REF_PREFIX}{holding_id}:{client_order_ref}",
                    broker_order_id="pending",
                    market=order.market,
                    symbol=order.symbol,
                    instrument=order.instrument,
                    action=LEGACY_ATTRIBUTION_ACTION,
                    side=order.side,
                    order_type="market",
                    quantity=legacy_context["source_holding_quantities"][holding_id],
                    price=legacy_context["source_holding_risk"][holding_id]["entry_price"],
                    status=LEGACY_ATTRIBUTION_STATUS,
                    state_version=state_version,
                    accepted_at=accepted_at,
                )
                attribution.broker_order_id = _control_broker_id(LEGACY_ATTRIBUTION_BROKER_PREFIX, attribution)
                session.add(attribution)
        if cash_delta:
            session.add(
                PaperCashMovement(
                    account_scope=order.account_scope,
                    account_generation=order.account_generation,
                    currency=account.currency,
                    amount=float(cash_delta),
                    movement_type="order_fill",
                    state_version=state_version,
                    occurred_at=accepted_at,
                )
            )
        account.state_version = state_version
        account.updated_at = accepted_at
        session.flush()
        if legacy_context is not None:
            current_attributions = [
                row
                for row in _validated_attributions(
                    session,
                    order.account_scope,
                    order.account_generation,
                    market=order.market,
                    symbol=order.symbol,
                    instrument=order.instrument,
                    side=order.side,
                )
                if _attribution_identity(row)[1] == client_order_ref
            ]
            if len(current_attributions) != len(legacy_context["source_holding_ids"]):
                raise BrokerCapabilityError("legacy close attribution was not durably recorded")
        snapshot = _order_snapshot(broker_order)
        session.commit()
        return snapshot
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class PaperBrokerAdapter(BrokerAdapter):
    """Stock paper adapter; decision recovery uses the independent DB ledger."""

    capabilities = PAPER_RECONCILIATION_CAPABILITIES

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._fills: dict[str, list[Fill]] = {}

    def login(self) -> bool:
        return True

    def place_order(self, order: Order, *, client_order_ref: str | None = None):
        """Use durable truth for decision orders and memory only for legacy calls."""
        durable = order.order_origin == "decision" or (
            order.order_origin in DURABLE_PROTECTIVE_ORIGINS and order.execution_key is not None
        )
        if durable and client_order_ref is None:
            raise BrokerCapabilityError("durable execution requires the stored client reference")
        if client_order_ref is not None:
            self._require_decision_submission(order, client_order_ref)
            if order.market != "tw_stock" or order.instrument != "spot":
                raise BrokerCapabilityError("stock paper execution requires tw_stock/spot")
            if _legacy_context(order) is None:
                replay = _existing_replay(self._session_factory, order, client_order_ref, "PAPER")
                if replay is not None:
                    return replay

        from poseidon.data.remote_repository import RemoteDataRepository

        repo = RemoteDataRepository.from_settings()
        latest = repo.read_ohlcv(order.symbol, order.market, "1d")
        if latest.empty:
            raise ValueError(f"No price data for {order.symbol}")
        fill_price = float(latest["close"].iloc[-1])
        if client_order_ref is not None:
            prefix = LEGACY_EXECUTION_PREFIX if _legacy_context(order) is not None else "PAPER"
            return _accept_decision_order(
                self._session_factory,
                order,
                client_order_ref,
                fill_price,
                broker_prefix=prefix,
            )

        broker_order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        self._fills[broker_order_id] = [
            Fill(
                order_id=order.id,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                fill_time=datetime.now(UTC),
                broker_fill_id=broker_order_id,
            )
        ]
        return broker_order_id

    def find_order_by_client_ref(self, client_order_ref, *, account_scope, account_generation):
        session = self._session_factory()
        try:
            record = session.scalar(
                select(PaperBrokerOrder).where(
                    PaperBrokerOrder.account_scope == account_scope,
                    PaperBrokerOrder.account_generation == account_generation,
                    PaperBrokerOrder.client_order_ref == client_order_ref,
                )
            )
            return _order_snapshot(record) if record is not None else None
        finally:
            session.close()

    def query_order(self, broker_order_id, *, account_scope, account_generation):
        session = self._session_factory()
        try:
            record = session.scalar(
                select(PaperBrokerOrder).where(
                    PaperBrokerOrder.account_scope == account_scope,
                    PaperBrokerOrder.account_generation == account_generation,
                    PaperBrokerOrder.broker_order_id == broker_order_id,
                )
            )
            return _order_snapshot(record) if record is not None else None
        finally:
            session.close()

    def query_fills(
        self,
        broker_order_id: str,
        *,
        account_scope: str | None = None,
        account_generation: str | None = None,
    ) -> list[Fill] | list[BrokerFillSnapshot]:
        if account_scope is None and account_generation is None:
            return self._fills.get(broker_order_id, [])
        if not account_scope or not account_generation:
            raise BrokerCapabilityError("durable fill lookup requires account identity")
        session = self._session_factory()
        try:
            order = session.scalar(
                select(PaperBrokerOrder).where(
                    PaperBrokerOrder.account_scope == account_scope,
                    PaperBrokerOrder.account_generation == account_generation,
                    PaperBrokerOrder.broker_order_id == broker_order_id,
                )
            )
            if order is None:
                return []
            records = session.scalars(
                select(PaperBrokerFill)
                .where(PaperBrokerFill.paper_broker_order_id == order.id)
                .order_by(PaperBrokerFill.fill_time, PaperBrokerFill.id)
            ).all()
            return [_fill_snapshot(record, broker_order_id) for record in records]
        finally:
            session.close()

    def query_account_snapshot(self, account_scope: str, account_generation: str) -> BrokerAccountSnapshot:
        session = self._session_factory()
        try:
            account = _validate_owner_account(
                session.scalar(
                    select(PaperBrokerAccount)
                    .where(
                        PaperBrokerAccount.account_scope == account_scope,
                        PaperBrokerAccount.account_generation == account_generation,
                    )
                    .with_for_update()
                )
            )
            movements = session.scalars(
                select(PaperCashMovement).where(
                    PaperCashMovement.account_scope == account_scope,
                    PaperCashMovement.account_generation == account_generation,
                    PaperCashMovement.currency == account.currency,
                    PaperCashMovement.state_version <= account.state_version,
                )
            ).all()
            rows = session.execute(
                select(PaperBrokerOrder, PaperBrokerFill)
                .join(PaperBrokerFill, PaperBrokerFill.paper_broker_order_id == PaperBrokerOrder.id)
                .where(
                    PaperBrokerOrder.account_scope == account_scope,
                    PaperBrokerOrder.account_generation == account_generation,
                    PaperBrokerFill.state_version <= account.state_version,
                )
            ).all()
            baselines = _validated_baselines(session, account_scope, account_generation)
            attributions = _validated_attributions(session, account_scope, account_generation)
            for baseline in baselines:
                if baseline.state_version > account.state_version:
                    raise BrokerCapabilityError("legacy position baseline exceeds account watermark")
            if any(row.state_version > account.state_version for row in attributions):
                raise BrokerCapabilityError("legacy position attribution exceeds account watermark")
            identities = {(row.market, row.symbol, row.instrument, row.side) for row in baselines}
            for _broker_order, fill in rows:
                identities.add((fill.market, fill.symbol, fill.instrument, fill.side))
            quantities = {
                identity: _position_state_for_identity(
                    session,
                    account_scope,
                    account_generation,
                    *identity,
                    state_version=account.state_version,
                )[0]
                for identity in identities
            }
            positions = tuple(
                BrokerPositionSnapshot(*identity, float(quantity))
                for identity, quantity in sorted(quantities.items())
                if quantity > 0
            )
            cash = Decimal(str(account.opening_cash)) + sum(
                (Decimal(str(movement.amount)) for movement in movements),
                start=Decimal("0"),
            )
            return BrokerAccountSnapshot(
                account_scope=account_scope,
                account_generation=account_generation,
                currency=account.currency,
                cash=float(cash),
                state_version=account.state_version,
                as_of=datetime.now(UTC),
                positions=positions,
            )
        finally:
            session.close()

    def query_positions(self) -> list[dict]:
        return []

    def logout(self) -> None:
        pass
