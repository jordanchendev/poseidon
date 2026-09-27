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
        try:
            uuid.UUID(row.client_order_ref.removeprefix(LEGACY_BASELINE_REF_PREFIX))
        except (AttributeError, ValueError) as error:
            raise BrokerCapabilityError("legacy position baseline identity is invalid") from error
        if (
            not row.client_order_ref.startswith(LEGACY_BASELINE_REF_PREFIX)
            or row.status != LEGACY_BASELINE_STATUS
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
    if (
        set(context) != expected_context
        or not valid_ids
        or not valid_maps
        or context["source_lot_ids"] != []
        or context["source_decision_ids"] != []
        or Decimal(str(order.quantity)) != total
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
    prior_closed = session.scalar(
        select(func.coalesce(func.sum(PaperBrokerFill.fill_quantity), 0.0))
        .join(PaperBrokerOrder, PaperBrokerOrder.id == PaperBrokerFill.paper_broker_order_id)
        .where(
            PaperBrokerOrder.account_scope == order.account_scope,
            PaperBrokerOrder.account_generation == order.account_generation,
            PaperBrokerOrder.market == order.market,
            PaperBrokerOrder.symbol == order.symbol,
            PaperBrokerOrder.instrument == order.instrument,
            PaperBrokerOrder.side == order.side,
            PaperBrokerOrder.broker_order_id.like(f"{LEGACY_EXECUTION_PREFIX}-%"),
            PaperBrokerOrder.client_order_ref != client_order_ref,
        )
    )
    remaining_closed = Decimal(str(prior_closed))
    remaining_by_ref = {}
    for row in baselines:
        quantity = Decimal(str(row.quantity))
        consumed = min(quantity, remaining_closed)
        remaining_by_ref[row.client_order_ref] = quantity - consumed
        remaining_closed -= consumed
    if remaining_closed:
        raise BrokerCapabilityError("legacy position baseline is smaller than prior durable closes")
    for holding_id, ref in refs.items():
        row = by_ref.get(ref)
        risk = context["source_holding_risk"][holding_id]
        if (
            row is None
            or Decimal(str(context["source_holding_quantities"][holding_id])) != remaining_by_ref[ref]
            or Decimal(str(risk["entry_price"])) != Decimal(str(row.price))
        ):
            raise BrokerCapabilityError("legacy position baseline changed")


def _position_state(session, order: Order) -> tuple[Decimal, Decimal]:
    baselines = _validated_baselines(
        session,
        order.account_scope,
        order.account_generation,
        market=order.market,
        symbol=order.symbol,
        instrument=order.instrument,
        side=order.side,
    )
    quantity = sum((Decimal(str(row.quantity)) for row in baselines), Decimal("0"))
    entry_cost = sum((Decimal(str(row.quantity)) * Decimal(str(row.price)) for row in baselines), Decimal("0"))
    rows = session.execute(
        select(PaperBrokerOrder, PaperBrokerFill)
        .join(PaperBrokerFill, PaperBrokerFill.paper_broker_order_id == PaperBrokerOrder.id)
        .where(
            PaperBrokerOrder.account_scope == order.account_scope,
            PaperBrokerOrder.account_generation == order.account_generation,
            PaperBrokerOrder.market == order.market,
            PaperBrokerOrder.symbol == order.symbol,
            PaperBrokerOrder.instrument == order.instrument,
            PaperBrokerOrder.side == order.side,
        )
        .order_by(PaperBrokerFill.state_version, PaperBrokerFill.id)
    ).all()
    opening_action = "buy" if order.side == "long" else "sell"
    for broker_order, fill in rows:
        fill_quantity = Decimal(str(fill.fill_quantity))
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
                average_entry = entry_cost / position_quantity
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
            quantities: dict[tuple[str, str, str, str], Decimal] = {}
            for baseline in _validated_baselines(session, account_scope, account_generation):
                if baseline.state_version > account.state_version:
                    raise BrokerCapabilityError("legacy position baseline exceeds account watermark")
                identity = (baseline.market, baseline.symbol, baseline.instrument, baseline.side)
                quantities[identity] = quantities.get(identity, Decimal("0")) + Decimal(str(baseline.quantity))
            for broker_order, fill in rows:
                identity = (fill.market, fill.symbol, fill.instrument, fill.side)
                opening_action = "buy" if fill.side == "long" else "sell"
                direction = Decimal("1") if broker_order.action == opening_action else Decimal("-1")
                quantities[identity] = quantities.get(identity, Decimal("0")) + direction * Decimal(
                    str(fill.fill_quantity)
                )
            positions = tuple(
                BrokerPositionSnapshot(*identity, float(quantity))
                for identity, quantity in sorted(quantities.items())
                if quantity != 0
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
