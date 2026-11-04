"""Decision-loop execution: claim, submit, reconcile, and recover decision orders, and the
durable protective-exit route shared with the portfolio and perp monitors."""

import math
import uuid
from datetime import UTC, datetime

from poseidon.core.authz import AuthPrincipal
from poseidon.core.canonical import content_sha256
from poseidon.core.config import settings
from poseidon.core.database import SessionLocal
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks.loop import _loop_reconcile_skip
from poseidon.workers.cpu_tasks.market_state import _get_latest_prices, _get_perp_mark_prices


def _decision_paper_adapter(market):
    from poseidon.broker.paper_adapter import PaperBrokerAdapter
    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter

    if market == "tw_stock":
        return PaperBrokerAdapter(SessionLocal)
    if market == "crypto_perp":
        return PerpPaperAdapter(SessionLocal)
    raise ValueError("decision execution requires a supported paper market")


def _decision_liquidation_nav(snapshot, policy, prices, ledger, *, position_economics=None):
    from poseidon.decision_loop.account_reconciliation import paper_liquidation_nav

    return paper_liquidation_nav(
        snapshot,
        policy,
        prices,
        ledger,
        position_economics=position_economics,
    )


def _ordinary_execution_authorized(scope, market, generation):
    return (
        settings.decision_loop_execution_mode == "decision"
        and settings.decision_loop_execution_enabled
        and bool(scope)
        and bool(generation)
        and (scope, market, generation)
        == (
            settings.decision_loop_approved_account_scope,
            settings.decision_loop_approved_market,
            settings.decision_loop_approved_account_generation,
        )
    )


def _protective_execution_route(mode, *, enabled, approved_identity, legacy_allowed):
    if mode in {"legacy", "shadow"}:
        return "legacy"
    if mode == "decision" and not enabled:
        return "skip"
    if approved_identity:
        return "durable"
    return "legacy" if legacy_allowed else "skip"


def _configured_protective_route(market):
    scope = settings.decision_loop_approved_account_scope
    generation = settings.decision_loop_approved_account_generation
    approved_identity = bool(
        scope
        and generation
        and settings.decision_loop_approved_market
        and market == settings.decision_loop_approved_market
    )
    return _protective_execution_route(
        settings.decision_loop_execution_mode,
        enabled=settings.decision_loop_execution_enabled,
        approved_identity=approved_identity,
        legacy_allowed=scope in settings.decision_loop_legacy_protective_scopes,
    )


def _run_durable_protective(*, origin, market, triggered, source_holding_ids_by_identity=None):
    """Commit protective intents per lot identity, then enqueue UUID-only submission."""
    from poseidon.decision_loop.protective_execution import ProtectiveExecutionService

    scope = settings.decision_loop_approved_account_scope
    generation = settings.decision_loop_approved_account_generation
    principal = AuthPrincipal("system:protective-worker", frozenset({"decision-worker"}), frozenset({scope}))
    order_ids = []
    for identity in sorted(triggered):
        identity_market, symbol, instrument, side = identity
        if identity_market != market:
            raise ValueError("protective trigger market changed")
        price = triggered[identity]
        source_holding_ids = (
            None if source_holding_ids_by_identity is None else source_holding_ids_by_identity[identity]
        )
        trigger_generation = content_sha256(
            {
                "origin": origin,
                "identity": {
                    "market": market,
                    "symbol": symbol,
                    "instrument": instrument,
                    "side": side,
                },
                "price": price,
                "source_holding_ids": source_holding_ids,
            }
        )
        with SessionLocal() as session, session.begin():
            response = ProtectiveExecutionService(session).materialize(
                account_scope=scope,
                account_generation=generation,
                market=market,
                symbol=symbol,
                instrument=instrument,
                side=side,
                origin=origin,
                trigger_generation=trigger_generation,
                price=price,
                principal=principal,
                source_holding_ids=source_holding_ids,
                allow_legacy_holdings=source_holding_ids is not None,
            )
        order_ids.extend(response["order_ids"])
    for order_id in order_ids:
        submit_decision_order.delay(order_id)
    return order_ids


def _approved_legacy_holdings(market):
    """Return only non-projected pre-lot holdings for the configured paper account."""
    from sqlalchemy import select

    from poseidon.models.paper_broker_account import PaperBrokerAccount
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord

    scope = settings.decision_loop_approved_account_scope
    generation = settings.decision_loop_approved_account_generation
    with SessionLocal() as session:
        account = session.scalar(
            select(PaperBrokerAccount).where(
                PaperBrokerAccount.account_scope == scope,
                PaperBrokerAccount.account_generation == generation,
            )
        )
        if account is None:
            return []
        marker = f"decision-lots:{account.id}"
        holdings = session.scalars(
            select(PortfolioHoldingRecord)
            .where(
                PortfolioHoldingRecord.market == market,
                PortfolioHoldingRecord.closed.is_(False),
                PortfolioHoldingRecord.strategy_name != marker,
                PortfolioHoldingRecord.strategy_name.not_like("decision-lots:%"),
            )
            .order_by(PortfolioHoldingRecord.symbol, PortfolioHoldingRecord.side, PortfolioHoldingRecord.id)
        ).all()
        if market != "crypto_perp":
            return holdings
        from poseidon.models.position_lot import PositionLot

        durable_symbols = set(
            session.scalars(
                select(PositionLot.symbol).where(
                    PositionLot.account_scope == scope,
                    PositionLot.account_generation == generation,
                    PositionLot.market == market,
                    PositionLot.open_quantity > 0,
                )
            ).all()
        )
        return [row for row in holdings if row.symbol not in durable_symbols]


def _durable_protective_triggers(market):
    """Evaluate the configured account's durable lots without legacy position state."""
    from collections import defaultdict

    from sqlalchemy import select

    from poseidon.decision_loop.protective_execution import ProtectiveExecutionService
    from poseidon.decision_loop.reservations import ExecutionConflictError
    from poseidon.models.position_lot import PositionLot
    from poseidon.models.strategy_version import StrategyVersion

    scope = settings.decision_loop_approved_account_scope
    generation = settings.decision_loop_approved_account_generation
    with SessionLocal() as session:
        lots = session.scalars(
            select(PositionLot)
            .where(
                PositionLot.account_scope == scope,
                PositionLot.account_generation == generation,
                PositionLot.market == market,
                PositionLot.open_quantity > 0,
            )
            .order_by(PositionLot.symbol, PositionLot.instrument, PositionLot.side, PositionLot.opened_at)
        ).all()
        source_settings = {}
        service = ProtectiveExecutionService(session)
        for decision_id in sorted({lot.opening_decision_id for lot in lots}, key=str):
            try:
                decision, _policy = service.source_decision_policy(decision_id, scope, generation, market)
            except ExecutionConflictError:
                return {"checked": len(lots), "triggered": {}, "skipped": "protective_policy_unresolved"}
            version = session.get(StrategyVersion, decision.strategy_version_id)
            config = version.config_json if isinstance(version.config_json, dict) else {}
            risk = config.get("protective_exit")
            if not isinstance(risk, dict):
                return {"checked": len(lots), "triggered": {}, "skipped": "protective_policy_unresolved"}
            required = ("stop_loss_pct",) if market == "tw_stock" else ("leverage", "margin_ratio_threshold")
            values = tuple(risk.get(field) for field in required)
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in values
            ) or (market == "tw_stock" and values[0] >= 1):
                return {"checked": len(lots), "triggered": {}, "skipped": "protective_policy_unresolved"}
            source_settings[decision_id] = values
        grouped = defaultdict(list)
        for lot in lots:
            grouped[(market, lot.symbol, lot.instrument, lot.side)].append(lot)
    legacy = (
        _approved_legacy_holdings(market)
        if market == "tw_stock" and scope in settings.decision_loop_legacy_protective_scopes
        else []
    )
    legacy_grouped = defaultdict(list)
    legacy_skipped = False
    for holding in legacy:
        values = (holding.shares, holding.entry_price, holding.stop_loss_pct)
        if (
            holding.side not in {"long", "short"}
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in values
            )
            or holding.stop_loss_pct >= 1
        ):
            legacy_grouped.clear()
            legacy_skipped = True
            break
        legacy_grouped[(market, holding.symbol, "spot", holding.side)].append(holding)
    if not grouped and not legacy_grouped:
        return {"checked": 0, "triggered": {}, "legacy_sources": {}, "legacy_triggered": {}}
    symbols = sorted({identity[1] for identity in grouped} | {identity[1] for identity in legacy_grouped})
    prices = _get_latest_prices(symbols) if market == "tw_stock" else _get_perp_mark_prices(symbols)
    triggered = {}
    for identity, identity_lots in grouped.items():
        _identity_market, symbol, _instrument, side = identity
        price = prices.get(symbol)
        if price is None or not math.isfinite(price) or price <= 0:
            continue
        if market == "tw_stock":
            breached = any(
                price <= lot.cost_basis_json["unit_price"] * (1 - source_settings[lot.opening_decision_id][0])
                if side == "long"
                else price >= lot.cost_basis_json["unit_price"] * (1 + source_settings[lot.opening_decision_id][0])
                for lot in identity_lots
            )
        else:
            direction = 1 if side == "long" else -1
            breached = any(
                (
                    lot.cost_basis_json["unit_price"] / source_settings[lot.opening_decision_id][0]
                    + (price - lot.cost_basis_json["unit_price"]) * direction
                )
                / price
                < source_settings[lot.opening_decision_id][1]
                for lot in identity_lots
            )
        if breached:
            triggered[identity] = price
    if market == "crypto_perp" and triggered:
        triggered = {identity: prices[identity[1]] for identity in grouped if identity[1] in prices}
    legacy_sources = {}
    legacy_triggered = {}
    for identity, holdings in legacy_grouped.items():
        _identity_market, symbol, _instrument, side = identity
        price = prices.get(symbol)
        if price is None or not math.isfinite(price) or price <= 0:
            continue
        breached = [
            holding
            for holding in holdings
            if (
                price <= holding.entry_price * (1 - holding.stop_loss_pct)
                if side == "long"
                else price >= holding.entry_price * (1 + holding.stop_loss_pct)
            )
        ]
        if breached:
            legacy_sources[identity] = [str(row.id) for row in breached]
            legacy_triggered[identity] = price
    result = {
        "checked": len(grouped) + len(legacy_grouped),
        "triggered": triggered,
        "legacy_sources": legacy_sources,
        "legacy_triggered": legacy_triggered,
    }
    if legacy_skipped:
        result["skipped"] = "legacy_protective_policy_unresolved"
    return result


def _approved_ordinary_candidate(session, market, now):
    from sqlalchemy import select

    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_version import StrategyVersion

    return session.scalar(
        select(DecisionRecord)
        .join(StrategyVersion, StrategyVersion.id == DecisionRecord.strategy_version_id)
        .join(StrategyRecord, StrategyRecord.id == StrategyVersion.strategy_id)
        .where(
            DecisionRecord.account_scope == settings.decision_loop_approved_account_scope,
            DecisionRecord.status == "approved",
            DecisionRecord.execution_key.is_(None),
            DecisionRecord.claimed_at.is_(None),
            DecisionRecord.valid_until > now,
            StrategyVersion.policy_json["market"].as_string() == market,
            StrategyVersion.policy_json["reconciliation"]["account_generation"].as_string()
            == settings.decision_loop_approved_account_generation,
            StrategyRecord.active_version_id == DecisionRecord.strategy_version_id,
        )
        .order_by(DecisionRecord.created_at, DecisionRecord.id)
        .limit(1)
    )


def _execute_approved_decision(market):
    """One ordinary writer: commit a validated claim before UUID-only worker calls."""
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.decision_loop.decisions import DecisionConflictError, DecisionService
    from poseidon.decision_loop.execution import DecisionExecutionService
    from poseidon.decision_loop.reservations import ExecutionConflictError

    scope = settings.decision_loop_approved_account_scope
    generation = settings.decision_loop_approved_account_generation
    if not _ordinary_execution_authorized(scope, market, generation):
        return {"skipped": "decision_execution_not_authorized"}
    now = datetime.now(UTC)
    principal = AuthPrincipal("system:decision-worker", frozenset({"decision-worker"}), frozenset({scope}))
    try:
        path = "config/broker.yaml" if market == "tw_stock" else "config/broker_perp.yaml"
        with open(path) as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict) or raw.get("mode") != "paper":
            raise ValueError("ordinary decision execution requires explicit paper broker mode")
        broker_config = BrokerConfig(**raw)
        if broker_config.execution_backend != "paper":
            raise ValueError("ordinary decision execution requires paper backend")
        adapter = _decision_paper_adapter(market)
        with SessionLocal() as session, session.begin():
            decision = _approved_ordinary_candidate(session, market, now)
            if decision is None:
                return {"skipped": "no_approved_decision"}
            DecisionExecutionService(session).preflight_approved(
                decision, principal=principal, market=market, account_generation=generation, adapter=adapter
            )
            response = DecisionService(session).claim_execution(
                decision.id, decision.revision, principal=principal, now=datetime.now(UTC)
            )
    except (ValueError, OSError, DecisionConflictError, ExecutionConflictError):
        return {"skipped": "decision_preflight_failed"}
    decision_id = response["decision_id"]
    materialized = materialize_execution_claim.run(decision_id)
    results = [submit_decision_order.run(order_id) for order_id in materialized["order_ids"]]
    return {"decision_id": decision_id, "orders": len(results), "mode": "decision"}


def _record_shadow_parity(market, orders):
    """Audit frozen target parity only; never claim, materialize, or submit."""
    from sqlalchemy import select

    from poseidon.decision_loop.execution import DecisionExecutionService
    from poseidon.decision_loop.reservations import ExecutionConflictError
    from poseidon.models.decision_event import DecisionEvent

    if settings.decision_loop_approved_market != market:
        return
    with SessionLocal() as session, session.begin():
        decision = _approved_ordinary_candidate(session, market, datetime.now(UTC))
        if decision is None:
            return
        # Serialize only the audit append against concurrent shadow deliveries.
        session.refresh(decision, with_for_update=True)
        if decision.status != "approved":
            return
        try:
            _, _, intents = DecisionExecutionService(session).policy_and_intents(decision)
        except ExecutionConflictError:
            return
        legacy = sorted(
            [
                [
                    market,
                    order.symbol,
                    getattr(order, "instrument", "spot" if market == "tw_stock" else None),
                    order.side,
                    ("enter" if order.current_weight == 0 else "add")
                    if order.delta_weight > 0
                    else (("exit" if order.target_weight == 0 else "reduce") if order.delta_weight < 0 else "hold"),
                    float(order.target_weight),
                ]
                for order in orders
            ]
        )
        frozen = sorted(
            [
                [
                    intent.market,
                    intent.symbol,
                    intent.instrument,
                    intent.side,
                    intent.action,
                    float(intent.target_weight),
                ]
                for intent in intents
            ]
        )
        payload = {
            "decision_id": str(decision.id),
            "policy_sha256": decision.policy_sha256,
            "legacy_sha256": content_sha256(legacy),
            "intents_sha256": content_sha256(frozen),
            "frozen_intents_sha256": content_sha256([intent.model_dump(mode="json") for intent in intents]),
            "legacy_targets": legacy,
            "frozen_targets": frozen,
            "matched": legacy == frozen,
        }
        digest = content_sha256(payload)
        key = "shadow:" + digest
        existing = session.scalar(
            select(DecisionEvent).where(DecisionEvent.decision_id == decision.id, DecisionEvent.idempotency_key == key)
        )
        if existing is not None:
            if existing.payload_json != payload:
                raise ExecutionConflictError("shadow audit replay changed content")
            return
        session.add(
            DecisionEvent(
                decision_id=decision.id,
                event_type="shadow:" + digest[:17],
                actor_id="system:shadow-parity",
                expected_revision=decision.revision,
                idempotency_key=key,
                request_sha256=digest,
                payload_json=payload,
            )
        )


@celery_app.task(name="poseidon.workers.cpu_tasks.materialize_execution_claim", max_retries=0)
def materialize_execution_claim(decision_id: str) -> dict:
    persisted_id = uuid.UUID(decision_id)
    from sqlalchemy import select

    from poseidon.broker.paper_adapter import validated_position_economics
    from poseidon.decision_loop.execution import DecisionExecutionService
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.paper_broker_fill import PaperBrokerFill
    from poseidon.models.paper_broker_order import PaperBrokerOrder

    with SessionLocal() as session:
        decision = session.get(DecisionRecord, persisted_id)
        if decision is None:
            raise ValueError("decision does not exist")
        _, policy, intents = DecisionExecutionService(session).policy_and_intents(decision)
        account_scope = decision.account_scope
        reconciliation = policy.reconciliation
        if not _ordinary_execution_authorized(account_scope, policy.market, reconciliation.account_generation):
            return {"skipped": "ordinary_execution_not_authorized"}
        reservation_identities = {
            (order.market, order.symbol, order.instrument)
            for order in DecisionExecutionService(session).pending_orders(decision, reconciliation.account_generation)
        }
    adapter = _decision_paper_adapter(policy.market)
    snapshot = adapter.query_account_snapshot(account_scope, reconciliation.account_generation)
    if (snapshot.account_scope, snapshot.account_generation, snapshot.currency) != (
        account_scope,
        reconciliation.account_generation,
        reconciliation.currency,
    ):
        raise ValueError("paper snapshot account identity changed")
    with SessionLocal() as session:
        ledger = session.execute(
            select(PaperBrokerOrder, PaperBrokerFill)
            .join(PaperBrokerFill, PaperBrokerFill.paper_broker_order_id == PaperBrokerOrder.id)
            .where(
                PaperBrokerOrder.account_scope == account_scope,
                PaperBrokerOrder.account_generation == reconciliation.account_generation,
                PaperBrokerFill.state_version <= snapshot.state_version,
            )
            .order_by(PaperBrokerFill.state_version, PaperBrokerFill.id)
        ).all()
        position_economics = validated_position_economics(
            session,
            account_scope,
            reconciliation.account_generation,
            snapshot.state_version,
        )
    identities = {(intent.market, intent.symbol, intent.instrument) for intent in intents}
    identities.update(reservation_identities)
    identities.update((position.market, position.symbol, position.instrument) for position in snapshot.positions)
    prices = {}
    for market in sorted({identity[0] for identity in identities}):
        symbols = sorted({identity[1] for identity in identities if identity[0] == market})
        if market == "tw_stock":
            marks = _get_latest_prices(symbols)
        elif market == "crypto_perp":
            marks = _get_perp_mark_prices(symbols)
        else:
            raise ValueError("unsupported materialization market")
        for identity in identities:
            if identity[0] == market:
                mark = marks.get(identity[1])
                if mark is None or isinstance(mark, bool) or not math.isfinite(mark) or mark <= 0:
                    raise ValueError("materialization requires every current mark")
                prices[identity] = mark
    nav = _decision_liquidation_nav(
        snapshot,
        reconciliation,
        prices,
        ledger,
        position_economics=position_economics,
    )
    principal = AuthPrincipal("system:decision-worker", frozenset({"decision-worker"}), frozenset({account_scope}))
    with SessionLocal() as session, session.begin():
        execution = DecisionExecutionService(session)
        decision = session.get(DecisionRecord, persisted_id)
        if decision is None:
            raise ValueError("decision does not exist")
        return execution.materialize(
            persisted_id,
            principal=principal,
            account_nav=nav,
            prices=prices,
            expected_account_state_version=snapshot.state_version,
        )


@celery_app.task(name="poseidon.workers.cpu_tasks.submit_decision_order", max_retries=0)
def submit_decision_order(order_id: str) -> dict:
    persisted_id = uuid.UUID(order_id)
    from poseidon.decision_loop.reconciliation import submit_or_reconcile_order
    from poseidon.models.order import OrderRecord
    from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS

    with SessionLocal() as session:
        order = session.get(OrderRecord, persisted_id)
        if order is None or order.order_origin not in {"decision", *DURABLE_PROTECTIVE_ORIGINS}:
            raise ValueError("decision order does not exist")
        if order.submit_attempted_at is None:
            authorized = (
                _ordinary_execution_authorized(order.account_scope, order.market, order.account_generation)
                if order.order_origin == "decision"
                else _configured_protective_route(order.market) == "durable"
                and (order.account_scope, order.account_generation)
                == (
                    settings.decision_loop_approved_account_scope,
                    settings.decision_loop_approved_account_generation,
                )
            )
            if not authorized:
                reason = (
                    "ordinary_execution_not_authorized"
                    if order.order_origin == "decision"
                    else "durable_execution_not_authorized"
                )
                return {"skipped": reason}
        market = order.market
    return submit_or_reconcile_order(SessionLocal, persisted_id, _decision_paper_adapter(market))


@celery_app.task(name="poseidon.workers.cpu_tasks.reconcile_execution_claim", max_retries=0)
def reconcile_execution_claim(decision_id: str) -> dict:
    persisted_id = uuid.UUID(decision_id)
    from sqlalchemy import select

    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.order import OrderRecord

    with SessionLocal() as session:
        if session.get(DecisionRecord, persisted_id) is None:
            raise ValueError("decision does not exist")
        order_ids = session.scalars(
            select(OrderRecord.id)
            .where(OrderRecord.decision_id == persisted_id, OrderRecord.submit_attempted_at.is_not(None))
            .order_by(OrderRecord.id)
        ).all()
    results, failures = [], []
    for order_id in order_ids:
        try:
            results.append(submit_decision_order.run(str(order_id)))
        except Exception as error:
            error.add_note(f"decision order {order_id}")
            failures.append(error)
    if failures:
        raise ExceptionGroup("decision order reconciliation failed", failures)
    return {"decision_id": str(persisted_id), "orders": results}


@celery_app.task(name="poseidon.workers.cpu_tasks.project_decision_fill", max_retries=0)
def project_decision_fill(fill_id: str) -> dict:
    persisted_id = uuid.UUID(fill_id)
    from poseidon.positions.lots import apply_fill_projection

    return apply_fill_projection(SessionLocal, persisted_id)


@celery_app.task(name="poseidon.workers.cpu_tasks.reconcile_paper_account", max_retries=0)
def reconcile_paper_account(account_id: str) -> dict:
    persisted_id = uuid.UUID(account_id)
    from sqlalchemy import select

    from poseidon.decision_loop.account_reconciliation import reconcile_account
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.paper_broker_account import PaperBrokerAccount
    from poseidon.models.paper_broker_order import PaperBrokerOrder
    from poseidon.models.position_lot import PositionLot
    from poseidon.models.strategy_version import StrategyVersion

    with SessionLocal() as session:
        account = session.get(PaperBrokerAccount, persisted_id)
        if account is None:
            raise ValueError("paper account does not exist")
        market = None
        for decision in session.scalars(
            select(DecisionRecord)
            .where(DecisionRecord.account_scope == account.account_scope)
            .order_by(DecisionRecord.created_at.desc(), DecisionRecord.id.desc())
        ):
            version = session.get(StrategyVersion, decision.strategy_version_id)
            policy = {} if version is None else version.policy_json
            reconciliation_policy = policy.get("reconciliation") if isinstance(policy, dict) else None
            if (
                isinstance(reconciliation_policy, dict)
                and reconciliation_policy.get("account_generation") == account.account_generation
            ):
                market = policy.get("market")
                break
        identities = set(
            session.execute(
                select(PaperBrokerOrder.market, PaperBrokerOrder.symbol, PaperBrokerOrder.instrument).where(
                    PaperBrokerOrder.account_scope == account.account_scope,
                    PaperBrokerOrder.account_generation == account.account_generation,
                )
            ).all()
        )
        identities.update(
            session.execute(
                select(PositionLot.market, PositionLot.symbol, PositionLot.instrument).where(
                    PositionLot.account_scope == account.account_scope,
                    PositionLot.account_generation == account.account_generation,
                    PositionLot.open_quantity > 0,
                )
            ).all()
        )
    prices = {}
    for position_market in sorted({identity[0] for identity in identities}):
        symbols = sorted({identity[1] for identity in identities if identity[0] == position_market})
        marks = (
            _get_latest_prices(symbols)
            if position_market == "tw_stock"
            else (_get_perp_mark_prices(symbols) if position_market == "crypto_perp" else {})
        )
        for identity in identities:
            if identity[0] == position_market and identity[1] in marks:
                prices[tuple(identity)] = marks[identity[1]]
    # Snapshot reads use the generic independent ledger even when owner policy is missing.
    # This does not authorize order placement or supply missing policy/tolerance values.
    adapter = _decision_paper_adapter(market if market in {"tw_stock", "crypto_perp"} else "tw_stock")
    return reconcile_account(SessionLocal, persisted_id, adapter, prices=prices)


@celery_app.task(name="poseidon.workers.cpu_tasks.recover_decision_execution", max_retries=0)
def recover_decision_execution() -> dict:
    from poseidon.decision_loop.recovery import RecoverySelector
    from poseidon.models.order import OrderRecord
    from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS

    tasks = {
        "materialize_decision": materialize_execution_claim,
        "submit_order": submit_decision_order,
        "reconcile_order": reconcile_execution_claim,
        "project_fill": project_decision_fill,
        "reconcile_account": reconcile_paper_account,
    }
    with SessionLocal() as session:
        actions = RecoverySelector(session).select()
        dispatches = []
        seen = set()
        for action in actions:
            operation = action.operation
            persisted_id = action.persisted_id
            if action.operation == "reconcile_order":
                order = session.get(OrderRecord, persisted_id)
                if order is None:
                    raise ValueError("recovery order lacks decision identity")
                if order.decision_id is None:
                    if order.order_origin not in DURABLE_PROTECTIVE_ORIGINS:
                        raise ValueError("recovery order lacks decision identity")
                    operation = "submit_order"
                else:
                    persisted_id = order.decision_id
            dispatch = operation, str(persisted_id)
            if dispatch not in seen:
                dispatches.append(dispatch)
                seen.add(dispatch)
    for operation, persisted_id in dispatches:
        tasks[operation].delay(persisted_id)
    return {"enqueued": len(dispatches)}


@celery_app.task(name="poseidon.workers.cpu_tasks.scheduled_recover_decision_execution", max_retries=0)
def scheduled_recover_decision_execution() -> dict:
    """Beat entry for the existing recovery selector; a no-op outside shadow/decision/halted."""
    skipped = _loop_reconcile_skip()
    if skipped:
        return skipped
    return recover_decision_execution.run()


@celery_app.task(name="poseidon.workers.cpu_tasks.scheduled_reconcile_paper_account", max_retries=0)
def scheduled_reconcile_paper_account() -> dict:
    """Beat entry reconciling only the approved paper account; a no-op outside shadow/decision/halted."""
    skipped = _loop_reconcile_skip()
    if skipped:
        return skipped
    from sqlalchemy import select

    from poseidon.models.paper_broker_account import PaperBrokerAccount

    scope = settings.decision_loop_approved_account_scope
    generation = settings.decision_loop_approved_account_generation
    if not scope or not generation:
        return {"skipped": "account_not_approved"}
    with SessionLocal() as session:
        account_id = session.scalar(
            select(PaperBrokerAccount.id).where(
                PaperBrokerAccount.account_scope == scope,
                PaperBrokerAccount.account_generation == generation,
            )
        )
    if account_id is None:
        return {"skipped": "paper_account_missing"}
    return reconcile_paper_account.run(str(account_id))
