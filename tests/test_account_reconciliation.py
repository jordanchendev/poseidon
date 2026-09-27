"""Independent PostgreSQL account truth, drift, and exposure reconciliation."""

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from threading import Barrier
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import delete, select, text, update

from poseidon.api.auth import AuthPrincipal
from poseidon.broker.base import BrokerCapabilityError
from poseidon.broker.paper_adapter import PaperBrokerAdapter
from poseidon.decision_loop.execution import (
    DecisionExecutionService,
    ExecutionConflictError,
    ProtectiveExecutionService,
    internal_state_watermark,
)
from poseidon.decision_loop.reconciliation import ReconciliationService, submit_or_reconcile_order
from poseidon.models.account_reconciliation import AccountReconciliation
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
from tests import test_position_lot_allocation as lot_tests

NOW = lot_tests.NOW + timedelta(days=2)


@pytest.fixture
def account():
    fixture = lot_tests.seed.__wrapped__()
    seed = next(fixture)
    initial = seed.fill()
    with seed.sessions() as session, session.begin():
        fill = session.get(OrderFillRecord, initial)
        order = session.get(OrderRecord, fill.order_id)
        decision_id = order.decision_id
        session.delete(fill)
        session.flush()
        session.delete(order)
        broker_account = session.scalar(
            select(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == seed.account)
        )
        account_id = broker_account.id
    adapter = PaperBrokerAdapter(session_factory=seed.sessions)
    original_query = adapter.query_account_snapshot
    adapter.query_account_snapshot = lambda scope, generation: replace(original_query(scope, generation), as_of=NOW)
    value = SimpleNamespace(
        seed=seed, sessions=seed.sessions, id=account_id, scope=seed.account, decision_id=decision_id, adapter=adapter
    )
    yield value
    with seed.sessions() as session, session.begin():
        session.execute(delete(AccountReconciliation).where(AccountReconciliation.account_scope == seed.account))
        session.execute(delete(PaperCashMovement).where(PaperCashMovement.account_scope == seed.account))
        session.execute(delete(PaperBrokerFill).where(PaperBrokerFill.account_scope == seed.account))
        session.execute(delete(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == seed.account))
        session.execute(
            delete(PortfolioHoldingRecord).where(PortfolioHoldingRecord.strategy_name == f"decision-lots:{account_id}")
        )
        session.execute(
            delete(PortfolioHoldingRecord).where(
                PortfolioHoldingRecord.strategy_name.startswith("phase98-reconcile-legacy:")
            )
        )
    next(fixture, None)


def reconcile(account, **kwargs):
    from poseidon.decision_loop.reconciliation import reconcile_account

    kwargs.setdefault("prices", {("tw_stock", "2330", "spot"): 100.0})
    return reconcile_account(account.sessions, account.id, account.adapter, now=NOW, **kwargs)


def trade(account, *, quantity=5, price=100, action="enter", side="long", project=True, time=None, fill_uuid=None):
    fill_id = account.seed.fill(quantity=quantity, price=price, action=action, side=side)
    with account.sessions() as session, session.begin():
        fill = session.get(OrderFillRecord, fill_id)
        order = session.get(OrderRecord, fill.order_id)
        order.reserved_cash_json = {"currency": "TWD", "amount": quantity * price if action in {"enter", "add"} else 0}
        broker_account = session.get(PaperBrokerAccount, account.id)
        broker_account.state_version += 1
        version = broker_account.state_version
        fill.fill_time = fill.fill_time + timedelta(seconds=version) if time is None else time
        if fill_uuid is not None:
            fill.id = fill_uuid
            fill_id = fill_uuid
        broker_order = PaperBrokerOrder(
            id=uuid.uuid4(),
            account_scope=account.scope,
            account_generation="generation-1",
            client_order_ref=order.client_order_ref,
            broker_order_id=order.broker_order_id,
            market=order.market,
            symbol=order.symbol,
            instrument=order.instrument,
            action=order.action,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            state_version=version,
            accepted_at=fill.fill_time,
        )
        session.add(broker_order)
        session.flush()
        session.add(
            PaperBrokerFill(
                paper_broker_order_id=broker_order.id,
                account_scope=account.scope,
                account_generation="generation-1",
                broker_fill_id=fill.broker_fill_id,
                market=order.market,
                symbol=order.symbol,
                instrument=order.instrument,
                side=order.side,
                fill_price=fill.fill_price,
                fill_quantity=fill.fill_quantity,
                fill_time=fill.fill_time,
                state_version=version,
            )
        )
        cash_delta = (-1 if action in {"enter", "add"} else 1) * quantity * price
        if side == "short" and action in {"reduce", "exit"}:
            openings = session.execute(
                select(PaperBrokerFill.fill_quantity, PaperBrokerFill.fill_price)
                .join(PaperBrokerOrder)
                .where(
                    PaperBrokerOrder.account_scope == account.scope,
                    PaperBrokerOrder.side == "short",
                    PaperBrokerOrder.action == "sell",
                )
            ).all()
            average = sum(amount * entry for amount, entry in openings) / sum(amount for amount, _ in openings)
            cash_delta = (2 * average - price) * quantity
        session.add(
            PaperCashMovement(
                account_scope=account.scope,
                account_generation="generation-1",
                currency="TWD",
                amount=cash_delta,
                movement_type="fill",
                state_version=version,
                occurred_at=fill.fill_time,
            )
        )
    if project:
        lot_tests.project(account.seed, fill_id)
    return fill_id


def stored(account, result):
    with account.sessions() as session:
        return session.get(AccountReconciliation, uuid.UUID(result["reconciliation_id"]))


def legacy_close(account, monkeypatch, *, shares=5):
    holding_id = uuid.uuid4()
    with account.sessions() as session, session.begin():
        session.add(
            PortfolioHoldingRecord(
                id=holding_id,
                strategy_name=f"phase98-reconcile-legacy:{holding_id}",
                symbol="2330",
                market="tw_stock",
                weight=0.1,
                shares=shares,
                entry_price=100,
                entry_date=NOW,
                closed=False,
                side="long",
                stop_loss_pct=0.1,
            )
        )
        materialized = ProtectiveExecutionService(session).materialize(
            account_scope=account.scope,
            account_generation="generation-1",
            market="tw_stock",
            symbol="2330",
            instrument="spot",
            side="long",
            origin="stop_loss",
            trigger_generation=f"legacy-reconciliation:{holding_id}",
            price=100,
            source_holding_ids=[holding_id],
            allow_legacy_holdings=True,
            principal=AuthPrincipal(
                "system:test-protective",
                frozenset({"decision-worker"}),
                frozenset({account.scope}),
            ),
            now=NOW,
        )
    repo = type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})()
    monkeypatch.setattr("poseidon.data.remote_repository.RemoteDataRepository.from_settings", lambda: repo)
    order_id = uuid.UUID(materialized["order_ids"][0])
    submit_or_reconcile_order(account.sessions, order_id, account.adapter, now=NOW)
    with account.sessions() as session:
        fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == order_id))
    lot_tests.project(account.seed, fill_id)
    return holding_id, order_id


def legacy_partial_close(account, monkeypatch, *, shares=5, filled=2):
    holding_id = uuid.uuid4()
    with account.sessions() as session, session.begin():
        session.add(
            PortfolioHoldingRecord(
                id=holding_id,
                strategy_name=f"phase98-reconcile-legacy:{holding_id}",
                symbol="2330",
                market="tw_stock",
                weight=0.1,
                shares=shares,
                entry_price=100,
                entry_date=NOW,
                closed=False,
                side="long",
                stop_loss_pct=0.1,
            )
        )
        materialized = ProtectiveExecutionService(session).materialize(
            account_scope=account.scope,
            account_generation="generation-1",
            market="tw_stock",
            symbol="2330",
            instrument="spot",
            side="long",
            origin="stop_loss",
            trigger_generation=f"legacy-reconciliation-partial:{holding_id}",
            price=100,
            source_holding_ids=[holding_id],
            allow_legacy_holdings=True,
            principal=AuthPrincipal(
                "system:test-protective",
                frozenset({"decision-worker"}),
                frozenset({account.scope}),
            ),
            now=NOW,
        )
    repo = type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})()
    monkeypatch.setattr("poseidon.data.remote_repository.RemoteDataRepository.from_settings", lambda: repo)
    order_id = uuid.UUID(materialized["order_ids"][0])
    with account.sessions() as session, session.begin():
        prepared = ReconciliationService(session).prepare_attempt(order_id, account.adapter, now=NOW)
    accepted = account.adapter.place_order(prepared.order, client_order_ref=prepared.order.client_order_ref)
    with account.sessions() as session, session.begin():
        broker_order = session.scalar(
            select(PaperBrokerOrder).where(PaperBrokerOrder.broker_order_id == accepted.broker_order_id)
        )
        broker_order.status = "cancelled"
        session.scalar(
            select(PaperBrokerFill).where(PaperBrokerFill.paper_broker_order_id == broker_order.id)
        ).fill_quantity = filled
        session.scalar(
            select(PaperCashMovement).where(
                PaperCashMovement.account_scope == account.scope,
                PaperCashMovement.state_version == broker_order.state_version,
            )
        ).amount = filled * 100
    snapshot = account.adapter.find_order_by_client_ref(
        prepared.order.client_order_ref,
        account_scope=account.scope,
        account_generation="generation-1",
    )
    fills = account.adapter.query_fills(
        snapshot.broker_order_id,
        account_scope=account.scope,
        account_generation="generation-1",
    )
    with account.sessions() as session, session.begin():
        ReconciliationService(session).import_broker_state(order_id, snapshot, fills, now=NOW)
    with account.sessions() as session:
        fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == order_id))
    lot_tests.project(account.seed, fill_id)
    return holding_id, order_id


def test_bootstrap_zero_trade_matched_exact_replay_is_immutable(account):
    result = reconcile(account)
    assert result["status"] == "matched"
    assert reconcile(account) == result
    row = stored(account, result)
    assert row.broker_snapshot_json["orders"] == row.internal_snapshot_json["orders"] == {}
    assert row.broker_snapshot_json["cash"] == row.internal_snapshot_json["cash"] == {"TWD": 100000.0}
    assert row.broker_state_watermark == "broker:0"
    with account.sessions() as session:
        assert (
            len(
                session.scalars(
                    select(AccountReconciliation).where(AccountReconciliation.account_scope == account.scope)
                ).all()
            )
            == 1
        )


def test_full_legacy_close_baseline_is_ledger_watermark_not_broker_order_drift(monkeypatch, account):
    _holding_id, order_id = legacy_close(account, monkeypatch)

    result = reconcile(account)

    row = stored(account, result)
    assert result["status"] == "matched", row.difference_json
    assert row.broker_state_watermark == "broker:2"
    assert set(row.broker_snapshot_json["orders"]) == {
        session_ref
        for session_ref in [
            next(
                value
                for value in row.internal_snapshot_json["orders"]
                if row.internal_snapshot_json["orders"][value]["broker_order_id"].startswith("PAPER-LEGACY-")
            )
        ]
    }
    with account.sessions() as session:
        baseline = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == account.scope,
                PaperBrokerOrder.action == "position_base",
            )
        )
        assert baseline is not None
        assert baseline.state_version == 1
        assert session.get(OrderRecord, order_id).status == "filled"


def test_partial_legacy_close_baseline_remains_honest_unresolved(monkeypatch, account):
    legacy_partial_close(account, monkeypatch)

    result = reconcile(account)

    assert result["status"] in {"mismatch", "unresolved"}
    row = stored(account, result)
    assert not any(ref.startswith("LEGACY-BASE-") for ref in row.broker_snapshot_json["orders"])
    assert row.broker_snapshot_json["positions"]


def test_two_generation_legacy_close_replays_and_reconciles(monkeypatch, account):
    holding_id, first_order_id = legacy_partial_close(account, monkeypatch)
    with account.sessions() as session, session.begin():
        second = ProtectiveExecutionService(session).materialize(
            account_scope=account.scope,
            account_generation="generation-1",
            market="tw_stock",
            symbol="2330",
            instrument="spot",
            side="long",
            origin="stop_loss",
            trigger_generation=f"legacy-reconciliation-second:{holding_id}",
            price=100,
            source_holding_ids=[holding_id],
            allow_legacy_holdings=True,
            principal=AuthPrincipal(
                "system:test-protective",
                frozenset({"decision-worker"}),
                frozenset({account.scope}),
            ),
            now=NOW,
        )
    second_order_id = uuid.UUID(second["order_ids"][0])
    submit_or_reconcile_order(account.sessions, second_order_id, account.adapter, now=NOW)
    with account.sessions() as session:
        first_fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == first_order_id))
        second_fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == second_order_id))
    lot_tests.project(account.seed, second_fill_id)
    lot_tests.project(account.seed, first_fill_id)

    result = reconcile(account)

    assert result["status"] == "matched", stored(account, result).difference_json


@pytest.mark.parametrize(
    "corruption",
    ["action", "status", "client_ref", "broker_id", "fill", "parent_identity", "state_version"],
)
def test_malformed_legacy_attribution_is_never_excluded_from_reconciliation(monkeypatch, account, corruption):
    legacy_close(account, monkeypatch)
    snapshot = account.adapter.query_account_snapshot(account.scope, "generation-1")
    with account.sessions() as session, session.begin():
        attribution = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == account.scope,
                PaperBrokerOrder.action == "position_alloc",
            )
        )
        if corruption == "action":
            attribution.action = "sell"
        elif corruption == "status":
            attribution.status = "filled"
        elif corruption == "client_ref":
            attribution.client_order_ref = "LEGACY-ALLOC-not-a-uuid"
        elif corruption == "broker_id":
            attribution.broker_order_id = "PAPER-ALLOC-corrupt"
        elif corruption == "parent_identity":
            attribution.symbol = "0050"
        elif corruption == "state_version":
            attribution.state_version += 1
        else:
            session.add(
                PaperBrokerFill(
                    paper_broker_order_id=attribution.id,
                    account_scope=account.scope,
                    account_generation="generation-1",
                    broker_fill_id=f"corrupt-{uuid.uuid4()}",
                    market=attribution.market,
                    symbol=attribution.symbol,
                    instrument=attribution.instrument,
                    side=attribution.side,
                    fill_price=attribution.price,
                    fill_quantity=1,
                    fill_time=NOW,
                    state_version=attribution.state_version,
                )
            )
    with pytest.raises(BrokerCapabilityError):
        account.adapter.query_account_snapshot(account.scope, "generation-1")
    with account.sessions() as session, session.begin():
        result = ReconciliationService(session).reconcile_account(
            account.id,
            snapshot,
            now=NOW,
            prices={("tw_stock", "2330", "spot"): 100.0},
        )

    assert result["status"] in {"mismatch", "unresolved"}
    row = stored(account, result)
    assert any("broker control ledger unresolved" in reason for reason in row.difference_json["unresolved"])


@pytest.mark.parametrize("corruption", ["action", "status", "client_ref", "broker_id", "fill"])
def test_malformed_legacy_baseline_is_never_excluded_from_reconciliation(monkeypatch, account, corruption):
    legacy_close(account, monkeypatch)
    snapshot = account.adapter.query_account_snapshot(account.scope, "generation-1")
    with account.sessions() as session, session.begin():
        baseline = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == account.scope,
                PaperBrokerOrder.action == "position_base",
            )
        )
        if corruption == "action":
            baseline.action = "sell"
        elif corruption == "status":
            baseline.status = "filled"
        elif corruption == "client_ref":
            baseline.client_order_ref = "LEGACY-BASE-not-a-uuid"
        elif corruption == "broker_id":
            baseline.broker_order_id = "PAPER-BASE-corrupt"
        else:
            session.add(
                PaperBrokerFill(
                    paper_broker_order_id=baseline.id,
                    account_scope=account.scope,
                    account_generation="generation-1",
                    broker_fill_id=f"corrupt-{uuid.uuid4()}",
                    market=baseline.market,
                    symbol=baseline.symbol,
                    instrument=baseline.instrument,
                    side=baseline.side,
                    fill_price=baseline.price,
                    fill_quantity=1,
                    fill_time=NOW,
                    state_version=baseline.state_version,
                )
            )
    with account.sessions() as session, session.begin():
        result = ReconciliationService(session).reconcile_account(
            account.id,
            snapshot,
            now=NOW,
            prices={("tw_stock", "2330", "spot"): 100.0},
        )

    assert result["status"] != "matched"
    row = stored(account, result)
    assert any(
        reason.startswith(("broker control ledger unresolved:", "broker order unresolved:"))
        for reason in row.difference_json["unresolved"]
    )


@pytest.mark.parametrize(
    "missing", ["policy", "cash_tolerance", "position_tolerance", "fill_tolerance", "opening_cash"]
)
def test_bootstrap_missing_owner_inputs_is_unresolved(account, missing):
    with account.sessions() as session, session.begin():
        decision = session.get(DecisionRecord, account.decision_id)
        version = session.get(StrategyVersion, decision.strategy_version_id)
        policy = copy.deepcopy(version.policy_json)
        if missing == "policy":
            policy.pop("reconciliation")
        else:
            policy["reconciliation"].pop(missing)
        session.execute(update(StrategyVersion).where(StrategyVersion.id == version.id).values(policy_json=policy))
    result = reconcile(account)
    assert result["status"] == "unresolved"
    assert stored(account, result).difference_json["unresolved"]


def test_bootstrap_opening_cash_corruption_does_not_self_compare(account):
    with account.sessions() as session, session.begin():
        session.get(PaperBrokerAccount, account.id).opening_cash += 1
    result = reconcile(account)
    assert result["status"] != "matched"
    row = stored(account, result)
    assert row.broker_snapshot_json["cash"]["TWD"] == 100001
    assert row.internal_snapshot_json["cash"]["TWD"] == 100000


@pytest.mark.parametrize("corruption", ["order", "fill", "lot", "cash", "allocation"])
def test_independent_internal_corruption_is_drift_and_broker_unchanged(account, corruption):
    fill_id = trade(account)
    if corruption == "allocation":
        fill_id = trade(account, quantity=2, action="reduce")
    before = account.adapter.query_account_snapshot(account.scope, "generation-1")
    with account.sessions() as session, session.begin():
        fill = session.get(OrderFillRecord, fill_id)
        order = session.get(OrderRecord, fill.order_id)
        if corruption == "order":
            order.quantity += 1
        elif corruption == "fill":
            fill.fill_quantity -= 1
        elif corruption == "lot":
            session.scalar(select(PositionLot).where(PositionLot.account_scope == account.scope)).open_quantity -= 1
        elif corruption == "cash":
            fill.fill_price += 1
        else:
            session.scalar(select(FillAllocation).where(FillAllocation.closing_fill_id == fill_id)).quantity -= 1
    result = reconcile(account)
    assert result["status"] == "mismatch"
    assert stored(account, result).difference_json["differences"]
    assert account.adapter.query_account_snapshot(account.scope, "generation-1") == before


def test_extra_broker_order_and_fill_are_discovered_as_drift(account):
    trade(account, project=False)
    with account.sessions() as session, session.begin():
        fill = session.scalar(
            select(OrderFillRecord).join(OrderRecord).where(OrderRecord.account_scope == account.scope)
        )
        session.delete(fill)
        session.flush()
        session.delete(session.scalar(select(OrderRecord).where(OrderRecord.account_scope == account.scope)))
        # Existing lot remains detectable as orphaned internal state.
    result = reconcile(account)
    assert result["status"] == "mismatch"
    assert stored(account, result).broker_snapshot_json["orders"]


def test_projection_pending_unresolved_then_match_terminalizes_decision(account):
    fill_id = trade(account, project=False)
    assert reconcile(account)["status"] == "unresolved"
    with account.sessions() as session:
        decision_id = session.get(OrderRecord, session.get(OrderFillRecord, fill_id).order_id).decision_id
        assert session.get(DecisionRecord, decision_id).status != "executed"
    lot_tests.project(account.seed, fill_id)
    result = reconcile(account)
    assert result["status"] == "matched"
    with account.sessions() as session:
        assert session.get(DecisionRecord, decision_id).status == "executed"


def test_matched_account_terminalizes_only_canonical_ordinary_orders_with_anchored_protective(
    monkeypatch,
    account,
):
    opening_fill_id = trade(account)
    with account.sessions() as session:
        opening_order = session.get(OrderRecord, session.get(OrderFillRecord, opening_fill_id).order_id)
        decision_id = opening_order.decision_id
    with account.sessions() as session, session.begin():
        protective = ProtectiveExecutionService(session).materialize(
            account_scope=account.scope,
            account_generation="generation-1",
            market="tw_stock",
            symbol="2330",
            instrument="spot",
            side="long",
            origin="stop_loss",
            trigger_generation="terminalization:protective",
            price=100.0,
            principal=AuthPrincipal(
                "system:test-protective",
                frozenset({"decision-worker"}),
                frozenset({account.scope}),
            ),
            now=NOW,
        )
    protective_order_id = uuid.UUID(protective["order_ids"][0])
    repo = type(
        "Repo",
        (),
        {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})},
    )()
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: repo,
    )
    submit_or_reconcile_order(account.sessions, protective_order_id, account.adapter, now=NOW)
    with account.sessions() as session:
        protective_fill_id = session.scalar(
            select(OrderFillRecord.id).where(OrderFillRecord.order_id == protective_order_id)
        )
    lot_tests.project(account.seed, protective_fill_id)

    result = reconcile(account)

    assert result["status"] == "matched"
    with account.sessions() as session:
        assert session.get(DecisionRecord, decision_id).status == "executed"
        protective_order = session.get(OrderRecord, protective_order_id)
        assert protective_order.status == "filled"
        assert protective_order.reservation_status == "released"


def test_freshness_exposure_gate_invalidates_match_on_new_broker_watermark(account):
    result = reconcile(account)
    assert result["status"] == "matched"
    with account.sessions() as session, session.begin():
        decision = session.get(DecisionRecord, account.decision_id)
        execution = DecisionExecutionService(session)
        _, policy, _ = execution._policy_and_intents(decision)
        broker_account = session.get(PaperBrokerAccount, account.id)
        execution._require_reconciled(decision, policy.reconciliation, broker_account, NOW)
        broker_account.state_version += 1
        with pytest.raises(ExecutionConflictError):
            execution._require_reconciled(decision, policy.reconciliation, broker_account, NOW)


@pytest.mark.parametrize("side", ["long", "short"])
def test_legacy_projection_matches_lot_aggregate_and_liquidation_nav(account, side):
    trade(account, quantity=5, side=side)
    result = reconcile(account, prices={("tw_stock", "2330", "spot"): 110})
    assert result["status"] == "matched"
    row = stored(account, result)
    expected_nav = 100050 if side == "long" else 99950
    assert row.internal_snapshot_json["projection"]["account_nav"] == expected_nav
    with account.sessions() as session:
        holding = session.scalar(
            select(PortfolioHoldingRecord).where(PortfolioHoldingRecord.strategy_name == f"decision-lots:{account.id}")
        )
        assert holding.shares == 5
        assert holding.entry_price == 100
        assert holding.side == side
        assert holding.weight == pytest.approx(550 / expected_nav)


def test_legacy_unowned_open_holding_is_unresolved_not_overwritten(account):
    trade(account)
    legacy_id = uuid.uuid4()
    try:
        with account.sessions() as session, session.begin():
            session.add(
                PortfolioHoldingRecord(
                    id=legacy_id,
                    strategy_name="pre-042",
                    symbol="2330",
                    market="tw_stock",
                    weight=0.2,
                    shares=9,
                    entry_price=90,
                    entry_date=NOW,
                    closed=False,
                    side="long",
                )
            )
        result = reconcile(account)
        assert result["status"] == "unresolved"
        assert any("legacy" in reason for reason in stored(account, result).difference_json["unresolved"])
        with account.sessions() as session:
            assert session.get(PortfolioHoldingRecord, legacy_id).shares == 9
    finally:
        with account.sessions() as session, session.begin():
            session.execute(delete(PortfolioHoldingRecord).where(PortfolioHoldingRecord.id == legacy_id))


def test_projection_missing_marks_unresolved_without_fabricated_holdings(account):
    trade(account)
    result = reconcile(account, prices={})
    assert result["status"] == "unresolved"
    with account.sessions() as session:
        assert (
            session.scalar(
                select(PortfolioHoldingRecord.id).where(
                    PortfolioHoldingRecord.strategy_name == f"decision-lots:{account.id}"
                )
            )
            is None
        )


def test_projection_replay_changed_price_content_never_mutates_record(account):
    from poseidon.decision_loop.reconciliation import ReconciliationConflictError

    trade(account)
    result = reconcile(account)
    before = copy.deepcopy(stored(account, result).internal_snapshot_json)
    with pytest.raises(ReconciliationConflictError):
        reconcile(account, prices={("tw_stock", "2330", "spot"): 110})
    assert stored(account, result).internal_snapshot_json == before


def test_freshness_internal_fill_economic_corruption_invalidates_existing_gate(account):
    fill_id = trade(account)
    with account.sessions() as session, session.begin():
        decision = session.get(DecisionRecord, account.decision_id)
        session.add(
            AccountReconciliation(
                account_scope=account.scope,
                account_generation="generation-1",
                as_of=NOW,
                broker_state_watermark="broker:1",
                internal_state_watermark=internal_state_watermark(session, account.scope, "generation-1"),
                broker_snapshot_sha256="a" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=decision.policy_sha256,
                status="matched",
            )
        )
    with account.sessions() as session, session.begin():
        session.get(OrderFillRecord, fill_id).fill_price += 1
    with account.sessions() as session:
        execution = DecisionExecutionService(session)
        decision = session.get(DecisionRecord, account.decision_id)
        _, policy, _ = execution._policy_and_intents(decision)
        with pytest.raises(ExecutionConflictError):
            execution._require_reconciled(
                decision, policy.reconciliation, session.get(PaperBrokerAccount, account.id), NOW
            )


@pytest.mark.parametrize("corruption", ["allocation_cost", "allocation_decision", "lot_reservation"])
def test_allocation_and_reservation_internal_corruption_cannot_be_blessed(account, corruption):
    trade(account)
    closing = trade(account, quantity=2, action="reduce")
    with account.sessions() as session, session.begin():
        if corruption == "lot_reservation":
            session.scalar(
                select(PositionLot).where(PositionLot.account_scope == account.scope)
            ).reserved_close_quantity = 1
        else:
            allocation = session.scalar(select(FillAllocation).where(FillAllocation.closing_fill_id == closing))
            if corruption == "allocation_decision":
                allocation.closing_decision_id = account.decision_id
            else:
                cost = copy.deepcopy(allocation.realized_cost_json)
                cost["entry_cost"] += 1
                allocation.realized_cost_json = cost
    assert reconcile(account)["status"] == "mismatch"


def test_actual_fill_price_is_independently_reconciled_not_materialization_quote(account):
    fill_id = trade(account, project=False)
    with account.sessions() as session, session.begin():
        session.get(OrderFillRecord, fill_id).fill_price = 101
        session.scalar(select(PaperBrokerFill).where(PaperBrokerFill.account_scope == account.scope)).fill_price = 101
        session.scalar(select(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == account.scope)).price = 101
        session.scalar(select(PaperCashMovement).where(PaperCashMovement.account_scope == account.scope)).amount = -505
    lot_tests.project(account.seed, fill_id)
    assert reconcile(account)["status"] == "matched"


@pytest.mark.parametrize("side", ["long", "short"])
def test_tied_fill_time_cash_is_uuid_order_independent(account, side):
    opening = trade(account, quantity=5, side=side, time=lot_tests.NOW)
    trade(account, quantity=2, action="reduce", side=side, time=lot_tests.NOW, fill_uuid=uuid.UUID(int=opening.int - 1))
    result = reconcile(account)
    assert result["status"] == ("matched" if side == "long" else "unresolved")
    if side == "short":
        assert any("ambiguous" in reason for reason in stored(account, result).difference_json["unresolved"])


def test_multi_price_partial_short_cash_uses_independent_weighted_average(account):
    trade(account, quantity=5, price=100, side="short")
    trade(account, quantity=5, price=120, side="short", action="add")
    trade(account, quantity=4, price=110, side="short", action="reduce")
    result = reconcile(account)
    assert result["status"] == "matched"
    row = stored(account, result)
    assert row.internal_snapshot_json["cash"] == row.broker_snapshot_json["cash"] == {"TWD": 99340.0}


def test_concurrent_legacy_account_projection_never_locks_foreign_lots(account):
    from poseidon.strategies.portfolio.position_tracker import PositionTracker

    trade(account)
    other_scope = account.scope + ":other"
    lot_tests.project(account.seed, account.seed.fill(account=other_scope))
    with account.sessions() as session:
        other_id = session.scalar(select(PaperBrokerAccount.id).where(PaperBrokerAccount.account_scope == other_scope))
    barrier = Barrier(2)

    def project(account_id, scope):
        with account.sessions() as session, session.begin():
            pid = session.scalar(text("select pg_backend_pid()"))
            session.scalars(select(PositionLot).where(PositionLot.account_scope == scope).with_for_update()).all()
            barrier.wait(timeout=10)
            result = PositionTracker.project_lots(session, account_id, {("tw_stock", "2330", "spot"): 100}, 100000)
            return pid, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(timeout=20)
            for future in [pool.submit(project, account.id, account.scope), pool.submit(project, other_id, other_scope)]
        ]
    assert len({pid for pid, _ in results}) == 2
    assert all(result["status"] == "unresolved" for _, result in results)


def test_bootstrap_cash_drift_within_owner_tolerance_is_preserved_and_matched(account):
    with account.sessions() as session, session.begin():
        session.get(PaperBrokerAccount, account.id).state_version = 1
        session.add(
            PaperCashMovement(
                account_scope=account.scope,
                account_generation="generation-1",
                currency="TWD",
                amount=0.005,
                movement_type="adjustment",
                state_version=1,
                occurred_at=NOW,
            )
        )
    result = reconcile(account)
    assert result["status"] == "matched"
    assert any(
        row["field"].startswith("cash.") and row["within_tolerance"]
        for row in stored(account, result).difference_json["differences"]
    )


def test_freshness_owner_max_age_and_policy_digest_block_exposure(account):
    reconcile(account)
    with account.sessions() as session:
        execution = DecisionExecutionService(session)
        decision = session.get(DecisionRecord, account.decision_id)
        _, policy, _ = execution._policy_and_intents(decision)
        broker_account = session.get(PaperBrokerAccount, account.id)
        with pytest.raises(ExecutionConflictError, match="stale"):
            execution._require_reconciled(decision, policy.reconciliation, broker_account, NOW + timedelta(seconds=301))
        decision.policy_sha256 = "b" * 64
        with pytest.raises(ExecutionConflictError, match="policy changed"):
            execution._require_reconciled(decision, policy.reconciliation, broker_account, NOW)


def test_account_worker_uuid_only_loads_current_marks_server_side(account, monkeypatch):
    from poseidon.decision_loop import reconciliation
    from poseidon.workers import cpu_tasks

    trade(account)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", account.sessions)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda symbols: {"2330": 123.0})
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: account.adapter)
    captured = {}

    def receive(sessions, account_id, adapter, **kwargs):
        captured.update(account_id=account_id, prices=kwargs.get("prices"))
        return {"status": "received"}

    monkeypatch.setattr(reconciliation, "reconcile_account", receive)
    assert cpu_tasks.reconcile_paper_account.run(str(account.id)) == {"status": "received"}
    assert captured == {"account_id": account.id, "prices": {("tw_stock", "2330", "spot"): 123.0}}


def test_account_worker_missing_policy_records_unresolved(account, monkeypatch):
    from poseidon.workers import cpu_tasks

    with account.sessions() as session, session.begin():
        decision = session.get(DecisionRecord, account.decision_id)
        version = session.get(StrategyVersion, decision.strategy_version_id)
        changed = copy.deepcopy(version.policy_json)
        changed.pop("reconciliation")
        session.execute(update(StrategyVersion).where(StrategyVersion.id == version.id).values(policy_json=changed))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", account.sessions)
    original = cpu_tasks._decision_paper_adapter

    def adapter(market):
        original(market)  # Preserve the real market selection guard.
        return account.adapter

    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", adapter)
    result = cpu_tasks.reconcile_paper_account.run(str(account.id))
    assert result["status"] == "unresolved"
    assert stored(account, result).difference_json["unresolved"]


@pytest.mark.parametrize("mark", [float("nan"), float("inf"), None, True])
def test_projection_nonfinite_or_invalid_marks_are_durable_unresolved(account, mark):
    trade(account)
    result = reconcile(account, prices={("tw_stock", "2330", "spot"): mark})
    assert result["status"] == "unresolved"
    assert stored(account, result).difference_json["unresolved"]


@pytest.mark.parametrize("field", ["cash", "currency", "positions", "state_version"])
def test_bootstrap_missing_broker_snapshot_fields_are_unresolved(account, field):
    original = account.adapter.query_account_snapshot
    account.adapter.query_account_snapshot = lambda scope, generation: replace(
        original(scope, generation), **{field: None}
    )
    result = reconcile(account)
    assert result["status"] == "unresolved"
    assert stored(account, result).difference_json["unresolved"]


def test_traded_missing_broker_positions_are_durable_unresolved(account):
    trade(account)
    original = account.adapter.query_account_snapshot
    account.adapter.query_account_snapshot = lambda scope, generation: replace(
        original(scope, generation), positions=None
    )
    result = reconcile(account)
    assert result["status"] == "unresolved"
    assert stored(account, result).difference_json["unresolved"]
