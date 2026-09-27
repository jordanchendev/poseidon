"""Explicit broker reconciliation capability and durable paper truth."""

from __future__ import annotations

import os
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from threading import Barrier, Event
from typing import get_args, get_type_hints

import pandas as pd
import pytest
from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from poseidon.broker.base import BrokerAdapter, BrokerCapabilityError
from poseidon.broker.ccxt_adapter import CCXTBrokerAdapter
from poseidon.broker.paper_adapter import PaperBrokerAdapter
from poseidon.broker.perp_paper_adapter import PerpPaperAdapter
from poseidon.broker.shioaji_adapter import ShioajiBrokerAdapter
from poseidon.decision_loop.manifest import content_sha256
from poseidon.models.base import Base
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.orders.schemas import Order

NOW = datetime(2026, 9, 27, 8, 0, tzinfo=UTC)
DATABASE_URL = os.environ["POSEIDON_DATABASE_URL"]


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'paper-ledger.db'}")
    Base.metadata.create_all(
        engine,
        tables=[
            PaperBrokerAccount.__table__,
            PaperBrokerOrder.__table__,
            PaperBrokerFill.__table__,
            PaperCashMovement.__table__,
        ],
    )
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def postgres_sessions():
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, expire_on_commit=False)
    scope = f"paper:test:adapter:{uuid.uuid4().hex}"
    yield factory, scope
    with factory() as session:
        session.execute(delete(PaperCashMovement).where(PaperCashMovement.account_scope == scope))
        session.execute(delete(PaperBrokerFill).where(PaperBrokerFill.account_scope == scope))
        session.execute(delete(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == scope))
        session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == scope))
        session.commit()
    engine.dispose()


def _seed_account(sessions, *, scope="paper:pilot", generation="generation-1", cash=100_000.0, currency="TWD"):
    with sessions() as session:
        session.add(
            PaperBrokerAccount(
                account_scope=scope,
                account_generation=generation,
                opening_cash=cash,
                currency=currency,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.commit()


def _decision_order(**changes):
    values = {
        "symbol": "2330",
        "market": "tw_stock",
        "instrument": "spot",
        "action": "buy",
        "order_type": "market",
        "target_weight": 0.1,
        "quantity": 100.0,
        "strategy_name": "decision-loop",
        "broker_mode": "paper",
        "side": "long",
        "order_origin": "decision",
        "decision_id": uuid.uuid4(),
        "execution_key": uuid.uuid4(),
        "account_scope": "paper:pilot",
        "account_generation": "generation-1",
        "client_order_ref": "DL-deadbeef-0000",
        "intent_json": {
            "frozen_intent": {"action": "enter"},
            "economics": {"sizing_rules": {"quantity_rounding": "whole_share_floor"}},
        },
    }
    values.update(changes)
    values.setdefault("intent_sha256", content_sha256(values["intent_json"]))
    return Order(**values)


class _MethodsOnlyAdapter(BrokerAdapter):
    def login(self):
        return True

    def place_order(self, order):
        raise AssertionError("must not submit")

    def query_fills(self, broker_order_id, *, account_scope=None, account_generation=None):
        return []

    def query_positions(self):
        return []

    def logout(self):
        return None

    def find_order_by_client_ref(self, client_order_ref, *, account_scope, account_generation):
        return None

    def query_order(self, broker_order_id, *, account_scope, account_generation):
        return None

    def query_account_snapshot(self, account_scope, account_generation):
        return None


def test_capability_is_explicit_immutable_and_not_inferred_from_methods():
    adapter = _MethodsOnlyAdapter()

    assert adapter.supports_reconciliation is False
    assert adapter.capabilities.missing_contract == (
        "stable_client_reference",
        "find_order_by_client_reference",
        "query_order",
        "query_fills",
        "query_account_snapshot",
    )
    with pytest.raises(FrozenInstanceError):
        adapter.capabilities.query_fills = True


def test_decision_origin_is_part_of_the_order_dto():
    annotation = get_type_hints(Order)["order_origin"]
    assert "decision" in get_args(annotation)


@pytest.mark.parametrize("origin", ["stop_loss", "liquidation", "manual_emergency"])
@pytest.mark.parametrize(
    ("adapter_type", "market", "instrument"),
    [
        (PaperBrokerAdapter, "tw_stock", "spot"),
        (PerpPaperAdapter, "crypto_perp", "BTC-USDT"),
    ],
)
def test_durable_protective_adapter_requires_stored_client_ref(
    sessions,
    origin,
    adapter_type,
    market,
    instrument,
):
    order = _decision_order(
        order_origin=origin,
        market=market,
        instrument=instrument,
        action="sell",
        intent_json={"frozen_intent": {"action": "exit"}, "economics": {}},
    )

    with pytest.raises(BrokerCapabilityError, match="stored client reference"):
        adapter_type(sessions).place_order(order)


@pytest.mark.parametrize(
    ("adapter_type", "market", "instrument"),
    [
        (PaperBrokerAdapter, "tw_stock", "spot"),
        (PerpPaperAdapter, "crypto_perp", "BTC-USDT"),
    ],
)
def test_ordinary_decision_without_execution_key_still_requires_stored_client_ref(
    sessions,
    adapter_type,
    market,
    instrument,
):
    order = _decision_order(
        market=market,
        instrument=instrument,
        execution_key=None,
    )

    with pytest.raises(BrokerCapabilityError, match="stored client reference"):
        adapter_type(sessions).place_order(order)


def test_legacy_protective_without_durable_identity_keeps_paper_path(sessions, monkeypatch):
    repo = type(
        "Repo",
        (),
        {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [612.0]})},
    )()
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: repo,
    )
    order = _decision_order(
        order_origin="stop_loss",
        decision_id=None,
        execution_key=None,
        client_order_ref=None,
        account_scope=None,
        account_generation=None,
        intent_json=None,
        intent_sha256=None,
    )

    assert PaperBrokerAdapter(sessions).place_order(order)


def test_stock_paper_acceptance_is_independent_and_restart_safe(sessions, monkeypatch):
    _seed_account(sessions)
    repo = type(
        "Repo",
        (),
        {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [612.0]})},
    )()
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: repo,
    )
    order = _decision_order()

    accepted = PaperBrokerAdapter(sessions).place_decision_order(
        order,
        client_order_ref=order.client_order_ref,
    )
    restarted = PaperBrokerAdapter(sessions)
    recovered = restarted.find_order_by_client_ref(
        order.client_order_ref,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )
    fills = restarted.query_fills(
        accepted.broker_order_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )
    account = restarted.query_account_snapshot(order.account_scope, order.account_generation)
    queried = restarted.query_order(
        accepted.broker_order_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )

    assert recovered == accepted
    assert queried == accepted
    assert recovered.status == "filled"
    assert len(fills) == 1
    assert fills[0].broker_fill_id == f"{accepted.broker_order_id}-fill"
    assert account.currency == "TWD"
    assert account.cash == 38_800.0
    assert account.positions[0].quantity == 100.0
    assert account.positions[0].identity == ("tw_stock", "2330", "spot", "long")
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 1
        assert session.scalar(select(func.count()).select_from(PaperBrokerFill)) == 1
        assert session.scalar(select(func.count()).select_from(PaperCashMovement)) == 1
        ledger_order = session.scalar(select(PaperBrokerOrder))
        ledger_fill = session.scalar(select(PaperBrokerFill))
        ledger_movement = session.scalar(select(PaperCashMovement))
        ledger_account = session.scalar(select(PaperBrokerAccount))
        assert {ledger_order.state_version, ledger_fill.state_version, ledger_movement.state_version} == {
            ledger_account.state_version
        }


def test_acceptance_commit_survives_response_loss(sessions, monkeypatch):
    _seed_account(sessions)
    repo = type(
        "Repo",
        (),
        {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [500.0]})},
    )()
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: repo,
    )

    class ResponseLost(RuntimeError):
        pass

    class CommitThenLoseResponse(Session):
        def commit(self):
            super().commit()
            raise ResponseLost("response lost after broker commit")

    crashing_factory = sessionmaker(
        bind=sessions.kw["bind"],
        class_=CommitThenLoseResponse,
        expire_on_commit=False,
    )
    order = _decision_order(client_order_ref="DL-deadbeef-0001")

    with pytest.raises(ResponseLost, match="after broker commit"):
        PaperBrokerAdapter(crashing_factory).place_decision_order(
            order,
            client_order_ref=order.client_order_ref,
        )

    recovered = PaperBrokerAdapter(sessions).find_order_by_client_ref(
        order.client_order_ref,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )
    assert recovered is not None
    assert recovered.client_order_ref == order.client_order_ref


def test_perp_requires_owner_rules_before_acceptance(sessions, monkeypatch):
    _seed_account(sessions, generation="perp-1", cash=20_000.0, currency="USDT")
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_latest_price": lambda _self, _symbol: 50_000.0})(),
    )
    order = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        quantity=0.1,
        account_generation="perp-1",
        client_order_ref="DL-perp-0000",
        intent_json={"frozen_intent": {"action": "enter"}, "economics": {"sizing_rules": {}}},
    )

    with pytest.raises(BrokerCapabilityError, match="owner-frozen perp rules"):
        PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 0

    order.intent_json["economics"]["sizing_rules"] = {
        "quantity_step": 0.001,
        "contract_multiplier": 0.01,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    order.intent_sha256 = content_sha256(order.intent_json)
    accepted = PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
    account = PerpPaperAdapter(sessions).query_account_snapshot(order.account_scope, order.account_generation)

    assert accepted.status == "filled"
    assert account.currency == "USDT"
    assert account.cash == 19_950.0
    assert account.positions[0].quantity == 0.1


@pytest.mark.parametrize(
    "rules",
    [
        {
            "quantity_step": 0.1,
            "contract_multiplier": 1.0,
            "margin_semantics": "isolated_3x",
            "funding_semantics": "excluded",
        },
        {
            "quantity_step": 0.1,
            "contract_multiplier": 1.0,
            "margin_semantics": "full_notional",
            "funding_semantics": "exchange_funding",
        },
    ],
)
def test_perp_rejects_unimplemented_owner_semantics(sessions, rules):
    _seed_account(sessions, generation="perp-policy", cash=20_000.0, currency="USDT")
    order = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        quantity=0.1,
        account_generation="perp-policy",
        client_order_ref="DL-perp-policy",
        intent_json={"frozen_intent": {"action": "enter"}, "economics": {"sizing_rules": rules}},
    )

    with pytest.raises(BrokerCapabilityError, match="unsupported perp"):
        PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)


def test_perp_rejects_quantity_off_owner_step(sessions):
    _seed_account(sessions, generation="perp-step", cash=20_000.0, currency="USDT")
    rules = {
        "quantity_step": 0.1,
        "contract_multiplier": 1.0,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    order = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        quantity=0.15,
        account_generation="perp-step",
        client_order_ref="DL-perp-step",
        intent_json={"frozen_intent": {"action": "enter"}, "economics": {"sizing_rules": rules}},
    )

    with pytest.raises(BrokerCapabilityError, match="quantity_step"):
        PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)


def test_owner_cash_and_generation_are_required_before_acceptance(sessions, monkeypatch):
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )
    order = _decision_order(client_order_ref="DL-no-account-0000")

    with pytest.raises(BrokerCapabilityError, match="owner-frozen paper account generation"):
        PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 0


@pytest.mark.parametrize("adapter_type", [CCXTBrokerAdapter, ShioajiBrokerAdapter])
def test_live_adapters_fail_capability_before_place_order(adapter_type):
    adapter = object.__new__(adapter_type)
    called = False

    def place_order(_order):
        nonlocal called
        called = True
        raise AssertionError("live place_order reached")

    adapter.place_order = place_order
    order = _decision_order(broker_mode="live")

    assert adapter.supports_reconciliation is False
    with pytest.raises(BrokerCapabilityError, match="missing reconciliation contract"):
        adapter.place_decision_order(order, client_order_ref=order.client_order_ref)
    assert called is False


@pytest.mark.parametrize("adapter_type", [CCXTBrokerAdapter, ShioajiBrokerAdapter])
def test_live_place_order_signature_rejects_decision_reference(adapter_type):
    adapter = object.__new__(adapter_type)
    order = _decision_order(broker_mode="live")

    with pytest.raises(BrokerCapabilityError, match="does not support decision execution"):
        adapter_type.place_order(adapter, order, client_order_ref=order.client_order_ref)


@pytest.mark.parametrize("adapter_type", [CCXTBrokerAdapter, ShioajiBrokerAdapter])
def test_live_fill_query_signature_rejects_decision_identity(adapter_type):
    adapter = object.__new__(adapter_type)

    with pytest.raises(BrokerCapabilityError, match="does not support normalized fill recovery"):
        adapter_type.query_fills(
            adapter,
            "broker-order",
            account_scope="paper:pilot",
            account_generation="generation-1",
        )


@pytest.mark.parametrize("adapter_type", [CCXTBrokerAdapter, ShioajiBrokerAdapter])
def test_live_adapter_rejects_decision_origin_without_keyword(adapter_type):
    adapter = object.__new__(adapter_type)
    order = _decision_order(broker_mode="live")

    with pytest.raises(BrokerCapabilityError, match="does not support decision execution"):
        adapter_type.place_order(adapter, order)


def test_stock_adapter_rejects_perpetual_market_before_price_lookup(sessions, monkeypatch):
    price_lookup = False

    def from_settings():
        nonlocal price_lookup
        price_lookup = True
        raise AssertionError("price lookup reached")

    monkeypatch.setattr("poseidon.data.remote_repository.RemoteDataRepository.from_settings", from_settings)
    order = _decision_order(market="crypto_perp", instrument="BTC-USDT")

    with pytest.raises(BrokerCapabilityError, match="tw_stock/spot"):
        PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
    assert price_lookup is False


def test_exact_replay_uses_fingerprint_without_new_quote(sessions, monkeypatch):
    _seed_account(sessions)
    calls = 0

    def read_ohlcv(*_args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("replay fetched a new quote")
        return pd.DataFrame({"close": [100.0]})

    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: read_ohlcv()})(),
    )
    order = _decision_order(client_order_ref="DL-replay")
    adapter = PaperBrokerAdapter(sessions)

    first = adapter.place_decision_order(order, client_order_ref=order.client_order_ref)
    replay = PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)

    assert replay == first
    assert calls == 1
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 1
        assert session.scalar(select(PaperBrokerAccount.state_version)) == 1


def test_replay_rejects_changed_frozen_economics(sessions, monkeypatch):
    _seed_account(sessions)
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )
    original = _decision_order(client_order_ref="DL-replay-conflict")
    PaperBrokerAdapter(sessions).place_decision_order(original, client_order_ref=original.client_order_ref)
    changed_json = {"frozen_intent": {"action": "exit"}, "economics": original.intent_json["economics"]}
    changed = _decision_order(
        client_order_ref=original.client_order_ref,
        action="sell",
        intent_json=changed_json,
    )

    with pytest.raises(BrokerCapabilityError, match="fingerprint"):
        PaperBrokerAdapter(sessions).place_decision_order(changed, client_order_ref=changed.client_order_ref)
    with sessions() as session:
        assert session.scalar(select(PaperBrokerAccount.state_version)) == 1


def test_scoped_fill_lookup_ignores_conflicting_legacy_memory(sessions, monkeypatch):
    _seed_account(sessions)
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )
    order = _decision_order(client_order_ref="DL-scoped")
    accepted = PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
    restarted = PaperBrokerAdapter(sessions)
    restarted._fills[accepted.broker_order_id] = []

    fills = restarted.query_fills(
        accepted.broker_order_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )

    assert [fill.broker_fill_id for fill in fills] == [f"{accepted.broker_order_id}-fill"]


def test_perp_scoped_fill_lookup_ignores_conflicting_legacy_memory(sessions, monkeypatch):
    _seed_account(sessions, generation="perp-scoped", cash=20_000.0, currency="USDT")
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_latest_price": lambda _self, _symbol: 50_000.0})(),
    )
    rules = {
        "quantity_step": 0.1,
        "contract_multiplier": 0.01,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    order = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        quantity=0.1,
        account_generation="perp-scoped",
        client_order_ref="DL-perp-scoped",
        intent_json={"frozen_intent": {"action": "enter"}, "economics": {"sizing_rules": rules}},
    )
    accepted = PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
    restarted = PerpPaperAdapter(sessions)
    restarted._fills[accepted.broker_order_id] = []

    fills = restarted.query_fills(
        accepted.broker_order_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )

    assert [fill.broker_fill_id for fill in fills] == [f"{accepted.broker_order_id}-fill"]


def test_short_close_realizes_profit_and_versions_advance_once(sessions, monkeypatch):
    _seed_account(sessions, generation="short-1", cash=1_000.0, currency="USDT")
    prices = iter([100.0, 80.0])
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_latest_price": lambda _self, _symbol: next(prices)})(),
    )
    rules = {
        "quantity_step": 1.0,
        "contract_multiplier": 1.0,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    opening = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        side="short",
        action="sell",
        quantity=1.0,
        account_generation="short-1",
        client_order_ref="DL-short-open",
        intent_json={"frozen_intent": {"action": "enter"}, "economics": {"sizing_rules": rules}},
    )
    closing = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        side="short",
        action="buy",
        quantity=1.0,
        account_generation="short-1",
        client_order_ref="DL-short-close",
        intent_json={"frozen_intent": {"action": "exit"}, "economics": {"sizing_rules": rules}},
    )

    first = PerpPaperAdapter(sessions).place_decision_order(opening, client_order_ref=opening.client_order_ref)
    second = PerpPaperAdapter(sessions).place_decision_order(closing, client_order_ref=closing.client_order_ref)
    account = PerpPaperAdapter(sessions).query_account_snapshot("paper:pilot", "short-1")

    assert (first.state_version, second.state_version, account.state_version) == (1, 2, 2)
    assert account.cash == 1_020.0
    assert account.positions == ()


def test_short_close_at_total_loss_records_no_zero_cash_movement(sessions, monkeypatch):
    _seed_account(sessions, generation="short-total-loss", cash=100.0, currency="USDT")
    prices = iter([100.0, 200.0])
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_latest_price": lambda _self, _symbol: next(prices)})(),
    )
    rules = {
        "quantity_step": 1.0,
        "contract_multiplier": 1.0,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    orders = [
        _decision_order(
            symbol="BTC",
            market="crypto_perp",
            instrument="BTC-USDT",
            side="short",
            action="sell" if index == 0 else "buy",
            quantity=1.0,
            account_generation="short-total-loss",
            client_order_ref=f"DL-short-total-loss-{index}",
            intent_json={
                "frozen_intent": {"action": "enter" if index == 0 else "exit"},
                "economics": {"sizing_rules": rules},
            },
        )
        for index in range(2)
    ]

    snapshots = [
        PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
        for order in orders
    ]

    account = PerpPaperAdapter(sessions).query_account_snapshot("paper:pilot", "short-total-loss")
    assert [snapshot.state_version for snapshot in snapshots] == [1, 2]
    assert account.state_version == 2
    assert account.cash == 0.0
    assert account.positions == ()
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperCashMovement)) == 1


def test_fractional_perp_snapshot_folds_decimal_before_dto(sessions, monkeypatch):
    _seed_account(sessions, generation="fractional", cash=1_000.0, currency="USDT")
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_latest_price": lambda _self, _symbol: 100.0})(),
    )
    rules = {
        "quantity_step": 0.1,
        "contract_multiplier": 1.0,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    for index, quantity in enumerate((0.1, 0.2)):
        order = _decision_order(
            symbol="BTC",
            market="crypto_perp",
            instrument="BTC-USDT",
            quantity=quantity,
            account_generation="fractional",
            client_order_ref=f"DL-fractional-{index}",
            intent_json={
                "frozen_intent": {"action": "enter" if index == 0 else "add"},
                "economics": {"sizing_rules": rules},
            },
        )
        PerpPaperAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)

    account = PerpPaperAdapter(sessions).query_account_snapshot("paper:pilot", "fractional")
    assert account.positions[0].quantity == 0.3
    assert account.cash == 970.0


def test_fractional_cash_capacity_folds_decimal_during_acceptance(sessions, monkeypatch):
    _seed_account(sessions, generation="fractional-cash", cash=0.3)
    prices = iter([0.1, 0.2])
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type(
            "Repo",
            (),
            {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [next(prices)]})},
        )(),
    )
    orders = [
        _decision_order(
            account_generation="fractional-cash",
            client_order_ref=f"DL-fractional-cash-{index}",
            quantity=1.0,
            intent_json={
                "frozen_intent": {"action": "enter" if index == 0 else "add"},
                "economics": {"sizing_rules": {"quantity_rounding": "whole_share_floor"}},
            },
        )
        for index in range(2)
    ]

    for order in orders:
        PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)

    account = PaperBrokerAdapter(sessions).query_account_snapshot("paper:pilot", "fractional-cash")
    assert account.state_version == 2
    assert account.cash == 0.0


@pytest.mark.parametrize("timestamp_order", ["regressed", "equal"])
def test_inventory_cost_fold_uses_state_version_not_wall_clock(sessions, monkeypatch, timestamp_order):
    _seed_account(sessions, generation=f"ledger-{timestamp_order}", cash=1_000.0)
    prices = iter([100.0, 110.0, 90.0])
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type(
            "Repo",
            (),
            {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [next(prices)]})},
        )(),
    )
    generation = f"ledger-{timestamp_order}"
    opening = _decision_order(account_generation=generation, client_order_ref="DL-ledger-open", quantity=2.0)
    closing = _decision_order(
        account_generation=generation,
        client_order_ref="DL-ledger-close",
        action="sell",
        quantity=1.0,
        intent_json={
            "frozen_intent": {"action": "reduce"},
            "economics": opening.intent_json["economics"],
        },
    )
    adding = _decision_order(
        account_generation=generation,
        client_order_ref="DL-ledger-add",
        quantity=1.0,
        intent_json={
            "frozen_intent": {"action": "add"},
            "economics": opening.intent_json["economics"],
        },
    )
    adapter = PaperBrokerAdapter(sessions)
    adapter.place_decision_order(opening, client_order_ref=opening.client_order_ref)
    adapter.place_decision_order(closing, client_order_ref=closing.client_order_ref)
    with sessions() as session:
        fills = session.scalars(select(PaperBrokerFill).order_by(PaperBrokerFill.state_version)).all()
        if timestamp_order == "regressed":
            fills[0].fill_time = NOW + timedelta(seconds=1)
            fills[1].fill_time = NOW
        else:
            fills[0].fill_time = fills[1].fill_time = NOW
            fills[0].id = uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
            fills[1].id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        session.commit()

    accepted = adapter.place_decision_order(adding, client_order_ref=adding.client_order_ref)
    account = adapter.query_account_snapshot("paper:pilot", generation)

    assert accepted.state_version == account.state_version == 3
    assert account.positions[0].quantity == 2.0


def test_over_close_is_rejected_by_independent_inventory(sessions, monkeypatch):
    _seed_account(sessions, generation="over-close", cash=1_000.0)
    prices = iter([100.0, 110.0])
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [next(prices)]})})(),
    )
    opening = _decision_order(account_generation="over-close", client_order_ref="DL-long-open", quantity=1.0)
    closing_json = {"frozen_intent": {"action": "exit"}, "economics": opening.intent_json["economics"]}
    closing = _decision_order(
        account_generation="over-close",
        client_order_ref="DL-long-over-close",
        action="sell",
        quantity=2.0,
        intent_json=closing_json,
    )
    PaperBrokerAdapter(sessions).place_decision_order(opening, client_order_ref=opening.client_order_ref)

    with pytest.raises(BrokerCapabilityError, match="exceeds independent broker inventory"):
        PaperBrokerAdapter(sessions).place_decision_order(closing, client_order_ref=closing.client_order_ref)

    account = PaperBrokerAdapter(sessions).query_account_snapshot("paper:pilot", "over-close")
    assert account.cash == 900.0
    assert account.positions[0].quantity == 1.0
    assert account.state_version == 1
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 1


def test_failure_after_order_flush_rolls_back_whole_acceptance(sessions, monkeypatch):
    _seed_account(sessions, generation="rollback")
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )

    class InjectedFailure(RuntimeError):
        pass

    class FailBeforeCommit(Session):
        def commit(self):
            raise InjectedFailure("after order flush")

    failing_factory = sessionmaker(bind=sessions.kw["bind"], class_=FailBeforeCommit, expire_on_commit=False)
    order = _decision_order(account_generation="rollback", client_order_ref="DL-rollback")

    with pytest.raises(InjectedFailure, match="after order flush"):
        PaperBrokerAdapter(failing_factory).place_decision_order(order, client_order_ref=order.client_order_ref)

    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 0
        assert session.scalar(select(func.count()).select_from(PaperBrokerFill)) == 0
        assert session.scalar(select(func.count()).select_from(PaperCashMovement)) == 0
        assert session.scalar(select(PaperBrokerAccount.state_version)) == 0


def test_fresh_perp_adapter_recovers_after_committed_response_loss(sessions, monkeypatch):
    _seed_account(sessions, generation="perp-crash", cash=20_000.0, currency="USDT")
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_latest_price": lambda _self, _symbol: 50_000.0})(),
    )

    class ResponseLost(RuntimeError):
        pass

    class CommitThenLoseResponse(Session):
        def commit(self):
            super().commit()
            raise ResponseLost("perp response lost after broker commit")

    crashing_factory = sessionmaker(
        bind=sessions.kw["bind"],
        class_=CommitThenLoseResponse,
        expire_on_commit=False,
    )
    rules = {
        "quantity_step": 0.1,
        "contract_multiplier": 0.01,
        "margin_semantics": "full_notional",
        "funding_semantics": "excluded",
    }
    order = _decision_order(
        symbol="BTC",
        market="crypto_perp",
        instrument="BTC-USDT",
        quantity=0.1,
        account_generation="perp-crash",
        client_order_ref="DL-perp-crash",
        intent_json={"frozen_intent": {"action": "enter"}, "economics": {"sizing_rules": rules}},
    )

    with pytest.raises(ResponseLost, match="after broker commit"):
        PerpPaperAdapter(crashing_factory).place_decision_order(order, client_order_ref=order.client_order_ref)

    restarted = PerpPaperAdapter(sessions)
    recovered = restarted.find_order_by_client_ref(
        order.client_order_ref,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )
    fills = restarted.query_fills(
        recovered.broker_order_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
    )
    account = restarted.query_account_snapshot(order.account_scope, order.account_generation)
    assert recovered.state_version == fills[0].state_version == account.state_version == 1
    assert account.cash == 19_950.0
    assert account.positions[0].quantity == 0.1


def test_postgres_same_reference_race_has_one_acceptance(postgres_sessions, monkeypatch):
    sessions, scope = postgres_sessions
    _seed_account(sessions, scope=scope, generation="same-ref", cash=1_000.0)
    barrier = Barrier(2)

    def price(*_args):
        barrier.wait()
        return pd.DataFrame({"close": [100.0]})

    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: price()})(),
    )
    order = _decision_order(
        account_scope=scope,
        account_generation="same-ref",
        client_order_ref="DL-pg-same-ref",
        quantity=1.0,
    )

    def submit():
        return PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: submit(), range(2)))

    assert results[0] == results[1]
    with sessions() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == scope)
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count()).select_from(PaperBrokerFill).where(PaperBrokerFill.account_scope == scope)
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count()).select_from(PaperCashMovement).where(PaperCashMovement.account_scope == scope)
            )
            == 1
        )
        assert (
            session.scalar(select(PaperBrokerAccount.state_version).where(PaperBrokerAccount.account_scope == scope))
            == 1
        )


def test_postgres_distinct_orders_serialize_cash_capacity(postgres_sessions, monkeypatch):
    sessions, scope = postgres_sessions
    _seed_account(sessions, scope=scope, generation="cash-race", cash=150.0)
    barrier = Barrier(2)

    def price(*_args):
        barrier.wait()
        return pd.DataFrame({"close": [100.0]})

    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: price()})(),
    )
    orders = [
        _decision_order(
            account_scope=scope,
            account_generation="cash-race",
            client_order_ref=f"DL-pg-cash-{symbol}",
            symbol=symbol,
            quantity=1.0,
        )
        for symbol in ("2330", "2317")
    ]

    def submit(order):
        try:
            return PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
        except BrokerCapabilityError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, orders))

    assert sum(isinstance(result, BrokerCapabilityError) for result in results) == 1
    assert "exceeds durable account cash" in str(next(result for result in results if isinstance(result, Exception)))
    with sessions() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == scope)
            )
            == 1
        )
        assert (
            session.scalar(select(PaperBrokerAccount.state_version).where(PaperBrokerAccount.account_scope == scope))
            == 1
        )


def test_postgres_distinct_closes_serialize_independent_inventory(postgres_sessions, monkeypatch):
    sessions, scope = postgres_sessions
    _seed_account(sessions, scope=scope, generation="close-race", cash=1_000.0)
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )
    opening = _decision_order(
        account_scope=scope,
        account_generation="close-race",
        client_order_ref="DL-pg-close-open",
        quantity=1.0,
    )
    PaperBrokerAdapter(sessions).place_decision_order(opening, client_order_ref=opening.client_order_ref)

    barrier = Barrier(2)

    def close_price(*_args):
        barrier.wait()
        return pd.DataFrame({"close": [110.0]})

    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: close_price()})(),
    )
    closing_json = {"frozen_intent": {"action": "exit"}, "economics": opening.intent_json["economics"]}
    closes = [
        _decision_order(
            account_scope=scope,
            account_generation="close-race",
            client_order_ref=f"DL-pg-close-{index}",
            action="sell",
            quantity=1.0,
            intent_json=closing_json,
        )
        for index in range(2)
    ]

    def submit(order):
        try:
            return PaperBrokerAdapter(sessions).place_decision_order(order, client_order_ref=order.client_order_ref)
        except BrokerCapabilityError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, closes))

    failures = [result for result in results if isinstance(result, BrokerCapabilityError)]
    assert len(failures) == 1
    assert str(failures[0]) == "close exceeds independent broker inventory"
    account = PaperBrokerAdapter(sessions).query_account_snapshot(scope, "close-race")
    assert account.state_version == 2
    assert account.positions == ()
    assert account.cash == 1_010.0
    with sessions() as session:
        assert (
            session.scalar(
                select(func.count()).select_from(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == scope)
            )
            == 2
        )
        assert (
            session.scalar(
                select(func.count()).select_from(PaperBrokerFill).where(PaperBrokerFill.account_scope == scope)
            )
            == 2
        )
        assert (
            session.scalar(
                select(func.count()).select_from(PaperCashMovement).where(PaperCashMovement.account_scope == scope)
            )
            == 2
        )


def test_postgres_snapshot_waits_for_account_writer_lock(postgres_sessions):
    sessions, scope = postgres_sessions
    _seed_account(sessions, scope=scope, generation="snapshot-lock", cash=1_000.0)
    pids: queue.Queue[int] = queue.Queue()

    def snapshot_factory():
        session = sessions()
        pids.put(session.scalar(text("SELECT pg_backend_pid()")))
        return session

    locker = sessions()
    locker.scalar(
        select(PaperBrokerAccount)
        .where(
            PaperBrokerAccount.account_scope == scope,
            PaperBrokerAccount.account_generation == "snapshot-lock",
        )
        .with_for_update()
    )
    locker_pid = locker.scalar(text("SELECT pg_backend_pid()"))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                PaperBrokerAdapter(snapshot_factory).query_account_snapshot,
                scope,
                "snapshot-lock",
            )
            snapshot_pid = pids.get(timeout=5)
            deadline = time.monotonic() + 5
            blocked = False
            while time.monotonic() < deadline:
                with sessions() as observer:
                    blockers = observer.scalar(text("SELECT pg_blocking_pids(:pid)"), {"pid": snapshot_pid})
                if locker_pid in blockers:
                    blocked = True
                    break
                if future.done():
                    break
                Event().wait(0.01)
            assert blocked, "snapshot query did not share the account-row lock order"
            locker.commit()
            snapshot = future.result(timeout=5)
    finally:
        locker.rollback()
        locker.close()

    assert snapshot.state_version == 0
    assert snapshot.cash == 1_000.0


def test_decision_order_without_client_reference_cannot_use_legacy_path(sessions):
    order = _decision_order(client_order_ref=None)

    with pytest.raises(BrokerCapabilityError, match="stored client reference"):
        PaperBrokerAdapter(sessions).place_order(order)


def test_legacy_origin_cannot_opt_into_the_decision_ledger(sessions):
    order = _decision_order(order_origin="signal")

    with pytest.raises(BrokerCapabilityError, match="order_origin=decision"):
        PaperBrokerAdapter(sessions).place_order(order, client_order_ref=order.client_order_ref)
