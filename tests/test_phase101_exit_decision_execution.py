"""Real PostgreSQL proof that producer exit decisions use the unchanged execution path (Phase 101-02).

A released strategy holds one open loop lot. A rule ``close`` on the next frozen bar becomes an
``exit`` decision through ``ProducerService``; after human approval the ordinary claimer selects it,
``claim_execution`` claims it, and ``DecisionExecutionService.materialize`` sells exactly the settled
unreserved lot quantity with zero reserved cash. ``execution.py`` and ``cpu_tasks.py`` are unchanged.

Each test owns a unique account scope/generation and a fresh released strategy, so rows never
collide; the pointer is moved only through ``ReleaseService`` (no trigger bypass).
"""

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionService
from poseidon.decision_loop.execution import DecisionExecutionService, ExecutionConflictError
from poseidon.decision_loop.manifest import ManifestService, content_sha256
from poseidon.decision_loop.producer import ProducerService
from poseidon.decision_loop.releases import ReleaseService
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.position_lot import PositionLot
from tests.test_decision_service import synthetic_reconciliation_policy
from tests.test_strategy_release import NOW as RELEASE_TIME
from tests.test_strategy_release import _seed_gate_graph, release_manager

DATABASE_URL = os.environ["POSEIDON_REAL_DATABASE_URL"]
ENGINE = create_engine(DATABASE_URL)
if ENGINE.dialect.name != "postgresql":
    raise RuntimeError("test_phase101_exit_decision_execution.py requires PostgreSQL")

LOT_QUANTITY = 1000.0
PRICE = 100.0
PRICES = {("tw_stock", "2330", "spot"): PRICE}
NAV = 1_000_000.0
LAST_BAR = datetime(2026, 10, 13, tzinfo=UTC)
ENTRY_AS_OF = datetime(2026, 10, 13, 3, 0, tzinfo=UTC)
EXIT_AS_OF = datetime(2026, 10, 13, 6, 0, tzinfo=UTC)
APPROVE_TIME = EXIT_AS_OF + timedelta(minutes=10)
CANDIDATE_TIME = EXIT_AS_OF + timedelta(minutes=15)
CLAIM_TIME = EXIT_AS_OF + timedelta(minutes=20)
MATERIALIZE_TIME = EXIT_AS_OF + timedelta(minutes=25)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def long_signal(_executable, _bars):
    return [{"action": "long", "confidence": 0.9, "quantity_pct": 0.1}]


def close_signal(_executable, _bars):
    return [{"action": "close", "confidence": 1.0, "quantity_pct": 1.0}]


@dataclass(frozen=True)
class ExitSeed:
    marker: str
    account: str
    generation: str
    strategy_id: uuid.UUID
    version_id: uuid.UUID
    entry_decision_id: uuid.UUID
    lot_id: uuid.UUID

    @property
    def worker(self):
        return AuthPrincipal("decision-worker:phase101-exit", frozenset({"decision-worker"}), frozenset({self.account}))

    @property
    def manager(self):
        return AuthPrincipal("human:phase101-manager", frozenset({"portfolio_manager"}), frozenset({self.account}))


def _manifest_request(account, marker, as_of):
    closes = [100.0, 101.0, 102.0, 101.5, 100.5]
    start = LAST_BAR - timedelta(days=len(closes) - 1)
    bars = [
        {
            "time": _iso(start + timedelta(days=index)),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
        }
        for index, close in enumerate(closes)
    ]
    payload = {"symbol": "2330", "market": "tw_stock", "interval": "1d", "bars": bars, "marker": marker}
    return {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": _iso(as_of),
        "account_scope": account,
        "universe_id": "tw-stock-pilot-v1",
        "capability_json": {"ohlcv": True},
        "required_data": {"ohlcv": {"max_age_seconds": 86400}},
        "evidence": [
            {
                "id": f"ohlcv:2330:{marker}",
                "kind": "ohlcv",
                "source_uri": f"fixture://phase101-exit/{marker}",
                "event_time": bars[-1]["time"],
                "available_at": bars[-1]["time"],
                "recorded_at": _iso(as_of),
                "payload": payload,
                "content_sha256": content_sha256(payload),
            }
        ],
    }


def _produce(session, seed, evaluator, as_of):
    manifest = ManifestService(session).freeze(_manifest_request(seed.account, seed.marker, as_of))
    return ProducerService(session, signal_evaluator=evaluator).produce(
        seed.version_id, manifest.id, decision_as_of=_iso(as_of), principal=seed.worker, mode="decision"
    )


def _approve(session, seed, decision_id, now):
    DecisionService(session).approve(
        decision_id,
        {"expected_revision": 1},
        principal=seed.manager,
        idempotency_key=f"phase101-exit-approve-{decision_id}",
        now=now,
    )


def _seed_open_lot(session, seed_values, entry_decision_id):
    account, generation = seed_values
    opened = ENTRY_AS_OF + timedelta(minutes=40)
    order_id, fill_id, lot_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session.add(
        OrderRecord(
            id=order_id,
            strategy_name="phase101-exit-opening-lot",
            symbol="2330",
            market="tw_stock",
            action="buy",
            order_type="market",
            target_weight=0.1,
            quantity=LOT_QUANTITY,
            price=PRICE,
            side="long",
            status="filled",
            broker_mode="paper",
            order_origin="decision",
            decision_id=entry_decision_id,
            account_scope=account,
            account_generation=generation,
            client_order_ref=f"DL-source-{order_id.hex}",
            instrument="spot",
            created_at=opened,
            updated_at=opened,
        )
    )
    session.flush()
    session.add(
        OrderFillRecord(
            id=fill_id,
            order_id=order_id,
            fill_price=PRICE,
            fill_quantity=LOT_QUANTITY,
            fill_time=opened,
            broker_fill_id=f"phase101-exit-source-{fill_id.hex}",
            projection_status="applied",
            created_at=opened,
        )
    )
    session.flush()
    session.add(
        PositionLot(
            id=lot_id,
            account_scope=account,
            account_generation=generation,
            market="tw_stock",
            symbol="2330",
            instrument="spot",
            side="long",
            opening_fill_id=fill_id,
            opening_decision_id=entry_decision_id,
            original_quantity=LOT_QUANTITY,
            open_quantity=LOT_QUANTITY,
            reserved_close_quantity=0.0,
            cost_basis_json={"price": PRICE},
            opened_at=opened,
            created_at=opened,
            updated_at=opened,
        )
    )
    session.flush()
    return lot_id


@pytest.fixture
def exit_seed():
    """Released pilot-shaped version, matched paper account, and one 1000-share lot from an enter decision."""
    marker = uuid.uuid4().hex
    account = f"paper:phase101-exit:{marker}"
    generation = f"phase101-exit-generation-{marker}"
    with Session(ENGINE) as session, session.begin():
        graph = _seed_gate_graph(
            session,
            policy_changes={
                "account_scope": account,
                "reconciliation": synthetic_reconciliation_policy(account_generation=generation),
            },
        )
        ReleaseService(session).release(
            graph.strategy_id,
            graph.candidate_id,
            graph.campaign_id,
            graph.review_id,
            graph.reconciliation_id,
            expected_revision=0,
            expected_active_version_id=None,
            principal=release_manager(account),
            idempotency_key=f"phase101-exit-release-{marker}",
            reason="phase101-02 exit execution fixture",
            now=RELEASE_TIME,
        )
        session.add(
            PaperBrokerAccount(
                account_scope=account,
                account_generation=generation,
                opening_cash=100000.0,
                currency="TWD",
                created_at=RELEASE_TIME,
                updated_at=RELEASE_TIME,
            )
        )
        session.flush()
        seed = ExitSeed(marker, account, generation, graph.strategy_id, graph.candidate_id, uuid.uuid4(), uuid.uuid4())

        # The prior enter decision that opened the lot: produced, approved, and claimed (no longer claimable).
        entered = _produce(session, seed, long_signal, ENTRY_AS_OF)
        entry_decision_id = uuid.UUID(entered["decision_id"])
        entry = session.get(DecisionRecord, entry_decision_id)
        assert entry.final_json["final_action"] == "enter"
        _approve(session, seed, entry_decision_id, ENTRY_AS_OF + timedelta(minutes=10))
        DecisionService(session).claim_execution(
            entry_decision_id, 2, principal=seed.worker, now=ENTRY_AS_OF + timedelta(minutes=20)
        )
        lot_id = _seed_open_lot(session, (account, generation), entry_decision_id)
        seed = ExitSeed(marker, account, generation, graph.strategy_id, graph.candidate_id, entry_decision_id, lot_id)
    return seed


def _approved_exit(seed):
    with Session(ENGINE) as session, session.begin():
        produced = _produce(session, seed, close_signal, EXIT_AS_OF)
        decision_id = uuid.UUID(produced["decision_id"])
        _approve(session, seed, decision_id, APPROVE_TIME)
    return decision_id


def _claim(seed, decision_id):
    with Session(ENGINE) as session, session.begin():
        decision = session.get(DecisionRecord, decision_id)
        return DecisionService(session).claim_execution(
            decision_id, decision.revision, principal=seed.worker, now=CLAIM_TIME
        )


def _materialize(seed, decision_id):
    with Session(ENGINE) as session, session.begin():
        return DecisionExecutionService(session).materialize(
            decision_id, principal=seed.worker, account_nav=NAV, prices=PRICES, now=MATERIALIZE_TIME
        )


def _decision_orders(session, decision_id):
    return session.scalars(select(OrderRecord).where(OrderRecord.decision_id == decision_id)).all()


def test_producer_exit_decision_validates_and_is_approved(exit_seed):
    with Session(ENGINE) as session, session.begin():
        produced = _produce(session, exit_seed, close_signal, EXIT_AS_OF)
        decision = session.get(DecisionRecord, uuid.UUID(produced["decision_id"]))
        assert decision.status == "pending_approval"
        assert decision.original_json["final_action"] == "exit"
        [intent] = decision.original_json["order_intents"]
        assert (intent["action"], intent["side"], intent["target_weight"]) == ("exit", "long", 0.0)
        assert "exit" in decision.risk_snapshot_json["allowed_actions"]
        assert decision.portfolio_snapshot_json["positions"] == [
            {
                "market": "tw_stock",
                "symbol": "2330",
                "instrument": "spot",
                "side": "long",
                "open_quantity": LOT_QUANTITY,
            }
        ]
        _approve(session, exit_seed, decision.id, APPROVE_TIME)
        session.refresh(decision)
        assert decision.status == "approved"
        assert decision.final_json["final_action"] == "exit"


def test_ordinary_claimer_selects_the_approved_exit_decision(exit_seed, monkeypatch):
    from poseidon.workers import cpu_tasks

    decision_id = _approved_exit(exit_seed)
    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        SimpleNamespace(
            decision_loop_approved_account_scope=exit_seed.account,
            decision_loop_approved_account_generation=exit_seed.generation,
        ),
    )
    with Session(ENGINE) as session:
        candidate = cpu_tasks._approved_ordinary_candidate(session, "tw_stock", CANDIDATE_TIME)
        assert candidate is not None
        assert candidate.id == decision_id
        assert candidate.final_json["final_action"] == "exit"


def test_claimed_exit_materializes_a_full_lot_sell_with_zero_reserved_cash(exit_seed):
    decision_id = _approved_exit(exit_seed)
    claimed = _claim(exit_seed, decision_id)
    assert claimed["status"] == "execution_claimed"

    response = _materialize(exit_seed, decision_id)

    assert response["status"] == "pending_submit"
    with Session(ENGINE) as session:
        [order] = _decision_orders(session, decision_id)
        assert order.intent_json["frozen_intent"]["action"] == "exit"
        assert order.intent_json["frozen_intent"]["side"] == "long"
        assert order.intent_json["economics"]["materialized_quantity"] == LOT_QUANTITY
        assert order.intent_json["economics"]["settled_quantity"] == LOT_QUANTITY
        assert (order.action, order.side, order.quantity, order.reserved_quantity) == (
            "sell",
            "long",
            LOT_QUANTITY,
            LOT_QUANTITY,
        )
        assert order.reserved_cash_json == {"currency": "TWD", "amount": 0.0}
        assert order.status == "pending_submit"
        assert str(order.execution_key) == claimed["execution_key"]
        lot = session.get(PositionLot, exit_seed.lot_id)
        assert lot.open_quantity == LOT_QUANTITY
        assert lot.reserved_close_quantity == LOT_QUANTITY


def test_rematerializing_the_same_claim_creates_no_duplicate_order(exit_seed):
    decision_id = _approved_exit(exit_seed)
    _claim(exit_seed, decision_id)
    first = _materialize(exit_seed, decision_id)

    replay = _materialize(exit_seed, decision_id)

    assert replay == first
    with Session(ENGINE) as session:
        orders = _decision_orders(session, decision_id)
        assert len(orders) == 1
        assert len({order.client_order_ref for order in orders}) == 1
        assert session.get(PositionLot, exit_seed.lot_id).reserved_close_quantity == LOT_QUANTITY


def test_exit_with_no_open_lot_is_rejected_without_orders(exit_seed):
    decision_id = _approved_exit(exit_seed)
    _claim(exit_seed, decision_id)
    # The lot closed between approval and materialization (e.g. a protective exit filled first).
    with Session(ENGINE) as session, session.begin():
        lot = session.get(PositionLot, exit_seed.lot_id)
        lot.open_quantity = 0.0

    with pytest.raises(ExecutionConflictError, match="reduce/exit requires a negative quantity delta"):
        _materialize(exit_seed, decision_id)

    with Session(ENGINE) as session:
        assert _decision_orders(session, decision_id) == []
        decision = session.get(DecisionRecord, decision_id)
        assert decision.status == "execution_claimed"
        assert session.get(PositionLot, exit_seed.lot_id).reserved_close_quantity == 0.0
