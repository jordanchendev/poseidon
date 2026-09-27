"""Real PostgreSQL row-lock proof for decision execution claims."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionService
from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.execution import ExecutionConflictError, internal_state_watermark
from poseidon.decision_loop.manifest import ManifestService, content_sha256
from poseidon.decision_loop.transactions import materialize_order_intents
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.data_manifest import DataManifest
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import (
    executable_intent,
    manifest_request,
    synthetic_policy,
    synthetic_reconciliation_policy,
)

DATABASE_URL = os.environ["POSEIDON_DATABASE_URL"]
ENGINE = create_engine(DATABASE_URL)
if ENGINE.dialect.name != "postgresql":
    raise RuntimeError("test_execution_concurrency_postgres.py requires PostgreSQL")

CLAIM_TIME = datetime(2026, 9, 26, 12, 45, tzinfo=UTC)


@dataclass(frozen=True)
class ClaimSeed:
    account: str
    decision_ids: tuple[uuid.UUID, uuid.UUID]
    strategy_ids: tuple[uuid.UUID, uuid.UUID]
    version_ids: tuple[uuid.UUID, uuid.UUID]
    manifest_id: uuid.UUID
    run_ids: tuple[uuid.UUID, uuid.UUID]

    @property
    def worker(self):
        return AuthPrincipal("service:phase98-claim", frozenset({"decision-worker"}), frozenset({self.account}))


@pytest.fixture
def claim_seed():
    marker = uuid.uuid4().hex
    account = f"paper:phase98:claim:{marker}"
    universe_id = f"phase98-claim-{marker}"
    strategy_ids = []
    version_ids = []
    run_ids = []
    decision_ids = []
    with Session(ENGINE) as session:
        request = manifest_request()
        request["account_scope"] = account
        request["universe_id"] = universe_id
        request["evidence"][0]["payload"]["marker"] = marker
        request["evidence"][0]["content_sha256"] = content_sha256(request["evidence"][0]["payload"])
        manifest = ManifestService(session).freeze(request)

        for index in range(2):
            strategy = StrategyRecord(
                name=f"phase98-claim-{marker}-{index}",
                strategy_type="technical",
                symbol="2330",
                market="tw_stock",
                interval="1d",
            )
            session.add(strategy)
            session.flush()
            policy_json = synthetic_policy(
                account_scope=account,
                universe_id=universe_id,
                reconciliation=synthetic_reconciliation_policy(account_generation=f"phase98-claim-generation-{marker}"),
            )
            version = StrategyVersion(
                strategy_id=strategy.id,
                version_no=1,
                config_json={"marker": marker, "index": index},
                policy_json=policy_json,
                artifact_json={},
                content_sha256=strategy_version_digest(
                    {"marker": marker, "index": index},
                    policy_json,
                    {},
                ),
            )
            session.add(version)
            session.flush()
            universe = [{"symbol": "2330", "market": "tw_stock", "instrument": "spot"}]
            run = EvaluationService(session).evaluate_run(
                version.id,
                manifest.id,
                universe,
                [
                    {
                        **universe[0],
                        "status": "evaluated",
                        "recommendation_json": {"research_status": "not_required", "side": "long"},
                        "reason_codes": [],
                        "valid_until": "2026-09-26T15:00:00Z",
                    }
                ],
            )
            session.flush()
            snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=run.id).one()
            decision = DecisionService(session).create_decision(
                run.id,
                principal=AuthPrincipal(
                    "service:phase98-create",
                    frozenset({"decision-worker"}),
                    frozenset({account}),
                ),
                account_scope=account,
                original_json={
                    "selected_evaluation_ids": [str(snapshot.id)],
                    "final_action": "enter",
                    "order_intents": [executable_intent(str(snapshot.id))],
                },
                portfolio_snapshot_json={"cash": 100000.0, "positions": []},
                risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
                now=datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
            )
            DecisionService(session).approve(
                decision.id,
                {"expected_revision": 1},
                principal=AuthPrincipal(
                    "human:phase98-manager",
                    frozenset({"portfolio_manager"}),
                    frozenset({account}),
                ),
                idempotency_key=f"phase98-claim-approve-{index}",
                now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
            )
            strategy_ids.append(strategy.id)
            version_ids.append(version.id)
            run_ids.append(run.id)
            decision_ids.append(decision.id)
        value = ClaimSeed(
            account,
            tuple(decision_ids),
            tuple(strategy_ids),
            tuple(version_ids),
            manifest.id,
            tuple(run_ids),
        )
        session.commit()

    yield value

    with Session(ENGINE) as session:
        session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id.in_(value.decision_ids)))
        session.execute(delete(DecisionRecord).where(DecisionRecord.id.in_(value.decision_ids)))
        session.execute(delete(EvaluationSnapshot).where(EvaluationSnapshot.evaluation_run_id.in_(value.run_ids)))
        session.execute(delete(EvaluationRun).where(EvaluationRun.id.in_(value.run_ids)))
        session.execute(delete(StrategyVersion).where(StrategyVersion.id.in_(value.version_ids)))
        session.execute(delete(DataManifest).where(DataManifest.id == value.manifest_id))
        session.execute(delete(StrategyRecord).where(StrategyRecord.id.in_(value.strategy_ids)))
        session.commit()


def test_concurrent_named_claim_has_one_key_revision_and_event(claim_seed):
    barrier = Barrier(2)

    def claim():
        with Session(ENGINE) as session:
            barrier.wait(timeout=10)
            try:
                result = DecisionService(session).claim_execution(
                    claim_seed.decision_ids[0],
                    2,
                    principal=claim_seed.worker,
                    now=CLAIM_TIME,
                )
                session.commit()
                return "claimed", result["execution_key"]
            except DecisionConflictError as error:
                session.rollback()
                return "conflict", str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=20) for future in [pool.submit(claim), pool.submit(claim)]]

    assert sorted(status for status, _ in results) == ["claimed", "conflict"]
    with Session(ENGINE) as session:
        decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
        assert decision.status == "execution_claimed"
        assert decision.revision == 3
        assert decision.execution_key is not None
        events = session.query(DecisionEvent).filter_by(
            decision_id=decision.id,
            event_type="execution_claimed",
        )
        assert events.count() == 1
        assert events.one().expected_revision == 2


def test_claim_next_skips_an_older_ineligible_approved_row(claim_seed):
    with Session(ENGINE) as session:
        ineligible = session.get(DecisionRecord, claim_seed.decision_ids[0])
        eligible = session.get(DecisionRecord, claim_seed.decision_ids[1])
        ineligible.final_json = {
            "selected_evaluation_ids": ineligible.final_json["selected_evaluation_ids"],
            "final_action": "hold",
        }
        ineligible.created_at = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
        eligible.created_at = datetime(2026, 9, 26, 12, 1, tzinfo=UTC)
        session.commit()

    with Session(ENGINE) as session:
        result = DecisionService(session).claim_next_approved(
            claim_seed.account,
            principal=claim_seed.worker,
            now=CLAIM_TIME,
        )
        session.commit()

    assert result["decision_id"] == str(claim_seed.decision_ids[1])
    with Session(ENGINE) as session:
        ineligible = session.get(DecisionRecord, claim_seed.decision_ids[0])
        assert (ineligible.status, ineligible.execution_key) == ("approved", None)


def test_skip_locked_claims_distinct_eligible_rows_without_waiting(claim_seed):
    selected = Barrier(2)

    def claim_next():
        with Session(ENGINE) as session:
            result = DecisionService(session).claim_next_approved(
                claim_seed.account,
                principal=claim_seed.worker,
                now=CLAIM_TIME,
            )
            selected.wait(timeout=10)
            session.commit()
            return result["decision_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed_ids = [future.result(timeout=20) for future in [pool.submit(claim_next), pool.submit(claim_next)]]

    assert set(claimed_ids) == {str(decision_id) for decision_id in claim_seed.decision_ids}
    with Session(ENGINE) as session:
        decisions = session.scalars(select(DecisionRecord).where(DecisionRecord.id.in_(claim_seed.decision_ids))).all()
        assert all(decision.status == "execution_claimed" for decision in decisions)
        assert len({decision.execution_key for decision in decisions}) == 2
        assert (
            session.query(DecisionEvent)
            .filter(
                DecisionEvent.decision_id.in_(claim_seed.decision_ids),
                DecisionEvent.event_type == "execution_claimed",
            )
            .count()
            == 2
        )


def test_same_account_materialization_serializes_reservations(claim_seed):
    principal = claim_seed.worker
    reduce_decision_ids = []
    source_order_id = uuid.uuid4()
    source_fill_id = uuid.uuid4()
    lot_id = uuid.uuid4()
    with Session(ENGINE) as session:
        approved = session.scalars(
            select(DecisionRecord).where(DecisionRecord.id.in_(claim_seed.decision_ids)).order_by(DecisionRecord.id)
        ).all()
        generation = session.get(StrategyVersion, approved[0].strategy_version_id).policy_json["reconciliation"][
            "account_generation"
        ]
        for index, existing in enumerate(approved):
            snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=existing.evaluation_run_id).one()
            decision = DecisionService(session).create_decision(
                existing.evaluation_run_id,
                principal=principal,
                account_scope=claim_seed.account,
                original_json={
                    "selected_evaluation_ids": [str(snapshot.id)],
                    "final_action": "exit",
                    "order_intents": [executable_intent(str(snapshot.id), action="exit", target_weight=0.0)],
                },
                portfolio_snapshot_json={"cash": 90_000.0, "positions": [{"symbol": "2330", "quantity": 100.0}]},
                risk_snapshot_json={"hard_failures": [], "allowed_actions": ["exit"]},
                now=datetime(2026, 9, 26, 12, 31, tzinfo=UTC),
            )
            DecisionService(session).approve(
                decision.id,
                {"expected_revision": 1},
                principal=AuthPrincipal(
                    "human:phase98-manager",
                    frozenset({"portfolio_manager"}),
                    frozenset({claim_seed.account}),
                ),
                idempotency_key=f"phase98-exit-approve-{index}-{decision.id}",
                now=datetime(2026, 9, 26, 12, 35, tzinfo=UTC),
            )
            DecisionService(session).claim_execution(decision.id, 2, principal=principal, now=CLAIM_TIME)
            reduce_decision_ids.append(decision.id)
        session.add(
            PaperBrokerAccount(
                account_scope=claim_seed.account,
                account_generation=generation,
                opening_cash=100_000.0,
                currency="TWD",
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.add(
            AccountReconciliation(
                account_scope=claim_seed.account,
                account_generation=generation,
                as_of=datetime(2026, 9, 26, 12, 44, tzinfo=UTC),
                broker_state_watermark="broker:0",
                internal_state_watermark="internal:2026-09-26T12:40:00Z",
                broker_snapshot_sha256="b" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=approved[0].policy_sha256,
                status="matched",
            )
        )
        session.add(
            OrderRecord(
                id=source_order_id,
                strategy_name="phase98-opening-lot",
                symbol="2330",
                market="tw_stock",
                action="buy",
                order_type="market",
                target_weight=0.1,
                quantity=100.0,
                price=100.0,
                side="long",
                status="filled",
                broker_mode="paper",
                order_origin="decision",
                decision_id=claim_seed.decision_ids[0],
                account_scope=claim_seed.account,
                account_generation=generation,
                client_order_ref=f"DL-source-{source_order_id.hex}",
                instrument="spot",
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.flush()
        session.add(
            OrderFillRecord(
                id=source_fill_id,
                order_id=source_order_id,
                fill_price=100.0,
                fill_quantity=100.0,
                fill_time=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                broker_fill_id=f"phase98-source-{source_fill_id.hex}",
                projection_status="applied",
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.flush()
        session.add(
            PositionLot(
                id=lot_id,
                account_scope=claim_seed.account,
                account_generation=generation,
                market="tw_stock",
                symbol="2330",
                instrument="spot",
                side="long",
                opening_fill_id=source_fill_id,
                opening_decision_id=claim_seed.decision_ids[0],
                original_quantity=100.0,
                open_quantity=100.0,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 100.0},
                opened_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.commit()

    barrier = Barrier(2)

    def materialize(decision_id):
        barrier.wait(timeout=10)
        try:
            return materialize_order_intents(
                lambda: Session(ENGINE),
                decision_id,
                principal=principal,
                account_nav=100_000.0,
                prices={("tw_stock", "2330", "spot"): 100.0},
                now=CLAIM_TIME,
            )["status"]
        except ExecutionConflictError as error:
            return str(error)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = [
                future.result(timeout=20)
                for future in [pool.submit(materialize, decision_id) for decision_id in reduce_decision_ids]
            ]
        assert sorted(statuses) == ["pending_submit", "reduce/exit requires a negative quantity delta"]
        with Session(ENGINE) as session:
            orders = session.scalars(
                select(OrderRecord).where(
                    OrderRecord.account_scope == claim_seed.account,
                    OrderRecord.account_generation == generation,
                )
            ).all()
            materialized = [order for order in orders if order.decision_id in reduce_decision_ids]
            assert len(materialized) == 1
            assert materialized[0].reserved_quantity == 100.0
            assert session.get(PositionLot, lot_id).reserved_close_quantity == 100.0
    finally:
        with Session(ENGINE) as session:
            session.execute(
                delete(OrderRecord).where(
                    OrderRecord.account_scope == claim_seed.account,
                    OrderRecord.id != source_order_id,
                )
            )
            session.execute(delete(PositionLot).where(PositionLot.id == lot_id))
            session.execute(delete(OrderFillRecord).where(OrderFillRecord.id == source_fill_id))
            session.execute(delete(OrderRecord).where(OrderRecord.id == source_order_id))
            session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id.in_(reduce_decision_ids)))
            session.execute(delete(DecisionRecord).where(DecisionRecord.id.in_(reduce_decision_ids)))
            session.execute(
                delete(AccountReconciliation).where(AccountReconciliation.account_scope == claim_seed.account)
            )
            session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == claim_seed.account))
            session.commit()


def test_same_account_materialization_serializes_cash_capacity(claim_seed):
    principal = claim_seed.worker
    source_order_id = uuid.uuid4()
    custom_run_id = None
    custom_snapshot_id = None
    custom_decision_id = None
    with Session(ENGINE) as session:
        first_decision = session.get(DecisionRecord, claim_seed.decision_ids[0])
        custom_universe = [{"symbol": "2317", "market": "tw_stock", "instrument": "spot"}]
        custom_run = EvaluationService(session).evaluate_run(
            claim_seed.version_ids[1],
            claim_seed.manifest_id,
            custom_universe,
            [
                {
                    **custom_universe[0],
                    "status": "evaluated",
                    "recommendation_json": {"research_status": "not_required", "side": "long"},
                    "reason_codes": [],
                    "valid_until": "2026-09-26T15:00:00Z",
                }
            ],
        )
        session.flush()
        custom_snapshot = session.query(EvaluationSnapshot).filter_by(evaluation_run_id=custom_run.id).one()
        custom_decision = DecisionService(session).create_decision(
            custom_run.id,
            principal=principal,
            account_scope=claim_seed.account,
            original_json={
                "selected_evaluation_ids": [str(custom_snapshot.id)],
                "final_action": "enter",
                "order_intents": [
                    executable_intent(str(custom_snapshot.id), symbol="2317", action="enter", target_weight=0.1)
                ],
            },
            portfolio_snapshot_json={"cash": 15_000.0, "positions": []},
            risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
            now=datetime(2026, 9, 26, 12, 1, tzinfo=UTC),
        )
        manager = AuthPrincipal(
            "human:phase98-manager",
            frozenset({"portfolio_manager"}),
            frozenset({claim_seed.account}),
        )
        DecisionService(session).approve(
            custom_decision.id,
            {"expected_revision": 1},
            principal=manager,
            idempotency_key=f"phase98-cash-approve-{custom_decision.id}",
            now=datetime(2026, 9, 26, 12, 35, tzinfo=UTC),
        )
        DecisionService(session).claim_execution(first_decision.id, 2, principal=principal, now=CLAIM_TIME)
        DecisionService(session).claim_execution(custom_decision.id, 2, principal=principal, now=CLAIM_TIME)
        generation = session.get(StrategyVersion, first_decision.strategy_version_id).policy_json["reconciliation"][
            "account_generation"
        ]
        session.add(
            PaperBrokerAccount(
                account_scope=claim_seed.account,
                account_generation=generation,
                opening_cash=100_000.0,
                currency="TWD",
                state_version=1,
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                updated_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.add(
            PaperCashMovement(
                account_scope=claim_seed.account,
                account_generation=generation,
                currency="TWD",
                amount=-85_000.0,
                movement_type="adjustment",
                state_version=1,
                occurred_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
                created_at=datetime(2026, 9, 26, 12, 40, tzinfo=UTC),
            )
        )
        session.add(
            OrderRecord(
                id=source_order_id,
                strategy_name="phase98-cash-watermark",
                symbol="2303",
                market="tw_stock",
                action="buy",
                order_type="market",
                target_weight=0.0,
                quantity=1.0,
                price=100.0,
                side="long",
                status="filled",
                broker_mode="paper",
                order_origin="decision",
                decision_id=first_decision.id,
                account_scope=claim_seed.account,
                account_generation=generation,
                client_order_ref=f"DL-cash-source-{source_order_id.hex}",
                instrument="spot",
                created_at=CLAIM_TIME,
                updated_at=CLAIM_TIME,
            )
        )
        session.flush()
        session.add(
            AccountReconciliation(
                account_scope=claim_seed.account,
                account_generation=generation,
                as_of=CLAIM_TIME,
                broker_state_watermark="broker:1",
                internal_state_watermark=internal_state_watermark(session, claim_seed.account, generation),
                broker_snapshot_sha256="c" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=first_decision.policy_sha256,
                status="matched",
            )
        )
        session.commit()
        custom_run_id = custom_run.id
        custom_snapshot_id = custom_snapshot.id
        custom_decision_id = custom_decision.id

    barrier = Barrier(2)
    decision_ids = (claim_seed.decision_ids[0], custom_decision_id)

    def materialize(decision_id):
        barrier.wait(timeout=10)
        try:
            return materialize_order_intents(
                lambda: Session(ENGINE),
                decision_id,
                principal=principal,
                account_nav=100_000.0,
                prices={
                    ("tw_stock", "2330", "spot"): 100.0,
                    ("tw_stock", "2317", "spot"): 100.0,
                },
                now=CLAIM_TIME,
            )["status"]
        except ExecutionConflictError as error:
            return str(error)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = [
                future.result(timeout=20) for future in [pool.submit(materialize, item) for item in decision_ids]
            ]
        # Winner's reservation invalidates the prior match before loser's capacity arithmetic.
        assert sorted(outcomes) == ["account reconciliation internal watermark changed", "pending_submit"]
        with Session(ENGINE) as session:
            orders = session.scalars(select(OrderRecord).where(OrderRecord.decision_id.in_(decision_ids))).all()
            materialized = [order for order in orders if order.id != source_order_id]
            assert len(materialized) == 1
            assert materialized[0].reserved_cash_json == {"currency": "TWD", "amount": 10_000.0}
    finally:
        with Session(ENGINE) as session:
            session.execute(delete(OrderRecord).where(OrderRecord.account_scope == claim_seed.account))
            session.execute(delete(DecisionEvent).where(DecisionEvent.decision_id == custom_decision_id))
            session.execute(delete(DecisionRecord).where(DecisionRecord.id == custom_decision_id))
            session.execute(delete(EvaluationSnapshot).where(EvaluationSnapshot.id == custom_snapshot_id))
            session.execute(delete(EvaluationRun).where(EvaluationRun.id == custom_run_id))
            session.execute(
                delete(AccountReconciliation).where(AccountReconciliation.account_scope == claim_seed.account)
            )
            session.execute(delete(PaperCashMovement).where(PaperCashMovement.account_scope == claim_seed.account))
            session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == claim_seed.account))
            session.commit()
