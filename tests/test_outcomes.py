"""Outcome maturity, frozen-input, and provenance behavior for Phase 99."""

import ast
import inspect
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from poseidon.decision_loop.evaluation import snapshot_payload
from poseidon.decision_loop.manifest import ManifestService, ValidationError, content_sha256
from poseidon.decision_loop.outcome_accounting import FillCostService, _account_reconciliation_sha256
from poseidon.decision_loop.outcomes import OutcomeService, validate_outcome_label_contract
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.fill_allocation import FillAllocation
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.outcome import (
    EconomicReconciliation,
    OutcomeLabelContract,
    OutcomeRecord,
    ResearchAssessment,
)
from poseidon.models.position_lot import PositionLot
from poseidon.models.research_revision import ResearchRevision
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest

ANCHOR = datetime(2026, 1, 2, 16, tzinfo=UTC)  # Friday
MONDAY = datetime(2026, 1, 5, 16, tzinfo=UTC)
TUESDAY = datetime(2026, 1, 6, 16, tzinfo=UTC)
WEDNESDAY = datetime(2026, 1, 7, 16, tzinfo=UTC)
SESSIONS = [ANCHOR, MONDAY, TUESDAY, WEDNESDAY, datetime(2026, 1, 8, 16, tzinfo=UTC)]


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def _contract_json(marker=None):
    calendar_identity = "XNYS:2026a" if marker is None else f"XNYS:2026a:{marker}"
    return {
        "calendar": {"identity": calendar_identity, "evidence_id": "calendar"},
        "horizons": {
            "signal": [{"key": "signal-next-session", "anchor": "decision_as_of", "session_offset": 1}],
            "trade": [{"key": "trade-second-session", "anchor": "decision_as_of", "session_offset": 2}],
            "research": [{"key": "research-third-session", "anchor": "decision_as_of", "session_offset": 3}],
        },
        "benchmark": {"symbol": "SPY", "evidence_id": "benchmark"},
        "reporting_currency": "USD",
        "required_cost_components": ["commission", "tax"],
        "cost_model_version": "paper-cost-v1",
        "fx_model_version": "wm-close-v1",
        "pnl_tolerance": "0.000000000000000001",
        "research_assessment": {
            "expiry_horizon_key": "research-third-session",
            "statuses": ["confirmed", "not_confirmed", "unavailable"],
        },
        "counterfactual_assumption_versions": ["paper-execution-v1"],
    }


def _evidence(evidence_id, kind, payload, event_time):
    return {
        "id": evidence_id,
        "kind": kind,
        "source_uri": f"fixture://{evidence_id}",
        "event_time": _iso(event_time),
        "available_at": _iso(event_time),
        "recorded_at": _iso(event_time),
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def _outcome_manifest_request(
    cutoff,
    *,
    calendar_identity="XNYS:2026a",
    include_horizon_price=True,
    counterfactuals=None,
):
    visible = [session for session in SESSIONS if session <= cutoff]
    prices = {
        _iso(session): value
        for session, value in zip(SESSIONS, ("100", "110", "108", "112", "115"), strict=True)
        if session <= cutoff
    }
    if not include_horizon_price:
        prices.pop(_iso(visible[-1]), None)
    benchmark = {
        _iso(session): value
        for session, value in zip(SESSIONS, ("200", "202", "204", "206", "208"), strict=True)
        if session <= cutoff
    }
    payload = {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": _iso(cutoff),
        "account_scope": "paper:phase99",
        "capability_json": {kind: True for kind in ("price", "benchmark", "fx", "calendar", "research")},
        "required_data": {
            kind: {"max_age_seconds": 31536000}
            for kind in ("price", "benchmark", "fx", "calendar", "research")
        },
        "evidence": [
            _evidence(
                "calendar",
                "calendar",
                {"identity": calendar_identity, "sessions": [_iso(value) for value in SESSIONS]},
                ANCHOR,
            ),
            _evidence("price", "price", {"symbol": "PH99", "values": prices}, visible[-1]),
            _evidence("benchmark", "benchmark", {"symbol": "SPY", "values": benchmark}, visible[-1]),
            _evidence("fx", "fx", {"base": "USD", "quote": "USD", "rate": "1"}, visible[-1]),
            _evidence("research-evidence", "research", {"fact": "bounded evidence"}, visible[-1]),
        ],
    }
    if counterfactuals is not None:
        payload["counterfactuals"] = counterfactuals
    return payload


def _evaluation_manifest_request(marker):
    price = {"symbol": "PH99", "close": "100", "marker": marker}
    return {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": _iso(ANCHOR),
        "account_scope": "paper:phase99",
        "capability_json": {"ohlcv": True},
        "required_data": {"ohlcv": {"max_age_seconds": 86400}},
        "evidence": [_evidence("evaluation-price", "ohlcv", price, ANCHOR)],
    }


@pytest.fixture
def outcome_session(phase99_session_factory):
    session = phase99_session_factory()
    transaction = session.begin()
    try:
        yield session
    finally:
        if transaction.is_active:
            transaction.rollback()
        session.close()


def _seed(outcome_session, cutoff, *, include_horizon_price=True, with_decision=True, counterfactuals=None):
    marker = uuid.uuid4().hex
    strategy = StrategyRecord(
        name=f"phase99-outcome-{marker}",
        strategy_type="technical",
        symbol="PH99",
        market="tw_stock",
        interval="1d",
    )
    outcome_session.add(strategy)
    outcome_session.flush()
    policy = {"fixture": marker}
    version = StrategyVersion(
        strategy_id=strategy.id,
        version_no=1,
        config_json={"fixture": marker},
        policy_json=policy,
        artifact_json={},
        content_sha256=strategy_version_digest({"fixture": marker}, policy, {}),
    )
    outcome_session.add(version)
    evaluation_manifest = ManifestService(outcome_session).freeze(_evaluation_manifest_request(marker))
    outcome_manifest = ManifestService(outcome_session).freeze(
        _outcome_manifest_request(
            cutoff,
            calendar_identity=f"XNYS:2026a:{marker}",
            include_horizon_price=include_horizon_price,
            counterfactuals=counterfactuals,
        )
    )
    contract_json = _contract_json(marker)
    contract = OutcomeLabelContract(
        version=f"label-{marker}",
        contract_json=contract_json,
        contract_sha256=content_sha256(contract_json),
    )
    outcome_session.add(contract)
    outcome_session.flush()
    run = EvaluationRun(
        strategy_version_id=version.id,
        manifest_id=evaluation_manifest.id,
        decision_as_of=ANCHOR,
        universe_json=[{"symbol": "PH99", "market": "tw_stock", "instrument": "spot"}],
        input_sha256=content_sha256({"evaluation": marker}),
        status="complete",
        coverage_json={"total": 1, "terminal": 1, "by_status": {"evaluated": 1}},
    )
    outcome_session.add(run)
    outcome_session.flush()
    snapshot_data = {
        "symbol": "PH99",
        "market": "tw_stock",
        "instrument": "spot",
        "status": "evaluated",
        "recommendation_json": {"research_status": "not_required"},
        "technical_json": {},
        "research_revision_ids": [],
        "reason_codes": [],
        "valid_until": None,
    }
    snapshot = EvaluationSnapshot(
        evaluation_run_id=run.id,
        content_sha256=content_sha256(snapshot_payload(snapshot_data)),
        **snapshot_data,
    )
    outcome_session.add(snapshot)
    outcome_session.flush()
    decision = None
    if with_decision:
        intent = {
            "evaluation_snapshot_id": str(snapshot.id),
            "symbol": "PH99",
            "market": "tw_stock",
            "instrument": "spot",
            "action": "enter",
            "side": "long",
            "target_weight": 0.1,
            "order_type": "market",
        }
        decision = DecisionRecord(
            evaluation_run_id=run.id,
            strategy_version_id=version.id,
            account_scope="paper:phase99",
            decision_as_of=ANCHOR,
            valid_until=ANCHOR + timedelta(days=30),
            status="approved",
            revision=1,
            creation_sha256=content_sha256({"decision": marker}),
            policy_sha256=content_sha256(policy),
            original_json={"selected_evaluation_ids": [str(snapshot.id)]},
            final_json={"selected_evaluation_ids": [str(snapshot.id)], "order_intents": [intent]},
            portfolio_snapshot_json={},
            risk_snapshot_json={},
        )
        outcome_session.add(decision)
        outcome_session.flush()
    return snapshot, decision, contract, outcome_manifest


def _label(outcome_session, seed, cutoff):
    snapshot, _decision, contract, manifest = seed
    return OutcomeService(outcome_session).label_mature_outcomes(
        evaluation_snapshot_ids=[snapshot.id],
        as_of=cutoff,
        label_definition_version=contract.version,
        outcome_manifest_id=manifest.id,
    )


@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        (MONDAY, {"signal"}),
        (TUESDAY, {"signal", "trade"}),
        (WEDNESDAY, {"signal", "trade", "research"}),
    ],
)
def test_independent_maturity_uses_three_named_manifest_calendar_horizons(outcome_session, cutoff, expected):
    seed = _seed(outcome_session, cutoff)
    rows = _label(outcome_session, seed, cutoff)
    assert {row.kind for row in rows} == expected
    assert {row.horizon_key for row in rows} == {
        {"signal": "signal-next-session", "trade": "trade-second-session", "research": "research-third-session"}[kind]
        for kind in expected
    }
    if "trade" in expected:
        trade = next(row for row in rows if row.kind == "trade")
        assert trade.status == "available"
        assert trade.metrics_json["actual"] == {"status": "not_applicable", "reason": "no_execution"}
        assert not {"gross", "net"} & trade.metrics_json["actual"].keys()
    if "research" in expected:
        research = next(row for row in rows if row.kind == "research")
        assert (research.status, research.reason_code) == ("unavailable", "research_assessment_missing")


def test_weekend_calendar_does_not_mature_signal_by_wall_clock(outcome_session):
    saturday = datetime(2026, 1, 3, 16, tzinfo=UTC)
    saturday_seed = _seed(outcome_session, saturday)
    assert _label(outcome_session, saturday_seed, saturday) == []

    monday_seed = _seed(outcome_session, MONDAY)
    signal = next(row for row in _label(outcome_session, monday_seed, MONDAY) if row.kind == "signal")
    assert signal.maturity_at == MONDAY
    assert signal.metrics_json["actual"] == {
        "status": "available",
        "forward_return": "0.100000000000000000",
        "benchmark_return": "0.010000000000000000",
        "excess_return": "0.090000000000000000",
        "mae": "0.000000000000000000",
        "mfe": "0.100000000000000000",
    }


@pytest.mark.parametrize("execution_state", ["open", "early_full_exit"])
def test_trade_mark_horizon_ignores_open_or_early_exit_state(outcome_session, execution_state):
    def add_execution(seed):
        _snapshot, decision, _contract, _manifest = seed
        stored_intent = {"frozen_intent": decision.final_json["order_intents"][0], "economics": {}}
        order = OrderRecord(
            id=uuid.uuid4(),
            strategy_name="phase99",
            symbol="PH99",
            market="tw_stock",
            action="buy",
            order_type="market",
            target_weight=0.1,
            quantity=1,
            price=100,
            side="long",
            status="filled" if execution_state == "early_full_exit" else "partially_filled",
            broker_mode="paper",
            order_origin="decision",
            decision_id=decision.id,
            account_scope=decision.account_scope,
            account_generation="phase99-generation",
            client_order_ref=f"PH99-{uuid.uuid4().hex}",
            instrument="spot",
            intent_json=stored_intent,
            intent_sha256=content_sha256(stored_intent),
            reserved_cash_json={"currency": "USD", "amount": 0},
            reserved_quantity=0,
            reservation_status="released",
            reconciliation_status="resolved",
        )
        outcome_session.add(order)
        outcome_session.flush()
        if execution_state == "early_full_exit":
            outcome_session.add(
                OrderFillRecord(
                    id=uuid.uuid4(),
                    order_id=order.id,
                    fill_price=101,
                    fill_quantity=1,
                    fill_time=MONDAY,
                    broker_fill_id=f"fill-{uuid.uuid4().hex}",
                    projection_status="applied",
                )
            )
            outcome_session.flush()

    early_seed = _seed(outcome_session, MONDAY)
    add_execution(early_seed)
    assert not [row for row in _label(outcome_session, early_seed, MONDAY) if row.kind == "trade"]

    mature_seed = _seed(outcome_session, TUESDAY)
    add_execution(mature_seed)
    trade = next(row for row in _label(outcome_session, mature_seed, TUESDAY) if row.kind == "trade")
    assert trade.maturity_at == TUESDAY
    expected = "not_applicable" if execution_state == "open" else "provisional"
    assert trade.metrics_json["actual"]["status"] == expected
    if execution_state == "open":
        assert trade.metrics_json["actual"]["reason"] == "no_execution"


def test_research_not_confirmed_is_available_negative_label(outcome_session):
    assessment_time = datetime(2026, 1, 4, 12, tzinfo=UTC)
    seed = _seed(outcome_session, assessment_time)
    snapshot, _decision, _contract, manifest = seed
    revision_json = {"bounded": "human assessment"}
    revision = ResearchRevision(
        scope_key=f"phase99:{snapshot.id}",
        manifest_id=manifest.id,
        request_sha256=content_sha256({"request": str(snapshot.id)}),
        policy_version="research-v1",
        provider="human",
        model="none",
        runtime_digest="host-v1",
        content_sha256=content_sha256(revision_json),
        status="completed",
        research_json=revision_json,
    )
    outcome_session.add(revision)
    outcome_session.flush()
    body = {"status": "not_confirmed", "citation_ids": ["research-evidence"]}
    assessment = ResearchAssessment(
        evaluation_snapshot_id=snapshot.id,
        research_revision_id=revision.id,
        manifest_id=manifest.id,
        assessment_type="invalidation",
        assessment_at=assessment_time,
        status="not_confirmed",
        citation_ids_json=["research-evidence"],
        input_sha256=content_sha256({"assessment": body, "snapshot_id": str(snapshot.id)}),
        content_sha256=content_sha256(body),
        assessment_json=body,
    )
    outcome_session.add(assessment)
    outcome_session.flush()

    research = next(row for row in _label(outcome_session, seed, assessment_time) if row.kind == "research")
    assert (research.status, research.reason_code) == ("available", "research_not_confirmed")
    assert research.metrics_json["actual"] == {
        "status": "available",
        "research_result": "not_confirmed",
        "assessment_id": str(assessment.id),
        "assessment_content_sha256": assessment.content_sha256,
    }


def test_mature_missing_price_is_explicit_unavailable(outcome_session):
    seed = _seed(outcome_session, MONDAY, include_horizon_price=False)
    signal = next(row for row in _label(outcome_session, seed, MONDAY) if row.kind == "signal")
    assert (signal.status, signal.reason_code) == ("unavailable", "price_missing")
    assert signal.metrics_json["actual"] == {"status": "unavailable", "reason": "price_missing"}


def test_manifest_rejects_unsupported_counterfactual_and_missing_research_citation(outcome_session):
    unsupported = _seed(
        outcome_session,
        MONDAY,
        counterfactuals=[{"assumption_version": "mutable-latest", "gross": "1"}],
    )
    with pytest.raises(ValidationError, match="counterfactual assumption version"):
        _label(outcome_session, unsupported, MONDAY)

    seed = _seed(outcome_session, MONDAY)
    snapshot, _decision, _contract, manifest = seed
    revision_json = {"bounded": "human assessment"}
    revision = ResearchRevision(
        scope_key=f"phase99:{snapshot.id}",
        manifest_id=manifest.id,
        request_sha256=content_sha256({"request": str(snapshot.id)}),
        policy_version="research-v1",
        provider="human",
        model="none",
        runtime_digest="host-v1",
        content_sha256=content_sha256(revision_json),
        status="completed",
        research_json=revision_json,
    )
    outcome_session.add(revision)
    outcome_session.flush()
    body = {"status": "confirmed", "citation_ids": ["not-in-manifest"]}
    outcome_session.add(
        ResearchAssessment(
            evaluation_snapshot_id=snapshot.id,
            research_revision_id=revision.id,
            manifest_id=manifest.id,
            assessment_type="invalidation",
            assessment_at=MONDAY,
            status="confirmed",
            citation_ids_json=["not-in-manifest"],
            input_sha256=content_sha256({"assessment": body, "snapshot_id": str(snapshot.id)}),
            content_sha256=content_sha256(body),
            assessment_json=body,
        )
    )
    outcome_session.flush()
    with pytest.raises(ValidationError, match="citations"):
        _label(outcome_session, seed, MONDAY)


def test_label_contract_and_static_service_boundaries():
    contract = _contract_json()
    assert validate_outcome_label_contract(contract) == contract
    for mutation, message in (
        (lambda value: value["calendar"].pop("identity"), "calendar.identity"),
        (lambda value: value["horizons"]["trade"][0].pop("anchor"), "anchor"),
        (lambda value: value.update(pnl_tolerance=0.1), "Decimal string"),
        (lambda value: value.update(counterfactual_assumption_versions=[]), "counterfactual"),
    ):
        invalid = _contract_json()
        mutation(invalid)
        with pytest.raises(ValidationError, match=message):
            validate_outcome_label_contract(invalid)

    import poseidon.decision_loop.outcomes as outcomes

    source = inspect.getsource(outcomes)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not {name for name in imported if "dsh" in name.lower() or "llm" in name.lower()}
    service = ast.parse(inspect.getsource(OutcomeService))
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"commit", "rollback"}
        for node in ast.walk(service)
    )
    runner_source = inspect.getsource(outcomes.label_mature_outcomes)
    assert ".begin()" in runner_source
    assert ".label_mature_outcomes(" in runner_source


def test_outcome_rows_are_append_only_and_same_input_replays(outcome_session):
    seed = _seed(outcome_session, MONDAY)
    snapshot, _decision, _contract, _manifest = seed
    first = _label(outcome_session, seed, MONDAY)
    second = _label(outcome_session, seed, MONDAY)
    assert [(row.id, row.content_sha256) for row in first] == [
        (row.id, row.content_sha256) for row in second
    ]
    assert (
        outcome_session.query(OutcomeRecord)
        .filter(OutcomeRecord.evaluation_snapshot_id == snapshot.id)
        .count()
        == len(first)
    )


def _add_reconciled_trade(
    outcome_session,
    seed,
    *,
    classifications=("actual", "actual"),
    component_types=("commission", "tax"),
    other_symbol=False,
    with_cost=True,
):
    snapshot, decision, _contract, _manifest = seed
    generation = f"phase99-generation-{uuid.uuid4().hex}"
    opening_order = OrderRecord(
        id=uuid.uuid4(),
        strategy_name="phase99",
        symbol="PH99",
        market="tw_stock",
        action="buy",
        order_type="market",
        target_weight=0.1,
        quantity=1,
        price=100,
        side="long",
        status="filled",
        broker_mode="paper",
        order_origin="decision",
        decision_id=decision.id,
        account_scope=decision.account_scope,
        account_generation=generation,
        client_order_ref=f"PH99-open-{uuid.uuid4().hex}",
        instrument="spot",
        intent_json={"evaluation_snapshot_id": str(uuid.uuid4())},
        intent_sha256=content_sha256({"opening": str(snapshot.id)}),
    )
    exact_intent = {
        "frozen_intent": decision.final_json["order_intents"][0],
        "economics": {"reported_net": "8.500000000000000000"},
    }
    closing_order = OrderRecord(
        id=uuid.uuid4(),
        strategy_name="phase99",
        symbol="PH99",
        market="tw_stock",
        action="sell",
        order_type="market",
        target_weight=0,
        quantity=1,
        price=110,
        side="long",
        status="filled",
        broker_mode="paper",
        order_origin="decision",
        decision_id=decision.id,
        account_scope=decision.account_scope,
        account_generation=generation,
        client_order_ref=f"PH99-close-{uuid.uuid4().hex}",
        instrument="spot",
        intent_json=exact_intent,
        intent_sha256=content_sha256(exact_intent),
    )
    outcome_session.add_all([opening_order, closing_order])
    outcome_session.flush()
    opening_fill = OrderFillRecord(
        id=uuid.uuid4(),
        order_id=opening_order.id,
        fill_price=100,
        fill_quantity=1,
        fill_time=MONDAY,
        broker_fill_id=f"open-{uuid.uuid4().hex}",
        projection_status="applied",
    )
    closing_fill = OrderFillRecord(
        id=uuid.uuid4(),
        order_id=closing_order.id,
        fill_price=110,
        fill_quantity=1,
        fill_time=TUESDAY,
        broker_fill_id=f"close-{uuid.uuid4().hex}",
        projection_status="applied",
    )
    outcome_session.add_all([opening_fill, closing_fill])
    outcome_session.flush()
    lot = PositionLot(
        id=uuid.uuid4(),
        account_scope=decision.account_scope,
        account_generation=generation,
        market="tw_stock",
        symbol="PH99",
        instrument="spot",
        side="long",
        opening_fill_id=opening_fill.id,
        opening_decision_id=decision.id,
        original_quantity=1,
        open_quantity=0,
        reserved_close_quantity=0,
        cost_basis_json={"unit_price": 100, "contract_multiplier": 1, "currency": "USD"},
        opened_at=MONDAY,
    )
    outcome_session.add(lot)
    outcome_session.flush()
    outcome_session.add(
        FillAllocation(
            id=uuid.uuid4(),
            closing_fill_id=closing_fill.id,
            position_lot_id=lot.id,
            closing_decision_id=decision.id,
            quantity=1,
            realized_cost_json={
                "entry_price": 100,
                "entry_cost": 100,
                "contract_multiplier": 1,
                "currency": "USD",
                "projection_sha256": content_sha256({"allocation": str(closing_fill.id)}),
            },
        )
    )
    reconciliation = AccountReconciliation(
        account_scope=decision.account_scope,
        account_generation=generation,
        as_of=TUESDAY,
        broker_state_watermark=f"broker:{uuid.uuid4().hex}",
        internal_state_watermark=f"internal:{uuid.uuid4().hex}",
        broker_snapshot_sha256=content_sha256({"broker": str(closing_fill.id)}),
        broker_snapshot_json={},
        internal_snapshot_json={},
        difference_json={},
        policy_sha256=decision.policy_sha256,
        status="matched",
    )
    outcome_session.add(reconciliation)
    outcome_session.flush()
    revision = None
    if with_cost:
        amounts = {"commission": "1", "tax": "0.5"}
        revision = FillCostService(outcome_session).append_revision(
            order_fill_id=closing_fill.id,
            reporting_currency="USD",
            cost_model_version="paper-cost-v1",
            components=[
                {
                    "component_type": component,
                    "native_amount": amounts[component],
                    "native_currency": "USD",
                    "reporting_amount": amounts[component],
                    "reporting_currency": "USD",
                    "classification": classification,
                    "source": "paper-model" if classification == "estimated" else "broker-statement",
                }
                for component, classification in zip(component_types, classifications, strict=True)
            ],
        )
    if other_symbol:
        unrelated = OrderRecord(
            id=uuid.uuid4(),
            strategy_name="phase99",
            symbol="OTHER",
            market="tw_stock",
            action="sell",
            order_type="market",
            target_weight=0,
            quantity=99,
            price=999,
            side="long",
            status="filled",
            broker_mode="paper",
            order_origin="decision",
            decision_id=decision.id,
            account_scope=decision.account_scope,
            account_generation=generation,
            client_order_ref=f"OTHER-{uuid.uuid4().hex}",
            instrument="spot",
            intent_json={"evaluation_snapshot_id": str(uuid.uuid4())},
            intent_sha256=content_sha256({"unrelated": str(snapshot.id)}),
        )
        outcome_session.add(unrelated)
        outcome_session.flush()
        outcome_session.add(
            OrderFillRecord(
                id=uuid.uuid4(),
                order_id=unrelated.id,
                fill_price=999,
                fill_quantity=99,
                fill_time=TUESDAY,
                broker_fill_id=f"unrelated-{uuid.uuid4().hex}",
                projection_status="applied",
            )
        )
        outcome_session.flush()
    return closing_fill, revision, reconciliation


@pytest.mark.parametrize("classifications", [("actual", "actual"), ("estimated", "estimated")])
def test_trade_pnl_uses_exact_cost_revision_and_reconciliation(outcome_session, classifications):
    seed = _seed(
        outcome_session,
        TUESDAY,
        counterfactuals=[{"assumption_version": "paper-execution-v1", "net": "7"}],
    )
    _closing_fill, revision, reconciliation = _add_reconciled_trade(
        outcome_session,
        seed,
        classifications=classifications,
        other_symbol=True,
    )
    trade = next(row for row in _label(outcome_session, seed, TUESDAY) if row.kind == "trade")
    assert (trade.status, trade.reason_code) == ("available", "pnl_reconciled")
    assert trade.metrics_json["actual"]["gross"] == "10.000000000000000000"
    assert trade.metrics_json["actual"]["cost_total"] == "1.500000000000000000"
    assert trade.metrics_json["actual"]["net"] == "8.500000000000000000"
    assert {item["classification"] for item in trade.metrics_json["actual"]["cost_components"]} == set(
        classifications
    )
    assert trade.metrics_json["counterfactual"] == [
        {"assumption_version": "paper-execution-v1", "net": "7"}
    ]
    economic = outcome_session.scalar(
        select(EconomicReconciliation).where(EconomicReconciliation.account_reconciliation_id == reconciliation.id)
    )
    assert economic is not None
    assert economic.cost_revision_ids_json == [
        {"id": str(revision.id), "content_sha256": revision.content_sha256}
    ]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_cost", "required_cost_component_missing"),
        ("missing_reconciliation", "account_reconciliation_missing"),
        ("equation_mismatch", "pnl_equation_mismatch"),
    ],
)
def test_trade_pnl_is_provisional_when_economics_are_incomplete(outcome_session, mutation, reason):
    seed = _seed(outcome_session, TUESDAY)
    kwargs = {"component_types": ("commission",), "classifications": ("actual",)} if mutation == "missing_cost" else {}
    closing_fill, _revision, reconciliation = _add_reconciled_trade(outcome_session, seed, **kwargs)
    if mutation == "missing_cost":
        pass
    elif mutation == "missing_reconciliation":
        outcome_session.delete(reconciliation)
        outcome_session.flush()
    else:
        order = outcome_session.get(OrderRecord, closing_fill.order_id)
        intent = dict(order.intent_json)
        intent["economics"] = {"reported_net": "999"}
        order.intent_json = intent
        order.intent_sha256 = content_sha256(intent)
        outcome_session.flush()
    trade = next(row for row in _label(outcome_session, seed, TUESDAY) if row.kind == "trade")
    assert (trade.status, trade.reason_code) == ("provisional", reason)
    assert trade.metrics_json["actual"] == {"status": "provisional", "reason": reason}


def test_legacy_fill_without_cost_provenance_never_becomes_zero_cost_actual(outcome_session):
    seed = _seed(outcome_session, TUESDAY)
    _add_reconciled_trade(outcome_session, seed, with_cost=False)
    trade = next(row for row in _label(outcome_session, seed, TUESDAY) if row.kind == "trade")
    assert (trade.status, trade.reason_code) == ("provisional", "fill_cost_provenance_missing")
    assert "gross" not in trade.metrics_json["actual"]


def test_cost_correction_appends_outcome_and_pins_exact_revision(outcome_session):
    seed = _seed(outcome_session, TUESDAY)
    snapshot, _decision, contract, manifest = seed
    closing_fill, first_cost, reconciliation = _add_reconciled_trade(outcome_session, seed)
    first = next(row for row in _label(outcome_session, seed, TUESDAY) if row.kind == "trade")
    second_cost = FillCostService(outcome_session).append_revision(
        order_fill_id=closing_fill.id,
        reporting_currency="USD",
        cost_model_version="paper-cost-v1",
        components=[
            {
                "component_type": component,
                "native_amount": amount,
                "native_currency": "USD",
                "reporting_amount": amount,
                "reporting_currency": "USD",
                "classification": "actual",
                "source": "broker-statement",
            }
            for component, amount in (("commission", "2"), ("tax", "0.5"))
        ],
    )
    order = outcome_session.get(OrderRecord, closing_fill.order_id)
    intent = dict(order.intent_json)
    intent["economics"] = {"reported_net": "7.500000000000000000"}
    order.intent_json = intent
    order.intent_sha256 = content_sha256(intent)
    outcome_session.flush()
    second = next(row for row in _label(outcome_session, seed, TUESDAY) if row.kind == "trade")
    assert (second.revision_no, second.previous_outcome_id, second.previous_revision_no) == (2, first.id, 1)
    assert second.input_sha256 != first.input_sha256

    economic = next(
        row
        for row in outcome_session.scalars(select(EconomicReconciliation)).all()
        if row.cost_revision_ids_json == [{"id": str(second_cost.id), "content_sha256": second_cost.content_sha256}]
    )
    references = {
        "order_ids": [str(order.id)],
        "fill_ids": [str(closing_fill.id)],
        "fill_cost_revisions": [{"id": str(second_cost.id), "content_sha256": second_cost.content_sha256}],
        "account_reconciliation_id": str(reconciliation.id),
        "account_reconciliation_sha256": _account_reconciliation_sha256(reconciliation),
        "economic_reconciliation_id": str(economic.id),
        "economic_reconciliation_sha256": economic.content_sha256,
    }
    assert second.input_sha256 == content_sha256(
        {
            "logical_key_sha256": second.logical_key_sha256,
            "evaluation_snapshot_sha256": snapshot.content_sha256,
            "outcome_manifest_id": str(manifest.id),
            "outcome_manifest_sha256": manifest.content_sha256,
            "label_contract_id": str(contract.id),
            "label_contract_sha256": contract.contract_sha256,
            "references": references,
            "counterfactual": [],
        }
    )
    assert first_cost.id != second_cost.id


def test_outcome_revalidates_cost_hash_after_lock(outcome_session):
    seed = _seed(outcome_session, TUESDAY)
    _add_reconciled_trade(outcome_session, seed)

    class TamperingOutcomeService(OutcomeService):
        def _trade_metrics(self, snapshot, decision, contract, counterfactuals):
            status, reason, metrics, references = super()._trade_metrics(
                snapshot,
                decision,
                contract,
                counterfactuals,
            )
            references["fill_cost_revisions"][0]["content_sha256"] = "f" * 64
            return status, reason, metrics, references

    snapshot, _decision, contract, manifest = seed
    with pytest.raises(ValidationError, match="fill cost revision hash drift"):
        TamperingOutcomeService(outcome_session).label_mature_outcomes(
            evaluation_snapshot_ids=[snapshot.id],
            as_of=TUESDAY,
            label_definition_version=contract.version,
            outcome_manifest_id=manifest.id,
        )
