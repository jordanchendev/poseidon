"""Typed nine-stage release trace API contracts."""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.api.traces import TraceService
from poseidon.decision_loop.manifest import content_sha256
from poseidon.decision_loop.releases import ReleaseService
from poseidon.main import app
from poseidon.models.data_manifest import DataManifest
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.experiment_campaign import CampaignReview
from poseidon.models.signal import SignalRecord
from tests.test_strategy_release import NOW, _release, _seed_gate_graph, release_manager

TRACE_PATHS = (
    "/api/v1/traces/decisions/{decision_id}",
    "/api/v1/traces/signals/{signal_id}",
    "/api/v1/traces/evaluations/{evaluation_snapshot_id}",
)
STAGE_NAMES = (
    "signal_or_no_trade",
    "manifest",
    "research_revision",
    "evaluation",
    "decision_events",
    "execution",
    "outcomes",
    "campaign_review",
    "release",
)


def _seed_no_trade_snapshot(session, version_id):
    marker = uuid.uuid4().hex
    manifest = DataManifest(
        content_sha256=content_sha256({"manifest": marker}),
        market="tw_stock",
        interval="1d",
        as_of=NOW,
        capability_json={},
        sources_json=[],
        payload_json={"marker": marker},
    )
    session.add(manifest)
    session.flush()
    run = EvaluationRun(
        strategy_version_id=version_id,
        manifest_id=manifest.id,
        decision_as_of=NOW,
        universe_json=[{"symbol": "2330", "market": "tw_stock", "instrument": "spot"}],
        input_sha256=content_sha256({"run": marker}),
        status="succeeded",
        coverage_json={"total": 1, "terminal": 1, "by_status": {"no_trade": 1}},
    )
    session.add(run)
    session.flush()
    snapshot = EvaluationSnapshot(
        evaluation_run_id=run.id,
        symbol="2330",
        market="tw_stock",
        instrument="spot",
        status="no_trade",
        recommendation_json={"research_status": "not_required"},
        technical_json={},
        research_revision_ids=[],
        reason_codes=["no_executable_signal"],
        valid_until=NOW + timedelta(hours=1),
        content_sha256=content_sha256({"snapshot": marker}),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def test_trace_openapi_publishes_three_typed_reads_and_error_models():
    schema = app.openapi()

    for path in TRACE_PATHS:
        operation = schema["paths"][path]["get"]
        assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/TraceResponse"
        }
        assert {"403", "404", "422"}.issubset(operation["responses"])

    stage = schema["components"]["schemas"]["TraceStage"]
    assert stage["properties"]["name"]["enum"] == list(STAGE_NAMES)


def test_trace_stage_order_is_a_closed_nine_stage_contract():
    from poseidon.api.traces import TRACE_STAGE_NAMES, TraceResponse

    assert TRACE_STAGE_NAMES == STAGE_NAMES
    assert TraceResponse.model_json_schema()["properties"]["stages"]["maxItems"] == 9
    assert TraceResponse.model_json_schema()["properties"]["stages"]["minItems"] == 9


def test_formal_trace_routes_reject_unknown_identities_without_guessing():
    client = TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        "viewer:trace-test",
        frozenset({"viewer"}),
        frozenset({"paper:pilot"}),
    )
    try:
        for path in (
            "/api/v1/traces/decisions/00000000-0000-0000-0000-000000000001",
            "/api/v1/traces/signals/00000000-0000-0000-0000-000000000001",
            "/api/v1/traces/evaluations/00000000-0000-0000-0000-000000000001",
        ):
            response = client.get(path)
            assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


def test_outcome_projection_keeps_actual_and_assumption_versions_separate():
    from poseidon.api.traces import project_outcome_rows

    outcome = SimpleNamespace(
        id=uuid.UUID("00000000-0000-0000-0000-000000000010"),
        kind="trade",
        horizon_key="5_sessions",
        status="provisional",
        reason_code="awaiting_cost_revision",
        content_sha256="a" * 64,
        maturity_at=datetime(2026, 10, 13, tzinfo=UTC),
        created_at=datetime(2026, 10, 13, tzinfo=UTC),
        metrics_json={
            "actual": {"net": "8.5", "narrative_confidence": 0.72},
            "counterfactual": [
                {"assumption_version": "paper-execution-v2", "net": "7.0"},
                {"assumption_version": "paper-execution-v1", "net": "6.5"},
            ],
        },
    )

    rows = project_outcome_rows([outcome])

    assert [row.label for row in rows] == [
        "Actual",
        "Counterfactual · paper-execution-v1",
        "Counterfactual · paper-execution-v2",
    ]
    assert rows[0].values["Narrative confidence"] == 0.72
    assert rows[0].values["confidence_note"] == "Not a probability of profit"
    assert all(row.state == "Provisional" for row in rows)
    assert rows[0].values["net"] == "8.5"
    assert rows[1].values["net"] == "6.5"


@pytest.mark.postgresql
def test_no_trade_trace_keeps_campaign_holdout_review_and_release_history(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)
        _release(service, graph)
        service.pause(
            graph.strategy_id,
            expected_revision=1,
            expected_active_version_id=graph.candidate_id,
            principal=release_manager(),
            idempotency_key="trace-pause",
            reason="trace pause",
        )
        service.rollback(
            graph.strategy_id,
            graph.candidate_id,
            graph.reconciliation_id,
            expected_revision=2,
            expected_active_version_id=None,
            principal=release_manager(),
            idempotency_key="trace-rollback",
            reason="trace rollback",
            now=NOW,
        )
        first_review = session.get(CampaignReview, graph.review_id)
        second_result = {"status": "inconclusive", "reason_code": "insufficient_followup"}
        session.add(
            CampaignReview(
                campaign_id=graph.campaign_id,
                input_sha256="8" * 64,
                result_sha256=content_sha256(second_result),
                status="inconclusive",
                result_json=second_result,
                holdout_use_id=first_review.holdout_use_id,
                input_identity_json=first_review.input_identity_json,
                created_at=NOW + timedelta(minutes=1),
            )
        )
        snapshot = _seed_no_trade_snapshot(session, graph.candidate_id)

        response = TraceService(session).by_evaluation(snapshot.id, principal=release_manager())
        campaign = response.stages[7]
        release = response.stages[8]

        assert response.anchor_type == "evaluation"
        assert response.stages[0].state == "No trade"
        assert response.stages[4].state == "Not required"
        assert campaign.state == "Inconclusive"
        assert [row.label for row in campaign.rows].count("Paired cell") == 6
        assert [row.label for row in campaign.rows].count("Campaign review") == 2
        assert {"Campaign", "Holdout use", "Campaign event"}.issubset({row.label for row in campaign.rows})
        assert release.state == "Rolled back"
        assert [row.state for row in release.rows] == ["Released", "Paused", "Rolled back"]
        assert all(row.identities for row in campaign.rows + release.rows)

        with pytest.raises(HTTPException) as denied:
            TraceService(session).by_evaluation(
                snapshot.id,
                principal=release_manager("paper:other"),
            )
        assert denied.value.status_code == 403
        session.rollback()


@pytest.mark.postgresql
def test_signal_trace_uses_explicit_identity_and_legacy_signal_stays_unresolved(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        snapshot = _seed_no_trade_snapshot(session, graph.candidate_id)
        linked = SignalRecord(
            strategy_id=graph.strategy_id,
            symbol="2330",
            market="tw_stock",
            instrument="spot",
            action="long",
            confidence=0.8,
            quantity_pct=0.1,
            signal_time=NOW,
            interval="1d",
            params={},
            status="passed",
            metadata_={
                "account_scope": "paper:pilot",
                "evaluation_snapshot_id": str(snapshot.id),
            },
        )
        legacy = SignalRecord(
            strategy_id=graph.strategy_id,
            symbol="2330",
            market="tw_stock",
            instrument="spot",
            action="hold",
            confidence=0.5,
            signal_time=NOW + timedelta(minutes=1),
            interval="1d",
            params={},
            status="passed",
            metadata_={"account_scope": "paper:pilot"},
        )
        session.add_all([linked, legacy])
        session.flush()

        linked_trace = TraceService(session).by_signal(linked.id, principal=release_manager())
        legacy_trace = TraceService(session).by_signal(legacy.id, principal=release_manager())

        assert linked_trace.anchor_type == "signal"
        assert linked_trace.stages[0].state == "Available"
        assert linked_trace.stages[0].rows[0].identities == {"signal_id": str(linked.id)}
        assert legacy_trace.stages[0].state == "Legacy unresolved"
        assert len(legacy_trace.stages) == 9
        with pytest.raises(HTTPException) as denied:
            TraceService(session).by_signal(linked.id, principal=release_manager("paper:other"))
        assert denied.value.status_code == 403
        session.rollback()
