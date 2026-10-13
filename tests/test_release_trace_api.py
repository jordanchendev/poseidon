"""Typed nine-stage release trace API contracts."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from poseidon.main import app

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

    for path in (
        "/api/v1/traces/decisions/00000000-0000-0000-0000-000000000001",
        "/api/v1/traces/signals/00000000-0000-0000-0000-000000000001",
        "/api/v1/traces/evaluations/00000000-0000-0000-0000-000000000001",
    ):
        response = client.get(path)
        assert response.status_code in {403, 404}


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
