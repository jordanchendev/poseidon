"""Fail-closed release gate contracts."""

import uuid
from datetime import UTC, datetime

import pytest

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.manifest import content_sha256
from poseidon.decision_loop.releases import ReleaseService


NOW = datetime(2026, 10, 13, 2, 0, tzinfo=UTC)
GATE_CODES = (
    "candidate_identity",
    "policy_identity",
    "campaign_identity",
    "incumbent_pointer",
    "six_cell_evidence",
    "review_identity",
    "holdout_ownership",
    "reconciliation",
    "release_authority",
)


def release_manager(scope="paper:pilot"):
    return AuthPrincipal(
        "human:phase100-release-manager",
        frozenset({"release_manager"}),
        frozenset({scope}),
    )


@pytest.mark.postgresql
def test_gate_retains_every_missing_predicate_in_stable_order(phase99_session_factory):
    missing = {name: uuid.uuid4() for name in ("strategy", "candidate", "campaign", "review", "reconciliation")}

    with phase99_session_factory() as session:
        result = ReleaseService(session).evaluate_gate(
            missing["strategy"],
            missing["candidate"],
            missing["campaign"],
            missing["review"],
            missing["reconciliation"],
            principal=release_manager(),
            now=NOW,
        )

    assert result.allowed is False
    assert tuple(predicate.code for predicate in result.predicates) == GATE_CODES
    assert all(predicate.state in {"blocked", "unavailable", "inconclusive"} for predicate in result.predicates)
    assert all(predicate.reason and predicate.evidence for predicate in result.predicates)


def test_gate_result_values_are_immutable_and_canonicalizable():
    result_type = ReleaseService.gate_result_type
    predicate_type = ReleaseService.predicate_type
    predicate = predicate_type(
        code="candidate_identity",
        state="passed",
        reason="candidate content and ownership match",
        evidence={"candidate_version_id": str(uuid.uuid4()), "content_sha256": "a" * 64},
    )
    result = result_type((predicate,))

    assert result.allowed is True
    assert content_sha256(result.as_json())
    with pytest.raises((AttributeError, TypeError)):
        predicate.state = "blocked"
