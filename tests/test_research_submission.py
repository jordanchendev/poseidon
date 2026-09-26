"""Research request identity and immutable human-completion contracts."""

import pytest
from fastapi import HTTPException

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.manifest import ManifestService, ValidationError
from poseidon.decision_loop.research import ResearchService
from tests.test_decision_loop_foundation import db as foundation_db
from tests.test_decision_loop_foundation import manifest_request

db = foundation_db
RESEARCHER = AuthPrincipal("human", frozenset({"researcher"}), frozenset({"paper:pilot"}))


def complete(service, revision, body, digest=None):
    return service.complete_human_research(
        revision.id,
        body,
        principal=RESEARCHER,
        account_scope="paper:pilot",
        request_sha256=digest or revision.request_sha256,
    )


def prepare(service, manifest, scope="tw_stock:spot:2330", **overrides):
    fields = dict(policy_version="pilot-v1", provider="jev", model="bounded", runtime_digest="host-v1")
    fields.update(overrides)
    return service.prepare(scope, manifest.id, **fields)


def narrative(manifest):
    return {
        "scope_key": "tw_stock:spot:2330",
        "symbol": "2330",
        "as_of": manifest.payload_json["as_of"],
        "thesis": "Bounded description",
        "stance": "watch",
        "change_summary": "Initial revision",
        "next_review_at": "next disclosure",
        "risks": ["uncertainty"],
        "invalidation_conditions": ["changed evidence"],
        "unknowns": ["valuation"],
        "claims": [
            {"kind": "fact", "text": "fundamental", "evidence_ids": ["fundamental"]},
            {"kind": "fact", "text": "price", "evidence_ids": ["ohlcv"]},
            {"kind": "inference", "text": "bounded", "evidence_ids": ["fundamental", "ohlcv"]},
        ],
    }


def test_prepare_identity_binds_every_request_dimension(db):
    manifest = ManifestService(db).freeze(manifest_request())
    service = ResearchService(db)
    revision = prepare(service, manifest)
    assert prepare(service, manifest).id == revision.id
    variants = [prepare(service, manifest, scope="tw_stock:sector:semiconductor")]
    for field in ("policy_version", "provider", "model", "runtime_digest"):
        variants.append(prepare(service, manifest, **{field: "changed"}))
    assert len({revision.request_sha256, *(item.request_sha256 for item in variants)}) == 6


def test_completion_is_idempotent_and_cannot_be_overwritten(db):
    manifest = ManifestService(db).freeze(manifest_request())
    service = ResearchService(db)
    revision = prepare(service, manifest)
    body = narrative(manifest)
    completed = complete(service, revision, body)
    assert completed.status == "completed"
    assert complete(service, revision, body).id == completed.id
    changed = dict(body, thesis="changed")
    with pytest.raises(ValidationError, match="immutable"):
        complete(service, revision, changed)


@pytest.mark.parametrize("case", ["citation", "cutoff", "scope", "schema", "digest"])
def test_human_completion_revalidates_request_and_claims(db, case):
    manifest = ManifestService(db).freeze(manifest_request())
    service = ResearchService(db)
    revision = prepare(service, manifest)
    body = narrative(manifest)
    digest = revision.request_sha256
    if case == "citation":
        body["claims"][0]["evidence_ids"] = ["invented"]
    elif case == "cutoff":
        body["as_of"] = "2026-09-27T00:00:00Z"
    elif case == "scope":
        body["scope_key"] = "another"
    elif case == "schema":
        body["risks"] = []
    else:
        digest = "0" * 64
    with pytest.raises(ValidationError):
        complete(service, revision, body, digest)
    assert revision.content_sha256 is None


def test_actual_predecessor_drift_requires_reprepare(db):
    manifests = ManifestService(db)
    earlier = manifests.freeze(manifest_request())
    later_request = manifest_request("2026-09-26T13:00:00Z")
    later = manifests.freeze(later_request)
    service = ResearchService(db)
    pending = prepare(service, later)
    first = prepare(service, earlier)
    complete(service, first, narrative(earlier))
    with pytest.raises(ValidationError, match="predecessor"):
        complete(service, pending, narrative(later))
    replacement = prepare(service, later)
    assert replacement.previous_revision_id == first.id
    assert replacement.request_sha256 != pending.request_sha256


def test_predecessor_does_not_cross_account_scope(db):
    manifests = ManifestService(db)
    first = manifests.freeze(manifest_request())
    other_request = manifest_request("2026-09-26T13:00:00Z")
    other_request["account_scope"] = "paper:other"
    other = manifests.freeze(other_request)
    service = ResearchService(db)
    completed = prepare(service, first)
    complete(service, completed, narrative(first))
    other_revision = prepare(service, other)
    assert other_revision.previous_revision_id is None


def test_corrupted_predecessor_manifest_fails_closed(db):
    manifests = ManifestService(db)
    earlier = manifests.freeze(manifest_request())
    later = manifests.freeze(manifest_request("2026-09-26T13:00:00Z"))
    service = ResearchService(db)
    completed = prepare(service, earlier)
    complete(service, completed, narrative(earlier))
    earlier.as_of = earlier.as_of.replace(year=2025)
    with pytest.raises(ValidationError, match="metadata"):
        prepare(service, later)


def test_provider_failure_is_review_without_narrative_fallback(db):
    manifest = ManifestService(db).freeze(manifest_request())
    service = ResearchService(db)
    revision = prepare(service, manifest)
    failed = service.provider_failed(revision.id)
    assert failed.status == "needs_review"
    assert failed.content_sha256 is None
    assert failed.research_json == {"review_reason_codes": ["provider_error"]}
    assert prepare(service, manifest).id == failed.id


@pytest.mark.parametrize(
    "principal,account_scope",
    [
        (AuthPrincipal("viewer", frozenset({"viewer"}), frozenset({"paper:pilot"})), "paper:pilot"),
        (
            AuthPrincipal("worker", frozenset({"decision-worker", "researcher"}), frozenset({"paper:pilot"})),
            "paper:pilot",
        ),
        (AuthPrincipal("human", frozenset({"researcher"}), frozenset({"paper:other"})), "paper:pilot"),
        (AuthPrincipal("human", frozenset({"researcher"}), frozenset({"paper:other"})), "paper:other"),
    ],
)
def test_completion_rejects_missing_role_worker_and_cross_account(db, principal, account_scope):
    manifest = ManifestService(db).freeze(manifest_request())
    service = ResearchService(db)
    revision = prepare(service, manifest)
    with pytest.raises(HTTPException) as error:
        service.complete_human_research(
            revision.id,
            narrative(manifest),
            principal=principal,
            account_scope=account_scope,
            request_sha256=revision.request_sha256,
        )
    assert error.value.status_code == 403
