"""Research request identity and immutable human-completion contracts."""

import pytest
from fastapi import HTTPException

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.manifest import ManifestService, ValidationError
from poseidon.decision_loop.research import ResearchService
from tests.test_decision_loop_foundation import db as foundation_db
from tests.test_decision_loop_foundation import manifest_request, with_excerpts

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


def excerpt_manifest(db, fundamental_tier=None, ohlcv_tier=None):
    request = with_excerpts(manifest_request(), index=0, tier=fundamental_tier)
    if ohlcv_tier is not None:
        request["evidence"][1]["source_tier"] = ohlcv_tier
    return ManifestService(db).freeze(request)


def complete_with_claim(db, manifest, claim):
    service = ResearchService(db)
    revision = prepare(service, manifest)
    body = narrative(manifest)
    body["claims"][0] = claim
    return complete(service, revision, body), revision


def test_excerpt_citations_happy_path(db):
    manifest = excerpt_manifest(db, fundamental_tier="official-prepared", ohlcv_tier="derived")
    claim = {
        "kind": "fact",
        "text": "revenue grew",
        "source_tier": "official-prepared",
        "evidence_ids": ["fundamental"],
        "citations": [
            {"evidence_id": "fundamental", "excerpt_locator": "p1:l1"},
            {"evidence_id": "fundamental", "excerpt_locator": "p1:l2"},
        ],
    }
    completed, _ = complete_with_claim(db, manifest, claim)
    assert completed.status == "completed"
    assert completed.research_json["claims"][0]["citations"][1]["excerpt_locator"] == "p1:l2"


@pytest.mark.parametrize(
    "evidence_ids,citations,match",
    [
        (["fundamental"], [{"evidence_id": "fundamental", "excerpt_locator": "p9:l9"}], "'p9:l9' not found"),
        (["ohlcv"], [{"evidence_id": "ohlcv", "excerpt_locator": "p1:l1"}], "not found in evidence ohlcv"),
        (["fundamental"], [{"evidence_id": "invented", "excerpt_locator": "p1:l1"}], "'invented' does not belong"),
        (
            ["fundamental", "ohlcv"],
            [{"evidence_id": "fundamental", "excerpt_locator": "p1:l1"}],
            "must equal the evidence cited",
        ),
        (
            ["fundamental"],
            [{"evidence_id": "fundamental", "excerpt_locator": "p1:l1"}] * 2,
            "duplicated",
        ),
        (["fundamental"], [], "non-empty list when present"),
    ],
)
def test_excerpt_citations_reject_bad_locator_or_mismatch(db, evidence_ids, citations, match):
    manifest = excerpt_manifest(db)
    claim = {"kind": "inference", "text": "x", "evidence_ids": evidence_ids, "citations": citations}
    service = ResearchService(db)
    revision = prepare(service, manifest)
    body = narrative(manifest)
    body["claims"][0] = claim
    with pytest.raises(ValidationError, match=match):
        complete(service, revision, body)
    assert revision.content_sha256 is None


def test_claim_source_tier_unknown_value_is_rejected(db):
    manifest = excerpt_manifest(db)
    claim = {"kind": "inference", "text": "x", "source_tier": "blog", "evidence_ids": ["fundamental"]}
    with pytest.raises(ValidationError, match=r"claim\.source_tier must be one of"):
        complete_with_claim(db, manifest, claim)


@pytest.mark.parametrize("tier", ["secondary", "third-party-transcript"])
def test_fact_on_non_official_evidence_is_rejected(db, tier):
    manifest = excerpt_manifest(db, fundamental_tier=tier)
    claim = {
        "kind": "fact",
        "text": "x",
        "evidence_ids": ["fundamental"],
        "citations": [{"evidence_id": "fundamental", "excerpt_locator": "p1:l1"}],
    }
    with pytest.raises(ValidationError, match="fact claims require official or derived evidence"):
        complete_with_claim(db, manifest, claim)


def test_fact_with_self_declared_secondary_tier_is_rejected(db):
    manifest = excerpt_manifest(db)
    claim = {"kind": "fact", "text": "x", "source_tier": "secondary", "evidence_ids": ["fundamental"]}
    with pytest.raises(ValidationError, match="fact claims require official or derived evidence"):
        complete_with_claim(db, manifest, claim)


def test_inference_on_secondary_evidence_is_allowed(db):
    manifest = excerpt_manifest(db, fundamental_tier="secondary")
    claim = {
        "kind": "inference",
        "text": "x",
        "source_tier": "secondary",
        "evidence_ids": ["fundamental"],
        "citations": [{"evidence_id": "fundamental", "excerpt_locator": "p1:l2"}],
    }
    service = ResearchService(db)
    revision = prepare(service, manifest)
    body = narrative(manifest)
    body["claims"][0] = claim
    # The untouched fact claim on "ohlcv" has no tier and keeps pre-v21 behaviour.
    assert complete(service, revision, body).status == "completed"


# SEED-002 C2: atomic lint and numeric precheck on new completions.
PRECHECK_EXCERPTS = [
    {"locator": "p1:l1", "text": "Revenue of $133.19 billion versus $37.38 billion for the prior year"},
    {"locator": "p1:l2", "text": "third quarter results represented a 17.6% increase in revenue"},
    {"locator": "p1:l3", "text": "Gross margin for the quarter was 66.4%, operating margin was 58.7%"},
]


def precheck_manifest(db, index=0):
    request = with_excerpts(
        manifest_request(),
        index=index,
        excerpts=PRECHECK_EXCERPTS,
        raw_text="\n".join(item["text"] for item in PRECHECK_EXCERPTS),
    )
    return ManifestService(db).freeze(request)


def cited_claim(text, locator, evidence_id="fundamental", kind="fact"):
    return {
        "kind": kind,
        "text": text,
        "evidence_ids": [evidence_id],
        "citations": [{"evidence_id": evidence_id, "excerpt_locator": locator}],
    }


def refused(db, manifest, claim, index=0):
    service = ResearchService(db)
    revision = prepare(service, manifest)
    body = narrative(manifest)
    body["claims"][index] = claim
    with pytest.raises(ValidationError) as error:
        complete(service, revision, body)
    assert revision.content_sha256 is None
    assert revision.status == "prepared"
    return str(error.value)


def test_non_atomic_claim_is_refused_with_index_and_clauses(db):
    manifest = precheck_manifest(db)
    claim = {"kind": "inference", "text": "第三季毛利率 66.4%、營業利益率 58.7%。", "evidence_ids": ["fundamental"]}
    message = refused(db, manifest, claim, index=2)
    assert "claims[2] is non-atomic" in message
    assert "'第三季毛利率 66.4%'" in message and "'營業利益率 58.7%'" in message


def test_causal_compound_claim_is_refused(db):
    manifest = ManifestService(db).freeze(manifest_request())
    claim = {
        "kind": "inference",
        "text": "第四季毛利率指引低於第三季是因為 2 奈米爬坡的稀釋效應。",
        "evidence_ids": ["fundamental"],
    }
    assert "causal/driver clause attached" in refused(db, manifest, claim)


@pytest.mark.parametrize(
    "text,locator,kind",
    [
        ("美光 2026 會計年度營收為 133.19 億美元。", "p1:l1", "scale"),
        ("Third-quarter revenue decreased 17.6% from the second quarter.", "p1:l2", "sign"),
    ],
)
def test_numeric_scale_or_sign_mismatch_is_refused(db, text, locator, kind):
    message = refused(db, precheck_manifest(db), cited_claim(text, locator))
    assert "claims[0] numeric mismatch vs cited excerpts ['fundamental']" in message
    assert f"({kind})" in message


@pytest.mark.parametrize(
    "text,locator,kind",
    [
        ("第三季毛利率為 56.4%。", "p1:l3", "value"),
        # 66.4% is in p1:l3; citing p1:l1 alone makes the percentage absent.
        ("第三季毛利率為 66.4%。", "p1:l1", "absent"),
    ],
)
def test_fact_claim_value_or_absent_mismatch_is_refused(db, text, locator, kind):
    message = refused(db, precheck_manifest(db), cited_claim(text, locator))
    assert f"({kind})" in message


@pytest.mark.parametrize(
    "text,locator,raw,kind",
    [("第三季毛利率為 56.4%。", "p1:l3", "56.4%", "value"), ("第三季毛利率為 66.4%。", "p1:l1", "66.4%", "absent")],
)
def test_inference_claim_value_or_absent_mismatch_is_recorded_not_refused(db, text, locator, raw, kind):
    manifest = precheck_manifest(db)
    completed, _ = complete_with_claim(db, manifest, cited_claim(text, locator, kind="inference"))
    assert completed.status == "completed"
    assert completed.claim_precheck[0] == {
        "claim": 0,
        "atomic": True,
        "numbers": "mismatch",
        "unmatched": [{"claim": raw, "kind": kind}],
        "period_mismatch": [],
    }
    assert "claim_precheck" not in completed.research_json


def test_inference_claim_scale_mismatch_is_refused(db):
    claim = cited_claim("美光 2026 會計年度營收為 133.19 億美元。", "p1:l1", kind="inference")
    assert "(scale)" in refused(db, precheck_manifest(db), claim)


def test_claim_without_citations_checks_every_excerpt_of_its_evidence(db):
    manifest = precheck_manifest(db)
    claim = {"kind": "fact", "text": "美光 2026 會計年度營收為 133.19 億美元。", "evidence_ids": ["fundamental"]}
    assert "(scale)" in refused(db, manifest, claim)


def test_price_evidence_skips_numeric_check(db):
    manifest = precheck_manifest(db, index=1)  # the excerpts sit on the ohlcv evidence
    claim = cited_claim("美光 2026 會計年度營收為 133.19 億美元。", "p1:l1", evidence_id="ohlcv")
    completed, _ = complete_with_claim(db, manifest, claim)
    assert completed.claim_precheck[0]["numbers"] == "skipped"
    assert completed.claim_precheck[0]["reason"] == "cites price_facts/ohlcv"


def test_legacy_evidence_without_excerpts_skips_numeric_check(db):
    manifest = ManifestService(db).freeze(manifest_request())
    completed, _ = complete_with_claim(
        db, manifest, {"kind": "fact", "text": "營收為 133.19 億美元。", "evidence_ids": ["fundamental"]}
    )
    assert completed.claim_precheck[0] == {
        "claim": 0,
        "atomic": True,
        "numbers": "skipped",
        "reason": "no cited excerpts",
        "unmatched": [],
        "period_mismatch": [],
    }
