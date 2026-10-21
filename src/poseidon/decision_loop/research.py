"""Host-validated, scoped research identity and human narrative completion."""

import json
import uuid

from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import aliased

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.claim_lint import check_numbers, lint_atomic
from poseidon.decision_loop.manifest import (
    NON_FACT_TIERS,
    ValidationError,
    canonical_json,
    content_sha256,
    insert_once,
    required_text,
    source_tier,
    timestamp,
    verify_manifest,
)
from poseidon.models.data_manifest import DataManifest
from poseidon.models.research_revision import CORRECTION_REASONS, ResearchRevision

_UNSET = object()
FAILURE_CODES = frozenset({"provider_error", "provider_timeout", "provider_invalid_response", "provider_unavailable"})


CORRECTION_PREFIX = "CORRECTION:"
# SEED-002 C3: reasons that fix how the evidence was written down may not change what it implies.
STANCE_CHANGE_REASONS = frozenset({"source_error", "stale_evidence"})


def request_identity(
    scope_key, manifest, predecessor, policy_version, provider, model, runtime_digest, supersedes_revision_id=None
):
    identity = {
        "scope_key": scope_key,
        "manifest_sha256": manifest.content_sha256,
        "previous_revision_id": str(predecessor.id) if predecessor else None,
        "previous_content_sha256": predecessor.content_sha256 if predecessor else None,
        "policy_version": policy_version,
        "provider": provider,
        "model": model,
        "runtime_digest": runtime_digest,
    }
    # Added only for a correction, so every pre-C3 request_sha256 stays byte-identical.
    if supersedes_revision_id is not None:
        identity["supersedes_revision_id"] = str(supersedes_revision_id)
    return identity


def superseded_revision_ids():
    """Subquery of revision ids superseded by a completed correction (SEED-002 C3).

    A superseded revision keeps status ``completed`` and its content hash, so evaluation chains that
    cite it still re-verify; supersession is derived from the correction row, never stored on the old one.
    """
    correction = aliased(ResearchRevision)
    return select(correction.supersedes_revision_id).where(
        correction.supersedes_revision_id.is_not(None), correction.status == "completed"
    )


def superseded_by(session, revision_id):
    """The completed correction that supersedes ``revision_id``, or None (at most one, by 048's unique index)."""
    return (
        session.query(ResearchRevision)
        .filter(ResearchRevision.supersedes_revision_id == revision_id, ResearchRevision.status == "completed")
        .one_or_none()
    )


def is_superseded(session, revision_id):
    return superseded_by(session, revision_id) is not None


def validate_correction(body, revision, superseded):
    """SEED-002 C3 body rules for a new completion; ``superseded`` is None for an ordinary revision.

    Runs only on new completions (not inside validate_research), so stored bodies keep re-verifying unchanged.
    """
    summary = body["change_summary"]
    if superseded is None:
        if summary.startswith(CORRECTION_PREFIX):
            raise ValidationError(
                f"change_summary starting with {CORRECTION_PREFIX!r} requires a correction prepared with "
                "supersedes_revision_id"
            )
        if "supersedes" in body or "correction_reason" in body:
            raise ValidationError("research.supersedes/correction_reason are only allowed on a correction")
        return None
    if body.get("supersedes") != str(revision.supersedes_revision_id):
        raise ValidationError("correction research.supersedes must equal the prepared supersedes_revision_id")
    if not summary.startswith(CORRECTION_PREFIX):
        raise ValidationError(f"correction requires change_summary to start with {CORRECTION_PREFIX!r}")
    reason = body.get("correction_reason")
    if reason not in CORRECTION_REASONS:
        raise ValidationError(f"correction requires research.correction_reason in {list(CORRECTION_REASONS)}")
    if reason not in STANCE_CHANGE_REASONS:
        previous = superseded.research_json
        for field in ("stance", "next_review_at"):
            if body.get(field) != previous.get(field):
                raise ValidationError(
                    f"correction_reason {reason!r} may not change research.{field}; "
                    f"only {sorted(STANCE_CHANGE_REASONS)} may"
                )
    return reason


def _excerpt_locators(item):
    excerpts = item.get("payload", {}).get("excerpts") if isinstance(item.get("payload"), dict) else None
    if not isinstance(excerpts, list):
        return set()
    return {excerpt.get("locator") for excerpt in excerpts if isinstance(excerpt, dict)}


def validate_claim_citations(claim, evidence):
    """SEED-002 B2: optional excerpt-level citations; evidence_ids must equal the cited set."""
    if "citations" not in claim:
        return
    citations = claim["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValidationError("claim.citations must be a non-empty list when present")
    cited_ids = set()
    pairs = set()
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValidationError("claim.citations entries must be objects")
        evidence_id = citation.get("evidence_id")
        if not isinstance(evidence_id, str) or evidence_id not in evidence:
            raise ValidationError(f"claim citation evidence_id {evidence_id!r} does not belong to frozen evidence")
        locator = required_text(citation.get("excerpt_locator"), "claim citation excerpt_locator")
        if locator not in _excerpt_locators(evidence[evidence_id]):
            raise ValidationError(f"claim citation excerpt_locator {locator!r} not found in evidence {evidence_id}")
        if (evidence_id, locator) in pairs:
            raise ValidationError(f"claim citation {evidence_id}#{locator} is duplicated")
        pairs.add((evidence_id, locator))
        cited_ids.add(evidence_id)
    if set(claim["evidence_ids"]) != cited_ids:
        raise ValidationError("claim.evidence_ids must equal the evidence cited in claim.citations")


def validate_claim_tier(claim, evidence):
    """SEED-002 B3: fact claims may not rest on third-party or secondary attribution."""
    claim_tier = source_tier(claim.get("source_tier"), "claim.source_tier")
    if claim["kind"] != "fact":
        return
    tiers = {claim_tier} | {evidence[item].get("source_tier") for item in claim["evidence_ids"]}
    if tiers & NON_FACT_TIERS:
        raise ValidationError(
            "fact claims require official or derived evidence; "
            f"cited tiers {sorted(tier for tier in tiers if tier)} include a non-fact tier"
        )


# SEED-002 C2: claims citing price-derived evidence carry computed numbers, not quoted ones.
NUMERIC_PRECHECK_SKIP_KINDS = frozenset({"price_facts", "ohlcv"})
# Numeric mismatch kinds that refuse completion, by claim kind: every mismatch refuses a fact claim; an
# inference claim is refused only on scale/sign, and its value/absent mismatches are recorded.
REFUSED_NUMERIC_MISMATCHES = {
    "fact": frozenset({"scale", "sign", "value", "absent"}),
    "inference": frozenset({"scale", "sign"}),
}


def _claim_excerpt_texts(claim, evidence):
    """Excerpt texts a claim cites: its citation locators when present, else every excerpt of its evidence."""
    texts = []
    pairs = {(item["evidence_id"], item["excerpt_locator"]) for item in claim.get("citations", [])}
    for evidence_id in claim["evidence_ids"]:
        payload = evidence[evidence_id].get("payload")
        excerpts = payload.get("excerpts") if isinstance(payload, dict) else None
        for excerpt in excerpts if isinstance(excerpts, list) else []:
            if not isinstance(excerpt, dict) or not isinstance(excerpt.get("text"), str):
                continue
            if pairs and (evidence_id, excerpt.get("locator")) not in pairs:
                continue
            texts.append(excerpt["text"])
    return texts


def precheck_claims(claims, evidence):
    """SEED-002 C2: atomic lint and numeric precheck for every claim of a new completion.

    Refuses non-atomic claims and numeric mismatches against the cited excerpts (any mismatch for a fact
    claim, scale/sign for an inference claim); returns the per-claim results, in which inference value/absent
    mismatches and all period mismatches are recorded, not refused.
    Runs only on new completions (not inside validate_research), so revisions completed before this
    check keep re-verifying in evaluation and outcome accounting.
    """
    problems = []
    results = []
    for index, claim in enumerate(claims):
        atomic = lint_atomic(claim["text"])
        if not atomic["ok"]:
            problems.append(f"claims[{index}] is non-atomic ({atomic['reason']}); clauses: {atomic['clauses']}")
        kinds = {evidence[item]["kind"] for item in claim["evidence_ids"]}
        excerpts = _claim_excerpt_texts(claim, evidence)
        if kinds & NUMERIC_PRECHECK_SKIP_KINDS:
            numbers = {"status": "skipped", "reason": "cites price_facts/ohlcv"}
        elif not excerpts:
            numbers = {"status": "skipped", "reason": "no cited excerpts"}
        else:
            numbers = check_numbers(claim["text"], "\n".join(excerpts))
        refusing = REFUSED_NUMERIC_MISMATCHES[claim["kind"]]
        refused = [item for item in numbers.get("unmatched", []) if item["kind"] in refusing]
        if refused:
            detail = "; ".join(f"{item['claim']} ({item['kind']})" for item in refused)
            problems.append(f"claims[{index}] numeric mismatch vs cited excerpts {claim['evidence_ids']}: {detail}")
        results.append(
            {
                "claim": index,
                "atomic": atomic["ok"],
                "numbers": numbers["status"],
                **({"reason": numbers["reason"]} if "reason" in numbers else {}),
                "unmatched": [{"claim": item["claim"], "kind": item["kind"]} for item in numbers.get("unmatched", [])],
                "period_mismatch": [
                    {key: item[key] for key in ("claim", "claim_period", "excerpt_periods")}
                    for item in numbers.get("period_mismatch", [])
                ],
            }
        )
    if problems:
        raise ValidationError("claim precheck refused completion: " + " | ".join(problems))
    return results


def validate_research(research, revision, payload):
    if not isinstance(research, dict):
        raise ValidationError("research must be a JSON object")
    body = json.loads(canonical_json(research))
    if body.get("scope_key") != revision.scope_key:
        raise ValidationError("research scope does not match request")
    if timestamp(body.get("as_of"), "research.as_of") != timestamp(payload["as_of"], "manifest.as_of"):
        raise ValidationError("research cutoff does not match manifest")
    if payload.get("symbol") is not None and body.get("symbol") != payload["symbol"]:
        raise ValidationError("research symbol does not match manifest")
    for field in ("thesis", "stance", "change_summary", "next_review_at"):
        required_text(body.get(field), f"research.{field}")
    for field in ("risks", "invalidation_conditions", "unknowns"):
        values = body.get(field)
        if not isinstance(values, list) or not values:
            raise ValidationError(f"research.{field} must be a non-empty text list")
        for value in values:
            required_text(value, f"research.{field}")
    claims = body.get("claims")
    if not isinstance(claims, list) or len(claims) < 3:
        raise ValidationError("research requires at least three claims")
    evidence = {item["id"]: item for item in payload["evidence"]}
    cited = set()
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("kind") not in {"fact", "inference"}:
            raise ValidationError("claim.kind must be fact or inference")
        required_text(claim.get("text"), "claim.text")
        citations = claim.get("evidence_ids")
        if (
            not isinstance(citations, list)
            or not citations
            or any(not isinstance(item, str) or item not in evidence for item in citations)
        ):
            raise ValidationError("claim citations must belong to frozen evidence")
        validate_claim_citations(claim, evidence)
        validate_claim_tier(claim, evidence)
        cited.update(citations)
    for kind in payload["required_data"]:
        if not any(evidence[item]["kind"] == kind for item in cited):
            raise ValidationError(f"research must cite required {kind} evidence")
    return body


class ResearchService:
    def __init__(self, session):
        self.session = session

    def _lock_scope(self, scope_key, account_scope):
        if self.session.get_bind().dialect.name == "postgresql":
            # Serialize a scope through commit so concurrent completions cannot change its predecessor unseen.
            lock_key = int(content_sha256({"scope_key": scope_key, "account_scope": account_scope})[:16], 16)
            if lock_key >= 2**63:
                lock_key -= 2**64
            self.session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})

    def _manifest(self, manifest_id):
        manifest = self.session.get(DataManifest, manifest_id)
        if manifest is None:
            raise ValidationError("manifest does not exist")
        verify_manifest(manifest)
        return manifest

    @staticmethod
    def _scope(payload):
        return (payload.get("account_scope"), payload["market"], payload["interval"], payload.get("symbol"))

    def _predecessor(self, scope_key, manifest):
        """Latest completed, not superseded revision of the scope strictly before the manifest's as_of.

        Ties on as_of are broken by created_at, then str(id). Revisions superseded by a completed
        correction are skipped (SEED-002 C3), so the correction takes the superseded revision's place.
        """
        scope = self._scope(manifest.payload_json)
        candidates = (
            self.session.query(ResearchRevision, DataManifest)
            .join(DataManifest, ResearchRevision.manifest_id == DataManifest.id)
            .filter(
                ResearchRevision.scope_key == scope_key,
                ResearchRevision.status == "completed",
                ResearchRevision.id.not_in(superseded_revision_ids()),
            )
            .all()
        )
        eligible = []
        for revision, candidate_manifest in candidates:
            payload = verify_manifest(candidate_manifest)
            cutoff = timestamp(payload["as_of"], "manifest.as_of")
            if self._scope(payload) == scope and cutoff < timestamp(manifest.payload_json["as_of"], "manifest.as_of"):
                eligible.append((cutoff, revision.created_at, str(revision.id), revision))
        return max(eligible, default=(None, None, "", None))[-1]

    def _correction_target(self, scope_key, manifest, supersedes_revision_id):
        """The completed, not yet superseded revision a correction replaces, checked against its scope."""
        try:
            target_id = uuid.UUID(str(supersedes_revision_id))
        except ValueError as error:
            raise ValidationError("supersedes_revision_id must be a UUID") from error
        target = self.session.get(ResearchRevision, target_id)
        if target is None:
            raise ValidationError("superseded research revision does not exist")
        if target.status != "completed":
            raise ValidationError("only a completed research revision can be superseded")
        target_payload = verify_manifest(self._manifest(target.manifest_id))
        if target.scope_key != scope_key or self._scope(target_payload) != self._scope(manifest.payload_json):
            raise ValidationError(
                "correction must share scope_key, account_scope, market, interval and symbol with the superseded revision"
            )
        if timestamp(target_payload["as_of"], "manifest.as_of") != timestamp(
            manifest.payload_json["as_of"], "manifest.as_of"
        ):
            raise ValidationError("correction manifest as_of must equal the superseded revision's as_of")
        if is_superseded(self.session, target.id):
            raise ValidationError("research revision is already superseded; correct the latest correction instead")
        return target

    def prepare(
        self,
        scope_key,
        manifest_id,
        *,
        policy_version,
        provider,
        model,
        runtime_digest,
        expected_previous_revision_id=_UNSET,
        supersedes_revision_id=None,
    ):
        for field, value in (
            ("scope_key", scope_key),
            ("policy_version", policy_version),
            ("provider", provider),
            ("model", model),
            ("runtime_digest", runtime_digest),
        ):
            required_text(value, field)
        manifest = self._manifest(manifest_id)
        account_scope = required_text(manifest.payload_json.get("account_scope"), "manifest.account_scope")
        self._lock_scope(scope_key, account_scope)
        target = None
        if supersedes_revision_id is not None:
            # A correction replaces the superseded revision in the chain: it inherits that revision's
            # predecessor instead of becoming its successor (SEED-002 C3).
            target = self._correction_target(scope_key, manifest, supersedes_revision_id)
            previous = (
                self.session.get(ResearchRevision, target.previous_revision_id) if target.previous_revision_id else None
            )
        else:
            previous = self._predecessor(scope_key, manifest)
        previous_id = previous.id if previous else None
        if expected_previous_revision_id is not _UNSET and expected_previous_revision_id != previous_id:
            raise ValidationError("actual predecessor changed; rerun prepare")
        target_id = target.id if target else None
        digest = content_sha256(
            request_identity(scope_key, manifest, previous, policy_version, provider, model, runtime_digest, target_id)
        )
        return insert_once(
            self.session,
            ResearchRevision,
            "request_sha256",
            digest,
            scope_key=scope_key,
            manifest_id=manifest.id,
            previous_revision_id=previous_id,
            policy_version=policy_version,
            provider=provider,
            model=model,
            runtime_digest=runtime_digest,
            supersedes_revision_id=target_id,
            status="prepared",
            research_json={},
        )

    def _locked_revision(self, revision_id):
        revision = (
            self.session.query(ResearchRevision)
            .filter(ResearchRevision.id == revision_id)
            .with_for_update()
            .populate_existing()
            .one_or_none()
        )
        if revision is None:
            raise ValidationError("research revision does not exist")
        return revision

    def complete_human_research(
        self, revision_id, research, *, principal: AuthPrincipal, account_scope: str, request_sha256: str
    ):
        principal.require_role("researcher")
        principal.require_account_scope(account_scope)
        requested = self.session.get(ResearchRevision, revision_id)
        if requested is None:
            raise ValidationError("research revision does not exist")
        manifest = self._manifest(requested.manifest_id)
        self._lock_scope(requested.scope_key, manifest.payload_json.get("account_scope"))
        revision = self._locked_revision(revision_id)
        if revision.manifest_id != manifest.id:
            raise ValidationError("frozen research request identity mismatch")
        payload = manifest.payload_json
        if payload.get("account_scope") != account_scope:
            raise HTTPException(status_code=403, detail="Research account scope not authorized")
        if request_sha256 != revision.request_sha256:
            raise ValidationError("research request digest mismatch")
        previous = (
            self.session.get(ResearchRevision, revision.previous_revision_id) if revision.previous_revision_id else None
        )
        identity = request_identity(
            revision.scope_key,
            manifest,
            previous,
            revision.policy_version,
            revision.provider,
            revision.model,
            revision.runtime_digest,
            revision.supersedes_revision_id,
        )
        if content_sha256(identity) != revision.request_sha256:
            raise ValidationError("frozen research request identity mismatch")
        body = validate_research(research, revision, payload)
        digest = content_sha256(body)
        if revision.content_sha256 is not None:
            if revision.status != "completed" or revision.content_sha256 != digest or revision.research_json != body:
                raise ValidationError("completed research is immutable")
            return revision
        target = None
        if revision.supersedes_revision_id is not None:
            # Re-checked under the scope lock: the target may have been corrected since prepare.
            target = self._correction_target(revision.scope_key, manifest, revision.supersedes_revision_id)
        correction_reason = validate_correction(body, revision, target)
        evidence = {item["id"]: item for item in payload["evidence"]}
        claim_precheck = precheck_claims(body["claims"], evidence)
        if target is not None:
            expected_previous = target.previous_revision_id
        else:
            actual = self._predecessor(revision.scope_key, manifest)
            expected_previous = actual.id if actual else None
        if expected_previous != revision.previous_revision_id:
            raise ValidationError("actual predecessor changed; rerun prepare")
        # Host-owned results live in precheck_json (048), never in research_json: content_sha256 hashes
        # research_json, evaluation/outcomes re-verify that hash, and the API serves it as the narrative.
        revision.precheck_json = claim_precheck
        revision.correction_reason = correction_reason
        revision.research_json = body
        revision.content_sha256 = digest
        revision.status = "completed"
        self.session.flush()
        return revision

    def provider_failed(self, revision_id, reason_code="provider_error"):
        if reason_code not in FAILURE_CODES:
            raise ValidationError("unknown provider failure reason code")
        revision = self._locked_revision(revision_id)
        if revision.content_sha256 is not None:
            raise ValidationError("completed research is immutable")
        revision.status = "needs_review"
        revision.research_json = {"review_reason_codes": [reason_code]}
        self.session.flush()
        return revision
