"""Host-validated, scoped research identity and human narrative completion."""

import json

from fastapi import HTTPException
from sqlalchemy import text

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    insert_once,
    required_text,
    timestamp,
    verify_manifest,
)
from poseidon.models.data_manifest import DataManifest
from poseidon.models.research_revision import ResearchRevision

_UNSET = object()
FAILURE_CODES = frozenset({"provider_error", "provider_timeout", "provider_invalid_response", "provider_unavailable"})


def request_identity(scope_key, manifest, predecessor, policy_version, provider, model, runtime_digest):
    return {
        "scope_key": scope_key,
        "manifest_sha256": manifest.content_sha256,
        "previous_revision_id": str(predecessor.id) if predecessor else None,
        "previous_content_sha256": predecessor.content_sha256 if predecessor else None,
        "policy_version": policy_version,
        "provider": provider,
        "model": model,
        "runtime_digest": runtime_digest,
    }


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

    def _predecessor(self, scope_key, manifest):
        scope = (
            manifest.payload_json.get("account_scope"),
            manifest.market,
            manifest.interval,
            manifest.payload_json.get("symbol"),
        )
        candidates = (
            self.session.query(ResearchRevision, DataManifest)
            .join(DataManifest, ResearchRevision.manifest_id == DataManifest.id)
            .filter(
                ResearchRevision.scope_key == scope_key,
                ResearchRevision.status == "completed",
            )
            .all()
        )
        eligible = []
        for revision, candidate_manifest in candidates:
            payload = verify_manifest(candidate_manifest)
            candidate_scope = (
                payload.get("account_scope"),
                payload["market"],
                payload["interval"],
                payload.get("symbol"),
            )
            cutoff = timestamp(payload["as_of"], "manifest.as_of")
            if candidate_scope == scope and cutoff < timestamp(manifest.payload_json["as_of"], "manifest.as_of"):
                eligible.append((cutoff, revision.created_at, str(revision.id), revision))
        return max(eligible, default=(None, None, "", None))[-1]

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
        previous = self._predecessor(scope_key, manifest)
        previous_id = previous.id if previous else None
        if expected_previous_revision_id is not _UNSET and expected_previous_revision_id != previous_id:
            raise ValidationError("actual predecessor changed; rerun prepare")
        digest = content_sha256(
            request_identity(scope_key, manifest, previous, policy_version, provider, model, runtime_digest)
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
        )
        if content_sha256(identity) != revision.request_sha256:
            raise ValidationError("frozen research request identity mismatch")
        body = validate_research(research, revision, payload)
        digest = content_sha256(body)
        if revision.content_sha256 is not None:
            if revision.status != "completed" or revision.content_sha256 != digest or revision.research_json != body:
                raise ValidationError("completed research is immutable")
            return revision
        actual = self._predecessor(revision.scope_key, manifest)
        if (actual.id if actual else None) != revision.previous_revision_id:
            raise ValidationError("actual predecessor changed; rerun prepare")
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
