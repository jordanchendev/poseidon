"""Validate and freeze point-in-time evidence without fetching external data."""

import hashlib
import json
import math
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError

from poseidon.models.data_manifest import DataManifest


class ValidationError(ValueError):
    """A frozen-input contract was violated."""


def canonical_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError("payload must be finite JSON") from error


def content_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def timestamp(value, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(UTC)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be a timezone-aware ISO timestamp") from error


def iso_time(value, field: str) -> str:
    return timestamp(value, field).isoformat().replace("+00:00", "Z")


def required_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be non-empty text")
    return value


def insert_once(session, record_type, digest_field: str, digest: str, **fields):
    """A savepoint contains only our insert; the caller still owns commit/rollback."""
    query = session.query(record_type).filter(getattr(record_type, digest_field) == digest)
    existing = query.one_or_none()
    if existing is not None:
        return existing
    try:
        with session.begin_nested():
            record = record_type(**{digest_field: digest}, **fields)
            session.add(record)
            session.flush()
        return record
    except IntegrityError:
        existing = query.one_or_none()
        if existing is None:
            raise
        return existing


def validate_manifest(request: dict, as_of=None) -> dict:
    if not isinstance(request, dict):
        raise ValidationError("manifest must be an object")
    payload = json.loads(canonical_json(request))
    cutoff = timestamp(as_of if as_of is not None else payload.get("as_of"), "as_of")
    if as_of is not None and "as_of" in payload and timestamp(payload["as_of"], "as_of") != cutoff:
        raise ValidationError("manifest cutoff mismatch")
    payload["as_of"] = iso_time(cutoff, "as_of")
    for field in ("market", "interval"):
        required_text(payload.get(field), field)
    capabilities = payload.get("capability_json")
    required = payload.get("required_data")
    evidence = payload.get("evidence")
    if not isinstance(capabilities, dict) or not isinstance(required, dict) or not required:
        raise ValidationError("manifest requires declared capabilities and required_data")
    if not isinstance(evidence, list) or not evidence:
        raise ValidationError("manifest evidence must be a non-empty list")
    ids = set()
    events_by_kind = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise ValidationError("evidence must be objects")
        evidence_id = required_text(item.get("id"), "evidence.id")
        if evidence_id in ids:
            raise ValidationError("evidence IDs must be unique")
        ids.add(evidence_id)
        kind = required_text(item.get("kind"), "evidence.kind")
        required_text(item.get("source_uri"), "evidence.source_uri")
        if capabilities.get(kind) is not True:
            raise ValidationError(f"unsupported evidence capability: {kind}")
        for field in ("event_time", "available_at", "recorded_at"):
            item[field] = iso_time(item.get(field), f"evidence.{field}")
        available = timestamp(item["available_at"], "available_at")
        if available > cutoff or timestamp(item["event_time"], "event_time") > cutoff:
            raise ValidationError("future evidence is not cutoff-visible")
        if "payload" not in item or item.get("content_sha256") != content_sha256(item["payload"]):
            raise ValidationError("evidence content hash mismatch")
        events_by_kind.setdefault(kind, []).append(timestamp(item["event_time"], "event_time"))
    for kind, rule in required.items():
        if capabilities.get(kind) is not True or not events_by_kind.get(kind):
            raise ValidationError(f"required evidence capability missing: {kind}")
        max_age = rule.get("max_age_seconds") if isinstance(rule, dict) else None
        if (
            isinstance(max_age, bool)
            or not isinstance(max_age, (int, float))
            or not math.isfinite(max_age)
            or max_age < 0
        ):
            raise ValidationError(f"invalid freshness policy: {kind}")
        if (cutoff - max(events_by_kind[kind])).total_seconds() > max_age:
            raise ValidationError(f"required evidence is stale: {kind}")
    if not isinstance(payload.get("sources_json", []), list):
        raise ValidationError("sources_json must be a list")
    return payload


def verify_manifest(manifest: DataManifest) -> dict:
    if manifest is None:
        raise ValidationError("manifest does not exist")
    payload = validate_manifest(manifest.payload_json)
    if content_sha256(payload) != manifest.content_sha256:
        raise ValidationError("frozen manifest content hash mismatch")
    if (
        manifest.market != payload["market"]
        or manifest.interval != payload["interval"]
        or timestamp(manifest.as_of, "manifest.as_of") != timestamp(payload["as_of"], "payload.as_of")
        or manifest.capability_json != payload["capability_json"]
        or manifest.sources_json != payload.get("sources_json", [])
    ):
        raise ValidationError("frozen manifest metadata mismatch")
    return payload


class ManifestService:
    def __init__(self, session):
        self.session = session

    def freeze(self, request: dict, as_of=None) -> DataManifest:
        payload = validate_manifest(request, as_of)
        manifest = insert_once(
            self.session,
            DataManifest,
            "content_sha256",
            content_sha256(payload),
            market=payload["market"],
            interval=payload["interval"],
            as_of=timestamp(payload["as_of"], "as_of"),
            capability_json=payload["capability_json"],
            sources_json=payload.get("sources_json", []),
            payload_json=payload,
        )
        verify_manifest(manifest)
        return manifest
