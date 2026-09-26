"""Persist supplied terminal evaluations; no strategy execution or order authority."""

import json
import uuid
from collections import Counter
from datetime import UTC

from sqlalchemy.exc import IntegrityError

from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    iso_time,
    required_text,
    timestamp,
    validate_manifest,
    verify_manifest,
)
from poseidon.models.data_manifest import DataManifest
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.research_revision import ResearchRevision
from poseidon.models.strategy_version import StrategyVersion

TERMINAL_STATUSES = frozenset({"evaluated", "excluded", "failed", "no_trade"})


def member_key(member):
    if not isinstance(member, dict):
        raise ValidationError("universe members must be objects")
    return tuple(required_text(member.get(field), field) for field in ("symbol", "market", "instrument"))


def snapshot_payload(snapshot):
    if not isinstance(snapshot, dict):
        raise ValidationError("snapshots must be objects")
    member_key(snapshot)
    if snapshot.get("status") not in TERMINAL_STATUSES:
        raise ValidationError("evaluation snapshot must be terminal")
    payload = {field: snapshot[field] for field in ("symbol", "market", "instrument", "status")}
    for field, kind in (
        ("recommendation_json", dict),
        ("technical_json", dict),
        ("research_revision_ids", list),
        ("reason_codes", list),
    ):
        value = snapshot.get(field, kind())
        if not isinstance(value, kind):
            raise ValidationError(f"{field} has invalid type")
        payload[field] = value
    if any(not isinstance(value, str) or not value for value in payload["reason_codes"]):
        raise ValidationError("reason_codes must contain non-empty strings")
    if payload["status"] in {"excluded", "failed", "no_trade"} and not payload["reason_codes"]:
        raise ValidationError("non-evaluated snapshots require reason codes")
    payload["valid_until"] = iso_time(snapshot["valid_until"], "valid_until") if snapshot.get("valid_until") else None
    return json.loads(canonical_json(payload))


class EvaluationService:
    def __init__(self, session):
        self.session = session

    def _replay(self, run, snapshots, *, version, manifest, cutoff, members, digest):
        existing = self.session.query(EvaluationSnapshot).filter(EvaluationSnapshot.evaluation_run_id == run.id).all()
        hashes = {
            member_key({"symbol": row.symbol, "market": row.market, "instrument": row.instrument}): row.content_sha256
            for row in existing
        }
        for row in existing:
            fields = {
                field: getattr(row, field)
                for field in (
                    "symbol",
                    "market",
                    "instrument",
                    "status",
                    "recommendation_json",
                    "technical_json",
                    "research_revision_ids",
                    "reason_codes",
                    "valid_until",
                )
            }
            if fields["valid_until"] is not None and fields["valid_until"].tzinfo is None:
                # SQLite strips timezone metadata; production DateTime(timezone=True) preserves it.
                fields["valid_until"] = fields["valid_until"].replace(tzinfo=UTC)
            if content_sha256(snapshot_payload(fields)) != row.content_sha256:
                raise ValidationError("evaluation snapshots are immutable")
        expected = {member_key(item): content_sha256(item) for item in snapshots}
        coverage = {
            "total": len(members),
            "terminal": len(snapshots),
            "by_status": dict(Counter(item["status"] for item in snapshots)),
        }
        parent_matches = (
            run.strategy_version_id == version.id
            and run.manifest_id == manifest.id
            and timestamp(run.decision_as_of, "run.decision_as_of") == cutoff
            and run.universe_json == members
            and run.input_sha256 == digest
            and run.coverage_json == coverage
        )
        if run.status != "complete" or not parent_matches or hashes != expected or len(existing) != len(snapshots):
            raise ValidationError("evaluation run and snapshots are immutable")
        return run

    def evaluate_run(self, strategy_version_id, manifest_id, universe, snapshots, decision_as_of=None):
        version = self.session.get(StrategyVersion, strategy_version_id)
        manifest = self.session.get(DataManifest, manifest_id)
        if version is None or manifest is None:
            raise ValidationError("strategy version and manifest must exist")
        try:
            version.verify_content()
        except (TypeError, ValueError) as error:
            raise ValidationError("strategy version content hash mismatch") from error
        payload = verify_manifest(manifest)
        cutoff = timestamp(decision_as_of if decision_as_of is not None else payload["as_of"], "decision_as_of")
        if cutoff < timestamp(payload["as_of"], "manifest.as_of"):
            raise ValidationError("manifest is later than decision cutoff")
        validate_manifest(dict(payload, as_of=iso_time(cutoff, "decision_as_of")))
        if not isinstance(universe, list) or not universe or not isinstance(snapshots, list):
            raise ValidationError("evaluation requires a non-empty universe and terminal snapshots")
        members = sorted(json.loads(canonical_json(universe)), key=member_key)
        keys = [member_key(member) for member in members]
        if len(set(keys)) != len(keys) or any(key[1] != manifest.market for key in keys):
            raise ValidationError("universe identities must be unique within the manifest market")
        checked = [snapshot_payload(snapshot) for snapshot in snapshots]
        checked_keys = [member_key(snapshot) for snapshot in checked]
        if len(set(checked_keys)) != len(checked_keys) or set(checked_keys) != set(keys):
            raise ValidationError("each universe member requires exactly one terminal snapshot")
        for snapshot in checked:
            if snapshot["valid_until"] and timestamp(snapshot["valid_until"], "valid_until") <= cutoff:
                raise ValidationError("snapshot validity must extend beyond decision cutoff")
            for revision_id in snapshot["research_revision_ids"]:
                try:
                    revision = self.session.get(ResearchRevision, uuid.UUID(revision_id))
                except (ValueError, TypeError, AttributeError) as error:
                    raise ValidationError("invalid research revision reference") from error
                if revision is None or revision.status != "completed":
                    raise ValidationError("evaluation research must be completed")
                if revision.content_sha256 != content_sha256(revision.research_json):
                    raise ValidationError("evaluation research content hash mismatch")
                research_manifest = self.session.get(DataManifest, revision.manifest_id)
                research_payload = verify_manifest(research_manifest)
                if (
                    research_manifest.market != manifest.market
                    or research_payload.get("account_scope") != payload.get("account_scope")
                    or (
                        research_payload.get("symbol") is not None
                        and research_payload.get("symbol") != snapshot["symbol"]
                    )
                ):
                    raise ValidationError("evaluation research scope does not match member")
                if timestamp(research_payload["as_of"], "research.as_of") > cutoff:
                    raise ValidationError("evaluation research is later than decision cutoff")
        digest = content_sha256(
            {
                "strategy_version_id": str(version.id),
                "strategy_sha256": version.content_sha256,
                "manifest_sha256": manifest.content_sha256,
                "decision_as_of": iso_time(cutoff, "decision_as_of"),
                "universe": members,
            }
        )
        query = self.session.query(EvaluationRun).filter(EvaluationRun.input_sha256 == digest)
        existing = query.one_or_none()
        if existing is not None:
            return self._replay(
                existing,
                checked,
                version=version,
                manifest=manifest,
                cutoff=cutoff,
                members=members,
                digest=digest,
            )
        try:
            with self.session.begin_nested():
                run = EvaluationRun(
                    strategy_version_id=version.id,
                    manifest_id=manifest.id,
                    decision_as_of=cutoff,
                    universe_json=members,
                    input_sha256=digest,
                    status="pending",
                    coverage_json={},
                )
                self.session.add(run)
                self.session.flush()
                for item in checked:
                    fields = dict(item)
                    fields["valid_until"] = (
                        timestamp(item["valid_until"], "valid_until") if item["valid_until"] else None
                    )
                    self.session.add(
                        EvaluationSnapshot(evaluation_run_id=run.id, content_sha256=content_sha256(item), **fields)
                    )
                self.session.flush()
                run.coverage_json = {
                    "total": len(members),
                    "terminal": len(checked),
                    "by_status": dict(Counter(item["status"] for item in checked)),
                }
                run.status = "complete"
                self.session.flush()
            return run
        except IntegrityError:
            existing = query.one_or_none()
            if existing is None:
                raise
            return self._replay(
                existing,
                checked,
                version=version,
                manifest=manifest,
                cutoff=cutoff,
                members=members,
                digest=digest,
            )
