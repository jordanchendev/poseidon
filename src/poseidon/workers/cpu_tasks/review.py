"""Paired review of a research campaign on its frozen global holdout."""

import uuid

import pandas as pd

from poseidon.core.canonical import canonical_json
from poseidon.core.database import SessionLocal
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks.loop import _loop_mode_skip


def _holdout_manifest_ids(session, campaign, holdout_identity: dict) -> list:
    """Metadata-only preflight; holdout payloads are read only after consumption commits."""
    from poseidon.models.data_manifest import DataManifest

    contract = campaign.contract_json if isinstance(campaign.contract_json, dict) else {}
    frozen_refs = {canonical_json(item) for item in contract.get("manifest_set") or []}
    references = holdout_identity.get("manifest_set")
    if not isinstance(references, list) or not references:
        raise ValueError("holdout identity requires a non-empty manifest_set")
    manifest_ids = []
    for reference in references:
        if canonical_json(reference) not in frozen_refs:
            raise ValueError("holdout manifest is not part of the frozen campaign contract")
        raw_id = reference.get("manifest_id") if isinstance(reference, dict) else reference
        expected = reference.get("content_sha256") if isinstance(reference, dict) else None
        try:
            manifest_id = uuid.UUID(str(raw_id))
        except ValueError as error:
            raise ValueError("holdout manifest reference must be a frozen manifest UUID") from error
        manifest = session.get(DataManifest, manifest_id)
        if manifest is None or (expected is not None and manifest.content_sha256 != expected):
            raise ValueError("holdout manifest reference does not match frozen truth")
        if (manifest.capability_json or {}).get("review_panel") is not True:
            raise ValueError("holdout manifest does not declare review_panel evidence")
        manifest_ids.append(manifest_id)
    return manifest_ids


def _holdout_panel(manifest_ids: list, holdout_identity: dict) -> pd.DataFrame:
    from poseidon.decision_loop.manifest import verify_manifest
    from poseidon.models.data_manifest import DataManifest

    window = holdout_identity.get("window") if isinstance(holdout_identity.get("window"), dict) else {}
    start, end = str(window.get("start", ""))[:10], str(window.get("end", ""))[:10]
    universe = {
        item if isinstance(item, str) else item.get("symbol")
        for item in holdout_identity.get("universe") or []
        if isinstance(item, (str, dict))
    }
    rows = []
    with SessionLocal() as session:
        for manifest_id in manifest_ids:
            payload = verify_manifest(session.get(DataManifest, manifest_id))
            for item in payload["evidence"]:
                if item.get("kind") != "review_panel" or not isinstance(item.get("payload"), dict):
                    continue
                for row in item["payload"].get("rows") or []:
                    date = str(row.get("date", ""))[:10] if isinstance(row, dict) else ""
                    if start <= date <= end and (not universe or row.get("symbol") in universe):
                        rows.append(row)
    return pd.DataFrame(rows)


@celery_app.task(name="poseidon.workers.cpu_tasks.run_review_campaign", max_retries=0)
def run_review_campaign(campaign_id: str, holdout_identity: dict) -> dict:
    """Consume the global holdout, then run one paired review; reuse fails the task without retry."""
    skipped = _loop_mode_skip()
    if skipped:
        return skipped
    persisted_id = uuid.UUID(campaign_id)
    from poseidon.models.experiment_campaign import ExperimentCampaign
    from poseidon.research.campaign import build_holdout_identity, run_paired_review_in_transactions

    digest = build_holdout_identity(holdout_identity)
    with SessionLocal() as session:
        campaign = session.get(ExperimentCampaign, persisted_id)
        if campaign is None:
            raise ValueError("campaign does not exist")
        manifest_ids = _holdout_manifest_ids(session, campaign, holdout_identity)

    def loader(*, permit):
        return _holdout_panel(manifest_ids, holdout_identity)

    review = run_paired_review_in_transactions(SessionLocal, persisted_id, digest, loader)
    return {
        "campaign_id": str(persisted_id),
        "review_id": str(review.id),
        "status": review.status,
        "holdout_identity_sha256": digest,
    }
