"""Release-manager routes that freeze campaigns and enqueue paired reviews.

Campaign and review reads are already served by
``GET /api/v1/strategies/{strategy_id}/release-state``; this router adds only the
missing writers. It never releases, pauses, rolls back, or moves a strategy pointer;
``api/releases.py`` stays the only release writer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.models.experiment_campaign import ExperimentCampaign
from poseidon.models.strategy_version import StrategyVersion
from poseidon.research.campaign import (
    CampaignContractValidationError,
    CampaignService,
    FrozenCampaignIdentityConflict,
    build_holdout_identity,
)
from poseidon.workers.celery_app import POSEIDON_CPU_QUEUE, celery_app

router = APIRouter()
RUN_REVIEW_TASK = "poseidon.workers.cpu_tasks.run_review_campaign"
Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")]


class CampaignAPIError(BaseModel):
    detail: str


CAMPAIGN_ERROR_RESPONSES = {status: {"model": CampaignAPIError} for status in (403, 404, 409, 422)}


class CampaignCreateRequest(BaseModel):
    """Frozen campaign contract; ``created_by`` always comes from the principal."""

    model_config = ConfigDict(extra="forbid")

    incumbent_strategy_version_id: uuid.UUID
    candidate_strategy_version_id: uuid.UUID
    incumbent_content_sha256: Digest
    candidate_content_sha256: Digest
    declared_difference_json: dict[str, JsonValue]
    hypothesis: Annotated[str, Field(min_length=1)]
    contract_json: dict[str, JsonValue]


class CampaignResponse(BaseModel):
    id: uuid.UUID
    contract_sha256: str
    incumbent_strategy_version_id: uuid.UUID
    candidate_strategy_version_id: uuid.UUID
    hypothesis: str
    created_by: str
    created_at: datetime


class CampaignReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holdout_identity: dict[str, JsonValue]


class CampaignReviewEnqueuedResponse(BaseModel):
    campaign_id: uuid.UUID
    task_id: str
    holdout_identity_sha256: str


def _raise_api_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, FrozenCampaignIdentityConflict):
        raise HTTPException(status_code=409, detail="Campaign identity conflict") from error
    if isinstance(error, (CampaignContractValidationError, PydanticValidationError)):
        raise HTTPException(status_code=422, detail=str(error) or "Invalid campaign input") from error
    raise error


def _require_version_scope(db: Session, version_id: uuid.UUID, principal: AuthPrincipal) -> None:
    version = db.get(StrategyVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Strategy version not found")
    policy = version.policy_json if isinstance(version.policy_json, dict) else {}
    account_scope = policy.get("account_scope")
    principal.require_account_scope(account_scope if isinstance(account_scope, str) else "")


def _campaign_response(campaign: ExperimentCampaign) -> dict:
    return {
        "id": campaign.id,
        "contract_sha256": campaign.contract_sha256,
        "incumbent_strategy_version_id": campaign.incumbent_strategy_version_id,
        "candidate_strategy_version_id": campaign.candidate_strategy_version_id,
        "hypothesis": campaign.hypothesis,
        "created_by": campaign.created_by,
        "created_at": campaign.created_at,
    }


@router.post("/", response_model=CampaignResponse, responses=CAMPAIGN_ERROR_RESPONSES)
def create_campaign(
    body: CampaignCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        principal.require_role("release_manager")
        _require_version_scope(db, body.incumbent_strategy_version_id, principal)
        _require_version_scope(db, body.candidate_strategy_version_id, principal)
        contract = body.model_dump(mode="json")
        contract["created_by"] = principal.actor_id
        campaign = CampaignService(db).create_frozen_campaign(contract)
        db.commit()
        db.refresh(campaign)
        return _campaign_response(campaign)
    except Exception as error:
        db.rollback()
        _raise_api_error(error)


@router.post(
    "/{campaign_id}/review",
    status_code=202,
    response_model=CampaignReviewEnqueuedResponse,
    responses=CAMPAIGN_ERROR_RESPONSES,
)
def enqueue_campaign_review(
    campaign_id: uuid.UUID,
    body: CampaignReviewRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        principal.require_role("release_manager")
        campaign = db.get(ExperimentCampaign, campaign_id)
        if campaign is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        _require_version_scope(db, campaign.incumbent_strategy_version_id, principal)
        _require_version_scope(db, campaign.candidate_strategy_version_id, principal)
        digest = build_holdout_identity(body.holdout_identity)
    except Exception as error:
        _raise_api_error(error)
    # The review consumes the global holdout and runs in the worker, never inline here.
    result = celery_app.send_task(
        RUN_REVIEW_TASK,
        args=[str(campaign_id), body.holdout_identity],
        queue=POSEIDON_CPU_QUEUE,
    )
    return {"campaign_id": campaign_id, "task_id": str(result.id), "holdout_identity_sha256": digest}
