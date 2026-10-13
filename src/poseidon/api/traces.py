"""Typed, account-scoped nine-stage decision and signal traces."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import (
    TRACE_STAGE_NAMES,
    DecisionNotFoundError,
    DecisionPolicy,
    DecisionService,
    build_trace_stages,
    load_trace_evidence,
    load_trace_signals,
    project_trace_signal,
)
from poseidon.decision_loop.decisions import (
    project_outcome_rows as _project_outcome_rows,
)
from poseidon.decision_loop.evaluation import verify_complete_run
from poseidon.decision_loop.manifest import canonical_json, content_sha256, iso_time
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.research_revision import ResearchRevision
from poseidon.models.signal import SignalRecord
from poseidon.signals.schemas import SignalTraceMetadata

router = APIRouter()
TraceAnchor = Literal["decision", "signal", "evaluation"]
TraceState = Literal[
    "Available",
    "No trade",
    "Not required",
    "Unavailable",
    "Provisional",
    "Inconclusive",
    "Legacy unresolved",
    "Filled",
    "Blocked",
    "Released",
    "Paused",
    "Rolled back",
    "Rejected",
]
TraceStageName = Literal[
    "signal_or_no_trade",
    "manifest",
    "research_revision",
    "evaluation",
    "decision_events",
    "execution",
    "outcomes",
    "campaign_review",
    "release",
]


class TraceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    state: TraceState
    timestamp: datetime | None = None
    identities: dict[str, str] = Field(default_factory=dict)
    values: dict[str, JsonValue] = Field(default_factory=dict)


class TraceStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Annotated[int, Field(strict=True, ge=1, le=9)]
    name: TraceStageName
    state: TraceState
    reason: Annotated[str, Field(min_length=1)]
    timestamp: datetime | None = None
    identities: dict[str, str] = Field(default_factory=dict)
    rows: list[TraceRow] = Field(default_factory=list)
    raw_payload: JsonValue | None = None


class TraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_type: TraceAnchor
    anchor_id: uuid.UUID
    account_scope: str
    stages: Annotated[list[TraceStage], Field(min_length=9, max_length=9)]

    @model_validator(mode="after")
    def require_closed_stage_order(self):
        if tuple(stage.name for stage in self.stages) != TRACE_STAGE_NAMES:
            raise ValueError("trace stages must use the canonical nine-stage order")
        if [stage.position for stage in self.stages] != list(range(1, 10)):
            raise ValueError("trace stage positions must be contiguous")
        return self


class TraceAPIError(BaseModel):
    detail: str
    error_code: str
    status_code: int


TRACE_ERROR_RESPONSES = {status: {"model": TraceAPIError} for status in (403, 404, 422)}


def project_outcome_rows(outcomes) -> list[TraceRow]:
    return [TraceRow.model_validate(row) for row in _project_outcome_rows(outcomes)]


def _stored_time(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return iso_time(value, "trace.timestamp")


def _evaluation_rows(session, snapshots):
    rows = []
    for snapshot in snapshots:
        revision_ids = [uuid.UUID(str(value)) for value in snapshot.research_revision_ids]
        revisions = (
            session.scalars(
                select(ResearchRevision)
                .where(ResearchRevision.id.in_(revision_ids))
                .order_by(ResearchRevision.created_at, ResearchRevision.id)
            ).all()
            if revision_ids
            else []
        )
        rows.append(
            {
                "id": str(snapshot.id),
                "content_sha256": snapshot.content_sha256,
                "symbol": snapshot.symbol,
                "market": snapshot.market,
                "instrument": snapshot.instrument,
                "status": snapshot.status,
                "selected": False,
                "recommendation_json": json.loads(canonical_json(snapshot.recommendation_json)),
                "technical_json": json.loads(canonical_json(snapshot.technical_json)),
                "reason_codes": list(snapshot.reason_codes),
                "valid_until": _stored_time(snapshot.valid_until),
                "research": {
                    "revision_ids": [str(value) for value in revision_ids],
                    "revisions": [
                        {"id": str(revision.id), "content_sha256": revision.content_sha256} for revision in revisions
                    ],
                }
                if revisions
                else {"research_status": snapshot.recommendation_json.get("research_status", "unavailable")},
            }
        )
    return rows


def _evidence(session, *, version, snapshots, account_scope):
    return load_trace_evidence(
        session,
        version=version,
        snapshots=snapshots,
        account_scope=account_scope,
    )


class TraceService:
    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def _response(anchor_type, anchor_id, account_scope, raw):
        raw["stages"] = build_trace_stages(raw)
        return TraceResponse(
            anchor_type=anchor_type,
            anchor_id=anchor_id,
            account_scope=account_scope,
            stages=raw["stages"],
        )

    def by_decision(self, decision_id, *, principal):
        raw = DecisionService(self.session).trace(decision_id, principal=principal)
        return self._response("decision", decision_id, raw["decision"]["account_scope"], raw)

    def by_evaluation(self, snapshot_id, *, principal):
        snapshot = self.session.get(EvaluationSnapshot, snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Evaluation snapshot not found")
        try:
            run, version, manifest, snapshots = verify_complete_run(self.session, snapshot.evaluation_run_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Evaluation trace identity is invalid") from error
        if snapshot.id not in {row.id for row in snapshots}:
            raise HTTPException(status_code=422, detail="Evaluation snapshot is outside the frozen run")
        try:
            policy = DecisionPolicy.model_validate(version.policy_json)
        except PydanticValidationError as error:
            raise HTTPException(status_code=422, detail="Evaluation policy identity is invalid") from error
        principal.require_account_scope(policy.account_scope)
        decisions = self.session.scalars(
            select(DecisionRecord)
            .where(DecisionRecord.evaluation_run_id == run.id, DecisionRecord.account_scope == policy.account_scope)
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
        ).all()
        linked = [
            decision
            for decision in decisions
            if str(snapshot.id) in {str(value) for value in decision.original_json.get("selected_evaluation_ids", [])}
        ]
        if linked:
            traces = [DecisionService(self.session).trace(decision.id, principal=principal) for decision in linked]
            decision_rows = [trace["decision"] for trace in traces]
            raw = traces[0]
            raw["decision"] = traces[0]["decision"] if len(traces) == 1 else None
            raw["decisions"] = decision_rows
            raw["trace_evaluation_ids"] = [str(snapshot.id)]
            for key in ("events", "orders", "fills", "lots", "allocations", "reconciliations"):
                unique = {row["id"]: row for trace in traces for row in trace[key]}
                raw[key] = [unique[key] for key in sorted(unique)]
            signals = {row["identities"]["signal_id"]: row for trace in traces for row in trace["signals"]}
            raw["signals"] = [signals[key] for key in sorted(signals)]
            raw.update(
                _evidence(
                    self.session,
                    version=version,
                    snapshots=[snapshot],
                    account_scope=policy.account_scope,
                )
            )
            raw["stages"] = build_trace_stages(raw)
            return self._response("evaluation", snapshot.id, policy.account_scope, raw)

        evaluations = _evaluation_rows(self.session, snapshots)
        raw = {
            "decision": None,
            "decisions": [],
            "strategy_version": {
                "id": str(version.id),
                "content_sha256": version.content_sha256,
                "policy_sha256": content_sha256(version.policy_json),
            },
            "evaluation_run": {
                "id": str(run.id),
                "input_sha256": run.input_sha256,
                "coverage_json": json.loads(canonical_json(run.coverage_json)),
                "decision_as_of": _stored_time(run.decision_as_of),
            },
            "manifest": {
                "id": str(manifest.id),
                "content_sha256": manifest.content_sha256,
                "created_at": _stored_time(manifest.created_at),
            },
            "evaluations": evaluations,
            "trace_evaluation_ids": [str(snapshot.id)],
            "events": [],
            "orders": [],
            "fills": [],
            "lots": [],
            "allocations": [],
            "reconciliations": [],
            "signals": load_trace_signals(
                self.session,
                account_scope=policy.account_scope,
                version=version,
                snapshots=[snapshot],
            ),
            **_evidence(
                self.session,
                version=version,
                snapshots=[snapshot],
                account_scope=policy.account_scope,
            ),
        }
        return self._response("evaluation", snapshot.id, policy.account_scope, raw)

    def by_signal(self, signal_id, *, principal):
        signal = self.session.get(SignalRecord, signal_id)
        if signal is None:
            raise HTTPException(status_code=404, detail="Signal not found")
        try:
            metadata = SignalTraceMetadata.model_validate(signal.metadata_)
        except PydanticValidationError as error:
            raise HTTPException(status_code=422, detail="Signal trace identity is invalid") from error
        if metadata.account_scope is None:
            raise HTTPException(status_code=422, detail="Legacy signal account scope is unresolved")
        principal.require_account_scope(metadata.account_scope)
        try:
            if metadata.decision_id is not None:
                response = self.by_decision(metadata.decision_id, principal=principal)
            elif metadata.evaluation_snapshot_id is not None:
                response = self.by_evaluation(metadata.evaluation_snapshot_id, principal=principal)
            else:
                response = None
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Signal trace identity is invalid") from error
        if response is None:
            stages = [
                TraceStage(
                    position=index,
                    name=name,
                    state="Legacy unresolved" if index == 1 else "Unavailable",
                    reason=(
                        "Legacy signal has no stored evaluation or decision identity"
                        if index == 1
                        else "Upstream identity is unavailable"
                    ),
                    identities={"signal_id": str(signal.id)} if index == 1 else {},
                )
                for index, name in enumerate(TRACE_STAGE_NAMES, start=1)
            ]
            return TraceResponse(
                anchor_type="signal",
                anchor_id=signal.id,
                account_scope=metadata.account_scope,
                stages=stages,
            )
        if response.account_scope != metadata.account_scope:
            raise HTTPException(status_code=422, detail="Signal account scope identity changed")
        signal_row = TraceRow.model_validate(project_trace_signal(signal, metadata))
        signal_stage = response.stages[0].model_copy(
            update={
                "state": "Available",
                "reason": "Stored signal carries exact trace identity",
                "timestamp": signal.signal_time,
                "identities": {"signal_id": str(signal.id)},
                "rows": [signal_row],
            }
        )
        return response.model_copy(
            update={
                "anchor_type": "signal",
                "anchor_id": signal.id,
                "stages": [signal_stage, *response.stages[1:]],
            }
        )


def _map_trace_error(error):
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, DecisionNotFoundError):
        raise HTTPException(status_code=404, detail="Decision not found") from error
    if isinstance(error, (PydanticValidationError, ValueError, TypeError)):
        raise HTTPException(status_code=422, detail="Trace identity is invalid") from error
    raise error


@router.get(
    "/decisions/{decision_id}",
    response_model=TraceResponse,
    responses=TRACE_ERROR_RESPONSES,
)
def get_decision_trace(
    decision_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        return TraceService(db).by_decision(decision_id, principal=principal)
    except Exception as error:
        _map_trace_error(error)


@router.get(
    "/signals/{signal_id}",
    response_model=TraceResponse,
    responses=TRACE_ERROR_RESPONSES,
)
def get_signal_trace(
    signal_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        return TraceService(db).by_signal(signal_id, principal=principal)
    except Exception as error:
        _map_trace_error(error)


@router.get(
    "/evaluations/{evaluation_snapshot_id}",
    response_model=TraceResponse,
    responses=TRACE_ERROR_RESPONSES,
)
def get_evaluation_trace(
    evaluation_snapshot_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        return TraceService(db).by_evaluation(evaluation_snapshot_id, principal=principal)
    except Exception as error:
        _map_trace_error(error)
