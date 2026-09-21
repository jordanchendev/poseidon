"""RD-Agent API with validated input and UUID-contained artifact paths."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from poseidon.core.database import get_db
from poseidon.core.schemas import (
    RDAgentRunDetailResponse,
    RDAgentRunListResponse,
    RDAgentRunRequest,
    RDAgentRunResponse,
)
from poseidon.models.rd_agent_run import RDAgentRun
from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app

router = APIRouter()
_ROOT = Path(os.environ.get("POSEIDON_AQUARIUM_ROOT", "/app"))


@router.post("/run", status_code=202, response_model=RDAgentRunResponse)
async def create_rdagent_run(request: RDAgentRunRequest, db: Session = Depends(get_db)):
    from poseidon.rdagent.sandbox import validate_challenge_text

    try:
        validate_challenge_text(request.challenge)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    run = RDAgentRun(**request.model_dump())
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        celery_app.send_task(
            "poseidon.workers.qlib_tasks.qlib_rdagent_run", args=[str(run.run_id)], queue=POSEIDON_QLIB_QUEUE
        )
    except Exception as exc:
        run.status, run.error, run.finished_at = "failed", f"dispatch failed: {type(exc).__name__}", datetime.now(UTC)
        db.commit()
        raise HTTPException(503, "RD-Agent dispatch failed") from exc
    return RDAgentRunResponse.model_validate(run)


@router.get("/runs", response_model=RDAgentRunListResponse)
async def list_rdagent_runs(
    limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db)
):
    query = db.query(RDAgentRun)
    return RDAgentRunListResponse(
        runs=[
            RDAgentRunResponse.model_validate(run)
            for run in query.order_by(RDAgentRun.created_at.desc()).offset(offset).limit(limit)
        ],
        total=query.count(),
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}", response_model=RDAgentRunDetailResponse)
async def get_rdagent_run(run_id: UUID, db: Session = Depends(get_db)):
    run = db.query(RDAgentRun).filter_by(run_id=run_id).first()
    if run is None:
        raise HTTPException(404, "RD-Agent run not found")
    return RDAgentRunDetailResponse.model_validate(run)


@router.post("/runs/{run_id}/cancel")
async def cancel_rdagent_run(run_id: UUID, db: Session = Depends(get_db)):
    run = db.query(RDAgentRun).filter_by(run_id=run_id).with_for_update().first()
    if run is None:
        raise HTTPException(404, "RD-Agent run not found")
    if run.status not in {"pending", "running"}:
        raise HTTPException(409, f"cannot cancel run in status={run.status}")
    run.cancel_requested = True
    run.cancel_reason = "user cancellation requested"
    if run.status == "pending":
        run.status = "cancelled"
        run.finished_at = datetime.now(UTC)
    db.commit()
    return {"run_id": str(run_id), "status": run.status, "cancel_requested": True}


@router.get("/runs/{run_id}/artifacts")
async def list_rdagent_artifacts(run_id: UUID):
    base = (_ROOT / "local_dev" / "rd-agent" / "runs").resolve()
    root = (base / str(run_id)).resolve()
    if not root.is_relative_to(base) or not root.is_dir():
        raise HTTPException(404, "artifact directory not found")
    artifacts = []
    for path in root.rglob("*"):
        if path.is_file() and path.resolve().is_relative_to(root):
            artifacts.append({"path": str(path.relative_to(root)), "size": path.stat().st_size})
    return {"run_id": str(run_id), "artifacts": artifacts}
