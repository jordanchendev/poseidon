"""RD-Agent research run lifecycle model (alembic revision 039)."""

import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class RDAgentRun(Base):
    """Durable pending → terminal record for one constrained RD-Agent run."""

    __tablename__ = "rd_agent_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    challenge = Column(Text, nullable=False)
    time_budget_hours = Column(Float, nullable=False, default=4.0)
    cost_cap_usd = Column(Float, nullable=False, default=20.0)
    use_gpu = Column(Boolean, nullable=False, default=False)
    cancel_requested = Column(Boolean, nullable=False, default=False)
    cancel_reason = Column(Text, nullable=True)
    token_cost_acc_usd = Column(Float, nullable=True)
    status = Column(String(16), nullable=False, default="pending")
    summary = Column(JSONB, nullable=True)
    verdict = Column(String(32), nullable=True)
    result_dir = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    requested_by = Column(String(16), nullable=False, default="api")
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="ck_rd_agent_runs_status",
        ),
    )
