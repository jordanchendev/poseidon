"""Immutable outcome, cost, research-assessment, and economic facts."""

import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.decision_loop.manifest import content_sha256
from poseidon.models.base import Base

_LABEL_CONTRACT_FIELDS = (
    "calendar",
    "horizons",
    "benchmark",
    "reporting_currency",
    "required_cost_components",
    "cost_model_version",
    "fx_model_version",
    "pnl_tolerance",
    "research_assessment",
    "counterfactual_assumption_versions",
)
_OUTCOME_KINDS = ("signal", "trade", "research")


def outcome_label_contract_sha256(contract_json: dict) -> str:
    """Return the canonical identity of a complete label contract."""

    return content_sha256(contract_json)


def _required_text(value, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


class OutcomeLabelContract(Base):
    __tablename__ = "outcome_label_contracts"
    __table_args__ = (
        UniqueConstraint("contract_sha256", name="uq_outcome_label_contracts_sha256"),
        CheckConstraint("btrim(version) <> ''", name="ck_outcome_label_contracts_version"),
        CheckConstraint(
            "char_length(contract_sha256) = 64",
            name="ck_outcome_label_contracts_sha256",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    contract_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    contract_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def verify_contract(self) -> None:
        if not isinstance(self.contract_json, dict):
            raise ValueError("contract_json must be an object")
        for field in _LABEL_CONTRACT_FIELDS:
            if field not in self.contract_json:
                raise ValueError(f"contract_json requires {field}")

        calendar = self.contract_json["calendar"]
        if not isinstance(calendar, dict) or not calendar:
            raise ValueError("calendar must be a non-empty object")
        horizons = self.contract_json["horizons"]
        if not isinstance(horizons, dict):
            raise ValueError("horizons must be an object")
        for kind in _OUTCOME_KINDS:
            values = horizons.get(kind)
            if not isinstance(values, list) or not values:
                raise ValueError(f"horizons requires an ordered non-empty {kind} list")

        benchmark = self.contract_json["benchmark"]
        if not ((isinstance(benchmark, str) and benchmark.strip()) or (isinstance(benchmark, dict) and benchmark)):
            raise ValueError("benchmark must be explicit")
        for field in ("reporting_currency", "cost_model_version", "fx_model_version"):
            _required_text(self.contract_json[field], field)
        if not isinstance(self.contract_json["required_cost_components"], list):
            raise ValueError("required_cost_components must be a list")
        if not isinstance(self.contract_json["research_assessment"], dict):
            raise ValueError("research_assessment must be an object")
        if not isinstance(self.contract_json["counterfactual_assumption_versions"], list):
            raise ValueError("counterfactual_assumption_versions must be a list")

        tolerance = self.contract_json["pnl_tolerance"]
        if not isinstance(tolerance, str):
            raise ValueError("pnl_tolerance must be a Decimal string")
        try:
            parsed = Decimal(tolerance)
        except InvalidOperation as error:
            raise ValueError("pnl_tolerance must be a Decimal string") from error
        if not parsed.is_finite() or parsed < 0:
            raise ValueError("pnl_tolerance must be a finite non-negative Decimal string")

        _required_text(self.version, "version")
        if self.contract_sha256 != outcome_label_contract_sha256(self.contract_json):
            raise ValueError("outcome label contract hash mismatch")


class ResearchAssessment(Base):
    __tablename__ = "research_assessments"
    __table_args__ = (
        UniqueConstraint(
            "evaluation_snapshot_id",
            "input_sha256",
            name="uq_research_assessments_replay",
        ),
        CheckConstraint(
            "assessment_type IN ('expiry', 'invalidation')",
            name="ck_research_assessments_type",
        ),
        CheckConstraint(
            "status IN ('confirmed', 'not_confirmed', 'unavailable')",
            name="ck_research_assessments_status",
        ),
        CheckConstraint(
            "jsonb_typeof(citation_ids_json) = 'array' AND jsonb_array_length(citation_ids_json) > 0",
            name="ck_research_assessments_citations",
        ),
        CheckConstraint(
            "char_length(input_sha256) = 64 AND char_length(content_sha256) = 64",
            name="ck_research_assessments_hashes",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    evaluation_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_snapshots.id"), nullable=False
    )
    research_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_revisions.id"), nullable=False
    )
    manifest_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("data_manifests.id"), nullable=False)
    assessment_type: Mapped[str] = mapped_column(String(24), nullable=False)
    assessment_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    citation_ids_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    assessment_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class FillCostRevision(Base):
    __tablename__ = "fill_cost_revisions"
    __table_args__ = (
        UniqueConstraint(
            "fill_key_sha256",
            "input_sha256",
            name="uq_fill_cost_revisions_replay",
        ),
        UniqueConstraint(
            "fill_key_sha256",
            "revision_no",
            name="uq_fill_cost_revisions_revision",
        ),
        UniqueConstraint(
            "id",
            "fill_key_sha256",
            "revision_no",
            name="uq_fill_cost_revisions_identity_key_revision",
        ),
        UniqueConstraint(
            "previous_fill_cost_revision_id",
            name="uq_fill_cost_revisions_previous",
        ),
        ForeignKeyConstraint(
            ["previous_fill_cost_revision_id", "fill_key_sha256", "previous_revision_no"],
            [
                "fill_cost_revisions.id",
                "fill_cost_revisions.fill_key_sha256",
                "fill_cost_revisions.revision_no",
            ],
            name="fk_fill_cost_revisions_previous_identity_key_revision",
        ),
        CheckConstraint(
            "(order_fill_id IS NOT NULL AND paper_broker_fill_id IS NULL) OR "
            "(order_fill_id IS NULL AND paper_broker_fill_id IS NOT NULL)",
            name="ck_fill_cost_revisions_fill_xor",
        ),
        CheckConstraint(
            "(revision_no = 1 AND previous_fill_cost_revision_id IS NULL AND previous_revision_no IS NULL) "
            "OR (revision_no > 1 AND previous_fill_cost_revision_id IS NOT NULL "
            "AND previous_revision_no = revision_no - 1)",
            name="ck_fill_cost_revisions_revision_chain",
        ),
        CheckConstraint(
            "char_length(fill_key_sha256) = 64 AND char_length(input_sha256) = 64 AND char_length(content_sha256) = 64",
            name="ck_fill_cost_revisions_hashes",
        ),
        Index(
            "uq_fill_cost_revisions_order_fill_revision",
            "order_fill_id",
            "revision_no",
            unique=True,
            postgresql_where=text("order_fill_id IS NOT NULL"),
        ),
        Index(
            "uq_fill_cost_revisions_paper_fill_revision",
            "paper_broker_fill_id",
            "revision_no",
            unique=True,
            postgresql_where=text("paper_broker_fill_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    order_fill_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("order_fills.id"), nullable=True
    )
    paper_broker_fill_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("paper_broker_fills.id"), nullable=True
    )
    fill_key_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    reporting_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    cost_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_fill_cost_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    previous_revision_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class FillCostComponent(Base):
    __tablename__ = "fill_cost_components"
    __table_args__ = (
        UniqueConstraint(
            "fill_cost_revision_id",
            "component_type",
            "source",
            name="uq_fill_cost_components_identity",
        ),
        CheckConstraint(
            "component_type IN ('commission', 'tax', 'funding', 'borrow', 'fx', 'other')",
            name="ck_fill_cost_components_type",
        ),
        CheckConstraint(
            "classification IN ('actual', 'estimated', 'not_applicable', 'unavailable')",
            name="ck_fill_cost_components_classification",
        ),
        CheckConstraint(
            "classification NOT IN ('actual', 'estimated') OR "
            "(native_amount IS NOT NULL AND reporting_amount IS NOT NULL)",
            name="ck_fill_cost_components_amounts",
        ),
        CheckConstraint(
            "classification <> 'unavailable' OR (reason IS NOT NULL AND btrim(reason) <> '')",
            name="ck_fill_cost_components_unavailable_reason",
        ),
        CheckConstraint(
            "classification NOT IN ('actual', 'estimated') OR native_currency = reporting_currency OR "
            "(fx_source IS NOT NULL AND btrim(fx_source) <> '' AND fx_rate IS NOT NULL "
            "AND fx_rate > 0 AND fx_as_of IS NOT NULL)",
            name="ck_fill_cost_components_cross_currency_fx",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    fill_cost_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("fill_cost_revisions.id"), nullable=False
    )
    component_type: Mapped[str] = mapped_column(String(32), nullable=False)
    native_amount: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    native_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    reporting_amount: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    reporting_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    classification: Mapped[str] = mapped_column(String(24), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    cost_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    fx_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fx_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18), nullable=True)
    fx_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EconomicReconciliation(Base):
    __tablename__ = "economic_reconciliations"
    __table_args__ = (
        UniqueConstraint(
            "account_reconciliation_id",
            "input_sha256",
            name="uq_economic_reconciliations_replay",
        ),
        CheckConstraint(
            "status IN ('matched', 'provisional', 'unavailable')",
            name="ck_economic_reconciliations_status",
        ),
        CheckConstraint(
            "char_length(input_sha256) = 64 AND char_length(content_sha256) = 64",
            name="ck_economic_reconciliations_hashes",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    account_reconciliation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("account_reconciliations.id"), nullable=False
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    reporting_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    cost_revision_ids_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    fx_facts_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    reason_codes_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OutcomeRecord(Base):
    __tablename__ = "outcome_records"
    __table_args__ = (
        UniqueConstraint(
            "logical_key_sha256",
            "input_sha256",
            name="uq_outcome_records_replay",
        ),
        UniqueConstraint(
            "logical_key_sha256",
            "revision_no",
            name="uq_outcome_records_revision",
        ),
        UniqueConstraint(
            "id",
            "logical_key_sha256",
            "revision_no",
            name="uq_outcome_records_identity_key_revision",
        ),
        UniqueConstraint("previous_outcome_id", name="uq_outcome_records_previous"),
        ForeignKeyConstraint(
            ["previous_outcome_id", "logical_key_sha256", "previous_revision_no"],
            ["outcome_records.id", "outcome_records.logical_key_sha256", "outcome_records.revision_no"],
            name="fk_outcome_records_previous_identity_key_revision",
        ),
        CheckConstraint(
            "kind IN ('signal', 'trade', 'research')",
            name="ck_outcome_records_kind",
        ),
        CheckConstraint(
            "(kind = 'trade' AND decision_id IS NOT NULL) OR (kind IN ('signal', 'research') AND decision_id IS NULL)",
            name="ck_outcome_records_kind_decision",
        ),
        CheckConstraint(
            "status IN ('available', 'provisional', 'unavailable')",
            name="ck_outcome_records_status",
        ),
        CheckConstraint(
            "(revision_no = 1 AND previous_outcome_id IS NULL AND previous_revision_no IS NULL) "
            "OR (revision_no > 1 AND previous_outcome_id IS NOT NULL "
            "AND previous_revision_no = revision_no - 1)",
            name="ck_outcome_records_revision_chain",
        ),
        CheckConstraint(
            "char_length(logical_key_sha256) = 64 AND char_length(input_sha256) = 64 "
            "AND char_length(content_sha256) = 64",
            name="ck_outcome_records_hashes",
        ),
        Index(
            "ix_outcome_records_evaluation_kind",
            "evaluation_snapshot_id",
            "kind",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    evaluation_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_snapshots.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    label_contract_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("outcome_label_contracts.id"), nullable=False
    )
    horizon_key: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("data_manifests.id"), nullable=False)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decision_records.id"), nullable=True
    )
    logical_key_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_outcome_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    previous_revision_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    maturity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


@event.listens_for(OutcomeLabelContract, "before_insert")
def _verify_outcome_label_contract(mapper, connection, target):
    target.verify_contract()


def _reject_outcome_fact_update(mapper, connection, target):
    if inspect(target).modified:
        raise ValueError(f"{target.__tablename__} is append-only")


def _reject_outcome_fact_delete(mapper, connection, target):
    raise ValueError(f"{target.__tablename__} is append-only")


for _append_only_model in (
    OutcomeLabelContract,
    ResearchAssessment,
    FillCostRevision,
    FillCostComponent,
    EconomicReconciliation,
    OutcomeRecord,
):
    event.listen(_append_only_model, "before_update", _reject_outcome_fact_update)
    event.listen(_append_only_model, "before_delete", _reject_outcome_fact_delete)
