"""Deterministic outcome maturity and append-only revision writes."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from poseidon.decision_loop.evaluation import snapshot_payload
from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    required_text,
    timestamp,
    verify_manifest,
)
from poseidon.models.data_manifest import DataManifest
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.outcome import OutcomeLabelContract, OutcomeRecord, ResearchAssessment
from poseidon.models.research_revision import ResearchRevision

_CAPABILITIES = frozenset({"price", "benchmark", "fx", "calendar", "research"})
_QUANTUM = Decimal("0.000000000000000001")


def _stored_time(value, field):
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return timestamp(value, field)


def _decimal(value, field):
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be a Decimal string")
    try:
        parsed = Decimal(value) if isinstance(value, str) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be a Decimal string") from error
    if not parsed.is_finite():
        raise ValidationError(f"{field} must be finite")
    return parsed


def _decimal_string(value):
    return format(value.quantize(_QUANTUM), "f")


def _uuid(value, field):
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be an immutable UUID") from error


def validate_outcome_label_contract(value) -> dict:
    """Validate the complete frozen label contract without supplying defaults."""
    if isinstance(value, OutcomeLabelContract):
        try:
            value.verify_contract()
        except ValueError as error:
            raise ValidationError(str(error)) from error
        contract = value.contract_json
    else:
        contract = value
    if not isinstance(contract, dict):
        raise ValidationError("label contract must be an object")
    contract = json.loads(canonical_json(contract))

    calendar = contract.get("calendar")
    if not isinstance(calendar, dict):
        raise ValidationError("calendar must be an object")
    required_text(calendar.get("identity"), "calendar.identity")
    required_text(calendar.get("evidence_id"), "calendar.evidence_id")

    horizons = contract.get("horizons")
    if not isinstance(horizons, dict):
        raise ValidationError("horizons must be an object")
    keys = set()
    for kind in ("signal", "trade", "research"):
        items = horizons.get(kind)
        if not isinstance(items, list) or not items:
            raise ValidationError(f"horizons requires {kind}")
        for item in items:
            if not isinstance(item, dict):
                raise ValidationError(f"{kind} horizon must be an object")
            key = required_text(item.get("key"), f"horizons.{kind}.key")
            if key in keys:
                raise ValidationError("horizon keys must be unique")
            keys.add(key)
            if item.get("anchor") != "decision_as_of":
                raise ValidationError(f"horizons.{kind}.anchor must be decision_as_of")
            offset = item.get("session_offset")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
                raise ValidationError(f"horizons.{kind}.session_offset must be positive")

    benchmark = contract.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValidationError("benchmark must be an object")
    required_text(benchmark.get("symbol"), "benchmark.symbol")
    required_text(benchmark.get("evidence_id"), "benchmark.evidence_id")
    for field in ("reporting_currency", "cost_model_version", "fx_model_version"):
        required_text(contract.get(field), field)

    required_costs = contract.get("required_cost_components")
    if not isinstance(required_costs, list) or not required_costs or len(set(required_costs)) != len(required_costs):
        raise ValidationError("required_cost_components must be a non-empty unique list")
    if any(not isinstance(item, str) or not item.strip() for item in required_costs):
        raise ValidationError("required_cost_components must contain text")
    tolerance = contract.get("pnl_tolerance")
    if not isinstance(tolerance, str) or _decimal(tolerance, "pnl_tolerance") < 0:
        raise ValidationError("pnl_tolerance must be a finite non-negative Decimal string")

    research = contract.get("research_assessment")
    if not isinstance(research, dict):
        raise ValidationError("research_assessment must be an object")
    research_keys = {item["key"] for item in horizons["research"]}
    if research.get("expiry_horizon_key") not in research_keys:
        raise ValidationError("research_assessment expiry horizon is invalid")
    statuses = research.get("statuses")
    if statuses != ["confirmed", "not_confirmed", "unavailable"]:
        raise ValidationError("research_assessment statuses must be frozen explicitly")

    assumptions = contract.get("counterfactual_assumption_versions")
    if (
        not isinstance(assumptions, list)
        or not assumptions
        or len(set(assumptions)) != len(assumptions)
        or any(not isinstance(item, str) or not item.strip() for item in assumptions)
    ):
        raise ValidationError("counterfactual assumption versions must be a non-empty unique text list")
    return contract


def _snapshot_body(snapshot):
    fields = {
        field: getattr(snapshot, field)
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
    if fields["valid_until"] is not None:
        fields["valid_until"] = _stored_time(fields["valid_until"], "valid_until")
    return snapshot_payload(fields)


def _manifest_facts(manifest, contract, as_of):
    payload = verify_manifest(manifest)
    if _stored_time(payload["as_of"], "manifest.as_of") != as_of:
        raise ValidationError("outcome manifest cutoff must equal as_of")
    capabilities = payload.get("capability_json")
    if not isinstance(capabilities, dict) or any(capabilities.get(kind) is not True for kind in _CAPABILITIES):
        raise ValidationError("outcome manifest lacks a required capability")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise ValidationError("outcome manifest evidence is invalid")
    by_id = {item.get("id"): item for item in evidence if isinstance(item, dict)}
    if len(by_id) != len(evidence):
        raise ValidationError("outcome manifest evidence identities are invalid")
    for kind in _CAPABILITIES:
        if not any(item.get("kind") == kind for item in evidence):
            raise ValidationError(f"outcome manifest lacks {kind} evidence")

    calendar_item = by_id.get(contract["calendar"]["evidence_id"])
    if calendar_item is None or calendar_item.get("kind") != "calendar":
        raise ValidationError("frozen calendar evidence is missing")
    calendar = calendar_item.get("payload")
    if not isinstance(calendar, dict) or calendar.get("identity") != contract["calendar"]["identity"]:
        raise ValidationError("frozen calendar identity drift")
    raw_sessions = calendar.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValidationError("frozen calendar sessions are missing")
    sessions = [_stored_time(value, "calendar.session") for value in raw_sessions]
    if sessions != sorted(set(sessions)):
        raise ValidationError("frozen calendar sessions must be ordered and unique")

    benchmark_item = by_id.get(contract["benchmark"]["evidence_id"])
    if benchmark_item is None or benchmark_item.get("kind") != "benchmark":
        raise ValidationError("frozen benchmark evidence is missing")
    benchmark = benchmark_item.get("payload")
    if not isinstance(benchmark, dict) or benchmark.get("symbol") != contract["benchmark"]["symbol"]:
        raise ValidationError("frozen benchmark identity drift")

    allowed = set(contract["counterfactual_assumption_versions"])
    counterfactuals = payload.get("counterfactuals", [])
    if not isinstance(counterfactuals, list):
        raise ValidationError("counterfactuals must be a list")
    for item in counterfactuals:
        if not isinstance(item, dict) or item.get("assumption_version") not in allowed:
            raise ValidationError("unsupported counterfactual assumption version")
    return payload, by_id, sessions, json.loads(canonical_json(counterfactuals))


def _horizon_time(anchor, horizon, sessions):
    eligible = [session for session in sessions if session > anchor]
    offset = horizon["session_offset"]
    if len(eligible) < offset:
        raise ValidationError("frozen calendar cannot resolve horizon")
    return eligible[offset - 1]


def _advisory_keys(digest):
    raw = bytes.fromhex(digest)
    return int.from_bytes(raw[:4], "big", signed=True), int.from_bytes(raw[4:8], "big", signed=True)


class OutcomeService:
    """Validate, add, and flush outcomes while the caller owns the transaction."""

    def __init__(self, session):
        self.session = session

    def _lock_digests(self, logical_digests):
        if self.session.get_bind().dialect.name != "postgresql":
            return
        for digest in sorted(logical_digests):
            key1, key2 = _advisory_keys(digest)
            self.session.execute(
                text("SELECT pg_advisory_xact_lock(:key1, :key2)"),
                {"key1": key1, "key2": key2},
            )

    def _contract(self, version):
        required_text(version, "label_definition_version")
        rows = self.session.scalars(
            select(OutcomeLabelContract).where(OutcomeLabelContract.version == version)
        ).all()
        if len(rows) != 1:
            raise ValidationError("label contract version must resolve exactly once")
        contract = rows[0]
        validate_outcome_label_contract(contract)
        return contract

    def _snapshot(self, snapshot_id):
        snapshot = self.session.get(EvaluationSnapshot, _uuid(snapshot_id, "evaluation_snapshot_id"))
        if snapshot is None:
            raise ValidationError("evaluation snapshot does not exist")
        if snapshot.content_sha256 != content_sha256(_snapshot_body(snapshot)):
            raise ValidationError("evaluation snapshot content hash drift")
        run = self.session.get(EvaluationRun, snapshot.evaluation_run_id)
        if run is None or run.status != "complete":
            raise ValidationError("evaluation run is not a frozen complete run")
        return snapshot, run

    def _decisions(self, snapshot, run):
        snapshot_id = str(snapshot.id)
        decisions = self.session.scalars(
            select(DecisionRecord).where(DecisionRecord.evaluation_run_id == run.id)
        ).all()
        matched = []
        for decision in decisions:
            final = decision.final_json
            intents = final.get("order_intents") if isinstance(final, dict) else None
            if not isinstance(intents, list):
                continue
            if any(isinstance(intent, dict) and intent.get("evaluation_snapshot_id") == snapshot_id for intent in intents):
                matched.append(decision)
        return matched

    def _research_assessment(self, snapshot, as_of):
        rows = self.session.scalars(
            select(ResearchAssessment)
            .where(ResearchAssessment.evaluation_snapshot_id == snapshot.id)
            .order_by(ResearchAssessment.assessment_at.desc(), ResearchAssessment.id.desc())
        ).all()
        eligible = [row for row in rows if _stored_time(row.assessment_at, "assessment_at") <= as_of]
        return eligible[0] if eligible else None

    def _validate_references(self, candidates):
        from poseidon.decision_loop.outcome_accounting import _account_reconciliation_sha256
        from poseidon.models.account_reconciliation import AccountReconciliation
        from poseidon.models.outcome import EconomicReconciliation, FillCostRevision

        for item in candidates:
            references = item.get("_references", {})
            costs = references.get("fill_cost_revisions", [])
            for reference in costs:
                row = self.session.get(
                    FillCostRevision,
                    _uuid(reference.get("id"), "fill_cost_revision_id"),
                )
                if row is None or row.content_sha256 != reference.get("content_sha256"):
                    raise ValidationError("fill cost revision hash drift")
            economic_id = references.get("economic_reconciliation_id")
            if economic_id is not None:
                economic = self.session.get(
                    EconomicReconciliation,
                    _uuid(economic_id, "economic_reconciliation_id"),
                )
                if (
                    economic is None
                    or economic.content_sha256 != references.get("economic_reconciliation_sha256")
                    or economic.cost_revision_ids_json != costs
                ):
                    raise ValidationError("economic reconciliation hash drift")
            account_id = references.get("account_reconciliation_id")
            if account_id is not None:
                reconciliation = self.session.get(
                    AccountReconciliation,
                    _uuid(account_id, "account_reconciliation_id"),
                )
                if (
                    reconciliation is None
                    or _account_reconciliation_sha256(reconciliation)
                    != references.get("account_reconciliation_sha256")
                ):
                    raise ValidationError("account reconciliation hash drift")
            assessment_id = references.get("assessment_id")
            if assessment_id is not None:
                assessment = self.session.get(
                    ResearchAssessment,
                    _uuid(assessment_id, "research_assessment_id"),
                )
                if assessment is None or assessment.content_sha256 != references.get(
                    "assessment_content_sha256"
                ):
                    raise ValidationError("research assessment hash drift")

    @staticmethod
    def _signal_metrics(snapshot, anchor, maturity, by_id, contract, sessions):
        price_items = [item for item in by_id.values() if item.get("kind") == "price"]
        benchmark_item = by_id[contract["benchmark"]["evidence_id"]]
        if len(price_items) != 1:
            return "unavailable", "price_missing", {"actual": {"status": "unavailable", "reason": "price_missing"}, "counterfactual": []}
        price_payload = price_items[0].get("payload")
        benchmark_payload = benchmark_item.get("payload")
        if not isinstance(price_payload, dict) or price_payload.get("symbol") != snapshot.symbol:
            raise ValidationError("frozen price identity drift")
        price_values = price_payload.get("values")
        benchmark_values = benchmark_payload.get("values") if isinstance(benchmark_payload, dict) else None
        anchor_key, maturity_key = _iso(anchor), _iso(maturity)
        if (
            not isinstance(price_values, dict)
            or not isinstance(benchmark_values, dict)
            or anchor_key not in price_values
            or maturity_key not in price_values
            or anchor_key not in benchmark_values
            or maturity_key not in benchmark_values
        ):
            return "unavailable", "price_missing", {"actual": {"status": "unavailable", "reason": "price_missing"}, "counterfactual": []}
        anchor_price = _decimal(price_values[anchor_key], "price.anchor")
        target_price = _decimal(price_values[maturity_key], "price.horizon")
        benchmark_anchor = _decimal(benchmark_values[anchor_key], "benchmark.anchor")
        benchmark_target = _decimal(benchmark_values[maturity_key], "benchmark.horizon")
        if min(anchor_price, target_price, benchmark_anchor, benchmark_target) <= 0:
            raise ValidationError("frozen prices must be positive")
        path = []
        for session in sessions:
            if anchor <= session <= maturity:
                key = _iso(session)
                if key not in price_values:
                    return "unavailable", "price_missing", {"actual": {"status": "unavailable", "reason": "price_missing"}, "counterfactual": []}
                path.append(_decimal(price_values[key], "price.path") / anchor_price - 1)
        forward = target_price / anchor_price - 1
        benchmark = benchmark_target / benchmark_anchor - 1
        actual = {
            "status": "available",
            "forward_return": _decimal_string(forward),
            "benchmark_return": _decimal_string(benchmark),
            "excess_return": _decimal_string(forward - benchmark),
            "mae": _decimal_string(min(path)),
            "mfe": _decimal_string(max(path)),
        }
        return "available", "available", {"actual": actual, "counterfactual": []}

    def _trade_metrics(self, snapshot, decision, contract, counterfactuals):
        from poseidon.decision_loop.outcome_accounting import OutcomeAccounting

        return OutcomeAccounting(self.session).compute_trade(
            evaluation_snapshot_id=snapshot.id,
            decision=decision,
            label_contract=contract,
            counterfactuals=counterfactuals,
        )

    def _research_metrics(self, snapshot, assessment, manifest, by_id):
        if assessment is None:
            return (
                "unavailable",
                "research_assessment_missing",
                {"actual": {"status": "unavailable", "reason": "research_assessment_missing"}, "counterfactual": []},
                {},
            )
        if assessment.manifest_id != manifest.id:
            raise ValidationError("research assessment manifest drift")
        if assessment.status not in {"confirmed", "not_confirmed", "unavailable"}:
            raise ValidationError("research assessment status is invalid")
        if assessment.content_sha256 != content_sha256(assessment.assessment_json):
            raise ValidationError("research assessment content hash drift")
        if assessment.assessment_json.get("status") != assessment.status:
            raise ValidationError("research assessment status drift")
        citations = assessment.citation_ids_json
        if (
            not isinstance(citations, list)
            or citations != assessment.assessment_json.get("citation_ids")
            or any(item not in by_id or by_id[item].get("kind") != "research" for item in citations)
        ):
            raise ValidationError("research assessment citations are absent from outcome manifest")
        revision = self.session.get(ResearchRevision, assessment.research_revision_id)
        if revision is None or revision.content_sha256 != content_sha256(revision.research_json):
            raise ValidationError("research assessment revision content hash drift")
        if assessment.status == "unavailable":
            return (
                "unavailable",
                "research_assessment_unavailable",
                {"actual": {"status": "unavailable", "reason": "research_assessment_unavailable"}, "counterfactual": []},
                {"assessment_id": str(assessment.id), "assessment_content_sha256": assessment.content_sha256},
            )
        reason = "research_confirmed" if assessment.status == "confirmed" else "research_not_confirmed"
        actual = {
            "status": "available",
            "research_result": assessment.status,
            "assessment_id": str(assessment.id),
            "assessment_content_sha256": assessment.content_sha256,
        }
        return "available", reason, {"actual": actual, "counterfactual": []}, actual

    def _append(self, item):
        item = {key: value for key, value in item.items() if key != "_references"}
        query = select(OutcomeRecord).where(
            OutcomeRecord.logical_key_sha256 == item["logical_key_sha256"],
            OutcomeRecord.input_sha256 == item["input_sha256"],
        )
        existing = self.session.scalar(query)
        if existing is not None:
            return existing
        previous = self.session.scalar(
            select(OutcomeRecord)
            .where(OutcomeRecord.logical_key_sha256 == item["logical_key_sha256"])
            .order_by(OutcomeRecord.revision_no.desc())
            .limit(1)
            .with_for_update()
        )
        fields = {
            **item,
            "revision_no": 1 if previous is None else previous.revision_no + 1,
            "previous_outcome_id": None if previous is None else previous.id,
            "previous_revision_no": None if previous is None else previous.revision_no,
        }
        content = {
            key: (str(value) if isinstance(value, uuid.UUID) else _iso(value) if isinstance(value, datetime) else value)
            for key, value in fields.items()
            if key != "content_sha256"
        }
        fields["content_sha256"] = content_sha256(content)
        try:
            with self.session.begin_nested():
                row = OutcomeRecord(**fields)
                self.session.add(row)
                self.session.flush()
            return row
        except IntegrityError:
            existing = self.session.scalar(query)
            if existing is None:
                raise
            return existing

    def label_mature_outcomes(
        self,
        *,
        evaluation_snapshot_ids,
        as_of,
        label_definition_version,
        outcome_manifest_id,
    ):
        if not isinstance(evaluation_snapshot_ids, (list, tuple)) or not evaluation_snapshot_ids:
            raise ValidationError("evaluation_snapshot_ids must be a non-empty bounded list")
        as_of = _stored_time(as_of, "as_of")
        contract_row = self._contract(label_definition_version)
        contract = validate_outcome_label_contract(contract_row)
        manifest = self.session.get(DataManifest, _uuid(outcome_manifest_id, "outcome_manifest_id"))
        if manifest is None:
            raise ValidationError("outcome manifest does not exist")
        _payload, by_id, sessions, counterfactuals = _manifest_facts(manifest, contract, as_of)

        candidates = []
        for snapshot_id in evaluation_snapshot_ids:
            snapshot, run = self._snapshot(snapshot_id)
            anchor = _stored_time(run.decision_as_of, "decision_as_of")
            assessment = self._research_assessment(snapshot, as_of)
            decisions = self._decisions(snapshot, run)
            for kind in ("signal", "trade", "research"):
                for horizon in contract["horizons"][kind]:
                    maturity = _horizon_time(anchor, horizon, sessions)
                    if kind == "research" and assessment is not None:
                        maturity = min(maturity, _stored_time(assessment.assessment_at, "assessment_at"))
                    if as_of < maturity:
                        continue
                    subjects = decisions if kind == "trade" else [None]
                    for decision in subjects:
                        if kind == "signal":
                            status, reason, metrics = self._signal_metrics(
                                snapshot, anchor, maturity, by_id, contract, sessions
                            )
                            references = {}
                        elif kind == "trade":
                            status, reason, metrics, references = self._trade_metrics(
                                snapshot, decision, contract, counterfactuals
                            )
                        else:
                            status, reason, metrics, references = self._research_metrics(
                                snapshot, assessment, manifest, by_id
                            )
                        logical = content_sha256(
                            {
                                "evaluation_snapshot_id": str(snapshot.id),
                                "kind": kind,
                                "label_contract_version": contract_row.version,
                                "label_contract_sha256": contract_row.contract_sha256,
                                "horizon_key": horizon["key"],
                                "decision_id": str(decision.id) if decision is not None else None,
                            }
                        )
                        input_sha = content_sha256(
                            {
                                "logical_key_sha256": logical,
                                "evaluation_snapshot_sha256": snapshot.content_sha256,
                                "outcome_manifest_id": str(manifest.id),
                                "outcome_manifest_sha256": manifest.content_sha256,
                                "label_contract_id": str(contract_row.id),
                                "label_contract_sha256": contract_row.contract_sha256,
                                "references": references,
                                "counterfactual": metrics["counterfactual"],
                            }
                        )
                        candidates.append(
                            {
                                "evaluation_snapshot_id": snapshot.id,
                                "kind": kind,
                                "label_contract_id": contract_row.id,
                                "horizon_key": horizon["key"],
                                "manifest_id": manifest.id,
                                "decision_id": None if decision is None else decision.id,
                                "logical_key_sha256": logical,
                                "input_sha256": input_sha,
                                "maturity_at": maturity,
                                "status": status,
                                "reason_code": reason,
                                "metrics_json": metrics,
                                "_references": references,
                            }
                        )

        self._lock_digests(item["logical_key_sha256"] for item in candidates)
        contract = validate_outcome_label_contract(contract_row)
        _payload, _by_id, _sessions, locked_counterfactuals = _manifest_facts(manifest, contract, as_of)
        if locked_counterfactuals != counterfactuals:
            raise ValidationError("counterfactual contract drift")
        for snapshot_id in evaluation_snapshot_ids:
            self._snapshot(snapshot_id)
        self._validate_references(candidates)
        return [self._append(item) for item in candidates]


def _iso(value):
    return _stored_time(value, "timestamp").isoformat().replace("+00:00", "Z")


def label_mature_outcomes(session_factory, **kwargs):
    """Own one short labeling transaction; the service never commits."""
    with session_factory() as session:
        with session.begin():
            rows = OutcomeService(session).label_mature_outcomes(**kwargs)
        return rows
