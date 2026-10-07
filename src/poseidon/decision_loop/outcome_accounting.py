"""Immutable fill-cost provenance and reconciled Decimal trade economics."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from poseidon.decision_loop.manifest import ValidationError, content_sha256, required_text
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.fill_allocation import FillAllocation
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.outcome import EconomicReconciliation, FillCostComponent, FillCostRevision
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.position_lot import PositionLot

_COMPONENT_TYPES = frozenset({"commission", "tax", "funding", "borrow", "fx", "other"})
_CLASSIFICATIONS = frozenset({"actual", "estimated", "not_applicable", "unavailable"})
_QUANTUM = Decimal("0.000000000000000001")


def _decimal(value, field):
    if isinstance(value, bool) or value is None:
        raise ValidationError(f"{field} must be a Decimal value")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be a Decimal value") from error
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


def _iso(value):
    if not isinstance(value, datetime):
        raise ValidationError("timestamp must be a datetime")
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")


def _advisory_keys(digest):
    raw = bytes.fromhex(digest)
    return int.from_bytes(raw[:4], "big", signed=True), int.from_bytes(raw[4:8], "big", signed=True)


def _account_reconciliation_sha256(row):
    return content_sha256(
        {
            "id": str(row.id),
            "account_scope": row.account_scope,
            "account_generation": row.account_generation,
            "as_of": _iso(row.as_of),
            "broker_state_watermark": row.broker_state_watermark,
            "internal_state_watermark": row.internal_state_watermark,
            "broker_snapshot_sha256": row.broker_snapshot_sha256,
            "broker_snapshot_json": row.broker_snapshot_json,
            "internal_snapshot_json": row.internal_snapshot_json,
            "difference_json": row.difference_json,
            "policy_sha256": row.policy_sha256,
            "status": row.status,
        }
    )


class FillCostService:
    """Append per-fill cost facts while the caller owns the transaction."""

    def __init__(self, session):
        self.session = session

    def _normalize_components(self, components, reporting_currency, cost_model_version):
        if not isinstance(components, (list, tuple)) or not components:
            raise ValidationError("components must be a non-empty bounded list")
        normalized = []
        identities = set()
        for raw in components:
            if not isinstance(raw, dict):
                raise ValidationError("cost component must be an object")
            component_type = raw.get("component_type")
            classification = raw.get("classification")
            if component_type not in _COMPONENT_TYPES:
                raise ValidationError("cost component type is unsupported")
            if classification not in _CLASSIFICATIONS:
                raise ValidationError("cost component classification is unsupported")
            source = required_text(raw.get("source"), "component.source")
            identity = component_type, source
            if identity in identities:
                raise ValidationError("cost component identities must be unique")
            identities.add(identity)
            native_currency = required_text(raw.get("native_currency"), "component.native_currency")
            component_reporting = required_text(
                raw.get("reporting_currency", reporting_currency),
                "component.reporting_currency",
            )
            if component_reporting != reporting_currency:
                raise ValidationError("cost component reporting currency drift")

            available = classification in {"actual", "estimated"}
            native_amount = _decimal(raw.get("native_amount"), "component.native_amount") if available else None
            reporting_amount = (
                _decimal(raw.get("reporting_amount"), "component.reporting_amount") if available else None
            )
            if not available and (raw.get("native_amount") is not None or raw.get("reporting_amount") is not None):
                raise ValidationError("non-valued cost component cannot carry amounts")

            fx_source = raw.get("fx_source")
            fx_rate = raw.get("fx_rate")
            fx_as_of = raw.get("fx_as_of")
            if available and native_currency != reporting_currency:
                required_text(fx_source, "component.fx_source")
                rate = _decimal(fx_rate, "component.fx_rate")
                if rate <= 0 or not isinstance(fx_as_of, datetime):
                    raise ValidationError("cross-currency cost requires complete frozen FX")
                if (native_amount * rate).quantize(_QUANTUM) != reporting_amount.quantize(_QUANTUM):
                    raise ValidationError("cross-currency cost FX conversion mismatch")
                fx_rate = rate
            elif fx_rate is not None:
                fx_rate = _decimal(fx_rate, "component.fx_rate")
                if fx_rate <= 0:
                    raise ValidationError("component FX rate must be positive")
            if classification == "unavailable":
                required_text(raw.get("reason"), "component.reason")

            normalized.append(
                {
                    "component_type": component_type,
                    "native_amount": None if native_amount is None else _decimal_string(native_amount),
                    "native_currency": native_currency,
                    "reporting_amount": None if reporting_amount is None else _decimal_string(reporting_amount),
                    "reporting_currency": reporting_currency,
                    "classification": classification,
                    "source": source,
                    "cost_model_version": cost_model_version,
                    "fx_source": fx_source,
                    "fx_rate": None if fx_rate is None else _decimal_string(fx_rate),
                    "fx_as_of": None if fx_as_of is None else _iso(fx_as_of),
                    "reason": raw.get("reason"),
                }
            )
        return sorted(normalized, key=lambda item: (item["component_type"], item["source"]))

    def append_revision(
        self,
        *,
        order_fill_id=None,
        paper_broker_fill_id=None,
        reporting_currency,
        cost_model_version,
        components,
    ):
        if (order_fill_id is None) == (paper_broker_fill_id is None):
            raise ValidationError("exactly one durable fill identity is required")
        reporting_currency = required_text(reporting_currency, "reporting_currency")
        cost_model_version = required_text(cost_model_version, "cost_model_version")
        if order_fill_id is not None:
            fill_id = _uuid(order_fill_id, "order_fill_id")
            if self.session.get(OrderFillRecord, fill_id) is None:
                raise ValidationError("order fill does not exist")
            fill_type = "order_fill"
        else:
            fill_id = _uuid(paper_broker_fill_id, "paper_broker_fill_id")
            if self.session.get(PaperBrokerFill, fill_id) is None:
                raise ValidationError("paper broker fill does not exist")
            fill_type = "paper_broker_fill"
        fill_key = content_sha256({"fill_type": fill_type, "fill_id": str(fill_id)})
        normalized = self._normalize_components(components, reporting_currency, cost_model_version)
        input_sha = content_sha256(
            {
                "fill_key_sha256": fill_key,
                "reporting_currency": reporting_currency,
                "cost_model_version": cost_model_version,
                "components": normalized,
            }
        )
        replay_query = select(FillCostRevision).where(
            FillCostRevision.fill_key_sha256 == fill_key,
            FillCostRevision.input_sha256 == input_sha,
        )
        existing = self.session.scalar(replay_query)
        if existing is not None:
            return existing
        if self.session.get_bind().dialect.name == "postgresql":
            key1, key2 = _advisory_keys(fill_key)
            self.session.execute(
                text("SELECT pg_advisory_xact_lock(:key1, :key2)"),
                {"key1": key1, "key2": key2},
            )
        existing = self.session.scalar(replay_query)
        if existing is not None:
            return existing
        previous = self.session.scalar(
            select(FillCostRevision)
            .where(FillCostRevision.fill_key_sha256 == fill_key)
            .order_by(FillCostRevision.revision_no.desc())
            .limit(1)
            .with_for_update()
        )
        revision_no = 1 if previous is None else previous.revision_no + 1
        content = {
            "fill_type": fill_type,
            "fill_id": str(fill_id),
            "fill_key_sha256": fill_key,
            "reporting_currency": reporting_currency,
            "cost_model_version": cost_model_version,
            "input_sha256": input_sha,
            "revision_no": revision_no,
            "previous_fill_cost_revision_id": None if previous is None else str(previous.id),
            "previous_revision_no": None if previous is None else previous.revision_no,
            "components": normalized,
        }
        try:
            with self.session.begin_nested():
                row = FillCostRevision(
                    order_fill_id=fill_id if fill_type == "order_fill" else None,
                    paper_broker_fill_id=fill_id if fill_type == "paper_broker_fill" else None,
                    fill_key_sha256=fill_key,
                    reporting_currency=reporting_currency,
                    cost_model_version=cost_model_version,
                    input_sha256=input_sha,
                    content_sha256=content_sha256(content),
                    revision_no=revision_no,
                    previous_fill_cost_revision_id=None if previous is None else previous.id,
                    previous_revision_no=None if previous is None else previous.revision_no,
                )
                self.session.add(row)
                self.session.flush()
                for item in normalized:
                    self.session.add(
                        FillCostComponent(
                            fill_cost_revision_id=row.id,
                            component_type=item["component_type"],
                            native_amount=item["native_amount"],
                            native_currency=item["native_currency"],
                            reporting_amount=item["reporting_amount"],
                            reporting_currency=item["reporting_currency"],
                            classification=item["classification"],
                            source=item["source"],
                            cost_model_version=item["cost_model_version"],
                            fx_source=item["fx_source"],
                            fx_rate=item["fx_rate"],
                            fx_as_of=None if item["fx_as_of"] is None else datetime.fromisoformat(item["fx_as_of"].replace("Z", "+00:00")),
                            reason=item["reason"],
                        )
                    )
                self.session.flush()
            return row
        except IntegrityError:
            existing = self.session.scalar(replay_query)
            if existing is None:
                raise
            return existing


class OutcomeAccounting:
    """Resolve exact durable facts and append one economic reconciliation."""

    def __init__(self, session):
        self.session = session

    @staticmethod
    def _order_matches(order, snapshot_id):
        intent = order.intent_json
        if not isinstance(intent, dict):
            return False
        frozen = intent.get("frozen_intent")
        return intent.get("evaluation_snapshot_id") == snapshot_id or (
            isinstance(frozen, dict) and frozen.get("evaluation_snapshot_id") == snapshot_id
        )

    @staticmethod
    def _provisional(reason, counterfactuals, references=None):
        return (
            "provisional",
            reason,
            {"actual": {"status": "provisional", "reason": reason}, "counterfactual": counterfactuals},
            references or {},
        )

    def _append_reconciliation(
        self,
        account_reconciliation,
        account_sha,
        reporting_currency,
        cost_references,
        fx_facts,
        economics,
        *,
        status="matched",
        reasons=(),
    ):
        body = {
            "account_reconciliation_id": str(account_reconciliation.id),
            "account_reconciliation_sha256": account_sha,
            "reporting_currency": reporting_currency,
            "cost_revision_ids": cost_references,
            "fx_facts": fx_facts,
            "economics": economics,
            "status": status,
            "reason_codes": list(reasons),
        }
        input_sha = content_sha256(body)
        existing = self.session.scalar(
            select(EconomicReconciliation).where(
                EconomicReconciliation.account_reconciliation_id == account_reconciliation.id,
                EconomicReconciliation.input_sha256 == input_sha,
            )
        )
        if existing is not None:
            return existing
        row = EconomicReconciliation(
            account_reconciliation_id=account_reconciliation.id,
            input_sha256=input_sha,
            content_sha256=content_sha256({**body, "input_sha256": input_sha}),
            reporting_currency=reporting_currency,
            cost_revision_ids_json=cost_references,
            fx_facts_json=fx_facts,
            status=status,
            reason_codes_json=list(reasons),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def compute_trade(self, *, evaluation_snapshot_id, decision, label_contract, counterfactuals):
        snapshot_id = str(_uuid(evaluation_snapshot_id, "evaluation_snapshot_id"))
        allowed = set(label_contract["counterfactual_assumption_versions"])
        if any(
            not isinstance(item, dict) or item.get("assumption_version") not in allowed
            for item in counterfactuals
        ):
            raise ValidationError("unsupported counterfactual assumption version")
        orders = self.session.scalars(select(OrderRecord).where(OrderRecord.decision_id == decision.id)).all()
        exact_orders = [order for order in orders if self._order_matches(order, snapshot_id)]
        if not exact_orders:
            return (
                "available",
                "no_execution",
                {"actual": {"status": "not_applicable", "reason": "no_execution"}, "counterfactual": counterfactuals},
                {},
            )
        fills = self.session.scalars(
            select(OrderFillRecord).where(OrderFillRecord.order_id.in_([order.id for order in exact_orders]))
        ).all()
        if not fills:
            return (
                "available",
                "no_execution",
                {"actual": {"status": "not_applicable", "reason": "no_execution"}, "counterfactual": counterfactuals},
                {"order_ids": sorted(str(order.id) for order in exact_orders)},
            )
        identities = {(order.account_scope, order.account_generation) for order in exact_orders}
        if len(identities) != 1 or None in next(iter(identities)):
            return self._provisional("account_identity_inconsistent", counterfactuals)
        account_scope, account_generation = next(iter(identities))
        account_reconciliation = self.session.scalar(
            select(AccountReconciliation)
            .where(
                AccountReconciliation.account_scope == account_scope,
                AccountReconciliation.account_generation == account_generation,
            )
            .order_by(AccountReconciliation.as_of.desc(), AccountReconciliation.id.desc())
            .limit(1)
        )
        if account_reconciliation is None:
            return self._provisional("account_reconciliation_missing", counterfactuals)
        if account_reconciliation.status != "matched" or account_reconciliation.policy_sha256 != decision.policy_sha256:
            return self._provisional("account_reconciliation_not_matched", counterfactuals)
        account_sha = _account_reconciliation_sha256(account_reconciliation)

        fill_by_id = {fill.id: fill for fill in fills}
        order_by_id = {order.id: order for order in exact_orders}
        allocations = self.session.scalars(
            select(FillAllocation).where(FillAllocation.closing_fill_id.in_(list(fill_by_id)))
        ).all()
        if not allocations:
            return self._provisional("fill_allocation_missing", counterfactuals)
        gross = Decimal(0)
        for allocation in allocations:
            if allocation.closing_decision_id != decision.id:
                return self._provisional("fill_allocation_decision_mismatch", counterfactuals)
            lot = self.session.get(PositionLot, allocation.position_lot_id)
            closing_fill = fill_by_id[allocation.closing_fill_id]
            closing_order = order_by_id[closing_fill.order_id]
            realized = allocation.realized_cost_json
            if lot is None or not isinstance(realized, dict):
                return self._provisional("fill_allocation_invalid", counterfactuals)
            if (lot.market, lot.symbol, lot.instrument, lot.side) != (
                closing_order.market,
                closing_order.symbol,
                closing_order.instrument,
                closing_order.side,
            ):
                return self._provisional("fill_allocation_identity_mismatch", counterfactuals)
            quantity = _decimal(allocation.quantity, "allocation.quantity")
            multiplier = _decimal(realized.get("contract_multiplier"), "allocation.contract_multiplier")
            entry_cost = _decimal(realized.get("entry_cost"), "allocation.entry_cost")
            closing_value = _decimal(closing_fill.fill_price, "fill.fill_price") * quantity * multiplier
            gross += closing_value - entry_cost if lot.side == "long" else entry_cost - closing_value

        reporting_currency = label_contract["reporting_currency"]
        required_components = set(label_contract["required_cost_components"])
        cost_total = Decimal(0)
        cost_references = []
        component_metrics = []
        fx_facts = []
        for fill in sorted(fills, key=lambda row: str(row.id)):
            revision = self.session.scalar(
                select(FillCostRevision)
                .where(FillCostRevision.order_fill_id == fill.id)
                .order_by(FillCostRevision.revision_no.desc())
                .limit(1)
            )
            if revision is None:
                return self._provisional("fill_cost_provenance_missing", counterfactuals)
            if (
                revision.reporting_currency != reporting_currency
                or revision.cost_model_version != label_contract["cost_model_version"]
            ):
                return self._provisional("fill_cost_contract_mismatch", counterfactuals)
            components = self.session.scalars(
                select(FillCostComponent)
                .where(FillCostComponent.fill_cost_revision_id == revision.id)
                .order_by(FillCostComponent.component_type, FillCostComponent.source)
            ).all()
            if not required_components.issubset({component.component_type for component in components}):
                return self._provisional("required_cost_component_missing", counterfactuals)
            if any(component.classification == "unavailable" for component in components):
                return self._provisional("cost_component_unavailable", counterfactuals)
            cost_references.append({"id": str(revision.id), "content_sha256": revision.content_sha256})
            for component in components:
                metric = {
                    "fill_id": str(fill.id),
                    "fill_cost_revision_id": str(revision.id),
                    "type": component.component_type,
                    "classification": component.classification,
                    "source": component.source,
                }
                if component.classification in {"actual", "estimated"}:
                    if component.reporting_amount is None:
                        return self._provisional("cost_component_amount_missing", counterfactuals)
                    amount = _decimal(component.reporting_amount, "component.reporting_amount")
                    cost_total += amount
                    metric["reporting_amount"] = _decimal_string(amount)
                    if component.native_currency != reporting_currency:
                        if component.fx_source is None or component.fx_rate is None or component.fx_as_of is None:
                            return self._provisional("fx_provenance_missing", counterfactuals)
                        fx_facts.append(
                            {
                                "fill_cost_component_id": str(component.id),
                                "source": component.fx_source,
                                "rate": _decimal_string(_decimal(component.fx_rate, "component.fx_rate")),
                                "as_of": _iso(component.fx_as_of),
                                "model_version": label_contract["fx_model_version"],
                            }
                        )
                component_metrics.append(metric)

        net = gross - cost_total
        reported_values = {
            order.intent_json.get("economics", {}).get("reported_net")
            for order in exact_orders
            if isinstance(order.intent_json, dict)
            and isinstance(order.intent_json.get("economics"), dict)
            and order.intent_json["economics"].get("reported_net") is not None
        }
        if len(reported_values) > 1:
            return self._provisional("pnl_equation_mismatch", counterfactuals)
        tolerance = _decimal(label_contract["pnl_tolerance"], "pnl_tolerance")
        if reported_values and abs(net - _decimal(next(iter(reported_values)), "reported_net")) > tolerance:
            return self._provisional("pnl_equation_mismatch", counterfactuals)
        economics = {
            "gross": _decimal_string(gross),
            "cost_total": _decimal_string(cost_total),
            "net": _decimal_string(net),
            "tolerance": _decimal_string(tolerance),
        }
        economic = self._append_reconciliation(
            account_reconciliation,
            account_sha,
            reporting_currency,
            cost_references,
            fx_facts,
            economics,
        )
        actual = {
            "status": "available",
            **economics,
            "reporting_currency": reporting_currency,
            "cost_provenance": "estimated"
            if any(item["classification"] == "estimated" for item in component_metrics)
            else "actual",
            "cost_components": component_metrics,
        }
        references = {
            "order_ids": sorted(str(order.id) for order in exact_orders),
            "fill_ids": sorted(str(fill.id) for fill in fills),
            "fill_cost_revisions": cost_references,
            "account_reconciliation_id": str(account_reconciliation.id),
            "account_reconciliation_sha256": account_sha,
            "economic_reconciliation_id": str(economic.id),
            "economic_reconciliation_sha256": economic.content_sha256,
        }
        return "available", "pnl_reconciled", {"actual": actual, "counterfactual": counterfactuals}, references
