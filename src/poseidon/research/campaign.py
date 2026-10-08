"""Frozen campaign and global holdout transaction boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import (
    CampaignEvent,
    CampaignReview,
    ExperimentCampaign,
    HoldoutUse,
    experiment_campaign_contract_sha256,
)
from poseidon.models.strategy_version import StrategyVersion
from poseidon.research.paired_review import PairedReview


class CampaignContractValidationError(ValueError):
    """A campaign or holdout contract is incomplete or inconsistent."""


class FrozenCampaignIdentityConflict(RuntimeError):
    """A campaign digest already identifies different immutable content."""


class CampaignEventIdentityConflict(RuntimeError):
    """An event idempotency digest already identifies different content."""


class HoldoutReuseDenied(RuntimeError):
    """Globally consumed holdout evidence belongs to another campaign."""

    def __init__(self, identity: str, owner_campaign_id: uuid.UUID) -> None:
        self.holdout_identity_sha256 = identity
        self.owner_campaign_id = owner_campaign_id
        super().__init__(f"holdout evidence {identity} is already owned by campaign {owner_campaign_id}")


_TOP_LEVEL_FIELDS = (
    "incumbent_strategy_version_id",
    "candidate_strategy_version_id",
    "incumbent_content_sha256",
    "candidate_content_sha256",
    "declared_difference_json",
    "hypothesis",
    "contract_json",
    "created_by",
)
_CONTRACT_FIELDS = (
    "manifest_set",
    "universe",
    "windows",
    "label",
    "benchmark",
    "cost_fx_contract",
    "purge_gap",
    "regimes",
    "uncertainty_estimator",
    "ablation_arms",
    "gates",
    "runtime_artifact_identity",
    "seed",
    "turnover",
    "capacity",
    "declared_trials",
)
_ARMS = ("fundamental_only", "technical_only", "combined")
_HOLDOUT_FIELDS = (
    "manifest_set",
    "evidence_slice",
    "window",
    "universe",
    "label",
    "benchmark",
    "purge_gap",
)
_PERMIT_TOKEN = object()


def _json_object(value, field: str, *, nonempty: bool = True) -> dict:
    if not isinstance(value, dict) or (nonempty and not value):
        raise CampaignContractValidationError(f"{field} must be a non-empty object")
    try:
        return json.loads(canonical_json(value))
    except ValueError as error:
        raise CampaignContractValidationError(f"{field} must be finite JSON") from error


def _json_value(value, field: str):
    try:
        return json.loads(canonical_json(value))
    except ValueError as error:
        raise CampaignContractValidationError(f"{field} must be finite JSON") from error


def _required_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignContractValidationError(f"{field} is required and must be non-empty text")
    return value


def _digest(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CampaignContractValidationError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CampaignContractValidationError(f"{field} must be a hexadecimal SHA-256 digest") from error
    return value.lower()


def _uuid(value, field: str) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise CampaignContractValidationError(f"{field} must be a UUID") from error


def _positive_number(value, field: str, *, maximum: Decimal | None = None) -> Decimal:
    if isinstance(value, bool):
        raise CampaignContractValidationError(f"{field} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise CampaignContractValidationError(f"{field} must be numeric") from error
    if not number.is_finite() or number <= 0 or (maximum is not None and number > maximum):
        raise CampaignContractValidationError(f"{field} is outside its allowed range")
    return number


def _positive_integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignContractValidationError(f"{field} must be a positive integer")
    return value


def _canonical_panel_scalar(value, *, missing: bool) -> dict:
    if value is None:
        return {"null": "none"}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "+inf" if value > 0 else "-inf"}
        return {"float": value.hex()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, (datetime, date, time)):
        return {"datetime": value.isoformat()}
    if isinstance(value, str):
        return {"string": value}
    if missing:
        return {"null": f"{type(value).__module__}.{type(value).__qualname__}"}
    item = getattr(value, "item", None)
    if callable(item):
        unboxed = item()
        if unboxed is not value:
            return _canonical_panel_scalar(unboxed, missing=False)
    raise CampaignContractValidationError(f"holdout panel contains unsupported scalar {type(value).__name__}")


def _advisory_keys(domain: str, identity: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{domain}{identity}".encode()).digest()
    return int.from_bytes(digest[:4], "big", signed=True), int.from_bytes(digest[4:8], "big", signed=True)


def _lock(session, domain: str, identity: str) -> None:
    key1, key2 = _advisory_keys(domain, identity)
    session.execute(text("SELECT pg_advisory_xact_lock(:key1, :key2)"), {"key1": key1, "key2": key2})


def _validate_complete_contract(value: dict) -> dict:
    if not isinstance(value, dict):
        raise CampaignContractValidationError("campaign contract must be an object")
    missing = [field for field in _TOP_LEVEL_FIELDS if field not in value]
    if missing:
        raise CampaignContractValidationError(f"campaign contract missing required fields: {missing}")

    incumbent_id = _uuid(value["incumbent_strategy_version_id"], "incumbent_strategy_version_id")
    candidate_id = _uuid(value["candidate_strategy_version_id"], "candidate_strategy_version_id")
    declared_difference = _json_object(value["declared_difference_json"], "declared_difference_json")
    hypothesis = _required_text(value["hypothesis"], "hypothesis")
    created_by = _required_text(value["created_by"], "created_by")
    contract = _json_object(value["contract_json"], "contract_json")
    nested_missing = [field for field in _CONTRACT_FIELDS if field not in contract]
    if nested_missing:
        raise CampaignContractValidationError(f"contract_json missing required fields: {nested_missing}")

    for field in ("manifest_set", "universe", "regimes", "declared_trials"):
        if not isinstance(contract[field], list) or not contract[field]:
            raise CampaignContractValidationError(f"contract_json.{field} must be a non-empty list")

    windows = _json_object(contract["windows"], "contract_json.windows")
    for field in ("train", "validation", "holdout"):
        window = _json_object(windows.get(field), f"contract_json.windows.{field}")
        _required_text(window.get("start"), f"contract_json.windows.{field}.start")
        _required_text(window.get("end"), f"contract_json.windows.{field}.end")

    label = _json_object(contract["label"], "contract_json.label")
    _required_text(label.get("version"), "contract_json.label.version")
    if not isinstance(label.get("horizons"), list) or not label["horizons"]:
        raise CampaignContractValidationError("contract_json.label.horizons must be a non-empty list")
    if any(not isinstance(horizon, str) or not horizon for horizon in label["horizons"]):
        raise CampaignContractValidationError("contract_json.label.horizons must contain text values")

    _json_object(contract["benchmark"], "contract_json.benchmark")
    costs = _json_object(contract["cost_fx_contract"], "contract_json.cost_fx_contract")
    _json_object(costs.get("fx"), "contract_json.cost_fx_contract.fx")
    scenarios = costs.get("cost_scenarios")
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        raise CampaignContractValidationError("contract_json.cost_fx_contract.cost_scenarios requires base and stress")
    scenario_names = set()
    amount_fields = ("commission_bps", "tax_bps", "slippage_bps")
    for index, scenario in enumerate(scenarios):
        item = _json_object(scenario, f"contract_json.cost_fx_contract.cost_scenarios[{index}]")
        name = _required_text(item.get("name"), f"contract_json.cost_fx_contract.cost_scenarios[{index}].name")
        if name in scenario_names:
            raise CampaignContractValidationError("frozen cost scenario names must be unique")
        scenario_names.add(name)
        missing_amounts = [field for field in amount_fields if field not in item]
        if missing_amounts:
            raise CampaignContractValidationError(f"frozen cost scenario missing required amounts: {missing_amounts}")
        amounts = [item[field] for field in amount_fields]
        try:
            parsed_amounts = [Decimal(str(amount)) for amount in amounts]
        except (InvalidOperation, TypeError, ValueError) as error:
            raise CampaignContractValidationError("frozen cost scenario amounts must be finite numerics") from error
        if not parsed_amounts or any(not amount.is_finite() or amount < 0 for amount in parsed_amounts):
            raise CampaignContractValidationError("frozen cost scenario amounts must be non-negative")
        if not any(amount > 0 for amount in parsed_amounts):
            raise CampaignContractValidationError("each frozen cost scenario must be non-zero")

    purge_gap = _json_object(contract["purge_gap"], "contract_json.purge_gap")
    _positive_integer(purge_gap.get("gap_eligible_sessions"), "contract_json.purge_gap.gap_eligible_sessions")
    _required_text(purge_gap.get("overlap_method"), "contract_json.purge_gap.overlap_method")

    estimator = _json_object(contract["uncertainty_estimator"], "contract_json.uncertainty_estimator")
    if estimator.get("name") != "ols_hac_intercept":
        raise CampaignContractValidationError("uncertainty estimator must be ols_hac_intercept")
    if estimator.get("kernel") != "bartlett":
        raise CampaignContractValidationError("uncertainty estimator kernel must be bartlett")
    if not isinstance(estimator.get("maxlags_by_horizon"), dict) or set(estimator["maxlags_by_horizon"]) != set(
        label["horizons"]
    ):
        raise CampaignContractValidationError("maxlags_by_horizon must cover every frozen horizon")
    if any(
        isinstance(maxlags, bool) or not isinstance(maxlags, int) or maxlags < 0
        for maxlags in estimator["maxlags_by_horizon"].values()
    ):
        raise CampaignContractValidationError("maxlags_by_horizon values must be non-negative integers")
    if estimator.get("small_sample_correction") is not True:
        raise CampaignContractValidationError("small_sample_correction must be explicitly frozen")
    alpha = _positive_number(estimator.get("alpha"), "contract_json.uncertainty_estimator.alpha")
    if alpha >= 1:
        raise CampaignContractValidationError("contract_json.uncertainty_estimator.alpha must be less than 1")

    if contract["ablation_arms"] != list(_ARMS):
        raise CampaignContractValidationError("ablation_arms must declare fundamental_only, technical_only, combined")
    gates = _json_object(contract["gates"], "contract_json.gates")
    for field in ("minimum_symbols_per_date", "minimum_dates_per_symbol", "minimum_effective_paired_dates"):
        _positive_integer(gates.get(field), f"contract_json.gates.{field}")
    _positive_number(gates.get("minimum_coverage"), "contract_json.gates.minimum_coverage", maximum=Decimal("1"))

    runtime = _json_object(contract["runtime_artifact_identity"], "contract_json.runtime_artifact_identity")
    _required_text(runtime.get("runtime"), "contract_json.runtime_artifact_identity.runtime")
    _digest(runtime.get("artifact_sha256"), "contract_json.runtime_artifact_identity.artifact_sha256")
    if isinstance(contract["seed"], bool) or not isinstance(contract["seed"], int):
        raise CampaignContractValidationError("contract_json.seed must be an integer")

    turnover = _json_object(contract["turnover"], "contract_json.turnover")
    if turnover.get("formula") != "0.5*sum(abs(w_t-w_t_minus_1))":
        raise CampaignContractValidationError("contract_json.turnover.formula is unsupported")
    if not isinstance(turnover.get("cash_included"), bool):
        raise CampaignContractValidationError("contract_json.turnover.cash_included must be boolean")
    capacity = _json_object(contract["capacity"], "contract_json.capacity")
    _positive_integer(capacity.get("adv_lookback_sessions"), "contract_json.capacity.adv_lookback_sessions")
    _positive_number(
        capacity.get("participation_cap"),
        "contract_json.capacity.participation_cap",
        maximum=Decimal("1"),
    )
    if capacity.get("price_volume_adjustment") != "split_adjusted":
        raise CampaignContractValidationError("contract_json.capacity.price_volume_adjustment is unsupported")
    if capacity.get("aggregation") != "min_symbol_capacity":
        raise CampaignContractValidationError("contract_json.capacity.aggregation is unsupported")

    expected_cells = {(str(incumbent_id), arm) for arm in _ARMS} | {(str(candidate_id), arm) for arm in _ARMS}
    actual_cells = set()
    original_trial_ids = set()
    for index, trial in enumerate(contract["declared_trials"]):
        trial = _json_object(trial, f"contract_json.declared_trials[{index}]")
        original_trial_id = _required_text(trial.get("original_trial_id"), "declared trial original_trial_id")
        if original_trial_id in original_trial_ids:
            raise CampaignContractValidationError("declared trial identities must be unique")
        original_trial_ids.add(original_trial_id)
        if trial.get("trial_role") != "paired_evaluation":
            raise CampaignContractValidationError("declared paired trials must use trial_role paired_evaluation")
        version_role = trial.get("version_role")
        if version_role not in ("incumbent", "candidate"):
            raise CampaignContractValidationError("declared trial version_role is invalid")
        version_id = str(_uuid(trial.get("strategy_version_id"), "declared trial strategy_version_id"))
        expected_version_id = str(incumbent_id if version_role == "incumbent" else candidate_id)
        if version_id != expected_version_id:
            raise CampaignContractValidationError("declared trial version_role does not match strategy_version_id")
        arm = trial.get("ablation_arm")
        if arm not in _ARMS:
            raise CampaignContractValidationError("declared trial ablation_arm is invalid")
        actual_cells.add((version_id, arm))
    if len(contract["declared_trials"]) != 6 or actual_cells != expected_cells:
        raise CampaignContractValidationError("declared_trials must contain the exact six paired cells")

    return {
        "incumbent_strategy_version_id": incumbent_id,
        "candidate_strategy_version_id": candidate_id,
        "incumbent_content_sha256": _digest(value["incumbent_content_sha256"], "incumbent_content_sha256"),
        "candidate_content_sha256": _digest(value["candidate_content_sha256"], "candidate_content_sha256"),
        "declared_difference_json": declared_difference,
        "hypothesis": hypothesis,
        "contract_json": contract,
        "created_by": created_by,
    }


def _campaign_equal(row: ExperimentCampaign, fields: dict) -> bool:
    return all(
        getattr(row, name) == fields[name]
        for name in (
            "incumbent_strategy_version_id",
            "candidate_strategy_version_id",
            "incumbent_content_sha256",
            "candidate_content_sha256",
            "declared_difference_json",
            "hypothesis",
            "contract_json",
            "created_by",
        )
    )


class CampaignService:
    """Validate and append campaign facts without owning the outer transaction."""

    def __init__(self, session) -> None:
        self.session = session

    def create_frozen_campaign(self, contract: dict) -> ExperimentCampaign:
        fields = _validate_complete_contract(contract)
        for role in ("incumbent", "candidate"):
            version = self.session.get(StrategyVersion, fields[f"{role}_strategy_version_id"])
            if version is None or version.content_sha256 != fields[f"{role}_content_sha256"]:
                raise CampaignContractValidationError(f"{role} strategy version/hash does not match frozen truth")

        digest = experiment_campaign_contract_sha256(
            incumbent_strategy_version_id=fields["incumbent_strategy_version_id"],
            candidate_strategy_version_id=fields["candidate_strategy_version_id"],
            incumbent_content_sha256=fields["incumbent_content_sha256"],
            candidate_content_sha256=fields["candidate_content_sha256"],
            declared_difference_json=fields["declared_difference_json"],
            hypothesis=fields["hypothesis"],
            contract_json=fields["contract_json"],
        )
        _lock(self.session, "phase99:campaign:", digest)
        query = self.session.query(ExperimentCampaign).filter_by(contract_sha256=digest)
        existing = query.one_or_none()
        if existing is not None:
            if not _campaign_equal(existing, fields):
                raise FrozenCampaignIdentityConflict("campaign digest identifies different immutable content")
            return existing

        try:
            with self.session.begin_nested():
                record = ExperimentCampaign(contract_sha256=digest, **fields)
                self.session.add(record)
                self.session.flush()
            return record
        except IntegrityError as error:
            existing = query.one_or_none()
            if existing is None:
                raise
            if not _campaign_equal(existing, fields):
                raise FrozenCampaignIdentityConflict(
                    "campaign digest identifies different immutable content"
                ) from error
            return existing

    def append_event(self, campaign_id, *, event_type: str, payload_json: dict) -> CampaignEvent:
        campaign_id = _uuid(campaign_id, "campaign_id")
        event_type = _required_text(event_type, "event_type")
        payload = _json_object(payload_json, "payload_json", nonempty=False)
        if self.session.get(ExperimentCampaign, campaign_id) is None:
            raise CampaignContractValidationError("campaign does not exist")
        identity = content_sha256({"campaign_id": str(campaign_id), "event_type": event_type, "payload_json": payload})
        query = self.session.query(CampaignEvent).filter_by(
            campaign_id=campaign_id,
            idempotency_sha256=identity,
        )
        existing = query.one_or_none()
        if existing is not None:
            if existing.event_type != event_type or existing.payload_json != payload:
                raise CampaignEventIdentityConflict("event digest identifies different immutable content")
            return existing
        try:
            with self.session.begin_nested():
                record = CampaignEvent(
                    campaign_id=campaign_id,
                    event_type=event_type,
                    payload_json=payload,
                    idempotency_sha256=identity,
                )
                self.session.add(record)
                self.session.flush()
            return record
        except IntegrityError as error:
            existing = query.one_or_none()
            if existing is None:
                raise
            if existing.event_type != event_type or existing.payload_json != payload:
                raise CampaignEventIdentityConflict("event digest identifies different immutable content") from error
            return existing


def build_holdout_identity(value: dict) -> str:
    """Hash only the frozen evidence slice, never campaign or candidate identity."""

    if not isinstance(value, dict):
        raise CampaignContractValidationError("holdout identity must be an object")
    missing = [field for field in _HOLDOUT_FIELDS if field not in value]
    if missing:
        raise CampaignContractValidationError(f"holdout identity missing required fields: {missing}")
    evidence = {field: _json_value(value[field], f"holdout.{field}") for field in _HOLDOUT_FIELDS}
    for field, item in evidence.items():
        if item in (None, "", [], {}):
            raise CampaignContractValidationError(f"holdout.{field} must be non-empty")
    return content_sha256(evidence)


class HoldoutReadPermit:
    """Non-forgeable in-process proof that holdout ownership already committed."""

    __slots__ = ("campaign_contract_sha256", "campaign_id", "holdout_identity_sha256", "holdout_use_id")

    def __init__(
        self,
        token,
        holdout_use_id: uuid.UUID,
        holdout_identity_sha256: str,
        campaign_id: uuid.UUID,
        campaign_contract_sha256: str,
    ) -> None:
        if token is not _PERMIT_TOKEN:
            raise TypeError("HoldoutReadPermit can only be issued after committed consumption")
        self.holdout_use_id = holdout_use_id
        self.holdout_identity_sha256 = holdout_identity_sha256
        self.campaign_id = campaign_id
        self.campaign_contract_sha256 = campaign_contract_sha256

    def materialize(self, loader, *args, **kwargs):
        if not callable(loader):
            raise TypeError("loader must be callable")
        return loader(*args, permit=self, **kwargs)


def consume_holdout_in_committed_transaction(
    session_factory,
    campaign_id,
    campaign_contract_sha256: str,
    holdout_identity: str,
) -> HoldoutReadPermit:
    """Commit global ownership and audit before returning loader authority."""

    campaign_id = _uuid(campaign_id, "campaign_id")
    contract_digest = _digest(campaign_contract_sha256, "campaign_contract_sha256")
    identity = _digest(holdout_identity, "holdout_identity_sha256")
    denied_owner = None
    use = None
    use_id = None

    with session_factory() as session, session.begin():
        campaign = session.get(ExperimentCampaign, campaign_id)
        if campaign is None:
            raise CampaignContractValidationError("campaign does not exist")
        if campaign.contract_sha256 != contract_digest:
            raise CampaignContractValidationError("campaign contract hash mismatch")

        query = session.query(HoldoutUse).filter_by(holdout_identity_sha256=identity)
        use = query.one_or_none()
        inserted = False
        if use is None:
            try:
                with session.begin_nested():
                    use = HoldoutUse(
                        holdout_identity_sha256=identity,
                        campaign_id=campaign_id,
                        campaign_contract_sha256=contract_digest,
                        audit_json={"campaign_id": str(campaign_id)},
                        consumed_at=datetime.now(UTC),
                    )
                    session.add(use)
                    session.flush()
                    inserted = True
            except IntegrityError:
                use = query.one_or_none()
                if use is None:
                    raise

        service = CampaignService(session)
        if use.campaign_id == campaign_id and use.campaign_contract_sha256 == contract_digest:
            service.append_event(
                campaign_id,
                event_type="holdout_consumed" if inserted else "holdout_replayed",
                payload_json={
                    "holdout_identity_sha256": identity,
                    "holdout_use_id": str(use.id),
                },
            )
        else:
            denied_owner = use.campaign_id
            service.append_event(
                campaign_id,
                event_type="holdout_reuse_denied",
                payload_json={
                    "holdout_identity_sha256": identity,
                    "owner_campaign_id": str(use.campaign_id),
                    "owner_campaign_contract_sha256": use.campaign_contract_sha256,
                },
            )
        use_id = use.id

    if denied_owner is not None:
        raise HoldoutReuseDenied(identity, denied_owner)
    return HoldoutReadPermit(_PERMIT_TOKEN, use_id, identity, campaign_id, contract_digest)


def run_paired_review_in_transactions(
    session_factory,
    campaign_id,
    holdout_identity,
    holdout_loader,
) -> CampaignReview:
    """Consume, materialize, compute, then persist one append-only paired review."""

    campaign_id = _uuid(campaign_id, "campaign_id")
    identity = _digest(holdout_identity, "holdout_identity_sha256")
    with session_factory() as session:
        campaign = session.get(ExperimentCampaign, campaign_id)
        if campaign is None:
            raise CampaignContractValidationError("campaign does not exist")
        contract_digest = campaign.contract_sha256

    permit = consume_holdout_in_committed_transaction(
        session_factory,
        campaign_id,
        contract_digest,
        identity,
    )
    loader = holdout_loader.materialize if hasattr(holdout_loader, "materialize") else holdout_loader
    panel = permit.materialize(loader)

    with session_factory() as session:
        campaign = session.get(ExperimentCampaign, campaign_id)
        trials = session.query(ExperimentRecord).filter_by(campaign_id=campaign_id).all()
        cells = [
            {
                "version_role": "incumbent"
                if trial.strategy_version_id == campaign.incumbent_strategy_version_id
                else "candidate",
                "ablation_arm": trial.ablation_arm,
                "terminal_state": trial.terminal_state,
                "paired_sample_key_sha256": trial.paired_sample_key_sha256,
                "sample_membership": (trial.metrics_json or {}).get("sample_membership"),
                "result_sha256": trial.result_sha256,
            }
            for trial in trials
        ]
        contract = json.loads(canonical_json(campaign.contract_json))
        trial_hashes = sorted(trial.result_sha256 for trial in trials)

    if not hasattr(panel, "to_dict") or not hasattr(panel, "isna"):
        raise CampaignContractValidationError("holdout loader must return a pandas DataFrame")
    panel_records = [
        {field: _canonical_panel_scalar(value, missing=bool(missing[field])) for field, value in record.items()}
        for record, missing in zip(
            panel.to_dict(orient="records"),
            panel.isna().to_dict(orient="records"),
            strict=True,
        )
    ]
    panel_digest = content_sha256(sorted(panel_records, key=canonical_json))
    input_digest = content_sha256(
        {
            "campaign_contract_sha256": contract_digest,
            "holdout_identity_sha256": identity,
            "trial_result_sha256": trial_hashes,
            "panel_sha256": panel_digest,
        }
    )
    result = PairedReview(contract, cells).run(panel)
    result_digest = content_sha256(result)

    with session_factory() as session, session.begin():
        _lock(session, "phase99:review:", f"{campaign_id}:{input_digest}")
        review = (
            session.query(CampaignReview).filter_by(campaign_id=campaign_id, input_sha256=input_digest).one_or_none()
        )
        if review is None:
            review = CampaignReview(
                campaign_id=campaign_id,
                input_sha256=input_digest,
                result_sha256=result_digest,
                status=result["status"],
                result_json=result,
            )
            session.add(review)
            session.flush()
        elif review.result_sha256 != result_digest or review.status != result["status"] or review.result_json != result:
            raise FrozenCampaignIdentityConflict("review input identifies different immutable content")
        CampaignService(session).append_event(
            campaign_id,
            event_type="campaign_review_recorded",
            payload_json={
                "campaign_review_id": str(review.id),
                "input_sha256": input_digest,
                "result_sha256": result_digest,
                "status": result["status"],
            },
        )
        session.expunge(review)
    return review
