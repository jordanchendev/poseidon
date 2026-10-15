"""Reviewed operator CLI for Phase 101 pilot configuration writes.

Run inside the Poseidon image (``python -m poseidon.decision_loop.admin``):

* ``wrap-legacy`` audits legacy ``strategies.config`` rows for the mechanical
  Phase 100 wrap (read-only transaction).
* ``create-version`` inserts one immutable draft ``StrategyVersion`` for one
  named strategy after the release-gate validations.
* ``seed-paper-account`` / ``seed-label-contract`` create exactly one row each
  from operator-supplied JSON.

No command writes the strategy release pointer, its revision, the legacy
``active`` flag, or release events. Release stays with the human ``bootstrap``
API. Every command prints exactly one JSON line; errors go to stderr as
``{"error": code, "detail": text}`` with exit status 2. Output carries ids and
hashes only, never a DSN or a key.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from poseidon.decision_loop.decisions import DecisionPolicy
from poseidon.decision_loop.manifest import ValidationError, canonical_json, content_sha256
from poseidon.decision_loop.outcomes import validate_outcome_label_contract
from poseidon.decision_loop.releases import (
    ExecutableStrategyConfig,
    LiveFeaturePlanError,
    ProtectiveExitConfig,
    RuleExecutablePayload,
    VotingExecutablePayload,
)
from poseidon.models.outcome import OutcomeLabelContract, outcome_label_contract_sha256
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest

WRAPPABLE_STRATEGY_TYPES = ("rule", "voting")
EXECUTABLE_MARKETS = ("tw_stock", "crypto_spot", "crypto_perp")
IDENTITY_KEYS = ("symbol", "market", "interval")
_PAYLOAD_MODELS = {"rule": RuleExecutablePayload, "voting": VotingExecutablePayload}

# Refusal codes (stable; consumed by the 101-16 operator checkpoint evidence).
STRATEGY_TYPE_NOT_RULE_OR_VOTING = "strategy_type_not_rule_or_voting"
MARKET_NOT_EXECUTABLE = "market_not_executable"
MULTI_SYMBOL_ROW = "multi_symbol_row"
PAYLOAD_IDENTITY_MISMATCH = "payload_identity_mismatch"
EXECUTABLE_CONFIG_INVALID = "executable_config_invalid"
LIVE_FEATURE_PLAN_BLOCKED = "live_feature_plan_blocked"
ROUTING_MISMATCH = "routing_mismatch"


class AdminRefusal(Exception):
    """A fail-closed refusal reported as ``{"error": code, "detail": text}``."""

    def __init__(self, code: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.extra = extra


@dataclass(frozen=True)
class WrapAssessment:
    wrapped: dict | None
    reason_codes: tuple[str, ...]
    first_error_loc: str | None = None
    live_feature_specs: list = field(default_factory=list)

    @property
    def wrappable(self) -> bool:
        return not self.reason_codes


def _instrument_for(market: str) -> str:
    return "perpetual" if market == "crypto_perp" else "spot"


def wrap_legacy_config(row) -> dict:
    """Mechanically wrap a legacy flat ``strategies.config`` into the Phase 100 shape.

    Only the ``symbol``/``market``/``interval`` identity keys move out of the
    payload; every other key, including the rules, is carried verbatim.
    """
    legacy = row.config if isinstance(row.config, dict) else {}
    payload = {key: value for key, value in legacy.items() if key not in IDENTITY_KEYS}
    return {
        "strategy_type": row.strategy_type,
        "symbol": row.symbol,
        "market": row.market,
        "instrument": _instrument_for(row.market),
        "interval": row.interval,
        "payload": json.loads(canonical_json(payload)),
    }


def _error_loc(error: PydanticValidationError, prefix: tuple = ()) -> str:
    errors = error.errors()
    if not errors:
        return ".".join(str(part) for part in prefix) or "<root>"
    loc = (*prefix, *errors[0].get("loc", ()))
    return ".".join(str(part) for part in loc) or "<root>"


def assess_legacy_row(row) -> WrapAssessment:
    """Return the wrap verdict for one legacy row without touching the database."""
    codes: list[str] = []
    if row.strategy_type not in WRAPPABLE_STRATEGY_TYPES:
        codes.append(STRATEGY_TYPE_NOT_RULE_OR_VOTING)
    if row.market not in EXECUTABLE_MARKETS:
        codes.append(MARKET_NOT_EXECUTABLE)
    if not isinstance(row.symbol, str) or any(character.isspace() for character in row.symbol):
        codes.append(MULTI_SYMBOL_ROW)
    legacy = row.config
    if isinstance(legacy, dict) and any(key in legacy and legacy[key] != getattr(row, key) for key in IDENTITY_KEYS):
        codes.append(PAYLOAD_IDENTITY_MISMATCH)
    if STRATEGY_TYPE_NOT_RULE_OR_VOTING in codes or MARKET_NOT_EXECUTABLE in codes:
        return WrapAssessment(wrapped=None, reason_codes=tuple(codes))

    if not isinstance(legacy, dict):
        codes.append(EXECUTABLE_CONFIG_INVALID)
        return WrapAssessment(wrapped=None, reason_codes=tuple(codes), first_error_loc="config")

    wrapped = wrap_legacy_config(row)
    try:
        # Validate the payload alone first so the recorded loc points into it.
        _PAYLOAD_MODELS[row.strategy_type].model_validate(wrapped["payload"])
    except PydanticValidationError as error:
        codes.append(EXECUTABLE_CONFIG_INVALID)
        loc = _error_loc(error, ("payload",))
        return WrapAssessment(wrapped=None, reason_codes=tuple(codes), first_error_loc=loc)
    try:
        executable = ExecutableStrategyConfig.model_validate(wrapped)
    except PydanticValidationError as error:
        codes.append(EXECUTABLE_CONFIG_INVALID)
        return WrapAssessment(wrapped=None, reason_codes=tuple(codes), first_error_loc=_error_loc(error))
    except (TypeError, ValueError) as error:
        codes.append(EXECUTABLE_CONFIG_INVALID)
        return WrapAssessment(wrapped=None, reason_codes=tuple(codes), first_error_loc=type(error).__name__)

    specs: list = []
    try:
        specs = [{"feature": name, "params": params} for name, params in executable.live_feature_specs()]
    except LiveFeaturePlanError:
        codes.append(LIVE_FEATURE_PLAN_BLOCKED)
    if not executable.routing_matches(row):
        codes.append(ROUTING_MISMATCH)
    if codes:
        return WrapAssessment(wrapped=None, reason_codes=tuple(codes), live_feature_specs=specs)
    return WrapAssessment(wrapped=wrapped, reason_codes=(), live_feature_specs=specs)


def refusal_codes(row) -> list[str]:
    """Return the ordered wrap refusal codes for one legacy row (empty = wrappable)."""
    return list(assess_legacy_row(row).reason_codes)


def _legacy_config_sha256(row) -> str | None:
    try:
        return content_sha256(row.config)
    except ValidationError:
        return None


def _row_report(row) -> dict:
    assessment = assess_legacy_row(row)
    report = {
        "strategy_id": str(row.id),
        "name": row.name,
        "symbol": row.symbol,
        "interval": row.interval,
        "verdict": "wrappable" if assessment.wrappable else "refused",
        "reason_codes": list(assessment.reason_codes),
        "config_sha256": None if assessment.wrapped is None else content_sha256(assessment.wrapped),
        "legacy_config_sha256": _legacy_config_sha256(row),
        "live_feature_specs": assessment.live_feature_specs,
    }
    if assessment.first_error_loc is not None:
        report["first_error_loc"] = assessment.first_error_loc
    return report


# --------------------------------------------------------------------------- input helpers


def _read_input(stream) -> dict:
    raw = stream.read()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise AdminRefusal("input_invalid", "stdin must be one JSON object") from error
    if not isinstance(value, dict):
        raise AdminRefusal("input_invalid", "stdin must be one JSON object")
    return value


def _require_keys(value: dict, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    missing = [key for key in required if key not in value]
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise AdminRefusal("input_invalid", f"unknown input keys: {', '.join(unknown)}")
    if missing:
        raise AdminRefusal("input_invalid", f"missing input keys: {', '.join(missing)}")


def _operator(value) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AdminRefusal("operator_invalid", "operator must be a non-empty trimmed actor id")
    return value


def _policy(value) -> DecisionPolicy:
    try:
        return DecisionPolicy.model_validate(value)
    except (PydanticValidationError, TypeError, ValueError) as error:
        loc = _error_loc(error) if isinstance(error, PydanticValidationError) else type(error).__name__
        raise AdminRefusal("policy_invalid", f"DecisionPolicy is invalid at {loc}") from error


def _approved_identity():
    from poseidon.core.config import settings

    return (
        settings.decision_loop_approved_account_scope,
        settings.decision_loop_approved_market,
        settings.decision_loop_approved_account_generation,
    )


def _check_settings_identity(policy: DecisionPolicy) -> None:
    """Refuse when a configured approved identity differs from the policy identity."""
    scope, market, generation = _approved_identity()
    mismatches = []
    if scope and scope != policy.account_scope:
        mismatches.append("account_scope")
    if market and market != policy.market:
        mismatches.append("market")
    if generation and (policy.reconciliation is None or generation != policy.reconciliation.account_generation):
        mismatches.append("account_generation")
    if mismatches:
        raise AdminRefusal(
            "settings_identity_mismatch",
            f"approved settings identity differs from policy: {', '.join(mismatches)}",
        )


def _session_factory(session_factory):
    if session_factory is not None:
        return session_factory
    from poseidon.core.database import SessionLocal

    return SessionLocal


# --------------------------------------------------------------------------- commands


def cmd_wrap_legacy(session_factory, *, market: str, strategy_type: str | None) -> dict:
    session = session_factory()
    try:
        session.execute(text("SET TRANSACTION READ ONLY"))
        query = select(StrategyRecord).where(StrategyRecord.market == market)
        if strategy_type is not None:
            query = query.where(StrategyRecord.strategy_type == strategy_type)
        rows = session.scalars(query.order_by(StrategyRecord.name, StrategyRecord.id)).all()
        reports = [_row_report(row) for row in rows]
    finally:
        session.rollback()
        session.close()
    wrappable = sum(1 for report in reports if report["verdict"] == "wrappable")
    return {"rows": reports, "counts": {"wrappable": wrappable, "refused": len(reports) - wrappable}}


def _strategy_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise AdminRefusal("input_invalid", "strategy id must be a UUID") from error


def _version_content(strategy, request: dict):
    _require_keys(request, ("policy", "stop_loss_pct", "operator"), ("artifact",))
    operator = _operator(request["operator"])
    assessment = assess_legacy_row(strategy)
    if not assessment.wrappable:
        raise AdminRefusal(
            "wrap_refused",
            f"legacy config is not wrappable: {', '.join(assessment.reason_codes)}",
            reason_codes=list(assessment.reason_codes),
        )
    if request["stop_loss_pct"] is None:
        raise AdminRefusal("stop_loss_pct_missing", "stop_loss_pct is required")
    try:
        protective = ProtectiveExitConfig.model_validate({"stop_loss_pct": request["stop_loss_pct"]})
    except PydanticValidationError as error:
        raise AdminRefusal("protective_exit_invalid", "stop_loss_pct must be a float in (0, 1)") from error

    policy = _policy(request["policy"])
    if policy.mode != "human":
        raise AdminRefusal("policy_not_human", "policy.mode must be human")
    if policy.release_gate.requires_human_release is not True:
        raise AdminRefusal("policy_release_not_human", "policy.release_gate.requires_human_release must be true")
    if policy.market != strategy.market:
        raise AdminRefusal("policy_market_mismatch", "policy.market does not match the strategy market")
    if policy.reconciliation is None:
        raise AdminRefusal("policy_reconciliation_missing", "policy.reconciliation is required")
    _check_settings_identity(policy)

    config = {**assessment.wrapped, "protective_exit": {"stop_loss_pct": protective.stop_loss_pct}}
    try:
        executable = ExecutableStrategyConfig.model_validate(config)
        executable.live_feature_specs()
    except LiveFeaturePlanError as error:
        raise AdminRefusal(LIVE_FEATURE_PLAN_BLOCKED, "wrapped config is not live-capable") from error
    except (PydanticValidationError, TypeError, ValueError) as error:
        raise AdminRefusal(EXECUTABLE_CONFIG_INVALID, "wrapped config with protective_exit is invalid") from error
    if not executable.routing_matches(strategy):
        raise AdminRefusal(ROUTING_MISMATCH, "wrapped config routing does not match the strategy")

    artifact = {
        "research_required": False,
        "wrapped_from": "strategies.config",
        "legacy_config_sha256": content_sha256(strategy.config),
        "operator": operator,
    }
    extra = request.get("artifact", {})
    if not isinstance(extra, dict):
        raise AdminRefusal("input_invalid", "artifact must be a JSON object")
    collisions = sorted(set(extra) & set(artifact))
    if collisions:
        raise AdminRefusal("artifact_key_collision", f"artifact keys are reserved: {', '.join(collisions)}")
    artifact.update(extra)

    try:
        policy_json = json.loads(canonical_json(request["policy"]))
        artifact = json.loads(canonical_json(artifact))
        config = json.loads(canonical_json(config))
    except ValidationError as error:
        raise AdminRefusal("input_invalid", "input must be finite JSON") from error
    return config, policy_json, artifact


def _version_output(version, *, replay: bool) -> dict:
    return {
        "strategy_version_id": str(version.id),
        "strategy_id": str(version.strategy_id),
        "version_no": version.version_no,
        "status": version.status,
        "content_sha256": version.content_sha256,
        "config_sha256": content_sha256(version.config_json),
        "policy_sha256": content_sha256(version.policy_json),
        "replay": replay,
    }


def cmd_create_version(session_factory, *, strategy_id: str, request: dict) -> dict:
    sid = _strategy_uuid(strategy_id)
    session = session_factory()
    try:
        strategy = session.scalar(
            select(StrategyRecord)
            .where(StrategyRecord.id == sid)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if strategy is None:
            raise AdminRefusal("strategy_not_found", "strategy does not exist")
        config, policy_json, artifact = _version_content(strategy, request)
        digest = strategy_version_digest(config, policy_json, artifact)
        existing = session.scalar(
            select(StrategyVersion).where(
                StrategyVersion.strategy_id == sid,
                StrategyVersion.content_sha256 == digest,
            )
        )
        if existing is not None:
            output = _version_output(existing, replay=True)
            session.rollback()
            return output
        latest = session.scalar(select(func.max(StrategyVersion.version_no)).where(StrategyVersion.strategy_id == sid))
        version = StrategyVersion(
            strategy_id=sid,
            version_no=(latest or 0) + 1,
            status="draft",
            config_json=config,
            policy_json=policy_json,
            artifact_json=artifact,
            content_sha256=digest,
        )
        session.add(version)
        session.flush()
        output = _version_output(version, replay=False)
        session.commit()
        return output
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def _account_output(account, *, operator: str, replay: bool) -> dict:
    return {
        "account_id": str(account.id),
        "account_scope": account.account_scope,
        "account_generation": account.account_generation,
        "opening_cash": account.opening_cash,
        "currency": account.currency,
        "operator": operator,
        "replay": replay,
    }


def _matching_account(session, scope: str, generation: str, reconciliation, operator: str):
    account = session.scalar(
        select(PaperBrokerAccount).where(
            PaperBrokerAccount.account_scope == scope,
            PaperBrokerAccount.account_generation == generation,
        )
    )
    if account is None:
        return None
    if account.opening_cash != reconciliation.opening_cash or account.currency != reconciliation.currency:
        raise AdminRefusal(
            "account_identity_conflict",
            "an account with this scope/generation exists with a different opening_cash or currency",
        )
    return _account_output(account, operator=operator, replay=True)


def cmd_seed_paper_account(session_factory, *, request: dict) -> dict:
    _require_keys(request, ("policy", "operator"))
    operator = _operator(request["operator"])
    policy = _policy(request["policy"])
    if policy.reconciliation is None:
        raise AdminRefusal("policy_reconciliation_missing", "policy.reconciliation is required")
    _check_settings_identity(policy)
    reconciliation = policy.reconciliation
    scope = policy.account_scope
    generation = reconciliation.account_generation

    session = session_factory()
    try:
        replay = _matching_account(session, scope, generation, reconciliation, operator)
        if replay is not None:
            session.rollback()
            return replay
        account = PaperBrokerAccount(
            account_scope=scope,
            account_generation=generation,
            opening_cash=reconciliation.opening_cash,
            currency=reconciliation.currency,
            state_version=0,
        )
        session.add(account)
        try:
            session.flush()
        except IntegrityError:
            # A concurrent seed won the unique (scope, generation) race: re-read it.
            session.rollback()
            replay = _matching_account(session, scope, generation, reconciliation, operator)
            if replay is None:
                raise
            session.rollback()
            return replay
        output = _account_output(account, operator=operator, replay=False)
        session.commit()
        return output
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def _contract_output(row, *, operator: str, replay: bool) -> dict:
    return {
        "label_contract_id": str(row.id),
        "version": row.version,
        "contract_sha256": row.contract_sha256,
        "operator": operator,
        "replay": replay,
    }


def _matching_contract(session, version: str, digest: str, operator: str):
    rows = session.scalars(
        select(OutcomeLabelContract).where(
            (OutcomeLabelContract.version == version) | (OutcomeLabelContract.contract_sha256 == digest)
        )
    ).all()
    if not rows:
        return None
    if any(row.version == version and row.contract_sha256 != digest for row in rows):
        raise AdminRefusal(
            "label_contract_version_conflict",
            "a label contract with this version exists with different content",
        )
    if any(row.contract_sha256 == digest and row.version != version for row in rows):
        raise AdminRefusal(
            "label_contract_content_conflict",
            "identical label contract content is already stored under another version",
        )
    return _contract_output(rows[0], operator=operator, replay=True)


def cmd_seed_label_contract(session_factory, *, request: dict) -> dict:
    _require_keys(request, ("version", "contract", "operator"))
    operator = _operator(request["operator"])
    version = request["version"]
    if not isinstance(version, str) or not version.strip() or version != version.strip():
        raise AdminRefusal("input_invalid", "version must be non-empty trimmed text")
    try:
        contract = validate_outcome_label_contract(request["contract"])
    except ValidationError as error:
        raise AdminRefusal("label_contract_invalid", str(error)) from error
    digest = outcome_label_contract_sha256(contract)

    session = session_factory()
    try:
        replay = _matching_contract(session, version, digest, operator)
        if replay is not None:
            session.rollback()
            return replay
        row = OutcomeLabelContract(version=version, contract_json=contract, contract_sha256=digest)
        try:
            row.verify_contract()
        except ValueError as error:
            raise AdminRefusal("label_contract_invalid", str(error)) from error
        session.add(row)
        try:
            session.flush()
        except IntegrityError:
            # A concurrent seed stored the same content first: re-read it.
            session.rollback()
            replay = _matching_contract(session, version, digest, operator)
            if replay is None:
                raise
            session.rollback()
            return replay
        output = _contract_output(row, operator=operator, replay=False)
        session.commit()
        return output
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- entry point


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m poseidon.decision_loop.admin")
    commands = parser.add_subparsers(dest="command", required=True)

    wrap = commands.add_parser("wrap-legacy", help="audit legacy strategies.config rows (read-only)")
    wrap.add_argument("--market", required=True)
    wrap.add_argument("--strategy-type", default=None)

    create = commands.add_parser("create-version", help="insert one immutable draft StrategyVersion")
    create.add_argument("--strategy-id", required=True)
    create.add_argument("--input", required=True, choices=["-"], help="read the request JSON from stdin")

    account = commands.add_parser("seed-paper-account", help="create the one paper account for a policy")
    account.add_argument("--input", required=True, choices=["-"], help="read the request JSON from stdin")

    contract = commands.add_parser("seed-label-contract", help="create one outcome label contract")
    contract.add_argument("--input", required=True, choices=["-"], help="read the request JSON from stdin")
    return parser


def _emit(stream, value: dict) -> None:
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    stream.flush()


def main(argv=None, *, session_factory=None, stdin=None, stdout=None, stderr=None) -> int:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    args = _parser().parse_args(argv)
    try:
        factory = _session_factory(session_factory)
        if args.command == "wrap-legacy":
            result = cmd_wrap_legacy(factory, market=args.market, strategy_type=args.strategy_type)
        elif args.command == "create-version":
            result = cmd_create_version(factory, strategy_id=args.strategy_id, request=_read_input(stdin))
        elif args.command == "seed-paper-account":
            result = cmd_seed_paper_account(factory, request=_read_input(stdin))
        else:
            result = cmd_seed_label_contract(factory, request=_read_input(stdin))
    except AdminRefusal as refusal:
        _emit(stderr, {"error": refusal.code, "detail": refusal.detail, **refusal.extra})
        return 2
    except SQLAlchemyError as error:
        # Never echo the driver message: it can carry connection details.
        _emit(stderr, {"error": "database_error", "detail": type(error).__name__})
        return 2
    _emit(stdout, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
