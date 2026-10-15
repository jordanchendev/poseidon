"""Deterministic producer: one frozen manifest plus one strategy version.

``build_run_inputs`` is pure: it derives the universe from the strategy version and
the terminal snapshots from manifest evidence only, so identical inputs always hash
identically. ``ProducerService`` persists the run through ``EvaluationService`` and,
only in ``decision`` mode, creates one pending human decision through
``DecisionService.create_decision``. Nothing here approves, claims, submits,
releases, or moves a strategy pointer.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import timedelta

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionPolicy, DecisionService
from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    iso_time,
    timestamp,
    verify_manifest,
)
from poseidon.decision_loop.releases import ExecutableStrategyConfig
from poseidon.models.data_manifest import DataManifest
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.position_lot import PositionLot
from poseidon.models.research_revision import ResearchRevision
from poseidon.models.signal import SignalRecord
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion

logger = logging.getLogger(__name__)

CANDIDATE_STATUS = "evaluated"
PRODUCING_MODES = frozenset({"shadow", "decision"})
LIVE_STRATEGY_TYPES = frozenset({"rule", "voting"})
ENTRY_ACTIONS = frozenset({"long", "short"})
SHADOW_PARITY_ACTOR = "system:shadow-parity"
DECISION_ALLOWED_ACTIONS = ("enter", "exit", "hold", "watch")
_INTERVAL = re.compile(r"^(\d+)([mhdw])$")
_INTERVAL_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def research_scope_key(member: dict) -> str:
    """Research scope for one universe member; SEED-002 owns any future canonical form."""
    return f"{member['market']}:{member['instrument']}:{member['symbol']}"


def interval_delta(interval: str) -> timedelta:
    match = _INTERVAL.fullmatch(interval or "")
    if match is None or int(match.group(1)) <= 0:
        raise ValidationError(f"unsupported interval: {interval}")
    return timedelta(**{_INTERVAL_UNITS[match.group(2)]: int(match.group(1))})


def resolve_worker_principal(actor_id: str, principals) -> AuthPrincipal:
    """Return the configured principal for ``actor_id`` only if it is a pure decision worker."""
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("decision_loop_worker_principal_id is not configured")
    matches = [principal for principal in principals if principal.actor_id == actor_id]
    if len(matches) != 1:
        raise ValueError(f"decision worker principal {actor_id!r} must resolve to exactly one configured principal")
    principal = matches[0]
    if principal.roles != frozenset({"decision-worker"}):
        raise ValueError(f"decision worker principal {actor_id!r} must hold only the decision-worker role")
    if not principal.account_scopes:
        raise ValueError(f"decision worker principal {actor_id!r} has no account scope")
    return principal


def executable_config(strategy_version) -> ExecutableStrategyConfig:
    try:
        return ExecutableStrategyConfig.model_validate(strategy_version.config_json)
    except (PydanticValidationError, TypeError, ValueError) as error:
        raise ValidationError("strategy version is not an executable strategy config") from error


def _member_evidence(evidence: list, kind: str, symbol: str) -> list:
    return [
        item
        for item in evidence
        if item.get("kind") == kind
        and isinstance(item.get("payload"), dict)
        and item["payload"].get("symbol") == symbol
    ]


def _frozen_bars(item: dict, cutoff) -> list[dict]:
    bars = item["payload"].get("bars")
    if not isinstance(bars, list):
        raise ValidationError("ohlcv evidence requires a bars list")
    visible = []
    for bar in bars:
        if not isinstance(bar, dict) or any(field not in bar for field in ("time", "open", "high", "low", "close")):
            raise ValidationError("ohlcv bars require time, open, high, low, close")
        if timestamp(bar["time"], "bar.time") <= cutoff:
            visible.append(bar)
    return sorted(visible, key=lambda bar: timestamp(bar["time"], "bar.time"))


def _terminal(member, status, reason_codes, *, valid_until, technical=None, recommendation=None, research_ids=()):
    # Evaluation requires research_status=not_required whenever no revision is referenced;
    # excluded/failed members therefore reference none and carry the reason in reason_codes.
    research_ids = sorted(research_ids)
    return {
        **member,
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        "recommendation_json": {
            "research_status": "completed" if research_ids else "not_required",
            **(recommendation or {}),
        },
        "technical_json": technical or {},
        "research_revision_ids": research_ids,
        "valid_until": valid_until,
    }


def _evidence_reasons(member, evidence, policy, cutoff) -> list[str]:
    reasons = []
    required = {kind: rule.max_age_seconds for kind, rule in policy.required_data.items()}
    required.setdefault("ohlcv", None)
    for kind in sorted(required):
        items = _member_evidence(evidence, kind, member["symbol"])
        if not items:
            reasons.append(f"{kind}_missing")
            continue
        newest = max(timestamp(item["event_time"], "evidence.event_time") for item in items)
        if required[kind] is not None and (cutoff - newest).total_seconds() > required[kind]:
            reasons.append(f"{kind}_stale")
    return reasons


def _signal_facts(signals) -> list[dict]:
    return sorted(
        (
            {
                "action": str(signal["action"]),
                "confidence": float(signal["confidence"]),
                "quantity_pct": None if signal.get("quantity_pct") is None else float(signal["quantity_pct"]),
            }
            for signal in signals
        ),
        key=canonical_json,
    )


def _member_snapshot(
    member,
    executable,
    evidence,
    policy,
    cutoff,
    valid_until,
    research_lookup,
    signal_evaluator,
    *,
    research_required,
    open_positions=frozenset(),
):
    def terminal(status, reasons, **fields):
        return _terminal(member, status, reasons, valid_until=valid_until, **fields)

    if executable.strategy_type not in LIVE_STRATEGY_TYPES:
        return terminal("excluded", ["strategy_not_live"])
    reasons = _evidence_reasons(member, evidence, policy, cutoff)
    if reasons:
        return terminal("excluded", reasons)

    ohlcv = max(
        _member_evidence(evidence, "ohlcv", member["symbol"]),
        key=lambda item: (timestamp(item["event_time"], "evidence.event_time"), item["id"]),
    )
    try:
        bars = _frozen_bars(ohlcv, cutoff)
    except ValidationError:
        return terminal("excluded", ["ohlcv_invalid"])
    if not bars:
        return terminal("excluded", ["ohlcv_missing"])
    technical = {
        "ohlcv_evidence_id": ohlcv["id"],
        "ohlcv_content_sha256": ohlcv["content_sha256"],
        "bar_count": len(bars),
        "last_bar_time": iso_time(bars[-1]["time"], "bar.time"),
        "strategy_type": executable.strategy_type,
    }

    research_ids = ()
    if research_required:
        revision_id = research_lookup(research_scope_key(member))
        if revision_id is None:
            return terminal("excluded", ["research_missing"], technical=technical)
        research_ids = (str(revision_id),)

    try:
        facts = _signal_facts(signal_evaluator(executable, bars))
    except Exception:  # a strategy failure is a terminal member outcome, never a crash
        logger.exception("frozen strategy evaluation failed for %s", member["symbol"])
        return terminal("failed", ["signal_evaluation_failed"], technical=technical)
    technical["signals"] = facts
    entries = [fact for fact in facts if fact["action"] in ENTRY_ACTIONS]
    closes = [fact for fact in facts if fact["action"] == "close"]
    if closes and entries:
        return terminal(
            "no_trade", ["conflicting_close_and_entry_signals"], technical=technical, research_ids=research_ids
        )
    if closes:
        # A rule close exits only a position the account already holds (D-23.11); never opens or flips.
        identity = (member["market"], member["symbol"], member["instrument"])
        held = sorted(
            side for (market, symbol, instrument, side) in open_positions if (market, symbol, instrument) == identity
        )
        if not held:
            return terminal("no_trade", ["signal_close"], technical=technical, research_ids=research_ids)
        if len(held) != 1:
            return terminal("no_trade", ["ambiguous_open_position"], technical=technical, research_ids=research_ids)
        close = max(closes, key=lambda fact: (fact["confidence"], canonical_json(fact)))
        intent = {
            "symbol": member["symbol"],
            "market": member["market"],
            "instrument": member["instrument"],
            "action": "exit",
            "side": held[0],
            "target_weight": 0.0,
            "order_type": "market",
        }
        return terminal(
            CANDIDATE_STATUS,
            [],
            technical=technical,
            research_ids=research_ids,
            recommendation={"side": held[0], "confidence": close["confidence"], "intent": intent},
        )
    if not entries:
        return terminal("no_trade", ["no_entry_signal"], technical=technical, research_ids=research_ids)
    if len({fact["action"] for fact in entries}) != 1:
        return terminal("no_trade", ["conflicting_entry_signals"], technical=technical, research_ids=research_ids)
    entry = max(entries, key=lambda fact: (fact["confidence"], canonical_json(fact)))
    limit = policy.hard_limits.max_position_weight
    weight = limit if entry["quantity_pct"] is None else min(entry["quantity_pct"], limit)
    if weight <= 0:
        return terminal("no_trade", ["zero_target_weight"], technical=technical, research_ids=research_ids)
    intent = {
        "symbol": member["symbol"],
        "market": member["market"],
        "instrument": member["instrument"],
        "action": "enter",
        "side": entry["action"],
        "target_weight": float(weight),
        "order_type": "market",
    }
    return terminal(
        CANDIDATE_STATUS,
        [],
        technical=technical,
        research_ids=research_ids,
        recommendation={"side": entry["action"], "confidence": entry["confidence"], "intent": intent},
    )


def build_run_inputs(
    manifest_payload: dict,
    strategy_version,
    policy: DecisionPolicy,
    *,
    research_lookup,
    signal_evaluator,
    open_positions=frozenset(),
) -> tuple[list, list]:
    """Pure manifest -> (universe, terminal snapshots); no database, network, Redis, or clock.

    Open positions are an input: a frozenset of ``(market, symbol, instrument, side)`` the caller
    read before calling. A rule ``close`` becomes an ``exit`` candidate only for a held identity.
    """
    executable = executable_config(strategy_version)
    if executable.market != manifest_payload.get("market") or executable.interval != manifest_payload.get("interval"):
        raise ValidationError("strategy version does not match the manifest market and interval")
    cutoff = timestamp(manifest_payload["as_of"], "manifest.as_of")
    valid_until = iso_time(cutoff + timedelta(seconds=policy.decision_ttl_seconds), "valid_until")
    artifact = strategy_version.artifact_json if isinstance(strategy_version.artifact_json, dict) else {}
    research_required = artifact.get("research_required") is True
    evidence = manifest_payload.get("evidence") or []
    universe = [{"symbol": executable.symbol, "market": executable.market, "instrument": executable.instrument}]
    snapshots = [
        _member_snapshot(
            member,
            executable,
            evidence,
            policy,
            cutoff,
            valid_until,
            research_lookup,
            signal_evaluator,
            research_required=research_required,
            open_positions=open_positions,
        )
        for member in universe
    ]
    return universe, snapshots


def decision_risk_snapshot(policy: DecisionPolicy, intents: list[dict]) -> dict:
    """Deterministic decision-time limits; execution re-checks with live NAV and prices."""
    failures = [
        f"{intent['symbol']}:position_limit"
        for intent in intents
        if intent["target_weight"] > policy.hard_limits.max_position_weight
    ]
    if sum(intent["target_weight"] for intent in intents) > policy.hard_limits.max_gross_exposure:
        failures.append("max_gross_exposure")
    return {
        "hard_failures": sorted(failures),
        "allowed_actions": list(DECISION_ALLOWED_ACTIONS),
        "limits": policy.hard_limits.model_dump(mode="json"),
    }


class ProducerService:
    """Persist one evaluation run and, in decision mode, one pending decision."""

    def __init__(self, session, *, signal_evaluator=None):
        self.session = session
        if signal_evaluator is None:
            from poseidon.decision_loop.strategy_signals import evaluate_strategy_signals

            signal_evaluator = evaluate_strategy_signals
        self.signal_evaluator = signal_evaluator

    def active_versions(self, market: str, interval: str, account_scope: str) -> list:
        """Read-only: active versions whose frozen config/policy match the approved identity."""
        rows = self.session.execute(
            select(StrategyRecord, StrategyVersion)
            .join(StrategyVersion, StrategyVersion.id == StrategyRecord.active_version_id)
            .order_by(StrategyRecord.id)
        ).all()
        versions = []
        for _record, version in rows:
            config = version.config_json if isinstance(version.config_json, dict) else {}
            policy = version.policy_json if isinstance(version.policy_json, dict) else {}
            if (config.get("market"), config.get("interval")) != (market, interval):
                continue
            if (policy.get("market"), policy.get("account_scope")) != (market, account_scope):
                continue
            versions.append(version)
        return versions

    def _research_lookup(self, manifest_payload, cutoff):
        def lookup(scope_key):
            rows = self.session.execute(
                select(ResearchRevision, DataManifest)
                .join(DataManifest, DataManifest.id == ResearchRevision.manifest_id)
                .where(ResearchRevision.scope_key == scope_key, ResearchRevision.status == "completed")
            ).all()
            eligible = []
            for revision, manifest in rows:
                payload = manifest.payload_json if isinstance(manifest.payload_json, dict) else {}
                as_of = timestamp(payload.get("as_of"), "research.as_of")
                if (
                    manifest.market == manifest_payload["market"]
                    and payload.get("account_scope") == manifest_payload.get("account_scope")
                    and as_of <= cutoff
                ):
                    eligible.append((as_of, str(revision.id), revision.id))
            return max(eligible)[-1] if eligible else None

        return lookup

    def _portfolio_snapshot(self, policy: DecisionPolicy) -> dict:
        generation = policy.reconciliation.account_generation if policy.reconciliation else None
        positions = []
        if generation is not None:
            lots = self.session.scalars(
                select(PositionLot).where(
                    PositionLot.account_scope == policy.account_scope,
                    PositionLot.account_generation == generation,
                    PositionLot.open_quantity > 0,
                )
            ).all()
            totals = {}
            for lot in lots:
                identity = (lot.market, lot.symbol, lot.instrument, lot.side)
                totals[identity] = totals.get(identity, 0.0) + float(lot.open_quantity)
            positions = [
                {"market": market, "symbol": symbol, "instrument": instrument, "side": side, "open_quantity": quantity}
                for (market, symbol, instrument, side), quantity in sorted(totals.items())
            ]
        return {"account_scope": policy.account_scope, "account_generation": generation, "positions": positions}

    def _shadow_parity(self, version, manifest, rows) -> dict:
        """Compare frozen candidates with legacy signals on the same bar; read-only."""
        candidates = [row for row in rows if row.status == CANDIDATE_STATUS]
        frozen = sorted(
            [row.symbol, row.recommendation_json.get("side")] for row in candidates if _intent_action(row) == "enter"
        )
        frozen_exits = sorted(
            [row.symbol, row.recommendation_json.get("side")] for row in candidates if _intent_action(row) == "exit"
        )
        bar_times = sorted(
            {row.technical_json.get("last_bar_time") for row in rows if row.technical_json.get("last_bar_time")}
        )
        legacy = []
        for bar_time in bar_times:
            signals = self.session.scalars(
                select(SignalRecord).where(
                    SignalRecord.strategy_id == version.strategy_id,
                    SignalRecord.market == manifest.market,
                    SignalRecord.interval == manifest.interval,
                    SignalRecord.signal_time == timestamp(bar_time, "bar.time"),
                    SignalRecord.action.in_(sorted(ENTRY_ACTIONS)),
                )
            ).all()
            legacy.extend([signal.symbol, signal.action] for signal in signals)
        legacy = sorted(legacy)
        payload = {
            "actor_id": SHADOW_PARITY_ACTOR,
            "strategy_version_id": str(version.id),
            "manifest_sha256": manifest.content_sha256,
            "bar_times": bar_times,
            "legacy_targets": legacy,
            "frozen_targets": frozen,
            "frozen_exits": frozen_exits,
            "matched": legacy == frozen,
        }
        payload["parity_sha256"] = content_sha256(payload)
        logger.info("decision-loop shadow parity %s", canonical_json(payload))
        return payload

    def produce(self, strategy_version_id, manifest_id, *, decision_as_of, principal: AuthPrincipal, mode: str) -> dict:
        if mode not in PRODUCING_MODES:
            return {"skipped": mode}
        principal.require_role("decision-worker")
        version = self.session.get(StrategyVersion, _uuid(strategy_version_id, "strategy_version_id"))
        manifest = self.session.get(DataManifest, _uuid(manifest_id, "manifest_id"))
        if version is None or manifest is None:
            raise ValidationError("strategy version and manifest must exist")
        try:
            policy = DecisionPolicy.model_validate(version.policy_json)
        except PydanticValidationError as error:
            raise ValidationError("strategy version has no valid decision policy") from error
        principal.require_account_scope(policy.account_scope)
        payload = verify_manifest(manifest)
        cutoff = timestamp(decision_as_of if decision_as_of is not None else payload["as_of"], "decision_as_of")
        portfolio = self._portfolio_snapshot(policy)
        universe, snapshots = build_run_inputs(
            payload,
            version,
            policy,
            research_lookup=self._research_lookup(payload, cutoff),
            signal_evaluator=self.signal_evaluator,
            open_positions=frozenset(
                (item["market"], item["symbol"], item["instrument"], item["side"]) for item in portfolio["positions"]
            ),
        )
        run = EvaluationService(self.session).evaluate_run(version.id, manifest.id, universe, snapshots, cutoff)
        rows = self.session.scalars(
            select(EvaluationSnapshot)
            .where(EvaluationSnapshot.evaluation_run_id == run.id)
            .order_by(EvaluationSnapshot.symbol, EvaluationSnapshot.instrument, EvaluationSnapshot.id)
        ).all()
        result = {"mode": mode, "evaluation_run_id": str(run.id), "coverage": run.coverage_json}
        if mode == "shadow":
            result["shadow_parity"] = self._shadow_parity(version, manifest, rows)
            return result

        held = {(item["market"], item["symbol"], item["instrument"], item["side"]) for item in portfolio["positions"]}
        candidates = [
            row
            for row in rows
            if row.status == CANDIDATE_STATUS
            and (
                _intent_action(row) == "exit"
                or (row.market, row.symbol, row.instrument, row.recommendation_json.get("side")) not in held
            )
        ]
        if not candidates:
            result["decision_id"] = None
            result["decision_skipped"] = "no_candidates"
            return result
        actions = {_intent_action(row) for row in candidates}
        if len(actions) > 1:
            # One decision carries one final_action; fail closed rather than split or guess.
            result["decision_id"] = None
            result["decision_skipped"] = "mixed_actions"
            return result
        final_action = actions.pop()
        intents = [{"evaluation_snapshot_id": str(row.id), **row.recommendation_json["intent"]} for row in candidates]
        decision = DecisionService(self.session).create_decision(
            run.id,
            principal=principal,
            account_scope=policy.account_scope,
            original_json={
                "policy_id": policy.policy_id,
                "policy_sha256": content_sha256(version.policy_json),
                "selected_evaluation_ids": [str(row.id) for row in candidates],
                "final_action": final_action,
                "order_intents": intents,
            },
            portfolio_snapshot_json=portfolio,
            risk_snapshot_json=decision_risk_snapshot(policy, intents),
        )
        result["decision_id"] = str(decision.id)
        result["decision_status"] = decision.status
        return result


def _intent_action(row):
    intent = row.recommendation_json.get("intent") if isinstance(row.recommendation_json, dict) else None
    return intent.get("action") if isinstance(intent, dict) else None


def _uuid(value, field):
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be a UUID") from error
