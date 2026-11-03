"""Decision-loop producers: freeze market manifests and produce evaluation runs.

Every task is a no-op unless the configured execution mode is shadow or decision."""

import math
import uuid
from datetime import UTC

import pandas as pd

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.config import settings
from poseidon.core.database import SessionLocal
from poseidon.core.timeutil import as_utc, iso_time, timestamp
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks.loop import (
    LOOP_BAR_FIELDS,
    _decision_worker_principal,
    _loop_as_of,
    _loop_market_skip,
    _loop_mode_skip,
)

LOOP_OHLCV_LOOKBACK_BARS = 400


def _frozen_ohlcv_evidence(market: str, interval: str, symbol: str, as_of: str) -> dict | None:
    from urllib.parse import urlencode

    from poseidon.data.remote_repository import RemoteDataRepository
    from poseidon.decision_loop.producer import interval_delta

    cutoff = timestamp(as_of, "as_of")
    start = cutoff - interval_delta(interval) * LOOP_OHLCV_LOOKBACK_BARS
    params = {
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "start": start.isoformat(),
        "end": cutoff.isoformat(),
    }
    frame = RemoteDataRepository.from_settings().read_ohlcv(symbol, market, interval, start, cutoff)
    bars = []
    for index, row in frame.iterrows():
        values = [row.get(field) for field in LOOP_BAR_FIELDS]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            continue
        bar_time = pd.Timestamp(index)
        bar_time = bar_time.tz_localize("UTC") if bar_time.tzinfo is None else bar_time.tz_convert("UTC")
        if bar_time.to_pydatetime() > cutoff:
            continue
        bars.append(
            {
                "time": iso_time(bar_time.to_pydatetime(), "bar.time"),
                **{field: float(value) for field, value in zip(LOOP_BAR_FIELDS, values, strict=True)},
            }
        )
    if not bars:
        return None
    payload = {"symbol": symbol, "market": market, "interval": interval, "bars": bars}
    return {
        "id": f"ohlcv:{symbol}",
        "kind": "ohlcv",
        "source_uri": f"{settings.thalassa_base_url.rstrip('/')}/api/v1/ohlcv?{urlencode(params)}",
        "event_time": bars[-1]["time"],
        "available_at": bars[-1]["time"],
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def _stale_required_evidence(required: dict, evidence: list[dict], as_of: str) -> dict:
    """Return the required kinds whose newest evidence is older than the policy ``max_age_seconds``.

    Phase 101 D-04: on non-trading days the newest TW 1d bar is a day or more old, so the
    manifest must not be frozen; the beat task reports a skip instead of raising, because a
    raised loop task marks the soak day red. Missing evidence is not stale and still fails
    closed inside ``ManifestService.freeze``.
    """

    cutoff = timestamp(as_of, "as_of")
    newest: dict[str, object] = {}
    for item in evidence:
        event_time = timestamp(item["event_time"], "event_time")
        if item["kind"] not in newest or event_time > newest[item["kind"]]:
            newest[item["kind"]] = event_time
    stale = {}
    for kind, rule in required.items():
        latest = newest.get(kind)
        max_age = rule.get("max_age_seconds") if isinstance(rule, dict) else None
        if latest is None or not isinstance(max_age, (int, float)) or isinstance(max_age, bool):
            continue
        if (cutoff - latest).total_seconds() > max_age:
            stale[kind] = latest.isoformat().replace("+00:00", "Z")
    return stale


@celery_app.task(name="poseidon.workers.cpu_tasks.freeze_market_manifest", max_retries=0)
def freeze_market_manifest(market: str, interval: str, as_of: str | None = None) -> dict:
    """Freeze one OHLCV manifest for the approved universe of active decision-loop strategies."""
    skipped = _loop_mode_skip() or _loop_market_skip(market)
    if skipped:
        return skipped
    from pydantic import ValidationError as PydanticValidationError

    from poseidon.decision_loop.manifest import ManifestService
    from poseidon.decision_loop.policy import DecisionPolicy
    from poseidon.decision_loop.producer import ProducerService, executable_config

    as_of = _loop_as_of(as_of)
    scope = settings.decision_loop_approved_account_scope
    with SessionLocal() as session:
        versions = ProducerService(session).active_versions(market, interval, scope)
        try:
            policies = [DecisionPolicy.model_validate(version.policy_json) for version in versions]
        except PydanticValidationError as error:
            raise ValueError("active decision-loop strategy has an invalid decision policy") from error
        symbols = sorted({executable_config(version).symbol for version in versions})
    if not versions:
        return {"skipped": "no_active_strategies"}
    required = {kind: rule.model_dump() for kind, rule in policies[0].required_data.items()}
    if any(
        (policy.universe_id, canonical_json({kind: rule.model_dump() for kind, rule in policy.required_data.items()}))
        != (policies[0].universe_id, canonical_json(required))
        for policy in policies
    ):
        raise ValueError("active decision policies disagree on universe_id or required_data")
    unsupported = sorted(set(required) - {"ohlcv"})
    if unsupported:
        raise ValueError(f"manifest freeze supplies only ohlcv evidence; policy requires {unsupported}")
    evidence = [
        item for symbol in symbols if (item := _frozen_ohlcv_evidence(market, interval, symbol, as_of)) is not None
    ]
    stale = _stale_required_evidence(required, evidence, as_of)
    if stale:
        return {"skipped": "evidence_stale", "stale": stale, "as_of": as_of}
    request = {
        "market": market,
        "interval": interval,
        "as_of": as_of,
        "account_scope": scope,
        "universe_id": policies[0].universe_id,
        "capability_json": {"ohlcv": True},
        "required_data": required,
        "sources_json": [{"source": "thalassa", "base_url": settings.thalassa_base_url}],
        "evidence": evidence,
    }
    with SessionLocal() as session, session.begin():
        manifest = ManifestService(session).freeze(request)
        return {"manifest_id": str(manifest.id), "content_sha256": manifest.content_sha256, "as_of": as_of}


@celery_app.task(name="poseidon.workers.cpu_tasks.produce_decision_run", max_retries=0)
def produce_decision_run(strategy_version_id: str, manifest_id: str, decision_as_of: str | None = None) -> dict:
    """Turn one frozen manifest plus one strategy version into one immutable run (and decision)."""
    skipped = _loop_mode_skip()
    if skipped:
        return skipped
    version_id = uuid.UUID(strategy_version_id)
    persisted_manifest_id = uuid.UUID(manifest_id)
    from poseidon.decision_loop.producer import ProducerService

    principal = _decision_worker_principal()
    with SessionLocal() as session, session.begin():
        # A task from a manifest the in-flight guard already gave up on may still run late on a
        # backlogged worker; whichever task lands first settles the day for this version.
        # Accepted threat: check-then-insert, so two tasks running concurrently on different
        # cpu-worker forks can both see no run and both produce. Fix after 101-19: a partial unique
        # index on evaluation_runs (strategy_version_id, decision_as_of::date); Phase 101 opens no
        # migration.
        day_as_of = decision_as_of
        if day_as_of is None:
            from poseidon.models.data_manifest import DataManifest

            manifest = session.get(DataManifest, persisted_manifest_id)
            stored = manifest.as_of if manifest is not None else None
            if stored is not None and stored.tzinfo is None:
                stored = stored.replace(tzinfo=UTC)
            day_as_of = stored.isoformat() if stored is not None else None
        produced = _complete_runs_on_day(session, [version_id], day_as_of) if day_as_of else []
        if produced:
            return {"skipped": "already_produced", "run_id": produced[0], "manifest_id": manifest_id}
        return ProducerService(session).produce(
            version_id,
            persisted_manifest_id,
            decision_as_of=decision_as_of,
            principal=principal,
            mode=settings.decision_loop_execution_mode,
        )


# EvaluationRun statuses that settle a run. The model writes "pending" then "complete"; "failed"
# is the terminal-failure spelling. Any other (or future) status counts as still in progress, so an
# unknown status fails closed (skip) instead of letting a retry slot freeze a second manifest.
_RUN_TERMINAL_STATUSES = frozenset({"complete", "failed"})


def _runs_on_day(session, version_ids: list, as_of: str) -> list:
    """Evaluation runs (any status) for ``version_ids`` whose ``decision_as_of`` is on the UTC day of ``as_of``."""
    from sqlalchemy import select

    from poseidon.models.evaluation_run import EvaluationRun

    if not version_ids:
        return []
    day = timestamp(as_of, "as_of").date()
    runs = session.scalars(
        select(EvaluationRun)
        .where(EvaluationRun.strategy_version_id.in_(version_ids))
        .order_by(EvaluationRun.decision_as_of.desc(), EvaluationRun.id)
        .limit(200)
    ).all()
    matched = []
    for run in runs:
        stored = run.decision_as_of
        if stored.tzinfo is None:
            stored = stored.replace(tzinfo=UTC)
        if stored.astimezone(UTC).date() == day:
            matched.append(run)
    return matched


def _complete_runs_on_day(session, version_ids: list, as_of: str) -> list[str]:
    """Complete evaluation runs for ``version_ids`` whose ``decision_as_of`` falls on the UTC day of ``as_of``.

    Phase 101-17.2: the beat retry slots call the producer several times per session; the
    first successful freeze settles the day, later slots must not freeze a second manifest.
    """
    return sorted(str(run.id) for run in _runs_on_day(session, version_ids, as_of) if run.status == "complete")


def _in_progress_runs_on_day(session, version_ids: list, as_of: str) -> list[str]:
    """Same-day runs for ``version_ids`` that are not yet terminal (not complete and not failed)."""
    return sorted(
        str(run.id) for run in _runs_on_day(session, version_ids, as_of) if run.status not in _RUN_TERMINAL_STATUSES
    )


# Phase 101 in-flight guard (proposal, pending 101-19 / owner approval). EvaluationRun rows only
# land as "complete" (produce_decision_run commits inside one transaction), so a backlogged
# producer leaves a same-day manifest with no run at all. A manifest that young is treated as in
# flight; an older one without runs is treated as a failed produce and the day is re-produced.
# 1200 s = one full 900 s retry-slot gap + 300 s margin: after a 06:40Z freeze the 06:55Z slot
# always skips (the common worker-backlog length) and only 07:10Z onwards treats it as dead. A
# produce itself takes seconds, so no run after a whole slot gap is not a backlog any more.
PRODUCER_IN_FLIGHT_SECONDS = 1200


def _in_flight_manifest_on_day(session, market: str, interval: str, account_scope: str, version_ids: list, as_of: str):
    """Latest same-day manifest for this market/interval/scope with no run of ``version_ids``, if still in flight."""
    from sqlalchemy import select

    from poseidon.models.data_manifest import DataManifest
    from poseidon.models.evaluation_run import EvaluationRun

    if not version_ids:
        return None
    cutoff = timestamp(as_of, "as_of")

    manifests = session.scalars(
        select(DataManifest)
        .where(DataManifest.market == market, DataManifest.interval == interval)
        .order_by(DataManifest.as_of.desc(), DataManifest.created_at.desc(), DataManifest.id)
        .limit(50)
    ).all()
    latest = None
    for manifest in manifests:
        payload = manifest.payload_json if isinstance(manifest.payload_json, dict) else {}
        if as_utc(manifest.as_of).date() != cutoff.date() or payload.get("account_scope") != account_scope:
            continue
        if latest is None or as_utc(manifest.created_at) > as_utc(latest.created_at):
            latest = manifest
    if latest is None:
        return None
    has_run = session.scalars(
        select(EvaluationRun.id)
        .where(EvaluationRun.manifest_id == latest.id, EvaluationRun.strategy_version_id.in_(version_ids))
        .limit(1)
    ).first()
    if has_run is not None:
        return None
    if (cutoff - as_utc(latest.created_at)).total_seconds() > PRODUCER_IN_FLIGHT_SECONDS:
        return None
    return str(latest.id)


@celery_app.task(name="poseidon.workers.cpu_tasks.produce_decision_runs_for_market", max_retries=0)
def produce_decision_runs_for_market(market: str, interval: str, as_of: str | None = None) -> dict:
    """Beat entry: freeze once, then enqueue one UUID-only producer task per active version.

    Scheduled at 06:40Z and at three retry slots (06:55Z, 07:10Z, 07:25Z) inside the ohlcv
    freshness window; a slot that finds the day already produced (a complete run) or still in
    progress (a non-terminal run) skips without freezing. A day whose runs all failed is re-produced.
    """
    skipped = _loop_mode_skip() or _loop_market_skip(market)
    if skipped:
        return skipped
    from poseidon.decision_loop.producer import ProducerService

    _decision_worker_principal()
    as_of = _loop_as_of(as_of)
    with SessionLocal() as session:
        versions = ProducerService(session).active_versions(
            market, interval, settings.decision_loop_approved_account_scope
        )
        produced = _complete_runs_on_day(session, [version.id for version in versions], as_of)
        in_progress = [] if produced else _in_progress_runs_on_day(session, [version.id for version in versions], as_of)
        in_flight = (
            None
            if produced or in_progress
            else _in_flight_manifest_on_day(
                session,
                market,
                interval,
                settings.decision_loop_approved_account_scope,
                [version.id for version in versions],
                as_of,
            )
        )
        version_ids = [str(version.id) for version in versions]
    if produced:
        return {"skipped": "already_produced", "as_of": as_of, "runs": produced}
    if in_progress:
        # An earlier slot's run has not settled yet; a second freeze would mean two runs (and, in
        # decision mode, two decisions) for one session. Only all-failed days are re-produced.
        return {"skipped": "in_progress", "as_of": as_of, "runs": in_progress}
    if in_flight:
        # An earlier slot froze a manifest whose producer tasks have not landed a run yet.
        return {"skipped": "in_progress", "as_of": as_of, "manifest_id": in_flight, "runs": []}
    frozen = freeze_market_manifest.run(market, interval, as_of)
    if "manifest_id" not in frozen:
        return frozen
    for version_id in version_ids:
        produce_decision_run.delay(version_id, frozen["manifest_id"], as_of)
    return {"manifest_id": frozen["manifest_id"], "as_of": as_of, "enqueued": len(version_ids)}
