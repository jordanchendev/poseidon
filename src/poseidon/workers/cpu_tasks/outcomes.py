"""Decision-loop housekeeping: decision expiry, outcome manifests, and outcome labeling.

Every task returns {"skipped": mode} outside shadow/decision."""

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from poseidon.core.config import settings
from poseidon.core.database import SessionLocal
from poseidon.core.errors import ValidationError
from poseidon.core.timeutil import iso_time, timestamp
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks.loop import LOOP_BAR_FIELDS, _loop_as_of, _loop_market_skip, _loop_mode_skip

LOOP_LABEL_BATCH = 50


LOOP_OUTCOME_LOOKBACK_DAYS = 7


OUTCOME_CAPABILITIES = ("price", "benchmark", "fx", "calendar", "research")


@celery_app.task(name="poseidon.workers.cpu_tasks.expire_due_decisions", max_retries=0)
def expire_due_decisions(as_of: str | None = None) -> dict:
    skipped = _loop_mode_skip()
    if skipped:
        return skipped
    from poseidon.decision_loop.decisions import DecisionService

    as_of = _loop_as_of(as_of)
    with SessionLocal() as session, session.begin():
        expired = DecisionService(session).expire_due(as_of)
        return {"as_of": as_of, "expired": [str(decision.id) for decision in expired]}


def _is_outcome_manifest(manifest) -> bool:
    capabilities = manifest.capability_json if isinstance(manifest.capability_json, dict) else {}
    return all(capabilities.get(kind) is True for kind in OUTCOME_CAPABILITIES)


def _latest_outcome_as_of(session, now) -> str | None:
    from sqlalchemy import select

    from poseidon.models.data_manifest import DataManifest

    manifests = session.scalars(
        select(DataManifest)
        .where(DataManifest.as_of <= now, DataManifest.as_of >= now - timedelta(days=LOOP_OUTCOME_LOOKBACK_DAYS))
        .order_by(DataManifest.as_of.desc(), DataManifest.id)
    ).all()
    latest = next((manifest for manifest in manifests if _is_outcome_manifest(manifest)), None)
    return None if latest is None else iso_time(latest.as_of, "as_of")


def _mature_outcome_batches(session, as_of: str, label_definition_version: str):
    """Group unlabeled snapshots with a mature horizon by their frozen outcome manifest."""
    from sqlalchemy import select

    from poseidon.decision_loop.outcomes import _horizon_time, _manifest_facts, validate_outcome_label_contract
    from poseidon.models.data_manifest import DataManifest
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.evaluation_run import EvaluationRun
    from poseidon.models.evaluation_snapshot import EvaluationSnapshot
    from poseidon.models.outcome import OutcomeLabelContract, OutcomeRecord

    contracts = session.scalars(
        select(OutcomeLabelContract).where(OutcomeLabelContract.version == label_definition_version)
    ).all()
    if len(contracts) != 1:
        raise ValueError(f"label contract {label_definition_version!r} must resolve exactly once")
    contract_row = contracts[0]
    contract = validate_outcome_label_contract(contract_row)
    cutoff = timestamp(as_of, "as_of")
    by_identity = {}
    for manifest in session.scalars(
        select(DataManifest).where(DataManifest.as_of == cutoff).order_by(DataManifest.id)
    ).all():
        if not _is_outcome_manifest(manifest):
            continue
        try:
            _payload, by_id, sessions, _counterfactuals = _manifest_facts(manifest, contract, cutoff)
        except ValidationError:
            continue
        prices = [item for item in by_id.values() if item.get("kind") == "price"]
        if len(prices) != 1 or not isinstance(prices[0].get("payload"), dict):
            continue
        identity = (manifest.market, prices[0]["payload"].get("symbol"))
        by_identity.setdefault(identity, []).append((manifest, sessions))

    batches, unlabeled = {}, {}
    for (market, symbol), candidates in sorted(by_identity.items(), key=str):
        if len(candidates) != 1:
            unlabeled[f"{market}:{symbol}"] = "outcome_manifest_ambiguous"
            continue
        manifest, sessions = candidates[0]
        rows = session.execute(
            select(EvaluationSnapshot, EvaluationRun)
            .join(EvaluationRun, EvaluationRun.id == EvaluationSnapshot.evaluation_run_id)
            .where(
                EvaluationSnapshot.market == market,
                EvaluationSnapshot.symbol == symbol,
                EvaluationRun.status == "complete",
                EvaluationRun.decision_as_of < cutoff,
            )
            .order_by(EvaluationRun.decision_as_of, EvaluationSnapshot.id)
        ).all()
        for snapshot, run in rows:
            anchor = timestamp(
                run.decision_as_of.replace(tzinfo=UTC) if run.decision_as_of.tzinfo is None else run.decision_as_of,
                "decision_as_of",
            )
            labeled = {
                (kind, horizon_key)
                for kind, horizon_key in session.execute(
                    select(OutcomeRecord.kind, OutcomeRecord.horizon_key).where(
                        OutcomeRecord.evaluation_snapshot_id == snapshot.id,
                        OutcomeRecord.label_contract_id == contract_row.id,
                    )
                ).all()
            }
            traded = (
                session.scalar(select(DecisionRecord.id).where(DecisionRecord.evaluation_run_id == run.id).limit(1))
                is not None
            )
            try:
                # The service resolves every horizon, so an unresolvable calendar would fail the whole batch.
                maturities = {
                    (kind, horizon["key"]): _horizon_time(anchor, horizon, sessions)
                    for kind in ("signal", "trade", "research")
                    for horizon in contract["horizons"][kind]
                }
            except ValidationError:
                unlabeled[str(snapshot.id)] = "calendar_unresolved"
                continue
            due = [
                key
                for key, maturity in maturities.items()
                if maturity <= cutoff and (key[0] != "trade" or traded) and key not in labeled
            ]
            if due:
                batches.setdefault(manifest.id, []).append(snapshot.id)
    return batches, unlabeled


@celery_app.task(name="poseidon.workers.cpu_tasks.label_decision_outcomes", max_retries=0)
def label_decision_outcomes(as_of: str | None = None) -> dict:
    """Label only mature, not-yet-labeled snapshots against one frozen outcome manifest per symbol."""
    skipped = _loop_mode_skip()
    if skipped:
        return skipped
    version = settings.decision_loop_label_definition_version
    if not version:
        return {"skipped": "label_definition_version_unset"}
    from poseidon.decision_loop.outcomes import label_mature_outcomes

    with SessionLocal() as session:
        if as_of is None:
            as_of = _latest_outcome_as_of(session, datetime.now(UTC))
            if as_of is None:
                return {"skipped": "outcome_manifest_missing"}
        as_of = _loop_as_of(as_of)
        batches, unlabeled = _mature_outcome_batches(session, as_of, version)
    outcomes = 0
    for manifest_id, snapshot_ids in sorted(batches.items(), key=lambda item: str(item[0])):
        for start in range(0, len(snapshot_ids), LOOP_LABEL_BATCH):
            outcomes += len(
                label_mature_outcomes(
                    SessionLocal,
                    evaluation_snapshot_ids=snapshot_ids[start : start + LOOP_LABEL_BATCH],
                    as_of=as_of,
                    label_definition_version=version,
                    outcome_manifest_id=manifest_id,
                )
            )
    return {
        "as_of": as_of,
        "label_definition_version": version,
        "snapshots": sum(len(snapshot_ids) for snapshot_ids in batches.values()),
        "outcomes": outcomes,
        "unlabeled": unlabeled,
    }


# Phase 101 D-23.4: daily outcome manifests for the labeler above.
OUTCOME_PRICE_LOOKBACK_DAYS = 120


OUTCOME_MARKET = "tw_stock"


OUTCOME_INTERVAL = "1d"


OUTCOME_CALENDAR_PATH = "config/calendars/xtai_1d_sessions.json"


def _outcome_bars(symbol: str, as_of: str) -> list[dict]:
    """Finite tw_stock 1d closes at or before as_of, fetched like `_frozen_ohlcv_evidence`."""
    from poseidon.data.remote_repository import RemoteDataRepository

    cutoff = timestamp(as_of, "as_of")
    start = cutoff - timedelta(days=OUTCOME_PRICE_LOOKBACK_DAYS)
    frame = RemoteDataRepository.from_settings().read_ohlcv(symbol, OUTCOME_MARKET, OUTCOME_INTERVAL, start, cutoff)
    bars = []
    for index, row in frame.iterrows():
        values = [row.get(field) for field in LOOP_BAR_FIELDS]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            continue
        bar_time = pd.Timestamp(index)
        bar_time = bar_time.tz_localize("UTC") if bar_time.tzinfo is None else bar_time.tz_convert("UTC")
        if bar_time.to_pydatetime() > cutoff:
            continue
        bars.append({"time": iso_time(bar_time.to_pydatetime(), "bar.time"), "close": float(row.get("close"))})
    return bars


def _outcome_targets(session, scope: str, cutoff: datetime) -> dict:
    """Per active pilot symbol: the policy's account scope and universe plus recent decision anchors."""
    from pydantic import ValidationError as PydanticValidationError
    from sqlalchemy import select

    from poseidon.decision_loop.decisions import DecisionPolicy
    from poseidon.decision_loop.producer import ProducerService, executable_config
    from poseidon.models.evaluation_run import EvaluationRun

    earliest = cutoff - timedelta(days=OUTCOME_PRICE_LOOKBACK_DAYS)
    targets = {}
    for version in ProducerService(session).active_versions(OUTCOME_MARKET, OUTCOME_INTERVAL, scope):
        try:
            policy = DecisionPolicy.model_validate(version.policy_json)
        except PydanticValidationError as error:
            raise ValueError("active decision-loop strategy has an invalid decision policy") from error
        symbol = executable_config(version).symbol
        identity = (policy.account_scope, policy.universe_id)
        target = targets.setdefault(symbol, {"identity": identity, "anchors": set()})
        if target["identity"] != identity:
            raise ValueError(f"active decision policies for {symbol} disagree on account_scope or universe_id")
        for run in session.scalars(
            select(EvaluationRun).where(
                EvaluationRun.strategy_version_id == version.id,
                EvaluationRun.status == "complete",
            )
        ).all():
            decision_as_of = run.decision_as_of
            if decision_as_of.tzinfo is None:
                decision_as_of = decision_as_of.replace(tzinfo=UTC)
            anchor = timestamp(decision_as_of, "decision_as_of")
            if earliest <= anchor <= cutoff:
                target["anchors"].add(anchor)
    return targets


@celery_app.task(name="poseidon.workers.cpu_tasks.freeze_outcome_manifests", max_retries=0)
def freeze_outcome_manifests(as_of: str | None = None) -> dict:
    """Freeze one outcome manifest per active tw_stock pilot symbol for `label_decision_outcomes`.

    Benchmark, reporting currency, fx model, and calendar identity come only from the
    `OutcomeLabelContract` row of the configured label version; nothing is defaulted.
    """
    skipped = _loop_mode_skip() or _loop_market_skip(settings.decision_loop_approved_market)
    if skipped:
        return skipped
    if settings.decision_loop_approved_market != OUTCOME_MARKET:
        return {"skipped": "market_not_supported"}
    version = settings.decision_loop_label_definition_version
    if not version:
        return {"skipped": "label_definition_version_unset"}
    from sqlalchemy import select

    from poseidon.decision_loop.manifest import ManifestService, validate_manifest
    from poseidon.decision_loop.outcome_manifest import build_outcome_manifest_request, load_session_calendar
    from poseidon.decision_loop.outcomes import validate_outcome_label_contract
    from poseidon.models.outcome import OutcomeLabelContract

    as_of = _loop_as_of(as_of)
    cutoff = timestamp(as_of, "as_of")
    scope = settings.decision_loop_approved_account_scope
    calendar = load_session_calendar(OUTCOME_CALENDAR_PATH)
    with SessionLocal() as session:
        contracts = session.scalars(select(OutcomeLabelContract).where(OutcomeLabelContract.version == version)).all()
        if len(contracts) != 1:
            raise ValueError(f"label contract {version!r} must resolve exactly once")
        contract = validate_outcome_label_contract(contracts[0])
        targets = _outcome_targets(session, scope, cutoff)
    if not targets:
        return {"skipped": "no_active_strategies"}

    manifests, skipped_symbols = {}, {}
    benchmark_bars = _outcome_bars(contract["benchmark"]["symbol"], as_of)
    for symbol, target in sorted(targets.items()):
        if not benchmark_bars:
            skipped_symbols[symbol] = "benchmark_missing"
            continue
        bars = _outcome_bars(symbol, as_of)
        if not bars:
            skipped_symbols[symbol] = "price_missing"
            continue
        account_scope, universe_id = target["identity"]
        request = build_outcome_manifest_request(
            market=OUTCOME_MARKET,
            interval=OUTCOME_INTERVAL,
            as_of=as_of,
            account_scope=account_scope,
            universe_id=universe_id,
            symbol=symbol,
            contract=contract,
            calendar=calendar,
            bars=bars,
            benchmark_bars=benchmark_bars,
            anchors=sorted(target["anchors"]),
            source_base_url=settings.thalassa_base_url,
        )
        try:
            validate_manifest(request)
        except ValidationError as error:
            if str(error) != "required evidence is stale: price":
                raise
            skipped_symbols[symbol] = "price_stale"
            continue
        with SessionLocal() as session, session.begin():
            manifests[symbol] = str(ManifestService(session).freeze(request).id)
    return {"as_of": as_of, "manifests": manifests, "skipped_symbols": skipped_symbols}
