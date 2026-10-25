"""Run lifecycle commands (create / rerun / abort) and read models.

Every command is idempotent on ``Idempotency-Key``: the same key with the same
request replays the stored response; the same key with a different request is
a conflict. The caller owns the transaction (commit, then enqueue).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline.dispatcher import add_event, latest_attempts, run_config, select_next_stage
from poseidon.pipeline.models import PipelineEvent, PipelineRun, PipelineTrial, StageRun
from poseidon.pipeline.plan import (
    CONFIG_VERSION,
    ROUTED_CLASSES,
    SPEC_V3_CONFIG_VERSION,
    V3_RUN_CLASSES,
    RouteError,
    StagePlanError,
    default_config,
    is_routed,
    strategy_class_of,
    validate_stage_plan,
)
from poseidon.pipeline.stages import STAGE_REGISTRY
from poseidon.pipeline.state import (
    RUN_ABORTABLE,
    RUN_RERUNNABLE,
    STAGE_OPEN,
    AttemptTrigger,
    RunStatus,
    StageStatus,
    touch_run,
    transition_run,
    transition_stage,
    utcnow,
)

SPEC_JSON_MAX_BYTES = 64 * 1024
MARKET_TZ = ZoneInfo("Asia/Taipei")


class PipelineNotFound(LookupError):
    pass


class PipelineConflict(Exception):
    pass


class PipelineRequestInvalid(ValueError):
    pass


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def run_json(run: PipelineRun) -> dict:
    return {
        "id": str(run.id),
        "run_no": run.run_no,
        "family_id": str(run.family_id),
        "source": run.source,
        "idea_text": run.idea_text,
        "market": run.market,
        "symbol": run.symbol,
        "input_spec_json": run.input_spec_json,
        "config_json": run.config_json,
        "config_sha256": run.config_sha256,
        "created_by": run.created_by,
        "status": run.status,
        "current_stage": run.current_stage,
        "revision": run.revision,
        "reason": run.reason,
        "abort_requested_at": _iso(run.abort_requested_at),
        "created_at": _iso(run.created_at),
        "updated_at": _iso(run.updated_at),
        "finished_at": _iso(run.finished_at),
    }


def attempt_json(row: StageRun, *, full: bool = False) -> dict:
    data = {
        "id": str(row.id),
        "stage": row.stage,
        "seq": row.seq,
        "attempt": row.attempt,
        "trigger": row.trigger,
        "status": row.status,
        "verdict": row.verdict,
        "reason": row.reason,
        "code_version": row.code_version,
        "input_sha256": row.input_sha256,
        "artifact_sha256": row.artifact_sha256,
        "artifact_uri": row.artifact_uri,
        "lease_expires_at": _iso(row.lease_expires_at),
        "heartbeat_at": _iso(row.heartbeat_at),
        "wait_until": _iso(row.wait_until),
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "created_at": _iso(row.created_at),
    }
    if full:
        data["artifact_json"] = row.artifact_json
        data["error_json"] = row.error_json
    return data


def event_json(event: PipelineEvent) -> dict:
    return {
        "id": str(event.id),
        "stage_run_id": str(event.stage_run_id) if event.stage_run_id else None,
        "event_type": event.event_type,
        "actor_id": event.actor_id,
        "expected_revision": event.expected_revision,
        "payload_json": event.payload_json,
        "created_at": _iso(event.created_at),
    }


def trial_json(trial: PipelineTrial) -> dict:
    return {
        "id": str(trial.id),
        "run_id": str(trial.run_id),
        "stage_run_id": str(trial.stage_run_id),
        "trial_no": trial.trial_no,
        "params_json": trial.params_json,
        "params_sha256": trial.params_sha256,
        "state": trial.state,
        "sharpe": trial.sharpe,
        "n_obs": trial.n_obs,
        "skew": trial.skew,
        "kurtosis": trial.kurtosis,
        "window_sha256": trial.window_sha256,
        "created_at": _iso(trial.created_at),
    }


def family_trial_count(session: Session, family_id) -> int:
    """DSR trial count: every ledger row, including errored/aborted attempts (owner decision)."""

    return session.scalar(select(func.count()).select_from(PipelineTrial).where(PipelineTrial.family_id == family_id))


def _replay(event: PipelineEvent, request_sha256: str) -> dict:
    if event.request_sha256 != request_sha256:
        raise PipelineConflict("idempotency key was used for a different request")
    response = event.payload_json.get("response")
    if not isinstance(response, dict):
        raise PipelineConflict("stored idempotency response is invalid")
    return json.loads(canonical_json(response))


def _required_key(idempotency_key: str) -> str:
    if not isinstance(idempotency_key, str) or not idempotency_key.strip() or len(idempotency_key) > 256:
        raise PipelineRequestInvalid("Idempotency-Key must be 1..256 characters")
    return idempotency_key


def is_spec_v3(spec) -> bool:
    return isinstance(spec, dict) and spec.get("spec_version") == 3


def _default_refdata():
    from poseidon.pipeline.refdata.store import RefdataStore

    return RefdataStore()


class PipelineService:
    def __init__(self, session: Session, *, registry=None, clock=utcnow, refdata=None):
        self.session = session
        self.registry = registry if registry is not None else STAGE_REGISTRY
        self.clock = clock
        self.refdata = refdata or _default_refdata  # () -> refdata store; read only for spec v3 runs

    # ------------------------------------------------------------- commands

    def create(self, body: dict, *, actor_id: str, idempotency_key: str) -> tuple[dict, bool]:
        """Return ``(response, replayed)``. ``body`` is the JSON-mode request."""

        key = _required_key(idempotency_key)
        request_sha256 = content_sha256({"operation": "create", "body": body})
        replayed = self._replay_create(actor_id, key, request_sha256)
        if replayed is not None:
            return replayed, True

        spec = body.get("spec_json")
        if spec is not None and len(canonical_json(spec).encode("utf-8")) > SPEC_JSON_MAX_BYTES:
            raise PipelineRequestInvalid(f"spec_json exceeds {SPEC_JSON_MAX_BYTES} bytes")
        stage_plan = body.get("stage_plan")
        v3 = is_spec_v3(spec)
        if stage_plan is not None and spec is not None:
            try:
                version = SPEC_V3_CONFIG_VERSION if v3 else CONFIG_VERSION
                stage_plan = validate_stage_plan(stage_plan, self.registry, version=version)
            except StagePlanError as error:
                raise PipelineRequestInvalid(str(error)) from error
        now = self.clock()
        run_id = uuid.uuid4()
        family_id = run_id
        root = None
        if body.get("family_id") is not None:
            root = self.session.get(PipelineRun, uuid.UUID(str(body["family_id"])))
            if root is None:
                raise PipelineRequestInvalid("family_id does not name an existing pipeline run")
            family_id = root.family_id
        as_of = now.astimezone(MARKET_TZ).date()
        instrument = {}
        if spec is None:
            # Idea-only: spec_generate v3 decides the class (CONFIG_VERSION 7). A family member can only
            # become its family's class. market/symbol keep the column defaults (tw_futures/TX): every
            # instrument a spec can trade today is a TAIFEX index future; the spec names the contract.
            classes = (self._family_class(root),) if root is not None else ROUTED_CLASSES
            config = self._idea_config(stage_plan, as_of, classes)
        elif v3:
            config, instrument = self._v3_config(spec, stage_plan, as_of)
        else:
            config = default_config(stage_plan=stage_plan, as_of=as_of)
        if root is not None and spec is not None:
            self._check_family_class(root, config)
        run = PipelineRun(
            id=run_id,
            family_id=family_id,
            source=body["source"],
            idea_text=body["idea_text"],
            input_spec_json=spec,
            config_json=config,
            config_sha256=content_sha256(config),
            created_by=actor_id,
            create_idempotency_key=key,
            create_request_sha256=request_sha256,
            status=RunStatus.QUEUED.value,
            current_stage=config["stage_plan"][0],
            revision=1,
            created_at=now,
            updated_at=now,
            **instrument,
        )
        savepoint = self.session.begin_nested()
        try:
            self.session.add(run)
            self.session.flush()
        except IntegrityError:
            savepoint.rollback()
            replayed = self._replay_create(actor_id, key, request_sha256)
            if replayed is None:
                raise
            return replayed, True
        response = run_json(run)
        add_event(
            self.session,
            run,
            "run_created",
            actor_id,
            {"response": response},
            idempotency_key=key,
            request_sha256=request_sha256,
        )
        self.session.flush()
        return response, False

    def _family_class(self, root: PipelineRun) -> str:
        """The family's strategy class: its first run's, from spec_generate's route for an idea-only first run."""

        family_root = root if root.id == root.family_id else self.session.get(PipelineRun, root.family_id)
        if not is_routed(family_root.config_json):
            return strategy_class_of(family_root.config_json)
        attempts = self.session.query(StageRun).filter(StageRun.run_id == family_root.id).all()
        try:
            config = run_config(family_root.config_json, attempts)
        except RouteError as error:
            raise PipelineRequestInvalid(f"family {family_root.family_id} has no usable class: {error}") from error
        if is_routed(config):
            raise PipelineRequestInvalid(
                f"family {family_root.family_id}: its first run has no strategy class yet (spec_generate has not "
                "passed); wait for it or start a new family (omit family_id)"
            )
        return strategy_class_of(config)

    def _check_family_class(self, root: PipelineRun, config: dict) -> None:
        """A family's first run decides its strategy class; DSR never mixes return and TWD Sharpes (§7)."""

        family_class = self._family_class(root)
        run_class = strategy_class_of(config)
        if run_class != family_class:
            raise PipelineRequestInvalid(
                f"family {root.family_id} is {family_class} (its first run decides the class); a "
                f"{run_class} run cannot join it, DSR would mix daily-return and TWD trial Sharpes. "
                "Start a new family (omit family_id)."
            )

    def _idea_config(self, stage_plan, as_of, classes: tuple[str, ...]) -> dict:
        """CONFIG_VERSION 7 for an idea-only run (``v3.config.idea_run_config``).

        A client ``stage_plan`` (it must start with spec_generate) is a filter: each class runs the stages
        of its own plan that the request lists, validated per class here at creation.
        """

        from poseidon.pipeline.refdata.store import RefdataError
        from poseidon.pipeline.v3.brief import render_brief
        from poseidon.pipeline.v3.config import REFDATA_DATASETS, class_plan, idea_run_config

        plans = None
        if stage_plan is not None:
            if not isinstance(stage_plan, list) or not stage_plan or stage_plan[0] != "spec_generate":
                raise PipelineRequestInvalid("an idea-only run's stage_plan must start with spec_generate")
            unknown = sorted(set(stage_plan) - {name for c in classes for name in class_plan(c)})
            if unknown:
                raise PipelineRequestInvalid(f"unknown stage for an idea-only run: {unknown[0]}")
            plans = {}
            for strategy_class in classes:
                plan = [name for name in class_plan(strategy_class) if name in stage_plan]
                daily = strategy_class in V3_RUN_CLASSES
                try:
                    plans[strategy_class] = validate_stage_plan(
                        plan,
                        self.registry,
                        version=SPEC_V3_CONFIG_VERSION if daily else CONFIG_VERSION,
                        routed=daily,
                    )
                except StagePlanError as error:
                    raise PipelineRequestInvalid(f"{strategy_class}: {error}") from error
        refdata, refdata_error = None, None
        if any(name in V3_RUN_CLASSES for name in classes):
            try:
                store = self.refdata()
                refdata = {name: store.content_sha256(name) for name in REFDATA_DATASETS}
            except (RefdataError, OSError) as error:
                refdata_error = str(error)
        return idea_run_config(
            as_of=as_of,
            brief_text=render_brief(),
            refdata=refdata,
            classes=classes,
            plans=plans,
            refdata_error=refdata_error,
        )

    def _v3_config(self, spec: dict, stage_plan, as_of) -> tuple[dict, dict]:
        """CONFIG_VERSION 6 config and the run's market/symbol for a spec v3 document (design §8)."""

        from poseidon.pipeline.refdata.store import RefdataError
        from poseidon.pipeline.v3.brief import render_brief
        from poseidon.pipeline.v3.config import daily_run_config, refdata_datasets, series_first_days
        from poseidon.pipeline.v3.instruments import INSTRUMENTS
        from poseidon.pipeline.v3.validation import FeasibilityContext, validate_spec_v3

        try:
            store = self.refdata()
            refdata = {name: store.content_sha256(name) for name in refdata_datasets(spec)}
            series_since = series_first_days(store)
        except (RefdataError, OSError) as error:
            raise PipelineRequestInvalid(f"refdata is not available for spec v3 runs: {error}") from error
        # The window start is chosen here, once, and frozen (a series that starts late moves it; owner 2026-10-25).
        result = validate_spec_v3(
            spec, feasibility=FeasibilityContext(end=as_of, series_since=series_since, choose_start=True)
        )
        if not result.ok:
            problems = [issue.text() for issue in (*result.errors, *result.unsupported)]
            raise PipelineRequestInvalid(f"spec_json (spec v3) does not validate: {'; '.join(problems)[:1500]}")
        if result.strategy_class not in V3_RUN_CLASSES:
            raise PipelineRequestInvalid(
                f"spec v3 runs are daily only for now ({', '.join(V3_RUN_CLASSES)}); this spec is "
                f"{result.strategy_class}: intraday specs run as DSL v2 until DSL parity (step 7)"
            )
        config = daily_run_config(
            result.resolved,
            as_of=as_of,
            brief_text=render_brief(),
            refdata=refdata,
            stage_plan=stage_plan,
            window=result.window,
        )
        rows = [INSTRUMENTS[decl["symbol"]] for decl in spec["instruments"].values()]
        if len(rows) == 1:
            market = {"future": "tw_futures", "stock": "tw_stock"}.get(rows[0].kind)
            instrument = {"market": market, "symbol": rows[0].symbol}
        else:
            instrument = {"market": "multi", "symbol": "PORTFOLIO"}
        if instrument["market"] is None:
            raise PipelineRequestInvalid(f"no pipeline market for {rows[0].symbol} ({rows[0].kind})")
        return config, instrument

    def _replay_create(self, actor_id, key, request_sha256) -> dict | None:
        run = (
            self.session.query(PipelineRun)
            .filter(PipelineRun.created_by == actor_id, PipelineRun.create_idempotency_key == key)
            .one_or_none()
        )
        if run is None:
            return None
        event = self.session.query(PipelineEvent).filter_by(run_id=run.id, idempotency_key=key).one()
        return _replay(event, request_sha256)

    def _locked_command(self, operation, run_id, body, idempotency_key):
        key = _required_key(idempotency_key)
        run = (
            self.session.query(PipelineRun)
            .filter(PipelineRun.id == run_id)
            .with_for_update()
            .populate_existing()
            .one_or_none()
        )
        if run is None:
            raise PipelineNotFound("pipeline run does not exist")
        request_sha256 = content_sha256({"operation": operation, "run_id": str(run_id), "body": body})
        event = self.session.query(PipelineEvent).filter_by(run_id=run.id, idempotency_key=key).one_or_none()
        if event is not None:
            return run, key, request_sha256, _replay(event, request_sha256)
        if body["expected_revision"] != run.revision:
            raise PipelineConflict(f"expected revision {body['expected_revision']}, run is at {run.revision}")
        return run, key, request_sha256, None

    def rerun(self, run_id, body: dict, *, actor_id: str, idempotency_key: str) -> tuple[dict, bool]:
        run, key, request_sha256, replayed = self._locked_command("rerun", run_id, body, idempotency_key)
        if replayed is not None:
            return replayed, True
        if run.status == RunStatus.FAILED:
            raise PipelineConflict("a failed gate is final; open a new run in the same family to try a variant")
        if run.status not in RUN_RERUNNABLE:
            raise PipelineConflict(f"run is {run.status}; only error, aborted or unsupported runs can be rerun")
        attempts = self.session.query(StageRun).filter(StageRun.run_id == run.id).with_for_update().all()
        if any(row.status in STAGE_OPEN for row in attempts):
            raise PipelineConflict("run still has an open stage attempt")
        try:
            plan = run_config(run.config_json, attempts)["stage_plan"]
        except RouteError as error:
            raise PipelineConflict(f"run cannot continue: {error}") from error
        stage, _ = select_next_stage(plan, attempts)
        if stage is None:
            raise PipelineConflict("every stage already passed")
        from_stage = body.get("from_stage")
        if from_stage is not None and from_stage != stage:
            raise PipelineRequestInvalid(
                f"rerun restarts at the first stage without a pass ({stage}); upstream passes are reused"
            )
        now = self.clock()
        previous = latest_attempts(attempts).get(stage)
        self.session.add(
            StageRun(
                id=uuid.uuid4(),
                run_id=run.id,
                stage=stage,
                seq=plan.index(stage),
                attempt=(previous.attempt + 1) if previous is not None else 1,
                trigger=AttemptTrigger.RERUN.value,
                status=StageStatus.PENDING.value,
                created_at=now,
                updated_at=now,
            )
        )
        run.abort_requested_at = None
        transition_run(run, RunStatus.QUEUED, now=now, reason=body["reason"])
        run.current_stage = stage
        self.session.flush()
        response = run_json(run)
        add_event(
            self.session,
            run,
            "run_rerun",
            actor_id,
            {"response": response, "stage": stage, "reason": body["reason"]},
            idempotency_key=key,
            request_sha256=request_sha256,
        )
        self.session.flush()
        return response, False

    def abort(self, run_id, body: dict, *, actor_id: str, idempotency_key: str) -> tuple[dict, bool]:
        run, key, request_sha256, replayed = self._locked_command("abort", run_id, body, idempotency_key)
        if replayed is not None:
            return replayed, True
        if run.status not in RUN_ABORTABLE:
            raise PipelineConflict(f"run is {run.status}; only queued, running or waiting runs can be aborted")
        if run.abort_requested_at is not None:
            raise PipelineConflict("abort was already requested")
        now = self.clock()
        run.abort_requested_at = now
        attempts = self.session.query(StageRun).filter(StageRun.run_id == run.id).with_for_update().all()
        running = False
        for row in attempts:
            if row.status in {StageStatus.PENDING, StageStatus.WAITING}:
                transition_stage(row, StageStatus.ABORTED, now=now, reason=f"aborted: {body['reason']}")
            elif row.status == StageStatus.RUNNING:
                running = True  # its worker stops at the next heartbeat; the fence drops its result
        if running:
            touch_run(run, now=now)
            run.reason = body["reason"]
        else:
            transition_run(run, RunStatus.ABORTED, now=now, reason=body["reason"])
        self.session.flush()
        response = run_json(run)
        add_event(
            self.session,
            run,
            "run_abort_requested",
            actor_id,
            {"response": response, "reason": body["reason"], "running_attempt_fenced": running},
            idempotency_key=key,
            request_sha256=request_sha256,
        )
        self.session.flush()
        return response, False

    # ---------------------------------------------------------------- reads

    def get_run(self, run_id) -> PipelineRun:
        run = self.session.get(PipelineRun, run_id)
        if run is None:
            raise PipelineNotFound("pipeline run does not exist")
        return run

    def list_runs(self, *, status=None, family_id=None, limit=50, offset=0) -> list[PipelineRun]:
        query = select(PipelineRun)
        if status is not None:
            query = query.where(PipelineRun.status == status)
        if family_id is not None:
            query = query.where(PipelineRun.family_id == family_id)
        return list(self.session.scalars(query.order_by(PipelineRun.run_no.desc()).limit(limit).offset(offset)))

    def stages(self, run_id) -> list[dict]:
        run = self.get_run(run_id)
        attempts = list(
            self.session.scalars(
                select(StageRun).where(StageRun.run_id == run.id).order_by(StageRun.seq, StageRun.attempt)
            )
        )
        by_stage: dict[str, list[StageRun]] = {}
        for row in attempts:
            by_stage.setdefault(row.stage, []).append(row)
        try:
            plan = run_config(run.config_json, attempts)["stage_plan"]
        except RouteError:
            plan = run.config_json["stage_plan"]
        result = []
        for seq, stage in enumerate(plan):
            rows = by_stage.get(stage, [])
            result.append(
                {
                    "stage": stage,
                    "seq": seq,
                    "latest": attempt_json(rows[-1]) if rows else None,
                    "attempts": [attempt_json(row) for row in rows],
                }
            )
        return result

    def attempt(self, run_id, stage: str, attempt: int) -> StageRun:
        self.get_run(run_id)
        row = self.session.scalars(
            select(StageRun).where(StageRun.run_id == run_id, StageRun.stage == stage, StageRun.attempt == attempt)
        ).one_or_none()
        if row is None:
            raise PipelineNotFound("stage attempt does not exist")
        return row

    def events(self, run_id, *, limit=200, offset=0) -> list[PipelineEvent]:
        self.get_run(run_id)
        return list(
            self.session.scalars(
                select(PipelineEvent)
                .where(PipelineEvent.run_id == run_id)
                .order_by(PipelineEvent.created_at, PipelineEvent.id)
                .limit(limit)
                .offset(offset)
            )
        )

    def family_trials(self, family_id, *, limit=1000, offset=0) -> tuple[int, list[PipelineTrial]]:
        if (
            self.session.scalar(select(func.count()).select_from(PipelineRun).where(PipelineRun.family_id == family_id))
            == 0
        ):
            raise PipelineNotFound("pipeline family does not exist")
        trials = list(
            self.session.scalars(
                select(PipelineTrial)
                .where(PipelineTrial.family_id == family_id)
                .order_by(PipelineTrial.created_at, PipelineTrial.trial_no, PipelineTrial.id)
                .limit(limit)
                .offset(offset)
            )
        )
        return family_trial_count(self.session, family_id), trials
