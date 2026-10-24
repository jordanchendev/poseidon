"""Stage interface, registry, and the implemented stages (spec, backtest, optimize)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Protocol

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline.plan import CANONICAL_ORDER, V1_STAGE_PLAN, config_version, requires_for
from poseidon.pipeline.spec_validation import V2_REQUIRED, get_spec_validator
from poseidon.pipeline.state import Verdict


class TransientStageError(Exception):
    """A failure worth one more automatic attempt (network blip, lock timeout...)."""


class StageStopped(Exception):
    """Raised by ``ctx.check_stop()`` once an abort or a lost lease was observed."""


@dataclass(frozen=True)
class UpstreamResult:
    stage: str
    attempt: int
    input_sha256: str
    artifact_sha256: str | None
    report: dict | None
    artifact_uri: str | None = None  # file:// directory holding the attempt's files
    files: tuple = ()  # the attempt's file manifest: ({"name", "sha256", "bytes"}, ...)


@dataclass
class StageContext:
    run_id: uuid.UUID
    family_id: uuid.UUID
    idea_text: str
    input_spec_json: dict | None
    config: dict
    config_sha256: str
    stage: str
    attempt: int
    upstream: Mapping[str, UpstreamResult]
    input_sha256: str = ""
    should_stop: Callable[[], bool] = lambda: False
    stage_run_id: uuid.UUID | None = None
    artifacts_root: Path | None = None
    trials: object | None = None  # a trial ledger (``poseidon.pipeline.trials``); None = in memory
    family: object | None = None  # a family view (``poseidon.pipeline.family``); None = this ledger only
    # Long-lived (waiting) stages: the dispatcher's clock at the claim, when the
    # stage's first attempt started, and the newest artifact any attempt of this
    # stage stored (a waiting attempt keeps its last one), to resume from.
    now: datetime | None = None
    stage_started_at: datetime | None = None
    resume: UpstreamResult | None = None

    def check_stop(self) -> None:
        if self.should_stop():
            raise StageStopped("stage stopped: abort requested or lease lost")


@dataclass(frozen=True)
class StageResult:
    verdict: Verdict
    reason: str | None = None
    report: dict | None = None
    files: Mapping[str, bytes] = field(default_factory=dict)
    wait_until: datetime | None = None
    hold: bool = False  # waiting with no wait_until: never auto-advances (awaiting_release)

    @classmethod
    def passed(cls, report: dict | None = None, *, reason: str | None = None, files=None) -> StageResult:
        return cls(Verdict.PASS, reason, report, files or {})

    @classmethod
    def failed(cls, reason: str, report: dict | None = None) -> StageResult:
        return cls(Verdict.FAIL, reason, report)

    @classmethod
    def unsupported(cls, reason: str, report: dict | None = None) -> StageResult:
        return cls(Verdict.UNSUPPORTED, reason, report)

    @classmethod
    def waiting(cls, until: datetime, reason: str | None = None, report=None, files=None) -> StageResult:
        """Come back at ``until``; ``report``/``files`` replace the attempt's artifact meanwhile."""

        return cls(Verdict.PASS, reason, report, files or {}, until)

    @classmethod
    def held(cls, reason: str, report: dict | None = None) -> StageResult:
        """Wait for a human: no ``wait_until``, the sweeper never advances it (abort still works)."""

        return cls(Verdict.PASS, reason, report, {}, None, True)

    @property
    def is_waiting(self) -> bool:
        return self.wait_until is not None or self.hold


class Stage(Protocol):
    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]]
    lease_seconds: ClassVar[int]
    max_auto_attempts: ClassVar[int]

    def input_payload(self, ctx: StageContext) -> dict: ...

    def run(self, ctx: StageContext) -> StageResult: ...


class BaseStage:
    name: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    lease_seconds: ClassVar[int] = 300
    max_auto_attempts: ClassVar[int] = 2

    def upstream_names(self, config) -> tuple[str, ...]:
        """``requires`` as of the run's frozen config version (stages added later are left out)."""

        return requires_for(self.requires, config_version(config))

    def input_payload(self, ctx: StageContext) -> dict:
        return {
            "stage": self.name,
            "config_sha256": ctx.config_sha256,
            "upstream": {name: ctx.upstream[name].artifact_sha256 for name in self.upstream_names(ctx.config)},
        }

    def run(self, ctx: StageContext) -> StageResult:  # pragma: no cover - interface
        raise NotImplementedError


class SpecGenerateStage(BaseStage):
    """Pass a hand-written spec through, or generate one from the idea with the LLM (Step 4)."""

    name = "spec_generate"

    def __init__(self, *, client_factory=None, pin_provider=None):
        self._client_factory = client_factory
        self._pin_provider = pin_provider

    def input_payload(self, ctx):
        payload = {
            "stage": self.name,
            "config_sha256": ctx.config_sha256,
            "idea_text": ctx.idea_text,
            "input_spec_json": ctx.input_spec_json,
        }
        if ctx.input_spec_json is None:
            from poseidon.pipeline.spec_llm import PROMPT_SHA256

            payload["prompt_sha256"] = PROMPT_SHA256
        return payload

    def run(self, ctx):
        if ctx.input_spec_json is not None:
            spec = ctx.input_spec_json
            return StageResult.passed(
                {"origin": "input_spec_json", "spec": spec, "spec_sha256": content_sha256(spec)},
            )
        from poseidon.core.config import settings
        from poseidon.pipeline.spec_llm import generate_spec

        ctx.check_stop()
        pin = self._pin_provider() if self._pin_provider else settings.pipeline_openrouter_key_label
        outcome = generate_spec(
            ctx.idea_text, ctx.config.get("spec_generate"), pin=pin, client_factory=self._client_factory
        )
        return StageResult(outcome.verdict, outcome.reason, outcome.report, outcome.files)


class SpecValidateStage(BaseStage):
    name = "spec_validate"
    requires = ("spec_generate",)

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload["validator"] = ctx.config["spec_validator"]
        return payload

    def run(self, ctx):
        spec = (ctx.upstream["spec_generate"].report or {}).get("spec")
        validator = get_spec_validator(ctx.config["spec_validator"])
        ok, errors, unsupported = validator.validate_spec(spec)
        report = {
            "validator": validator.name,
            "spec_sha256": content_sha256(spec),
            "errors": list(errors),
            "unsupported": list(unsupported),
        }
        if errors:
            return StageResult.failed(f"spec invalid: {len(errors)} error(s)", report)
        if unsupported:
            return StageResult.unsupported("; ".join(unsupported), report)
        if not ok:  # a validator that reports not-ok without saying why
            return StageResult.failed("spec rejected by validator", report)
        resolve = getattr(validator, "resolve", None)
        if resolve is None:
            report["spec"] = spec
            return StageResult.passed(report)
        # The concrete spec ($param defaults substituted) is what backtest runs;
        # the template and its grid are what optimize searches.
        resolved = resolve(spec)
        report.update(
            {
                "spec": resolved["spec"],
                "resolved_spec_sha256": content_sha256(resolved["spec"]),
                "template": resolved["template"],
                "params": resolved["params"],
            }
        )
        interval = (ctx.config.get("data") or {}).get("interval")
        if interval is not None and resolved["spec"]["interval"] != interval:
            reason = f"pipeline data window is {interval} bars, spec asks for {resolved['spec']['interval']}"
            return StageResult.unsupported(reason, report)
        return StageResult.passed(report)


STEP2_CONFIG_KEYS = ("data", "backtest", "optimize")


def _step2_config_missing(ctx) -> StageResult | None:
    missing = [key for key in STEP2_CONFIG_KEYS if key not in ctx.config]
    if missing or "holdout_start" not in ctx.config.get("data", {}):
        return StageResult.unsupported(
            "run config predates Step 2 (no frozen data/backtest/optimize); create a new run"
        )
    return None


class _DataStage(BaseStage):
    """A stage that reads the run's frozen window through the holdout guard."""

    def __init__(self, data_source=None, strategy_factory=None):
        self.data_source = data_source
        # ``(spec=..., bar_label=...) -> DslV2StrategyFactory``-like; tests inject
        # a test-only strategy (the random-entry control) through it.
        self.strategy_factory = strategy_factory

    def bars(self, ctx, part: str = "train"):
        from poseidon.pipeline import artifacts
        from poseidon.pipeline.data import DataSourceError, OhlcvCache, load_bars

        root = ctx.artifacts_root if ctx.artifacts_root is not None else artifacts.artifacts_root()
        try:
            return load_bars(ctx.stage, ctx.config, part, OhlcvCache(Path(root), self.data_source))
        except DataSourceError as error:
            raise TransientStageError(str(error)) from error

    def strategy_builder(self, ctx, template: dict):
        from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory

        factory = self.strategy_factory or DslV2StrategyFactory
        return factory(spec=template, bar_label=ctx.config["backtest"]["bar_label"])


class BacktestStage(_DataStage):
    """The concrete spec on the train part of the frozen window (never the holdout)."""

    name = "backtest"
    requires = ("spec_validate",)
    lease_seconds = 1800

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update(
            {
                "spec_sha256": (ctx.upstream["spec_validate"].report or {}).get("spec_sha256"),
                "data": ctx.config.get("data"),
                "backtest": ctx.config.get("backtest"),
                "part": "train",
            }
        )
        return payload

    def run(self, ctx):
        from poseidon.pipeline.backtesting import backtest_settings, run_backtest

        blocked = _step2_config_missing(ctx)
        if blocked is not None:
            return blocked
        upstream = ctx.upstream["spec_validate"].report or {}
        spec = upstream.get("spec")
        if not isinstance(spec, dict) or spec.get("dsl_version") != 2:
            return StageResult.unsupported(V2_REQUIRED)
        bars = self.bars(ctx, "train")
        strategy = self.strategy_builder(ctx, spec).build()
        outcome = run_backtest(strategy, bars.frame, ctx.config["backtest"], should_stop=ctx.should_stop)
        report = {
            "spec_sha256": content_sha256(spec),
            "data": bars.provenance(),
            "settings": backtest_settings(ctx.config["backtest"]),
            "metrics": outcome.metrics,
            "files": {"trades": "trades.csv", "daily_pnl": "daily_pnl.csv"},
        }
        return StageResult.passed(report, files=outcome.files())


class OptimizeStage(_DataStage):
    """TPE over the spec's ``$param`` grid on the train part; every trial goes to the ledger."""

    name = "optimize"
    requires = ("spec_validate", "backtest")
    lease_seconds = 1800

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update(
            {
                "spec_sha256": (ctx.upstream["spec_validate"].report or {}).get("spec_sha256"),
                "data": ctx.config.get("data"),
                "backtest": ctx.config.get("backtest"),
                "optimize": ctx.config.get("optimize"),
                "seed": ctx.config.get("seed"),
                "part": "train",
            }
        )
        return payload

    def run(self, ctx):
        from poseidon.pipeline import optimization as opt
        from poseidon.pipeline.backtesting import backtest_settings
        from poseidon.pipeline.trials import TRIAL_COMPLETE, MemoryTrialLedger
        from poseidon.strategies.dsl.v2 import validate_v2

        blocked = _step2_config_missing(ctx)
        if blocked is not None:
            return blocked
        validated = ctx.upstream["spec_validate"].report or {}
        backtest = ctx.upstream["backtest"].report or {}
        template = validated.get("template") or validated.get("spec")
        if not isinstance(template, dict) or template.get("dsl_version") != 2:
            return StageResult.unsupported(V2_REQUIRED)
        settings = ctx.config["optimize"]
        ledger = ctx.trials if ctx.trials is not None else MemoryTrialLedger()
        grid = opt.ParamGrid(validate_v2(template).params)
        started = time.monotonic()

        report = {
            "objective": settings["objective"],
            "objective_convention": "per-trading-day Sharpe of daily PnL (not annualised); see backtest report",
            "sampler": {"name": settings["sampler"], "seed": ctx.config["seed"]},
            "max_trials": settings["max_trials"],
            "param_grid": grid.describe(),
            "grid_size": grid.size() if grid.names else 1,
            "settings": backtest_settings(ctx.config["backtest"]),
        }
        files: dict[str, bytes] = {}
        if not grid.names:
            # Nothing to tune: the base backtest is the only trial, so N counts it.
            data = backtest.get("data") or {}
            metrics = backtest.get("metrics") or {}
            runner = opt.TrialRunner(
                build=None,
                bars=None,
                backtest_config=ctx.config["backtest"],
                ledger=ledger,
                window_sha256=data["window_sha256"],
                should_stop=ctx.should_stop,
                check_stop=ctx.check_stop,
            )
            state = TRIAL_COMPLETE if metrics.get("sharpe_daily") is not None else opt.TRIAL_PRUNED
            runner.record({}, "base", state, metrics)
            report.update({"data": data, "search": {"stopped_because": "no_params", "duplicates_skipped": 0}})
            neighbours, settled, factory, build, extension = [], True, None, None, None
        else:
            bars = self.bars(ctx, "train")
            factory = self.strategy_builder(ctx, template)
            # Off-grid values (edge extensions) are built from a widened copy of the template.
            build = opt.off_grid_builder(
                factory, template, grid, lambda effective: self.strategy_builder(ctx, effective)
            )
            runner = opt.TrialRunner(
                build=build,
                bars=bars.frame,
                backtest_config=ctx.config["backtest"],
                ledger=ledger,
                window_sha256=bars.window_sha256,
                should_stop=ctx.should_stop,
                check_stop=ctx.check_stop,
            )
            search = opt.tpe_search(
                runner,
                grid,
                factory.defaults,
                seed=int(ctx.config["seed"]),
                max_trials=int(settings["max_trials"]),
                max_consecutive_duplicates=int(settings["max_consecutive_duplicates"]),
                budget_seconds=settings.get("budget_seconds"),
                reserve_trials=2 * len(grid.names) * int(settings["neighbour_steps"]),
            )
            extension = opt.EdgeExtension(
                admissible=opt.hard_bound_check(template, grid),
                budget_seconds=settings.get("budget_seconds"),
                started=started,
                reserve_trials=2 * len(grid.names) * int(settings["neighbour_steps"]),
            )
            neighbours, settled = opt.complete_neighbourhood(runner, grid, int(settings["neighbour_steps"]), extension)
            report.update(
                {
                    "data": bars.provenance(),
                    "search": {
                        "stopped_because": search.stopped_because,
                        "duplicates_skipped": search.duplicates_skipped,
                        "asks": search.asks,
                        "budget_exhausted": search.budget_exhausted,
                    },
                }
            )

        results = runner.results
        by_state: dict[str, int] = {}
        for result in results:
            by_state[result.state] = by_state.get(result.state, 0) + 1
        report.update(
            {
                "n_trials": len(results),
                "n_trials_by_source": {
                    source: sum(1 for r in results if r.source == source)
                    for source in ("base", "search", "edge_extension", "neighbour")
                },
                "n_trials_by_state": dict(sorted(by_state.items())),
                "neighbour_steps": int(settings["neighbour_steps"]),
                "neighbours": neighbours,
                "neighbourhood_settled": settled,
                "trials_file": "trials.json",
            }
        )
        if grid.names:
            # The spec's declared grid stays as declared (``param_grid``); this is what was evaluated.
            report["edge_extension"] = extension.report()
            report["effective_range"] = grid.effective_range(results, extension.max_steps)
        files["trials.json"] = canonical_json([result.row() for result in results]).encode("utf-8")
        best = runner.best
        elapsed = time.monotonic() - started
        if best is None:
            report["best"] = None
            return StageResult(
                Verdict.FAIL,
                f"no trial produced a daily Sharpe ({len(results)} trials, {elapsed:.0f}s)",
                report,
                files,
            )
        best_spec = factory.to_config_dict(build(best.params)) if grid.names else validated["spec"]
        report["best"] = {
            "trial_no": best.trial_no,
            "params": best.params,
            "metrics": best.metrics,
        }
        report["best_spec"] = best_spec
        report["best_spec_sha256"] = content_sha256(best_spec)
        if best.outcome is not None:
            for name, data in best.outcome.files().items():
                files[f"best_{name}"] = data
        reason = (
            f"{len(results)} trials in {elapsed:.0f}s; best trial {best.trial_no} "
            f"sharpe_daily={best.sharpe:.4f} params={canonical_json(best.params)}"
        )
        return StageResult.passed(report, reason=reason, files=files)


class EngineRouted:
    """One stage name, two bodies (design 2026-10-24 §8).

    A run frozen at CONFIG_VERSION <= 5 (no ``engine`` key) gets the ``legacy`` DSL v2 stage byte for
    byte -- same ``requires``, input payload and code path -- and a spec v3 run (CONFIG_VERSION 6,
    engine ``target_v1``) gets the ``daily`` stage of :mod:`poseidon.pipeline.daily_stages`.
    """

    def __init__(self, legacy, daily):
        if legacy.name != daily.name:
            raise ValueError(f"cannot route {legacy.name} to {daily.name}")
        self.legacy = legacy
        self.daily = daily
        self.name = legacy.name
        self.requires = legacy.requires  # what plans of the legacy versions validate against
        self.lease_seconds = max(legacy.lease_seconds, daily.lease_seconds)
        self.max_auto_attempts = legacy.max_auto_attempts

    def body(self, config):
        from poseidon.pipeline.v3.config import is_v3_run

        return self.daily if is_v3_run(config) else self.legacy

    def requires_at(self, version: int) -> tuple[str, ...]:
        from poseidon.pipeline.plan import SPEC_V3_CONFIG_VERSION

        stage = self.daily if version >= SPEC_V3_CONFIG_VERSION else self.legacy
        return requires_for(stage.requires, version)

    def upstream_names(self, config) -> tuple[str, ...]:
        stage = self.body(config)
        names = getattr(stage, "upstream_names", None)
        return names(config) if names is not None else requires_for(stage.requires, config_version(config))

    def input_payload(self, ctx: StageContext) -> dict:
        return self.body(ctx.config).input_payload(ctx)

    def run(self, ctx: StageContext) -> StageResult:
        return self.body(ctx.config).run(ctx)


def build_default_registry(*, data_source=None, strategy_factory=None, refdata=None) -> dict[str, Stage]:
    """``data_source`` overrides the Thalassa reader and ``strategy_factory`` the DSL v2
    strategy factory (tests pass synthetic bars / a test-only strategy); ``refdata`` (() -> store)
    the refdata store of spec v3 daily runs."""

    from poseidon.pipeline import daily_stages, gate_stages, shadow_paper
    from poseidon.pipeline.faithfulness import SpecFaithfulnessStage

    legacy: dict[str, Stage] = {
        "spec_generate": SpecGenerateStage(),
        "spec_validate": SpecValidateStage(),
        "backtest": BacktestStage(data_source, strategy_factory),
        "optimize": OptimizeStage(data_source, strategy_factory),
    }
    legacy.update(gate_stages.build_gate_stages(data_source=data_source, strategy_factory=strategy_factory))
    legacy.update(shadow_paper.build_paper_stages(data_source=data_source, strategy_factory=strategy_factory))
    if tuple(legacy) != V1_STAGE_PLAN:
        raise RuntimeError("stage registry drifted from the v1 stage plan")
    daily = daily_stages.build_daily_stages(refdata=refdata)
    daily[SpecFaithfulnessStage.name] = SpecFaithfulnessStage()
    registry: dict[str, Stage] = {}
    for name in CANONICAL_ORDER:
        if name in legacy and name in daily:
            registry[name] = EngineRouted(legacy[name], daily[name])
        else:
            # spec_generate / awaiting_release / spec_faithfulness / execution_timing
            registry[name] = legacy.get(name) or daily[name]
    return registry


_DEFAULT_REGISTRY: dict[str, Stage] | None = None


def __getattr__(name: str):
    """``STAGE_REGISTRY`` is built on first use (PEP 562), not at import time.

    The registry imports the stage modules (gate_stages, shadow_paper), which
    import this module; building it lazily lets any of them be imported first.
    """

    global _DEFAULT_REGISTRY
    if name == "STAGE_REGISTRY":
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = build_default_registry()
        return _DEFAULT_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
