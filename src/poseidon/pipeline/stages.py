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
from poseidon.pipeline.plan import V1_STAGE_PLAN
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
    def waiting(cls, until: datetime, reason: str | None = None) -> StageResult:
        return cls(Verdict.PASS, reason, None, {}, until)


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

    def input_payload(self, ctx: StageContext) -> dict:
        return {
            "stage": self.name,
            "config_sha256": ctx.config_sha256,
            "upstream": {name: ctx.upstream[name].artifact_sha256 for name in self.requires},
        }

    def run(self, ctx: StageContext) -> StageResult:  # pragma: no cover - interface
        raise NotImplementedError


class SpecGenerateStage(BaseStage):
    """Step 1: pass a hand-written spec through. LLM generation lands in Step 4."""

    name = "spec_generate"

    def input_payload(self, ctx):
        return {
            "stage": self.name,
            "config_sha256": ctx.config_sha256,
            "idea_text": ctx.idea_text,
            "input_spec_json": ctx.input_spec_json,
        }

    def run(self, ctx):
        if ctx.input_spec_json is None:
            return StageResult.unsupported("llm generation not implemented")
        spec = ctx.input_spec_json
        return StageResult.passed(
            {"origin": "input_spec_json", "spec": spec, "spec_sha256": content_sha256(spec)},
        )


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

    def __init__(self, data_source=None):
        self.data_source = data_source

    def bars(self, ctx, part: str = "train"):
        from poseidon.pipeline import artifacts
        from poseidon.pipeline.data import DataSourceError, OhlcvCache, load_bars

        root = ctx.artifacts_root if ctx.artifacts_root is not None else artifacts.artifacts_root()
        try:
            return load_bars(ctx.stage, ctx.config, part, OhlcvCache(Path(root), self.data_source))
        except DataSourceError as error:
            raise TransientStageError(str(error)) from error

    @staticmethod
    def strategy_builder(ctx, template: dict):
        from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory

        return DslV2StrategyFactory(spec=template, bar_label=ctx.config["backtest"]["bar_label"])


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
            neighbours, settled, factory = [], True, None
        else:
            bars = self.bars(ctx, "train")
            factory = self.strategy_builder(ctx, template)
            runner = opt.TrialRunner(
                build=factory.build,
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
            neighbours, settled = opt.complete_neighbourhood(runner, grid, int(settings["neighbour_steps"]))
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
                    source: sum(1 for r in results if r.source == source) for source in ("base", "search", "neighbour")
                },
                "n_trials_by_state": dict(sorted(by_state.items())),
                "neighbour_steps": int(settings["neighbour_steps"]),
                "neighbours": neighbours,
                "neighbourhood_settled": settled,
                "trials_file": "trials.json",
            }
        )
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
        best_spec = factory.to_config_dict(factory.build(best.params)) if grid.names else validated["spec"]
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


class NotImplementedStage(BaseStage):
    def run(self, ctx):
        return StageResult.unsupported("not implemented")


def _not_implemented(stage_name: str, required: tuple[str, ...]) -> NotImplementedStage:
    stage_class = type(
        f"{stage_name.title().replace('_', '')}Stage",
        (NotImplementedStage,),
        {"name": stage_name, "requires": required},
    )
    return stage_class()


_GATES = ("wfe", "holdout", "cost_stress", "dsr", "param_stability", "regime_split")


def build_default_registry(*, data_source=None) -> dict[str, Stage]:
    """``data_source`` overrides the Thalassa reader (tests pass synthetic bars)."""

    registry: dict[str, Stage] = {
        "spec_generate": SpecGenerateStage(),
        "spec_validate": SpecValidateStage(),
        "backtest": BacktestStage(data_source),
        "optimize": OptimizeStage(data_source),
    }
    for gate in _GATES:
        registry[gate] = _not_implemented(gate, ("optimize",))
    registry["shadow_paper"] = _not_implemented("shadow_paper", _GATES)
    registry["awaiting_release"] = _not_implemented("awaiting_release", ("shadow_paper",))
    if tuple(registry) != V1_STAGE_PLAN:
        raise RuntimeError("stage registry drifted from the v1 stage plan")
    return registry


STAGE_REGISTRY: dict[str, Stage] = build_default_registry()
