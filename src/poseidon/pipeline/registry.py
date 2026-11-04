"""The stage registry: every stage by name, v1 bodies and v3 bodies routed by run config version."""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import gate_stages, shadow_paper
from poseidon.pipeline.daily_gates import (
    DailyBenchmarkStage,
    DailyCostStressStage,
    DailyHoldoutStage,
    DailyRegimeSplitStage,
    DailyWfeStage,
    ExecutionTimingStage,
)
from poseidon.pipeline.daily_stages import ComponentDrift, DailyWindow, check_components
from poseidon.pipeline.daily_validate import (
    DailyBacktestStage,
    DailyDsrStage,
    DailyOptimizeStage,
    DailyParamStabilityStage,
    DailySpecValidateStage,
)
from poseidon.pipeline.faithfulness import SpecFaithfulnessStage
from poseidon.pipeline.plan import (
    CANONICAL_ORDER,
    SPEC_V3_CONFIG_VERSION,
    V1_STAGE_PLAN,
    V3_GATE_STAGES,
    config_version,
    requires_for,
)
from poseidon.pipeline.stage_api import Stage, StageContext, StageResult
from poseidon.pipeline.stages import BacktestStage, OptimizeStage, SpecGenerateStage, SpecValidateStage
from poseidon.pipeline.v3.compiler import NotCompilable
from poseidon.pipeline.v3.config import ENGINE, is_v3_run
from poseidon.pipeline.v3.instruments import contract_spec
from poseidon.pipeline.v3.runtime import Conventions


class EngineRouted:
    """One stage name, two bodies (design 2026-10-24 §8).

    A run frozen at CONFIG_VERSION <= 5 (no ``engine`` key) gets the ``legacy`` DSL v2 stage byte for
    byte -- same ``requires``, input payload and code path -- and a spec v3 run (CONFIG_VERSION 6,
    engine ``target_v1``) gets the ``daily`` stage of :mod:`poseidon.pipeline.daily_validate` / :mod:`poseidon.pipeline.daily_gates`.
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
        return self.daily if is_v3_run(config) else self.legacy

    def requires_at(self, version: int) -> tuple[str, ...]:
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

    legacy: dict[str, Stage] = {
        "spec_generate": SpecGenerateStage(refdata=refdata),
        "spec_validate": SpecValidateStage(),
        "backtest": BacktestStage(data_source, strategy_factory),
        "optimize": OptimizeStage(data_source, strategy_factory),
    }
    legacy.update(gate_stages.build_gate_stages(data_source=data_source, strategy_factory=strategy_factory))
    legacy.update(
        shadow_paper.build_paper_stages(data_source=data_source, strategy_factory=strategy_factory, refdata=refdata)
    )
    if tuple(legacy) != V1_STAGE_PLAN:
        raise RuntimeError("stage registry drifted from the v1 stage plan")
    daily = build_daily_stages(refdata=refdata)
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

    Building the registry instantiates every stage; deferring it keeps ``import poseidon.pipeline.registry`` cheap.
    """

    global _DEFAULT_REGISTRY
    if name == "STAGE_REGISTRY":
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = build_default_registry()
        return _DEFAULT_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compile_strategy(ctx):
    """``shadow_paper_daily.compile_strategy`` for a v3 run: optimize's selected spec -> CompiledStrategy.

    The engine starts on the train start (the paper re-simulates the whole history every wake-up and
    books only days from its anchor on); the protection basis is the train window ``[start,
    holdout_start)``. The fingerprint covers the selected spec, the registry version and the frozen
    component fingerprints, all verified against the live registry first. Aux series: only the indices
    the spec's signals read (``Plan.signal_indices``, e.g. ``index:taiex``), so the paper's cutoff
    waits for that day's index row and for nothing it does not use. The warm-up loads as the backtest's
    does (``compile_spec``): in sessions for a run frozen with ``data.warmup``, else in calendar days.
    Hedges run only for a run frozen with ``data.option_chain`` (``v3.config.option_chain_record``); an
    older hedged run stays unsupported, as it was frozen.
    """

    from poseidon.pipeline.shadow_paper_daily import CompiledStrategy, PaperInstrument, StrategyNotCompiled
    from poseidon.pipeline.v3.compiler import AUX_DATASETS, engine_config, hedge_policies, plan, strategy_factory
    from poseidon.pipeline.v3.validation import warmup_calendar_days

    try:
        components = check_components(ctx.config)
    except ComponentDrift as error:
        raise StrategyNotCompiled(str(error)) from error
    optimize = ctx.upstream["optimize"].report or {}
    best_spec = optimize.get("best_spec")
    if not isinstance(best_spec, dict) or best_spec.get("spec_version") != 3:
        raise StrategyNotCompiled("optimize selected no spec v3 point")
    try:
        spec_plan = plan(best_spec)
    except NotCompilable as error:
        raise StrategyNotCompiled(str(error)) from error
    if spec_plan.hedges and "option_chain" not in ctx.config.get("data", {}):
        raise StrategyNotCompiled(
            "daily shadow paper runs hedges only for runs frozen with data.option_chain; this run predates it "
            "(create a new run)"
        )
    window = DailyWindow.from_config(ctx.config)
    conventions = Conventions()
    warmup = spec_plan.warmup_bars + conventions.signal_lag
    record = ctx.config["data"].get("warmup")
    data_start = None if record else window.start - timedelta(days=warmup_calendar_days(warmup, "1d"))
    instruments = {
        alias: PaperInstrument(
            product=row.price_proxy or row.symbol,
            spec=contract_spec(row),
            roll_days_before=spec_plan.rolls.get(alias, 0),
        )
        for alias, row in spec_plan.instruments.items()
    }
    fingerprint = content_sha256(
        {
            "best_spec_sha256": content_sha256(best_spec),
            "registry_version": ctx.config["registry_version"],
            "components": components,
            "engine": ENGINE,
        }
    )
    return CompiledStrategy(
        factory=strategy_factory(spec_plan, conventions),
        instruments=instruments,
        engine=engine_config(spec_plan, start=window.start, end=None, conventions=conventions),
        data_start=data_start,
        train_end=window.holdout_start,
        fingerprint=fingerprint,
        aux={name: AUX_DATASETS[name] for name in spec_plan.signal_indices},
        warmup=(record, warmup) if record else None,
        calendar_rule=ctx.config["data"].get("trading_calendar"),
        hedges=tuple(hedge_policies(spec_plan)),
    )


def build_daily_stages(*, refdata=None) -> dict:
    """The v3 bodies, by stage name (``stages.build_default_registry`` routes v3 runs to them)."""

    from poseidon.pipeline.shadow_paper import ShadowPaperStage

    class DailyShadowPaperStage(ShadowPaperStage):
        requires: ClassVar[tuple[str, ...]] = ("backtest", "optimize", *V3_GATE_STAGES)

    stages = [
        DailySpecValidateStage(refdata=refdata),
        DailyBacktestStage(refdata=refdata),
        DailyOptimizeStage(refdata=refdata),
        DailyParamStabilityStage(),
        DailyDsrStage(),
        DailyCostStressStage(refdata=refdata),
        DailyRegimeSplitStage(refdata=refdata),
        DailyBenchmarkStage(refdata=refdata),
        ExecutionTimingStage(refdata=refdata),
        DailyWfeStage(refdata=refdata),
        DailyHoldoutStage(refdata=refdata),
        DailyShadowPaperStage(refdata=refdata, compile_strategy=compile_strategy),
    ]
    return {stage.name: stage for stage in stages}
