"""The stage registry: every stage by name, v1 bodies and v3 bodies routed by run config version."""

from __future__ import annotations

from poseidon.pipeline import daily_stages, gate_stages, shadow_paper
from poseidon.pipeline.faithfulness import SpecFaithfulnessStage
from poseidon.pipeline.plan import (
    CANONICAL_ORDER,
    SPEC_V3_CONFIG_VERSION,
    V1_STAGE_PLAN,
    config_version,
    requires_for,
)
from poseidon.pipeline.stage_api import Stage, StageContext, StageResult
from poseidon.pipeline.stages import BacktestStage, OptimizeStage, SpecGenerateStage, SpecValidateStage
from poseidon.pipeline.v3.config import is_v3_run


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

    Building the registry instantiates every stage; deferring it keeps ``import poseidon.pipeline.registry`` cheap.
    """

    global _DEFAULT_REGISTRY
    if name == "STAGE_REGISTRY":
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = build_default_registry()
        return _DEFAULT_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
