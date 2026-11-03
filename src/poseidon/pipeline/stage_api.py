"""The stage interface: context, result, base classes and the shared stage helpers.

Every stage module imports from here, never from :mod:`poseidon.pipeline.stages` (which holds the concrete
spec / backtest / optimize stages) or :mod:`poseidon.pipeline.registry` (which imports them all).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Protocol

from poseidon.pipeline.plan import (
    config_version,
    requires_for,
)
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


@dataclass(frozen=True)
class IdeaOriginal:
    """The owner's own material behind a drafted ``idea_text`` (the draft is what the owner confirmed)."""

    text: str
    sha256: str
    drafted_by: str


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
    original: IdeaOriginal | None = None  # read by spec_faithfulness; a stage that reads it hashes original.sha256

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
