"""Stage interface, registry, and the Step-1 stage implementations."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Protocol

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline.plan import V1_STAGE_PLAN
from poseidon.pipeline.spec_validation import get_spec_validator
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
        report["spec"] = spec
        return StageResult.passed(report)


class BacktestStubStage(BaseStage):
    """Deterministic stand-in until Step 2 wires the real runner.

    Every number derives from ``input_sha256`` so the same spec + frozen config
    always yields the same report (and artifact hash).
    """

    name = "backtest"
    requires = ("spec_validate",)
    lease_seconds = 1800

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update(
            {
                "spec_sha256": (ctx.upstream["spec_validate"].report or {}).get("spec_sha256"),
                "data": ctx.config["data"],
                "holdout": ctx.config["holdout"],
                "seed": ctx.config["seed"],
            }
        )
        return payload

    def run(self, ctx):
        digest = bytes.fromhex(ctx.input_sha256)

        def unit(index: int) -> float:
            return int.from_bytes(digest[index : index + 2], "big") / 65535

        trades = 20 + int(unit(0) * 180)
        report = {
            "stub": True,
            "note": "deterministic stub keyed by input_sha256; the real backtest runner lands in Step 2",
            "input_sha256": ctx.input_sha256,
            "metrics": {
                "sharpe": round(unit(2) * 3 - 1, 4),
                "total_pnl_twd": round((unit(4) - 0.4) * 500_000, 2),
                "max_drawdown_twd": round(unit(6) * 200_000, 2),
                "trades": trades,
                "win_rate": round(0.3 + unit(8) * 0.4, 4),
            },
        }
        trade_rows = [{"n": n, "pnl_points": round((unit(10 + n % 20) - 0.5) * 80, 1)} for n in range(trades)]
        files = {"trades.json": canonical_json(trade_rows).encode("utf-8")}
        return StageResult.passed(report, files=files)


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


def build_default_registry() -> dict[str, Stage]:
    registry: dict[str, Stage] = {
        "spec_generate": SpecGenerateStage(),
        "spec_validate": SpecValidateStage(),
        "backtest": BacktestStubStage(),
        "optimize": _not_implemented("optimize", ("backtest",)),
    }
    for gate in _GATES:
        registry[gate] = _not_implemented(gate, ("optimize",))
    registry["shadow_paper"] = _not_implemented("shadow_paper", _GATES)
    registry["awaiting_release"] = _not_implemented("awaiting_release", ("shadow_paper",))
    if tuple(registry) != V1_STAGE_PLAN:
        raise RuntimeError("stage registry drifted from the v1 stage plan")
    return registry


STAGE_REGISTRY: dict[str, Stage] = build_default_registry()
