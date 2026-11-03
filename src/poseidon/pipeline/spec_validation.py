"""Pluggable strategy-spec validators.

``spec_validate`` looks its validator up by the run's frozen
``config_json["spec_validator"]``. ``dsl_v1`` (the Step-1 ``RuleConfig`` +
``CONDITION_REGISTRY`` check) stays registered for runs frozen before Step 2;
new runs freeze ``dsl_v2``, the strict DSL v2 parser. ``spec_v3`` (CONFIG_VERSION 6) is
registered for spec v3 runs, which nothing creates before step 4.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

V1_ACTIONS = frozenset({"long", "short", "close"})
V1_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d"})
V1_MAX_DEPTH = 10  # mirrors evaluate_condition(max_depth=10)
COMBINATORS = ("all", "any", "none", "vote")
# Registered in CONDITION_REGISTRY but not executable inside the pipeline.
V1_PIPELINE_UNAVAILABLE = {"ml_prediction": "needs qlib_prediction features, which the pipeline does not produce"}


class SpecValidation(NamedTuple):
    ok: bool
    errors: tuple[str, ...]
    unsupported: tuple[str, ...]


class SpecValidator(Protocol):
    name: str

    def validate_spec(self, spec: dict) -> SpecValidation: ...


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class DslV1Validator:
    """Validate against today's ``RuleConfig`` and ``CONDITION_REGISTRY``."""

    name = "dsl_v1"

    def __init__(self, *, market: str = "tw_futures", symbol: str = "TX"):
        self.market = market
        self.symbol = symbol

    def validate_spec(self, spec: dict) -> SpecValidation:
        from pydantic import ValidationError

        from poseidon.strategies.dsl.conditions import CONDITION_REGISTRY
        from poseidon.strategies.dsl.schema import RuleConfig

        errors: list[str] = []
        unsupported: list[str] = []
        if not isinstance(spec, dict):
            return SpecValidation(False, ("spec must be a JSON object",), ())
        try:
            config = RuleConfig.model_validate(spec)
        except ValidationError as error:
            for item in error.errors():
                location = ".".join(str(part) for part in item["loc"]) or "spec"
                errors.append(f"{location}: {item['msg']}")
            return SpecValidation(False, tuple(errors), ())
        if config.market != self.market or config.symbol != self.symbol:
            errors.append(f"spec must target {self.market}/{self.symbol}, got {config.market}/{config.symbol}")
        if config.interval not in V1_INTERVALS:
            errors.append(f"interval {config.interval!r} is not one of {sorted(V1_INTERVALS)}")
        for index, rule in enumerate(config.rules):
            path = f"rules.{index}"
            if rule.action not in V1_ACTIONS:
                errors.append(f"{path}.action: must be one of {sorted(V1_ACTIONS)}")
            self._node(rule.condition, f"{path}.condition", 0, CONDITION_REGISTRY, errors, unsupported)
        unique_unsupported = tuple(dict.fromkeys(unsupported))
        return SpecValidation(not errors and not unique_unsupported, tuple(errors), unique_unsupported)

    def _node(self, node, path, depth, registry, errors, unsupported):
        if depth > V1_MAX_DEPTH:
            errors.append(f"{path}: condition nesting exceeds depth {V1_MAX_DEPTH}")
            return
        if not isinstance(node, dict):
            errors.append(f"{path}: condition must be an object")
            return
        combinators = [key for key in COMBINATORS if key in node]
        if combinators:
            if len(combinators) > 1 or "type" in node:
                errors.append(f"{path}: a node is either one combinator or one leaf")
                return
            key = combinators[0]
            if key == "vote":
                vote = node["vote"]
                children = vote.get("conditions") if isinstance(vote, dict) else None
                if not isinstance(children, list) or not children:
                    errors.append(f"{path}.vote.conditions: must be a non-empty list")
                    return
                min_votes = vote.get("min_votes", 4)
                if not _positive_int(min_votes) or min_votes > len(children):
                    errors.append(f"{path}.vote.min_votes: must be an integer in 1..{len(children)}")
            else:
                children = node[key]
                if not isinstance(children, list) or not children:
                    errors.append(f"{path}.{key}: must be a non-empty list")
                    return
            child_path = f"{path}.{key}" + (".conditions" if key == "vote" else "")
            for index, child in enumerate(children):
                self._node(child, f"{child_path}.{index}", depth + 1, registry, errors, unsupported)
            return
        condition_type = node.get("type")
        if not isinstance(condition_type, str) or not condition_type:
            errors.append(f"{path}: leaf needs a condition type")
            return
        if condition_type not in registry:
            unsupported.append(f"condition type {condition_type!r} is not in the DSL v1 registry")
            return
        if condition_type in V1_PIPELINE_UNAVAILABLE:
            unsupported.append(f"condition type {condition_type!r} {V1_PIPELINE_UNAVAILABLE[condition_type]}")
            return
        self._leaf(condition_type, node, path, errors)

    @staticmethod
    def _leaf(condition_type, node, path, errors):
        params = node.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"{path}.params: must be an object")
            return

        def need(test, field, message):
            if not test:
                errors.append(f"{path}.{field}: {message}")

        if condition_type in {"indicator_above", "indicator_below"}:
            need(isinstance(node.get("indicator"), str) and node["indicator"], "indicator", "required string")
            need(_number(node.get("threshold")), "threshold", "required finite number")
        elif condition_type == "price_crosses":
            need(isinstance(node.get("indicator"), str) and node["indicator"], "indicator", "required string")
            need(node.get("direction", "up") in {"up", "down"}, "direction", "must be up or down")
        elif condition_type == "indicator_crosses":
            need(_positive_int(params.get("fast")), "params.fast", "required positive integer")
            need(_positive_int(params.get("slow")), "params.slow", "required positive integer")
            need(node.get("direction", "up") in {"up", "down"}, "direction", "must be up or down")
        elif condition_type == "price_change_pct":
            need(_number(node.get("threshold")), "threshold", "required finite number")
        elif condition_type == "volume_spike":
            need(_positive_int(params.get("period", 20)), "params.period", "must be a positive integer")
            need(_number(params.get("multiplier", 2.0)), "params.multiplier", "must be a finite number")
        elif condition_type == "bollinger_width_percentile":
            need(_positive_int(params.get("period", 20)), "params.period", "must be a positive integer")
            need(_positive_int(params.get("lookback", 168)), "params.lookback", "must be a positive integer")
            need(_number(node.get("threshold", 0.85)), "threshold", "must be a finite number")
        elif condition_type == "indicator_comparison":
            for field in ("indicator_a", "indicator_b"):
                need(isinstance(node.get(field), str) and node[field], field, "required string")
            need(node.get("direction", "above") in {"above", "below"}, "direction", "must be above or below")
        elif condition_type in {"feature_above", "feature_below"}:
            need(isinstance(node.get("column"), str) and node["column"], "column", "required string")
            need(_number(node.get("threshold")), "threshold", "required finite number")


SPEC_VALIDATORS: dict[str, SpecValidator] = {}


def register_spec_validator(validator: SpecValidator) -> SpecValidator:
    SPEC_VALIDATORS[validator.name] = validator
    return validator


class DslV2Validator:
    """The strategy-pipeline validator: strict DSL v2 only (Step 2 onward).

    ``dsl_version`` missing or 1 is ``unsupported`` (the pipeline needs v2's
    sessions, exits and position awareness); a :class:`DslUnsupportedError`
    (e.g. ``htf``) is ``unsupported``; every other v2 validation error is an
    error (verdict ``fail``). :meth:`resolve` returns the concrete spec with
    each ``$param`` at its default plus the parameter grid for ``optimize``.
    """

    name = "dsl_v2"

    def __init__(self, *, market: str = "tw_futures", symbol: str = "TX"):
        self.market = market
        self.symbol = symbol

    def validate_spec(self, spec: dict) -> SpecValidation:
        from poseidon.strategies.dsl.v2 import DslUnsupportedError, DslValidationError, validate_v2

        if not isinstance(spec, dict):
            return SpecValidation(False, ("spec must be a JSON object",), ())
        version = spec.get("dsl_version", 1)
        if version == 1 and not isinstance(version, bool):
            return SpecValidation(False, (), (V2_REQUIRED,))
        try:
            validated = validate_v2(spec)
        except DslUnsupportedError as error:
            reasons = tuple(dict.fromkeys(f"{item['path']}: {item['msg']}" for item in error.errors))
            return SpecValidation(False, (), reasons)
        except DslValidationError as error:
            return SpecValidation(False, tuple(f"{item['path'] or 'spec'}: {item['msg']}" for item in error.errors), ())
        model = validated.model
        if model.market != self.market or model.symbol != self.symbol:
            message = f"spec must target {self.market}/{self.symbol}, got {model.market}/{model.symbol}"
            return SpecValidation(False, (message,), ())
        policy = param_count_errors(validated)
        if policy:
            return SpecValidation(False, tuple(f"{item['path']}: {item['msg']}" for item in policy), ())
        return SpecValidation(True, (), ())

    def resolve(self, spec: dict) -> dict:
        """``{"spec", "template", "params"}`` for a spec that passed :meth:`validate_spec`."""

        from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory

        factory = DslV2StrategyFactory(spec=spec)
        params = {
            name: {"type": ptype, "low": low, "high": high, "step": step, "default": factory.defaults[name]}
            for name, (low, high, ptype, step) in factory.param_bounds().items()
        }
        return {
            "spec": DslV2StrategyFactory.to_config_dict(factory.build()),
            "template": factory.template,
            "params": params,
        }


V2_REQUIRED = "pipeline requires dsl_version 2"

# Owner decision (Step 3b): every pipeline spec -- LLM output or hand-written --
# declares 1-4 ``$param``s, so optimize always searches and param_stability
# always has a neighbourhood. The DSL itself allows more (``MAX_PARAMS``).
PIPELINE_MIN_PARAMS = 1
PIPELINE_MAX_PARAMS = 4
PARAM_GUIDANCE = (
    "parameterise the values worth tuning, e.g. window minutes (opening-range minutes, session_elapsed "
    "bounds, max_hold_bars), stop/target distances (stop points or ATR multiple, target points or R multiple, "
    "stop offset) or thresholds (const levels, indicator periods, cross/compare offsets)"
)


def param_count_errors(validated) -> list[dict]:
    """Pipeline policy: ``PIPELINE_MIN_PARAMS``..``PIPELINE_MAX_PARAMS`` declared ``$param``s."""

    count = len(validated.params)
    if PIPELINE_MIN_PARAMS <= count <= PIPELINE_MAX_PARAMS:
        return []
    message = (
        f"the pipeline requires {PIPELINE_MIN_PARAMS}-{PIPELINE_MAX_PARAMS} tunable $params, "
        f"the spec declares {count}; {PARAM_GUIDANCE}"
    )
    return [{"path": "params", "msg": message, "type": "value_error"}]


class SpecV3Validator:
    """Spec v3 (CONFIG_VERSION 6, design 2026-10-24 §3.7); see :mod:`poseidon.pipeline.v3.validation`.

    ``config_json["spec_validator"] == "spec_v3"``. Imported lazily: ``v3.validation`` pulls in the DSL, the
    component registry and the cost models.
    """

    name = "spec_v3"

    def validate_spec(self, spec: dict) -> SpecValidation:
        from poseidon.pipeline.v3.validation import validate_spec_v3

        result = validate_spec_v3(spec)
        return SpecValidation(
            result.ok,
            tuple(issue.text() for issue in result.errors),
            tuple(issue.text() for issue in result.unsupported),
        )

    def resolve(self, spec: dict) -> dict:
        """``{"spec", "template", "params", "notes", "strategy_class"}`` for a spec that passed."""

        from poseidon.pipeline.v3.validation import validate_spec_v3

        result = validate_spec_v3(spec)
        if not result.ok:
            raise ValueError("resolve() needs a spec that passed validate_spec()")
        return {
            "spec": result.resolved,
            "template": result.template,
            "params": {
                name: {
                    "type": p.type,
                    "low": p.cast(p.low),
                    "high": p.cast(p.high),
                    "step": p.step,
                    "default": p.cast(p.default),
                }
                for name, p in sorted(result.params.items())
            },
            "notes": list(result.notes),
            "strategy_class": result.strategy_class,
        }


register_spec_validator(DslV1Validator())
register_spec_validator(DslV2Validator())
register_spec_validator(SpecV3Validator())


def get_spec_validator(name: str) -> SpecValidator:
    try:
        return SPEC_VALIDATORS[name]
    except KeyError:
        raise LookupError(f"spec validator {name!r} is not registered") from None
