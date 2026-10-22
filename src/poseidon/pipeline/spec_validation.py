"""Pluggable strategy-spec validators.

``spec_validate`` looks its validator up by the run's frozen
``config_json["spec_validator"]``. Step 1 ships the current v1 ``RuleConfig`` +
``CONDITION_REGISTRY`` validator; the DSL v2 strict parser registers itself
under a new name later without touching the stage or the dispatcher.
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


register_spec_validator(DslV1Validator())


def get_spec_validator(name: str) -> SpecValidator:
    try:
        return SPEC_VALIDATORS[name]
    except KeyError:
        raise LookupError(f"spec validator {name!r} is not registered") from None
