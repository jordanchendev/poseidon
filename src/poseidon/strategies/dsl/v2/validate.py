"""DSL v2 validator: strict parse, $param substitution, cross-field checks.

Entry points:

- :func:`validate_v2` — validate a raw document (a template with ``params``
  or a fully resolved spec) and return a :class:`ValidatedV2`.
- :func:`dsl_v2_json_schema` — JSON Schema of the grammar, for LLM prompts.

Errors carry dotted paths into the raw document (``rules.0.condition.all.1.b``).
Anything that asks for a capability v2 does not have (``htf`` today) raises
:class:`DslUnsupportedError` so the pipeline can close the idea as
``unsupported`` instead of retrying it as a malformed spec.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from poseidon.strategies.dsl.v2.schema import (
    MAX_DEPTH,
    MAX_ENTRY_RULES,
    MAX_LEAVES_PER_RULE,
    MAX_PARAMS,
    PARAM_NAME_PATTERN,
    AllNode,
    AnyNode,
    AtrTrailing,
    ChannelOperand,
    CloseRule,
    CompareCondition,
    ConstOperand,
    CrossCondition,
    EntryRule,
    NoneNode,
    OpeningRangeOperand,
    ParamSpec,
    PointsTrailing,
    PriceOperand,
    RiskBlock,
    RMultipleTarget,
    StrategyV2,
)

UNSUPPORTED_KEYS = {"htf": "higher-timeframe conditions (htf) are not supported yet (deferred past DSL v2.1)"}
PARAM_REF_KEY = "$param"


class DslV2Error(ValueError):
    """Base class: ``errors`` is a list of ``{"path", "msg", "type"}`` dicts."""

    kind = "invalid"

    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        summary = "; ".join(f"{e['path'] or '<root>'}: {e['msg']}" for e in errors[:5])
        more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        super().__init__(f"DSL v2 {self.kind}: {summary}{more}")


class DslValidationError(DslV2Error):
    """The document is malformed or out of bounds (fixable by the author)."""

    kind = "invalid"


class DslUnsupportedError(DslV2Error):
    """The document needs a capability DSL v2 does not have (verdict ``unsupported``)."""

    kind = "unsupported"


@dataclass(frozen=True)
class ValidatedV2:
    """A validated v2 spec.

    ``model`` is the resolved strategy (every ``$param`` substituted).
    ``template`` is the raw document as given; ``params``/``values`` describe
    the tunable parameters and the values used for ``model``.
    """

    model: StrategyV2
    template: dict[str, Any]
    params: dict[str, ParamSpec] = field(default_factory=dict)
    values: dict[str, int | float] = field(default_factory=dict)


def _err(path: str, msg: str, kind: str = "value_error") -> dict[str, str]:
    return {"path": path, "msg": msg, "type": kind}


def _join(loc: tuple[Any, ...] | list[Any]) -> str:
    return ".".join(str(p) for p in loc if not (isinstance(p, str) and p.startswith("~")))


def _pydantic_errors(exc: ValidationError, prefix: tuple[Any, ...] = ()) -> list[dict[str, str]]:
    return [_err(_join(prefix + tuple(e["loc"])), e["msg"], e["type"]) for e in exc.errors()]


# --------------------------------------------------------------------------- raw-doc walks


def _walk_raw(node: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_raw(value, (*path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_raw(value, (*path, i))


def find_unsupported(doc: Any) -> list[dict[str, str]]:
    """Paths of keys asking for capabilities that v2 does not support."""
    errors = []
    for path, node in _walk_raw(doc):
        if isinstance(node, dict):
            for key in node:
                if key in UNSUPPORTED_KEYS:
                    errors.append(_err(_join((*path, key)), UNSUPPORTED_KEYS[key], "unsupported"))
    return errors


def _param_refs(doc: Any) -> tuple[list[tuple[tuple[Any, ...], str]], list[dict[str, str]]]:
    refs, errors = [], []
    for path, node in _walk_raw(doc):
        if isinstance(node, dict) and PARAM_REF_KEY in node:
            name = node[PARAM_REF_KEY]
            if len(node) != 1 or not isinstance(name, str):
                errors.append(_err(_join(path), 'a parameter reference must be exactly {"$param": "<name>"}'))
                continue
            refs.append((path, name))
    return refs, errors


def _substitute(doc: Any, values: dict[str, int | float]) -> Any:
    if isinstance(doc, dict):
        if PARAM_REF_KEY in doc and len(doc) == 1:
            return values[doc[PARAM_REF_KEY]]
        return {k: _substitute(v, values) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_substitute(v, values) for v in doc]
    return doc


_PARAMS_ADAPTER = TypeAdapter(
    Annotated[dict[Annotated[str, Field(pattern=PARAM_NAME_PATTERN)], ParamSpec], Field(max_length=MAX_PARAMS)]
)


# --------------------------------------------------------------------------- model walks


def walk_model(node: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, node
    if isinstance(node, BaseModel):
        for name in type(node).model_fields:
            yield from walk_model(getattr(node, name), (*path, name))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk_model(value, (*path, i))
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from walk_model(value, (*path, key))


_COMBINATORS = {AllNode: "all", AnyNode: "any", NoneNode: "none"}


def condition_children(node: Any) -> list[Any]:
    key = _COMBINATORS.get(type(node))
    return list(getattr(node, key)) if key else []


def condition_depth(node: Any) -> int:
    children = condition_children(node)
    return 1 + max((condition_depth(c) for c in children), default=0)


def condition_leaves(node: Any) -> int:
    children = condition_children(node)
    return sum(condition_leaves(c) for c in children) if children else 1


RISK_FIELDS = tuple(RiskBlock.model_fields)


def merge_risk(strategy_risk: RiskBlock | None, rule_risk: RiskBlock | None) -> RiskBlock:
    """Rule-level fields override the strategy-level block field by field."""
    merged = {}
    for name in RISK_FIELDS:
        rule_value = getattr(rule_risk, name) if rule_risk is not None else None
        base_value = getattr(strategy_risk, name) if strategy_risk is not None else None
        merged[name] = rule_value if rule_value is not None else base_value
    return RiskBlock.model_construct(**merged)


def _risk_source(spec: StrategyV2, rule_index: int, field_name: str) -> str:
    rule = spec.rules[rule_index]
    if isinstance(rule, EntryRule) and rule.risk is not None and getattr(rule.risk, field_name) is not None:
        return f"rules.{rule_index}.risk.{field_name}"
    return f"risk.{field_name}"


# Prices that an include-current channel bounds by construction: highest(high) >= every price of
# the bar, highest(close) >= close, ... (lowest mirrors it).
_CHANNEL_BOUNDS = {
    "high": {"highest": {"open", "high", "low", "close"}, "lowest": {"high"}},
    "low": {"highest": {"low"}, "lowest": {"open", "high", "low", "close"}},
    "close": {"highest": {"close"}, "lowest": {"close"}},
}
_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">="}


def _degenerate_channel_compare(node: CompareCondition) -> str | None:
    """Why ``price op channel`` (include-current, no b_mult/offset) is constant, or None."""
    if node.b_mult is not None or node.offset is not None:
        return None
    a, b, op = node.a, node.b, node.op
    if isinstance(a, ChannelOperand) and isinstance(b, PriceOperand):
        a, b, op = b, a, _FLIP[op]
    if not (isinstance(a, PriceOperand) and isinstance(b, ChannelOperand)) or b.exclude_current:
        return None
    if a.price not in _CHANNEL_BOUNDS[b.source][b.channel]:
        return None
    # highest: price <= channel always; lowest: price >= channel always.
    never, always = (">", "<=") if b.channel == "highest" else ("<", ">=")
    if op == never:
        return "is never true"
    if op == always:
        return "is always true"
    return None


def operand_errors(spec: StrategyV2) -> list[dict[str, str]]:
    """Per-operand / per-comparison checks (opening-range multiples, v2.1 channel and b_mult rules)."""
    errors: list[dict[str, str]] = []
    interval = spec.interval_minutes
    for path, node in walk_model(spec):
        if isinstance(node, OpeningRangeOperand) and node.minutes % interval != 0:
            errors.append(
                _err(_join((*path, "minutes")), f"opening range minutes must be a multiple of the {interval}m interval")
            )
        if isinstance(node, CrossCondition):
            for side in ("a", "b"):
                operand = getattr(node, side)
                if isinstance(operand, ChannelOperand) and not operand.exclude_current:
                    errors.append(
                        _err(
                            _join((*path, side, "exclude_current")),
                            "a channel in a cross must have exclude_current true: a channel that includes the "
                            "current bar moves with that bar, so the bar's own price can never break it",
                        )
                    )
        if (
            isinstance(node, CrossCondition | CompareCondition)
            and node.b_mult is not None
            and isinstance(node.b, ConstOperand)
        ):
            errors.append(_err(_join((*path, "b_mult")), "b_mult on a const b is redundant; scale the const"))
        if isinstance(node, CompareCondition):
            why = _degenerate_channel_compare(node)
            if why is not None:
                errors.append(
                    _err(
                        _join(path),
                        f"this compare {why}: a channel with exclude_current false contains the bar's own price; "
                        "use exclude_current true or a different op",
                    )
                )
    return errors


def semantic_errors(spec: StrategyV2) -> list[dict[str, str]]:
    """Checks that need the whole document."""
    errors: list[dict[str, str]] = operand_errors(spec)

    entries = [i for i, r in enumerate(spec.rules) if isinstance(r, EntryRule)]
    if not entries:
        errors.append(_err("rules", "at least one long/short entry rule is required"))
    if len(entries) > MAX_ENTRY_RULES:
        errors.append(_err("rules", f"at most {MAX_ENTRY_RULES} entry rules are allowed"))

    seen: dict[str, int] = {}
    for i, rule in enumerate(spec.rules):
        if rule.id in seen:
            errors.append(_err(f"rules.{i}.id", f"duplicate rule id {rule.id!r} (also rules.{seen[rule.id]})"))
        seen.setdefault(rule.id, i)
        depth = condition_depth(rule.condition)
        if depth > MAX_DEPTH:
            errors.append(_err(f"rules.{i}.condition", f"condition depth {depth} exceeds {MAX_DEPTH}"))
        leaves = condition_leaves(rule.condition)
        if leaves > MAX_LEAVES_PER_RULE:
            errors.append(_err(f"rules.{i}.condition", f"{leaves} leaf conditions exceed {MAX_LEAVES_PER_RULE}"))

    for i in entries:
        rule = spec.rules[i]
        risk = merge_risk(spec.risk, rule.risk)
        if isinstance(risk.take_profit, RMultipleTarget) and risk.stop_loss is None:
            errors.append(_err(_risk_source(spec, i, "take_profit"), "r_multiple take_profit needs a stop_loss"))
        trailing = risk.trailing
        if (
            isinstance(trailing, AtrTrailing | PointsTrailing)
            and trailing.activate_after_r is not None
            and risk.stop_loss is None
        ):
            errors.append(_err(_risk_source(spec, i, "trailing"), "trailing.activate_after_r needs a stop_loss"))

    expected_market = {"TX": "tw_futures", "MTX": "tw_futures_mtx"}[spec.symbol]
    if spec.market != expected_market:
        errors.append(_err("market", f"symbol {spec.symbol} trades on market {expected_market!r}"))
    return errors


def _parse_resolved(doc: dict[str, Any]) -> StrategyV2:
    try:
        spec = StrategyV2.model_validate(doc)
    except ValidationError as exc:
        raise DslValidationError(_pydantic_errors(exc)) from None
    errors = semantic_errors(spec)
    if errors:
        raise DslValidationError(errors)
    return spec


def validate_v2(doc: Any, *, param_values: dict[str, Any] | None = None) -> ValidatedV2:
    """Validate a v2 document.

    Args:
        doc: Raw JSON-like document with ``dsl_version: 2``. May declare
            ``params`` and reference them with ``{"$param": "<name>"}``.
        param_values: Values to substitute (e.g. an optimizer trial). When
            omitted, each parameter's ``default`` is used and every grid value
            of every parameter is checked against the fields that use it.

    Raises:
        DslUnsupportedError: the document asks for a capability v2 lacks.
        DslValidationError: anything else that is wrong, with paths.
    """
    if not isinstance(doc, dict):
        raise DslValidationError([_err("", "a DSL v2 document must be a JSON object")])
    unsupported = find_unsupported(doc)
    if unsupported:
        raise DslUnsupportedError(unsupported)

    template = copy.deepcopy(doc)
    body = {k: v for k, v in template.items() if k != "params"}

    params: dict[str, ParamSpec] = {}
    if "params" in template and template["params"] is not None:
        try:
            params = _PARAMS_ADAPTER.validate_python(template["params"])
        except ValidationError as exc:
            raise DslValidationError(_pydantic_errors(exc, ("params",))) from None

    refs, errors = _param_refs(body)
    used = set()
    for path, name in refs:
        if name not in params:
            errors.append(_err(_join(path), f"unknown parameter {name!r}"))
        used.add(name)
    for name in params:
        if name not in used:
            errors.append(_err(f"params.{name}", f"parameter {name!r} is declared but never used"))
    if errors:
        raise DslValidationError(errors)

    if param_values is None:
        values = {name: spec.cast(spec.default) for name, spec in params.items()}
    else:
        bad = [_err(f"params.{n}", f"unknown parameter {n!r}") for n in param_values if n not in params]
        for name, spec in params.items():
            if name not in param_values:
                bad.append(_err(f"params.{name}", f"missing value for parameter {name!r}"))
            elif not spec.contains(param_values[name]):
                bad.append(_err(f"params.{name}", f"value {param_values[name]!r} is not on the {name!r} grid"))
        if bad:
            raise DslValidationError(bad)
        values = {name: spec.cast(param_values[name]) for name, spec in params.items()}

    model = _parse_resolved(_substitute(body, values))

    if param_values is None and params:
        range_errors = []
        for name, spec in params.items():
            for value in spec.grid():
                if value == values[name]:
                    continue
                trial = {**values, name: spec.cast(value)}
                try:
                    _parse_resolved(_substitute(body, trial))
                except DslValidationError as exc:
                    first = exc.errors[0]
                    range_errors.append(
                        _err(
                            f"params.{name}",
                            f"value {value!r} is invalid at {first['path']}: {first['msg']}",
                            "param_range",
                        )
                    )
                    break
        if range_errors:
            raise DslValidationError(range_errors)

    return ValidatedV2(model=model, template=template, params=params, values=values)


def dsl_v2_json_schema() -> dict[str, Any]:
    """JSON Schema of a DSL v2 document (same models the validator uses)."""
    return StrategyV2.model_json_schema()


__all__ = [
    "CloseRule",
    "DslUnsupportedError",
    "DslV2Error",
    "DslValidationError",
    "EntryRule",
    "ValidatedV2",
    "dsl_v2_json_schema",
    "find_unsupported",
    "merge_risk",
    "semantic_errors",
    "validate_v2",
]
