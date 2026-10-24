"""Spec v3 document shape (design 2026-10-24 §3.1-3.2) and helpers shared by validation, rendering and mapping.

The envelope here only fixes the frame: slots, sleeves, capital, params.
Each ``{"component": ..., "params": {...}}`` is checked against its registry
entry by :mod:`poseidon.pipeline.v3.validation`, which reports paths into the
raw document so an LLM can repair it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from poseidon.strategies.dsl.v2.schema import NAME_PATTERN, PARAM_NAME_PATTERN, ParamSpec

SPEC_VERSION = 3
PARAM_REF_KEY = "$param"
ALIAS_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
SLEEVE_ID_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
RESERVED_ALIASES = frozenset({"self"})

# §12 #14: initial capital default and allowed range (TWD).
DEFAULT_INITIAL_CAPITAL_TWD = 1_000_000
MIN_INITIAL_CAPITAL_TWD = 300_000
MAX_INITIAL_CAPITAL_TWD = 20_000_000
MAX_SLEEVES = 4
MAX_INSTRUMENTS = 4
MAX_LEVERAGE_LINKS = 6
MAX_LIST_COMPONENTS = 4
# §12 #15: every spec declares 1-4 tunable $params (same policy as DSL v2 pipeline specs).
MIN_PARAMS = 1
MAX_PARAMS = 4


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ComponentRef(_Strict):
    component: Annotated[str, Field(min_length=1, max_length=64)]
    params: dict[str, Any] = Field(default_factory=dict)


class InstrumentDecl(_Strict):
    symbol: Annotated[str, Field(min_length=1, max_length=16)]
    roll: ComponentRef | None = None


class Capital(_Strict):
    initial_twd: Annotated[float, Field(ge=MIN_INITIAL_CAPITAL_TWD, le=MAX_INITIAL_CAPITAL_TWD)] = (
        DEFAULT_INITIAL_CAPITAL_TWD
    )


class Sleeve(_Strict):
    id: Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)]
    instrument: Annotated[str, Field(min_length=1, max_length=32)]
    direction: ComponentRef
    leverage: Annotated[list[ComponentRef], Field(max_length=MAX_LEVERAGE_LINKS)] = Field(default_factory=list)
    sizing: ComponentRef
    rebalance: ComponentRef
    exits: Annotated[list[ComponentRef], Field(max_length=MAX_LIST_COMPONENTS)] = Field(default_factory=list)
    hedges: Annotated[list[ComponentRef], Field(max_length=MAX_LIST_COMPONENTS)] = Field(default_factory=list)


class SpecV3(_Strict):
    """The frame of a spec v3 document; component params are validated per component."""

    spec_version: Literal[3]
    name: Annotated[str, Field(pattern=NAME_PATTERN)]
    description: Annotated[str, Field(max_length=500)] = ""
    frequency: Literal["1d", "1m", "5m", "15m"]
    capital: Capital = Field(default_factory=Capital)
    instruments: Annotated[
        dict[Annotated[str, Field(pattern=ALIAS_PATTERN)], InstrumentDecl],
        Field(min_length=1, max_length=MAX_INSTRUMENTS),
    ]
    sleeves: Annotated[list[Sleeve], Field(min_length=1, max_length=MAX_SLEEVES)]
    allocation: ComponentRef
    execution: ComponentRef
    params: dict[Annotated[str, Field(pattern=PARAM_NAME_PATTERN)], ParamSpec] = Field(default_factory=dict)


def component_slots(doc: dict) -> Iterator[tuple[tuple[Any, ...], str, dict]]:
    """``(path, slot, component_ref)`` for every component reference in a shape-valid document."""

    for alias, decl in doc["instruments"].items():
        if decl.get("roll") is not None:
            yield ("instruments", alias, "roll"), "roll", decl["roll"]
    for index, sleeve in enumerate(doc["sleeves"]):
        base = ("sleeves", index)
        yield (*base, "direction"), "direction", sleeve["direction"]
        for link, ref in enumerate(sleeve.get("leverage") or []):
            yield (*base, "leverage", link), "leverage", ref
        yield (*base, "sizing"), "sizing", sleeve["sizing"]
        yield (*base, "rebalance"), "rebalance", sleeve["rebalance"]
        for position, ref in enumerate(sleeve.get("exits") or []):
            yield (*base, "exits", position), "exit", ref
        for position, ref in enumerate(sleeve.get("hedges") or []):
            yield (*base, "hedges", position), "hedge", ref
    yield ("allocation",), "allocation", doc["allocation"]
    yield ("execution",), "execution", doc["execution"]


def join_path(parts) -> str:
    return ".".join(str(part) for part in parts)


def walk(node: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, (*path, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, (*path, index))


def is_param_ref(node: Any) -> bool:
    return isinstance(node, dict) and PARAM_REF_KEY in node


def substitute(node: Any, values: dict[str, int | float]) -> Any:
    """Replace every ``{"$param": name}`` with ``values[name]``."""

    if isinstance(node, dict):
        if PARAM_REF_KEY in node and len(node) == 1 and node[PARAM_REF_KEY] in values:
            return values[node[PARAM_REF_KEY]]
        return {key: substitute(value, values) for key, value in node.items()}
    if isinstance(node, list):
        return [substitute(value, values) for value in node]
    return node


def strategy_class(resolved: dict) -> str:
    """Frozen per run (§7): ``{intraday|daily}_{fixed|equity}`` from frequency and the sizing basis."""

    from poseidon.pipeline.v3.registry import REGISTRY

    frequency = "daily" if resolved["frequency"] == "1d" else "intraday"
    bases = {REGISTRY[sleeve["sizing"]["component"]].basis for sleeve in resolved["sleeves"]}
    if len(bases) != 1:
        raise ValueError("all sleeves of a spec must share one sizing basis")
    return f"{frequency}_{bases.pop()}"
