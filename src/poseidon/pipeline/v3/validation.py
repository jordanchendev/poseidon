"""``spec_validate`` for spec v3 (design 2026-10-24 §3.7), in order:

1. shape: the frame (:class:`~poseidon.pipeline.v3.spec.SpecV3`), every component
   known and in the right slot, every instrument known and tradable;
2. ``$param``: references well-formed, declared, used, on tunable fields, 1-4 of
   them; every grid value valid for every field that uses it;
3. component parameters (strict models with bounds; DSL v2 via ``validate_v2``);
4. cross rules: frequencies, chain order, duplicates, sleeve/instrument links,
   allocation, execution, DSL fast-path constraints, component checks;
5. data feasibility: every instrument has bars over ``[start - warm-up, end)``, and every
   index a signal reads (``index:taiex`` from 1999-01-05, ``index:taiex_tr`` from 2003-01-02)
   has rows there; proxy price intervals become ``notes``;
6. contract-count feasibility: no grid point sizes to 0 contracts or misses its
   target leverage by more than 25 %;
7. margin feasibility: initial margin at the largest leverage <= 80 % of equity;
8. policy: leverage cap 5x on equity-sized sleeves.

Stages 1-4 stop at the first stage with problems (later stages need their
output); 5-8 all run and accumulate. Every message names the path into the
raw document and what to change, so it can go back to the LLM for repair.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from poseidon.pipeline.v3 import components as _components  # noqa: F401  (registers the components)
from poseidon.pipeline.v3.dsl_bridge import DSL_MARKETS
from poseidon.pipeline.v3.instruments import (
    INSTRUMENTS,
    INTRADAY_FREQUENCIES,
    MAX_MARGIN_UTILISATION,
    REFERENCE_INDEX_LEVEL,
    SAME_INDEX_FAMILY,
    TX_INTRADAY_DATA_SINCE,
    Instrument,
    data_since,
    ordered_frequencies,
    signal_index,
)
from poseidon.pipeline.v3.registry import (
    PLANNED_COMPONENTS,
    REGISTRY,
    CheckContext,
    Component,
    LeverageComponent,
    SizingComponent,
    components_for,
)
from poseidon.pipeline.v3.spec import (
    DEFAULT_INITIAL_CAPITAL_TWD,
    MAX_PARAMS,
    MIN_PARAMS,
    PARAM_REF_KEY,
    RESERVED_ALIASES,
    SpecV3,
    component_slots,
    is_param_ref,
    join_path,
    strategy_class,
    substitute,
    walk,
)

LEVERAGE_CAP = 5.0  # §12 #7, on equity-sized sleeves
MAX_ROUNDING_ERROR = 0.25  # §3.7.5: |actual - target| / target
DAILY_WINDOW_START = date(2001, 1, 2)  # §12 #3
INTRADAY_WINDOW_START = TX_INTRADAY_DATA_SINCE  # plan.DATA_START
TRADING_DAYS_PER_YEAR = 245
GRID_ENUMERATION_LIMIT = 20_000  # beyond this, sizing feasibility checks the grid corners only
DSL_SYMBOLS = tuple(DSL_MARKETS)


@dataclass(frozen=True)
class SpecIssue:
    path: str
    msg: str
    type: str = "value_error"

    def text(self) -> str:
        return f"{self.path or 'spec'}: {self.msg}"

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "msg": self.msg, "type": self.type}


@dataclass(frozen=True)
class FeasibilityContext:
    """What the validator knows about the run's data. Missing parts fall back to checked-in defaults.

    ``prices`` maps a price-source symbol (TX for the TX family) to the price
    points to size at, e.g. ``(first_bar_close, max_close)`` of the window.
    """

    start: date | None = None
    end: date | None = None
    prices: Mapping[str, tuple[float, ...]] | None = None


@dataclass(frozen=True)
class V3Validation:
    ok: bool
    errors: tuple[SpecIssue, ...] = ()
    unsupported: tuple[SpecIssue, ...] = ()
    notes: tuple[str, ...] = ()
    template: dict | None = None  # the raw document as given
    resolved: dict | None = None  # every $param at its default
    params: dict = field(default_factory=dict)  # name -> ParamSpec
    strategy_class: str | None = None


class _Issues:
    def __init__(self):
        self.errors: list[SpecIssue] = []
        self.unsupported: list[SpecIssue] = []

    def error(self, path, msg, kind="value_error"):
        self.errors.append(SpecIssue(path if isinstance(path, str) else join_path(path), msg, kind))

    def unsupported_(self, path, msg):
        self.unsupported.append(SpecIssue(path if isinstance(path, str) else join_path(path), msg, "unsupported"))

    def any(self) -> bool:
        return bool(self.errors or self.unsupported)


def _pydantic_issues(issues: _Issues, error: ValidationError, prefix: tuple) -> None:
    for item in error.errors():
        issues.error((*prefix, *item["loc"]), item["msg"], item["type"])


def _choices(slot: str) -> str:
    return ", ".join(component.name for component in components_for(slot))


def _result(issues: _Issues, **kwargs) -> V3Validation:
    return V3Validation(
        ok=not issues.any(), errors=tuple(issues.errors), unsupported=tuple(issues.unsupported), **kwargs
    )


# --------------------------------------------------------------------------- stage 1: shape


def _check_shape(doc: dict, issues: _Issues) -> None:
    for path, slot, ref in component_slots(doc):
        name = ref["component"]
        component = REGISTRY.get(name)
        if component is None:
            if name in PLANNED_COMPONENTS:
                issues.unsupported_(
                    (*path, "component"),
                    f"component {name!r} is not available yet (planned: {PLANNED_COMPONENTS[name]})",
                )
            else:
                choices = _choices(slot) or "none registered yet"
                issues.error((*path, "component"), f"unknown {slot} component {name!r}; choose one of: {choices}")
        elif component.slot != slot:
            choices = _choices(slot) or "none registered yet"
            issues.error(
                (*path, "component"),
                f"{name} is a {component.slot} component, not a {slot} component; choose one of: {choices}",
            )
    for alias, decl in doc["instruments"].items():
        base = ("instruments", alias)
        if alias in RESERVED_ALIASES:
            issues.error(
                base,
                f"alias {alias!r} is reserved in series refs (self: the sleeve's own instrument, index: a TWSE index)",
            )
        instrument = INSTRUMENTS.get(decl["symbol"])
        if instrument is None:
            tradable = ", ".join(symbol for symbol, row in INSTRUMENTS.items() if row.tradable)
            issues.error((*base, "symbol"), f"unknown symbol {decl['symbol']!r}; choose one of: {tradable}")
            continue
        if not instrument.tradable:
            issues.unsupported_(
                (*base, "symbol"), f"{instrument.symbol} is not available yet (planned: {instrument.planned})"
            )
            continue
        if instrument.kind == "future" and decl.get("roll") is None:
            issues.error(
                (*base, "roll"),
                "a futures instrument needs a roll policy: roll.before_settlement (1d) or roll.at_settlement (intraday)",
            )
        if instrument.kind != "future" and decl.get("roll") is not None:
            issues.error((*base, "roll"), f"{instrument.symbol} is not a future; remove roll")


# --------------------------------------------------------------------------- stage 2: $param


def _walk_outside(node: Any, path: tuple, skip: set[tuple]):
    if path in skip:
        return
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_outside(value, (*path, key), skip)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_outside(value, (*path, index), skip)


def _check_param_refs(doc: dict, params: dict, issues: _Issues) -> dict[str, list[tuple[tuple, Component]]]:
    """Return ``{param name: [(component path, component)]}`` for the grid sweep."""

    users: dict[str, list[tuple[tuple, Component]]] = {}
    param_paths = set()
    for path, _slot, ref in component_slots(doc):
        component = REGISTRY[ref["component"]]
        base = (*path, "params")
        param_paths.add(base)
        for ref_path, node in walk(ref.get("params") or {}, base):
            if not is_param_ref(node):
                continue
            name = node[PARAM_REF_KEY]
            if len(node) != 1 or not isinstance(name, str):
                issues.error(ref_path, 'a parameter reference must be exactly {"$param": "<name>"}')
                continue
            if name not in params:
                declared = ", ".join(sorted(params)) or "none"
                issues.error(ref_path, f"unknown parameter {name!r}; declared in params: {declared}")
                continue
            users.setdefault(name, []).append((path, component))
            if component.name == "direction.dsl_rules":
                continue  # the DSL validator checks where its $params may go
            if len(ref_path) != len(base) + 1:
                issues.error(ref_path, f"a $param must replace a whole top-level parameter of {component.name}")
                continue
            field_name = ref_path[-1]
            if field_name not in component.tunable:
                tunable = ", ".join(sorted(component.tunable)) or "none"
                issues.error(ref_path, f"{component.name}.{field_name} cannot be a $param; tunable fields: {tunable}")
                continue
            annotation = component.Params.model_fields[field_name].annotation
            if annotation is int and params[name].type != "int":
                issues.error(
                    ref_path, f"{component.name}.{field_name} is an integer; declare params.{name} with type int"
                )
    for path, node in _walk_outside(doc, (), param_paths | {("params",)}):
        if is_param_ref(node):
            issues.error(path, "a $param is only allowed inside a component's params")
    for name in params:
        if name not in users:
            issues.error(("params", name), f"parameter {name!r} is declared but never used")
    count = len(params)
    if not MIN_PARAMS <= count <= MAX_PARAMS:
        issues.error(
            "params",
            f"a spec declares {MIN_PARAMS}-{MAX_PARAMS} tunable $params, this one declares {count}; parameterise the "
            "values worth tuning (moving-average days, band, volatility window or k, base leverage)",
        )
    return users


# --------------------------------------------------------------------------- stage 3: component params


def _at(doc: dict, path: tuple) -> Any:
    node = doc
    for part in path:
        node = node[part]
    return node


def _dsl_document(sleeve: dict, doc: dict, params: dict) -> tuple[dict, dict[str, str]]:
    """The DSL v2 document a dsl_rules sleeve stands for, and a map of DSL top-level keys to spec paths."""

    alias = sleeve["instrument"]
    symbol = doc["instruments"][alias]["symbol"]
    body = dict(sleeve["direction"].get("params") or {})
    used = {node[PARAM_REF_KEY] for _path, node in walk(body) if is_param_ref(node) and node[PARAM_REF_KEY] in params}
    v2 = {**body, "symbol": symbol, "market": DSL_MARKETS.get(symbol, ""), "interval": doc["frequency"]}
    if used:
        v2["params"] = {name: params[name] for name in sorted(used)}
    outside = {
        "symbol": f"instruments.{alias}.symbol",
        "market": f"instruments.{alias}.symbol",
        "interval": "frequency",
    }
    return v2, outside


def _check_dsl_rules(path: tuple, sleeve: dict, doc: dict, issues: _Issues) -> None:
    from poseidon.strategies.dsl.v2 import DslUnsupportedError, DslValidationError, validate_v2

    raw_params = doc.get("params") or {}
    v2, outside = _dsl_document(sleeve, doc, raw_params)
    misplaced = [
        key for key in ("symbol", "market", "interval", "params") if key in (sleeve["direction"].get("params") or {})
    ]
    for key in misplaced:
        where = {"params": "the spec's top-level params", "interval": "frequency"}.get(key, "the instrument")
        issues.error((*path, "params", key), f"remove {key!r}: in spec v3 it comes from {where}")
    if misplaced:
        return
    base = (*path, "params")
    try:
        validate_v2(v2)
    except DslUnsupportedError as error:
        for item in error.errors:
            issues.unsupported_(_dsl_path(item["path"], base, outside), f"direction.dsl_rules: {item['msg']}")
    except DslValidationError as error:
        for item in error.errors:
            issues.error(_dsl_path(item["path"], base, outside), f"direction.dsl_rules: {item['msg']}", item["type"])


def _dsl_path(dsl_path: str, base: tuple, outside: dict[str, str]) -> str:
    head = dsl_path.split(".", 1)[0]
    if head == "params":
        return dsl_path
    if head in outside:
        return outside[head]
    return join_path((*base, dsl_path)) if dsl_path else join_path(base)


def _parse_components(resolved: dict, template: dict, issues: _Issues) -> dict[tuple, Any]:
    parsed: dict[tuple, Any] = {}
    for path, _slot, ref in component_slots(resolved):
        component = REGISTRY[ref["component"]]
        if component.name == "direction.dsl_rules":
            _check_dsl_rules(path, _at(template, path[:2]), template, issues)
            parsed[path] = component.parse(ref.get("params") or {})
            continue
        try:
            parsed[path] = component.parse(ref.get("params") or {})
        except ValidationError as error:
            _pydantic_issues(issues, error, (*path, "params"))
    return parsed


def _sweep_grid(template: dict, params: dict, users: dict, values: dict, issues: _Issues) -> None:
    """Every grid value of every $param must be valid for every (non-DSL) field that uses it."""

    for name, spec in params.items():
        for path, component in users.get(name, []):
            if component.name == "direction.dsl_rules":
                continue
            raw = _at(template, path).get("params") or {}
            failure = None
            for value in spec.grid():
                trial = substitute(raw, {**values, name: spec.cast(value)})
                try:
                    component.parse(trial)
                except ValidationError as error:
                    first = error.errors()[0]
                    where = join_path((*path, "params", *first["loc"]))
                    failure = f"value {value!r} is invalid at {where}: {first['msg']}"
                    break
            if failure:
                issues.error(("params", name), failure, "param_range")
                break


# --------------------------------------------------------------------------- stage 4: cross rules


def _check_cross(resolved: dict, parsed: dict, issues: _Issues) -> None:
    frequency = resolved["frequency"]
    aliases = {alias: INSTRUMENTS[decl["symbol"]] for alias, decl in resolved["instruments"].items()}
    for path, _slot, ref in component_slots(resolved):
        component = REGISTRY[ref["component"]]
        if frequency not in component.frequencies:
            supported = ", ".join(ordered_frequencies(component.frequencies))
            issues.error(
                (*path, "component"),
                f"{component.name} supports frequency {supported}; the spec frequency is {frequency}",
            )

    seen_ids: dict[str, int] = {}
    used_aliases: set[str] = set()
    bases: set[str] = set()
    has_dsl = False
    for index, sleeve in enumerate(resolved["sleeves"]):
        base = ("sleeves", index)
        if sleeve["id"] in seen_ids:
            issues.error((*base, "id"), f"duplicate sleeve id {sleeve['id']!r} (also sleeves.{seen_ids[sleeve['id']]})")
        seen_ids.setdefault(sleeve["id"], index)
        alias = sleeve["instrument"]
        instrument = aliases.get(alias)
        if instrument is None:
            declared = ", ".join(aliases)
            issues.error((*base, "instrument"), f"unknown instrument alias {alias!r}; declared: {declared}")
        used_aliases.add(alias)

        sizing: SizingComponent = REGISTRY[sleeve["sizing"]["component"]]
        bases.add(sizing.basis)
        chain = sleeve.get("leverage") or []
        names = [link["component"] for link in chain]
        if sizing.basis == "equity":
            if not chain:
                issues.error((*base, "leverage"), f"{sizing.name} needs a leverage chain starting with leverage.base")
            elif names[0] != "leverage.base":
                issues.error((*base, "leverage", 0, "component"), "the leverage chain must start with leverage.base")
        elif chain:
            issues.error(
                (*base, "leverage"),
                f"{sizing.name} ignores leverage; remove the leverage chain or use sizing.target_leverage",
            )
        for link, name in enumerate(names):
            if link > 0 and name == "leverage.base":
                issues.error((*base, "leverage", link, "component"), "leverage.base may only be the first link")
        for list_name in ("leverage", "exits", "hedges"):
            listed = [ref["component"] for ref in sleeve.get(list_name) or []]
            for position, name in enumerate(listed):
                if name in listed[:position]:
                    issues.error(
                        (*base, list_name, position, "component"),
                        f"{name} appears twice in {list_name}; keep one (the same slot cannot repeat a component)",
                    )

        if sleeve["direction"]["component"] == "direction.dsl_rules":
            has_dsl = True
            _check_dsl_sleeve(base, sleeve, instrument, issues)

        for path, _slot, ref in component_slots({**resolved, "sleeves": [sleeve], "instruments": {}}):
            if path[0] != "sleeves":
                continue
            real_path = (*base, *path[2:])
            component = REGISTRY[ref["component"]]
            ctx = CheckContext(resolved, frequency, sleeve, instrument, aliases)
            params = parsed[real_path]
            for rel, message in component.check(params, ctx):
                issues.error((*real_path, "params", rel) if rel else real_path, message)
            signal = getattr(params, "signal", None)
            if isinstance(signal, str) and signal.split(":")[0] != "self":
                used_aliases.add(signal.split(":")[0])

    for alias in aliases:
        if alias not in used_aliases:
            issues.error(("instruments", alias), f"instrument alias {alias!r} is not used by any sleeve or series")
    if len(bases) > 1:
        issues.error("sleeves", "all sleeves must share one sizing basis (all equity-sized or all fixed contracts)")

    allocation = REGISTRY[resolved["allocation"]["component"]]
    count = len(resolved["sleeves"])
    if not allocation.min_sleeves <= count <= allocation.max_sleeves:
        issues.error(
            "allocation.component",
            f"{allocation.name} takes {allocation.min_sleeves}-{allocation.max_sleeves} sleeves, the spec has {count}",
        )
    execution = REGISTRY[resolved["execution"]["component"]]
    all_dsl = all(sleeve["direction"]["component"] == "direction.dsl_rules" for sleeve in resolved["sleeves"])
    if execution.fill == "signal_close" and not all_dsl:
        issues.error("execution.component", "execution.signal_close is only for direction.dsl_rules (DSL v2 parity)")
    if has_dsl and execution.fill != "signal_close":
        issues.error(
            "execution.component", "direction.dsl_rules fills at the signal bar's close: use execution.signal_close"
        )
    for alias, decl in resolved["instruments"].items():
        roll = decl.get("roll")
        if roll is not None:
            component = REGISTRY[roll["component"]]
            ctx = CheckContext(resolved, frequency, None, aliases[alias], aliases)
            for rel, message in component.check(parsed[("instruments", alias, "roll")], ctx):
                path = ("instruments", alias, "roll", "params", rel) if rel else ("instruments", alias, "roll")
                issues.error(path, message)


def _check_dsl_sleeve(base: tuple, sleeve: dict, instrument: Instrument | None, issues: _Issues) -> None:
    """Until DSL parity (step 7) the fast path is exactly the v1 shape: fixed contracts, trade on signal change."""

    if sleeve["sizing"]["component"] != "sizing.fixed_contracts":
        issues.error((*base, "sizing", "component"), "direction.dsl_rules sizes with sizing.fixed_contracts")
    if sleeve["rebalance"]["component"] != "rebalance.on_signal_change":
        issues.error((*base, "rebalance", "component"), "direction.dsl_rules trades with rebalance.on_signal_change")
    for list_name, where in (("exits", "the DSL risk block"), ("hedges", "a later milestone")):
        if sleeve.get(list_name):
            issues.error((*base, list_name), f"direction.dsl_rules takes no {list_name}; use {where}")
    if instrument is not None and instrument.symbol not in DSL_SYMBOLS:
        issues.error((*base, "instrument"), f"direction.dsl_rules trades {' or '.join(DSL_SYMBOLS)} only")


# --------------------------------------------------------------------------- stages 5-8: feasibility and policy


def warmup_calendar_days(bars: int, frequency: str) -> int:
    """Calendar days of data to load before the window start for ``bars`` warm-up bars."""

    if frequency in INTRADAY_FREQUENCIES:
        return 0  # intraday runs warm up inside the data window (plan.DEFAULT_GATES["warmup_days"])
    return math.ceil(bars * 365 / TRADING_DAYS_PER_YEAR) + 7


def _history_bars(template: dict, params: dict) -> int:
    """Warm-up at each $param's largest grid value (history grows with window lengths)."""

    high = {name: spec.cast(spec.high) for name, spec in params.items()}
    resolved = substitute(template, high)
    bars = 0
    for _path, _slot, ref in component_slots(resolved):
        component = REGISTRY[ref["component"]]
        if component.name == "direction.dsl_rules":
            continue
        bars = max(bars, component.history(component.parse(ref.get("params") or {})))
    return bars


def _window_uncovered(series: str, since: date, start: date) -> str:
    """The run's data window is frozen in its config, not in the spec, so no spec edit can fix this."""

    return (
        f"{series} data starts {since.isoformat()}, after the run's data window start {start.isoformat()}; "
        "the window is fixed by the run config, so this series cannot be read in this run"
    )


def _check_data(resolved: dict, template: dict, params: dict, feasibility: FeasibilityContext, issues, notes) -> None:
    frequency = resolved["frequency"]
    default_start = INTRADAY_WINDOW_START if frequency in INTRADAY_FREQUENCIES else DAILY_WINDOW_START
    start = feasibility.start or default_start
    bars = _history_bars(template, params)
    warmup = timedelta(days=warmup_calendar_days(bars, frequency))
    need_from = start - warmup
    for alias, decl in resolved["instruments"].items():
        instrument = INSTRUMENTS[decl["symbol"]]
        since = data_since(instrument, frequency)
        path = ("instruments", alias, "symbol")
        if since is None:
            issues.error(path, f"{instrument.symbol} has no {frequency} data source")
            continue
        if start < since:
            issues.unsupported_(path, _window_uncovered(f"{instrument.symbol} {frequency}", since, start))
        elif need_from < since:
            issues.error(
                path,
                f"{instrument.symbol} {frequency} data starts {since.isoformat()}; the window start {start.isoformat()} "
                f"minus the warm-up ({bars} bars, {warmup.days} calendar days) needs data from {need_from.isoformat()}; "
                "shorten the warm-up windows",
            )
        note = instrument.proxy_note(start)
        if note:
            notes.append(note)
    for path, _slot, ref in component_slots(resolved):
        component = REGISTRY[ref["component"]]
        if component.name == "direction.dsl_rules":
            continue
        signal = getattr(component.parse(ref.get("params") or {}), "signal", None)
        index = signal_index(signal) if isinstance(signal, str) else None
        if index is None:
            continue
        if start < index.data_since:
            issues.unsupported_(
                (*path, "params", "signal"), _window_uncovered(f"{signal} ({index.name_en})", index.data_since, start)
            )
        elif need_from < index.data_since:
            issues.error(
                (*path, "params", "signal"),
                f"{signal} ({index.name_en}) data starts {index.data_since.isoformat()}; the window start "
                f"{start.isoformat()} minus the warm-up ({bars} bars, {warmup.days} calendar days) needs data from "
                f"{need_from.isoformat()}; shorten the warm-up windows or read another series",
            )
    if feasibility.end is not None and feasibility.end <= start:
        issues.error(
            "", f"the data window end {feasibility.end.isoformat()} is not after its start {start.isoformat()}"
        )


def _grid_points(names: list[str], params: dict):
    if not names:
        return [{}]
    size = math.prod(params[name].grid_size() for name in names)
    if size <= GRID_ENUMERATION_LIMIT:
        axes = [params[name].grid() for name in names]
    else:
        axes = [[params[name].cast(params[name].low), params[name].cast(params[name].high)] for name in names]
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*axes)]


def _chain(links: list[tuple[LeverageComponent, Any]]) -> tuple[float, float]:
    nominal = upper = 0.0
    for component, params in links:
        nominal = component.nominal(params, nominal)
        upper = component.upper(params, upper)
    return nominal, upper


def _contracts(units: int) -> str:
    return f"{units} contract" + ("" if units == 1 else "s")


def _family_hint(price: float, leverage: float, equity: float, current: Instrument, sizing, sizing_params) -> str:
    options = []
    for symbol in SAME_INDEX_FAMILY:
        other = INSTRUMENTS[symbol]
        if other.symbol == current.symbol or other.point_value_twd >= current.point_value_twd or not other.tradable:
            continue
        units = sizing.units(
            sizing_params, equity=equity, direction=1, leverage=leverage, weight=1.0, price=price, instrument=other
        )
        actual = units * other.notional_twd(price) / equity
        options.append(f"{symbol} ({other.notional_twd(price):,.0f} TWD each): {_contracts(units)} ~ {actual:.2f}x")
    if not options:
        return "raise initial capital or the target leverage"
    return "use a smaller contract on the same index: " + "; ".join(options)


def _check_sizing(resolved: dict, template: dict, params: dict, feasibility: FeasibilityContext, issues) -> None:
    equity = float(resolved.get("capital", {}).get("initial_twd", DEFAULT_INITIAL_CAPITAL_TWD))
    for index, sleeve in enumerate(resolved["sleeves"]):
        sizing: SizingComponent = REGISTRY[sleeve["sizing"]["component"]]
        if sizing.basis != "equity":
            continue
        base = ("sleeves", index)
        instrument = INSTRUMENTS[resolved["instruments"][sleeve["instrument"]]["symbol"]]
        source = instrument.price_proxy or instrument.symbol
        prices = tuple((feasibility.prices or {}).get(source) or (REFERENCE_INDEX_LEVEL,))
        raw_sleeve = template["sleeves"][index]
        refs = [raw_sleeve["sizing"], *raw_sleeve.get("leverage", [])]
        names = sorted(
            {node[PARAM_REF_KEY] for ref in refs for _path, node in walk(ref.get("params") or {}) if is_param_ref(node)}
        )
        sizing_params = sizing.parse(sleeve["sizing"].get("params") or {})
        defaults = {name: spec.cast(spec.default) for name, spec in params.items()}
        reported = {"units": False, "margin": False, "cap": False}
        grid = []
        for point in _grid_points(names, params):
            values = {**defaults, **point}
            links = []
            for link in raw_sleeve.get("leverage", []):
                component = REGISTRY[link["component"]]
                links.append((component, component.parse(substitute(link.get("params") or {}, values))))
            grid.append((point, *_chain(links)))
        # Name the grid point only when the leverage actually varies over the grid.
        varies = len({(nominal, upper) for _point, nominal, upper in grid}) > 1
        for point, nominal, upper in grid:
            where = f" at {point}" if varies else ""
            if upper > LEVERAGE_CAP + 1e-9 and not reported["cap"]:
                reported["cap"] = True
                issues.error(
                    (*base, "leverage"), f"leverage can reach {upper:g}x{where}, above the {LEVERAGE_CAP:g}x policy cap"
                )
            for price in prices:
                notional = instrument.notional_twd(price)
                units = sizing.units(
                    sizing_params,
                    equity=equity,
                    direction=1,
                    leverage=nominal,
                    weight=1.0,
                    price=price,
                    instrument=instrument,
                )
                actual = units * notional / equity
                if not reported["units"] and (units == 0 or abs(actual - nominal) / nominal > MAX_ROUNDING_ERROR):
                    reported["units"] = True
                    hint = _family_hint(price, nominal, equity, instrument, sizing, sizing_params)
                    bought = f"buys {_contracts(units)} of {instrument.symbol} ({notional:,.0f} TWD notional each)"
                    if units:
                        bought += f" = {actual:.2f}x, {abs(actual - nominal) / nominal:.0%} off target"
                    issues.error(
                        (*base, "sizing"),
                        f"target {nominal:g}x{where} on {equity:,.0f} TWD at price {price:,.0f} {bought}; "
                        f"0 contracts or more than {MAX_ROUNDING_ERROR:.0%} rounding error is infeasible; {hint}",
                    )
                upper_units = sizing.units(
                    sizing_params,
                    equity=equity,
                    direction=1,
                    leverage=upper,
                    weight=1.0,
                    price=price,
                    instrument=instrument,
                )
                margin = upper_units * notional * instrument.initial_margin_ratio
                if not reported["margin"] and margin > equity * MAX_MARGIN_UTILISATION:
                    reported["margin"] = True
                    issues.error(
                        (*base, "leverage"),
                        f"at {upper:g}x{where} the initial margin {margin:,.0f} TWD exceeds "
                        f"{MAX_MARGIN_UTILISATION:.0%} of equity ({equity * MAX_MARGIN_UTILISATION:,.0f} TWD); "
                        "lower the leverage",
                    )


# --------------------------------------------------------------------------- entry point


def validate_spec_v3(doc: Any, *, feasibility: FeasibilityContext | None = None) -> V3Validation:
    """Validate a spec v3 document (template with ``$param`` references) in the §3.7 order."""

    issues = _Issues()
    feasibility = feasibility or FeasibilityContext()
    if not isinstance(doc, dict):
        issues.error("", "a spec v3 document must be a JSON object")
        return _result(issues)
    try:
        frame = SpecV3.model_validate(doc)
    except ValidationError as error:
        _pydantic_issues(issues, error, ())
        return _result(issues)
    template = copy.deepcopy(doc)
    params = dict(frame.params)
    _check_shape(template, issues)
    if issues.any():
        return _result(issues, template=template, params=params)

    users = _check_param_refs(template, params, issues)
    if issues.any():
        return _result(issues, template=template, params=params)
    values = {name: spec.cast(spec.default) for name, spec in params.items()}
    resolved = substitute(template, values)

    parsed = _parse_components(resolved, template, issues)
    if not issues.any():
        _sweep_grid(template, params, users, values, issues)
    if issues.any():
        return _result(issues, template=template, resolved=resolved, params=params)

    _check_cross(resolved, parsed, issues)
    if issues.any():
        return _result(issues, template=template, resolved=resolved, params=params)

    notes: list[str] = []
    _check_data(resolved, template, params, feasibility, issues, notes)
    _check_sizing(resolved, template, params, feasibility, issues)
    return _result(
        issues,
        notes=tuple(dict.fromkeys(notes)),
        template=template,
        resolved=resolved,
        params=params,
        strategy_class=strategy_class(resolved),
    )


class SpecV3Validator:
    """``config_json["spec_validator"] == "spec_v3"`` (CONFIG_VERSION 6). No run freezes it before step 4."""

    name = "spec_v3"

    def validate_spec(self, spec: dict):
        from poseidon.pipeline.spec_validation import SpecValidation

        result = validate_spec_v3(spec)
        return SpecValidation(
            result.ok,
            tuple(issue.text() for issue in result.errors),
            tuple(issue.text() for issue in result.unsupported),
        )

    def resolve(self, spec: dict) -> dict:
        """``{"spec", "template", "params", "notes", "strategy_class"}`` for a spec that passed."""

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
