"""Capability brief, LLM catalog and JSON Schema, all rendered from the registry (design 2026-10-24 §3.9).

One source: components (CAN), ``PLANNED_COMPONENTS`` (not yet),
``UNSUPPORTED_CAPABILITIES`` (CANNOT), the instrument table and the policy
constants. The rendered files are checked in under ``generated/``; a golden
test fails when they drift from what the registry renders now. Regenerate
with ``python -m poseidon.pipeline.v3 brief``.

Consumers (step 6 switches them over): scout reads ``capability_brief.md``
instead of its ``CAPABILITY_BRIEF`` constant, ``bridge_sync`` pushes the file
to Sisyphus, ``spec_generate`` v3 builds its system prompt from
:func:`catalog`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from poseidon.pipeline.v3 import components as _components  # noqa: F401  (registers the components)
from poseidon.pipeline.v3.describe import describe_spec
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.instruments import (
    INITIAL_MARGIN_RATIO,
    INSTRUMENTS,
    MAINTENANCE_MARGIN_RATIO,
    MAX_MARGIN_UTILISATION,
    TX_INTRADAY_DATA_SINCE,
    ordered_frequencies,
)
from poseidon.pipeline.v3.registry import (
    PLANNED_COMPONENTS,
    REGISTRY_VERSION,
    SLOTS,
    UNSUPPORTED_CAPABILITIES,
    Component,
    components_for,
)
from poseidon.pipeline.v3.spec import (
    ALIAS_PATTERN,
    MAX_INITIAL_CAPITAL_TWD,
    MAX_PARAMS,
    MAX_SLEEVES,
    MIN_INITIAL_CAPITAL_TWD,
    MIN_PARAMS,
    SPEC_VERSION,
    Capital,
)
from poseidon.pipeline.v3.validation import DAILY_WINDOW_START, LEVERAGE_CAP, MAX_ROUNDING_ERROR, validate_spec_v3
from poseidon.strategies.dsl.v2.schema import NAME_PATTERN, PARAM_NAME_PATTERN, ParamSpec

GENERATED_DIR = Path(__file__).parent / "generated"
BRIEF_FILE = "capability_brief.md"
SCHEMA_FILE = "spec_v3.schema.json"

SLOT_TITLES = {
    "roll": "roll (per futures instrument: `instruments.<alias>.roll`)",
    "direction": "direction (per sleeve: d in {-1, 0, +1})",
    "leverage": "leverage (per sleeve: an ordered chain; `leverage.base` first, then filters and caps)",
    "sizing": "sizing (per sleeve: d x L -> integer contracts)",
    "rebalance": "rebalance (per sleeve: when a changed target becomes an order)",
    "exit": "exits (per sleeve: protective orders)",
    "hedge": "hedges (per sleeve)",
    "allocation": "allocation (weights across sleeves)",
    "execution": "execution (decision -> fill timing)",
}

# DSL v2.1 as the pipeline validator accepts it (same content as scout.CAPABILITY_BRIEF, 2026-10-24).
DSL_V21_CAN = """\
DSL v2.1 for TX (Taiwan index futures) on 5m bars by default (1m/15m possible), 1 fixed contract, data from 2020-03.
It CAN express:
  - Sessions: day 08:45-13:45 and night 15:00-05:00 (Asia/Taipei); time windows, minutes elapsed since session open.
  - Price (open/high/low/close); EMA / SMA (of close), ATR, RSI of the strategy interval (periods 2-400, ATR/RSI
    2-100); constants.
  - One level of smoothing of such an indicator: SMA or EMA (2-400 bars) of EMA/SMA/ATR/RSI, e.g. SMA50 of ATR14
    (volatility contraction / expansion), EMA of RSI.
  - Rolling N-bar channel (Donchian): highest/lowest of high, low or close over the previous N bars (2-400; counted
    in bars across sessions), for N-bar breakouts and N-bar swing stops.
  - Opening-range levels (high/low/mid/range of the first N minutes, 5-120), session open/high/low/VWAP,
    previous-session and previous-day high/low/close (gap and prev-level ideas).
  - Conditions: crosses above/below, compares (> >= < <=) with point or ATR-multiple offsets, combined with
    all/any/none. A multiplier on the compared side gives ratio thresholds (e.g. ATR < 0.7 x its SMA50; on a price
    level only a 0.9-1.1 percent envelope, e.g. close > 1.005 x EMA20).
  - Exits: stop loss (points, ATR multiple, or a level such as the N-bar low), take profit (points, ATR multiple,
    R multiple, level), trailing stop (points / ATR, optionally after reaching some R), max hold bars, flat before
    session end, max trades per session, no re-entry after stop, close on session change. Long and short entry
    rules, 1-6 rules."""

CANNOT_DSL = (
    "Inside direction.dsl_rules (intraday): higher timeframes (e.g. 60m EMA, daily RSI), smoothing a smoothed "
    "series, standard deviation / Bollinger bands / z-score / percentile rank, entries on anything but TX price, "
    "time, session levels, the indicators above and N-bar channels."
)
HIGHER_TIMEFRAME_RULE = (
    "Never stand in for a higher-timeframe indicator by reusing its length on intraday bars: a 200-day MA is NOT "
    "SMA(200) on 5m bars (that spans 1-4 trading days, depending on the sessions). On intraday bars such an idea is "
    "unsupported as written; say so instead of substituting. A daily indicator belongs on frequency 1d (a 200-day MA "
    "regime is regime.ma_band with ma_days 200)."
)


# --------------------------------------------------------------------------- JSON Schema (export only)


class ParamRef(BaseModel):
    """``{"$param": "<name>"}``: the value is the tunable spec-level parameter ``<name>``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    ref: Annotated[str, Field(alias="$param", pattern=r"^[a-z][a-z0-9_]{0,31}$")]


def _template_params(component: Component) -> type[BaseModel]:
    """The component's Params with each tunable field also accepting a ParamRef."""

    fields: dict[str, Any] = {}
    for name, info in component.Params.model_fields.items():
        annotation: Any = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation
        if name in component.tunable:
            annotation = Union[annotation, ParamRef]  # noqa: UP007  (runtime-built union)
        default = ... if info.is_required() else info.default
        fields[name] = (annotation, Field(default))
    model_name = "".join(part.title() for part in component.name.replace(".", "_").split("_")) + "Params"
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


def _ref_model(component: Component) -> type[BaseModel]:
    if component.name == "direction.dsl_rules":
        params: Any = (
            dict[str, Any],
            Field(description="a DSL v2 document without symbol/market/interval/params (see the DSL v2 schema)"),
        )
    else:
        params = (_template_params(component), Field(default_factory=dict))
    model_name = "".join(part.title() for part in component.name.replace(".", "_").split("_"))
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid", title=component.name),
        __doc__=component.summary,
        component=(Literal[component.name], ...),
        params=params,
    )


def _slot_type(slot: str) -> Any:
    refs = [_ref_model(component) for component in components_for(slot)]
    if not refs:
        return None
    if len(refs) == 1:
        return refs[0]
    return Annotated[Union[tuple(refs)], Field(discriminator="component")]  # noqa: UP007


def schema_model() -> type[BaseModel]:
    """A pydantic model of the whole document, discriminated by component (export and tests only)."""

    config = ConfigDict(extra="forbid")
    empty = (Annotated[list[dict[str, Any]], Field(max_length=0)], Field(default_factory=list))

    def listed(slot):
        kind = _slot_type(slot)
        return (list[kind], Field(default_factory=list)) if kind is not None else empty

    instrument = create_model(
        "InstrumentDecl",
        __config__=config,
        symbol=(Literal[tuple(symbol for symbol, row in INSTRUMENTS.items() if row.tradable)], ...),
        roll=(_slot_type("roll") | None, None),
    )
    sleeve = create_model(
        "Sleeve",
        __config__=config,
        id=(Annotated[str, Field(pattern=ALIAS_PATTERN)], ...),
        instrument=(str, ...),
        direction=(_slot_type("direction"), ...),
        leverage=listed("leverage"),
        sizing=(_slot_type("sizing"), ...),
        rebalance=(_slot_type("rebalance"), ...),
        exits=listed("exit"),
        hedges=listed("hedge"),
    )
    return create_model(
        "SpecV3Document",
        __config__=ConfigDict(extra="forbid", title=f"Strategy spec v{SPEC_VERSION} (registry {REGISTRY_VERSION})"),
        spec_version=(Literal[3], ...),
        name=(Annotated[str, Field(pattern=NAME_PATTERN)], ...),
        description=(Annotated[str, Field(max_length=500)], ""),
        frequency=(Literal["1d", "1m", "5m", "15m"], ...),
        capital=(Capital, Field(default_factory=Capital)),
        instruments=(dict[Annotated[str, Field(pattern=ALIAS_PATTERN)], instrument], ...),
        sleeves=(Annotated[list[sleeve], Field(min_length=1, max_length=MAX_SLEEVES)], ...),
        allocation=(_slot_type("allocation"), ...),
        execution=(_slot_type("execution"), ...),
        params=(dict[Annotated[str, Field(pattern=PARAM_NAME_PATTERN)], ParamSpec], Field(default_factory=dict)),
    )


def spec_json_schema() -> dict:
    schema = schema_model().model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["description"] = (
        f"Spec v{SPEC_VERSION}, REGISTRY_VERSION {REGISTRY_VERSION}. Shape only: cross rules, $param policy "
        f"({MIN_PARAMS}-{MAX_PARAMS}), feasibility and the {LEVERAGE_CAP:g}x leverage cap are checked by spec_validate."
    )
    return schema


# --------------------------------------------------------------------------- catalog (machine form)


def _param_rows(component: Component) -> list[dict]:
    if component.name == "direction.dsl_rules":
        return [
            {"name": "<DSL v2 document>", "spec": "rules, risk, name, dsl_version 2 (see DSL v2.1)", "tunable": True}
        ]
    schema = component.Params.model_json_schema()
    rows = []
    for name, prop in schema.get("properties", {}).items():
        rows.append(
            {
                "name": name,
                "spec": _prop_text(prop, name in schema.get("required", [])),
                "tunable": name in component.tunable,
            }
        )
    return rows


def _prop_text(prop: dict, required: bool) -> str:
    if "enum" in prop:
        text = " | ".join(str(value) for value in prop["enum"])
    elif "const" in prop:
        text = str(prop["const"])
    else:
        text = prop.get("type", "value")
        low = prop.get("minimum", prop.get("exclusiveMinimum"))
        high = prop.get("maximum", prop.get("exclusiveMaximum"))
        if low is not None or high is not None:
            left = "(" if "exclusiveMinimum" in prop else "["
            right = ")" if "exclusiveMaximum" in prop else "]"
            text += f" {left}{low}, {high}{right}"
        if "pattern" in prop:
            text += ' "<alias|self>:<open|high|low|close|adjusted>"'
    if required:
        return text + ", required"
    return f"{text}, default {json.dumps(prop.get('default'), ensure_ascii=False)}"


def catalog() -> dict:
    """The registry as data: what ``spec_generate`` v3 renders into its system prompt."""

    return {
        "spec_version": SPEC_VERSION,
        "registry_version": REGISTRY_VERSION,
        "components": [
            {
                "name": component.name,
                "slot": component.slot,
                "version": component.version,
                "frequencies": ordered_frequencies(component.frequencies),
                "summary": component.summary,
                "capabilities": list(component.capabilities),
                "params": _param_rows(component),
                "example": component.example,
            }
            for slot in SLOTS
            for component in components_for(slot)
        ],
        "planned": dict(PLANNED_COMPONENTS),
        "unsupported": dict(UNSUPPORTED_CAPABILITIES),
        "instruments": [_instrument_row(row) for row in INSTRUMENTS.values()],
        "policy": _policy(),
    }


def _instrument_row(row) -> dict:
    return {
        "symbol": row.symbol,
        "kind": row.kind,
        "name_en": row.name_en,
        "name_zh": row.name_zh,
        "point_value_twd": row.point_value_twd,
        "tick": row.tick_size,
        "fee_twd_per_side": row.cost_model.commission_per_contract,
        "tax_rate": row.cost_model.tax_rate,
        "listed_since": row.listed_since.isoformat(),
        "price_proxy": row.price_proxy,
        "daily_data_since": row.daily_data_since.isoformat() if row.daily_data_since else None,
        "intraday_data_since": row.intraday_data_since.isoformat() if row.intraday_data_since else None,
        "tradable": row.tradable,
        "planned": row.planned,
    }


def _policy() -> dict:
    return {
        "leverage_cap": LEVERAGE_CAP,
        "initial_capital_twd": {"default": 1_000_000, "min": MIN_INITIAL_CAPITAL_TWD, "max": MAX_INITIAL_CAPITAL_TWD},
        "params_per_spec": [MIN_PARAMS, MAX_PARAMS],
        "max_rounding_error": MAX_ROUNDING_ERROR,
        "initial_margin_ratio": INITIAL_MARGIN_RATIO,
        "maintenance_margin_ratio": MAINTENANCE_MARGIN_RATIO,
        "max_margin_utilisation": MAX_MARGIN_UTILISATION,
        "daily_window_start": DAILY_WINDOW_START.isoformat(),
    }


# --------------------------------------------------------------------------- brief (human form)


def _component_lines(component: Component) -> list[str]:
    frequencies = ", ".join(ordered_frequencies(component.frequencies))
    lines = [f"- `{component.name}` ({frequencies}): {component.summary}"]
    if component.name == "direction.dsl_rules":
        lines += ["  " + line if line else "" for line in DSL_V21_CAN.splitlines()]
        return lines
    for row in _param_rows(component):
        mark = "; tunable ($param)" if row["tunable"] else ""
        lines.append(f"  - `{row['name']}`: {row['spec']}{mark}")
    return lines


def render_brief() -> str:
    acceptance = tmf_dynamic_leverage_3x()
    checked = validate_spec_v3(acceptance)
    if not checked.ok:
        raise RuntimeError("the acceptance example no longer validates; fix the registry first")
    lines = [
        "# Strategy pipeline capability brief",
        "",
        f"Spec v{SPEC_VERSION}, REGISTRY_VERSION {REGISTRY_VERSION}. Rendered from the component registry and the "
        "instrument table by `python -m poseidon.pipeline.v3 brief`; edit the registry, not this file.",
        "",
        "The downstream pipeline turns an idea into a spec v3 and backtests it. A spec is one or more sleeves; each "
        "sleeve is one instrument + a direction + an ordered leverage chain + sizing + rebalance (+ exits, hedges), "
        "plus one allocation and one execution rule. Every slot is filled by picking a registered component below and "
        f"setting its parameters. {MIN_PARAMS}-{MAX_PARAMS} numeric parameters per spec are tunable ranges (`$param`) "
        "that the optimizer searches; only fields marked tunable can be one.",
        "",
        "## Frequencies",
        "",
        "- `1d`: decide after every instrument's daily close, fill at the next session's open (TX family: the 15:00 "
        f"night session). Daily runs start {DAILY_WINDOW_START.isoformat()} by default; the warm-up comes from "
        "earlier data.",
        f"- `1m` / `5m` / `15m` (intraday): only `direction.dsl_rules` (DSL v2.1), data from "
        f"{TX_INTRADAY_DATA_SINCE.isoformat()}.",
        "",
        "## Instruments",
        "",
        "| symbol | name | point value (TWD) | tick | fee (TWD/contract/side) | listed | prices | available |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in INSTRUMENTS.values():
        prices = f"{row.price_proxy} prices (proxy)" if row.price_proxy else "own"
        available = "yes" if row.tradable else f"not yet: {row.planned}"
        lines.append(
            f"| {row.symbol} | {row.name_en} ({row.name_zh}) | {row.point_value_twd:g} | {row.tick_size:g} | "
            f"{row.cost_model.commission_per_contract:g} | {row.listed_since.isoformat()} | {prices} | {available} |"
        )
    lines += [
        "",
        "Futures: tax 0.002% of notional on both legs, 1 tick slippage per side; margin approximated as "
        f"{INITIAL_MARGIN_RATIO:.1%} initial / {MAINTENANCE_MARGIN_RATIO:.1%} maintenance of notional. Daily prices "
        "are TAIFEX contract-month data (signals use the back-adjusted continuous series); MTX and TMF trade on TX "
        "prices with their own contract specs, flagged as proxy in reports (before the TMF listing the contract did "
        "not exist).",
        "",
        "## Components (what a spec CAN use)",
    ]
    for slot in SLOTS:
        registered = components_for(slot)
        lines += ["", f"### {SLOT_TITLES[slot]}", ""]
        if not registered:
            lines.append("- none yet (see Planned)")
        for component in registered:
            lines += _component_lines(component)
    lines += [
        "",
        "## Policy",
        "",
        f"- Initial capital {MIN_INITIAL_CAPITAL_TWD:,}-{MAX_INITIAL_CAPITAL_TWD:,} TWD (default 1,000,000).",
        f"- Equity-sized sleeves: leverage at most {LEVERAGE_CAP:g}x; integer contracts; at the first bar every grid "
        "point must buy at least 1 contract and land within "
        f"{MAX_ROUNDING_ERROR:.0%} of its target leverage (otherwise use MTX or TMF); initial margin at the largest "
        f"leverage at most {MAX_MARGIN_UTILISATION:.0%} of equity.",
        "- Shorts only through futures; no option selling.",
        "- Each slot holds a component at most once (e.g. one volatility cap).",
        "",
        "## Planned (not available yet; ideas needing these come back unsupported for now)",
        "",
    ]
    lines += [f"- `{name}`: {when}" for name, when in PLANNED_COMPONENTS.items()]
    lines += [
        "",
        '## It CANNOT express (such ideas come back "unsupported"; do not propose them)',
        "",
    ]
    lines += [f"- {text} (`{token}`)" for token, text in UNSUPPORTED_CAPABILITIES.items()]
    lines += [
        "- options or implied volatility beyond the planned protective put; standalone volume / OI conditions.",
        f"- {CANNOT_DSL}",
        f"- {HIGHER_TIMEFRAME_RULE}",
        "",
        "## Example",
        "",
        "The M1 acceptance spec (`tmf_dynamic_leverage_3x`) reads:",
        "",
        "> " + describe_spec(checked.template, checked.params),
        "",
    ]
    return "\n".join(lines)


def render_schema() -> str:
    return json.dumps(spec_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def rendered_files() -> dict[str, str]:
    return {BRIEF_FILE: render_brief(), SCHEMA_FILE: render_schema()}


def write_generated(out_dir: Path = GENERATED_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in rendered_files().items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
