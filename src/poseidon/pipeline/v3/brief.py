"""Capability brief, LLM catalog and JSON Schema, all rendered from the registry (design 2026-10-24 §3.9).

One source: components (CAN), ``PLANNED_COMPONENTS`` (not yet),
``UNSUPPORTED_CAPABILITIES`` (CANNOT), the instrument table and the policy
constants. The rendered files are checked in under ``generated/``; a golden
test fails when they drift from what the registry renders now. Regenerate
with ``python -m poseidon.pipeline.v3 brief``.

Consumers: ``spec_generate`` v3 builds its system prompt from :func:`catalog`.
The compact two-engine brief (:func:`render_capability_brief`, intraday DSL v2.1
plus the daily v3 registry) is checked in at ``scripts/strategy_scout/
capability_brief.md``; scout and ``bridge_sync`` (which pushes it to Sisyphus)
read that file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from poseidon.pipeline.v3 import components as _components  # noqa: F401  (registers the components)
from poseidon.pipeline.v3.components import SERIES_REF_PATTERN
from poseidon.pipeline.v3.describe import describe_spec
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050, tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.instruments import (
    BOARD_LOT,
    INITIAL_MARGIN_RATIO,
    INSTRUMENTS,
    MAINTENANCE_MARGIN_RATIO,
    MAX_MARGIN_UTILISATION,
    SIGNAL_INDICES,
    STOCK_DATA_SNAPSHOT,
    TX_DAILY_DATA_SINCE,
    TX_INTRADAY_DATA_SINCE,
    ordered_frequencies,
    stocks,
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
    SLEEVE_ID_PATTERN,
    SPEC_VERSION,
    Capital,
)
from poseidon.pipeline.v3.validation import (
    DAILY_WINDOW_START,
    LEVERAGE_CAP,
    MAX_ROUNDING_ERROR,
    FeasibilityContext,
    validate_spec_v3,
)
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


# The intraday section of the combined brief, moved verbatim from scout.CAPABILITY_BRIEF (2026-10-24, f8c9f1d
# rule included). Scout's prompt is exactly this text until step 7 lets scout propose daily ideas.
DSL_V21_BRIEF = """\
The downstream pipeline turns an idea into a DSL v2.1 strategy for TX (Taiwan index futures) and backtests it
(5m bars by default, 1m/15m possible, 1 fixed contract, data from 2020-03). It CAN express:
- Sessions: day 08:45-13:45 and night 15:00-05:00 (Asia/Taipei); time windows, minutes elapsed since session open.
- Price (open/high/low/close); EMA / SMA (of close), ATR, RSI of the strategy interval (periods 2-400, ATR/RSI 2-100);
  constants.
- One level of smoothing of such an indicator: SMA or EMA (2-400 bars) of EMA/SMA/ATR/RSI, e.g. SMA50 of ATR14
  (volatility contraction / expansion), EMA of RSI.
- Rolling N-bar channel (Donchian): highest/lowest of high, low or close over the previous N bars (2-400; counted in
  bars across sessions), for N-bar breakouts and N-bar swing stops.
- Opening-range levels (high/low/mid/range of the first N minutes, 5-120), session open/high/low/VWAP, previous-session
  and previous-day high/low/close (gap and prev-level ideas).
- Conditions: crosses above/below, compares (> >= < <=) with point or ATR-multiple offsets, combined with all/any/none.
  A multiplier on the compared side gives ratio thresholds (e.g. ATR < 0.7 x its SMA50; on a price level only a
  0.9-1.1 percent envelope, e.g. close > 1.005 x EMA20).
- Exits: stop loss (points, ATR multiple, or a level such as the N-bar low), take profit (points, ATR multiple,
  R multiple, level), trailing stop (points / ATR, optionally after reaching some R), max hold bars, flat before session
  end, max trades per session, no re-entry after stop, close on session change. Long and short entry rules, 1-6 rules.
- 1-4 tunable numeric parameters per strategy (windows, stop/target sizes, thresholds).
It CANNOT express (such ideas come back "unsupported"; do not propose them): higher timeframes (e.g. 60m EMA, daily
RSI), smoothing a smoothed series, standard deviation / Bollinger bands / z-score / percentile rank, volume / volume
profile / OI, options or implied volatility, order book, other instruments (NQ/ES/TE/stocks), news, fundamentals, ML
models, position sizing, pyramiding. Companion data such as foreign flows is NOT available to the DSL. Entries must be
rules on TX price/time/session levels, the indicators above and N-bar channels.
Never stand in for a higher-timeframe indicator by reusing its length on intraday bars: a 200-day MA is NOT SMA(200) on
5m bars (that spans 1-4 trading days, depending on the sessions). Such an idea is unsupported as written; say so instead of substituting.
"""


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
        unit=(Literal["board_lot", "odd_lot"] | None, None),
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
        if text == "array" and prop.get("description"):
            text = f"array: {prop['description']}"
        low = prop.get("minimum", prop.get("exclusiveMinimum"))
        high = prop.get("maximum", prop.get("exclusiveMaximum"))
        if low is not None or high is not None:
            left = "(" if "exclusiveMinimum" in prop else "["
            right = ")" if "exclusiveMaximum" in prop else "]"
            text += f" {left}{low}, {high}{right}"
        if prop.get("pattern") == SERIES_REF_PATTERN:
            text += ' "<alias|self>:<open|high|low|close|adjusted>" | "index:<' + "|".join(SIGNAL_INDICES) + '>"'
        elif prop.get("pattern") == SLEEVE_ID_PATTERN:
            text += " (a sleeve id)"
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
        "commission_rate": row.cost_model.buy_commission_rate,
        "tax_rate": row.cost_model.tax_rate,
        "lot": row.lot,
        "etf": row.etf,
        "odd_lot": row.odd_lot,
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


VIX_NOTE = (
    "`index:vix` is the previous US session's close: a Taiwan decision on date t reads US dates t-1 and earlier only "
    '(the VIX of US date t is not known yet). Scale-in ideas are CAN: "add when price falls N%" is '
    "`leverage.drawdown_ladder` (the opposite of the planned `risk.drawdown_brake`, which cuts leverage after the "
    'strategy\'s own equity drawdown) and "add when VIX is above X" is `leverage.level_ladder` on `index:vix`. '
    "Both only raise the incoming leverage, within the 5x cap and the margin limit."
)


def _signal_series_lines() -> list[str]:
    """Owner decision 2026-10-24 (afternoon) #1: the signal series is always named, never assumed."""

    lines = [
        "## Signal series",
        "",
        "`regime.ma_band`, `risk.vol_cap`, `leverage.drawdown_ladder` and `leverage.level_ladder` read the daily "
        "series named by their `signal` parameter, which is required (no default). The choice is part of the strategy: the same 200-day MA band over 2000-2014 ends 49% "
        "apart on the TX adjusted series vs the TAIEX price index (49 vs 33 regime switches).",
        "",
        f"- `self:adjusted`: the traded futures' back-adjusted continuous price (MTX/TMF: TX's), from "
        f"{TX_DAILY_DATA_SINCE.isoformat()}.",
    ]
    lines += [
        f"- `index:{index.name}`: {index.name_en} ({index.name_zh}), from {index.data_since.isoformat()}."
        for index in SIGNAL_INDICES.values()
    ]
    lines += [
        "",
        'An idea that only says "台指", "TAIEX" or "大盤" does not say which of these it means: that is a question for '
        "the owner, answered before the idea reaches the pipeline. Never pick one silently.",
        "",
        VIX_NOTE,
        "",
    ]
    return lines


def _stock_rules() -> list[str]:
    stock, etf = INSTRUMENTS["2330"].cost_model, INSTRUMENTS["0050"].cost_model
    return [
        f"Stocks / ETFs: long only, paid in full (no margin loans, no short selling); one unit is a board lot of "
        f"{BOARD_LOT:,} shares, or one share when the declaration says `unit: odd_lot`. Commission "
        f"{stock.buy_commission_rate:.4%} of notional per side, transaction tax {stock.tax_rate:.1%} (ETF "
        f"{etf.tax_rate:.1%}) on sells, 1 tick slippage per side on the TWSE tick ladder (stocks 0.01 under 10 up to "
        "5 from 1,000; ETFs 0.01 under 50, else 0.05). Prices are Thalassa's unadjusted daily bars; splits, dividends, "
        "rights offerings and capital reductions are booked at read time from `tw_corporate_actions` (a position's "
        "value is kept at the reference price, cash dividends are paid out), so a signal on a stock reads its "
        f"adjusted close (`<alias>` 收盤價). The listed symbols have data to {STOCK_DATA_SNAPSHOT.isoformat()}. "
        "The benchmark and regime underlying of a stock / ETF spec is 0050 total return.",
    ]


def render_brief() -> str:
    acceptance = tmf_dynamic_leverage_3x()
    checked = validate_spec_v3(acceptance)
    # the ETF data starts after the default window start: the window is chosen as the service does
    rotation = validate_spec_v3(etf_rotation_00631l_0050(), feasibility=FeasibilityContext(choose_start=True))
    if not checked.ok or not rotation.ok:
        raise RuntimeError("an acceptance example no longer validates; fix the registry first")
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
        f"night session; stocks / ETFs: the 09:00 open). Daily runs start {DAILY_WINDOW_START.isoformat()} by "
        "default; a series that starts later (an index a signal reads, a stock's own bars) moves the run's start to "
        "that series' first day plus the warm-up. The warm-up comes from earlier data.",
        f"- `1m` / `5m` / `15m` (intraday): only `direction.dsl_rules` (DSL v2.1), data from "
        f"{TX_INTRADAY_DATA_SINCE.isoformat()}.",
        "",
        "## Instruments",
        "",
        "### TAIFEX futures and options",
        "",
        "| symbol | name | point value (TWD) | tick | fee (TWD/contract/side) | listed | prices | available |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in INSTRUMENTS.values():
        if row.kind == "stock":
            continue
        prices = f"{row.price_proxy} prices (proxy)" if row.price_proxy else "own"
        available = (
            "yes"
            if row.tradable
            else f"hedges only (`{row.hedge_component}`)"
            if row.hedge_component
            else f"not yet: {row.planned}"
        )
        lines.append(
            f"| {row.symbol} | {row.name_en} ({row.name_zh}) | {row.point_value_twd:g} | {row.tick_size:g} | "
            f"{row.cost_model.commission_per_contract:g} | {row.listed_since.isoformat()} | {prices} | {available} |"
        )
    lines += [
        "",
        "Futures: tax 0.002% of notional on both legs, 1 tick slippage per side; margin approximated as "
        f"{INITIAL_MARGIN_RATIO:.1%} initial / {MAINTENANCE_MARGIN_RATIO:.1%} maintenance of notional. Daily prices "
        "are TAIFEX contract-month data (a signal on the futures reads the back-adjusted continuous series); MTX and "
        "TMF trade on TX prices with their own contract specs, flagged as proxy in reports (before the TMF listing "
        "the contract did not exist).",
        "",
        "### TWSE stocks and ETFs",
        "",
        "| symbol | name | ETF | daily data from |",
        "|---|---|---|---|",
    ]
    for row in stocks():
        lines.append(
            f"| {row.symbol} | {row.name_en} ({row.name_zh}) | {'yes' if row.etf else 'no'} | "
            f"{row.daily_data_since.isoformat()} |"
        )
    lines += [
        "",
        *_stock_rules(),
        "",
        *_signal_series_lines(),
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
        "- Stock / ETF sleeves: bought in full, exposure at most 1x of equity (`sizing.fixed_notional` or "
        "`sizing.fixed_contracts`); at the first bar every grid point must afford at least 1 unit (a board lot, or a "
        "share with `unit: odd_lot`).",
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
        "- options or implied volatility beyond bought protective puts; standalone volume / OI conditions.",
        f"- {CANNOT_DSL}",
        f"- {HIGHER_TIMEFRAME_RULE}",
        "",
        "## Example",
        "",
        "The M1 acceptance spec (`tmf_dynamic_leverage_3x`) reads:",
        "",
        "> " + describe_spec(checked.template, checked.params),
        "",
        "The M3 acceptance spec (`etf_rotation_00631l_0050`, two ETF sleeves under `allocation.regime_switch`) reads:",
        "",
        "> " + describe_spec(rotation.template, rotation.params),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- combined brief (scout, Sisyphus)

CAPABILITY_FILE = Path("scripts/strategy_scout/capability_brief.md")  # relative to the repo root
INTRADAY_HEADING = "## A. Intraday TX strategies (DSL v2.1)"
DAILY_HEADING = "## B. Daily strategies (spec v3, component registry)"


def intraday_section(text: str) -> str:
    """Part A of the combined brief, exactly :data:`DSL_V21_BRIEF` (what scout puts in its prompt)."""

    _, _, rest = text.partition(INTRADAY_HEADING + "\n\n")
    if not rest:
        raise ValueError("no intraday section in the capability brief")
    return rest.split("\n" + DAILY_HEADING, 1)[0].rstrip("\n") + "\n"


def _compact_param(row: dict) -> str:
    spec = "series ref, required" if row["name"] == "signal" else row["spec"]
    return f"{row['name']} {spec}{' *' if row['tunable'] else ''}"


def _compact_component(component: Component) -> str:
    params = "; ".join(_compact_param(row) for row in _param_rows(component))
    return f"- `{component.name}`: {component.summary}" + (f" [{params}]" if params else "")


def render_capability_brief() -> str:
    """One file for both engines: intraday DSL v2.1 (verbatim) and the daily registry (compact)."""

    checked = validate_spec_v3(tmf_dynamic_leverage_3x())
    if not checked.ok:
        raise RuntimeError("the acceptance example no longer validates; fix the registry first")
    lines = [
        "# Strategy pipeline capability brief",
        "",
        f"Rendered from the component registry by `python -m poseidon.pipeline.v3 brief` (REGISTRY_VERSION "
        f"{REGISTRY_VERSION}); edit the code, not this file. Two engines: A = intraday TX rules, B = daily "
        "strategies. An idea belongs to B when it needs daily indicators, leverage, sizing or a long history; "
        "to A when it is a bar-by-bar rule on TX price/time/session levels.",
        "",
        INTRADAY_HEADING,
        "",
        DSL_V21_BRIEF.rstrip("\n"),
        "",
        DAILY_HEADING,
        "",
        "Decide after each daily close, fill at the next session's open (TX family: the 15:00 night session; stocks "
        f"09:00). One or more sleeves = instrument + direction + leverage chain + sizing + rebalance. {MIN_PARAMS}-{MAX_PARAMS} "
        "tunable parameters per spec (`*` = may be one). Daily runs start "
        f"{DAILY_WINDOW_START.isoformat()}, or later when a series the spec reads starts later (its first day plus "
        "the warm-up, e.g. `index:taiex_tr`); warm-up comes from earlier data.",
        "",
        "Instruments (point value TWD/pt, fee TWD/contract/side, daily data from):",
    ]
    for row in INSTRUMENTS.values():
        if row.kind == "stock":
            continue
        if row.tradable:
            prices = "TX prices, proxy" if row.price_proxy else "own prices"
            lines.append(
                f"- {row.symbol} ({row.name_zh}): {row.point_value_twd:g}, {row.cost_model.commission_per_contract:g}, "
                f"{row.daily_data_since.isoformat() if row.daily_data_since else 'n/a'}; {prices}; listed "
                f"{row.listed_since.isoformat()}"
            )
        elif row.hedge_component:
            lines.append(
                f"- {row.symbol} ({row.name_zh}): bought puts only, through `{row.hedge_component}`; "
                f"daily data from {row.daily_data_since.isoformat()}"
            )
        else:
            lines.append(f"- {row.symbol} ({row.name_zh}): not available yet ({row.planned})")
    lines += [
        f"Futures: tax 0.002% both legs, 1 tick slippage/side, margin ~{INITIAL_MARGIN_RATIO:.1%} initial. "
        f"Capital {MIN_INITIAL_CAPITAL_TWD:,}-{MAX_INITIAL_CAPITAL_TWD:,} TWD (default 1,000,000); leverage at most "
        f"{LEVERAGE_CAP:g}x; integer contracts (every grid point must buy 1+ contract within "
        f"{MAX_ROUNDING_ERROR:.0%} of target leverage, else use MTX/TMF); initial margin at most "
        f"{MAX_MARGIN_UTILISATION:.0%} of equity; shorts only via futures.",
        f"- TWSE stocks / ETFs ({len(stocks())}): long only, paid in full (exposure <= 1x), unit = board lot of "
        f"{BOARD_LOT:,} shares (`unit: odd_lot` = 1 share), commission 0.1425%/side, sell tax 0.3% (ETF 0.1%), "
        "corporate actions booked from tw_corporate_actions, benchmark 0050 total return; data from each symbol's "
        "first Thalassa day (0050 2003-06-30, 00631L 2014-10-31, most stocks 2020-21). ETFs "
        + ",".join(row.symbol for row in stocks() if row.etf)
        + "; stocks "
        + ",".join(row.symbol for row in stocks() if not row.etf)
        + ".",
        "",
        "### Signal series (required by regime.ma_band, risk.vol_cap and the two ladders; never assumed)",
        "",
        "The same 200-day MA band over 2000-2014 ends 49% apart on the TX adjusted series vs the TAIEX price index "
        '(49 vs 33 regime switches). If the idea only says "台指", "TAIEX", "大盤" or a bare 均線, ask the owner which '
        "series; never pick silently.",
        f"- `self:adjusted`: the traded futures' back-adjusted continuous price, from "
        f"{TX_DAILY_DATA_SINCE.isoformat()}.",
    ]
    lines += [
        f"- `index:{index.name}`: {index.name_en} ({index.name_zh}), from {index.data_since.isoformat()}."
        for index in SIGNAL_INDICES.values()
    ]
    lines += ["", VIX_NOTE, "", "### What B CAN use (`*` = tunable)", ""]
    for slot in SLOTS:
        for component in components_for(slot):
            if "1d" in component.frequencies and component.name != "direction.dsl_rules":
                lines.append(_compact_component(component))
    lines += [
        "",
        "### What B CANNOT express (comes back unsupported)",
        "",
        "Not yet (planned): " + ", ".join(f"`{name}`" for name in PLANNED_COMPONENTS) + ".",
    ]
    lines += [f"- {text} (`{token}`)" for token, text in UNSUPPORTED_CAPABILITIES.items()]
    lines += [
        "- A daily indicator is never emulated on intraday bars (a 200-day MA is NOT SMA(200) on 5m bars): such an idea "
        "is a B strategy on frequency 1d (e.g. `regime.ma_band` with ma_days 200).",
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
