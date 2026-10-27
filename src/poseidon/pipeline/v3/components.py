"""Registered spec v3 components (REGISTRY_VERSION 3).

Parameters with bounds, ``$param``-able fields, zh-TW ``describe()``,
warm-up, capability tokens, component-level checks, and the bar-by-bar
behaviour each docstring states (design 2026-10-24 §3.3, §5). The behaviour
hooks (``step``, ``units``, ``should_trade``, ``roll_due``, ``weights``) see
only what :mod:`poseidon.pipeline.v3.runtime` hands them: a
:class:`~poseidon.pipeline.v3.runtime.SleeveView` of data up to the decision
close, never later, so every component is causal by construction.

REGISTRY_VERSION 2 holds what the M1 acceptance spec (§3.2) needs plus the
DSL v2 fast path (§3.4), ``leverage.drawdown_ladder`` and ``leverage.level_ladder``, the
``index:vix`` signal series (``instruments.SIGNAL_INDICES``), ``hedge.protective_put``,
``allocation.fixed_weights`` / ``inverse_vol`` / ``regime_switch``, ``sizing.fixed_notional`` and the
stock instrument kinds. Version 1 stopped at the M1 set; wave 4 added the rest and the bump to 2
(§3.6) followed, because ``registry_version`` is part of the pinned daily config hash. Version 3 adds
``rebalance.monthly`` (trade on one calendar day a month, any instrument). The rest of the Phase 1 list is
in ``registry.PLANNED_COMPONENTS``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from poseidon.pipeline.engine.options import HedgePolicy
from poseidon.pipeline.v3.instruments import (
    INDEX_ALIAS,
    INSTRUMENTS,
    INTRADAY_FREQUENCIES,
    SAME_INDEX_FAMILY,
    SIGNAL_INDICES,
    Instrument,
    signal_index,
)
from poseidon.pipeline.v3.registry import (
    AllocationComponent,
    CheckContext,
    DescribeContext,
    DirectionComponent,
    ExecutionComponent,
    HedgeComponent,
    LeverageComponent,
    RebalanceComponent,
    RollComponent,
    SizingComponent,
    StrictParams,
    fmt,
    pct,
    register,
)
from poseidon.pipeline.v3.spec import MAX_SLEEVES, SLEEVE_ID_PATTERN
from poseidon.strategies.dsl.v2.examples import tx_orb30_day

DAILY = frozenset({"1d"})
INTRADAY = INTRADAY_FREQUENCIES
ANY_FREQUENCY = DAILY | INTRADAY

# A signal series ref (owner decision 2026-10-24 afternoon #1; required, no default):
#   "<alias>:<field>" -- a price series of a declared instrument alias ("self" = the sleeve's own;
#                        "adjusted" = the back-adjusted continuous futures price);
#   "index:<name>"    -- a TWSE index (instruments.SIGNAL_INDICES: taiex price, taiex_tr total return) or the
#                        CBOE VIX ("vix", a US close read one calendar day late: no look-ahead).
SERIES_FIELDS = ("open", "high", "low", "close", "adjusted")
SERIES_REF_PATTERN = (
    r"^((self|[a-z][a-z0-9_]{0,31}):(" + "|".join(SERIES_FIELDS) + r")"
    rf"|{INDEX_ALIAS}:(" + "|".join(SIGNAL_INDICES) + r"))$"
)
SeriesRef = Annotated[str, Field(pattern=SERIES_REF_PATTERN)]
SERIES_ZH = {
    "adjusted": "調整後的連續價格",
    "close": "收盤價",
    "open": "開盤價",
    "high": "最高價",
    "low": "最低價",
}
INTERVAL_ZH = {"1m": "1 分鐘", "5m": "5 分鐘", "15m": "15 分鐘"}
LEVERAGE_MAX = 5.0  # parameter bound; the policy cap (validation.LEVERAGE_CAP) is checked on the whole chain


def series_text(ref: str, ctx: DescribeContext) -> str:
    """Which series a signal reads, in zh-TW: the index, or the instrument's (price source's) series."""

    index = signal_index(ref)
    if index is not None:
        return index.name_zh
    alias, field = ref.split(":")
    if alias in ("self", ctx.alias):
        source = ctx.instrument.price_proxy or ctx.instrument.symbol
        return f"{source} 期貨{SERIES_ZH[field]}" if ctx.instrument.kind == "future" else f"{source} {SERIES_ZH[field]}"
    return f"{alias} 的{SERIES_ZH[field]}"


def series_alias_errors(field: str, ref: str, ctx: CheckContext) -> list[tuple[str, str]]:
    alias = ref.split(":")[0]
    if alias == INDEX_ALIAS:
        if signal_index(ref) is None:
            names = ", ".join(f"{INDEX_ALIAS}:{name}" for name in SIGNAL_INDICES)
            return [(field, f"series {ref!r}: an index series is one of {names}")]
        return []
    if alias != "self" and alias not in ctx.aliases:
        return [(field, f"series {ref!r} names instrument alias {alias!r}, which is not declared in instruments")]
    return []


# --------------------------------------------------------------------------- direction


@register
class ConstantDirection(DirectionComponent):
    """Always long (+1) or always short (-1)."""

    name = "direction.constant"
    frequencies = DAILY
    capabilities = ("constant_direction", "always_long", "always_short")
    summary = "Always long or always short; the leverage chain then decides how much."

    class Params(StrictParams):
        side: Literal["long", "short"]

    example = {"side": "long"}

    def step(self, params, state, view):
        return 1 if params.side == "long" else -1

    def describe(self, params, ctx):
        return "一直做多" if params.side == "long" else "一直做空"


@register
class DslRulesDirection(DirectionComponent):
    """DSL v2 rules (``RuleStrategyV2``) as the direction: LONG/SHORT/CLOSE -> +1/-1/0 (§3.4).

    ``params`` is a DSL v2 document without ``symbol``/``market``/``interval``/
    ``params``: those come from the sleeve's instrument, the spec frequency and
    the spec-level ``params``. The DSL's own risk block (stops, targets,
    trailing, session rules) stays inside; the engine turns the initial
    SL/TP and trailing updates into protective orders it checks intrabar
    with the v1 gap-aware logic. Validated by ``validate_v2`` (strict DSL v2.1).
    The run binds the state to a ``RuleStrategyV2`` and its bars
    (``dsl_runtime.DslRulesState.bind``) before the first bar.
    """

    name = "direction.dsl_rules"
    frequencies = INTRADAY
    capabilities = (
        "dsl_v2_rules",
        "intraday_rules",
        "session_windows",
        "opening_range",
        "indicator_conditions",
        "donchian_channel",
        "stop_loss",
        "take_profit",
        "trailing_stop",
    )
    summary = (
        "DSL v2.1 intraday rules for TX/MTX (sessions, opening range, EMA/SMA/ATR/RSI, N-bar channels, "
        "exits in the DSL risk block); 1 fixed contract, fills at the signal bar's close."
    )
    example = {
        key: value for key, value in tx_orb30_day().items() if key not in ("symbol", "market", "interval", "params")
    }

    def parse(self, raw):
        return dict(raw)  # validated as a whole DSL v2 document by validation._check_dsl_rules

    def init(self, params):
        from poseidon.pipeline.v3.dsl_runtime import DslRulesState

        return DslRulesState()

    def step(self, params, state, view):
        return state.step(view)

    def protect(self, params, state):
        return state.pending_protect

    def describe(self, params: Mapping[str, Any], ctx):
        action_zh = {"long": "做多", "short": "做空", "close": "平倉"}
        rules = params.get("rules") or []
        listed = "、".join(
            f"{rule.get('id')} {action_zh.get(rule.get('action'), rule.get('action'))}" for rule in rules
        )
        return f"依 DSL v2 規則「{params.get('name')}」：{len(rules)} 條規則（{listed}），停損停利等出場寫在規則的 risk 區塊"


# --------------------------------------------------------------------------- leverage chain


@register
class BaseLeverage(LeverageComponent):
    """L0: a constant multiple of equity. Must be the first link of the chain."""

    name = "leverage.base"
    frequencies = DAILY
    capabilities = ("static_leverage",)
    summary = "Base leverage L0 = value x equity (first link of the leverage chain)."

    class Params(StrictParams):
        value: Annotated[float, Field(gt=0.0, le=LEVERAGE_MAX)]

    tunable = frozenset({"value"})
    example = {"value": 3.0}

    def nominal(self, params, leverage_in):
        return params.value

    def upper(self, params, leverage_in):
        return params.value

    def step(self, params, state, view, leverage_in):
        return params.value

    def describe(self, params, ctx):
        return f"基礎 {fmt(params.value)} 倍"


@register
class MaBandRegime(LeverageComponent):
    """Risk-off below MA x (1 - band), back on above MA x (1 + band); in between keep the previous state.

    MA = simple moving average of ``signal`` over ``ma_days`` rows of the
    signal series (including the latest row). Risk-on passes the incoming
    leverage through; risk-off replaces it with ``off_leverage``. The state
    starts at the first signal row dated on or after the first decision day:
    risk-on iff signal >= MA there (risk-on while the MA is undefined). Every
    later row moves the state at most once, in date order, so a signal with
    rows on days the instrument does not trade (an index) is still read row by
    row. The hysteresis state lives in the component state, never in the data.
    """

    name = "regime.ma_band"
    frequencies = DAILY
    capabilities = ("ma_regime", "hysteresis_band", "trend_filter")
    summary = (
        "Moving-average regime with a hysteresis band: below MA x (1 - band) switch to off_leverage, "
        "back above MA x (1 + band) restore the incoming leverage."
    )

    class Params(StrictParams):
        signal: SeriesRef
        ma_days: Annotated[int, Field(ge=20, le=400)] = 200
        band_pct: Annotated[float, Field(ge=0.0, le=0.10)] = 0.02
        off_leverage: Annotated[float, Field(ge=0.0, le=LEVERAGE_MAX)] = 0.5

    tunable = frozenset({"ma_days", "band_pct", "off_leverage"})
    example = {"signal": "index:taiex", "ma_days": 200, "band_pct": 0.02, "off_leverage": 0.5}

    def history(self, params):
        return params.ma_days

    def upper(self, params, leverage_in):
        return max(leverage_in, params.off_leverage)

    def init(self, params):
        # start: first decision day; rows: signal rows consumed; on: None until the start row;
        # switches: (signal row date, risk-on after the switch), for reports.
        return {"start": None, "rows": 0, "on": None, "switches": []}

    def step(self, params, state, view, leverage_in):
        series = view.series(params.signal)
        if state["start"] is None:
            state["start"] = np.datetime64(view.date, "ns")
        values, dates = series.values, series.dates
        n = params.ma_days
        for k in range(state["rows"], len(values)):
            close = float(values[k])
            ma = float(values[k - n + 1 : k + 1].mean()) if k + 1 >= n else math.nan
            if state["on"] is None:
                if dates[k] >= state["start"]:
                    state["on"] = not (math.isfinite(ma) and close < ma)
            elif math.isfinite(ma):
                if state["on"] and close < ma * (1 - params.band_pct):
                    state["on"] = False
                    state["switches"].append((dates[k], False))
                elif not state["on"] and close > ma * (1 + params.band_pct):
                    state["on"] = True
                    state["switches"].append((dates[k], True))
        state["rows"] = len(values)
        return leverage_in if state["on"] is not False else params.off_leverage

    def check(self, params, ctx):
        return series_alias_errors("signal", params.signal, ctx)

    def describe(self, params, ctx):
        back = f"{fmt(ctx.leverage_in)} 倍" if ctx.leverage_in is not None else "原倍數"
        band = pct(params.band_pct)
        return (
            f"{series_text(params.signal, ctx)}低於 {params.ma_days} 日均線 ×(1−{band}%) 時改成 "
            f"{fmt(params.off_leverage)} 倍，高於 ×(1+{band}%) 才回到 {back}"
        )


@register
class VolCap(LeverageComponent):
    """L = min(L_in, k / sigma), sigma = annualised realised volatility of ``signal``.

    sigma = sample standard deviation (ddof 1) of the last ``window_days``
    daily log returns of ``signal`` up to its latest row, times sqrt(252).
    Fewer than ``window_days + 1`` rows, or sigma = 0, leaves L_in unchanged.
    """

    name = "risk.vol_cap"
    frequencies = DAILY
    capabilities = ("volatility_cap", "realized_volatility")
    summary = "Volatility cap: leverage = min(incoming, k / annualised realised volatility over window_days)."

    class Params(StrictParams):
        signal: SeriesRef
        window_days: Annotated[int, Field(ge=5, le=120)] = 20
        k: Annotated[float, Field(ge=0.1, le=3.0)] = 0.75

    tunable = frozenset({"window_days", "k"})
    example = {"signal": "index:taiex", "window_days": 20, "k": 0.75}

    def history(self, params):
        return params.window_days + 1

    def check(self, params, ctx):
        return series_alias_errors("signal", params.signal, ctx)

    def step(self, params, state, view, leverage_in):
        values = view.series(params.signal).values
        window = params.window_days
        if len(values) < window + 1:
            return leverage_in
        sigma = float(np.diff(np.log(values[-(window + 1) :])).std(ddof=1)) * math.sqrt(252)
        if not math.isfinite(sigma) or sigma <= 0:
            return leverage_in
        return min(leverage_in, params.k / sigma)

    def describe(self, params, ctx):
        return f"再取 min(目前倍數, {fmt(params.k)} ÷ {series_text(params.signal, ctx)} {params.window_days} 日年化實現波動)"


# Both ladders add leverage when a signal series gets worse for a long position (it falls, or fear rises).
# They only ever raise the incoming leverage: output = max(incoming, the reached step's leverage). They are
# NOT risk.drawdown_brake (planned), which cuts leverage after the strategy's own equity drawdown.
LADDER_MAX_STEPS = 6
FLOAT_EPS = 1e-12


class DrawdownStep(StrictParams):
    """From a drawdown of ``drawdown_pct`` (a fraction of the rolling high) on, the leverage is ``leverage``."""

    drawdown_pct: Annotated[float, Field(gt=0.0, le=0.9)]
    leverage: Annotated[float, Field(gt=0.0, le=LEVERAGE_MAX)]


class LevelStep(StrictParams):
    """From a signal level of ``level`` on, the leverage is ``leverage``."""

    level: Annotated[float, Field(gt=0.0, le=1000.0)]
    leverage: Annotated[float, Field(gt=0.0, le=LEVERAGE_MAX)]


def ladder_problems(steps: list, key: str) -> list[str]:
    """Rungs must climb in both ``key`` and ``leverage`` (strictly): a deeper fall never buys less leverage."""

    problems = []
    for lower, upper in itertools.pairwise(steps):
        if getattr(upper, key) <= getattr(lower, key):
            problems.append(
                f"steps must have strictly increasing {key}: {fmt(getattr(lower, key))} then {fmt(getattr(upper, key))}"
            )
        if upper.leverage <= lower.leverage:
            problems.append(
                f"steps must have strictly increasing leverage: {fmt(lower.leverage)} then {fmt(upper.leverage)}"
            )
    return problems


def ladder_text(pairs: list[tuple[str, float]]) -> str:
    return "、".join(f"{at}→{fmt(leverage)} 倍" for at, leverage in pairs)


@register
class DrawdownLadder(LeverageComponent):
    """Leverage steps UP as ``signal`` falls from its rolling ``lookback_days``-row high (buy the dip).

    drawdown = 1 - signal / max(signal over the last ``lookback_days`` rows, latest row included). The
    highest step whose ``drawdown_pct`` the drawdown has reached sets the leverage: output =
    max(incoming leverage, that step's leverage); below the first step the incoming leverage passes
    through. With fewer than ``lookback_days`` rows nothing is reached. Without ``hold`` the step follows
    the current drawdown and moves down as the series recovers. With ``hold`` the highest step reached
    stays until the drawdown shrinks to ``release_drawdown_pct`` or less (then all steps clear), so it does
    not flap around a threshold. Rows are read one by one in date order, from the first row dated on or
    after the first decision day; the held step lives in the component state.
    This adds leverage when the PRICE falls. It is not ``risk.drawdown_brake`` (planned), which cuts
    leverage after the STRATEGY's own equity drawdown.
    """

    name = "leverage.drawdown_ladder"
    frequencies = DAILY
    capabilities = ("drawdown_ladder", "buy_the_dip", "scale_in_on_decline", "leverage_step_up")
    summary = (
        "Drawdown ladder (adds leverage when the PRICE falls): leverage steps UP as the signal falls from its rolling "
        "high, e.g. -10% -> 2x, -20% -> 3x; hold keeps the reached step until the drawdown recovers to "
        "release_drawdown_pct. Never lowers leverage. NOT the planned risk.drawdown_brake (cuts leverage after the "
        "strategy's own equity drawdown)."
    )

    class Params(StrictParams):
        signal: SeriesRef
        lookback_days: Annotated[int, Field(ge=20, le=500)] = 120
        steps: Annotated[
            list[DrawdownStep],
            Field(
                min_length=1,
                max_length=LADDER_MAX_STEPS,
                description=(
                    "1-6 rungs {drawdown_pct (fraction of the rolling high, 0-0.9), leverage}, both strictly "
                    "increasing, e.g. [{drawdown_pct: 0.1, leverage: 2}, {drawdown_pct: 0.2, leverage: 3}]"
                ),
            ),
        ]
        hold: bool = False
        release_drawdown_pct: Annotated[float, Field(ge=0.0, lt=0.9)] = 0.05

        @model_validator(mode="after")
        def _ladder(self):
            problems = ladder_problems(self.steps, "drawdown_pct")
            if self.hold and self.release_drawdown_pct >= self.steps[0].drawdown_pct:
                problems.append(
                    f"release_drawdown_pct {fmt(self.release_drawdown_pct)} must be below the first step's "
                    f"drawdown_pct {fmt(self.steps[0].drawdown_pct)}"
                )
            if problems:
                raise ValueError("; ".join(problems))
            return self

    tunable = frozenset({"lookback_days"})
    example = {
        "signal": "index:taiex",
        "lookback_days": 120,
        "steps": [{"drawdown_pct": 0.1, "leverage": 2.0}, {"drawdown_pct": 0.2, "leverage": 3.0}],
        "hold": True,
        "release_drawdown_pct": 0.05,
    }

    def history(self, params):
        return params.lookback_days

    def upper(self, params, leverage_in):
        return max(leverage_in, params.steps[-1].leverage)

    def check(self, params, ctx):
        return series_alias_errors("signal", params.signal, ctx)

    def init(self, params):
        # start: first decision day; rows: signal rows consumed; held: reached rung count (0 = none)
        return {"start": None, "rows": 0, "held": 0}

    def step(self, params, state, view, leverage_in):
        series = view.series(params.signal)
        if state["start"] is None:
            state["start"] = np.datetime64(view.date, "ns")
        values, dates = series.values, series.dates
        n = params.lookback_days
        for k in range(state["rows"], len(values)):
            if k + 1 < n or dates[k] < state["start"]:
                continue
            drawdown = 1.0 - float(values[k]) / float(values[k - n + 1 : k + 1].max())
            reached = sum(drawdown >= step.drawdown_pct - FLOAT_EPS for step in params.steps)
            if not params.hold:
                state["held"] = reached
            elif drawdown <= params.release_drawdown_pct:
                state["held"] = 0
            else:
                state["held"] = max(state["held"], reached)
        state["rows"] = len(values)
        if state["held"] == 0:
            return leverage_in
        return max(leverage_in, params.steps[state["held"] - 1].leverage)

    def describe(self, params, ctx):
        hold = (
            f"，已達的倍數維持到跌幅收斂至 {pct(params.release_drawdown_pct)}% 以內才放開"
            if params.hold
            else "，跌幅縮小倍數就跟著回降"
        )
        return (
            f"價格下跌時加碼：{series_text(params.signal, ctx)}自近 {params.lookback_days} 日高點回落達 "
            f"{ladder_text([(pct(step.drawdown_pct) + '%', step.leverage) for step in params.steps])}（只升不降，取目前倍數與階梯倍數的較大者；"
            f"這是價格跌時加碼，不是策略虧損時減碼）{hold}"
        )


@register
class LevelLadder(LeverageComponent):
    """Leverage steps UP as the LEVEL of ``signal`` reaches thresholds (e.g. VIX at or above 30 -> 2x).

    The highest step whose ``level`` the latest signal value has reached or passed sets the leverage:
    output = max(incoming leverage, that step's leverage); below the first level the incoming leverage
    passes through. Without ``hold`` the step follows the current level. With ``hold`` the highest step
    reached stays until the signal falls to ``release_below`` or lower (then all steps clear). Rows are read
    one by one in date order from the first row dated on or after the first decision day (the same rule
    as ``leverage.drawdown_ladder``), so a spike between two decision days is not missed.
    """

    name = "leverage.level_ladder"
    frequencies = DAILY
    capabilities = ("level_ladder", "signal_level_threshold", "vix_threshold", "leverage_step_up")
    summary = (
        "Level ladder (adds leverage when a signal LEVEL is high): leverage steps UP as the signal reaches "
        "thresholds, e.g. VIX >= 30 -> 2x, >= 40 -> 3x; hold keeps the step until the signal falls to "
        "release_below. Never lowers leverage."
    )

    class Params(StrictParams):
        signal: SeriesRef
        steps: Annotated[
            list[LevelStep],
            Field(
                min_length=1,
                max_length=LADDER_MAX_STEPS,
                description=(
                    "1-6 rungs {level, leverage}, both strictly increasing, "
                    "e.g. [{level: 30, leverage: 2}, {level: 40, leverage: 3}]"
                ),
            ),
        ]
        hold: bool = False
        release_below: Annotated[float, Field(ge=0.0, le=1000.0)] = 0.0

        @model_validator(mode="after")
        def _ladder(self):
            problems = ladder_problems(self.steps, "level")
            if self.hold and not 0.0 < self.release_below < self.steps[0].level:
                problems.append(
                    f"with hold, release_below ({fmt(self.release_below)}) must be above 0 and below the first "
                    f"step's level ({fmt(self.steps[0].level)})"
                )
            if problems:
                raise ValueError("; ".join(problems))
            return self

    example = {
        "signal": "index:vix",
        "steps": [{"level": 30.0, "leverage": 2.0}, {"level": 40.0, "leverage": 3.0}],
        "hold": True,
        "release_below": 22.0,
    }

    def upper(self, params, leverage_in):
        return max(leverage_in, params.steps[-1].leverage)

    def check(self, params, ctx):
        return series_alias_errors("signal", params.signal, ctx)

    def init(self, params):
        return {"start": None, "rows": 0, "held": 0}

    def step(self, params, state, view, leverage_in):
        series = view.series(params.signal)
        if state["start"] is None:
            state["start"] = np.datetime64(view.date, "ns")
        values, dates = series.values, series.dates
        for k in range(state["rows"], len(values)):
            if dates[k] < state["start"]:
                continue
            level = float(values[k])
            reached = sum(level >= step.level for step in params.steps)
            if not params.hold:
                state["held"] = reached
            elif level <= params.release_below:
                state["held"] = 0
            else:
                state["held"] = max(state["held"], reached)
        state["rows"] = len(values)
        if state["held"] == 0:
            return leverage_in
        return max(leverage_in, params.steps[state["held"] - 1].leverage)

    def describe(self, params, ctx):
        hold = (
            f"，已達的倍數維持到降至 {fmt(params.release_below)} 以下才放開"
            if params.hold
            else "，水位回落倍數就跟著回降"
        )
        return (
            f"{series_text(params.signal, ctx)}水位偏高時加碼：達 {ladder_text([(fmt(step.level), step.leverage) for step in params.steps])}"
            f"（只升不降，取目前倍數與階梯倍數的較大者）{hold}"
        )


# --------------------------------------------------------------------------- sizing


ROUNDING_EPS = 1e-9  # relative: keeps exact ratios such as 3e6 / 2e5 from flooring to 14 on float noise


def round_units(raw: float, rounding: str) -> int:
    """``floor`` rounds the magnitude down (never above the target leverage, long or short);
    ``nearest`` rounds half to even (Python ``round``, as the offline levtx backtest does)."""

    if rounding == "floor":
        return math.trunc(raw + math.copysign(ROUNDING_EPS * max(1.0, abs(raw)), raw))
    if rounding == "nearest":
        return round(raw)
    raise ValueError(f"unknown rounding {rounding!r}")


@register
class TargetLeverageSizing(SizingComponent):
    """units = round(equity x d x L x weight / (price x point value x lot)), floor by default (§5.3).

    The one sizing formula: the validator's feasibility checks and the engine run both call
    :meth:`units`. Units are contracts (board lots for stocks). Equity <= 0 sizes to 0.
    """

    name = "sizing.target_leverage"
    frequencies = DAILY
    basis = "equity"
    capabilities = ("equity_sizing", "integer_contracts", "target_leverage")
    summary = "Equity-based integer contracts: units = equity x direction x leverage / (price x point value)."

    class Params(StrictParams):
        rounding: Literal["floor", "nearest"] = "floor"

    example = {"rounding": "floor"}

    def units(self, params, *, equity, direction, leverage, weight, price, instrument: Instrument):
        if equity <= 0:
            return 0
        raw = equity * direction * leverage * weight / instrument.notional_twd(price)
        return round_units(raw, params.rounding)

    def describe(self, params, ctx):
        lot = f" × {ctx.instrument.lot}" if ctx.instrument.lot != 1 else ""
        rounding = "無條件捨去" if params.rounding == "floor" else "四捨五入"
        return f"口數 = 權益 × 倍數 ÷ (價格 × {fmt(ctx.instrument.point_value_twd)}{lot})，{rounding}"


@register
class FixedNotionalSizing(SizingComponent):
    """units = floor(equity x d x weight x pct_of_equity / 100 / (price x point value x lot)): a fixed share
    of equity, no leverage chain. The stock / ETF sizing (100 % = fully invested); futures may use it too."""

    name = "sizing.fixed_notional"
    frequencies = DAILY
    basis = "equity"
    uses_leverage = False
    capabilities = ("equity_sizing", "integer_contracts", "fixed_notional")
    summary = (
        "A fixed share of equity as notional, no leverage chain: "
        "units = equity x weight x pct_of_equity% / (price x point value x lot)."
    )

    class Params(StrictParams):
        pct_of_equity: Annotated[int, Field(ge=1, le=300)] = 100
        rounding: Literal["floor", "nearest"] = "floor"

    tunable = frozenset({"pct_of_equity"})
    example = {"pct_of_equity": 100, "rounding": "floor"}

    def exposure(self, params, leverage):
        return params.pct_of_equity / 100

    def units(self, params, *, equity, direction, leverage, weight, price, instrument: Instrument):
        if equity <= 0:
            return 0
        raw = equity * direction * weight * params.pct_of_equity / 100 / instrument.notional_twd(price)
        return round_units(raw, params.rounding)

    def describe(self, params, ctx):
        unit = "股" if ctx.instrument.lot == 1 else ("張" if ctx.instrument.kind == "stock" else "口")
        rounding = "無條件捨去" if params.rounding == "floor" else "四捨五入"
        return f"部位名目金額 = 權益的 {params.pct_of_equity}%，換成整數{unit}數（{rounding}）"


@register
class FixedContractsSizing(SizingComponent):
    """units = d x n (the v1 behaviour; PnL judged in TWD)."""

    name = "sizing.fixed_contracts"
    frequencies = ANY_FREQUENCY
    basis = "fixed"
    capabilities = ("fixed_contracts",)
    summary = "A fixed number of contracts n whenever the direction is non-zero (v1 behaviour)."

    class Params(StrictParams):
        n: Annotated[int, Field(ge=1, le=20)] = 1

    tunable = frozenset({"n"})
    example = {"n": 1}

    def units(self, params, *, equity, direction, leverage, weight, price, instrument):
        return direction * params.n

    def describe(self, params, ctx):
        return f"口數固定 {params.n} 口"


# --------------------------------------------------------------------------- rebalance


class _MinChange(StrictParams):
    min_change_units: Annotated[int, Field(ge=1, le=100)] = 1


@register
class DailyRebalance(RebalanceComponent):
    """At each daily decision, trade iff |target - current| >= min_change_units."""

    name = "rebalance.daily"
    frequencies = DAILY
    capabilities = ("daily_rebalance",)
    summary = "Re-size every trading day when the target differs by at least min_change_units."
    Params = _MinChange
    tunable = frozenset({"min_change_units"})
    example = {"min_change_units": 1}

    def should_trade(self, params, state, *, current_units, target_units, ctx):
        return abs(target_units - current_units) >= params.min_change_units

    def describe(self, params, ctx):
        return f"差 {params.min_change_units} 口以上才調整"


def _side(units: int) -> int:
    return (units > 0) - (units < 0)


@register
class MonthlyAtRollRebalance(RebalanceComponent):
    """Re-size only on a roll day (the held contract changed at this close), by at least
    ``min_change_units``; a change of side (enter, exit, reverse) trades at once."""

    name = "rebalance.monthly_at_roll"
    frequencies = DAILY
    capabilities = ("monthly_rebalance",)
    summary = "Re-size only when the futures position rolls (monthly); direction changes still trade at once."
    Params = _MinChange
    tunable = frozenset({"min_change_units"})
    example = {"min_change_units": 1}

    def check(self, params, ctx):
        if ctx.instrument is not None and ctx.instrument.kind != "future":
            return [("", "rebalance.monthly_at_roll needs a futures instrument (it re-sizes at the roll)")]
        return []

    def should_trade(self, params, state, *, current_units, target_units, ctx):
        if _side(target_units) != _side(current_units):
            return True
        return ctx.rolled_today and abs(target_units - current_units) >= params.min_change_units

    def describe(self, params, ctx):
        return f"只在每月換月時調整口數（差 {params.min_change_units} 口以上），方向改變時立即調整"


@register
class ThresholdBandRebalance(RebalanceComponent):
    """Trade iff |actual leverage - target leverage| / |target leverage| > band_pct, or the side changes.

    Actual leverage = current units x today's close x point value / equity; target = d x L x weight.
    """

    name = "rebalance.threshold_band"
    frequencies = DAILY
    capabilities = ("threshold_rebalance",)
    summary = "Re-size when the actual leverage drifts more than band_pct from the target (relative)."

    class Params(StrictParams):
        band_pct: Annotated[float, Field(ge=0.01, le=0.50)] = 0.10

    tunable = frozenset({"band_pct"})
    example = {"band_pct": 0.1}

    def should_trade(self, params, state, *, current_units, target_units, ctx):
        if target_units == current_units:
            return False
        if _side(target_units) != _side(current_units):
            return True
        drift = abs(ctx.actual_leverage - ctx.target_leverage) / abs(ctx.target_leverage)
        return drift > params.band_pct

    def describe(self, params, ctx):
        return f"實際槓桿偏離目標超過 {pct(params.band_pct)}% 才調整，方向改變時立即調整"


@register
class OnSignalChangeRebalance(RebalanceComponent):
    """Trade only when the direction changes (enter, exit, reverse); size is fixed at entry."""

    name = "rebalance.on_signal_change"
    frequencies = ANY_FREQUENCY
    capabilities = ("signal_rebalance",)
    summary = "Trade only when the direction changes (enter / exit / reverse); no re-sizing in between."
    example = {}

    def should_trade(self, params, state, *, current_units, target_units, ctx):
        return _side(target_units) != _side(current_units)

    def describe(self, params, ctx):
        return "訊號改變時才進出場"


def month_sessions(calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.DatetimeIndex:
    """The sessions of ``calendar`` in ``day``'s calendar month."""

    first = day.normalize().replace(day=1)
    return calendar[calendar.searchsorted(first) : calendar.searchsorted(first + pd.offsets.MonthBegin(1))]


MONTH_DAY_ZH = {"first_trading_day": "第一個交易日", "last_trading_day": "最後一個交易日"}


@register
class MonthlyRebalance(RebalanceComponent):
    """Trade only on the month's decision day, plus the run's first decision (the initial entry). There the
    target goes through when it differs by at least ``min_change_units``: a regime switch and a re-size to the
    sizing target alike. On every other day nothing is sent, so the position drifts with prices; the engine still
    books corporate actions, rolls futures and runs margin calls and forced liquidation.

    The decision day is a session of the run's trading calendar (``RebalanceContext.calendar``: the sessions
    that traded, then the official TWSE schedule): the first, the last or the ``n``-th session of the calendar
    month, the last one when the month has fewer than ``n``. An unscheduled closure (the 2026-10-23 typhoon) is
    not a session: when it takes the scheduled last day, the decision falls on the last day that traded, and
    ``n`` counts traded sessions only. The decision reads data through its own close; ``execution.next_open``
    fills it at the next session's open. The other components still step at every close (an allocation's
    hysteresis band sees every daily row); this rule only decides on which day their target becomes an order.
    """

    name = "rebalance.monthly"
    frequencies = DAILY
    capabilities = ("monthly_rebalance", "calendar_rebalance", "month_end_rebalance")
    summary = (
        "Trade once a month only, on the first, last or n-th trading day of the month (n only with nth_trading_day; "
        "TWSE / TAIFEX calendar, closures included): switch and re-size to the target there, hold in between; a "
        "run's first day enters."
    )

    class Params(StrictParams):
        day: Literal["last_trading_day", "first_trading_day", "nth_trading_day"] = "last_trading_day"
        n: Annotated[int, Field(ge=1, le=20)] = 1
        min_change_units: Annotated[int, Field(ge=1, le=100)] = 1

        @model_validator(mode="after")
        def _n(self):
            if self.n != 1 and self.day != "nth_trading_day":
                raise ValueError("n is only read with day nth_trading_day")
            return self

    tunable = frozenset({"min_change_units"})
    example = {"day": "last_trading_day", "min_change_units": 1}

    def init(self, params):
        return {"entered": False}

    def check(self, params, ctx):
        for sleeve in ctx.spec["sleeves"]:
            ref = sleeve["rebalance"]
            if ref["component"] != self.name or self.parse(ref.get("params") or {}) != params:
                return [("", f"every sleeve needs the same {self.name} (a switch sells and buys on one day)")]
        return []

    def decision_day(self, params, calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp:
        sessions = month_sessions(calendar, day)
        if day not in sessions:
            raise ValueError(f"{day.date()} is not a session of the run's trading calendar")
        if params.day == "first_trading_day":
            return sessions[0]
        if params.day == "last_trading_day":
            return sessions[-1]
        return sessions[min(params.n, len(sessions)) - 1]

    def should_trade(self, params, state, *, current_units, target_units, ctx):
        entry, state["entered"] = not state["entered"], True
        if not entry and ctx.date != self.decision_day(params, ctx.calendar, ctx.date):
            return False
        return abs(target_units - current_units) >= params.min_change_units

    def describe(self, params, ctx):
        day = MONTH_DAY_ZH.get(params.day, f"第 {params.n} 個交易日（不足 {params.n} 個交易日的月份取最後一個）")
        unit = "口" if ctx.instrument.kind == "future" else ("股" if ctx.instrument.lot == 1 else "張")
        return (
            f"開始時先建立部位，之後每月只在當月{day}收盤後檢查一次（依交易所行事曆，颱風等臨時休市日不算），"
            f"換到目標部位、差 {params.min_change_units} {unit}以上就調整，其餘交易日不調整"
        )


# --------------------------------------------------------------------------- roll


@register
class BeforeSettlementRoll(RollComponent):
    """Roll to the next month at the close ``days_before`` trading days before settlement (levtx ``build_r1``).

    Both legs pay costs and each contract settles at its own close, so the
    roll spread is never PnL; the signal series is back-adjusted from the same data.
    The engine executes the roll (``FuturesInput.roll_days_before = days_before``):
    after the close of t it holds the nearest expiry strictly after the trading day
    ``days_before`` sessions after t, so the position moves at t iff an expiry falls
    in (ref(t-1), ref(t)], which is what :meth:`roll_due` answers.
    """

    name = "roll.before_settlement"
    frequencies = DAILY
    capabilities = ("futures_roll",)
    summary = "Daily futures: roll to the next contract month days_before trading days before settlement."

    class Params(StrictParams):
        days_before: Annotated[int, Field(ge=0, le=5)] = 1

    example = {"days_before": 1}

    def roll_due(self, params, *, session_date, calendar):
        days = calendar.trading_days
        position = days.get_loc(session_date)
        last = len(days) - 1
        ref = days[min(position + params.days_before, last)]
        if position == 0:
            return False
        prev_ref = days[min(position - 1 + params.days_before, last)]
        return any(prev_ref < expiry <= ref for expiry in calendar.expiries)

    def describe(self, params, ctx):
        if params.days_before == 0:
            return "結算日當天收盤換月"
        return f"結算日前 {params.days_before} 個交易日換月"


@register
class AtSettlementRoll(RollComponent):
    """Intraday R1: a synthetic roll on settlement day, close the old month at 13:25, open the new at 13:30 (§5.5)."""

    name = "roll.at_settlement"
    frequencies = INTRADAY
    capabilities = ("futures_roll",)
    summary = "Intraday R1 futures: synthetic roll on settlement day (13:25 close -> 13:30 open, both legs costed)."
    example = {}

    def describe(self, params, ctx):
        return "結算日 13:25→13:30 合成換月"


# --------------------------------------------------------------------------- hedge


@register
class ProtectivePut(HedgeComponent):
    """Bought TXO monthly puts against the sleeve's long futures position (design §5.7).

    At each decision the engine sizes the hedge from the sleeve's final target units: puts =
    floor(cover_ratio x units x futures point value / 50), so at cover 1 one put covers five TMF
    (a flat or short position holds none). It keeps the held put until it has
    ``roll_days_before_expiry`` or fewer trading days left, then sells it and buys a new one:
    the ``front`` (or ``next``) monthly expiry with more trading days left than that, at the
    listed strike nearest to the futures close x (1 - moneyness_pct) that has a settlement price.
    A nearest strike more than 5 % from that target is ``hedge_unavailable`` (no new put that day;
    over 5 % of the hedged days fails the run). Valued and filled at daily settlement prices plus
    max(1 tick, 5 % of premium) slippage; a put held to expiry is cash-settled at the final
    settlement price. Behaviour lives in ``engine.options``.
    """

    name = "hedge.protective_put"
    frequencies = DAILY
    option_symbol = "TXO"
    capabilities = ("protective_put", "option_hedge", "tail_hedge")
    summary = (
        "Buy TXO monthly puts against the long futures position: strike nearest close x (1 - moneyness), "
        "roll before expiry, cover_ratio 1 = 1 put per 5 TMF; valued at daily settlement prices."
    )

    class Params(StrictParams):
        moneyness_pct: Annotated[float, Field(ge=0.0, le=0.20)] = 0.05
        tenor: Literal["front", "next"] = "front"
        roll_days_before_expiry: Annotated[int, Field(ge=1, le=15)] = 5
        cover_ratio: Annotated[float, Field(ge=0.25, le=1.5)] = 1.0

    tunable = frozenset({"moneyness_pct", "roll_days_before_expiry", "cover_ratio"})
    example = {"moneyness_pct": 0.05, "tenor": "front", "roll_days_before_expiry": 5, "cover_ratio": 1.0}

    def check(self, params, ctx):
        problems = []
        if ctx.instrument is not None and ctx.instrument.symbol not in SAME_INDEX_FAMILY:
            family = ", ".join(SAME_INDEX_FAMILY)
            problems.append(("", f"hedge.protective_put buys TXO puts and only hedges {family} sleeves"))
        direction = (ctx.sleeve or {}).get("direction") or {}
        if (
            direction.get("component") == "direction.constant"
            and (direction.get("params") or {}).get("side") == "short"
        ):
            problems.append(("", "a protective put hedges a long position; this sleeve is always short"))
        return problems

    def policy(self, params, *, name, underlying, chain):
        return HedgePolicy(
            name=name,
            underlying=underlying,
            chain=chain,
            moneyness=params.moneyness_pct,
            roll_days=params.roll_days_before_expiry,
            cover_ratio=params.cover_ratio,
            tenor=0 if params.tenor == "front" else 1,
            cp="P",
        )

    def describe(self, params, ctx):
        option = INSTRUMENTS[self.option_symbol]
        source = ctx.instrument.price_proxy or ctx.instrument.symbol
        per_put = option.point_value_twd / (ctx.instrument.point_value_twd * ctx.instrument.lot * params.cover_ratio)
        tenor = "近月" if params.tenor == "front" else "次近月"
        return (
            f"買進{option.name_zh}月選賣權保護多單：履約價取 {source} 期貨收盤價 ×(1−{pct(params.moneyness_pct)}%) "
            f"最近、有結算價的序列，{tenor}（剩餘超過 {params.roll_days_before_expiry} 個交易日的到期），"
            f"到期前 {params.roll_days_before_expiry} 個交易日換到下一個到期；口數 = {ctx.instrument.symbol} 口數 × "
            f"{fmt(ctx.instrument.point_value_twd * ctx.instrument.lot)} × {fmt(params.cover_ratio)} ÷ "
            f"{fmt(option.point_value_twd)}，無條件捨去（每 {fmt(per_put)} 口 {ctx.instrument.symbol} 一口賣權）；"
            "以每日結算價估值和成交（另加滑價），到期以最後結算價現金結算"
        )


# --------------------------------------------------------------------------- execution


@register
class NextOpenExecution(ExecutionComponent):
    """Decide after every instrument's close; fill at the next tradable bar's open + slippage (§5.1, §12 #1)."""

    name = "execution.next_open"
    frequencies = DAILY
    fill = "next_open"
    capabilities = ("next_open_execution",)
    summary = "Decide after the daily close, fill at the next session's open (TX family: 15:00 night open)."
    example = {}

    def describe(self, params, ctx):
        return "每個交易日收盤後決策，次一交易時段開盤成交"


@register
class SignalCloseExecution(ExecutionComponent):
    """Fill at the signal bar's close (the v1 runner's convention; DSL v2 parity only, §3.4)."""

    name = "execution.signal_close"
    frequencies = INTRADAY
    fill = "signal_close"
    capabilities = ("signal_close_execution",)
    summary = "Fill at the signal bar's close; only for direction.dsl_rules (DSL v2 parity)."
    example = {}

    def describe(self, params, ctx):
        return f"每根 {INTERVAL_ZH.get(ctx.frequency, ctx.frequency)} K 棒收盤時決策，以該根收盤價成交"


# --------------------------------------------------------------------------- allocation


@register
class SingleAllocation(AllocationComponent):
    """Exactly one sleeve with weight 1."""

    name = "allocation.single"
    frequencies = ANY_FREQUENCY
    capabilities = ("single_instrument",)
    summary = "One sleeve, weight 1."
    example = {}

    def weights(self, params, state, view, sleeves):
        return {sleeves[0]: 1.0}

    def describe(self, params, ctx):
        return "單一部位"


def _equal(sleeves: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(sleeves, 1.0 / len(sleeves))


@register
class FixedWeightsAllocation(AllocationComponent):
    """Constant weight per sleeve id; weights sum to at most 1 (each sleeve's own sizing applies on top)."""

    name = "allocation.fixed_weights"
    frequencies = DAILY
    min_sleeves = 2
    max_sleeves = MAX_SLEEVES
    capabilities = ("multi_sleeve", "fixed_weights")
    summary = "Constant weights per sleeve id (sum <= 1); each sleeve's sizing is scaled by its weight."

    class Params(StrictParams):
        weights: Annotated[
            dict[Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)], Annotated[float, Field(gt=0.0, le=1.0)]],
            Field(min_length=2, max_length=MAX_SLEEVES),
        ]

        @model_validator(mode="after")
        def _sum(self):
            if sum(self.weights.values()) > 1.0 + 1e-9:
                raise ValueError(f"weights sum to {sum(self.weights.values()):g}; at most 1")
            return self

    example = {"weights": {"core": 0.6, "satellite": 0.4}}

    def check(self, params, ctx):
        ids = [sleeve["id"] for sleeve in ctx.spec["sleeves"]]
        if sorted(params.weights) != sorted(ids):
            return [("weights", f"weights must name every sleeve id exactly once: {', '.join(ids)}")]
        return []

    def weights(self, params, state, view, sleeves):
        return dict(params.weights)

    def weight_bounds(self, params, sleeves):
        return {sleeve: (params.weights[sleeve], params.weights[sleeve]) for sleeve in sleeves}

    def describe(self, params, ctx):
        return "固定權重：" + "、".join(f"{sleeve} {pct(weight)}%" for sleeve, weight in params.weights.items())


@register
class InverseVolAllocation(AllocationComponent):
    """Weight each sleeve by 1 / its realised volatility (annualised, ``window_days`` log returns of its
    own adjusted series), normalised to sum 1; equal weights until every sleeve has a defined volatility."""

    name = "allocation.inverse_vol"
    frequencies = DAILY
    min_sleeves = 2
    max_sleeves = MAX_SLEEVES
    capabilities = ("multi_sleeve", "inverse_volatility", "realized_volatility")
    summary = "Weights proportional to 1 / realised volatility of each sleeve's own series over window_days, sum 1."

    class Params(StrictParams):
        window_days: Annotated[int, Field(ge=10, le=120)] = 20

    tunable = frozenset({"window_days"})
    example = {"window_days": 20}

    def history(self, params):
        return params.window_days

    def weights(self, params, state, view, sleeves):
        inverse = {}
        for sleeve in sleeves:
            values = view.sleeve_series(sleeve, "adjusted").values
            if len(values) < params.window_days + 1:
                return _equal(sleeves)
            sigma = float(np.diff(np.log(values[-(params.window_days + 1) :])).std(ddof=1)) * math.sqrt(252)
            if not math.isfinite(sigma) or sigma <= 0:
                return _equal(sleeves)
            inverse[sleeve] = 1.0 / sigma
        total = sum(inverse.values())
        return {sleeve: value / total for sleeve, value in inverse.items()}

    def weight_bounds(self, params, sleeves):
        return dict.fromkeys(sleeves, (1.0 / len(sleeves), 1.0))

    def describe(self, params, ctx):
        return f"權重與各部位 {params.window_days} 日年化實現波動成反比（合計 100%）"


@register
class RegimeSwitchAllocation(AllocationComponent):
    """All-in on sleeve ``on`` while the signal is above its MA band, all-in on ``off`` below it; the
    hysteresis is ``regime.ma_band``'s (same band rule and start rule) with 1 / 0 as the two states."""

    name = "allocation.regime_switch"
    frequencies = DAILY
    min_sleeves = 2
    max_sleeves = 2
    capabilities = ("multi_sleeve", "ma_regime", "hysteresis_band", "regime_switch")
    summary = (
        "Two sleeves: all weight on sleeve `on` while signal > MA x (1 + band), on sleeve `off` once < MA x (1 - band) "
        "(regime.ma_band hysteresis)."
    )

    class Params(StrictParams):
        signal: SeriesRef
        ma_days: Annotated[int, Field(ge=20, le=400)] = 200
        band_pct: Annotated[float, Field(ge=0.0, le=0.10)] = 0.02
        on: Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)]
        off: Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)]

    tunable = frozenset({"ma_days", "band_pct"})
    example = {"signal": "index:taiex", "ma_days": 200, "band_pct": 0.02, "on": "risk_on", "off": "risk_off"}

    @staticmethod
    def _regime(params):
        return MaBandRegime.Params(
            signal=params.signal, ma_days=params.ma_days, band_pct=params.band_pct, off_leverage=0.0
        )

    def history(self, params):
        return params.ma_days

    def init(self, params):
        return MaBandRegime().init(self._regime(params))

    def check(self, params, ctx):
        problems = []
        ids = [sleeve["id"] for sleeve in ctx.spec["sleeves"]]
        for field in ("on", "off"):
            if getattr(params, field) not in ids:
                problems.append((field, f"{getattr(params, field)!r} is not a sleeve id ({', '.join(ids)})"))
        if params.on == params.off:
            problems.append(("off", "on and off must be different sleeves"))
        if params.signal.startswith("self:"):
            problems.append(("signal", "an allocation has no own instrument: name an alias or an index series"))
        return problems + series_alias_errors("signal", params.signal, ctx)

    def weights(self, params, state, view, sleeves):
        on = MaBandRegime().step(self._regime(params), state, view, 1.0) == 1.0
        return {params.on: 1.0 if on else 0.0, params.off: 0.0 if on else 1.0}

    def describe(self, params, ctx):
        band = pct(params.band_pct)
        return (
            f"{series_text(params.signal, ctx)}高於 {params.ma_days} 日均線 ×(1+{band}%) 時全部押 {params.on}，"
            f"低於 ×(1−{band}%) 時全部換成 {params.off}"
        )
