"""Spec v3 components: direction and the leverage chain (REGISTRY_VERSION 3)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from poseidon.pipeline.v3.component_common import (
    DAILY,
    INTRADAY,
    LEVERAGE_MAX,
    SeriesRef,
    series_alias_errors,
    series_text,
)
from poseidon.pipeline.v3.registry import DirectionComponent, LeverageComponent, StrictParams, fmt, pct, register
from poseidon.strategies.dsl.v2.examples import tx_orb30_day


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
