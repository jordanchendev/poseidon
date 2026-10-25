"""Registered spec v3 components (REGISTRY_VERSION 1).

Parameters with bounds, ``$param``-able fields, zh-TW ``describe()``,
warm-up, capability tokens, component-level checks, and the bar-by-bar
behaviour each docstring states (design 2026-10-24 §3.3, §5). The behaviour
hooks (``step``, ``units``, ``should_trade``, ``roll_due``, ``weights``) see
only what :mod:`poseidon.pipeline.v3.runtime` hands them: a
:class:`~poseidon.pipeline.v3.runtime.SleeveView` of data up to the decision
close, never later, so every component is causal by construction.

REGISTRY_VERSION 1 holds what the M1 acceptance spec (§3.2) needs plus the
DSL v2 fast path (§3.4). The rest of the Phase 1 list is in
``registry.PLANNED_COMPONENTS``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from poseidon.pipeline.v3.instruments import INDEX_ALIAS, INTRADAY_FREQUENCIES, SIGNAL_INDICES, Instrument, signal_index
from poseidon.pipeline.v3.registry import (
    AllocationComponent,
    CheckContext,
    DescribeContext,
    DirectionComponent,
    ExecutionComponent,
    LeverageComponent,
    RebalanceComponent,
    RollComponent,
    SizingComponent,
    StrictParams,
    fmt,
    pct,
    register,
)
from poseidon.strategies.dsl.v2.examples import tx_orb30_day

DAILY = frozenset({"1d"})
INTRADAY = INTRADAY_FREQUENCIES
ANY_FREQUENCY = DAILY | INTRADAY

# A signal series ref (owner decision 2026-10-24 afternoon #1; required, no default):
#   "<alias>:<field>" -- a price series of a declared instrument alias ("self" = the sleeve's own;
#                        "adjusted" = the back-adjusted continuous futures price);
#   "index:<name>"    -- a TWSE index (instruments.SIGNAL_INDICES: taiex price, taiex_tr total return).
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
        return f"{source} 期貨{SERIES_ZH[field]}" if ctx.instrument.kind == "future" else SERIES_ZH[field]
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
