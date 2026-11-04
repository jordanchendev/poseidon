"""Spec v3 components: rebalance and roll."""

from __future__ import annotations

from typing import Annotated, Literal

import pandas as pd
from pydantic import Field, model_validator

from poseidon.pipeline.v3.component_common import ANY_FREQUENCY, DAILY, INTRADAY
from poseidon.pipeline.v3.registry import RebalanceComponent, RollComponent, StrictParams, pct, register


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
    # inside the class, so its fingerprint covers the schedule and the sentence
    DAY_ZH = {"first_trading_day": "第一個交易日", "last_trading_day": "最後一個交易日"}

    def init(self, params):
        return {"entered": False}

    def check(self, params, ctx):
        for sleeve in ctx.spec["sleeves"]:
            ref = sleeve["rebalance"]
            if ref["component"] != self.name or self.parse(ref.get("params") or {}) != params:
                return [("", f"every sleeve needs the same {self.name} (a switch sells and buys on one day)")]
        return []

    def decision_day(self, params, calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp:
        first = day.normalize().replace(day=1)
        sessions = calendar[calendar.searchsorted(first) : calendar.searchsorted(first + pd.offsets.MonthBegin(1))]
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
        day = self.DAY_ZH.get(params.day, f"第 {params.n} 個交易日（不足 {params.n} 個交易日的月份取最後一個）")
        unit = "口" if ctx.instrument.kind == "future" else ("股" if ctx.instrument.lot == 1 else "張")
        return (
            f"開始時先建立部位，之後每月只在當月{day}收盤後檢查一次（依交易所行事曆，颱風等臨時休市日不算），"
            f"換到目標部位、差 {params.min_change_units} {unit}以上就調整，其餘交易日不調整"
        )


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
