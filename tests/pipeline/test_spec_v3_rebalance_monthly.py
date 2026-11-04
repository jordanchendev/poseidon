"""``rebalance.monthly``: one decision day a month on the run's trading calendar (REGISTRY_VERSION 3).

Schedule by hand (month ends on weekends and holidays, the 2026-10-23 typhoon, first / n-th session),
the trade rule, then the compiled 00631L / 0050 rotation through the engine: switches and re-sizes only at
the month's decision day, filled at the next open; corporate actions and forced liquidation still happen
in between; a decision on day D never sees D + 1.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poseidon.pipeline.engine import EngineConfig, FuturesInput, run, truncation_audit
from poseidon.pipeline.engine.data import stock_input
from poseidon.pipeline.refdata.engine_input import trading_calendar
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.v3.compiler import engine_config, plan, strategy_factory
from poseidon.pipeline.v3.describe import narrative
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050
from poseidon.pipeline.v3.instruments import INSTRUMENTS, contract_spec
from poseidon.pipeline.v3.registry import REGISTRY, DescribeContext
from poseidon.pipeline.v3.runtime import Conventions, RebalanceContext
from poseidon.pipeline.v3.spec import substitute
from poseidon.pipeline.v3.validation import validate_spec_v3
from poseidon.pipeline.v3.validation_types import FeasibilityContext

from .engine.support import FAR, TMF_PLAIN, flat_bars
from .spec_support import resolved_spec

MONTHLY = REGISTRY["rebalance.monthly"]
T = pd.Timestamp


def days(*isos: str) -> list[pd.Timestamp]:
    return [T(iso) for iso in isos]


def decision(calendar, on: str, **params) -> pd.Timestamp:
    return MONTHLY.decision_day(MONTHLY.parse(params), pd.DatetimeIndex(calendar), T(on))


# October 2026 on the TWSE schedule: 10-09 (National Day, observed) and 10-26 (Retrocession Day, observed)
# are scheduled holidays; 10-23 closed for a typhoon, which no schedule lists.
SCHEDULE = MarketSchedule.from_frame(
    pd.DataFrame({"date": days("2026-01-01", "2026-10-09", "2026-10-26"), "closed": [True] * 3})
)
TYPHOON = T("2026-10-23")


def october(*, closure_known: bool) -> pd.DatetimeIndex:
    """The engine calendar as ``shadow_paper_daily`` builds it: sessions that traded through ``through``,
    then the official schedule. Before the closure is in the data, 10-23 is a scheduled session."""

    through = T("2026-10-27") if closure_known else T("2026-10-22")
    traded = SCHEDULE.open_days(T("2026-09-30"), through)
    traded = traded[traded != TYPHOON]
    return trading_calendar(traded, SCHEDULE, through, horizon_days=60)


# ----------------------------------------------------------------------------- the schedule


def test_month_ends_on_weekends_and_holidays():
    weekdays = pd.bdate_range("2026-05-01", "2026-06-30")
    assert decision(weekdays, "2026-05-04") == T("2026-05-29")  # 05-31 is a Sunday
    assert decision(weekdays, "2026-05-04", day="first_trading_day") == T("2026-05-01")
    june = weekdays[weekdays != T("2026-06-30")]  # a holiday on the last weekday
    assert decision(june, "2026-06-01") == T("2026-06-29")


def test_october_2026_with_the_scheduled_holidays_and_the_typhoon():
    known, before = october(closure_known=True), october(closure_known=False)
    assert TYPHOON not in known and TYPHOON in before
    assert T("2026-10-09") not in known and T("2026-10-26") not in known
    for calendar in (known, before):
        assert decision(calendar, "2026-10-01") == T("2026-10-30")
        assert decision(calendar, "2026-10-12", day="first_trading_day") == T("2026-10-01")
    # n counts the sessions that traded: the typhoon pushes the 16th session from 10-23 to 10-27
    assert decision(known, "2026-10-01", day="nth_trading_day", n=16) == T("2026-10-27")
    assert decision(before, "2026-10-01", day="nth_trading_day", n=16) == TYPHOON
    assert decision(known, "2026-11-02") == T("2026-11-30")  # the scheduled part of the calendar


def test_a_closed_last_day_moves_the_decision_to_the_last_day_that_traded():
    september = pd.bdate_range("2026-09-01", "2026-09-30")
    assert decision(september, "2026-09-01") == T("2026-09-30")
    closed = september[september != T("2026-09-30")]  # an unscheduled closure on the last weekday
    assert decision(closed, "2026-09-01") == T("2026-09-29")


def test_nth_session_past_the_month_end_is_the_last_session_and_n_needs_nth():
    february = pd.bdate_range("2026-02-02", "2026-02-27")
    february = february[(february < T("2026-02-12")) | (february > T("2026-02-20"))]  # Lunar New Year
    assert len(february) == 13
    assert decision(february, "2026-02-02", day="nth_trading_day", n=20) == T("2026-02-27")
    assert decision(february, "2026-02-02", day="nth_trading_day", n=2) == T("2026-02-03")
    with pytest.raises(ValueError, match="n is only read with day nth_trading_day"):
        MONTHLY.parse({"day": "last_trading_day", "n": 3})
    with pytest.raises(ValueError, match="not a session of the run's trading calendar"):
        decision(february, "2026-02-12")


# ----------------------------------------------------------------------------- the trade rule


def trades(state, on: str, current: int, target: int, calendar, **params) -> bool:
    ctx = RebalanceContext(T(on), False, 1.0, 1.0, pd.DatetimeIndex(calendar))
    return MONTHLY.should_trade(MONTHLY.parse(params), state, current_units=current, target_units=target, ctx=ctx)


def test_trades_at_the_first_decision_then_only_on_decision_days():
    calendar = pd.bdate_range("2026-01-05", "2026-03-31")
    state = MONTHLY.init(MONTHLY.parse({}))
    assert trades(state, "2026-01-14", 0, 9, calendar)  # the initial entry, mid-month
    assert not trades(state, "2026-01-15", 9, 0, calendar)  # a switch waits for the decision day
    assert not trades(state, "2026-01-30", 9, 9, calendar)  # decision day, nothing to send
    assert trades(state, "2026-01-30", 9, 0, calendar)
    assert trades(state, "2026-01-30", 9, 10, calendar)  # re-size to the target on the decision day
    assert not trades(state, "2026-01-30", 9, 10, calendar, min_change_units=2)
    assert not trades(state, "2026-02-02", 9, 12, calendar)


# ----------------------------------------------------------------------------- spec, validation, narrative


def monthly_rotation(**rebalance) -> dict:
    """The M3 rotation with ``rebalance.monthly`` on both sleeves, every $param bound."""

    doc = substitute(etf_rotation_00631l_0050(), {"ma_days": 20, "band": 0.02})
    doc["params"] = {}
    for sleeve in doc["sleeves"]:
        sleeve["rebalance"] = {"component": "rebalance.monthly", "params": dict(rebalance)}
    return doc


def errors(doc) -> list[str]:
    result = validate_spec_v3(doc, feasibility=FeasibilityContext(choose_start=True))
    return [issue.text() for issue in (*result.errors, *result.unsupported)]


def test_validation_wants_the_same_monthly_rule_on_every_sleeve():
    def template(*refs: dict) -> dict:
        doc = etf_rotation_00631l_0050()
        for sleeve, ref in zip(doc["sleeves"], refs, strict=True):
            sleeve["rebalance"] = ref
        return doc

    monthly = {"component": "rebalance.monthly", "params": {}}
    assert errors(template(monthly, {"component": "rebalance.monthly", "params": {"day": "last_trading_day"}})) == []
    mixed = errors(template(monthly, {"component": "rebalance.on_signal_change", "params": {}}))
    assert [e for e in mixed if "every sleeve needs the same" in e] == [
        "sleeves.0.rebalance: every sleeve needs the same rebalance.monthly (a switch sells and buys on one day)"
    ]
    first = {"component": "rebalance.monthly", "params": {"day": "first_trading_day"}}
    assert len([e for e in errors(template(monthly, first)) if "every sleeve needs the same" in e]) == 2


def test_the_narrative_says_once_a_month():
    text = narrative(monthly_rotation())
    assert "開始時先建立部位，之後每月只在當月最後一個交易日收盤後檢查一次" in text
    assert "颱風等臨時休市日不算" in text and "差 1 張以上就調整，其餘交易日不調整" in text
    tmf_text = MONTHLY.describe(
        MONTHLY.parse({"day": "nth_trading_day", "n": 5, "min_change_units": 2}),
        DescribeContext("1d", INSTRUMENTS["TMF"]),
    )
    assert "當月第 5 個交易日（不足 5 個交易日的月份取最後一個）" in tmf_text and "差 2 口以上" in tmf_text


# ----------------------------------------------------------------------------- through the engine

DATES = pd.bdate_range("2026-01-05", "2026-04-30")
START = T("2026-02-02")  # 20 sessions of warm-up before it


def stock(symbol: str, closes, actions: pd.DataFrame | None = None):
    frame = pd.DataFrame({"date": DATES, "open": closes, "high": closes, "low": closes, "close": closes})
    return stock_input(contract_spec(INSTRUMENTS[symbol]), frame, actions)


def index_dropping_on(drop: str | None) -> pd.DataFrame:
    """TAIEX at 100, falling to 90 (below the 20-day MA band) from ``drop`` on."""

    level = [90.0 if drop is not None and d >= T(drop) else 100.0 for d in DATES]
    return pd.DataFrame({"close": level}, index=DATES)


def rotate(spec: dict, aux: pd.DataFrame, base_actions: pd.DataFrame | None = None, base_close=None):
    spec_plan = plan(spec)
    inputs = {
        "lev": stock("00631L", [41.0] * len(DATES)),
        "base": stock("0050", base_close or [101.0] * len(DATES), base_actions),
    }
    config = engine_config(spec_plan, start=START)
    factory = strategy_factory(spec_plan, Conventions())
    return run(factory(), inputs, config, aux={"taiex": aux}), (factory, inputs, config, {"taiex": aux})


def fill_days(result, instrument: str | None = None, reason: str = "rebalance") -> list[pd.Timestamp]:
    fills = result.fills[result.fills["reason"] == reason]
    if instrument is not None:
        fills = fills[fills["instrument"] == instrument]
    return sorted({T(d) for d in fills["date"]})


def daily_switch_rotation() -> dict:
    doc = monthly_rotation()
    for sleeve in doc["sleeves"]:
        sleeve["rebalance"] = {"component": "rebalance.on_signal_change", "params": {}}
    return doc


def test_the_switch_waits_for_the_month_end_and_fills_at_the_next_open():
    aux = index_dropping_on("2026-02-16")
    monthly, _ = rotate(monthly_rotation(), aux)
    daily, _ = rotate(daily_switch_rotation(), aux)
    # entry decided 02-02, filled 02-03; the regime turns off at the 02-16 close
    assert fill_days(daily) == days("2026-02-03", "2026-02-17")
    assert fill_days(monthly) == days("2026-02-03", "2026-03-02")  # decided at 02-27, the last February session
    swap = monthly.fills[monthly.fills["date"] == T("2026-03-02")]
    assert swap["instrument"].tolist() == ["lev", "base"] and swap["units"].iloc[0] < 0 < swap["units"].iloc[1]
    held = monthly.held_units("lev")
    assert (held[T("2026-02-03") : T("2026-02-27")] == held[T("2026-02-03")]).all() and held[T("2026-03-02")] == 0


def test_a_cash_dividend_is_booked_at_once_and_reinvested_on_the_decision_day():
    """All in 0050 (index below its band from the start). The 02-11 dividend is paid in cash that day; the
    monthly check re-sizes to the target (one more lot) at the February decision, the daily switch never."""

    closes = [101.0 if d < T("2026-02-11") else 91.0 for d in DATES]
    dividend = pd.DataFrame(
        [
            {
                "date": T("2026-02-11"),
                "symbol": "0050",
                "action_type": "cash_dividend",
                "prev_close": 101.0,
                "reference_price": 91.0,
                "cash_dividend": 10.0,
            }
        ]
    )
    aux = index_dropping_on("2026-01-20")  # MA 95 at the 02-02 start: the regime starts off
    monthly, _ = rotate(monthly_rotation(), aux, dividend, closes)
    daily, _ = rotate(daily_switch_rotation(), aux, dividend, closes)
    assert fill_days(daily) == days("2026-02-03")
    assert fill_days(monthly) == days("2026-02-03", "2026-03-02")
    held = monthly.held_units("base")
    assert held[T("2026-02-11")] == held[T("2026-02-03")] == 9  # the dividend is cash, the units stay
    assert held[T("2026-03-02")] == 10
    equity = monthly.daily_pnl.set_index("date")["equity"]
    assert equity[T("2026-02-11")] == pytest.approx(equity[T("2026-02-10")], abs=1.0)  # paid out, not lost


def test_forced_liquidation_happens_between_decision_days():
    closes = [20000.0] * 40
    frame = flat_bars(closes)
    frame.loc[frame.index[30], "low"] = 15600.0  # 22 TMF at 4.5x: equity / initial margin 0.13 at the low
    spec = resolved_spec(ma_days=20, vol_k=None, base=4.5, rebalance=("rebalance.monthly", {}))
    futures = {"core": FuturesInput(spec=TMF_PLAIN, bars=frame, expiries={"A": FAR})}
    result = run(strategy_factory(plan(spec))(), futures, EngineConfig(execution="next_open", start=T("2026-02-02")))
    assert fill_days(result, reason="forced_liquidation") == days("2026-02-16")  # a Monday mid-month
    assert fill_days(result) == days("2026-02-03")
    assert result.held_units("core")[T("2026-02-13")] == 22 and result.held_units("core")[T("2026-02-16")] == 0


def test_a_decision_on_day_d_does_not_read_day_d_plus_1():
    """The index breaks below its band at the close of 03-02, the session after the February decision day
    (02-27). That decision must keep 00631L; the switch is March's decision (03-31), filled 04-01. The leak
    control dates every index row one session early (the D+1 row is visible at D) and does switch at 02-27,
    so the assertion would catch a decision that reads D + 1."""

    aux = index_dropping_on("2026-03-02")
    result, (factory, inputs, config, aux_frames) = rotate(monthly_rotation(), aux)
    assert fill_days(result) == days("2026-02-03", "2026-04-01")

    leaked = aux.shift(-1).ffill()
    leaked_result, _ = rotate(monthly_rotation(), leaked)
    assert fill_days(leaked_result) == days("2026-02-03", "2026-03-02")

    decision_days = pd.DatetimeIndex(result.decisions["date"].unique())
    audit = truncation_audit(factory, inputs, config, aux=aux_frames, n_cuts=len(decision_days) - 1)
    assert audit.ok, audit.mismatches
    assert {T("2026-02-27"), T("2026-03-31")} <= set(audit.cuts)
