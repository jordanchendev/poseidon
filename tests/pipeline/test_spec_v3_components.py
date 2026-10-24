"""Hand-computed cases for the daily components' behaviour (design 2026-10-24 §3.3, §5.3, §5.5)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import EngineConfig, FuturesInput, MarketView, run
from poseidon.pipeline.engine.view import AuxView, SeriesBuffer, SeriesView, _Buffer
from poseidon.pipeline.v3.compiler import plan, strategy_factory
from poseidon.pipeline.v3.instruments import INSTRUMENTS, REFERENCE_INDEX_LEVEL
from poseidon.pipeline.v3.registry import REGISTRY
from poseidon.pipeline.v3.runtime import Conventions, RebalanceContext, RollCalendar, Series, SleeveView

from .engine.support import TMF, bars, day, synthetic_contracts
from .spec_support import resolved_spec

DATES = pd.bdate_range("2026-01-05", periods=40)
E0 = 1_000_000.0


class FakeView:
    """A SleeveView stand-in: ``series()`` returns the rows up to ``n``."""

    def __init__(self, values, n=None, dates=DATES):
        self.values = np.asarray(values, dtype=float)
        self.n = len(self.values) if n is None else n
        self.all_dates = dates[: len(self.values)].values

    @property
    def date(self):
        return pd.Timestamp(self.all_dates[self.n - 1])

    def series(self, ref):
        return Series(self.all_dates[: self.n], self.values[: self.n])


def parse(name, **params):
    component = REGISTRY[name]
    return component, component.parse(params)


# ----------------------------------------------------------------------------- direction, base leverage


def test_constant_direction_and_base_leverage():
    component, params = parse("direction.constant", side="long")
    assert component.step(params, None, FakeView([1.0])) == 1
    component, params = parse("direction.constant", side="short")
    assert component.step(params, None, FakeView([1.0])) == -1
    component, params = parse("leverage.base", value=2.5)
    assert component.step(params, None, FakeView([1.0]), 0.0) == 2.5


# ----------------------------------------------------------------------------- regime.ma_band

# 20 rows at 100 (MA 100), then 97 (MA 99.85, below 99.85 x 0.98 = 97.853 -> off),
# 99 (MA 99.80, inside the band -> stay off), 103 (MA 99.95, above 99.95 x 1.02 = 101.949 -> on)
BAND_SERIES = [100.0] * 20 + [97.0, 99.0, 103.0]


def run_ma_band(values, decision_rows, **params):
    component, p = parse("regime.ma_band", ma_days=20, band_pct=0.02, off_leverage=0.5, **params)
    state = component.init(p)
    out = [component.step(p, state, FakeView(values, n + 1), 3.0) for n in decision_rows]
    return out, state


def test_ma_band_hysteresis_by_hand():
    out, state = run_ma_band(BAND_SERIES, [19, 20, 21, 22])
    assert out == [3.0, 0.5, 0.5, 3.0]
    assert state["switches"] == [(DATES.values[20], False), (DATES.values[22], True)]


def test_ma_band_reads_every_signal_row_even_between_decisions():
    out, state = run_ma_band(BAND_SERIES, [19, 22])  # rows 20-21 arrive with the second decision
    assert out == [3.0, 3.0]
    assert state["switches"] == [(DATES.values[20], False), (DATES.values[22], True)]


def test_ma_band_starts_off_below_the_ma_even_inside_the_band():
    out, state = run_ma_band([100.0] * 20 + [99.0], [20])  # MA 99.95: 99 is inside the band but below the MA
    assert out == [0.5] and state["switches"] == []


def test_ma_band_is_risk_on_while_the_ma_is_undefined():
    out, state = run_ma_band([100.0, 50.0, 25.0], [0, 1, 2])
    assert out == [3.0, 3.0, 3.0] and state["on"] is True and state["switches"] == []


def test_ma_band_warm_up_rows_only_feed_the_ma():
    # the drop to 97 happens before the first decision, so it does not switch: the state starts at row 21
    out, state = run_ma_band(BAND_SERIES[:22], [21])
    assert out == [0.5]  # row 21: 99 < MA 99.80 -> starts off
    assert state["switches"] == []


# ----------------------------------------------------------------------------- risk.vol_cap


def alternating_prices(n_returns, step=0.01, start=100.0):
    returns = [step if i % 2 == 0 else -step for i in range(n_returns)]
    return start * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))


def test_vol_cap_by_hand():
    component, p = parse("risk.vol_cap", window_days=20, k=0.3)
    sigma = 0.01 * math.sqrt(20 / 19) * math.sqrt(252)  # sample std (ddof 1) of +-1% log returns, annualised
    assert component.step(p, None, FakeView(alternating_prices(20)), 3.0) == pytest.approx(0.3 / sigma)
    assert component.step(p, None, FakeView(alternating_prices(20)), 1.0) == 1.0  # min(L_in, k / sigma)
    # only the last window_days returns count
    noisy = np.concatenate([[1.0, 500.0, 3.0], alternating_prices(20)])
    assert component.step(p, None, FakeView(noisy), 3.0) == pytest.approx(0.3 / sigma)


def test_vol_cap_passes_through_without_enough_rows_or_without_volatility():
    component, p = parse("risk.vol_cap", window_days=20, k=0.3)
    assert component.step(p, None, FakeView(alternating_prices(19)), 3.0) == 3.0
    assert component.step(p, None, FakeView([100.0] * 25), 3.0) == 3.0


# ----------------------------------------------------------------------------- sizing.target_leverage


def units(leverage, price, rounding="floor", direction=1, weight=1.0, equity=E0, symbol="TMF"):
    component, p = parse("sizing.target_leverage", rounding=rounding)
    return component.units(
        p,
        equity=equity,
        direction=direction,
        leverage=leverage,
        weight=weight,
        price=price,
        instrument=INSTRUMENTS[symbol],
    )


def test_target_leverage_sizing_by_hand():
    assert units(3.0, 20000) == 15  # 3e6 / 2e5 exactly, not 14 on float noise
    assert units(3.0, 20001) == 14  # 14.9993 -> floor
    assert units(3.0, 20001, rounding="nearest") == 15
    assert units(3.0, 20001, direction=-1) == -14  # magnitude floored for shorts too
    assert units(0.5, 20000, weight=0.5) == 1  # 1.25
    assert units(3.0, 20000, equity=-5.0) == 0
    assert units(2.5 * 20000 * 10 / E0, 20000, rounding="nearest") == 2  # 2.5 -> half to even
    assert units(3.0, REFERENCE_INDEX_LEVEL, symbol="MTX") == 1  # §3.7.5 example


# ----------------------------------------------------------------------------- rebalance


def ctx(rolled=False, target=3.0, actual=3.0):
    return RebalanceContext(pd.Timestamp("2026-01-05"), rolled, target, actual)


def trades(name, current, target, context=None, **params):
    component, p = parse(name, **params)
    return component.should_trade(p, None, current_units=current, target_units=target, ctx=context or ctx())


def test_daily_rebalance_band():
    assert not trades("rebalance.daily", 5, 5)
    assert trades("rebalance.daily", 5, 6)
    assert not trades("rebalance.daily", 5, 7, min_change_units=3)
    assert trades("rebalance.daily", 5, 2, min_change_units=3)


def test_monthly_at_roll_resizes_only_on_roll_days_but_side_changes_at_once():
    name = "rebalance.monthly_at_roll"
    assert not trades(name, 5, 9, ctx(rolled=False))
    assert trades(name, 5, 9, ctx(rolled=True))
    assert not trades(name, 5, 6, ctx(rolled=True), min_change_units=2)
    assert trades(name, 0, 5) and trades(name, 5, 0) and trades(name, 5, -3)


def test_threshold_band_on_relative_leverage_drift():
    name = "rebalance.threshold_band"
    assert not trades(name, 14, 15, ctx(target=3.0, actual=2.8), band_pct=0.1)  # 6.7 % drift
    assert trades(name, 13, 15, ctx(target=3.0, actual=2.6), band_pct=0.1)  # 13.3 % drift
    assert not trades(name, 15, 15, ctx(target=3.0, actual=2.0), band_pct=0.1)  # nothing to send
    assert trades(name, 0, 15, ctx(target=3.0, actual=0.0)) and trades(name, 15, 0, ctx(target=0.0, actual=3.0))


def test_on_signal_change_trades_only_on_side_changes():
    name = "rebalance.on_signal_change"
    assert not trades(name, 3, 5)
    assert trades(name, 0, 5) and trades(name, 5, -5) and trades(name, 5, 0)


# ----------------------------------------------------------------------------- roll, allocation


def test_roll_due_k_sessions_before_expiry():
    days = pd.bdate_range("2026-01-05", periods=10)
    calendar = RollCalendar(days, (days[5],))
    component = REGISTRY["roll.before_settlement"]
    due = {
        k: [
            t
            for t in days
            if component.roll_due(component.parse({"days_before": k}), session_date=t, calendar=calendar)
        ]
        for k in (0, 1, 2)
    }
    assert due == {0: [days[5]], 1: [days[4]], 2: [days[3]]}


def test_roll_due_matches_the_engine_roll_schedule():
    frame, expiries, _ = synthetic_contracts()
    futures = {"core": FuturesInput(spec=TMF, bars=frame, expiries=expiries, roll_days_before=1)}
    res = run(strategy_factory(plan(resolved_spec(ma_days=20, vol_k=None)))(), futures, EngineConfig())
    held = res.positions_daily.set_index("date")["contract"]
    engine_rolls = list(held.index[held != held.shift(1)][1:])
    component = REGISTRY["roll.before_settlement"]
    calendar = RollCalendar(pd.DatetimeIndex(held.index), tuple(pd.Timestamp(e) for e in expiries.values()))
    due = [
        t
        for t in held.index
        if component.roll_due(component.parse({"days_before": 1}), session_date=t, calendar=calendar)
    ]
    assert len(engine_rolls) >= 8 and due == engine_rolls


def test_single_allocation():
    component, p = parse("allocation.single")
    assert component.weights(p, None, None, ("main",)) == {"main": 1.0}


# ----------------------------------------------------------------------------- SleeveView


def market(n_bars=4, n_aux=4):
    buffer = SeriesBuffer(n_bars)
    for i in range(n_bars):
        close = 100.0 + i
        buffer.append_bar(
            DATES[i], "202601", dict(open=close, high=close, low=close, close=close, adjusted_close=close * 2)
        )
    aux = _Buffer(n_aux, ("close",))
    for i in range(n_aux):
        aux.append(DATES[i], {"close": 1000.0 + i})
    return MarketView(DATES[n_bars - 1], {"core": SeriesView(buffer, buffer.n)}, {"taiex": AuxView(aux, aux.n)})


def test_sleeve_view_resolves_self_fields_aux_and_lag():
    view = SleeveView(market(), "core", Conventions())
    assert view.series("self:close").values.tolist() == [100.0, 101.0, 102.0, 103.0]
    assert view.series("core:adjusted").values.tolist() == [200.0, 202.0, 204.0, 206.0]
    routed = SleeveView(market(), "core", Conventions(signal_aux={"core:adjusted": "taiex"}, signal_lag=1))
    series = routed.series("self:adjusted")
    assert series.values.tolist() == [1000.0, 1001.0, 1002.0]
    assert list(series.dates) == list(DATES.values[:3])
    assert routed.series("core:close").values.tolist() == [100.0, 101.0, 102.0]  # lag applies to every ref
    with pytest.raises(ValueError):
        Conventions(signal_lag=-1)


# ----------------------------------------------------------------------------- SpecStrategy through the engine


def rising(n_days=30, expiry_day=25):
    rows = []
    for i in range(n_days):
        price = round(20000 * 1.02**i)
        rows.append((i, "A", price, price, price, price))
        rows.append((i, "B", price + 10, price + 10, price + 10, price + 10))
    frame = bars([r for r in rows if not (r[1] == "A" and r[0] > expiry_day)])
    return {"core": FuturesInput(spec=TMF, bars=frame, expiries={"A": day(expiry_day), "B": "2100-01-01"})}


@pytest.mark.parametrize(
    "rebalance,expected_days",
    [
        # next_open: the entry decided on day 20 fills on day 21; the roll at day 24's close is followed by
        # the re-size decided on day 24, filled on day 25; nothing in between
        (("rebalance.monthly_at_roll", {"min_change_units": 1}), [21, 25]),
        (("rebalance.daily", {"min_change_units": 1}), None),
    ],
)
def test_rebalance_rule_decides_which_targets_become_orders(rebalance, expected_days):
    spec = resolved_spec(ma_days=20, vol_k=None, rebalance=rebalance)
    res = run(strategy_factory(plan(spec))(), rising(), EngineConfig(execution="next_open", start=day(20)))
    fills = res.fills[res.fills["reason"] == "rebalance"]
    days = sorted({DATES.get_loc(pd.Timestamp(d)) for d in fills["date"]})
    if expected_days is not None:
        assert days == expected_days
    else:
        assert len(days) >= 4  # equity grows ~6 % a day at 3x: the floor target moves on most days


def test_strategy_sizes_with_the_one_sizing_formula_at_the_decision_close():
    spec = resolved_spec(ma_days=20, vol_k=None)
    strategy = strategy_factory(plan(spec))()
    res = run(strategy, rising(), EngineConfig(execution="signal_close", start=day(20)))
    row = res.positions_daily.iloc[0]
    price = round(20000 * 1.02**20)
    assert row["target_units"] == math.floor(E0 * 3.0 / (price * 10))
    assert strategy.trace[0]["links"] == (3.0, 3.0) and strategy.trace[0]["traded"]
