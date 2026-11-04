"""Step 8: stocks / ETFs in spec v3 (odd lots, fixed-notional sizing, multi-sleeve allocation, validation)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from poseidon.pipeline.engine import run
from poseidon.pipeline.engine.data import stock_input
from poseidon.pipeline.v3.compiler import engine_config, plan, strategy_factory
from poseidon.pipeline.v3.describe import narrative
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050, tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.instruments import INSTRUMENTS, contract_spec, declared
from poseidon.pipeline.v3.registry import REGISTRY, CheckContext, DescribeContext
from poseidon.pipeline.v3.runtime import AllocationView, Conventions, Series
from poseidon.pipeline.v3.spec import strategy_class, substitute
from poseidon.pipeline.v3.validation import validate_spec_v3
from poseidon.pipeline.v3.validation_types import FeasibilityContext

ETF_0050 = INSTRUMENTS["0050"]


def rotation() -> dict:
    """The acceptance template (with its $params), for the validator."""

    return etf_rotation_00631l_0050()


def resolved(**values) -> dict:
    """The acceptance spec with every $param bound, for the compiler and the narrative."""

    doc = substitute(etf_rotation_00631l_0050(), {"ma_days": 200, "band": 0.02, **values})
    doc["params"] = {}
    return doc


def with_allocation(allocation: dict) -> dict:
    """The template with another allocation; its $params move to the first sleeve's sizing."""

    doc = rotation()
    doc["allocation"] = allocation
    doc["params"] = {"pct": {"type": "int", "low": 50, "high": 100, "step": 25, "default": 100}}
    doc["sleeves"][0]["sizing"]["params"]["pct_of_equity"] = {"$param": "pct"}
    return doc


def errors(doc, **feasibility) -> list[str]:
    """Issues of ``doc`` with the window start chosen as the service does (stock data starts late)."""

    result = validate_spec_v3(doc, feasibility=FeasibilityContext(choose_start=True, **feasibility))
    return [issue.text() for issue in (*result.errors, *result.unsupported)]


# ----------------------------------------------------------------------------- instruments


def test_declared_odd_lot_counts_single_shares():
    board = declared({"symbol": "2330"})
    odd = declared({"symbol": "2330", "unit": "odd_lot"})
    assert (board.lot, odd.lot, board.symbol) == (1000, 1, "2330")
    assert board.notional_twd(2385) == 2_385_000 and odd.notional_twd(2385) == 2385
    assert contract_spec(odd).point_value == 1.0
    assert declared({"symbol": "TMF"}) is INSTRUMENTS["TMF"]


# ----------------------------------------------------------------------------- components by hand


def test_fixed_notional_sizing_by_hand():
    sizing = REGISTRY["sizing.fixed_notional"]
    params = sizing.parse({"pct_of_equity": 100})
    assert not sizing.uses_leverage and sizing.basis == "equity" and sizing.exposure(params, 3.0) == 1.0
    kwargs = dict(equity=1_000_000, direction=1, leverage=0.0, price=106.9, instrument=ETF_0050)
    assert sizing.units(params, weight=1.0, **kwargs) == 9  # 1,000,000 / 106,900 = 9.35
    assert sizing.units(params, weight=0.5, **kwargs) == 4
    assert sizing.units(sizing.parse({"pct_of_equity": 50}), weight=1.0, **kwargs) == 4
    assert sizing.units(params, weight=1.0, **{**kwargs, "equity": 0}) == 0
    odd = dataclasses.replace(ETF_0050, lot=1)
    assert sizing.units(params, weight=1.0, **{**kwargs, "instrument": odd}) == 9354
    assert "張" in sizing.describe(params, DescribeContext("1d", ETF_0050))
    assert "股" in sizing.describe(params, DescribeContext("1d", odd))
    with pytest.raises(ValueError):
        sizing.parse({"pct_of_equity": 301})


class _View:
    """A stand-in AllocationView over fixed arrays per sleeve id / ref."""

    def __init__(self, sleeves: dict[str, list[float]], refs: dict[str, list[float]] | None = None, date="2026-01-30"):
        self.date = pd.Timestamp(date)
        self._sleeves = sleeves
        self._refs = refs or {}

    def _series(self, values):
        values = np.asarray(values, float)
        dates = pd.date_range(end=self.date, periods=len(values), freq="B").to_numpy()
        return Series(dates, values)

    def sleeve_series(self, sleeve, field):
        return self._series(self._sleeves[sleeve])

    def series(self, ref):
        return self._series(self._refs[ref])


def test_fixed_weights_parse_check_and_bounds():
    allocation = REGISTRY["allocation.fixed_weights"]
    params = allocation.parse({"weights": {"a": 0.6, "b": 0.4}})
    assert allocation.weights(params, None, None, ("a", "b")) == {"a": 0.6, "b": 0.4}
    assert allocation.weight_bounds(params, ("a", "b")) == {"a": (0.6, 0.6), "b": (0.4, 0.4)}
    with pytest.raises(ValueError, match="at most 1"):
        allocation.parse({"weights": {"a": 0.7, "b": 0.4}})
    with pytest.raises(ValueError):
        allocation.parse({"weights": {"a": 1.0}})
    spec = {"sleeves": [{"id": "a"}, {"id": "c"}]}
    [(field, message)] = allocation.check(params, CheckContext(spec, "1d", None, None, {}))
    assert field == "weights" and "every sleeve id" in message and "a, c" in message
    assert allocation.describe(params, DescribeContext("1d", ETF_0050)) == "固定權重：a 60%、b 40%"


def test_inverse_vol_weights_by_hand():
    allocation = REGISTRY["allocation.inverse_vol"]
    params = allocation.parse({"window_days": 10})
    assert allocation.history(params) == 10
    calm = [100 * 1.001**k for k in range(12)]
    wild = [100 * (1.05 if k % 2 else 0.95) ** k for k in range(12)]
    weights = allocation.weights(params, None, _View({"a": calm, "b": wild}), ("a", "b"))
    sigma = {name: np.diff(np.log(values[-11:])).std(ddof=1) for name, values in (("a", calm), ("b", wild))}
    expected = {name: (1 / sigma[name]) / sum(1 / s for s in sigma.values()) for name in sigma}
    assert weights == approx(expected) and weights["a"] > 0.9
    assert allocation.weights(params, None, _View({"a": calm[:5], "b": wild[:5]}), ("a", "b")) == {"a": 0.5, "b": 0.5}
    assert allocation.weights(params, None, _View({"a": [100] * 12, "b": wild}), ("a", "b")) == {"a": 0.5, "b": 0.5}
    assert allocation.weight_bounds(params, ("a", "b")) == {"a": (0.5, 1.0), "b": (0.5, 1.0)}


def test_regime_switch_follows_the_ma_band_with_hysteresis():
    allocation = REGISTRY["allocation.regime_switch"]
    params = allocation.parse({"signal": "index:taiex", "ma_days": 20, "band_pct": 0.02, "on": "lev", "off": "base"})
    state = allocation.init(params)
    index = [100.0] * 20
    view = _View({}, {"index:taiex": index}, date="2026-02-02")
    assert allocation.weights(params, state, view, ("lev", "base")) == {"lev": 1.0, "base": 0.0}  # at the MA: on
    index = [*index, 99.0]  # inside the band: unchanged
    assert allocation.weights(params, state, _View({}, {"index:taiex": index}), ("lev", "base"))["lev"] == 1.0
    index = [*index, 90.0]  # below MA x 0.98: off
    assert allocation.weights(params, state, _View({}, {"index:taiex": index}), ("lev", "base")) == {
        "lev": 0.0,
        "base": 1.0,
    }
    index = [*index, 100.0]  # back inside the band: still off
    assert allocation.weights(params, state, _View({}, {"index:taiex": index}), ("lev", "base"))["base"] == 1.0
    index = [*index, 120.0]  # above MA x 1.02: on again
    assert allocation.weights(params, state, _View({}, {"index:taiex": index}), ("lev", "base"))["lev"] == 1.0
    assert [on for _date, on in state["switches"]] == [False, True]

    spec = {"sleeves": [{"id": "lev"}, {"id": "base"}]}
    ctx = CheckContext(spec, "1d", None, None, {"lev": INSTRUMENTS["00631L"], "base": ETF_0050})
    assert allocation.check(params, ctx) == []
    bad = allocation.parse({"signal": "self:close", "ma_days": 20, "on": "lev", "off": "lev"})
    fields = [field for field, _message in allocation.check(bad, ctx)]
    assert fields == ["off", "signal"]
    missing = allocation.parse({"signal": "other:close", "ma_days": 20, "on": "x", "off": "base"})
    assert [field for field, _m in allocation.check(missing, ctx)] == ["on", "signal"]
    text = allocation.describe(params, DescribeContext("1d", ETF_0050))
    assert text == "加權股價指數（不含息）高於 20 日均線 ×(1+2%) 時全部押 lev，低於 ×(1−2%) 時全部換成 base"


def test_a_stock_sleeve_cannot_be_sold_short():
    doc = rotation()
    doc["sleeves"][0]["direction"]["params"]["side"] = "short"
    assert any("00631L cannot be sold short (short_stock)" in text for text in errors(doc))
    tmf = tmf_dynamic_leverage_3x()
    tmf["sleeves"][0]["direction"]["params"]["side"] = "short"
    assert not any("short_stock" in text for text in errors(tmf))


# ----------------------------------------------------------------------------- validation


def test_the_rotation_spec_validates_as_daily_equity():
    result = validate_spec_v3(etf_rotation_00631l_0050(), feasibility=FeasibilityContext(choose_start=True))
    assert result.ok, [issue.text() for issue in (*result.errors, *result.unsupported)]
    assert result.strategy_class == "daily_equity"
    assert strategy_class(resolved()) == "daily_equity"
    assert result.window.moved and result.window.binding == "thalassa_tw_stock_daily/00631L"


def test_the_rotation_spec_renders_its_sentence():
    text = narrative(resolved())
    assert text == (
        "每個交易日收盤後決策，次一交易時段開盤成交。"
        "配置：加權股價指數（不含息）高於 200 日均線 ×(1+2%) 時全部押 risk_on，低於 ×(1−2%) 時全部換成 risk_off。"
        "risk_on：商品：元大台灣50正2（00631L）。方向：一直做多。部位名目金額 = 權益的 100%，換成整數張數（無條件捨去）；"
        "實際槓桿偏離目標超過 10% 才調整，方向改變時立即調整。"
        "risk_off：商品：元大台灣50（0050）。方向：一直做多。部位名目金額 = 權益的 100%，換成整數張數（無條件捨去）；"
        "實際槓桿偏離目標超過 10% 才調整，方向改變時立即調整。"
    )
    odd = resolved()
    odd["instruments"]["base"]["unit"] = "odd_lot"
    assert "元大台灣50（0050），以零股（1 股為單位）交易" in narrative(odd)
    assert "換成整數股數" in narrative(odd)


def test_stock_rules_in_the_validator():
    doc = rotation()
    doc["instruments"]["lev"]["unit"] = "odd_lot"
    assert errors(doc) == []

    futures_unit = tmf_dynamic_leverage_3x()
    futures_unit["instruments"]["core"]["unit"] = "odd_lot"
    assert any(
        e.startswith("instruments.core.unit: unit (board_lot / odd_lot) is for stocks") for e in errors(futures_unit)
    )

    short = rotation()
    short["sleeves"][0]["direction"]["params"]["side"] = "short"
    assert any(e.startswith("sleeves.0.direction.params.side: 00631L cannot be sold short") for e in errors(short))

    roll = rotation()
    roll["instruments"]["base"]["roll"] = {"component": "roll.before_settlement", "params": {"days_before": 1}}
    assert any("0050 is not a future; remove roll" in e for e in errors(roll))

    chain = rotation()
    chain["sleeves"][0]["leverage"] = [{"component": "leverage.base", "params": {"value": 1.0}}]
    assert any("sizing.fixed_notional ignores leverage" in e for e in errors(chain))

    leveraged = rotation()
    leveraged["sleeves"][0]["sizing"] = {"component": "sizing.target_leverage", "params": {}}
    leveraged["sleeves"][0]["leverage"] = [{"component": "leverage.base", "params": {"value": 1.5}}]
    [cash] = [e for e in errors(leveraged) if "paid in full" in e]
    assert cash.startswith("sleeves.0.leverage: 00631L is paid in full (no margin loans): the position can reach 1.50x")

    over = rotation()
    over["sleeves"][1]["sizing"]["params"]["pct_of_equity"] = 120
    assert any(e.startswith("sleeves.1.sizing: 0050 is paid in full") for e in errors(over))

    weights = with_allocation(
        {"component": "allocation.fixed_weights", "params": {"weights": {"risk_on": 0.5, "x": 0.5}}}
    )
    assert any(e.startswith("allocation.params.weights: weights must name every sleeve id") for e in errors(weights))

    switch = rotation()
    switch["allocation"]["params"]["on"] = "nope"
    assert any(e.startswith("allocation.params.on: 'nope' is not a sleeve id") for e in errors(switch))

    one = rotation()
    del one["sleeves"][1]
    del one["instruments"]["base"]
    assert any("allocation.regime_switch takes 2-2 sleeves, the spec has 1" in e for e in errors(one))


def test_expensive_stocks_round_to_zero_lots_and_the_hint_says_odd_lot():
    doc = rotation()
    doc["instruments"]["lev"]["symbol"] = "2330"
    [message] = [e for e in errors(doc) if e.startswith("sleeves.0.sizing")]
    assert "buys 0 lots of 2330 (2,385,000 TWD notional each)" in message
    assert "0 lots or more than 25% rounding error is infeasible" in message
    assert "set instruments.lev.unit to odd_lot" in message and "419 shares ~ 1.00x" in message
    doc["instruments"]["lev"]["unit"] = "odd_lot"
    assert errors(doc) == []
    # the window's own prices win over the table's reference close
    priced = rotation()
    assert errors(priced, prices={"00631L": (900.0,), "0050": (106.9,)}) == []
    [message] = [e for e in errors(priced, prices={"00631L": (1500.0,)}) if e.startswith("sleeves.0.sizing")]
    assert "at price 1,500" in message


def test_fixed_weights_scale_the_rounding_check_by_the_sleeve_weight():
    doc = with_allocation(
        {"component": "allocation.fixed_weights", "params": {"weights": {"risk_on": 0.9, "risk_off": 0.1}}}
    )
    [message] = [e for e in errors(doc) if e.startswith("sleeves.1.sizing")]
    assert "target 0.1x" in message and "buys 0 lots of 0050" in message


def test_stock_data_series_and_intraday_are_checked():
    daily = rotation()
    result = validate_spec_v3(daily, feasibility=FeasibilityContext(start=pd.Timestamp("2014-01-02").date()))
    [late] = [issue.text() for issue in result.unsupported]
    assert late.startswith("instruments.lev.symbol: 00631L 1d data starts 2014-10-31")
    chosen = validate_spec_v3(daily, feasibility=FeasibilityContext(choose_start=True))
    assert chosen.ok and chosen.window.series_since["thalassa_tw_stock_daily/0050"].isoformat() == "2003-06-30"
    assert chosen.window.start.isoformat() > "2014-10-31"
    intraday = rotation()
    intraday["frequency"] = "5m"
    assert any("supports frequency 1d; the spec frequency is 5m" in e for e in errors(intraday))


# ----------------------------------------------------------------------------- compiled plan and runtime


def test_plan_accepts_stock_instruments_and_the_allocation_warm_up():
    spec_plan = plan(resolved())
    assert [s.instrument.symbol for s in spec_plan.sleeves] == ["00631L", "0050"]
    assert spec_plan.rolls == {} and spec_plan.warmup_bars == 200 and spec_plan.signal_indices == ("taiex",)
    odd = resolved()
    odd["instruments"]["base"]["unit"] = "odd_lot"
    assert plan(odd).sleeves[1].instrument.lot == 1
    config = engine_config(spec_plan, start="2026-01-05")
    assert config.execution == "next_open" and config.initial_capital == 1_000_000


def _stock(symbol: str, closes: list[float], dates: pd.DatetimeIndex):
    spec = contract_spec(INSTRUMENTS[symbol])
    frame = pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes})
    return stock_input(spec, frame)


def test_rotation_strategy_switches_sleeves_in_the_engine():
    """The compiled rotation on synthetic prices: all-in 00631L while the index is above its MA band,
    all-in 0050 after the index drops below it, the swap filled at one open with the sale first."""

    n = 30
    dates = pd.bdate_range("2026-01-05", periods=n)
    index = [100.0] * 22 + [90.0] * 8
    lev, base = [41.0] * n, [101.0] * n  # flat: 24 lots of 00631L, then 9 lots of 0050, fit in cash with costs
    spec_plan = plan(resolved(ma_days=20))
    strategy = strategy_factory(spec_plan, Conventions())()
    inputs = {"lev": _stock("00631L", lev, dates), "base": _stock("0050", base, dates)}
    aux = {"taiex": pd.DataFrame({"close": index}, index=dates)}
    result = run(strategy, inputs, engine_config(spec_plan, start=dates[21]), aux=aux)

    held_lev, held_base = result.held_units("lev"), result.held_units("base")
    assert held_lev.iloc[1] > 0 and held_base.iloc[1] == 0  # bought at the first open after the start
    swap = result.fills[result.fills["date"] == dates[23]]
    assert swap["instrument"].tolist() == ["lev", "base"] and swap["units"].iloc[0] < 0 < swap["units"].iloc[1]
    assert held_lev.iloc[-1] == 0 and held_base.iloc[-1] > 0
    assert result.events[result.events["kind"].isin(["clip_cash", "trim_cash"])].empty
    trace = pd.DataFrame(strategy.trace)
    first = trace[trace["date"] == dates[21]].set_index("sleeve")
    assert first.loc["risk_on", "target_leverage"] == 1.0 and first.loc["risk_off", "target_leverage"] == 0.0
    fills = result.fills
    costs = (fills["tax"] + fills["commission"] + fills["slippage_twd"]).sum()
    assert fills["units"].tolist() == [24, -24, 9]
    assert result.final_equity == approx(1_000_000 - costs)  # flat prices: only fees and the 1-tick slippage


def test_allocation_view_resolves_refs_and_sleeve_series():
    class Market:
        date = pd.Timestamp("2026-01-30")

        def __getitem__(self, alias):
            class View:
                dates = np.array(["2026-01-29", "2026-01-30"], dtype="datetime64[ns]")
                close = np.array([1.0, 2.0]) if alias == "a" else np.array([10.0, 20.0])
                adjusted_close = close * 2

            return View()

        def aux(self, name):
            class Aux:
                dates = np.array(["2026-01-30"], dtype="datetime64[ns]")

                def __getitem__(self, column):
                    return np.array([7.0])

            return Aux()

    view = AllocationView(Market(), {"x": "a", "y": "b"}, Conventions())
    assert view.sleeve_series("y", "close").values.tolist() == [10.0, 20.0]
    assert view.sleeve_series("x", "adjusted").values.tolist() == [2.0, 4.0]
    assert view.series("b:close").values.tolist() == [10.0, 20.0]
    assert view.series("index:taiex").values.tolist() == [7.0]
    lagged = AllocationView(Market(), {"x": "a"}, Conventions(signal_lag=1))
    assert lagged.sleeve_series("x", "close").values.tolist() == [1.0]
