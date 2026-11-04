"""spec_validate v3 (design 2026-10-24 §3.7): the acceptance spec passes, malformed specs fail at the expected path."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from poseidon.pipeline.spec_validation import get_spec_validator
from poseidon.pipeline.v3 import validation, validation_types
from poseidon.pipeline.v3.describe import describe_spec, narrative
from poseidon.pipeline.v3.dsl_bridge import v2_to_v3
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x, tmf_dynamic_leverage_3x_put
from poseidon.pipeline.v3.validation import validate_spec_v3
from poseidon.pipeline.v3.validation_types import FeasibilityContext
from poseidon.strategies.dsl.v2.examples import tx_atr_squeeze_breakout

# Design §3.2, the describe() rendering of the acceptance spec, with the signal series named
# (owner decision 2026-10-24 afternoon #1: both links read the TAIEX price index).
ACCEPTANCE_NARRATIVE = (
    "每個交易日收盤後決策，次一交易時段開盤成交。商品：微台指（TMF，價格以 TX 代理），結算日前 1 個交易日換月。"
    "方向：一直做多。槓桿：基礎 3 倍；加權股價指數（不含息）低於 200 日均線 ×(1−2%) 時改成 0.5 倍，高於 ×(1+2%) 才回到 "
    "3 倍；再取 min(目前倍數, 0.75 ÷ 加權股價指數（不含息） 20 日年化實現波動)。口數 = 權益 × 倍數 ÷ (價格 × 10)，"
    "無條件捨去；差 1 口以上才調整。"
)


def test_the_acceptance_spec_passes():
    result = validate_spec_v3(tmf_dynamic_leverage_3x())
    assert result.ok, [issue.text() for issue in result.errors + result.unsupported]
    assert result.strategy_class == "daily_equity"
    assert result.resolved["sleeves"][0]["leverage"][1]["params"]["ma_days"] == 200
    assert result.resolved["sleeves"][0]["leverage"][1]["params"]["band_pct"] == 0.02
    assert any("TMF" in note and "proxy" in note and "2024-07-29" in note for note in result.notes)
    assert result.template == tmf_dynamic_leverage_3x()


def test_the_acceptance_spec_renders_the_design_sentence():
    result = validate_spec_v3(tmf_dynamic_leverage_3x())
    assert narrative(result.resolved) == ACCEPTANCE_NARRATIVE
    assert describe_spec(result.template, result.params) == (
        ACCEPTANCE_NARRATIVE
        + "可調參數：band 0.01–0.03（間隔 0.005，預設 0.02）、ma_days 150–250（間隔 25，預設 200）、"
        "vol_k 0.5–1（間隔 0.125，預設 0.75）。"
    )


def test_registered_spec_v3_validator_adapts_to_spec_validation():
    validator = get_spec_validator("spec_v3")
    assert validator.validate_spec(tmf_dynamic_leverage_3x()).ok
    resolved = validator.resolve(tmf_dynamic_leverage_3x())
    assert resolved["strategy_class"] == "daily_equity"
    assert resolved["params"]["ma_days"] == {"type": "int", "low": 150, "high": 250, "step": 25.0, "default": 200}
    bad = tmf_dynamic_leverage_3x()
    bad["instruments"]["core"]["symbol"] = "TX"
    verdict = validator.validate_spec(bad)
    assert not verdict.ok and verdict.errors[0].startswith("sleeves.0.sizing: ")


# --------------------------------------------------------------------------- invalid specs


def _sleeve(doc):
    return doc["sleeves"][0]


def _lev(doc, index):
    return _sleeve(doc)["leverage"][index]


def m(fn):
    def apply():
        doc = tmf_dynamic_leverage_3x()
        out = fn(doc)
        return out if isinstance(out, dict) else doc

    return apply


def _set(path, value):
    def fn(doc):
        node = doc
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value

    return m(fn)


def _drop(path):
    def fn(doc):
        node = doc
        for part in path[:-1]:
            node = node[part]
        del node[path[-1]]

    return m(fn)


def _five_params(doc):
    doc["params"]["off"] = {"type": "float", "low": 0.0, "high": 1.0, "step": 0.5, "default": 0.5}
    doc["params"]["win"] = {"type": "int", "low": 10, "high": 30, "step": 10, "default": 20}
    _lev(doc, 1)["params"]["off_leverage"] = {"$param": "off"}
    _lev(doc, 2)["params"]["window_days"] = {"$param": "win"}


def _no_params(doc):
    _lev(doc, 1)["params"].update(ma_days=200, band_pct=0.02)
    _lev(doc, 2)["params"]["k"] = 0.75
    doc["params"] = {}


def _no_chain(doc):
    _sleeve(doc)["leverage"] = []
    doc["params"] = {"mc": {"type": "int", "low": 1, "high": 3, "step": 1, "default": 1}}
    _sleeve(doc)["rebalance"]["params"]["min_change_units"] = {"$param": "mc"}


def _two_sleeves(doc):
    second = dict(_sleeve(doc), id="second")
    doc["sleeves"].append(second)


def _dsl_spec(mutate=None):
    def make():
        dsl = tx_atr_squeeze_breakout()
        dsl["params"] = {"stop_atr": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5, "default": 2.0}}
        dsl["risk"]["stop_loss"]["value"] = {"$param": "stop_atr"}
        spec = v2_to_v3(dsl)
        if mutate is not None:
            mutate(spec)
        return spec

    return make


def _dsl_body(spec):
    return spec["sleeves"][0]["direction"]["params"]


def _bad_dsl_operator(spec):
    _dsl_body(spec)["rules"][0]["condition"]["all"][1]["direction"] = "sideways"


def _dsl_htf(spec):
    _dsl_body(spec)["rules"][0]["condition"]["all"][0]["htf"] = "60m"


INVALID = [
    # (case id, document factory, expected path, message fragment, kind)
    ("not_an_object", lambda: ["spec"], "", "JSON object", "error"),
    ("spec_version_2", _set(["spec_version"], 2), "spec_version", "", "error"),
    ("missing_frequency", _drop(["frequency"]), "frequency", "required", "error"),
    ("frequency_1h", _set(["frequency"], "1h"), "frequency", "", "error"),
    ("unknown_top_level_key", _set(["stop_loss"], 100), "stop_loss", "Extra inputs", "error"),
    ("capital_too_small", _set(["capital", "initial_twd"], 100_000), "capital.initial_twd", "300000", "error"),
    ("capital_too_large", _set(["capital", "initial_twd"], 30_000_000), "capital.initial_twd", "20000000", "error"),
    ("bad_name", _set(["name"], "Bad Name"), "name", "pattern", "error"),
    (
        "unknown_symbol",
        _set(["instruments", "core", "symbol"], "NQ"),
        "instruments.core.symbol",
        "TX, MTX, TMF",
        "error",
    ),
    (
        "txo_is_no_sleeve_instrument",
        _set(["instruments", "core", "symbol"], "TXO"),
        "instruments.core.symbol",
        "buy it with hedge.protective_put",
        "error",
    ),
    (
        "reserved_alias",
        m(lambda d: d.update(instruments={"self": d["instruments"]["core"]})),
        "instruments.self",
        "reserved",
        "error",
    ),
    (
        "reserved_alias_index",
        m(lambda d: d.update(instruments={"index": d["instruments"]["core"]})),
        "instruments.index",
        "reserved",
        "error",
    ),
    (
        "signal_missing",
        m(lambda d: _lev(d, 1)["params"].pop("signal")),
        "sleeves.0.leverage.1.params.signal",
        "required",
        "error",
    ),
    (
        "signal_unknown_index",
        m(lambda d: _lev(d, 2)["params"].update(signal="index:nikkei")),
        "sleeves.0.leverage.2.params.signal",
        "pattern",
        "error",
    ),
    (
        "signal_index_field",
        m(lambda d: _lev(d, 1)["params"].update(signal="index:close")),
        "sleeves.0.leverage.1.params.signal",
        "index:taiex, index:taiex_tr",
        "error",
    ),
    ("future_without_roll", _drop(["instruments", "core", "roll"]), "instruments.core.roll", "roll policy", "error"),
    (
        "unknown_component",
        m(lambda d: _lev(d, 2).update(component="leverage.turbo")),
        "sleeves.0.leverage.2.component",
        "choose one of: leverage.base, regime.ma_band, risk.vol_cap",
        "error",
    ),
    (
        "component_in_wrong_slot",
        m(lambda d: _sleeve(d).update(sizing={"component": "rebalance.daily", "params": {}})),
        "sleeves.0.sizing.component",
        "is a rebalance component, not a sizing component",
        "error",
    ),
    (
        "planned_component",
        m(lambda d: _sleeve(d)["exits"].append({"component": "exit.stop_loss", "params": {}})),
        "sleeves.0.exits.0.component",
        "planned: step 3",
        "unsupported",
    ),
    (
        "put_moneyness_in_percent",
        m(
            lambda d: _sleeve(d)["hedges"].append(
                {"component": "hedge.protective_put", "params": {"moneyness_pct": 5.0}}
            )
        ),
        "sleeves.0.hedges.0.params.moneyness_pct",
        "less than or equal to 0.2",
        "error",
    ),
    (
        "put_on_a_short_sleeve",
        m(
            lambda d: (
                _sleeve(d)["direction"]["params"].update(side="short"),
                _sleeve(d)["hedges"].append({"component": "hedge.protective_put"}),
            )
        ),
        "sleeves.0.hedges.0",
        "always short",
        "error",
    ),
    (
        "put_front_expiry_roll_on_a_front_put",
        m(
            lambda d: _sleeve(d)["hedges"].append(
                {"component": "hedge.protective_put", "params": {"tenor": "front", "roll_when": "front_expiry"}}
            )
        ),
        "sleeves.0.hedges.0.params.roll_when",
        "set tenor next",
        "error",
    ),
    (
        "put_take_profit_below_bound",
        m(
            lambda d: _sleeve(d)["hedges"].append(
                {"component": "hedge.protective_put", "params": {"take_profit_multiple": 1.2}}
            )
        ),
        "sleeves.0.hedges.0.params.take_profit_multiple",
        "greater than or equal to 1.5",
        "error",
    ),
    (
        "base_above_bound",
        m(lambda d: _lev(d, 0)["params"].update(value=6.0)),
        "sleeves.0.leverage.0.params.value",
        "less than or equal to 5",
        "error",
    ),
    (
        "ma_band_extra_param",
        m(lambda d: _lev(d, 1)["params"].update(foo=1)),
        "sleeves.0.leverage.1.params.foo",
        "Extra inputs",
        "error",
    ),
    (
        "off_leverage_above_bound",
        m(lambda d: _lev(d, 1)["params"].update(off_leverage=7.0)),
        "sleeves.0.leverage.1.params.off_leverage",
        "",
        "error",
    ),
    (
        "vol_window_below_bound",
        m(lambda d: _lev(d, 2)["params"].update(window_days=2)),
        "sleeves.0.leverage.2.params.window_days",
        "greater than or equal to 5",
        "error",
    ),
    (
        "int_field_given_float",
        m(lambda d: _lev(d, 2)["params"].update(window_days=20.0)),
        "sleeves.0.leverage.2.params.window_days",
        "integer",
        "error",
    ),
    (
        "series_bad_field",
        m(lambda d: _lev(d, 1)["params"].update(signal="core:vwap")),
        "sleeves.0.leverage.1.params.signal",
        "pattern",
        "error",
    ),
    (
        "series_unknown_alias",
        m(lambda d: _lev(d, 1)["params"].update(signal="bench:close")),
        "sleeves.0.leverage.1.params.signal",
        "not declared",
        "error",
    ),
    (
        "rounding_unknown",
        m(lambda d: _sleeve(d)["sizing"]["params"].update(rounding="ceil")),
        "sleeves.0.sizing.params.rounding",
        "floor",
        "error",
    ),
    (
        "param_unknown_name",
        m(lambda d: _lev(d, 1)["params"].update(ma_days={"$param": "nope"})),
        "sleeves.0.leverage.1.params.ma_days",
        "unknown parameter 'nope'",
        "error",
    ),
    (
        "param_ref_malformed",
        m(lambda d: _lev(d, 1)["params"].update(ma_days={"$param": "ma_days", "x": 1})),
        "sleeves.0.leverage.1.params.ma_days",
        "exactly",
        "error",
    ),
    (
        "param_declared_unused",
        m(lambda d: d["params"].update(spare={"type": "int", "low": 1, "high": 3, "step": 1, "default": 2})),
        "params.spare",
        "never used",
        "error",
    ),
    ("five_params", m(_five_params), "params", "1-4 tunable $params, this one declares 5", "error"),
    ("no_params", m(_no_params), "params", "declares 0", "error"),
    (
        "param_on_untunable_field",
        m(lambda d: _sleeve(d)["sizing"]["params"].update(rounding={"$param": "band"})),
        "sleeves.0.sizing.params.rounding",
        "cannot be a $param",
        "error",
    ),
    (
        "float_param_on_int_field",
        m(lambda d: d["params"]["ma_days"].update(type="float")),
        "sleeves.0.leverage.1.params.ma_days",
        "declare params.ma_days with type int",
        "error",
    ),
    (
        "param_grid_out_of_bounds",
        m(lambda d: d["params"]["ma_days"].update(high=450)),
        "params.ma_days",
        "value 425 is invalid at sleeves.0.leverage.1.params.ma_days",
        "error",
    ),
    (
        "param_spec_bad_grid",
        m(lambda d: d["params"]["band"].update(step=0.007)),
        "params.band",
        "multiple of step",
        "error",
    ),
    (
        "chain_not_starting_with_base",
        m(lambda d: _sleeve(d)["leverage"].reverse()),
        "sleeves.0.leverage.0.component",
        "must start with leverage.base",
        "error",
    ),
    (
        "base_not_first",
        m(lambda d: _sleeve(d)["leverage"].append({"component": "leverage.base", "params": {"value": 1.0}})),
        "sleeves.0.leverage.3.component",
        "only be the first link",
        "error",
    ),
    (
        "duplicate_vol_cap",
        m(
            lambda d: _sleeve(d)["leverage"].append(
                {"component": "risk.vol_cap", "params": {"signal": "self:adjusted", "k": 1.0}}
            )
        ),
        "sleeves.0.leverage.3.component",
        "appears twice",
        "error",
    ),
    ("equity_sizing_without_chain", m(_no_chain), "sleeves.0.leverage", "needs a leverage chain", "error"),
    (
        "fixed_contracts_with_chain",
        m(lambda d: _sleeve(d).update(sizing={"component": "sizing.fixed_contracts", "params": {"n": 1}})),
        "sleeves.0.leverage",
        "ignores leverage",
        "error",
    ),
    (
        "intraday_with_daily_components",
        _set(["frequency"], "5m"),
        "sleeves.0.leverage.0.component",
        "supports frequency 1d",
        "error",
    ),
    (
        "unknown_sleeve_instrument",
        m(lambda d: _sleeve(d).update(instrument="other")),
        "sleeves.0.instrument",
        "unknown instrument alias",
        "error",
    ),
    (
        "unused_instrument",
        m(lambda d: d["instruments"].update(spare={"symbol": "TX", "roll": {"component": "roll.before_settlement"}})),
        "instruments.spare",
        "not used",
        "error",
    ),
    ("two_sleeves_single_allocation", m(_two_sleeves), "allocation.component", "1-1 sleeves, the spec has 2", "error"),
    (
        "signal_close_without_dsl",
        m(lambda d: d.update(execution={"component": "execution.signal_close"})),
        "execution.component",
        "only for direction.dsl_rules",
        "error",
    ),
    (
        "intraday_roll_on_daily",
        m(lambda d: d["instruments"]["core"].update(roll={"component": "roll.at_settlement"})),
        "instruments.core.roll.component",
        "supports frequency",
        "error",
    ),
    (
        "zero_contracts_on_tx",
        _set(["instruments", "core", "symbol"], "TX"),
        "sleeves.0.sizing",
        "buys 0 contracts of TX",
        "error",
    ),
    (
        "rounding_error_above_25pct",
        m(lambda d: (d["instruments"]["core"].update(symbol="MTX"), _lev(d, 0)["params"].update(value=4.0))),
        "sleeves.0.sizing",
        "off target",
        "error",
    ),
    ("dsl_rules_on_daily", _dsl_spec(lambda s: s.update(frequency="1d")), "frequency", "'5m'", "error"),
    (
        "dsl_rules_keeps_symbol",
        _dsl_spec(lambda s: _dsl_body(s).update(symbol="TX")),
        "sleeves.0.direction.params.symbol",
        "comes from the instrument",
        "error",
    ),
    (
        "dsl_rule_error_path",
        _dsl_spec(_bad_dsl_operator),
        "sleeves.0.direction.params.rules.0.condition.all.1.direction",
        "",
        "error",
    ),
    (
        "dsl_htf_unsupported",
        _dsl_spec(_dsl_htf),
        "sleeves.0.direction.params.rules.0.condition.all.0.htf",
        "higher-timeframe",
        "unsupported",
    ),
    (
        "dsl_rules_with_equity_sizing",
        _dsl_spec(lambda s: s["sleeves"][0].update(sizing={"component": "sizing.target_leverage", "params": {}})),
        "sleeves.0.sizing.component",
        "supports frequency 1d",
        "error",
    ),
    (
        "dsl_rules_with_daily_rebalance_intraday",
        _dsl_spec(
            lambda s: s["sleeves"][0].update(rebalance={"component": "rebalance.on_signal_change", "params": {"x": 1}})
        ),
        "sleeves.0.rebalance.params.x",
        "Extra inputs",
        "error",
    ),
]


@pytest.mark.parametrize(("case", "make", "path", "fragment", "kind"), INVALID, ids=[row[0] for row in INVALID])
def test_invalid_spec_is_rejected_at_the_expected_path(case, make, path, fragment, kind):
    result = validate_spec_v3(make())
    assert not result.ok
    issues = result.errors if kind == "error" else result.unsupported
    matching = [issue for issue in issues if issue.path == path]
    assert matching, f"{case}: no {kind} at {path!r}; got {[i.text() for i in result.errors + result.unsupported]}"
    assert any(fragment in issue.msg for issue in matching), [issue.msg for issue in matching]


def test_there_are_about_thirty_invalid_cases():
    assert len(INVALID) >= 30


def test_zero_contract_error_suggests_smaller_contracts():
    doc = tmf_dynamic_leverage_3x()
    doc["instruments"]["core"]["symbol"] = "TX"
    message = validate_spec_v3(doc).errors[0].msg
    assert "MTX" in message and "TMF" in message and "6 contracts ~ 2.74x" in message


def test_data_window_before_the_daily_source_fails():
    result = validate_spec_v3(tmf_dynamic_leverage_3x(), feasibility=FeasibilityContext(start=date(1999, 1, 4)))
    assert result.errors[0].path == "instruments.core.symbol"
    assert "data starts 1998-07-21" in result.errors[0].msg and len(result.errors) == 1
    # index:taiex starts 1999-01-05, after the window start: no spec edit can reach it
    assert result.unsupported and all(issue.path.endswith(".params.signal") for issue in result.unsupported)


def test_a_source_that_starts_after_the_window_start_is_unsupported_not_an_error():
    early = validate_spec_v3(tmf_dynamic_leverage_3x(), feasibility=FeasibilityContext(start=date(1998, 1, 5)))
    assert "instruments.core.symbol" in {issue.path for issue in early.unsupported}
    assert not early.errors
    assert "1d data starts 1998-07-21, after the run's data window start 1998-01-05" in early.unsupported[0].msg


def test_signal_index_coverage_is_checked_at_the_largest_window():
    # the acceptance spec reads index:taiex (rows from 1999-01-05); ma_days up to 250 needs ~380 calendar days
    ok = validate_spec_v3(tmf_dynamic_leverage_3x(), feasibility=FeasibilityContext(start=date(2000, 2, 1)))
    assert ok.ok, [issue.text() for issue in ok.errors]
    early = validate_spec_v3(tmf_dynamic_leverage_3x(), feasibility=FeasibilityContext(start=date(1999, 12, 1)))
    assert {issue.path for issue in early.errors} == {
        "sleeves.0.leverage.1.params.signal",
        "sleeves.0.leverage.2.params.signal",
    }
    assert "index:taiex (TAIEX price index) data starts 1999-01-05" in early.errors[0].msg
    total_return = tmf_dynamic_leverage_3x()
    for link in total_return["sleeves"][0]["leverage"][1:]:
        link["params"]["signal"] = "index:taiex_tr"
    uncovered = validate_spec_v3(total_return)  # the daily window starts 2001-01-02
    assert not uncovered.errors and {issue.path for issue in uncovered.unsupported} == {
        "sleeves.0.leverage.1.params.signal",
        "sleeves.0.leverage.2.params.signal",
    }
    assert uncovered.unsupported[0].msg.startswith(
        "index:taiex_tr (TAIEX total-return index) data starts 2003-01-02, after the run's data window start"
    )
    assert validate_spec_v3(total_return, feasibility=FeasibilityContext(start=date(2004, 6, 1))).ok
    adjusted = tmf_dynamic_leverage_3x()
    for link in adjusted["sleeves"][0]["leverage"][1:]:
        link["params"]["signal"] = "self:adjusted"
    assert validate_spec_v3(adjusted, feasibility=FeasibilityContext(start=date(1999, 12, 1))).ok


# --------------------------------------------------------------------------- the chosen window start
# Owner decision 2026-10-25: a series that starts after the default daily start moves the run's start to that
# series' first day plus the warm-up. ``series_since`` stands for the refdata manifests (v3.config.series_first_days).

SERIES = {
    "taifex_futures_daily/TX": date(1998, 7, 21),
    "twse_taiex_daily": date(1999, 1, 5),
    "twse_taiex_total_return": date(2003, 1, 2),
}


def total_return_spec() -> dict:
    doc = tmf_dynamic_leverage_3x()
    for link in doc["sleeves"][0]["leverage"][1:]:
        link["params"]["signal"] = "index:taiex_tr"
    return doc


def choose(doc, series=SERIES, **kw):
    return validate_spec_v3(doc, feasibility=FeasibilityContext(series_since=series, choose_start=True, **kw))


def test_a_series_that_starts_late_moves_the_chosen_start_by_its_warm_up():
    result = choose(total_return_spec())
    assert result.ok, [issue.text() for issue in result.errors + result.unsupported]
    window = result.window
    # ma_days up to 250 (the largest grid value) plus the execution_timing scenario's one-row signal lag
    bars = validation._history_bars(result.template, result.params)
    assert window.warmup_bars == bars + validation_types.SIGNAL_LAG_WARMUP_BARS
    assert window.warmup_calendar_days == validation.warmup_calendar_days(window.warmup_bars, "1d")
    assert window.start == date(2003, 1, 2) + timedelta(days=window.warmup_calendar_days)
    assert window.moved and window.binding == "twse_taiex_total_return"
    assert window.default_start == validation_types.DAILY_WINDOW_START == date(2001, 1, 2)
    assert window.series_since == {
        "taifex_futures_daily/TX": date(1998, 7, 21),
        "twse_taiex_total_return": date(2003, 1, 2),
    }
    # spec_validate later checks the same spec against the frozen start (fixed window): it passes
    frozen = FeasibilityContext(start=window.start, series_since=SERIES)
    assert validate_spec_v3(total_return_spec(), feasibility=frozen).ok
    assert window.as_config() == {
        "rule": "start = max(default_start, latest first day of a series the spec reads + warm-up)",
        "default_start": "2001-01-02",
        "series_since": {"taifex_futures_daily/TX": "1998-07-21", "twse_taiex_total_return": "2003-01-02"},
        "binding_series": "twse_taiex_total_return",
        "warmup_bars": window.warmup_bars,
        "warmup_calendar_days": window.warmup_calendar_days,
        "source": "refdata dataset manifests (coverage.start; futures: the product's first row)",
    }


def test_the_default_start_stands_when_every_series_starts_early_enough():
    result = choose(tmf_dynamic_leverage_3x())  # index:taiex from 1999-01-05: 2001-01-02 has its warm-up
    assert result.ok and result.window.start == date(2001, 1, 2)
    assert not result.window.moved and result.window.binding is None
    assert validate_spec_v3(tmf_dynamic_leverage_3x()).window is None  # no choice without choose_start


def test_the_first_days_come_from_the_caller_not_the_checked_in_table():
    late = {**SERIES, "twse_taiex_daily": date(2005, 3, 1)}  # as if the manifest started later
    result = choose(tmf_dynamic_leverage_3x(), late)
    assert result.ok and result.window.binding == "twse_taiex_daily" and result.window.start > date(2005, 3, 1)
    fixed = validate_spec_v3(
        tmf_dynamic_leverage_3x(), feasibility=FeasibilityContext(start=date(2004, 1, 5), series_since=late)
    )
    assert fixed.unsupported and "data starts 2005-03-01, after the run's data window start" in fixed.unsupported[0].msg


def test_a_series_that_starts_too_late_for_any_window_is_unsupported():
    late = {**SERIES, "twse_taiex_total_return": date(2026, 6, 1)}
    result = choose(total_return_spec(), late, end=date(2026, 10, 24))
    assert not result.ok and not result.errors
    assert result.unsupported[0].path == "sleeves.0.leverage.1.params.signal"
    assert "data starts 2026-06-01" in result.unsupported[0].msg
    assert "not before the data end 2026-10-24" in result.unsupported[0].msg


def test_the_warm_up_leaves_room_for_the_execution_timing_signal_lag():
    from poseidon.pipeline.gates.execution_timing import scenarios_for

    lags = [request.signal_lag_extra for request in scenarios_for({}, base_execution="next_open").values()]
    assert max(lags) == validation_types.SIGNAL_LAG_WARMUP_BARS


# --------------------------------------------------------------------------- the protective put (M4)

PUT_NARRATIVE = (
    "避險：買進台指選擇權月選賣權保護多單：履約價取 TX 期貨收盤價 ×(1−5%) 最近、有結算價的序列，近月（剩餘超過 5 個交易日的到期），"
    "到期前 5 個交易日換到下一個到期；口數 = TMF 口數 × 10 × 1 ÷ 50，無條件捨去（每 5 口 TMF 一口賣權）；"
    "以每日結算價估值和成交（另加滑價），到期以最後結算價現金結算。"
)


def test_the_put_spec_reads_txo_from_its_listing():
    put = tmf_dynamic_leverage_3x_put()
    # the default daily window starts 2001-01-02, before the first TXO day: no spec edit reaches it
    uncovered = validate_spec_v3(put)
    assert not uncovered.errors and [issue.path for issue in uncovered.unsupported] == ["sleeves.0.hedges.0.component"]
    assert "TXO 1d data starts 2001-12-24, after the run's data window start 2001-01-02" in uncovered.unsupported[0].msg
    assert validate_spec_v3(put, feasibility=FeasibilityContext(start=date(2014, 11, 3))).ok
    # the chain needs no warm-up: a window may start on the first TXO day, while the signals still get theirs
    chosen = choose(put)
    assert chosen.ok and chosen.window.binding == "taifex_txo_daily" and chosen.window.start == date(2001, 12, 24)
    assert chosen.window.as_config()["no_warmup_series"] == ["taifex_txo_daily"]
    assert validate_spec_v3(put, feasibility=FeasibilityContext(start=date(2001, 12, 24))).ok
    assert "no_warmup_series" not in choose(total_return_spec()).window.as_config()


def test_the_put_spec_renders_the_hedge_sentence():
    result = validate_spec_v3(tmf_dynamic_leverage_3x_put(), feasibility=FeasibilityContext(start=date(2014, 11, 3)))
    assert narrative(result.resolved) == ACCEPTANCE_NARRATIVE + PUT_NARRATIVE


def test_a_put_that_sizes_to_zero_contracts_is_an_error():
    small = tmf_dynamic_leverage_3x_put()
    small["capital"]["initial_twd"] = 600_000  # 3x at 45,654 buys 3 TMF (2.28x), which cover 0.6 puts
    result = validate_spec_v3(small, feasibility=FeasibilityContext(start=date(2014, 11, 3)))
    assert [issue.path for issue in result.errors] == ["sleeves.0.hedges.0"]
    assert "sizes to 0 TXO puts" in result.errors[0].msg
    assert validate_spec_v3(
        small, feasibility=FeasibilityContext(start=date(2014, 11, 3), prices={"TX": (9_000.0,)})
    ).ok  # at 9,000 the same capital buys 20 TMF: 4 puts


def test_first_bar_price_decides_contract_feasibility():
    doc = tmf_dynamic_leverage_3x()
    doc["instruments"]["core"]["symbol"] = "TX"
    # At a 2001 index level of ~5,000 one TX contract is 1,000,000 TWD: 3x buys 3 contracts.
    assert validate_spec_v3(doc, feasibility=FeasibilityContext(prices={"TX": (5_000.0,)})).ok
    assert not validate_spec_v3(doc, feasibility=FeasibilityContext(prices={"TX": (5_000.0, 45_654.0)})).ok


def test_margin_feasibility(monkeypatch):
    monkeypatch.setattr(validation, "MAX_MARGIN_UTILISATION", 0.1)
    result = validate_spec_v3(tmf_dynamic_leverage_3x())
    assert any(issue.path == "sleeves.0.leverage" and "initial margin" in issue.msg for issue in result.errors)


def test_leverage_policy_cap(monkeypatch):
    monkeypatch.setattr(validation, "LEVERAGE_CAP", 2.0)
    result = validate_spec_v3(tmf_dynamic_leverage_3x())
    assert any(issue.path == "sleeves.0.leverage" and "policy cap" in issue.msg for issue in result.errors)


def test_dsl_fast_path_spec_passes_and_is_intraday_fixed():
    result = validate_spec_v3(_dsl_spec()())
    assert result.ok, [issue.text() for issue in result.errors + result.unsupported]
    assert result.strategy_class == "intraday_fixed" and result.notes == ()
