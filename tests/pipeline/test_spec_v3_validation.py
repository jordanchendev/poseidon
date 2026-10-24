"""spec_validate v3 (design 2026-10-24 §3.7): the acceptance spec passes, malformed specs fail at the expected path."""

from __future__ import annotations

from datetime import date

import pytest

from poseidon.pipeline.spec_validation import get_spec_validator
from poseidon.pipeline.v3 import validation
from poseidon.pipeline.v3.describe import describe_spec, narrative
from poseidon.pipeline.v3.dsl_bridge import v2_to_v3
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.validation import FeasibilityContext, validate_spec_v3
from poseidon.strategies.dsl.v2.examples import tx_atr_squeeze_breakout

# Design §3.2, the describe() rendering of the acceptance spec, verbatim.
ACCEPTANCE_NARRATIVE = (
    "每個交易日收盤後決策，次一交易時段開盤成交。商品：微台指（TMF，價格以 TX 代理），結算日前 1 個交易日換月。"
    "方向：一直做多。槓桿：基礎 3 倍；調整後的連續價格低於 200 日均線 ×(1−2%) 時改成 0.5 倍，高於 ×(1+2%) 才回到 3 倍；"
    "再取 min(目前倍數, 0.75 ÷ 20 日年化實現波動)。口數 = 權益 × 倍數 ÷ (價格 × 10)，無條件捨去；差 1 口以上才調整。"
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
    ("txo_not_yet", _set(["instruments", "core", "symbol"], "TXO"), "instruments.core.symbol", "M4", "unsupported"),
    (
        "reserved_alias",
        m(lambda d: d.update(instruments={"self": d["instruments"]["core"]})),
        "instruments.self",
        "reserved",
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
        m(lambda d: _sleeve(d)["hedges"].append({"component": "hedge.protective_put", "params": {"moneyness_pct": 5}})),
        "sleeves.0.hedges.0.component",
        "M4",
        "unsupported",
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
        m(lambda d: _sleeve(d)["leverage"].append({"component": "risk.vol_cap", "params": {"k": 1.0}})),
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
    assert [issue.path for issue in result.errors] == ["instruments.core.symbol"]
    assert "data starts 1998-07-21" in result.errors[0].msg


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
