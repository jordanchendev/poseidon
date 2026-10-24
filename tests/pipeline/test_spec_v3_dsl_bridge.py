"""DSL v2 -> spec v3 -> DSL v2 (design 2026-10-24 §3.4, §10 step 1): every DSL v2 example round-trips."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poseidon.pipeline.spec_validation import DslV2Validator
from poseidon.pipeline.v3.dsl_bridge import NotAFastPathSpec, v2_to_v3, v3_to_v2
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.validation import validate_spec_v3
from poseidon.strategies.dsl.v2.examples import tx_atr_squeeze_breakout, tx_orb30_day
from tests.pipeline.conftest import orb_template, tx_spec
from tests.pipeline.gate_support import overnight_long_spec, random_entry_spec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _mtx(doc):
    doc = copy.deepcopy(doc)
    doc.update(symbol="MTX", market="tw_futures_mtx")
    return doc


DSL_V2_DOCS = {
    "example_tx_orb30_day": tx_orb30_day(),
    "example_tx_atr_squeeze_breakout": tx_atr_squeeze_breakout(),
    "fixture_llm_v21_atr_contraction": json.loads((FIXTURES / "llm_v21_atr_contraction_spec.json").read_text()),
    "pipeline_orb_template": orb_template(),
    "control_overnight_long": overnight_long_spec(),
    "control_random_entry": random_entry_spec(),
    "mtx_orb_template": _mtx(orb_template()),
}


@pytest.mark.parametrize("name", sorted(DSL_V2_DOCS))
def test_v2_to_v3_round_trips_exactly(name):
    doc = DSL_V2_DOCS[name]
    spec = v2_to_v3(doc)
    assert spec["spec_version"] == 3 and spec["frequency"] == doc["interval"]
    assert spec["instruments"]["core"]["symbol"] == doc["symbol"]
    assert set(spec["sleeves"][0]["direction"]["params"]) == set(doc) - {"symbol", "market", "interval", "params"}
    assert v3_to_v2(spec) == doc
    assert json.dumps(v3_to_v2(json.loads(json.dumps(spec)))) == json.dumps(v3_to_v2(spec))


@pytest.mark.parametrize("name", sorted(name for name, doc in DSL_V2_DOCS.items() if doc["symbol"] == "TX"))
def test_v3_verdict_matches_the_dsl_v2_pipeline_validator(name):
    doc = DSL_V2_DOCS[name]
    v2 = DslV2Validator().validate_spec(doc)
    v3 = validate_spec_v3(v2_to_v3(doc))
    assert v3.ok == v2.ok, (v2, [issue.text() for issue in v3.errors + v3.unsupported])
    if v3.ok:
        assert v3.strategy_class == "intraday_fixed"


def test_mtx_dsl_is_a_valid_v3_spec():
    result = validate_spec_v3(v2_to_v3(_mtx(tx_spec())))
    assert result.ok, [issue.text() for issue in result.errors]


def test_empty_params_normalise_to_absent():
    doc = tx_orb30_day()
    for empty in (None, {}):
        assert v3_to_v2(v2_to_v3({**doc, "params": empty})) == doc


def test_symbol_market_mismatch_has_no_v3_form():
    with pytest.raises(ValueError, match="symbol/market"):
        v2_to_v3({**tx_orb30_day(), "market": "tw_futures_mtx"})


def test_non_fast_path_specs_have_no_dsl_form():
    with pytest.raises(NotAFastPathSpec):
        v3_to_v2(tmf_dynamic_leverage_3x())
    spec = v2_to_v3(tx_spec())
    spec["sleeves"][0]["sizing"]["params"]["n"] = 2
    with pytest.raises(NotAFastPathSpec):
        v3_to_v2(spec)
    spec = v2_to_v3(tx_spec())
    spec["capital"] = {"initial_twd": 2_000_000}
    with pytest.raises(NotAFastPathSpec):
        v3_to_v2(spec)
