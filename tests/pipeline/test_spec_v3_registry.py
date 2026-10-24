"""Spec v3 registry and instrument table (design 2026-10-24 §3.3, §5.2, §5.3, §12 #4/#6/#18)."""

from __future__ import annotations

import re
from datetime import date

import pytest

from poseidon.pipeline.v3.components import round_units
from poseidon.pipeline.v3.instruments import INSTRUMENTS, REFERENCE_INDEX_LEVEL, ordered_frequencies
from poseidon.pipeline.v3.registry import (
    PLANNED_COMPONENTS,
    REGISTRY,
    REGISTRY_VERSION,
    DescribeContext,
    LeverageComponent,
    component_fingerprints,
)

M1_COMPONENTS = {
    "direction.constant",
    "direction.dsl_rules",
    "leverage.base",
    "regime.ma_band",
    "risk.vol_cap",
    "sizing.target_leverage",
    "sizing.fixed_contracts",
    "rebalance.daily",
    "rebalance.monthly_at_roll",
    "rebalance.threshold_band",
    "rebalance.on_signal_change",
    "roll.before_settlement",
    "roll.at_settlement",
    "execution.next_open",
    "execution.signal_close",
    "allocation.single",
}


def test_registry_version_and_contents():
    assert REGISTRY_VERSION == 1
    assert set(REGISTRY) == M1_COMPONENTS
    assert not set(REGISTRY) & set(PLANNED_COMPONENTS)
    assert "hedge.protective_put" in PLANNED_COMPONENTS


@pytest.mark.parametrize("name", sorted(M1_COMPONENTS))
def test_every_component_declares_and_describes_its_example(name):
    component = REGISTRY[name]
    assert component.summary and component.capabilities
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", token) for token in component.capabilities)
    params = component.parse(component.example)
    instrument = INSTRUMENTS["TX"]
    frequency = ordered_frequencies(component.frequencies)[0]
    leverage_in = 3.0 if isinstance(component, LeverageComponent) else None
    text = component.describe(params, DescribeContext(frequency, instrument, leverage_in, "core"))
    assert text and re.search(r"[一-鿿]", text)  # one zh-TW fragment
    assert text == component.describe(params, DescribeContext(frequency, instrument, leverage_in, "core"))


def test_fingerprints_are_name_at_version_and_sha256():
    prints = component_fingerprints()
    assert set(prints) == {f"{name}@1" for name in M1_COMPONENTS}
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in prints.values())
    assert component_fingerprints(["leverage.base"]) == {"leverage.base@1": prints["leverage.base@1"]}


def test_daily_components_implement_their_hook_and_intraday_ones_wait_for_step_7():
    from poseidon.pipeline.v3.compiler import HOOKS

    missing = []
    for component in REGISTRY.values():
        hook = HOOKS.get(component.slot)
        base = next(c for c in type(component).__mro__ if c.__dict__.get("slot") == component.slot)
        if hook and getattr(type(component), hook) is getattr(base, hook):
            missing.append(component.name)
    assert sorted(missing) == ["direction.dsl_rules", "roll.at_settlement"]
    with pytest.raises(NotImplementedError, match="no daily behaviour"):
        REGISTRY["direction.dsl_rules"].step({}, None, None)
    ma_band = REGISTRY["regime.ma_band"]
    assert ma_band.history(ma_band.parse({"ma_days": 250})) == 250
    assert REGISTRY["risk.vol_cap"].history(REGISTRY["risk.vol_cap"].parse({"window_days": 20})) == 21


def test_instrument_table_matches_the_design_defaults():
    tx, mtx, tmf, txo = (INSTRUMENTS[s] for s in ("TX", "MTX", "TMF", "TXO"))
    assert (tx.point_value_twd, mtx.point_value_twd, tmf.point_value_twd, txo.point_value_twd) == (200, 50, 10, 50)
    assert (
        tx.cost_model.commission_per_contract,
        mtx.cost_model.commission_per_contract,
        tmf.cost_model.commission_per_contract,
        txo.cost_model.commission_per_contract,
    ) == (50, 25, 20, 25)
    for row in (tx, mtx, tmf):
        assert row.tick_size == 1 and row.kind == "future" and row.tradable
        assert row.cost_model.tax_rate == 0.00002 and row.cost_model.tax_on_entry
        assert (row.initial_margin_ratio, row.maintenance_margin_ratio) == (0.07, 0.054)
    assert tmf.listed_since == date(2024, 7, 29) and tmf.price_proxy == "TX" and mtx.price_proxy == "TX"
    assert tx.price_proxy is None and not txo.tradable and "M4" in txo.planned
    # margin is a notional ratio, so MTX = TX / 4 and TMF = TX / 20
    assert mtx.notional_twd(20_000) * 4 == tx.notional_twd(20_000) == tmf.notional_twd(20_000) * 20
    assert "TMF" in tmf.proxy_note(date(2001, 1, 2)) and "2024-07-29" in tmf.proxy_note(date(2001, 1, 2))
    assert tx.proxy_note(date(2001, 1, 2)) is None


@pytest.mark.parametrize(
    ("symbol", "units"),
    # §3.7.5: 1,000,000 TWD, 3x, index 45,654 -> TX 0, MTX 1, TMF 6 contracts
    [("TX", 0), ("MTX", 1), ("TMF", 6)],
)
def test_target_leverage_sizing_reproduces_the_design_example(symbol, units):
    sizing = REGISTRY["sizing.target_leverage"]
    got = sizing.units(
        sizing.parse({}),
        equity=1_000_000,
        direction=1,
        leverage=3.0,
        weight=1.0,
        price=REFERENCE_INDEX_LEVEL,
        instrument=INSTRUMENTS[symbol],
    )
    assert got == units


def test_rounding_keeps_the_sign_and_floors_the_magnitude():
    assert round_units(6.57, "floor") == 6 and round_units(-6.57, "floor") == -6
    assert round_units(6.57, "nearest") == 7 and round_units(-6.57, "nearest") == -7
    assert round_units(6.5, "nearest") == 6 and round_units(7.5, "nearest") == 8  # half to even, as levtx
    assert round_units(0.99, "floor") == 0 and round_units(2.9999999999, "floor") == 3  # float noise
    with pytest.raises(ValueError, match="rounding"):
        round_units(1.0, "ceil")
    fixed = REGISTRY["sizing.fixed_contracts"]
    assert (
        fixed.units(fixed.parse({"n": 2}), equity=1, direction=-1, leverage=0, weight=1, price=1, instrument=None) == -2
    )
