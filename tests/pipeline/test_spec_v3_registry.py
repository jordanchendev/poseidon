"""Spec v3 registry and instrument table (design 2026-10-24 §3.3, §5.2, §5.3, §12 #4/#6/#18)."""

from __future__ import annotations

import re
from datetime import date

import pytest

from poseidon.pipeline.v3.components_sizing import round_units
from poseidon.pipeline.v3.instruments import INSTRUMENTS, REFERENCE_INDEX_LEVEL, ordered_frequencies
from poseidon.pipeline.v3.registry import (
    PLANNED_COMPONENTS,
    REGISTRY,
    REGISTRY_VERSION,
    DescribeContext,
    LeverageComponent,
    component_fingerprints,
)

# sha256 of each class's source as the base image poseidon-pipeline:28d1ae6c4 computes it: a run frozen
# before step 8 must still pass check_components, so an existing class is never edited (new behaviour is a
# new component or a new version). ``direction.dsl_rules`` is the one exception: step 7 (DSL parity, 21bbe73)
# re-based it on the intraday engine; it is intraday-only, so no daily run (the only ones check_components
# re-checks) froze the base value.
BASE_FINGERPRINTS = {
    "allocation.single@1": "6b6caa2124b9b1b654728b9a4b91e81f481895fbd4e569243151bc3bf055a9d2",
    "direction.constant@1": "94e458eb5aa40fb940c882d7d7a7f9c85a1d7fae5ada660099ca6b100366a2a9",
    "direction.dsl_rules@1": "6ac5cd3b1b3aa9947e58e710a5aa500f7ed771a1c343d06b0f12db998c73bdbb",
    "execution.next_open@1": "19df267b3d44c0f5aa32bfdde168fae264088dbda795ff9bd98e732a4aaa118b",
    "execution.signal_close@1": "0b65f531e3916886ce07a9589e071135fb77745d113d4749771663b10df4897a",
    "leverage.base@1": "c15e6fcd24e64c0a2cfe2dcbb0c95230960edb452f9366c02c11bc054eb9cd16",
    "rebalance.daily@1": "d6182204b2f65e2294eb154e645ae1dabf32ffcc2ed9420ad695ec997c870696",
    "rebalance.monthly_at_roll@1": "88c99293dd3031d212e0c3e52de320bd27dc6867429836a1b5d76fc41e0843c0",
    "rebalance.on_signal_change@1": "761705bda3b62d990cd1c34887981d1e6d6c7f20fd4ad44cf6b56acc73128369",
    "rebalance.threshold_band@1": "c93c39109c53a8b2b69c312850922e582688f49d041372b11dae2835186382f2",
    "regime.ma_band@1": "f8a83761f5526f9a187e6d62583caf084f336a8d0da9b6c3357ed80b3b31f651",
    "risk.vol_cap@1": "6a7ca2470b7f3103d4b7ded5a7849afe615e36309a3f624aaed4751418bfbd44",
    "roll.at_settlement@1": "1d6f63d777631416710469c2bb904b278cd2cf9faaec24d7686e70b3ba3dc392",
    "roll.before_settlement@1": "62355055d7924ac17fbd12a7116f2523029c43fafa8857e905774d387761dbd6",
    "sizing.fixed_contracts@1": "57f7e14f3d47d3851de07bc2d174a2f57867616ba36b4fe2f5acbf2d82c4459e",
    "sizing.target_leverage@1": "9dd0fba490bf5815bc8af649265d335196f7d95e5297645e97cbca5046f32334",
}
# The components REGISTRY_VERSION 2 added, as the deployed runner image poseidon-pipeline:764f41bc3 computes them:
# runs frozen at version 2 (e.g. run #16, the 00631L / 0050 rotation) re-check these in check_components.
V2_FINGERPRINTS = {
    "allocation.fixed_weights@1": "35f5dc519362de72e52afd95b585e7d45e5c3559e420e50bc0b169af85686a50",
    "allocation.inverse_vol@1": "a13d145f1044e04e6e1df25dde6fccd11be82d3a53206e9a78d554a708cc5bc0",
    "allocation.regime_switch@1": "90056bcf1f812448f2749529146c060132de7219b9b6c19e9e06349144d286f5",
    "hedge.protective_put@1": "08a358fc1d55bddb7c28c29a80a6ba879ce16302b8488e46c3f2199b12e6cec9",
    "leverage.drawdown_ladder@1": "2147e2a667034bd0d0f8ab701a999e57c9e0c877f79684d56b6276de49458224",
    "leverage.level_ladder@1": "aa401d7d8ff44c4f7d47863970a40f6aaaada7c7411813257da70828c9c0a2ab",
    "sizing.fixed_notional@1": "f29aaeb220dc1201768a67ce5289e4c3638a2c8b32231c9e0ec4d0204681d316",
}

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
    # step 8 (M3): stocks / ETFs, fixed-notional sizing, multi-sleeve allocation
    "sizing.fixed_notional",
    "allocation.fixed_weights",
    "allocation.inverse_vol",
    "allocation.regime_switch",
    "leverage.drawdown_ladder",
    "leverage.level_ladder",
}
M4_COMPONENTS = {"hedge.protective_put"}
V3_COMPONENTS = {"rebalance.monthly"}
V3_VERSIONS = {"rebalance.monthly@1", "hedge.protective_put@2"}  # what REGISTRY_VERSION 3 adds, as keys
COMPONENTS = M1_COMPONENTS | M4_COMPONENTS | V3_COMPONENTS


def test_registry_version_and_contents():
    assert REGISTRY_VERSION == 3
    assert set(REGISTRY) == COMPONENTS
    assert not set(REGISTRY) & set(PLANNED_COMPONENTS)
    assert "hedge.collar" in PLANNED_COMPONENTS


@pytest.mark.parametrize(
    "version,prints",
    [(1, {"leverage.base@1": BASE_FINGERPRINTS["leverage.base@1"]}), (2, {**BASE_FINGERPRINTS, **V2_FINGERPRINTS})],
)
def test_a_run_frozen_at_an_earlier_registry_version_still_passes_the_component_check(version, prints):
    from poseidon.pipeline.daily_stages import check_components

    frozen = {"registry_version": version, "components": prints}
    assert frozen["registry_version"] != REGISTRY_VERSION
    assert check_components(frozen) == frozen["components"]


@pytest.mark.parametrize("name", sorted(COMPONENTS))
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
    assert set(prints) == {f"{name}@1" for name in COMPONENTS} - {"hedge.protective_put@1"} | V3_VERSIONS
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in prints.values())
    assert component_fingerprints(["leverage.base"]) == {"leverage.base@1": prints["leverage.base@1"]}


def test_every_component_implements_its_hook_but_the_r1_roll_which_the_intraday_engine_runs():
    from poseidon.pipeline.v3.compiler import HOOKS

    missing = []
    for component in REGISTRY.values():
        hook = HOOKS.get(component.slot)
        base = next(c for c in type(component).__mro__ if c.__dict__.get("slot") == component.slot)
        if hook and getattr(type(component), hook) is getattr(base, hook):
            missing.append(component.name)
    assert sorted(missing) == ["roll.at_settlement"]
    with pytest.raises(RuntimeError, match="not bound"):
        dsl = REGISTRY["direction.dsl_rules"]
        dsl.step({}, dsl.init({}), None)
    ma_band = REGISTRY["regime.ma_band"]
    assert ma_band.history(ma_band.parse({"signal": "index:taiex", "ma_days": 250})) == 250
    assert (
        REGISTRY["risk.vol_cap"].history(REGISTRY["risk.vol_cap"].parse({"signal": "self:adjusted", "window_days": 20}))
        == 21
    )


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
    assert tx.price_proxy is None and not txo.tradable and txo.hedge_component == "hedge.protective_put"
    assert txo.daily_data_since == date(2001, 12, 24) and txo.tick_ladder[:2] == ((10.0, 0.1), (50.0, 0.5))
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


def test_components_frozen_into_earlier_runs_have_not_drifted():
    frozen = {**BASE_FINGERPRINTS, **V2_FINGERPRINTS}
    assert component_fingerprints(sorted(frozen)) == frozen
    assert set(frozen) - {"hedge.protective_put@1"} | V3_VERSIONS == set(component_fingerprints())


def test_a_new_version_supersedes_the_old_one_which_stays_checkable():
    from poseidon.pipeline.daily_stages import ComponentDrift, check_components
    from poseidon.pipeline.v3.registry import SUPERSEDED, register

    put = REGISTRY["hedge.protective_put"]
    assert put.version == 2 and set(SUPERSEDED) == {"hedge.protective_put@1"}
    run_17 = {"components": {"hedge.protective_put@1": V2_FINGERPRINTS["hedge.protective_put@1"]}}
    assert check_components(run_17) == run_17["components"]
    with pytest.raises(ComponentDrift, match="no longer registered"):
        check_components({"components": {"hedge.protective_put@3": "0" * 64}})
    with pytest.raises(ValueError, match="already registered"):
        register(type(put))  # the same version again


def test_protective_put_v2_with_v1_params_builds_the_v1_policy_and_describes_the_author_put():
    from poseidon.pipeline.v3.registry import SUPERSEDED

    v1, v2 = SUPERSEDED["hedge.protective_put@1"], REGISTRY["hedge.protective_put"]
    for params in ({}, v1.example, {"moneyness_pct": 0.1, "tenor": "next", "roll_days_before_expiry": 3}):
        kwargs = {"name": "core.protective_put", "underlying": "core", "chain": "TXO"}
        assert v2.policy(v2.parse(params), **kwargs) == v1.policy(v1.parse(params), **kwargs)
    author = v2.parse(v2.example)
    policy = v2.policy(author, name="p", underlying="core", chain="TXO")
    assert (policy.tenor, policy.roll_days, policy.roll_with_tenor, policy.take_profit) == (1, 0, True, 3.0)
    text = v2.describe(author, DescribeContext("1d", INSTRUMENTS["TMF"], None, "core"))
    assert "次近月" in text and "近月到期日當天換成新的次近月" in text and "成交均價的 3 倍時賣出" in text
    assert v2.parse({}).take_profit_multiple is None
