"""Tests for RuleStrategyFactory -- Optuna parameter search factory (FACT-01).

Covers factory requirements:
- PARAM_BOUNDS contains expected keys, does NOT contain period_N keys
- from_config() builds valid RuleStrategy
- build_from_trial() suggests parameters from PARAM_BOUNDS
- to_config_dict() round-trips through from_config()
- Flat params map to RuleConfig-valid condition structure
- All PARAM_BOUNDS values have valid (low, high, type) tuples
- build_trial_factory() returns (callable, param_bounds) for pipeline injection
- feature_names stored on instance and used in build_trial_factory()
- Fewer features trims PARAM_BOUNDS appropriately
"""

from __future__ import annotations

from typing import Any

import pytest

from poseidon.backtest.rule_strategy_factory import (
    PARAM_BOUNDS,
    RuleStrategyFactory,
    _build_config_from_params,
)
from poseidon.strategies.dsl.schema import RuleConfig
from poseidon.strategies.rule_strategy import RuleStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTrial:
    """Mock Optuna trial that records suggested parameters."""

    def __init__(self) -> None:
        self.suggested: dict[str, Any] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        val = (low + high) // 2
        self.suggested[name] = val
        return val

    def suggest_float(self, name: str, low: float, high: float) -> float:
        val = (low + high) / 2
        self.suggested[name] = val
        return val


DEFAULT_FEATURE_NAMES = ["revenue_yoy_growth", "roe", "gross_margin"]


def _make_sample_params() -> dict[str, Any]:
    """Create sample flat params dict with midpoint values from PARAM_BOUNDS."""
    params: dict[str, Any] = {}
    for name, (low, high, ptype) in PARAM_BOUNDS.items():
        if ptype == "int":
            params[name] = (int(low) + int(high)) // 2
        else:
            params[name] = (float(low) + float(high)) / 2
    return params


# ---------------------------------------------------------------------------
# Test 1: PARAM_BOUNDS contains all expected keys
# ---------------------------------------------------------------------------


EXPECTED_PARAM_KEYS = {
    "threshold_0",
    "threshold_1",
    "threshold_2",
    "composite_threshold",
    "position_pct",
}


def test_param_bounds_contains_all_expected_keys():
    """PARAM_BOUNDS contains all required searchable parameter keys."""
    assert set(PARAM_BOUNDS.keys()) == EXPECTED_PARAM_KEYS


# ---------------------------------------------------------------------------
# Test 2: PARAM_BOUNDS does NOT contain period_N keys
# ---------------------------------------------------------------------------


def test_param_bounds_no_period_keys():
    """PARAM_BOUNDS does NOT contain period_0, period_1, period_2 (feature_above doesn't use period)."""
    for key in PARAM_BOUNDS:
        assert not key.startswith("period_"), f"Unexpected period key: {key}"


# ---------------------------------------------------------------------------
# Test 3: from_config returns a valid RuleStrategy
# ---------------------------------------------------------------------------


def test_from_config_returns_valid_strategy():
    """from_config(config_dict) returns a valid RuleStrategy instance."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="2330", market="tw_stock", interval="1d",
        feature_names=DEFAULT_FEATURE_NAMES,
    )
    strategy = RuleStrategyFactory.from_config(config)
    assert isinstance(strategy, RuleStrategy)
    assert strategy.symbol == "2330"
    assert strategy.market == "tw_stock"
    assert strategy.interval == "1d"


# ---------------------------------------------------------------------------
# Test 4: build_from_trial suggests parameters from PARAM_BOUNDS
# ---------------------------------------------------------------------------


def test_build_from_trial_suggests_parameters():
    """build_from_trial(trial) suggests expected params and returns valid RuleStrategy."""
    trial = MockTrial()
    strategy = RuleStrategyFactory.build_from_trial(
        trial, symbol="2330", market="tw_stock", interval="1d",
        feature_names=DEFAULT_FEATURE_NAMES,
    )
    assert isinstance(strategy, RuleStrategy)
    # All PARAM_BOUNDS keys should have been suggested
    assert set(trial.suggested.keys()) == set(PARAM_BOUNDS.keys())
    assert strategy.symbol == "2330"


# ---------------------------------------------------------------------------
# Test 5: to_config_dict round-trips through from_config
# ---------------------------------------------------------------------------


def test_to_config_dict_round_trips():
    """to_config_dict(strategy) returns dict that round-trips through from_config()."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="2330", market="tw_stock", interval="1d",
        feature_names=DEFAULT_FEATURE_NAMES,
    )
    strategy1 = RuleStrategyFactory.from_config(config)
    config_out = RuleStrategyFactory.to_config_dict(strategy1)
    strategy2 = RuleStrategyFactory.from_config(config_out)

    # Key attributes must match
    assert strategy2.symbol == strategy1.symbol
    assert strategy2.market == strategy1.market
    assert strategy2.interval == strategy1.interval
    assert strategy2.name == strategy1.name


# ---------------------------------------------------------------------------
# Test 6: Flat params map to rule config condition structure
# ---------------------------------------------------------------------------


def test_flat_params_map_to_rule_config():
    """Config from _build_config_from_params has rules with condition.vote.conditions."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="2330", market="tw_stock", interval="1d",
        feature_names=DEFAULT_FEATURE_NAMES,
    )

    assert "rules" in config
    assert len(config["rules"]) >= 1

    rule = config["rules"][0]
    assert "condition" in rule
    assert "vote" in rule["condition"]
    vote = rule["condition"]["vote"]
    assert "conditions" in vote
    assert "min_votes" in vote

    # Each condition should be feature_above type
    for cond in vote["conditions"]:
        assert cond["type"] == "feature_above"
        assert "column" in cond
        assert "threshold" in cond

    # Columns should match feature_names
    columns = [c["column"] for c in vote["conditions"]]
    assert columns == DEFAULT_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Test 7: All PARAM_BOUNDS values have valid (low, high, type) tuples
# ---------------------------------------------------------------------------


def test_param_bounds_values_are_valid():
    """All PARAM_BOUNDS values have valid (low, high, type) tuples where low < high."""
    for name, bounds in PARAM_BOUNDS.items():
        assert len(bounds) == 3, f"{name} should have 3 elements"
        low, high, ptype = bounds
        assert ptype in ("int", "float"), f"{name} has invalid type: {ptype}"
        assert low < high, f"{name}: low ({low}) must be < high ({high})"
        if ptype == "int":
            assert isinstance(low, (int, float)), f"{name}: low should be numeric"
            assert isinstance(high, (int, float)), f"{name}: high should be numeric"


# ---------------------------------------------------------------------------
# Test 8: build_trial_factory returns callable + bounds
# ---------------------------------------------------------------------------


def test_build_trial_factory_returns_callable_and_bounds():
    """build_trial_factory(symbol, market, interval) returns (callable, dict)."""
    factory = RuleStrategyFactory(feature_names=DEFAULT_FEATURE_NAMES)
    factory_fn, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    assert callable(factory_fn)
    assert isinstance(bounds, dict)
    assert set(bounds.keys()) == set(PARAM_BOUNDS.keys())


# ---------------------------------------------------------------------------
# Test 9: build_trial_factory callable produces strategy
# ---------------------------------------------------------------------------


def test_build_trial_factory_callable_produces_strategy():
    """build_trial_factory callable accepts params dict and returns a RuleStrategy."""
    factory = RuleStrategyFactory(feature_names=DEFAULT_FEATURE_NAMES)
    factory_fn, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    params = _make_sample_params()
    strategy = factory_fn(params)
    assert isinstance(strategy, RuleStrategy)
    assert strategy.symbol == "2330"
    assert strategy.market == "tw_stock"


# ---------------------------------------------------------------------------
# Test 10: RuleConfig validation passes for factory-produced configs
# ---------------------------------------------------------------------------


def test_rule_config_validation_passes():
    """RuleConfig(**config) does not raise for factory-produced configs."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="2330", market="tw_stock", interval="1d",
        feature_names=DEFAULT_FEATURE_NAMES,
    )
    # Must not raise
    rule_config = RuleConfig(**config)
    assert rule_config.name == "rule_optuna_search"
    assert rule_config.symbol == "2330"
    assert len(rule_config.rules) >= 1


# ---------------------------------------------------------------------------
# Test 11: feature_names stored on instance
# ---------------------------------------------------------------------------


def test_feature_names_stored_on_instance():
    """RuleStrategyFactory(feature_names=[...])._feature_names equals the passed list."""
    names = ["revenue_yoy_growth", "roe"]
    factory = RuleStrategyFactory(feature_names=names)
    assert factory._feature_names == names


# ---------------------------------------------------------------------------
# Test 12: composite_threshold bounds in PARAM_BOUNDS
# ---------------------------------------------------------------------------


def test_composite_threshold_bounds_in_param_bounds():
    """composite_threshold has range (1, 3, 'int')."""
    assert "composite_threshold" in PARAM_BOUNDS
    low, high, ptype = PARAM_BOUNDS["composite_threshold"]
    assert low == 1
    assert high == 3
    assert ptype == "int"


# ---------------------------------------------------------------------------
# Test 13: Fewer features trims param bounds
# ---------------------------------------------------------------------------


def test_fewer_features_trims_param_bounds():
    """If feature_names has only 1 entry, returned bounds should NOT contain threshold_1, threshold_2."""
    factory = RuleStrategyFactory(feature_names=["revenue_yoy_growth"])
    _, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    assert "threshold_0" in bounds
    assert "threshold_1" not in bounds
    assert "threshold_2" not in bounds
    # Non-threshold keys should still be present
    assert "composite_threshold" in bounds
    assert "position_pct" in bounds
