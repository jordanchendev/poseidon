"""Tests for LiquiditySweepStrategyFactory -- Optuna parameter search factory.

Covers SWEEP-05 factory requirements:
- PARAM_BOUNDS contains all expected keys
- from_config() builds valid LiquiditySweepStrategy
- build_from_trial() suggests parameters from PARAM_BOUNDS (per D-01)
- to_config_dict() round-trips through from_config()
- Flat PARAM_BOUNDS names correctly map to nested config structure
- All PARAM_BOUNDS values have valid (low, high, type) tuples
- build_trial_factory() returns (callable, param_bounds) for polymorphic injection (per D-02)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from poseidon.backtest.liquidity_sweep_factory import (
    PARAM_BOUNDS,
    LiquiditySweepStrategyFactory,
    _build_config_from_params,
)
from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy


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
    # Detection
    "lookback_bars",
    "wick_ratio_min",
    "breakout_distance_min",
    "oi_buildup_min",
    "confirmation_threshold",
    "w_oi_drop",
    "w_volume",
    "w_funding",
    # Entry
    "fib_level",
    "atr_mult_regime_0",
    "atr_mult_regime_1",
    "atr_mult_regime_2",
    "atr_mult_regime_3",
    # Exit
    "cooldown_bars",
    # Risk
    "risk_pct",
    # Trailing stop
    "trailing_activation_r",
    "trail_atr_multiplier",
}


def test_param_bounds_contains_all_expected_keys():
    """PARAM_BOUNDS contains all required searchable parameter keys."""
    assert set(PARAM_BOUNDS.keys()) == EXPECTED_PARAM_KEYS


# ---------------------------------------------------------------------------
# Test 2: from_config returns a valid LiquiditySweepStrategy
# ---------------------------------------------------------------------------


def test_from_config_returns_valid_strategy():
    """from_config(config_dict) returns a valid LiquiditySweepStrategy instance."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="BTCUSDT", market="crypto_perp", interval="1h",
    )
    strategy = LiquiditySweepStrategyFactory.from_config(config)
    assert isinstance(strategy, LiquiditySweepStrategy)
    assert strategy.symbol == "BTCUSDT"
    assert strategy.market == "crypto_perp"
    assert strategy.interval == "1h"


# ---------------------------------------------------------------------------
# Test 3: build_from_trial suggests parameters from PARAM_BOUNDS (per D-01)
# ---------------------------------------------------------------------------


def test_build_from_trial_suggests_parameters():
    """build_from_trial(trial) suggests all params from PARAM_BOUNDS and returns valid strategy."""
    trial = MockTrial()
    strategy = LiquiditySweepStrategyFactory.build_from_trial(
        trial, symbol="ETHUSDT", market="crypto_perp", interval="1h",
    )
    assert isinstance(strategy, LiquiditySweepStrategy)
    # All PARAM_BOUNDS keys should have been suggested
    assert set(trial.suggested.keys()) == set(PARAM_BOUNDS.keys())
    assert strategy.symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# Test 4: to_config_dict round-trips through from_config
# ---------------------------------------------------------------------------


def test_to_config_dict_round_trips():
    """to_config_dict(strategy) returns dict that can round-trip through from_config()."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="BTCUSDT", market="crypto_perp", interval="1h",
    )
    strategy1 = LiquiditySweepStrategyFactory.from_config(config)
    config_out = LiquiditySweepStrategyFactory.to_config_dict(strategy1)
    strategy2 = LiquiditySweepStrategyFactory.from_config(config_out)

    # Key attributes must match
    assert strategy2.symbol == strategy1.symbol
    assert strategy2.market == strategy1.market
    assert strategy2.interval == strategy1.interval
    assert strategy2._lookback_bars == strategy1._lookback_bars
    assert strategy2._wick_ratio_min == strategy1._wick_ratio_min
    assert strategy2._fib_level == strategy1._fib_level
    assert strategy2._cooldown_bars == strategy1._cooldown_bars
    assert strategy2._trailing_activation_r == strategy1._trailing_activation_r
    assert strategy2._trail_atr_multiplier == strategy1._trail_atr_multiplier
    assert strategy2._atr_multipliers == strategy1._atr_multipliers
    assert strategy2._direction_mode == strategy1._direction_mode


# ---------------------------------------------------------------------------
# Test 5: Flat PARAM_BOUNDS names correctly map to nested config structure
# ---------------------------------------------------------------------------


def test_flat_params_map_to_nested_config():
    """Flat PARAM_BOUNDS names correctly map to nested config structure (detection.*, entry.*, exit.*, trailing.*)."""
    params = _make_sample_params()
    config = _build_config_from_params(
        params, symbol="BTCUSDT", market="crypto_perp", interval="1h",
    )

    # Detection section
    assert "detection" in config
    assert config["detection"]["lookback_bars"] == params["lookback_bars"]
    assert config["detection"]["wick_ratio_min"] == params["wick_ratio_min"]
    assert config["detection"]["breakout_distance_min"] == params["breakout_distance_min"]
    assert config["detection"]["oi_buildup_min"] == params["oi_buildup_min"]
    assert config["detection"]["confirmation_threshold"] == params["confirmation_threshold"]
    assert config["detection"]["w_oi_drop"] == params["w_oi_drop"]
    assert config["detection"]["w_volume"] == params["w_volume"]
    assert config["detection"]["w_funding"] == params["w_funding"]

    # Entry section
    assert "entry" in config
    assert config["entry"]["fib_level"] == params["fib_level"]
    assert config["entry"]["atr_multipliers"][0] == params["atr_mult_regime_0"]
    assert config["entry"]["atr_multipliers"][1] == params["atr_mult_regime_1"]
    assert config["entry"]["atr_multipliers"][2] == params["atr_mult_regime_2"]
    assert config["entry"]["atr_multipliers"][3] == params["atr_mult_regime_3"]

    # Exit section
    assert "exit" in config
    assert config["exit"]["cooldown_bars"] == params["cooldown_bars"]

    # Trailing section
    assert "trailing" in config
    assert config["trailing"]["activation_r"] == params["trailing_activation_r"]
    assert config["trailing"]["atr_multiplier"] == params["trail_atr_multiplier"]


# ---------------------------------------------------------------------------
# Test 6: All PARAM_BOUNDS values have valid (low, high, type) tuples
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
# Test 7: build_trial_factory returns callable + bounds (per D-02)
# ---------------------------------------------------------------------------


def test_build_trial_factory_returns_callable_and_bounds():
    """build_trial_factory(symbol, market, interval) returns (callable, param_bounds)."""
    factory_fn, bounds = LiquiditySweepStrategyFactory.build_trial_factory(
        symbol="BTCUSDT", market="crypto_perp", interval="1h",
    )
    assert callable(factory_fn)
    assert isinstance(bounds, dict)
    assert set(bounds.keys()) == set(PARAM_BOUNDS.keys())


def test_build_trial_factory_callable_produces_strategy():
    """build_trial_factory callable accepts params dict and returns a LiquiditySweepStrategy."""
    factory_fn, bounds = LiquiditySweepStrategyFactory.build_trial_factory(
        symbol="SOLUSDT", market="crypto_perp", interval="1h",
    )
    params = _make_sample_params()
    strategy = factory_fn(params)
    assert isinstance(strategy, LiquiditySweepStrategy)
    assert strategy.symbol == "SOLUSDT"
    assert strategy.market == "crypto_perp"


# ---------------------------------------------------------------------------
# Test 8: build_trial_factory exposes PARAM_BOUNDS (get_param_bounds equivalent)
# ---------------------------------------------------------------------------


def test_build_trial_factory_exposes_param_bounds():
    """build_trial_factory returns PARAM_BOUNDS dict as second element."""
    _, bounds = LiquiditySweepStrategyFactory.build_trial_factory(
        symbol="BTCUSDT", market="crypto_perp", interval="1h",
    )
    # Must match module-level PARAM_BOUNDS
    for key in PARAM_BOUNDS:
        assert key in bounds, f"Missing key: {key}"
        assert bounds[key] == PARAM_BOUNDS[key], f"Mismatch for key: {key}"
