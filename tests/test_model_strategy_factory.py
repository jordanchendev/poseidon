"""Tests for ModelStrategyFactory -- Optuna parameter search factory (FACT-02).

Covers factory requirements:
- PARAM_BOUNDS contains expected keys
- All PARAM_BOUNDS values have valid (low, high, type) tuples
- build_trial_factory() returns (callable, param_bounds)
- Dynamic model_version_idx injection when multiple models available
- No model_version_idx when 0 or 1 model available
- Factory callable returns usable config dict
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from poseidon.backtest.model_strategy_factory import (
    PARAM_BOUNDS,
    ModelStrategyFactory,
)


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


def _make_mock_model_version(model_id: str | None = None) -> MagicMock:
    """Create a mock ModelVersion with .id, .name, .status, .artifacts_path."""
    mock = MagicMock()
    mock.id = model_id or str(uuid4())
    mock.name = "tw_stock_xgb"
    mock.status = "ready"
    mock.artifacts_path = "/tmp/test"
    return mock


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
    "prediction_threshold",
    "position_pct",
    "stop_loss_pct",
    "take_profit_pct",
}


def test_param_bounds_contains_all_expected_keys():
    """PARAM_BOUNDS contains all required searchable parameter keys."""
    assert set(PARAM_BOUNDS.keys()) == EXPECTED_PARAM_KEYS


# ---------------------------------------------------------------------------
# Test 2: All PARAM_BOUNDS values have valid (low, high, type) tuples
# ---------------------------------------------------------------------------


def test_param_bounds_values_are_valid():
    """All PARAM_BOUNDS values have valid (low, high, type) tuples where low < high."""
    for name, bounds in PARAM_BOUNDS.items():
        assert len(bounds) == 3, f"{name} should have 3 elements"
        low, high, ptype = bounds
        assert ptype in ("int", "float"), f"{name} has invalid type: {ptype}"
        assert low < high, f"{name}: low ({low}) must be < high ({high})"


# ---------------------------------------------------------------------------
# Test 3: build_trial_factory returns callable + bounds
# ---------------------------------------------------------------------------


def test_build_trial_factory_returns_callable_and_bounds():
    """build_trial_factory(symbol, market, interval) returns (callable, dict)."""
    factory = ModelStrategyFactory()
    factory_fn, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    assert callable(factory_fn)
    assert isinstance(bounds, dict)
    # Base bounds should match PARAM_BOUNDS
    for key in PARAM_BOUNDS:
        assert key in bounds


# ---------------------------------------------------------------------------
# Test 4: Multiple models adds model_version_idx bound
# ---------------------------------------------------------------------------


def test_build_trial_factory_with_multiple_models_adds_model_idx_bound():
    """With available_models=[mock1, mock2], returned bounds contain model_version_idx."""
    mock1 = _make_mock_model_version("model-1")
    mock2 = _make_mock_model_version("model-2")
    factory = ModelStrategyFactory(available_models=[mock1, mock2])
    _, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    assert "model_version_idx" in bounds
    low, high, ptype = bounds["model_version_idx"]
    assert low == 0
    assert high == 1  # 2 models -> idx 0-1
    assert ptype == "int"


# ---------------------------------------------------------------------------
# Test 5: No models -> no model_version_idx bound
# ---------------------------------------------------------------------------


def test_build_trial_factory_with_no_models_no_idx_bound():
    """With available_models=[], returned bounds do NOT contain model_version_idx."""
    factory = ModelStrategyFactory(available_models=[])
    _, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    assert "model_version_idx" not in bounds


# ---------------------------------------------------------------------------
# Test 6: Single model -> no model_version_idx bound
# ---------------------------------------------------------------------------


def test_build_trial_factory_with_single_model_no_idx_bound():
    """With available_models=[mock1], returned bounds do NOT contain model_version_idx."""
    mock1 = _make_mock_model_version("model-1")
    factory = ModelStrategyFactory(available_models=[mock1])
    _, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    assert "model_version_idx" not in bounds


# ---------------------------------------------------------------------------
# Test 7: Factory callable returns config dict
# ---------------------------------------------------------------------------


def test_build_trial_factory_callable_returns_config_or_strategy():
    """Factory callable returns something usable (config dict)."""
    mock1 = _make_mock_model_version("model-1")
    factory = ModelStrategyFactory(available_models=[mock1])
    factory_fn, bounds = factory.build_trial_factory(
        symbol="2330", market="tw_stock", interval="1d",
    )
    params = _make_sample_params()
    result = factory_fn(params)
    assert isinstance(result, dict)
    assert result["symbol"] == "2330"
    assert result["market"] == "tw_stock"
    assert result["interval"] == "1d"
    assert result["model_version_id"] == "model-1"
    assert "prediction_threshold" in result
    assert "position_pct" in result
    assert "stop_loss_pct" in result
    assert "take_profit_pct" in result
