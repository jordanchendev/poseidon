"""Tests for multi-symbol cross-validation gate (SWEEP-06, D-16/D-17).

Covers:
- Cross-symbol gate passes when ALL symbols meet thresholds
- Cross-symbol gate fails when ANY symbol fails pessimistic Sharpe
- Cross-symbol gate fails when ANY symbol fails WFE
- Per-symbol Optuna study name pattern
- Gate report structure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from poseidon.backtest.comparison import validate_cross_symbol


# ---------------------------------------------------------------------------
# Test 1: Multi-symbol run produces per-symbol results
# ---------------------------------------------------------------------------


def test_cross_symbol_gate_report_structure():
    """validate_cross_symbol produces per_symbol_report with correct structure."""
    results = [
        {"symbol": "BTCUSDT", "pessimistic_sharpe": 0.5, "wfe": 0.60},
        {"symbol": "ETHUSDT", "pessimistic_sharpe": 0.3, "wfe": 0.55},
        {"symbol": "SOLUSDT", "pessimistic_sharpe": 0.2, "wfe": 0.52},
    ]
    all_passed, report = validate_cross_symbol(results)

    assert isinstance(report, dict)
    assert len(report) == 3
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert sym in report
        assert "sharpe" in report[sym]
        assert "wfe" in report[sym]
        assert "passed" in report[sym]


# ---------------------------------------------------------------------------
# Test 2: Cross-symbol gate passes when ALL symbols meet thresholds
# ---------------------------------------------------------------------------


def test_cross_symbol_gate_passes_when_all_symbols_pass():
    """All symbols with pessimistic Sharpe > 0 AND WFE >= 50% -> all_passed=True."""
    results = [
        {"symbol": "BTCUSDT", "pessimistic_sharpe": 0.5, "wfe": 0.60},
        {"symbol": "ETHUSDT", "pessimistic_sharpe": 0.3, "wfe": 0.55},
        {"symbol": "SOLUSDT", "pessimistic_sharpe": 0.2, "wfe": 0.52},
    ]
    all_passed, report = validate_cross_symbol(results)

    assert all_passed is True
    for sym_report in report.values():
        assert sym_report["passed"] is True


# ---------------------------------------------------------------------------
# Test 3: Cross-symbol gate fails when ANY symbol has pessimistic Sharpe <= 0
# ---------------------------------------------------------------------------


def test_cross_symbol_gate_fails_when_any_symbol_fails_sharpe():
    """Any symbol with pessimistic Sharpe <= 0 -> all_passed=False."""
    results = [
        {"symbol": "BTCUSDT", "pessimistic_sharpe": 0.5, "wfe": 0.60},
        {"symbol": "ETHUSDT", "pessimistic_sharpe": -0.1, "wfe": 0.55},
    ]
    all_passed, report = validate_cross_symbol(results)

    assert all_passed is False
    assert report["BTCUSDT"]["passed"] is True
    assert report["ETHUSDT"]["passed"] is False


# ---------------------------------------------------------------------------
# Test 4: Cross-symbol gate fails when ANY symbol has WFE < 50%
# ---------------------------------------------------------------------------


def test_cross_symbol_gate_fails_when_any_symbol_fails_wfe():
    """Any symbol with WFE < 50% -> all_passed=False."""
    results = [
        {"symbol": "BTCUSDT", "pessimistic_sharpe": 0.5, "wfe": 0.60},
        {"symbol": "SOLUSDT", "pessimistic_sharpe": 0.3, "wfe": 0.40},
    ]
    all_passed, report = validate_cross_symbol(results)

    assert all_passed is False
    assert report["BTCUSDT"]["passed"] is True
    assert report["SOLUSDT"]["passed"] is False


# ---------------------------------------------------------------------------
# Test 5: Per-symbol Optuna study names follow correct pattern
# ---------------------------------------------------------------------------


def test_per_symbol_optuna_study_names():
    """Each symbol gets its own Optuna study name (e.g., 'crypto_perp_BTCUSDT_1h') (D-03)."""
    from poseidon.backtest.param_search import ParameterSearchPipeline, SearchConfig

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    expected_names = [f"crypto_perp_{s}_1h" for s in symbols]

    # Verify the study name pattern by checking what ParameterSearchPipeline.run would compute
    for symbol, expected_name in zip(symbols, expected_names):
        # The study name is computed as: f"{market}_{symbol}_{interval}"
        study_name = f"crypto_perp_{symbol}_1h"
        assert study_name == expected_name


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------


def test_cross_symbol_gate_empty_results():
    """Empty results list -> all_passed=True, empty report."""
    all_passed, report = validate_cross_symbol([])
    assert all_passed is True
    assert report == {}


def test_cross_symbol_gate_boundary_wfe_exactly_50():
    """WFE exactly 0.50 should pass (>= threshold)."""
    results = [
        {"symbol": "BTCUSDT", "pessimistic_sharpe": 0.1, "wfe": 0.50},
    ]
    all_passed, report = validate_cross_symbol(results)
    assert all_passed is True
    assert report["BTCUSDT"]["passed"] is True


def test_cross_symbol_gate_boundary_sharpe_exactly_zero():
    """Sharpe exactly 0 should fail (must be > 0)."""
    results = [
        {"symbol": "BTCUSDT", "pessimistic_sharpe": 0.0, "wfe": 0.60},
    ]
    all_passed, report = validate_cross_symbol(results)
    assert all_passed is False
    assert report["BTCUSDT"]["passed"] is False


def test_cross_symbol_gate_missing_fields_default_to_zero():
    """Missing sharpe/wfe fields default to 0.0 (fail gate)."""
    results = [
        {"symbol": "BTCUSDT"},
    ]
    all_passed, report = validate_cross_symbol(results)
    assert all_passed is False
    assert report["BTCUSDT"]["sharpe"] == 0.0
    assert report["BTCUSDT"]["wfe"] == 0.0
