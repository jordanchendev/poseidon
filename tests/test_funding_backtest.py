"""Tests for funding rate settlement in backtest engine.

Phase 52-02: Validates BacktestResult funding cost fields and
BacktestRunner funding settlement logic at 8h boundaries.

All tests run without DB/GPU dependencies (synthetic data only).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.schemas import BacktestConfig, BacktestResult


# ---------------------------------------------------------------------------
# Task 1: BacktestResult schema extension tests
# ---------------------------------------------------------------------------


class TestBacktestResultFundingFields:
    """Verify BacktestResult funding cost fields are present and backward-compatible."""

    def _make_config(self) -> BacktestConfig:
        return BacktestConfig(
            strategy_type="rule",
            symbol="BTCUSDT",
            market="crypto_perp",
        )

    def test_default_funding_costs_total(self):
        """BacktestResult can be instantiated with funding_costs_total=0.0 (default)."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
        )
        assert result.funding_costs_total == 0.0

    def test_explicit_funding_costs_total(self):
        """BacktestResult can be instantiated with funding_costs_total=123.45."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
            funding_costs_total=123.45,
        )
        assert result.funding_costs_total == 123.45

    def test_funding_costs_by_trade(self):
        """BacktestResult can be instantiated with funding_costs_by_trade as list of floats."""
        costs = [10.5, -5.2, 3.1]
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
            funding_costs_by_trade=costs,
        )
        assert result.funding_costs_by_trade == costs

    def test_pnl_with_and_without_funding(self):
        """BacktestResult can be instantiated with pnl_with_funding and pnl_without_funding."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
            pnl_with_funding=950.0,
            pnl_without_funding=1000.0,
        )
        assert result.pnl_with_funding == 950.0
        assert result.pnl_without_funding == 1000.0

    def test_backward_compatibility(self):
        """Existing BacktestResult instantiation without funding fields still works."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={"total_pnl": 500.0},
            trade_count=3,
            equity_curve_length=100,
            status="completed",
            trades=[{"symbol": "BTCUSDT"}],
        )
        # All funding fields have defaults
        assert result.funding_costs_total == 0.0
        assert result.funding_costs_by_trade == []
        assert result.pnl_with_funding is None
        assert result.pnl_without_funding is None
        # Existing fields still work
        assert result.trade_count == 3
        assert result.metrics["total_pnl"] == 500.0
