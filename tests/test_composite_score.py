"""Tests for compute_composite_score() in backtest/metrics.py."""

import math

import pytest

from poseidon.backtest.metrics import compute_composite_score


class TestCompositeScore:
    """Tests for composite scoring function with hard cutoffs and formula."""

    def test_hard_cutoff_low_trade_count(self):
        """trade_count=9 -> returns 0.0 (hard cutoff <10 trades)."""
        metrics = {
            "trade_count": 9,
            "max_drawdown": 0.1,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        assert compute_composite_score(metrics) == 0.0

    def test_hard_cutoff_high_drawdown(self):
        """max_drawdown=0.51 -> returns 0.0 (hard cutoff >50%)."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.51,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        assert compute_composite_score(metrics) == 0.0

    def test_hard_cutoff_capital_loss(self):
        """total_return=-0.51 -> returns 0.0 (hard cutoff >50% capital loss)."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.2,
            "total_return": -0.51,
            "sharpe_ratio": 2.0,
        }
        assert compute_composite_score(metrics) == 0.0

    def test_standard_good_metrics(self):
        """sharpe=2.0, trade_count=100, max_drawdown=0.2, total_return=0.5 -> 1.96."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.2,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # 2.0 * sqrt(min(100/50, 1.0)) - 0.2**2 - max(0, (100-200)/1000)
        # = 2.0 * 1.0 - 0.04 - 0.0 = 1.96
        assert compute_composite_score(metrics) == pytest.approx(1.96)

    def test_low_trade_count_penalty(self):
        """trade_count=25 (below 50) -> trade_factor = sqrt(25/50) penalizes score."""
        metrics = {
            "trade_count": 25,
            "max_drawdown": 0.2,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # 2.0 * sqrt(0.5) - 0.04 - 0.0
        expected = 2.0 * math.sqrt(0.5) - 0.04 - 0.0
        assert compute_composite_score(metrics) == pytest.approx(expected)

    def test_turnover_penalty(self):
        """trade_count=300 -> turnover_penalty = (300-200)/1000 = 0.1."""
        metrics = {
            "trade_count": 300,
            "max_drawdown": 0.2,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # 2.0 * 1.0 - 0.04 - 0.1 = 1.86
        assert compute_composite_score(metrics) == pytest.approx(1.86)

    def test_zero_sharpe(self):
        """sharpe=0.0 -> score near 0 or negative (dd_penalty still applies)."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.2,
            "total_return": 0.1,
            "sharpe_ratio": 0.0,
        }
        # 0.0 * 1.0 - 0.04 - 0.0 = -0.04
        assert compute_composite_score(metrics) == pytest.approx(-0.04)

    def test_missing_keys_defaults_safely(self):
        """Missing keys in metrics dict default safely (no KeyError)."""
        # Empty dict: trade_count defaults to 0 -> hard cutoff
        assert compute_composite_score({}) == 0.0
