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

    def test_dd_below_15pct_threshold_no_penalty(self):
        """max_drawdown=0.10 (<15%) -> dd_penalty=0.0 (Nunchi D-15)."""
        metrics = {
            "trade_count": 60,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # dd_penalty = max(0, 0.10 - 0.15) * 0.05 = 0.0
        # turnover: no avg_trade_value -> turnover_ratio=0 -> penalty=0
        # score = 2.0 * sqrt(min(60/50, 1.0)) - 0.0 - 0.0 = 2.0
        assert compute_composite_score(metrics) == pytest.approx(2.0)

    def test_dd_at_30pct_has_penalty(self):
        """max_drawdown=0.30 -> dd_penalty=max(0, 0.30-0.15)*0.05=0.0075 (D-15)."""
        metrics = {
            "trade_count": 60,
            "max_drawdown": 0.30,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # dd_penalty = (0.30 - 0.15) * 0.05 = 0.0075
        expected = 2.0 * 1.0 - 0.0075 - 0.0
        assert compute_composite_score(metrics) == pytest.approx(expected)

    def test_dd_at_exactly_15pct_no_penalty(self):
        """max_drawdown=0.15 (exactly at threshold) -> dd_penalty=0.0."""
        metrics = {
            "trade_count": 60,
            "max_drawdown": 0.15,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        assert compute_composite_score(metrics) == pytest.approx(2.0)

    def test_turnover_ratio_below_500_no_penalty(self):
        """turnover_ratio < 500 -> turnover_penalty=0.0 (D-16)."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
            "avg_trade_value": 1000.0,
            "initial_capital": 100000.0,
        }
        # turnover_ratio = 100 * 1000 / 100000 = 1.0 (well under 500)
        # turnover_penalty = max(0, 1.0 - 500) * 0.001 = 0.0
        assert compute_composite_score(metrics) == pytest.approx(2.0)

    def test_turnover_ratio_above_500_has_penalty(self):
        """turnover_ratio > 500 -> turnover_penalty > 0 (D-16)."""
        metrics = {
            "trade_count": 10000,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
            "avg_trade_value": 10000.0,
            "initial_capital": 100000.0,
        }
        # turnover_ratio = 10000 * 10000 / 100000 = 1000
        # turnover_penalty = max(0, 1000 - 500) * 0.001 = 0.5
        expected = 2.0 * 1.0 - 0.0 - 0.5
        assert compute_composite_score(metrics) == pytest.approx(expected)

    def test_no_avg_trade_value_defaults_zero_turnover(self):
        """Without avg_trade_value, turnover_ratio=0, penalty=0."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # No avg_trade_value -> defaults to 0.0 -> ratio=0 -> penalty=0
        assert compute_composite_score(metrics) == pytest.approx(2.0)

    def test_low_trade_count_penalty(self):
        """trade_count=25 (below 50) -> trade_factor = sqrt(25/50) penalizes score."""
        metrics = {
            "trade_count": 25,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # dd_penalty = 0 (under 15%), turnover = 0 (no avg_trade_value)
        expected = 2.0 * math.sqrt(0.5) - 0.0 - 0.0
        assert compute_composite_score(metrics) == pytest.approx(expected)

    def test_zero_sharpe(self):
        """sharpe=0.0 -> score is 0 (no dd or turnover penalty when below thresholds)."""
        metrics = {
            "trade_count": 100,
            "max_drawdown": 0.10,
            "total_return": 0.1,
            "sharpe_ratio": 0.0,
        }
        # 0.0 * 1.0 - 0.0 - 0.0 = 0.0
        assert compute_composite_score(metrics) == pytest.approx(0.0)

    def test_missing_keys_defaults_safely(self):
        """Missing keys in metrics dict default safely (no KeyError)."""
        # Empty dict: trade_count defaults to 0 -> hard cutoff
        assert compute_composite_score({}) == 0.0
