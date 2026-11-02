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
        """max_drawdown=0.10 (<15%) -> dd_penalty=0.0."""
        metrics = {
            "trade_count": 60,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # dd_penalty = max(0, 0.10 - 0.15) * 0.05 = 0.0
        # score = 2.0 * sqrt(min(60/50, 1.0)) - 0.0 - 0.0 = 2.0
        assert compute_composite_score(metrics) == pytest.approx(2.0)

    def test_dd_at_30pct_has_penalty(self):
        """max_drawdown=0.30 -> dd_penalty=max(0, 0.30-0.15)*0.05=0.0075."""
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

    def test_optimal_trade_zone_has_no_trade_penalty(self):
        """50-200 trades -> trade_factor 1.0 (the turnover penalty was replaced by this zone in b4756d9)."""
        metrics = {"trade_count": 200, "max_drawdown": 0.10, "total_return": 0.5, "sharpe_ratio": 2.0}
        assert compute_composite_score(metrics) == pytest.approx(2.0)

    def test_overtrading_decays_the_score(self):
        """>200 trades -> trade_factor = 1/sqrt(trades/200); 3200 trades -> 0.25."""
        metrics = {"trade_count": 3200, "max_drawdown": 0.10, "total_return": 0.5, "sharpe_ratio": 2.0}
        assert compute_composite_score(metrics) == pytest.approx(2.0 * 0.25)

    def test_low_trade_count_penalty(self):
        """trade_count=25 (below 50) -> trade_factor = sqrt(25/50) penalizes score."""
        metrics = {
            "trade_count": 25,
            "max_drawdown": 0.10,
            "total_return": 0.5,
            "sharpe_ratio": 2.0,
        }
        # dd_penalty = 0 (under 15%)
        expected = 2.0 * math.sqrt(0.5) - 0.0 - 0.0
        assert compute_composite_score(metrics) == pytest.approx(expected)

    def test_zero_sharpe(self):
        """sharpe=0.0 -> score is 0 (no dd penalty below the threshold)."""
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
