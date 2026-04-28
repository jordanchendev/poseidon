"""Unit tests for poseidon.services.portfolio_metrics (Phase 87 / TRUTH-01).

Covers Modified Dietz daily return, geometric TWR linking, Sharpe ratio,
and max-drawdown helpers. Pure-function tests, no DB / no fixtures.
"""

from __future__ import annotations

import math

import pytest

from poseidon.services.portfolio_metrics import (
    cumulative_twr,
    max_drawdown_pct,
    modified_dietz_daily_return,
    sharpe_from_returns,
)

# ---------------------------------------------------------------------------
# modified_dietz_daily_return
# ---------------------------------------------------------------------------


class TestModifiedDietzDailyReturn:
    def test_zero_cash_flow_simple_return(self):
        """No CF: R = (V_end - V_begin) / V_begin."""
        r = modified_dietz_daily_return(v_begin=100.0, v_end=110.0, cash_flow=0.0)
        assert r == pytest.approx(0.10)

    def test_positive_cash_flow_isolated_from_return(self):
        """D-01 core invariant: a deposit must NOT inflate the return.

        V_begin=1M, V_end=11M, CF=+9.87M (the 2026-04-28 deposit case
        scaled down). Naive (V_end - V_begin) / V_begin = 1000% would be
        wrong. Modified Dietz strips the CF and weights it: numerator
        reflects only investment gains.
        """
        v_begin = 1_000_000.0
        v_end = 11_000_000.0
        cf = 9_870_000.0
        # Investment gain = V_end - V_begin - CF = 130_000
        # Avg capital = V_begin + 0.5 * CF = 5_935_000
        # Expected R ~= 130_000 / 5_935_000 ~= 0.0219
        r = modified_dietz_daily_return(v_begin=v_begin, v_end=v_end, cash_flow=cf)
        assert r == pytest.approx(130_000.0 / 5_935_000.0)
        assert 0 < r < 0.05  # NOT 10x

    def test_negative_cash_flow_withdrawal(self):
        """A withdrawal during the period should also be stripped from numerator."""
        v_begin = 1_000_000.0
        v_end = 600_000.0
        cf = -500_000.0  # withdrawal
        # Numerator = V_end - V_begin - CF = 600k - 1M - (-500k) = 100_000 (gain)
        # Denominator = V_begin + 0.5 * CF = 1M + 0.5 * (-500k) = 750_000
        r = modified_dietz_daily_return(v_begin=v_begin, v_end=v_end, cash_flow=cf)
        assert r == pytest.approx(100_000.0 / 750_000.0)

    def test_zero_v_begin_returns_zero(self):
        """Degenerate: empty portfolio start. Avoids div-by-zero."""
        r = modified_dietz_daily_return(v_begin=0.0, v_end=100.0, cash_flow=0.0)
        assert r == 0.0

    def test_negative_denominator_returns_zero(self):
        """Withdrawal larger than starting capital: degenerate."""
        r = modified_dietz_daily_return(v_begin=100.0, v_end=50.0, cash_flow=-1000.0)
        # Denominator = 100 + 0.5 * -1000 = -400 → degenerate → 0.0
        assert r == 0.0

    def test_custom_weight(self):
        """Caller may override the time-weight (e.g. CF at end-of-day)."""
        r = modified_dietz_daily_return(v_begin=1000.0, v_end=2000.0, cash_flow=500.0, weight=1.0)
        # Numerator = 2000 - 1000 - 500 = 500
        # Denominator = 1000 + 1.0 * 500 = 1500
        assert r == pytest.approx(500.0 / 1500.0)


# ---------------------------------------------------------------------------
# cumulative_twr
# ---------------------------------------------------------------------------


class TestCumulativeTwr:
    def test_empty_returns_zero(self):
        assert cumulative_twr([]) == 0.0

    def test_single_return_returns_itself(self):
        assert cumulative_twr([0.05]) == pytest.approx(0.05)

    def test_geometric_linking(self):
        """+10% then -10% should NOT cancel: (1.1)(0.9) - 1 = -0.01."""
        twr = cumulative_twr([0.10, -0.10])
        assert twr == pytest.approx(-0.01)

    def test_three_period_compounding(self):
        """+5%, +3%, +2% → (1.05)(1.03)(1.02) - 1."""
        twr = cumulative_twr([0.05, 0.03, 0.02])
        expected = 1.05 * 1.03 * 1.02 - 1.0
        assert twr == pytest.approx(expected)


# ---------------------------------------------------------------------------
# sharpe_from_returns
# ---------------------------------------------------------------------------


class TestSharpeFromReturns:
    def test_empty_returns_none(self):
        assert sharpe_from_returns([]) is None

    def test_single_return_returns_none(self):
        """Need at least 2 observations for std."""
        assert sharpe_from_returns([0.01]) is None

    def test_zero_std_returns_none(self):
        """Constant returns → no risk → undefined Sharpe."""
        assert sharpe_from_returns([0.01, 0.01, 0.01, 0.01]) is None

    def test_normal_case_annualized(self):
        """Verify mean/std/sqrt(252) wiring."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        sharpe = sharpe_from_returns(returns)
        assert sharpe is not None
        # Sanity: positive mean → positive sharpe
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var)
        expected = (mean / std) * math.sqrt(252)
        assert sharpe == pytest.approx(expected)


# ---------------------------------------------------------------------------
# max_drawdown_pct
# ---------------------------------------------------------------------------


class TestMaxDrawdownPct:
    def test_monotonically_increasing_zero_drawdown(self):
        assert max_drawdown_pct([100.0, 110.0, 120.0]) == 0.0

    def test_simple_drawdown(self):
        """Peak 120 → trough 90 → DD = 25%."""
        dd = max_drawdown_pct([100.0, 120.0, 90.0, 110.0])
        assert dd == pytest.approx(0.25)
