"""Tests for VaR calculators: Parametric, Historical Simulation, Cornish-Fisher.

Uses synthetic data with known properties for deterministic validation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from scipy.stats import norm

from poseidon.risk.var.calculators import VaRCalculator
from poseidon.risk.var.types import VaRMethod, VaRResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Known 2x2 covariance matrix: 20% and 30% annual vol, ~0.167 correlation
COV_2X2 = np.array([[0.04, 0.01], [0.01, 0.09]])

# Equal weights for two assets
WEIGHTS_2 = np.array([0.5, 0.5])

AS_OF = datetime(2026, 3, 1, tzinfo=UTC)
PORTFOLIO_VALUE = 100_000.0


@pytest.fixture
def calc() -> VaRCalculator:
    return VaRCalculator()


@pytest.fixture
def rng_returns() -> np.ndarray:
    """252-day normal returns with seed 42 for reproducibility."""
    return np.random.RandomState(42).normal(0.0005, 0.02, 252)


@pytest.fixture
def fat_tail_returns() -> np.ndarray:
    """Returns with mild negative skew and positive excess kurtosis.

    Uses t-distribution (df=5) which has excess kurtosis = 6 and mild
    negative skew from the seed. The CF expansion is reliable for
    moderate departures from normality.
    """
    rng = np.random.RandomState(99)
    # t-distribution with df=5 has excess kurtosis = 6/(5-4) = 6
    base = rng.standard_t(df=5, size=1000) * 0.02
    # Shift slightly negative to ensure mild negative skew
    base[:50] -= 0.03
    return base


# ---------------------------------------------------------------------------
# Parametric VaR tests
# ---------------------------------------------------------------------------


class TestParametricVaR:
    def test_produces_var_result(self, calc: VaRCalculator) -> None:
        result = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC.value

    def test_var_within_expected_range(self, calc: VaRCalculator) -> None:
        """With known covariance, portfolio std is computable analytically."""
        result = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        # Portfolio variance = w' * Cov * w
        port_var = WEIGHTS_2 @ COV_2X2 @ WEIGHTS_2
        port_std = np.sqrt(port_var)
        expected_var95 = norm.ppf(0.95) * port_std
        assert abs(result.var_95 - expected_var95) < 1e-10

    def test_cvar_greater_than_var(self, calc: VaRCalculator) -> None:
        """CVaR (Expected Shortfall) is always >= VaR for same confidence."""
        result = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.cvar_95 > result.var_95
        assert result.cvar_99 > result.var_99

    def test_var99_greater_than_var95(self, calc: VaRCalculator) -> None:
        """Higher confidence = larger loss amount."""
        result = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.var_99 > result.var_95

    def test_single_asset(self, calc: VaRCalculator) -> None:
        """1x1 covariance works correctly."""
        cov_1x1 = np.array([[0.04]])
        weights_1 = np.array([1.0])
        result = calc.parametric(
            weights=weights_1,
            cov_matrix=cov_1x1,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        expected_var95 = norm.ppf(0.95) * 0.2  # sqrt(0.04) = 0.2
        assert abs(result.var_95 - expected_var95) < 1e-10

    def test_empty_weights_returns_zeros(self, calc: VaRCalculator) -> None:
        """Empty portfolio returns VaRResult with all zeros."""
        result = calc.parametric(
            weights=np.array([]),
            cov_matrix=np.array([]).reshape(0, 0),
            portfolio_value=0.0,
            as_of=AS_OF,
        )
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0
        assert result.cvar_95 == 0.0
        assert result.cvar_99 == 0.0

    def test_as_of_stored_in_result(self, calc: VaRCalculator) -> None:
        result = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.as_of == AS_OF

    def test_positive_values(self, calc: VaRCalculator) -> None:
        """VaR/CVaR must be positive loss fractions."""
        result = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.var_95 > 0
        assert result.var_99 > 0
        assert result.cvar_95 > 0
        assert result.cvar_99 > 0


# ---------------------------------------------------------------------------
# Historical Simulation VaR tests
# ---------------------------------------------------------------------------


class TestHistoricalSimulationVaR:
    def test_produces_var_result(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL.value

    def test_var_matches_percentile(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        """VaR should equal the negative of the corresponding percentile."""
        result = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        expected_var95 = -np.percentile(rng_returns, 5)
        assert abs(result.var_95 - expected_var95) < 1e-10

    def test_cvar_equals_mean_of_tail(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        """CVaR = mean of losses beyond VaR threshold."""
        result = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        var_95 = -np.percentile(rng_returns, 5)
        tail = rng_returns[rng_returns <= -var_95]
        expected_cvar_95 = -np.mean(tail) if len(tail) > 0 else var_95
        assert abs(result.cvar_95 - expected_cvar_95) < 1e-10

    def test_cvar_greater_than_var(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99

    def test_few_returns_no_crash(self, calc: VaRCalculator) -> None:
        """With fewer than 10 returns, should still produce a result."""
        short_returns = np.array([-0.01, 0.02, -0.03, 0.01, 0.005])
        result = calc.historical_simulation(
            portfolio_returns=short_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert isinstance(result, VaRResult)
        assert result.var_95 >= 0.0

    def test_empty_returns_zeros(self, calc: VaRCalculator) -> None:
        result = calc.historical_simulation(
            portfolio_returns=np.array([]),
            portfolio_value=0.0,
            as_of=AS_OF,
        )
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0

    def test_as_of_stored(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.as_of == AS_OF


# ---------------------------------------------------------------------------
# Cornish-Fisher VaR tests
# ---------------------------------------------------------------------------


class TestCornishFisherVaR:
    def test_produces_var_result(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.CORNISH_FISHER.value

    def test_equals_parametric_for_normal_returns(self, calc: VaRCalculator) -> None:
        """With zero skew and zero excess kurtosis, CF should equal parametric."""
        # Generate large sample from standard normal -> skew ~ 0, kurt ~ 0
        rng = np.random.RandomState(123)
        normal_returns = rng.normal(0, 0.02, 10000)

        result_cf = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=normal_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        result_param = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        # Should be very close (within 1% relative tolerance)
        assert abs(result_cf.var_95 - result_param.var_95) / result_param.var_95 < 0.01
        assert abs(result_cf.var_99 - result_param.var_99) / result_param.var_99 < 0.01

    def test_fat_tails_higher_var_than_parametric(self, calc: VaRCalculator, fat_tail_returns: np.ndarray) -> None:
        """Positive excess kurtosis should produce higher VaR than parametric."""
        result_cf = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=fat_tail_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        result_param = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result_cf.var_99 > result_param.var_99

    def test_negative_skew_increases_tail_risk(self, calc: VaRCalculator, fat_tail_returns: np.ndarray) -> None:
        """Negative skew + excess kurtosis increases VaR at high confidence.

        The Cornish-Fisher expansion adjusts the normal quantile. With
        negative skew and positive excess kurtosis, the 99% VaR should
        exceed the parametric 99% VaR because the kurtosis term (z^3-3z)*k/24
        dominates at extreme quantiles.
        """
        from scipy.stats import skew

        s = skew(fat_tail_returns)
        assert s < 0, "Test data should be negatively skewed"

        result_cf = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=fat_tail_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        result_param = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        # At 99% confidence, kurtosis + skew combined produce higher VaR
        assert result_cf.var_99 > result_param.var_99

    def test_details_contain_skew_and_kurtosis(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.details is not None
        assert "skewness" in result.details
        assert "excess_kurtosis" in result.details

    def test_empty_weights_returns_zeros(self, calc: VaRCalculator) -> None:
        result = calc.cornish_fisher(
            weights=np.array([]),
            cov_matrix=np.array([]).reshape(0, 0),
            portfolio_returns=np.array([0.01, 0.02, 0.03]),
            portfolio_value=0.0,
            as_of=AS_OF,
        )
        assert result.var_95 == 0.0

    def test_few_returns_returns_zeros(self, calc: VaRCalculator) -> None:
        """Fewer than 3 returns -> cannot compute skew/kurt -> zeros."""
        result = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=np.array([0.01, -0.01]),
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.var_95 == 0.0

    def test_as_of_stored(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
        )
        assert result.as_of == AS_OF


# ---------------------------------------------------------------------------
# Holding period scaling tests
# ---------------------------------------------------------------------------


class TestHoldingPeriodScaling:
    def test_parametric_holding_period_10_greater_than_1(self, calc: VaRCalculator) -> None:
        result_1 = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
            holding_period=1,
        )
        result_10 = calc.parametric(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
            holding_period=10,
        )
        # sqrt(10) scaling
        assert result_10.var_95 > result_1.var_95
        ratio = result_10.var_95 / result_1.var_95
        assert abs(ratio - np.sqrt(10)) < 1e-10

    def test_historical_holding_period_scaling(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result_1 = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
            holding_period=1,
        )
        result_10 = calc.historical_simulation(
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
            holding_period=10,
        )
        assert result_10.var_95 > result_1.var_95

    def test_cornish_fisher_holding_period_scaling(self, calc: VaRCalculator, rng_returns: np.ndarray) -> None:
        result_1 = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
            holding_period=1,
        )
        result_10 = calc.cornish_fisher(
            weights=WEIGHTS_2,
            cov_matrix=COV_2X2,
            portfolio_returns=rng_returns,
            portfolio_value=PORTFOLIO_VALUE,
            as_of=AS_OF,
            holding_period=10,
        )
        assert result_10.var_95 > result_1.var_95


# ---------------------------------------------------------------------------
# compute_portfolio_returns tests
# ---------------------------------------------------------------------------


class TestComputePortfolioReturns:
    def test_basic_computation(self, calc: VaRCalculator) -> None:
        """Portfolio returns = aligned_returns @ weights."""
        import pandas as pd

        data = pd.DataFrame(
            {"A": [0.01, -0.02, 0.03], "B": [0.02, 0.01, -0.01]},
            index=pd.date_range("2026-01-01", periods=3),
        )
        weights = np.array([0.6, 0.4])
        result = calc.compute_portfolio_returns(data, weights)
        expected = data.values @ weights
        np.testing.assert_array_almost_equal(result, expected)
