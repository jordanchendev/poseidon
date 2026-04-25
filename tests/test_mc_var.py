"""Tests for Monte Carlo VaR computation on VaRCalculator.

Verifies:
- Correct VaRResult output with method="monte_carlo"
- Deterministic results with seed
- Edge cases: empty weights, non-positive-definite matrix
- as_of preservation
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from poseidon.risk.var.calculators import VaRCalculator
from poseidon.risk.var.types import VaRMethod, VaRResult


@pytest.fixture
def calculator() -> VaRCalculator:
    return VaRCalculator(confidence_levels=(0.95, 0.99))


@pytest.fixture
def as_of() -> datetime:
    return datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)


class TestMonteCarloVaR:
    """Test suite for VaRCalculator.monte_carlo()."""

    def test_basic_identity_covariance(self, calculator: VaRCalculator, as_of: datetime) -> None:
        """MC VaR with 2x2 identity cov, equal weights, returns valid VaRResult."""
        weights = np.array([0.5, 0.5])
        cov_matrix = np.eye(2)

        result = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=100_000.0,
            as_of=as_of,
            seed=42,
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO.value
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.portfolio_value == 100_000.0
        assert result.holding_period == 1

    def test_empty_weights_returns_zero(self, calculator: VaRCalculator, as_of: datetime) -> None:
        """MC VaR with empty weights returns zero result."""
        weights = np.array([])
        cov_matrix = np.eye(0)

        result = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=100_000.0,
            as_of=as_of,
            seed=42,
        )

        assert result.method == VaRMethod.MONTE_CARLO.value
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0
        assert result.cvar_95 == 0.0
        assert result.cvar_99 == 0.0

    def test_deterministic_with_same_seed(self, calculator: VaRCalculator, as_of: datetime) -> None:
        """MC VaR with same seed produces identical results."""
        weights = np.array([0.5, 0.5])
        cov_matrix = np.eye(2)

        r1 = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=100_000.0,
            as_of=as_of,
            seed=42,
        )
        r2 = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=100_000.0,
            as_of=as_of,
            seed=42,
        )

        assert r1.var_95 == r2.var_95
        assert r1.var_99 == r2.var_99

    def test_non_positive_definite_matrix_returns_zero(self, calculator: VaRCalculator, as_of: datetime) -> None:
        """MC VaR with all-zeros matrix (non-PD) returns zero result gracefully."""
        weights = np.array([0.5, 0.5])
        cov_matrix = np.zeros((2, 2))

        result = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=50_000.0,
            as_of=as_of,
            seed=42,
        )

        assert result.method == VaRMethod.MONTE_CARLO.value
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0
        assert result.cvar_95 == 0.0
        assert result.cvar_99 == 0.0

    def test_as_of_preserved_and_computed_at_set(self, calculator: VaRCalculator, as_of: datetime) -> None:
        """as_of is preserved in result, computed_at is set."""
        weights = np.array([0.5, 0.5])
        cov_matrix = np.eye(2)

        result = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=100_000.0,
            as_of=as_of,
            seed=42,
        )

        assert result.as_of == as_of
        assert result.computed_at is not None
        assert result.computed_at >= as_of
