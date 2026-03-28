"""Stress test engine (STRESS-02, STRESS-03, STRESS-04).

Runs stress test scenarios against portfolio VaR by manipulating
returns or covariance before delegating to VaRCalculator.

Three scenario types:
- Historical: Replay actual returns during a crisis period.
- Hypothetical: Apply shock vectors to portfolio positions.
- Correlation stress: Replace off-diagonal correlations and re-run parametric VaR.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from poseidon.risk.stress.scenarios import load_scenario
from poseidon.risk.stress.types import ScenarioConfig, StressTestResult
from poseidon.risk.var.calculators import VaRCalculator


class StressTestEngine:
    """Runs stress test scenarios against portfolio VaR.

    Parameters
    ----------
    calculator:
        VaRCalculator instance for computing VaR under stressed conditions.
    scenarios_dir:
        Path to directory containing scenario JSON config files.
    """

    def __init__(
        self,
        calculator: VaRCalculator,
        scenarios_dir: str = "config/stress_scenarios",
    ) -> None:
        self.calculator = calculator
        self.scenarios_dir = scenarios_dir

    def run_scenario(
        self,
        scenario_name: str,
        weights: np.ndarray,
        symbols: list[str],
        cov_matrix: np.ndarray,
        aligned_returns: pd.DataFrame | None = None,
        portfolio_value: float = 1_000_000.0,
    ) -> StressTestResult:
        """Run a named scenario. Dispatches by type field.

        Parameters
        ----------
        scenario_name:
            Name of the scenario (must match a JSON file in scenarios_dir).
        weights:
            Portfolio weight vector.
        symbols:
            Asset names corresponding to weight positions.
        cov_matrix:
            Covariance matrix of asset returns.
        aligned_returns:
            DataFrame of aligned returns (required for historical scenarios).
        portfolio_value:
            Current portfolio value in currency units.
        """
        config = load_scenario(scenario_name, self.scenarios_dir)
        return self.run_config(
            config, weights, symbols, cov_matrix, aligned_returns, portfolio_value
        )

    def run_config(
        self,
        config: ScenarioConfig,
        weights: np.ndarray,
        symbols: list[str],
        cov_matrix: np.ndarray,
        aligned_returns: pd.DataFrame | None = None,
        portfolio_value: float = 1_000_000.0,
    ) -> StressTestResult:
        """Run a ScenarioConfig. Dispatches to scenario-specific handler."""
        if config.type == "historical":
            return self._run_historical(
                config, weights, symbols, aligned_returns, portfolio_value
            )
        elif config.type == "hypothetical":
            return self._run_hypothetical(
                config, weights, symbols, portfolio_value
            )
        elif config.type == "correlation_stress":
            return self._run_correlation_stress(
                config, weights, cov_matrix, portfolio_value
            )
        else:
            raise ValueError(f"Unknown scenario type: {config.type}")

    # ------------------------------------------------------------------
    # Historical scenario (D-02, STRESS-02)
    # ------------------------------------------------------------------

    def _run_historical(
        self,
        config: ScenarioConfig,
        weights: np.ndarray,
        symbols: list[str],
        aligned_returns: pd.DataFrame | None,
        portfolio_value: float,
    ) -> StressTestResult:
        """Replay actual returns during crisis period.

        Filters aligned_returns to the scenario's date_range, computes
        portfolio P&L, and runs historical VaR on the crisis subset.
        """
        if aligned_returns is None or aligned_returns.empty:
            return StressTestResult(
                scenario_name=config.name,
                scenario_type=config.type,
                portfolio_pnl=0.0,
                worst_case_loss=0.0,
                details={"error": "no aligned returns provided"},
                computed_at=datetime.now(timezone.utc),
            )

        # Filter to crisis date range
        crisis_returns = aligned_returns.copy()
        if config.date_range:
            start = pd.Timestamp(config.date_range["start"])
            end = pd.Timestamp(config.date_range["end"])
            crisis_returns = crisis_returns.loc[
                (crisis_returns.index >= start) & (crisis_returns.index <= end)
            ]

        if crisis_returns.empty:
            return StressTestResult(
                scenario_name=config.name,
                scenario_type=config.type,
                portfolio_pnl=0.0,
                worst_case_loss=0.0,
                details={"error": "no data in date range"},
                computed_at=datetime.now(timezone.utc),
            )

        # Compute portfolio returns during crisis
        portfolio_rets = crisis_returns.values @ weights

        # Cumulative P&L
        cumulative_pnl = portfolio_value * (np.cumprod(1 + portfolio_rets) - 1)
        final_pnl = float(cumulative_pnl[-1])
        worst_case = float(np.min(cumulative_pnl))

        # Run historical VaR on crisis period
        var_result = self.calculator.historical_simulation(
            portfolio_returns=portfolio_rets,
            portfolio_value=portfolio_value,
            as_of=datetime.now(timezone.utc),
        )

        return StressTestResult(
            scenario_name=config.name,
            scenario_type=config.type,
            var_result=var_result,
            portfolio_pnl=final_pnl,
            worst_case_loss=worst_case,
            details={
                "crisis_days": len(crisis_returns),
                "date_range": config.date_range,
            },
            computed_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Hypothetical scenario (D-03, STRESS-03)
    # ------------------------------------------------------------------

    def _run_hypothetical(
        self,
        config: ScenarioConfig,
        weights: np.ndarray,
        symbols: list[str],
        portfolio_value: float,
    ) -> StressTestResult:
        """Apply shock vectors to portfolio positions.

        Maps config.shocks (market -> shock factor) to the symbol positions
        and computes the immediate P&L impact.
        """
        if not config.shocks:
            return StressTestResult(
                scenario_name=config.name,
                scenario_type=config.type,
                portfolio_pnl=0.0,
                worst_case_loss=0.0,
                details={"error": "no shocks defined"},
                computed_at=datetime.now(timezone.utc),
            )

        # Build shock vector aligned to symbol positions
        shock_vector = np.zeros(len(symbols))
        for i, sym in enumerate(symbols):
            if sym in config.shocks:
                shock_vector[i] = config.shocks[sym]

        # P&L = portfolio_value * sum(weight_i * shock_i)
        pnl = float(portfolio_value * (weights @ shock_vector))
        worst_case = pnl  # single point shock

        return StressTestResult(
            scenario_name=config.name,
            scenario_type=config.type,
            portfolio_pnl=pnl,
            worst_case_loss=worst_case,
            details={
                "shocks_applied": {
                    sym: config.shocks.get(sym, 0.0) for sym in symbols
                },
                "shock_vector": shock_vector.tolist(),
            },
            computed_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Correlation stress (D-04, STRESS-04)
    # ------------------------------------------------------------------

    def _run_correlation_stress(
        self,
        config: ScenarioConfig,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_value: float,
    ) -> StressTestResult:
        """Replace off-diagonal correlations with target and re-run parametric VaR.

        Extracts standard deviations from the original covariance matrix,
        builds a stressed correlation matrix with target_correlation for all
        off-diagonal entries, converts back to covariance, and runs parametric VaR.
        """
        target_corr = config.target_correlation or 0.9

        stressed_cov = _apply_correlation_stress(cov_matrix, target_corr)

        # Run parametric VaR with stressed covariance
        var_result = self.calculator.parametric(
            weights=weights,
            cov_matrix=stressed_cov,
            portfolio_value=portfolio_value,
            as_of=datetime.now(timezone.utc),
        )

        # Also compute baseline VaR for comparison
        baseline_var = self.calculator.parametric(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=portfolio_value,
            as_of=datetime.now(timezone.utc),
        )

        return StressTestResult(
            scenario_name=config.name,
            scenario_type=config.type,
            var_result=var_result,
            portfolio_pnl=0.0,  # correlation stress doesn't have direct P&L
            worst_case_loss=0.0,
            details={
                "target_correlation": target_corr,
                "stressed_var_95": var_result.var_95,
                "baseline_var_95": baseline_var.var_95,
                "var_increase_pct": (
                    (var_result.var_95 - baseline_var.var_95) / baseline_var.var_95 * 100
                    if baseline_var.var_95 > 0
                    else 0.0
                ),
            },
            computed_at=datetime.now(timezone.utc),
        )


def _apply_correlation_stress(
    cov_matrix: np.ndarray, target_correlation: float
) -> np.ndarray:
    """Replace off-diagonal correlations with target_correlation.

    Parameters
    ----------
    cov_matrix:
        Original covariance matrix (n x n).
    target_correlation:
        Correlation value for all off-diagonal entries (e.g. 0.9).

    Returns
    -------
    np.ndarray
        Stressed covariance matrix: D @ R_stressed @ D + epsilon * I,
        where D = diag(stds) and R_stressed has target_correlation off-diagonal.
    """
    n = cov_matrix.shape[0]

    # Extract standard deviations from diagonal
    stds = np.sqrt(np.diag(cov_matrix))
    D = np.diag(stds)

    # Build stressed correlation matrix
    stressed_corr = np.full((n, n), target_correlation)
    np.fill_diagonal(stressed_corr, 1.0)

    # Convert back to covariance: D @ R_stressed @ D
    stressed_cov = D @ stressed_corr @ D

    # Epsilon regularization for numerical stability
    stressed_cov += np.eye(n) * 1e-8

    return stressed_cov
