"""VaR calculators: Parametric, Historical Simulation, Cornish-Fisher, Monte Carlo.

Convention: VaR/CVaR are positive numbers representing maximum loss fraction.
E.g., VaR=0.05 means 5% portfolio loss at the given confidence level.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from poseidon.risk.var.types import VaRMethod, VaRResult


class VaRCalculator:
    """Portfolio VaR calculator supporting four methods.

    All methods return :class:`VaRResult` with positive loss fractions.
    The ``confidence_levels`` tuple must contain exactly two values
    corresponding to the 95% and 99% confidence levels.
    """

    def __init__(
        self, confidence_levels: tuple[float, ...] = (0.95, 0.99)
    ) -> None:
        self.confidence_levels = confidence_levels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_result(
        method: str, as_of: datetime, portfolio_value: float, holding_period: int = 1
    ) -> VaRResult:
        """Return a VaRResult with all-zero VaR/CVaR values."""
        return VaRResult(
            method=method,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            as_of=as_of,
            computed_at=datetime.now(timezone.utc),
            portfolio_value=portfolio_value,
            holding_period=holding_period,
        )

    def compute_portfolio_returns(
        self, aligned_returns: pd.DataFrame, weights: np.ndarray
    ) -> np.ndarray:
        """Compute weighted portfolio returns from aligned asset returns.

        Parameters
        ----------
        aligned_returns:
            DataFrame where columns are assets and rows are dates.
        weights:
            Weight vector aligned to column order.

        Returns
        -------
        np.ndarray
            1-D array of portfolio returns.
        """
        return aligned_returns.values @ weights

    # ------------------------------------------------------------------
    # Parametric VaR (variance-covariance method)
    # ------------------------------------------------------------------

    def parametric(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_value: float,
        as_of: datetime,
        holding_period: int = 1,
    ) -> VaRResult:
        """Compute Parametric (variance-covariance) VaR and CVaR.

        Parameters
        ----------
        weights:
            Asset weight vector.  Empty array triggers zero result.
        cov_matrix:
            Covariance matrix of asset returns (n x n).
        portfolio_value:
            Current portfolio value in currency units.
        as_of:
            Point-in-time for the computation (PRISK-03).
        holding_period:
            Number of days.  VaR is scaled by sqrt(holding_period).

        Returns
        -------
        VaRResult
        """
        if len(weights) == 0:
            return self._zero_result(
                VaRMethod.PARAMETRIC.value, as_of, portfolio_value, holding_period
            )

        # Portfolio standard deviation with holding period scaling
        port_std = float(
            np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(holding_period)
        )

        conf_95, conf_99 = self.confidence_levels

        z_95 = norm.ppf(conf_95)
        z_99 = norm.ppf(conf_99)

        var_95 = z_95 * port_std
        var_99 = z_99 * port_std

        # CVaR (Expected Shortfall) under normality:
        # CVaR = sigma * phi(z) / (1 - alpha)
        cvar_95 = port_std * norm.pdf(z_95) / (1 - conf_95)
        cvar_99 = port_std * norm.pdf(z_99) / (1 - conf_99)

        return VaRResult(
            method=VaRMethod.PARAMETRIC.value,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            as_of=as_of,
            computed_at=datetime.now(timezone.utc),
            portfolio_value=portfolio_value,
            holding_period=holding_period,
        )

    # ------------------------------------------------------------------
    # Historical Simulation VaR
    # ------------------------------------------------------------------

    def historical_simulation(
        self,
        portfolio_returns: np.ndarray,
        portfolio_value: float,
        as_of: datetime,
        holding_period: int = 1,
    ) -> VaRResult:
        """Compute Historical Simulation VaR and CVaR.

        Parameters
        ----------
        portfolio_returns:
            1-D array of historical portfolio returns.
        portfolio_value:
            Current portfolio value.
        as_of:
            Point-in-time for the computation.
        holding_period:
            Number of days.  Returns scaled by sqrt(holding_period).

        Returns
        -------
        VaRResult
        """
        if len(portfolio_returns) == 0:
            return self._zero_result(
                VaRMethod.HISTORICAL.value, as_of, portfolio_value, holding_period
            )

        # Scale for holding period
        scaled = portfolio_returns * np.sqrt(holding_period)

        conf_95, conf_99 = self.confidence_levels

        def _hs_var_cvar(
            returns: np.ndarray, confidence: float
        ) -> tuple[float, float]:
            var = float(-np.percentile(returns, (1 - confidence) * 100))
            var = max(var, 0.0)  # ensure non-negative
            tail = returns[returns <= -var]
            if len(tail) > 0:
                cvar = float(-np.mean(tail))
            else:
                cvar = var
            cvar = max(cvar, 0.0)
            return var, cvar

        var_95, cvar_95 = _hs_var_cvar(scaled, conf_95)
        var_99, cvar_99 = _hs_var_cvar(scaled, conf_99)

        return VaRResult(
            method=VaRMethod.HISTORICAL.value,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            as_of=as_of,
            computed_at=datetime.now(timezone.utc),
            portfolio_value=portfolio_value,
            holding_period=holding_period,
        )

    # ------------------------------------------------------------------
    # Cornish-Fisher VaR
    # ------------------------------------------------------------------

    def cornish_fisher(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_returns: np.ndarray,
        portfolio_value: float,
        as_of: datetime,
        holding_period: int = 1,
    ) -> VaRResult:
        """Compute Cornish-Fisher adjusted VaR and CVaR.

        Uses the Cornish-Fisher expansion to adjust the normal quantile
        for skewness and excess kurtosis observed in the portfolio return
        distribution.

        Parameters
        ----------
        weights:
            Asset weight vector.
        cov_matrix:
            Covariance matrix of asset returns.
        portfolio_returns:
            1-D array of historical portfolio returns (for skew/kurt).
        portfolio_value:
            Current portfolio value.
        as_of:
            Point-in-time for the computation.
        holding_period:
            Number of days.

        Returns
        -------
        VaRResult
        """
        if len(weights) == 0 or len(portfolio_returns) < 3:
            return self._zero_result(
                VaRMethod.CORNISH_FISHER.value,
                as_of,
                portfolio_value,
                holding_period,
            )

        port_std = float(
            np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(holding_period)
        )

        # Fisher's skewness and excess kurtosis
        s = float(skew(portfolio_returns))
        k = float(kurtosis(portfolio_returns))  # excess kurtosis by default

        conf_95, conf_99 = self.confidence_levels

        def _cf_var_cvar(
            port_std: float, s: float, k: float, confidence: float
        ) -> tuple[float, float]:
            z = norm.ppf(confidence)
            # Cornish-Fisher expansion
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
            var = z_cf * port_std
            # CVaR approximation using adjusted quantile
            cvar = port_std * norm.pdf(z_cf) / (1 - confidence)
            return var, cvar

        var_95, cvar_95 = _cf_var_cvar(port_std, s, k, conf_95)
        var_99, cvar_99 = _cf_var_cvar(port_std, s, k, conf_99)

        return VaRResult(
            method=VaRMethod.CORNISH_FISHER.value,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            as_of=as_of,
            computed_at=datetime.now(timezone.utc),
            portfolio_value=portfolio_value,
            holding_period=holding_period,
            details={"skewness": s, "excess_kurtosis": k},
        )

    # ------------------------------------------------------------------
    # Monte Carlo VaR (Cholesky decomposition)
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_value: float,
        as_of: datetime,
        n_simulations: int = 10_000,
        holding_period: int = 1,
        seed: int | None = None,
    ) -> VaRResult:
        """Compute Monte Carlo VaR using Cholesky decomposition.

        Generates correlated random returns via Cholesky factorisation of the
        covariance matrix, then computes VaR/CVaR from the simulated portfolio
        return distribution.

        Parameters
        ----------
        weights:
            Asset weight vector.  Empty array triggers zero result.
        cov_matrix:
            Covariance matrix of asset returns (n x n).
        portfolio_value:
            Current portfolio value in currency units.
        as_of:
            Point-in-time for the computation.
        n_simulations:
            Number of Monte Carlo paths (default 10,000).
        holding_period:
            Number of days.  Simulated returns scaled by sqrt(holding_period).
        seed:
            Random seed for reproducibility (testing only; production uses None).

        Returns
        -------
        VaRResult
        """
        method_name = VaRMethod.MONTE_CARLO.value

        if len(weights) == 0:
            return self._zero_result(method_name, as_of, portfolio_value, holding_period)

        # Cholesky decomposition — graceful fallback for non-PD matrices
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            return self._zero_result(method_name, as_of, portfolio_value, holding_period)

        rng = np.random.default_rng(seed)
        uncorrelated = rng.standard_normal((n_simulations, len(weights)))
        correlated = uncorrelated @ L.T
        # Scale for holding period
        correlated *= np.sqrt(holding_period)
        portfolio_sims = correlated @ weights

        conf_95, conf_99 = self.confidence_levels

        var_95 = float(-np.percentile(portfolio_sims, (1 - conf_95) * 100))
        var_99 = float(-np.percentile(portfolio_sims, (1 - conf_99) * 100))

        # CVaR (Expected Shortfall): mean of tail beyond VaR
        tail_95 = portfolio_sims[portfolio_sims <= -var_95]
        cvar_95 = float(-tail_95.mean()) if len(tail_95) > 0 else var_95
        tail_99 = portfolio_sims[portfolio_sims <= -var_99]
        cvar_99 = float(-tail_99.mean()) if len(tail_99) > 0 else var_99

        return VaRResult(
            method=method_name,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            as_of=as_of,
            computed_at=datetime.now(timezone.utc),
            portfolio_value=portfolio_value,
            holding_period=holding_period,
            details={"n_simulations": n_simulations, "seed": seed},
        )
