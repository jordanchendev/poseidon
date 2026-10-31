"""Gate 4 -- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Conventions (all per-period, never annualized):

* ``sharpe`` -- observed Sharpe of the selected strategy per bar/period.
* ``n_obs`` -- T, the number of returns behind ``sharpe``.
* ``skew`` -- skewness of those returns.
* ``kurtosis`` -- **non-excess** (Pearson) kurtosis: 3.0 for normal returns.
  ``scipy.stats.kurtosis`` defaults to *excess*; pass ``fisher=False``.
* ``trial_sharpes`` -- one entry per trial in the family ledger, **including**
  errored / aborted-attempt / pruned trials (owner decision 2026-10-22). Their
  count is N; entries with a ``None``/non-finite Sharpe count toward N but not
  toward the variance. Trial Sharpes must use the same period as ``sharpe``.

    SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))
    DSR = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))

with g the Euler-Mascheroni constant and V the sample variance (ddof=1) of the
finite trial Sharpes. N = 1 gives SR0 = 0 (the Probabilistic Sharpe Ratio).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from scipy.stats import norm

from poseidon.pipeline.gates.base import GateResult, failed, invalid, is_finite, non_finite, passed
from poseidon.pipeline.gates.daily import Basis, DailyResult, moments

GATE = "dsr"
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class DsrInputs:
    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float  # non-excess: normal returns -> 3.0
    trial_sharpes: Sequence[float | None]


def trial_sharpes_from_ledger(rows: Iterable) -> list[float | None]:
    """Every ``pipeline_trials`` row of the family, whatever its ``state``."""

    return [getattr(row, "sharpe", None) for row in rows]


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """SR0: the expected maximum Sharpe of ``n_trials`` skill-less trials."""

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    return math.sqrt(variance) * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(sharpe: float, n_obs: int, skew: float, kurtosis: float, sr0: float) -> float:
    """Phi of the PSR z-score against benchmark ``sr0``; kurtosis is non-excess."""

    denominator = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        raise ValueError(f"non-positive PSR denominator {denominator}")
    return float(norm.cdf((sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denominator)))


def evaluate(inputs: DsrInputs, thresholds: Mapping) -> GateResult:
    dsr_min = float(thresholds["dsr_min"])
    min_obs = int(thresholds["dsr_min_obs"])
    bad = non_finite(
        [("sharpe", inputs.sharpe), ("n_obs", inputs.n_obs), ("skew", inputs.skew), ("kurtosis", inputs.kurtosis)]
    )
    if bad:
        return invalid(GATE, "non-finite inputs: " + ", ".join(bad))

    n_trials = len(inputs.trial_sharpes)
    finite = [float(value) for value in inputs.trial_sharpes if is_finite(value)]
    report = {
        "sharpe": float(inputs.sharpe),
        "n_obs": int(inputs.n_obs),
        "skew": float(inputs.skew),
        "kurtosis": float(inputs.kurtosis),
        "kurtosis_convention": "non_excess",
        "n_trials": n_trials,
        "n_trials_with_sharpe": len(finite),
        "trial_sharpe_variance": None,
        "sr0": None,
        "psr_denominator": None,
        "dsr": None,
        "dsr_min": dsr_min,
        "min_obs": min_obs,
    }
    if n_trials < 1:
        return failed(GATE, "no_trials", "trial ledger is empty; N must count the selected trial", "no trials", report)
    if inputs.n_obs < min_obs:
        return failed(
            GATE,
            "insufficient_observations",
            f"n_obs {inputs.n_obs} < {min_obs}",
            f"{inputs.n_obs} observations",
            report,
        )
    if inputs.kurtosis < 1.0 + inputs.skew**2:
        # Pearson kurtosis is bounded below by 1 + skew^2; smaller values are
        # almost certainly excess kurtosis passed by mistake.
        return failed(
            GATE,
            "invalid_moments",
            f"kurtosis {inputs.kurtosis:.3f} < 1 + skew^2; expected non-excess kurtosis (normal = 3)",
            "invalid kurtosis",
            report,
        )
    denominator = 1.0 - inputs.skew * inputs.sharpe + (inputs.kurtosis - 1.0) / 4.0 * inputs.sharpe**2
    report["psr_denominator"] = denominator
    if denominator <= 0:
        return failed(
            GATE,
            "invalid_denominator",
            f"PSR denominator 1 - skew*SR + (kurt-1)/4*SR^2 = {denominator:.4g} <= 0",
            "invalid DSR denominator",
            report,
        )
    if n_trials > 1:
        if len(finite) < 2:
            return failed(
                GATE,
                "insufficient_trial_sharpes",
                f"N={n_trials} but only {len(finite)} trials have a Sharpe; variance is undefined",
                "trial variance undefined",
                report,
            )
        variance = statistics.variance(finite)
        report["trial_sharpe_variance"] = variance
    else:
        variance = 0.0
    sr0 = expected_max_sharpe(n_trials, variance)
    dsr = deflated_sharpe(inputs.sharpe, inputs.n_obs, inputs.skew, inputs.kurtosis, sr0)
    report.update(sr0=sr0, dsr=dsr)
    headline = f"DSR {dsr:.2f} (N={n_trials})"
    if dsr < dsr_min:
        return failed(GATE, "dsr_below_min", f"DSR {dsr:.4f} < {dsr_min}", headline, report)
    return passed(GATE, headline, report)


def inputs_daily(result: DailyResult, basis: Basis, trial_sharpes: Sequence[float | None]) -> DsrInputs:
    """Daily classes (§7): the selected Sharpe and moments on the train backtest's daily metric.

    T = trading days (``dsr_min_obs`` 250 for daily classes). ``trial_sharpes`` is the family ledger
    as before; daily trials must be recorded with ``daily.moments`` of the same metric (the
    ``trials.make_record`` keys), so N and V stay on one basis. Missing moments (fewer than 2
    days, zero variance) are non-finite and fail ``invalid_input``.
    """

    stats = moments(result.metric(basis))
    return DsrInputs(
        sharpe=stats["sharpe_daily"],
        n_obs=stats["n_obs"],
        skew=stats["skew"],
        kurtosis=stats["kurtosis"],
        trial_sharpes=list(trial_sharpes),
    )
