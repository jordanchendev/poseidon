"""Pure paired-review validation and frozen statistical inference."""

from __future__ import annotations

import re
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from poseidon.decision_loop.manifest import canonical_json, content_sha256

ARMS = ("fundamental_only", "technical_only", "combined")
ROLES = ("incumbent", "candidate")
REQUIRED_CELLS = tuple(product(ROLES, ARMS))


class ReviewCapabilityUnavailable(RuntimeError):
    """A frozen review capability is absent from the runtime."""


def _load_statsmodels():
    try:
        import statsmodels
        import statsmodels.api as sm
    except ImportError as error:
        raise ReviewCapabilityUnavailable("statsmodels is required for frozen HAC inference") from error
    return statsmodels.__version__, sm


def _empty_hac(status: str, reason_code: str, sample_count: int) -> dict:
    return {
        "status": status,
        "reason_code": reason_code,
        "effect": None,
        "standard_error": None,
        "confidence_interval": None,
        "sample_count": sample_count,
    }


def paired_hac_effect(
    incumbent: pd.Series,
    candidate: pd.Series,
    *,
    kernel: str,
    maxlags: int,
    small_sample_correction: bool,
    alpha: float,
    minimum_effective_observations: int,
) -> dict:
    """Estimate a paired candidate-minus-incumbent mean using frozen OLS HAC."""

    if maxlags < 0 or minimum_effective_observations <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid frozen HAC settings")
    if not incumbent.index.equals(candidate.index):
        return _empty_hac("unavailable", "paired_sample_mismatch", 0)

    paired = pd.concat(
        [incumbent.rename("incumbent"), candidate.rename("candidate")],
        axis=1,
    ).dropna()
    sample_count = len(paired)
    if sample_count < minimum_effective_observations:
        return _empty_hac("inconclusive", "insufficient_effective_observations", sample_count)

    try:
        version, sm = _load_statsmodels()
    except (ImportError, ReviewCapabilityUnavailable):
        return _empty_hac("unavailable", "estimator_capability_unavailable", sample_count)

    effects = (paired["candidate"] - paired["incumbent"]).astype(float).to_numpy()
    fitted = sm.OLS(effects, np.ones((sample_count, 1))).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "kernel": kernel,
            "use_correction": small_sample_correction,
        },
    )
    confidence_interval = fitted.conf_int(alpha=alpha)[0]
    return {
        "status": "available",
        "reason_code": None,
        "effect": float(fitted.params[0]),
        "standard_error": float(fitted.bse[0]),
        "confidence_interval": [float(confidence_interval[0]), float(confidence_interval[1])],
        "sample_count": sample_count,
        "estimator": {"name": "ols_hac_intercept", "version": version},
        "kernel": kernel,
        "maxlags": maxlags,
        "small_sample_correction": small_sample_correction,
        "alpha": alpha,
    }


def _cell_value(cell: Any, name: str):
    if isinstance(cell, dict):
        return cell.get(name)
    return getattr(cell, name, None)


def _normalized_membership(cell: Any) -> list:
    membership = _cell_value(cell, "sample_membership")
    if membership is None:
        metrics = _cell_value(cell, "metrics_json") or {}
        membership = metrics.get("sample_membership")
    if not isinstance(membership, list) or not membership:
        return []
    return sorted(membership, key=canonical_json)


def _maximum_horizon(contract: dict) -> int:
    values = []
    for horizon in contract.get("label", {}).get("horizons", []):
        match = re.match(r"^(\d+)", str(horizon))
        if match is None:
            raise ValueError("frozen horizons must begin with an eligible-session count")
        values.append(int(match.group(1)))
    if not values:
        raise ValueError("frozen horizons are required")
    return max(values)


class PairedReview:
    """Validate the exact successful six-cell matrix before computing metrics."""

    def __init__(self, campaign_contract: dict, cells: list[Any]) -> None:
        self.contract = campaign_contract
        self.cells = cells

    def validate(self) -> dict:
        coverage = {}
        indexed = {}
        for cell in self.cells:
            key = (_cell_value(cell, "version_role"), _cell_value(cell, "ablation_arm"))
            if key in indexed:
                return self._unavailable("duplicate_required_cell", coverage)
            indexed[key] = cell
            coverage[f"{key[0]}:{key[1]}"] = {
                "present": True,
                "terminal_state": _cell_value(cell, "terminal_state"),
            }

        if set(indexed) != set(REQUIRED_CELLS):
            for role, arm in REQUIRED_CELLS:
                coverage.setdefault(f"{role}:{arm}", {"present": False, "terminal_state": None})
            return self._unavailable("required_cell_missing", coverage)
        if any(_cell_value(cell, "terminal_state") != "succeeded" for cell in indexed.values()):
            return self._unavailable("required_cell_unavailable", coverage)

        sample_keys = {_cell_value(cell, "paired_sample_key_sha256") for cell in indexed.values()}
        if len(sample_keys) != 1 or not isinstance(next(iter(sample_keys)), str) or len(next(iter(sample_keys))) != 64:
            return self._unavailable("paired_sample_key_mismatch", coverage)
        memberships = [_normalized_membership(cell) for cell in indexed.values()]
        membership_json = {canonical_json(membership) for membership in memberships if membership}
        if any(not membership for membership in memberships) or len(membership_json) != 1:
            return self._unavailable("sample_membership_mismatch", coverage)

        if self.contract["purge_gap"]["gap_eligible_sessions"] < _maximum_horizon(self.contract):
            return self._unavailable("purge_gap_below_maximum_horizon", coverage)

        membership = memberships[0]
        return {
            "status": "available",
            "reason_code": None,
            "coverage_matrix": coverage,
            "paired_sample_key_sha256": next(iter(sample_keys)),
            "paired_sample_digest": content_sha256(membership),
        }

    @staticmethod
    def _unavailable(reason_code: str, coverage: dict) -> dict:
        return {
            "status": "unavailable",
            "reason_code": reason_code,
            "coverage_matrix": coverage,
            "paired_sample_key_sha256": None,
            "paired_sample_digest": None,
        }
