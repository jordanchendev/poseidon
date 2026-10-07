from __future__ import annotations

from itertools import product

import pandas as pd
import pytest

from poseidon.decision_loop.manifest import content_sha256
from poseidon.research import paired_review
from poseidon.research.ic_analysis import (
    compute_cross_sectional_rank_ic,
    compute_time_series_rank_ic,
)
from poseidon.research.paired_review import PairedReview, paired_hac_effect

ARMS = ("fundamental_only", "technical_only", "combined")
ROLES = ("incumbent", "candidate")
SAMPLE_MEMBERSHIP = [
    {"date": "2026-01-02", "symbol": "0050"},
    {"date": "2026-01-02", "symbol": "2317"},
    {"date": "2026-01-02", "symbol": "2330"},
]


def _review_contract() -> dict:
    return {
        "label": {"horizons": ["1_session", "5_sessions"]},
        "purge_gap": {"gap_eligible_sessions": 5, "overlap_method": "purged_embargo"},
        "regimes": ["risk_on", "risk_off"],
        "uncertainty_estimator": {
            "name": "ols_hac_intercept",
            "kernel": "bartlett",
            "maxlags_by_horizon": {"1_session": 0, "5_sessions": 4},
            "small_sample_correction": True,
            "alpha": "0.05",
        },
        "gates": {
            "minimum_symbols_per_date": 3,
            "minimum_dates_per_symbol": 3,
            "minimum_effective_paired_dates": 4,
            "minimum_coverage": "0.80",
        },
    }


def _paired_cells() -> list[dict]:
    return [
        {
            "version_role": role,
            "ablation_arm": arm,
            "terminal_state": "succeeded",
            "paired_sample_key_sha256": "a" * 64,
            "sample_membership": SAMPLE_MEMBERSHIP,
        }
        for role, arm in product(ROLES, ARMS)
    ]


def test_cross_sectional_and_time_series_ic_are_distinct_estimands() -> None:
    panel = pd.DataFrame(
        [
            {"date": date, "symbol": symbol, "signal": signal, "forward_return": scale * signal}
            for date, scale, signals in (
                ("2026-01-02", 10.0, (1.0, 2.0, 3.0)),
                ("2026-01-05", 0.1, (2.0, 4.0, 6.0)),
                ("2026-01-06", 1.0, (3.0, 6.0, 9.0)),
            )
            for symbol, signal in zip(("0050", "2317", "2330"), signals, strict=True)
        ]
    )

    cross_sectional = compute_cross_sectional_rank_ic(
        panel,
        "signal",
        "forward_return",
        "date",
        min_symbols=3,
    )
    time_series = compute_time_series_rank_ic(
        panel,
        "signal",
        "forward_return",
        "symbol",
        min_dates=3,
    )

    assert cross_sectional == {
        "series": {"2026-01-02": 1.0, "2026-01-05": 1.0, "2026-01-06": 1.0},
        "sample_counts": {"2026-01-02": 3, "2026-01-05": 3, "2026-01-06": 3},
        "eligible_group_count": 3,
        "total_group_count": 3,
        "coverage": 1.0,
    }
    assert time_series["series"] == {"0050": -0.5, "2317": -0.5, "2330": -0.5}
    assert time_series["coverage"] == 1.0


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (lambda cells: cells.pop(), "required_cell_missing"),
        (
            lambda cells: cells[-1].update(sample_membership=SAMPLE_MEMBERSHIP[:-1]),
            "sample_membership_mismatch",
        ),
        (lambda cells: cells[-1].pop("sample_membership"), "sample_membership_mismatch"),
        (lambda cells: cells[-1].update(terminal_state="optimizer_failed"), "required_cell_unavailable"),
    ],
)
def test_six_cells_require_identical_successful_sample_membership(mutate, reason_code: str) -> None:
    cells = _paired_cells()
    mutate(cells)

    result = PairedReview(_review_contract(), cells).validate()

    assert result["status"] == "unavailable"
    assert result["reason_code"] == reason_code
    assert result["paired_sample_digest"] is None
    assert result.get("metric") is None


def test_six_cell_validation_returns_one_membership_digest() -> None:
    result = PairedReview(_review_contract(), _paired_cells()).validate()

    assert result["status"] == "available"
    assert result["paired_sample_key_sha256"] == "a" * 64
    assert result["paired_sample_digest"] == content_sha256(SAMPLE_MEMBERSHIP)
    assert set(result["coverage_matrix"]) == {f"{role}:{arm}" for role, arm in product(ROLES, ARMS)}


def test_paired_hac_effect_uses_frozen_statsmodels_settings() -> None:
    index = pd.date_range("2026-01-02", periods=5, freq="B")
    incumbent = pd.Series([0.01, 0.02, -0.01, 0.00, 0.01], index=index)
    candidate = incumbent + pd.Series([0.01, 0.02, 0.03, 0.02, 0.04], index=index)

    result = paired_hac_effect(
        incumbent,
        candidate,
        kernel="bartlett",
        maxlags=1,
        small_sample_correction=True,
        alpha=0.05,
        minimum_effective_observations=4,
    )

    assert result["status"] == "available"
    assert result["effect"] == pytest.approx(0.024)
    assert result["sample_count"] == 5
    assert result["estimator"]["name"] == "ols_hac_intercept"
    assert result["estimator"]["version"]
    assert result["kernel"] == "bartlett"
    assert result["maxlags"] == 1
    assert result["small_sample_correction"] is True
    assert result["alpha"] == 0.05
    assert result["standard_error"] is not None
    assert len(result["confidence_interval"]) == 2


def test_paired_hac_effect_fails_closed_without_statsmodels(monkeypatch) -> None:
    def unavailable():
        raise ImportError("statsmodels is absent")

    monkeypatch.setattr(paired_review, "_load_statsmodels", unavailable)
    series = pd.Series([0.01, 0.02, 0.03, 0.04])

    result = paired_hac_effect(
        series,
        series + 0.01,
        kernel="bartlett",
        maxlags=1,
        small_sample_correction=True,
        alpha=0.05,
        minimum_effective_observations=4,
    )

    assert result["status"] == "unavailable"
    assert result["reason_code"] == "estimator_capability_unavailable"
    assert result["effect"] is None
    assert result["standard_error"] is None
    assert result["confidence_interval"] is None


def test_paired_hac_effect_is_inconclusive_below_frozen_minimum() -> None:
    series = pd.Series([0.01, 0.02, 0.03])

    result = paired_hac_effect(
        series,
        series + 0.01,
        kernel="bartlett",
        maxlags=1,
        small_sample_correction=True,
        alpha=0.05,
        minimum_effective_observations=4,
    )

    assert result["status"] == "inconclusive"
    assert result["reason_code"] == "insufficient_effective_observations"
    assert result["effect"] is None
