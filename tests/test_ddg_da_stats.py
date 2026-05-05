"""Phase 92 Plan 92-03 — paired bootstrap on per-window ΔSharpe (D-16 stat test).

Pattern C from 92-RESEARCH.md §Code Examples (lines 614-643). Pure numpy —
no qlib dependency, fully Mac-runnable.

NOTE (W-5 per checker review): Plan 92-01 ships this file with an inline
`paired_bootstrap_delta_sharpe` definition so the test passes immediately
(no Plan 92-03 dependency in Wave 0). Plan 92-03 Step 6 then SWAPS the
inline definition for `from poseidon.autoresearch.ddg_da_compare import
paired_bootstrap_delta_sharpe` once the library lands — single source of
truth, no drift. The two assertion tests below continue to assert behavior
on whichever symbol is currently bound.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_delta_sharpe(
    sharpe_with: np.ndarray,
    sharpe_without: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap on per-window ΔSharpe.
    Both arrays MUST be aligned (same fold k in both). Returns mean ΔSharpe + CI95.
    """
    assert len(sharpe_with) == len(sharpe_without), "fold counts must match"
    rng = np.random.default_rng(seed)
    deltas = sharpe_with - sharpe_without
    n = len(deltas)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = deltas[idx].mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_delta_sharpe": float(deltas.mean()),
        "std_delta_sharpe": float(deltas.std()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_positive": float((boot_means > 0).mean()),
        "windows_improved_pct": float((deltas > 0).mean() * 100),
    }


def test_bootstrap_ci_shape():
    """DDGDA-02: bootstrap returns 6-key dict; CI95 brackets mean for normal-ish input."""
    rng = np.random.default_rng(0)
    sharpe_with = rng.normal(loc=0.5, scale=0.3, size=20)
    sharpe_without = rng.normal(loc=0.0, scale=0.3, size=20)
    result = paired_bootstrap_delta_sharpe(sharpe_with, sharpe_without, n_resamples=500, seed=0)
    assert set(result.keys()) == {
        "mean_delta_sharpe",
        "std_delta_sharpe",
        "ci95_low",
        "ci95_high",
        "p_positive",
        "windows_improved_pct",
    }
    assert result["ci95_low"] <= result["mean_delta_sharpe"] <= result["ci95_high"]
    # With loc difference 0.5 and σ 0.3, mean delta should be detectably positive
    assert result["mean_delta_sharpe"] > 0.2


def test_bootstrap_unequal_lengths_raises():
    """Defensive: paired bootstrap requires aligned fold counts."""
    import pytest as _pt

    with _pt.raises(AssertionError, match="fold counts must match"):
        paired_bootstrap_delta_sharpe(np.zeros(5), np.zeros(6))
