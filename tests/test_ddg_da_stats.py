"""Phase 92 Plan 92-03 — paired bootstrap on per-window ΔSharpe (D-16 stat test).

Pattern C from 92-RESEARCH.md §Code Examples (lines 614-643). Pure numpy —
no qlib dependency, fully Mac-runnable.

W-5 fix (per checker review, applied in Plan 92-03): Plan 92-01 shipped
this file with an inline ``paired_bootstrap_delta_sharpe`` definition so
the test passed immediately (no Plan 92-03 dependency in Wave 0). Plan
92-03 then promoted the canonical implementation into
``poseidon.autoresearch.ddg_da_compare`` and this file now imports from
there — single source of truth, no drift. The two assertion tests below
exercise the library's behaviour contract.
"""

from __future__ import annotations

import numpy as np

from poseidon.autoresearch.ddg_da_compare import paired_bootstrap_delta_sharpe


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
