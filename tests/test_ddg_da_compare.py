"""Phase 92 Plan 92-03 — comparison_summary.parquet writer unit tests (scaffolds).

D-20 mandates per-fold rows aligned with/without/Δ. Sample-based test on
synthetic with/without arrays.
"""

from __future__ import annotations

import pytest


def test_summary_alignment():
    """DDGDA-02: comparison_summary writes one row per fold with sharpe_with / sharpe_without / delta_sharpe columns."""
    pytest.skip("scaffold — Plan 92-03 fills body")


def test_summary_handles_unequal_fold_counts():
    """DDGDA-02: if with-DDG-DA and without-DDG-DA produce mismatched fold counts (RESEARCH OQ-2 edge case), writer raises ValueError."""
    pytest.skip("scaffold — Plan 92-03 fills body")
