"""Phase 92 Plan 92-03 — per-fold Sharpe extractor unit tests (scaffolds).

Validates Pattern B from 92-RESEARCH.md §Code Examples lines 575-610 —
extractor walks qlib mlflow recorders and computes per-fold OOS Sharpe.
"""

from __future__ import annotations

import pytest


def test_per_fold_sharpe_shape():
    """DDGDA-02: extractor returns DataFrame with columns recorder_id/fold_start/fold_end/n_engaged/sharpe/cum."""
    pytest.skip("scaffold — Plan 92-03 fills body using make_synthetic_anchor_signal")


def test_per_fold_sharpe_aligns_with_threshold_logic():
    """DDGDA-02: trigger logic (long if pred < -threshold, short if pred > +threshold) reproducible from synthetic anchor signal."""
    pytest.skip("scaffold — Plan 92-03 fills body")
