"""Phase 92 Plan 92-04 — stormtrooper end-to-end smoke (scaffold).

Module-level pytestmark: skip on Mac collect; only runs inside qlib-research
container with STORMTROOPER=1 env var (Pattern S4 from Phase 94/95 PATTERNS.md).
Body filled by Plan 92-04 — runs run_ddg_da_comparison.py with smoke=True
(last 2 walk-forward folds only per D-25).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


@pytest.mark.stormtrooper
def test_ddg_da_compare_smoke():
    """D-25/D-26: smoke run produces with_ddg_da/, without_ddg_da/, comparison_summary.parquet, verdict.md."""
    pytest.skip("scaffold — Plan 92-04 fills body")
