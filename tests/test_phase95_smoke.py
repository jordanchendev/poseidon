"""Phase 95 Wave 6 — cross-prong E2E smoke (stormtrooper-only).

Wall-clock budget per CONTEXT D-31 / D-33:
  * Total budget:  ≤ 30 minutes for the full suite.
  * Per-prong budget: ≤ 6 minutes per prong (5 prongs × 6 min = 30 min ceiling).
  * Failure tolerance: any single prong may fail without aborting the suite;
    the suite-level verdict requires ≥ 4 of 5 prongs to PASS.

Five prongs covered (all currently Wave 6 stubs — implementations land in their
respective waves first; this file aggregates them for the cross-node smoke):
  1. ``test_poseidon_prong_alpha158``         — ACTIVATE-01 (Wave 1)
  2. ``test_poseidon_prong_qrun``             — ACTIVATE-02 (Wave 2)
  3. ``test_poseidon_prong_signal_analysis``  — ACTIVATE-03 (Wave 3)
  4. ``test_poseidon_prong_portfolio_report`` — ACTIVATE-05 (Wave 5)
  5. ``test_macminim4_data_health``           — ACTIVATE-04 (Wave 4, cross-node SSH)

Pattern S4 STORMTROOPER gate at module level. Cross-node prong (#5) shells out
to ``ssh macminim4-lan`` per the Phase 95 deployment topology — that prong only
runs when both STORMTROOPER=1 AND the SSH config is in place.
"""

from __future__ import annotations

import os
import subprocess  # SSH bridge to macminim4 is intentional (Wave 6)
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


def _smoke_dir(prong: str) -> Path:
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_poseidon_prong_alpha158() -> None:
    """Wave 6 — exercises ACTIVATE-01 production-signal Alpha158 path."""
    pytest.skip("Wave 6 implementation pending")


def test_poseidon_prong_qrun() -> None:
    """Wave 6 — exercises ACTIVATE-02 qrun YAML pipeline."""
    pytest.skip("Wave 6 implementation pending")


def test_poseidon_prong_signal_analysis() -> None:
    """Wave 6 — exercises ACTIVATE-03 calc_ic / SigAnaRecord."""
    pytest.skip("Wave 6 implementation pending")


def test_poseidon_prong_portfolio_report() -> None:
    """Wave 6 — exercises ACTIVATE-05 portfolio attribution report."""
    pytest.skip("Wave 6 implementation pending")


def test_macminim4_data_health() -> None:
    """Wave 6 — cross-node SSH to macminim4-lan running ACTIVATE-04 health check.

    Wave 6 will shell out to ``ssh macminim4-lan ... docker compose exec
    -T data-worker python -c '...'`` and assert the qlib_data_health Celery
    task wrote a row to PostgreSQL within the per-prong budget. Stub for now.

    The ``subprocess`` import is intentionally referenced here so that lint
    tools accept the module-top import that Wave 6 will need.
    """
    _ = subprocess  # mark the import as used until Wave 6 wires the SSH call
    pytest.skip("Wave 6 implementation pending")
