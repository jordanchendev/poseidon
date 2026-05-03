"""Phase 95 ACTIVATE-03 — Signal Analysis / IC report smoke (stormtrooper-only).

Wave 0 stub: Pattern S4 STORMTROOPER gate + Pattern P9 ``importorskip`` inside
the body. Wave 3 will run qlib's ``calc_ic`` / ``SigAnaRecord`` on the
``make_synthetic_anchor_signal`` fixture (added to conftest in Plan 95-01
Task 3) and persist artifacts to ``.planning/phases/95-*/smoke/signal_analysis/``.
"""

from __future__ import annotations

import os
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


def test_signal_analysis_smoke() -> None:
    """Wave 3 implementation pending — calc_ic + SigAnaRecord harness."""
    pytest.importorskip("qlib")
    pytest.skip("Wave 3 implementation pending")
