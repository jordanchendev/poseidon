"""Phase 95 ACTIVATE-01 — Alpha158 production-signal smoke (stormtrooper-only).

Wave 0 stub: Pattern S4 (STORMTROOPER gate) + Pattern P9 (importorskip qlib
inside test body). Wave 1 lands the real Alpha158 evaluation harness — this
file currently exists only to satisfy the Mac-side ``pytest --collect-only``
gate (Pitfall 2) and reserve the test name in CI.
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
    """Resolve .planning/phases/95-*/smoke/{prong}/ from this file's path.

    poseidon/tests/test_alpha158_eval.py → parents[0]=tests/, parents[1]=poseidon/,
    parents[2]=aquarium root (where .planning/ lives). NEVER take a path from
    user input — D-13 + Pitfall 6.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_alpha158_eval_smoke() -> None:
    """Wave 1 implementation pending — real Alpha158 IC/Rank-IC eval harness."""
    pytest.importorskip("qlib")
    pytest.skip("Wave 1 implementation pending")
