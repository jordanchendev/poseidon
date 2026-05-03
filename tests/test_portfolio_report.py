"""Phase 95 ACTIVATE-05 — Portfolio attribution / plotly report smoke.

Stormtrooper-only (Pattern S4). Wave 5 will load the PortAnaRecord pickles
emitted by the Wave 2 qrun pipeline and render the qlib + plotly portfolio
attribution report. The skip-with-reason explicitly notes the cross-wave
dependency so a partial Wave 5 run does not silently swallow the failure.
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


def test_portfolio_report_smoke() -> None:
    """Wave 5 implementation pending — depends on Wave 2 PortAnaRecord pickles."""
    pytest.importorskip("qlib")
    pytest.importorskip("plotly")
    pytest.skip("Wave 5 implementation pending — depends on Wave 2 PortAnaRecord pickles")
