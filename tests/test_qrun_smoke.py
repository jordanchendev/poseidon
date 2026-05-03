"""Phase 95 ACTIVATE-02 — qrun YAML pipeline smoke.

Two distinct gates:
  1. ``test_qrun_yaml_loads`` runs on Mac (no STORMTROOPER guard) — Wave 2 will
     assert the tx_basis_vol.yml file resolves through the allowlist.
  2. ``test_qrun_smoke`` is stormtrooper-only (Pattern S4) — Wave 2 fills in the
     real qrun execution.

Pattern P9: ``import qlib`` only via ``pytest.importorskip`` inside test bodies.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _smoke_dir(prong: str) -> Path:
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_qrun_yaml_loads() -> None:
    """Wave 2 will load tx_basis_vol.yml and assert handler_class allowlists.

    No STORMTROOPER gate — must collect AND run on Mac so CI catches YAML
    breakage early. Wave 0 keeps this as a stub: the YAML file does not exist
    yet (Wave 2 creates it), so we cannot meaningfully execute the body.
    """
    pytest.skip("Wave 2: implement after tx_basis_vol.yml exists")


@pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)
def test_qrun_smoke() -> None:
    """Wave 2 implementation pending — full qrun pipeline execution."""
    pytest.importorskip("qlib")
    pytest.skip("Wave 2 implementation pending")
