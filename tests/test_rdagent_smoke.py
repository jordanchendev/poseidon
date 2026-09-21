"""Phase 91 / Wave W4 — RD-Agent end-to-end smoke test (CONTEXT D-28..D-31).

STORMTROOPER-only: this smoke depends on the qlib-research image plus
rdagent + OPENAI_API_KEY. ``pytestmark`` skips collection on any host
without ``STORMTROOPER=1`` set — the local Mac dev loop never runs it.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


def test_e2e_smoke():
    """Smoke: single end-to-end RD-Agent run on D-28 challenge."""
    pytest.skip("Plan 91-05 fills body — runs the real RD-Agent loop on stormtrooper")
