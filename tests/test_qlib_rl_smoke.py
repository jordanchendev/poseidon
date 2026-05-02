"""Phase 90 / Wave 5 — qlib RL real-environment smoke test.

Wave 0 scaffold (Plan 90-01 Task 3). Wave 5 implements the single-trigger-day
smoke that runs PPO + TWAP through the simulator on real Thalassa 1-min
data and emits PA(bps) without crashing.

STORMTROOPER-only: this smoke depends on the qlib-research image plus
Thalassa-served 1-min OHLCV. ``pytestmark`` skips collection on any host
without ``STORMTROOPER=1`` set — the local Mac dev loop never runs it.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


def test_run_one_trigger_day():
    """Smoke: single basis-arb trigger day end-to-end through PPO + TWAP."""
    pytest.skip("Wave 5 implements")
