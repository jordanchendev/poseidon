"""Phase 90 / Wave 1 — qlib RL runner tests.

Wave 0 scaffold (Plan 90-01 Task 3). Wave 1 fills in the actual runner
that drives all four algos (PPO / OPDS / OPDT / TWAP-baseline) end-to-end
through ``qlib.rl.contrib.train_onpolicy``.

The module-level ``importorskip("qlib")`` keeps this file collectable on
machines without the qlib stack installed (Mac dev → no torch). Inside the
qlib-research container on stormtrooper, qlib resolves and the body runs
(currently a placeholder skip — Wave 1 owns the real assertions).
"""

import pytest

pytest.importorskip("qlib")


def test_run_all_four_algos():
    """Drive PPO + OPDS + OPDT + TWAP through the runner and aggregate metrics."""
    pytest.skip("Wave 1 implements")
