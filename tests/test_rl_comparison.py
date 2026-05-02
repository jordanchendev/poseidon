"""Phase 90 / Wave 2 — RL comparison-table tests.

Wave 0 scaffold (Plan 90-01 Task 3). Wave 2 fills in the comparison-table
emitter that produces the (PPO / OPDS / OPDT / TWAP) × (PA bps, IS bps,
slippage, Sh delta) DataFrame.
"""

import pytest


def test_table_shape():
    """Comparison table has 4 rows (algos) × N metric cols, dtype float."""
    pytest.skip("Wave 2 implements")
