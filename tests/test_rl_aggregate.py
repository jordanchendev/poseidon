"""Phase 90 / Waves 1-2 — RL aggregator unit tests.

Wave 0 scaffold (Plan 90-01 Task 3). Filled in across Waves 1 & 2:

  * ``test_pa_to_bps_correct``         — Wave 1: PA(bps) formula correctness
  * ``test_naive_baseline_matches_v18``— Wave 1: 09:00 single-shot baseline
                                         must reproduce v18 OOS Sh 2.08 entry
  * ``test_pair_metrics_golden``       — Wave 2: paired (TX, 0050) cost
                                         metrics + Sharpe deltas vs golden
"""

import pytest


def test_pa_to_bps_correct():
    """PA(bps) = 10000 * (executed_avg_px - twap_px) / twap_px."""
    pytest.skip("Wave 1 implements")


def test_naive_baseline_matches_v18():
    """09:00 single-shot baseline matches v18 OOS Sh 2.08 entry."""
    pytest.skip("Wave 1 implements")


def test_pair_metrics_golden():
    """Paired (TX, 0050) cost metrics + Sharpe deltas vs golden snapshot."""
    pytest.skip("Wave 2 implements")
