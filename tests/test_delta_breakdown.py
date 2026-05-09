# Source: poseidon/tests/test_ddg_da_compare.py:48-49 (np.testing.assert_array_almost_equal pattern)
"""Phase 93 W0 — delta_breakdown computation unit test.

Wave 0 deliverable. Tests the three D-18 deltas + D-19 cost-delta sentence
generation:

  cost_delta_bps:        mean(NestedExecutor TWAP cost_bps) - mean(v18 |gap|/4 cost_bps)
  slippage_delta_bps:    mean(NestedExecutor TWAP slippage_bps) - 0 (naive = 0 slippage)
  fill_failure_rate_pct: NestedExecutor trigger-days with fill_failure=True / total × 100

Mac-collectable. Import deferred inside test body (RED until Wave 2 / Plan 93-03
lands the helper).

D-19 mandates a SINGLE-SENTENCE STRING surfaced in
``delta_breakdown["cost_delta_sentence"]`` (e.g. "NestedExecutor TWAP is ~2.0 bps
cheaper than v18 |gap|/4."). This sentence ends up appended to 93-RESEARCH.md
§ Cost Delta — fulfilling the ROADMAP literal "the cost delta is surfaced in
writing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_delta_breakdown_computes_three_deltas():
    """NESTEXEC-03 / D-18: cost_delta_bps + slippage_delta_bps + fill_failure_rate_pct."""
    from poseidon.backtest.nested_runner import compute_delta_breakdown

    # Synthetic per-trigger-day comparison frame (D-17 shape with all three algos)
    comparison = pd.DataFrame(
        {
            "trigger_date": pd.to_datetime(["2024-08-05", "2024-08-06", "2024-08-07"]),
            "naive_pair_pnl_bps": [10.0, 5.0, -2.0],
            "naive_slippage_bps_per_leg": [0.0, 0.0, 0.0],
            "naive_fill_failure": [False, False, False],
            "v18_gap4_pair_pnl_bps": [8.0, 3.0, -4.0],
            "v18_gap4_slippage_bps_per_leg": [0.0, 0.0, 0.0],
            "v18_gap4_cost_bps": [6.0, 6.0, 6.0],  # |gap|/4 cost
            "v18_gap4_fill_failure": [False, False, False],
            "nested_twap_pair_pnl_bps": [9.0, 4.0, -3.0],
            "nested_twap_slippage_bps_per_leg": [3.7, 3.5, 4.0],
            "nested_twap_cost_bps": [4.0, 4.0, 4.0],  # TWAP cheaper than |gap|/4
            "nested_twap_fill_failure": [False, False, True],
        }
    )

    deltas = compute_delta_breakdown(comparison)

    # Cost delta: TWAP 4.0 - |gap|/4 6.0 = -2.0 (cheaper)
    np.testing.assert_almost_equal(deltas["cost_delta_bps"], 4.0 - 6.0)
    # Slippage delta: mean([3.7, 3.5, 4.0]) - 0 = 3.733...
    np.testing.assert_almost_equal(deltas["slippage_delta_bps"], (3.7 + 3.5 + 4.0) / 3)
    # Fill failure rate: 1 of 3 trigger days = 33.33...%
    np.testing.assert_almost_equal(deltas["fill_failure_rate_pct"], 1 / 3 * 100)

    # D-19: cost_delta_sentence present and contains direction word
    assert "cost_delta_sentence" in deltas
    sentence = deltas["cost_delta_sentence"]
    assert isinstance(sentence, str)
    assert "cheaper" in sentence.lower() or "dearer" in sentence.lower() or "expensive" in sentence.lower()


def test_delta_breakdown_dearer_direction():
    """D-19: cost-delta sentence flips to "dearer"/"expensive" wording when
    NestedExecutor TWAP cost > v18 |gap|/4 cost.
    """
    from poseidon.backtest.nested_runner import compute_delta_breakdown

    comparison = pd.DataFrame(
        {
            "trigger_date": pd.to_datetime(["2024-08-05"]),
            "naive_pair_pnl_bps": [10.0],
            "naive_slippage_bps_per_leg": [0.0],
            "naive_fill_failure": [False],
            "v18_gap4_pair_pnl_bps": [8.0],
            "v18_gap4_slippage_bps_per_leg": [0.0],
            "v18_gap4_cost_bps": [4.0],  # |gap|/4 cheap baseline
            "v18_gap4_fill_failure": [False],
            "nested_twap_pair_pnl_bps": [7.0],
            "nested_twap_slippage_bps_per_leg": [3.7],
            "nested_twap_cost_bps": [10.0],  # TWAP MORE expensive than |gap|/4
            "nested_twap_fill_failure": [False],
        }
    )
    deltas = compute_delta_breakdown(comparison)

    # 10.0 - 4.0 = +6.0 (dearer)
    np.testing.assert_almost_equal(deltas["cost_delta_bps"], 6.0)
    sentence = deltas["cost_delta_sentence"].lower()
    assert "dearer" in sentence or "expensive" in sentence or "more" in sentence
