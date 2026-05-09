# Source: poseidon/tests/test_ddg_da_compare.py:52-69 (synthetic baseline + alignment-failure ValueError pattern)
"""Phase 93 W0 — compare_to_baseline() join unit test.

Wave 0 deliverable. Tests Phase 93's ``compare_to_baseline()`` helper that joins
NestedExecutor TWAP per-trigger-day output against Phase 90 wave2-full-002
baseline (D-26).

Mac-collectable. Import deferred inside test bodies (RED until Wave 2 / Plan 93-03
lands the helper).

Test contracts:
- happy path: synthetic NestedExecutor frame + synthetic Phase 90 baseline →
  joined per-trigger-day rows × {Naive, v18 |gap|/4, NestedExecutor TWAP} cols
  per D-17
- disjoint trigger_date values → ValueError matching "alignment failed"
- CSV fallback: pass `local_dev/nested-executor/baseline-cache/comparison.csv`
  path explicitly (skip if absent)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _mk_nested_result_frame(trigger_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Synthetic NestedExecutor TWAP per-trigger-day output frame."""
    return pd.DataFrame(
        {
            "trigger_date": trigger_dates,
            "nested_twap_pair_pnl_bps": [9.0, 4.5, -2.5],
            "nested_twap_slippage_bps_per_leg": [3.7, 3.5, 4.1],
            "nested_twap_cost_bps": [4.0, 4.0, 4.0],
            "nested_twap_fill_failure": [False, False, False],
        }
    )


def test_compare_to_baseline_joins_on_trigger_date(make_synthetic_phase90_baseline):
    """NESTEXEC-03 / D-17: compare_to_baseline returns per-trigger-day rows × algos cols."""
    from poseidon.backtest.nested_runner import compare_to_baseline

    trigger_dates = [
        pd.Timestamp("2024-08-05"),
        pd.Timestamp("2024-08-06"),
        pd.Timestamp("2024-08-07"),
    ]
    nested_frame = _mk_nested_result_frame(trigger_dates)
    baseline_path, _baseline_df = make_synthetic_phase90_baseline(
        trigger_dates=trigger_dates,
        format="csv",
    )

    comparison = compare_to_baseline(nested_frame, baseline_path)

    # D-17: per-trigger-day rows, all 3 algos present
    assert len(comparison) == 3
    assert "trigger_date" in comparison.columns
    # Naive (Phase 90) cols
    assert any("naive" in c.lower() for c in comparison.columns)
    # v18 |gap|/4 cols
    assert any("v18_gap4" in c.lower() or "gap4" in c.lower() for c in comparison.columns)
    # NestedExecutor TWAP cols
    assert any("nested_twap" in c.lower() for c in comparison.columns)


def test_compare_to_baseline_raises_on_disjoint_dates(make_synthetic_phase90_baseline):
    """NESTEXEC-03 / D-26: matched row counts but disjoint trigger_date values raise."""
    from poseidon.backtest.nested_runner import compare_to_baseline

    nested_frame = _mk_nested_result_frame(
        [
            pd.Timestamp("2024-08-05"),
            pd.Timestamp("2024-08-06"),
            pd.Timestamp("2024-08-07"),
        ]
    )
    # Baseline at COMPLETELY different dates → join produces zero overlap
    baseline_path, _ = make_synthetic_phase90_baseline(
        trigger_dates=[
            pd.Timestamp("2024-12-01"),
            pd.Timestamp("2024-12-02"),
            pd.Timestamp("2024-12-03"),
        ],
        format="csv",
    )

    with pytest.raises(ValueError, match="alignment failed"):
        compare_to_baseline(nested_frame, baseline_path)


def test_compare_to_baseline_csv_fallback():
    """NESTEXEC-03 / D-16: explicit CSV path against staged Phase 90 baseline.

    Skip if the staged CSV is absent (e.g. fresh checkout where W0 baseline-cache
    hasn't been populated yet — happens in CI / clean clones).
    """
    from poseidon.backtest.nested_runner import compare_to_baseline

    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    csv_path = aquarium_root / "local_dev" / "nested-executor" / "baseline-cache" / "comparison.csv"
    if not csv_path.exists():
        pytest.skip(f"baseline CSV not staged at {csv_path}")

    # Synthetic minimal NestedExecutor frame — alignment may or may not succeed
    # depending on the staged CSV's row schema (rollup vs per-trigger-day);
    # the helper must accept the path argument without raising on path resolution.
    nested_frame = _mk_nested_result_frame([pd.Timestamp("2024-08-05")])
    try:
        compare_to_baseline(nested_frame, csv_path)
    except ValueError as exc:
        # Acceptable: the rollup CSV has different row schema than per-day frames;
        # the test only validates that the helper accepts a CSV path arg.
        assert "alignment" in str(exc).lower() or "schema" in str(exc).lower()
