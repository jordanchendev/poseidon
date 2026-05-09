# Source: poseidon/tests/test_ddg_da_compare.py:30-49 (pd.read_parquet round-trip + column-set assertion)
"""Phase 93 W0 — fill_log.parquet D-21 schema unit test.

Wave 0 deliverable. Synthetic indicator_dict (via conftest fixture
``make_synthetic_indicator_dict``) → ``harvest_fill_log()`` → parquet round-trip
→ assert D-21 schema columns + fill_failure semantics.

Mac-collectable. ``harvest_fill_log`` import deferred inside test bodies (RED
until Wave 2 / Plan 93-03 lands the helper).

D-21 schema (one row per fill-event; ≤ N_triggers × twap_window × 2 legs):
  run_id, trigger_date, decision_ts, leg, fill_ts,
  planned_qty, filled_qty, fill_price,
  bar_open, bar_high, bar_low, bar_close,
  slippage_bps, cost_bps, fill_failure
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

D21_COLUMNS = {
    "run_id",
    "trigger_date",
    "decision_ts",
    "leg",
    "fill_ts",
    "planned_qty",
    "filled_qty",
    "fill_price",
    "bar_open",
    "bar_high",
    "bar_low",
    "bar_close",
    "slippage_bps",
    "cost_bps",
    "fill_failure",
}


def test_fill_log_schema_columns(tmp_path: Path, make_synthetic_indicator_dict):
    """NESTEXEC-02 / D-21: fill_log.parquet has the D-21 byte-frozen columns."""
    from poseidon.backtest.nested_runner import harvest_fill_log

    indicator_dict, _expected_n_rows = make_synthetic_indicator_dict(
        n_triggers=2,
        twap_window_minutes=5,
    )
    fill_log = harvest_fill_log(
        indicator_dict,
        run_id="test-run",
        triggers=[pd.Timestamp("2024-08-05"), pd.Timestamp("2024-08-06")],
        twap_window_minutes=5,
    )
    out_path = tmp_path / "fill_log.parquet"
    fill_log.to_parquet(out_path)
    assert out_path.exists()

    df = pd.read_parquet(out_path)
    assert D21_COLUMNS.issubset(set(df.columns)), f"missing D-21 columns: {D21_COLUMNS - set(df.columns)}"
    # ≤ N_triggers × twap_window × 2 legs (some bars may collapse if synthetic
    # produces identical (fill_ts, leg) keys, hence ≤ not ==)
    assert len(df) <= 2 * 5 * 2


def test_fill_failure_set_on_residual_qty(make_synthetic_indicator_dict):
    """NESTEXEC-02 / D-21: ``fill_failure=True`` on rows where filled_qty < planned_qty.

    Synthetic indicator_dict where one leg's deal_amount sums to less than the
    planned per-leg notional → harvest_fill_log produces residual rows with
    fill_failure=True.
    """
    from poseidon.backtest.nested_runner import harvest_fill_log

    indicator_dict, _ = make_synthetic_indicator_dict(
        n_triggers=1,
        twap_window_minutes=5,
    )
    # Halve the deal_amount on one leg → residual qty → fill_failure=True
    inner = indicator_dict["1min"]
    inner["deal_amount"] = inner["deal_amount"] * 0.5

    fill_log = harvest_fill_log(
        indicator_dict,
        run_id="test-residual",
        triggers=[pd.Timestamp("2024-08-05")],
        twap_window_minutes=5,
    )
    assert "fill_failure" in fill_log.columns
    # At least one row must have fill_failure=True (the residual marker row OR
    # rows where filled_qty < planned_qty per bar — both are valid encodings).
    assert fill_log["fill_failure"].any()
