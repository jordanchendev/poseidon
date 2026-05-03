"""Phase 90 / Wave 2 — RL comparison-table tests.

Wave 2 (Plan 90-03) implements the 6-column × 9-row comparison-table
emitter that Wave 5's verdict gate will consume. Tests cover:

* Shape (9 rows × 6 columns), column order, row order.
* PARTIAL note handling for PPO / OPDS placeholders.
* Sharpe Δ vs naive computation (Naive column = 0.0 by definition).
* v18 baseline column routes through ``rl_aggregate.v18_gap_cost_per_leg``
  (no formula re-implementation — D-16 single-source-of-truth invariant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poseidon.research.rl_aggregate import v18_gap_cost_per_leg
from poseidon.research.rl_comparison import (
    _COLUMNS,
    _ROWS,
    build_comparison_table,
    build_v18_gap_baseline,
)


def _make_realistic_algo_dict(sh: float, cum: float, mdd: float) -> dict:
    """Build a realistic per-algo summary dict in the aggregate_pair_metrics shape."""
    return {
        "n_total": 30,
        "n_engaged": 30,
        "freq": 1.0,
        "sh_full": sh,
        "sh_engaged_only": sh,
        "sortino": sh * 1.2,
        "downside_std_ann": 0.05,
        "cum": cum,
        "mdd": mdd,
        "calmar": cum / abs(mdd) if mdd != 0 else 0.0,
        "pct_time_in_dd_ge_1pct": 0.1,
        "max_dd_duration_bars": 5,
        "mean_per_leg_slippage_tx_bps": 1.5,
        "mean_per_leg_slippage_etf_bps": 1.2,
        "mean_net_slippage_bps": 0.3,
    }


def _make_v18_gap_input(seed: int = 42, n_days: int = 30):
    """Hand-crafted 30-day input for build_v18_gap_baseline."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02", periods=n_days, freq="B")
    prev_close = pd.Series(16500.0 + rng.normal(0, 50, n_days).cumsum(), index=dates)
    # Big-gap days will sit on the |gap|/4 branch; small-gap days hit floor.
    open_ = prev_close * (1.0 + rng.normal(0, 0.005, n_days))
    naive = pd.Series(rng.normal(0.0001, 0.01, n_days), index=dates)
    return open_, prev_close, naive


def test_table_shape():
    """build_comparison_table emits exactly 9 rows × 6 columns with the locked schema."""
    per_algo = {
        "twap": _make_realistic_algo_dict(sh=2.0, cum=0.05, mdd=-0.02),
        "vwap": _make_realistic_algo_dict(sh=2.1, cum=0.06, mdd=-0.018),
        "ppo": {"status": "PARTIAL", "error": "Wave 3 implements"},
        "opds": {"status": "PARTIAL", "error": "Wave 3 implements"},
    }
    naive = _make_realistic_algo_dict(sh=0.5, cum=0.05, mdd=-0.02)
    # Naive baseline has zero leg-slippage by construction (D-13).
    naive["mean_per_leg_slippage_tx_bps"] = 0.0
    naive["mean_per_leg_slippage_etf_bps"] = 0.0
    naive["mean_net_slippage_bps"] = 0.0

    open_, prev_close, naive_ret = _make_v18_gap_input()
    v18_gap = build_v18_gap_baseline(open_, prev_close, naive_ret, naive_ret)

    df = build_comparison_table(per_algo, naive, v18_gap)

    # Exact shape.
    assert df.shape == (9, 6), f"expected (9, 6), got {df.shape}"
    # Column order.
    assert list(df.columns) == [
        "Naive",
        "TWAP",
        "VWAP",
        "PPO",
        "OPDS",
        "v18_gap_over_4",
    ]
    # Row order.
    assert list(df.index) == _ROWS

    # PARTIAL notes for PPO / OPDS; OK for TWAP / VWAP / Naive / v18.
    assert df.loc["Note", "PPO"] == "PARTIAL"
    assert df.loc["Note", "OPDS"] == "PARTIAL"
    assert df.loc["Note", "TWAP"] == "OK"
    assert df.loc["Note", "VWAP"] == "OK"
    assert df.loc["Note", "Naive"] == "OK"
    assert df.loc["Note", "v18_gap_over_4"] == "OK"

    # Sharpe Δ vs naive: Naive column is 0.0 by definition.
    assert df.loc["Sharpe Δ vs naive", "Naive"] == 0.0
    # TWAP Sharpe (2.0) - Naive Sharpe (0.5) = 1.5
    assert df.loc["Sharpe Δ vs naive", "TWAP"] == pytest.approx(1.5, rel=1e-6)
    # VWAP Sharpe (2.1) - Naive Sharpe (0.5) = 1.6
    assert df.loc["Sharpe Δ vs naive", "VWAP"] == pytest.approx(1.6, rel=1e-6)

    # PARTIAL columns: Sharpe Δ NaN (cannot compute against PARTIAL Sharpe).
    assert pd.isna(df.loc["Sharpe Δ vs naive", "PPO"])
    assert pd.isna(df.loc["Sharpe Δ vs naive", "OPDS"])

    # Naive leg-slippage rows are 0.0 (D-13).
    assert df.loc["TX leg avg slippage (bps)", "Naive"] == 0.0
    assert df.loc["0050 leg avg slippage (bps)", "Naive"] == 0.0
    assert df.loc["Net pair slippage (bps)", "Naive"] == 0.0


def test_v18_baseline_uses_aggregate():
    """build_v18_gap_baseline routes through aggregate_pair_metrics, not a re-impl."""
    open_, prev_close, naive_ret = _make_v18_gap_input()
    out = build_v18_gap_baseline(open_, prev_close, naive_ret, naive_ret)

    # Proves the dict came from aggregate_pair_metrics (which adds these keys
    # on top of perf_full's 12-key contract — see Wave 1 SUMMARY).
    assert "mean_net_slippage_bps" in out
    assert "mean_per_leg_slippage_tx_bps" in out
    assert "mean_per_leg_slippage_etf_bps" in out
    # And from perf_full underneath.
    assert "sh_full" in out
    assert "cum" in out
    assert "mdd" in out
    assert "n_total" in out


def test_v18_gap_baseline_calls_v18_formula():
    """build_v18_gap_baseline cost_per_leg is byte-identical to v18_gap_cost_per_leg."""
    open_, prev_close, naive_ret = _make_v18_gap_input()

    # Manually compute what the function does internally.
    cost_per_leg = v18_gap_cost_per_leg(open_, prev_close)
    cost_bps_expected = cost_per_leg * 1e4

    # Both legs see -cost_bps as PA → mean_per_leg_slippage_*_bps = mean(-cost_bps)
    expected_tx_bps = float((-cost_bps_expected).mean())

    out = build_v18_gap_baseline(open_, prev_close, naive_ret, naive_ret)
    assert out["mean_per_leg_slippage_tx_bps"] == pytest.approx(expected_tx_bps, rel=1e-9)
    assert out["mean_per_leg_slippage_etf_bps"] == pytest.approx(expected_tx_bps, rel=1e-9)
    # Net = TX - ETF = 0 since both legs symmetric.
    assert out["mean_net_slippage_bps"] == pytest.approx(0.0, abs=1e-9)


def test_partial_algo_keeps_table_shape():
    """A fully-PARTIAL per_algo still produces a (9, 6) DataFrame."""
    per_algo = {
        "twap": {"status": "PARTIAL"},
        "vwap": {"status": "PARTIAL"},
        "ppo": {"status": "PARTIAL"},
        "opds": {"status": "PARTIAL"},
    }
    naive = _make_realistic_algo_dict(sh=0.5, cum=0.05, mdd=-0.02)
    open_, prev_close, naive_ret = _make_v18_gap_input()
    v18_gap = build_v18_gap_baseline(open_, prev_close, naive_ret, naive_ret)

    df = build_comparison_table(per_algo, naive, v18_gap)
    assert df.shape == (9, 6)
    # All 4 algo columns are PARTIAL.
    for col in ("TWAP", "VWAP", "PPO", "OPDS"):
        assert df.loc["Note", col] == "PARTIAL"


def test_columns_constant_locked():
    """Column-order constant matches the verdict-gate contract (D-16)."""
    assert _COLUMNS == ["Naive", "TWAP", "VWAP", "PPO", "OPDS", "v18_gap_over_4"]
