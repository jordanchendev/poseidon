"""Phase 90 / Wave 1 — RL aggregator unit tests.

Wave 1 implementations (replacing Wave 0 stubs) of:

  * ``test_pa_to_bps_correct``                — D-15 separate + net slippage.
  * ``test_naive_baseline_matches_v18``       — D-16 |gap|/4 cost formula.
  * ``test_naive_intraday_ret_first_to_last_bar`` — Q1 RESOLVED golden test.
  * ``test_pair_metrics_golden``              — perf_full Sharpe/cum/MDD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.research.rl_aggregate import (
    BARS_PER_YEAR,
    aggregate_pair_metrics,
    annualised,
    compute_naive_intraday_ret,
    perf_full,
    v18_gap_cost_per_leg,
)


def _trigger_dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def test_pa_to_bps_correct():
    """PA(bps) → mean_per_leg + mean_net diagnostics; D-15 separate+net."""
    dates = _trigger_dates(3)
    tx_pa = pd.Series([5.0, 5.0, 5.0], index=dates)
    etf_pa = pd.Series([3.0, 3.0, 3.0], index=dates)
    naive_zero = pd.Series([0.0, 0.0, 0.0], index=dates)

    out = aggregate_pair_metrics(tx_pa, etf_pa, naive_zero, naive_zero)

    assert out["mean_per_leg_slippage_tx_bps"] == 5.0
    assert out["mean_per_leg_slippage_etf_bps"] == 3.0
    assert out["mean_net_slippage_bps"] == 2.0


def test_naive_baseline_matches_v18():
    """v18 |gap|/4 cost formula — D-16 verbatim ``np.maximum(0.00032, |gap|/4)``."""
    # Construct gap = (open - prev_close) / prev_close = [0.0, 0.02, -0.05]
    tx_open = pd.Series([100.0, 102.0, 95.0], index=_trigger_dates(3))
    prev_close = pd.Series([100.0, 100.0, 100.0], index=_trigger_dates(3))

    cost = v18_gap_cost_per_leg(tx_open, prev_close)

    # gap = [0, 0.02, -0.05] → |gap|/4 = [0, 0.005, 0.0125]
    # max(3.2bps=0.00032, |gap|/4) = [0.00032, 0.005, 0.0125]
    expected = [0.00032, 0.005, 0.0125]
    assert cost.round(6).tolist() == expected
    # Index preserved
    assert list(cost.index) == list(tx_open.index)


def test_naive_intraday_ret_first_to_last_bar():
    """Q1 RESOLVED — naive baseline = first-1m-open ⇒ last-1m-close per day."""
    # Day 1: 09:00–13:30, 270 bars; first_open=16500, last_close=16550
    # Day 2: 09:00–13:30, 270 bars; first_open=16600, last_close=16500
    day1 = pd.date_range("2020-03-02 09:00", periods=270, freq="1min", tz="UTC")
    day2 = pd.date_range("2020-03-03 09:00", periods=270, freq="1min", tz="UTC")
    idx = day1.append(day2)

    # Build OHLCV. We only care about first open + last close per day.
    n = len(idx)
    open_ = np.full(n, 16500.0)
    close_ = np.full(n, 16500.0)
    open_[0] = 16500.0  # day1 first
    close_[269] = 16550.0  # day1 last
    open_[270] = 16600.0  # day2 first
    close_[539] = 16500.0  # day2 last

    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close_) + 5.0,
            "low": np.minimum(open_, close_) - 5.0,
            "close": close_,
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )

    triggers = [pd.Timestamp("2020-03-02"), pd.Timestamp("2020-03-03")]
    out = compute_naive_intraday_ret(df, triggers)

    assert len(out) == 2
    # Day 1: (16550 - 16500) / 16500 = 0.0030303030...
    assert abs(float(out.iloc[0]) - (50.0 / 16500.0)) < 1e-9
    # Day 2: (16500 - 16600) / 16600 = -0.006024096...
    assert abs(float(out.iloc[1]) - (-100.0 / 16600.0)) < 1e-9

    # Index = trigger dates (normalised tz-naive)
    assert list(out.index) == [
        pd.Timestamp("2020-03-02"),
        pd.Timestamp("2020-03-03"),
    ]


def test_pair_metrics_golden():
    """Hand-computed Sharpe / cum / MDD on a deterministic 100-day fixture."""
    # Build a deterministic pair_ret series with known statistics.
    # Use naive=0 + tx_pa - etf_pa = pair_ret (in bps → /1e4).
    # If tx_pa - etf_pa is a fixed sequence of bps, pair_ret/day is known.
    rng = np.random.default_rng(42)
    n = 100
    dates = _trigger_dates(n)
    # tx_pa: random ~N(10, 5) bps; etf_pa: random ~N(2, 5) bps.
    # net per-day pair_ret in bps = tx_pa - etf_pa (since naive=0 and
    # pair_ret = (naive_tx + tx_pa/1e4) - (naive_etf - etf_pa/1e4)
    #          = (tx_pa + etf_pa) / 1e4 when naives=0).
    # Wait — sign convention: tx_realized = naive + tx_pa/1e4 (long buy),
    #                          etf_realized = naive - etf_pa/1e4 (short sell)
    # pair_ret = tx_realized - etf_realized = (tx_pa + etf_pa)/1e4.
    tx_pa = pd.Series(rng.normal(10.0, 5.0, n), index=dates)
    etf_pa = pd.Series(rng.normal(2.0, 5.0, n), index=dates)
    naive_zero = pd.Series(np.zeros(n), index=dates)

    out = aggregate_pair_metrics(tx_pa, etf_pa, naive_zero, naive_zero)

    # Hand-compute expected pair_ret = (tx_pa + etf_pa) / 1e4
    pair_ret_expected = (tx_pa + etf_pa) / 1e4
    eq = (1.0 + pair_ret_expected).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    expected_mdd = float(dd.min())
    expected_cum = float(eq.iloc[-1] - 1.0)
    expected_sh = annualised(pair_ret_expected.mean(), pair_ret_expected.std())

    assert abs(out["sh_full"] - expected_sh) < 1e-6
    assert abs(out["cum"] - expected_cum) < 1e-6
    assert abs(out["mdd"] - expected_mdd) < 1e-6
    # n_total should equal n (no NaNs in fixture)
    assert out["n_total"] == n
    # All 12 perf_full keys present
    expected_keys = {
        "n_total",
        "n_engaged",
        "freq",
        "sh_full",
        "sh_engaged_only",
        "sortino",
        "downside_std_ann",
        "cum",
        "mdd",
        "calmar",
        "pct_time_in_dd_ge_1pct",
        "max_dd_duration_bars",
    }
    assert expected_keys.issubset(out.keys())


def test_perf_full_keys_match_v18_driver():
    """perf_full output dict must have the exact 12 keys from v18 driver."""
    n = 30
    dates = _trigger_dates(n)
    net = pd.Series(np.full(n, 0.001), index=dates)
    engaged = pd.Series(np.ones(n, dtype=bool), index=dates)

    p = perf_full(net, engaged)
    # The 12 keys per test_tx_walkforward_v2.py:81-94
    expected = {
        "n_total",
        "n_engaged",
        "freq",
        "sh_full",
        "sh_engaged_only",
        "sortino",
        "downside_std_ann",
        "cum",
        "mdd",
        "calmar",
        "pct_time_in_dd_ge_1pct",
        "max_dd_duration_bars",
    }
    assert set(p.keys()) == expected


def test_bars_per_year_constant():
    """BARS_PER_YEAR matches v18 driver convention."""
    assert BARS_PER_YEAR == 240
