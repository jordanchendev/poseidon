"""opening_range / session_levels (DSL v2 slice 3) and the no-look-ahead truncation test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.features.session import compute_opening_range, compute_session_levels

FIXTURE = Path(__file__).parent / "fixtures" / "tx_5m_2026-09-01_2w.csv"

SESSION_SPECS = [
    ("tx_session", {"interval_minutes": 5}),
    ("opening_range", {"minutes": 15, "interval_minutes": 5}),
    ("opening_range", {"minutes": 30, "interval_minutes": 5}),
    ("opening_range", {"minutes": 60, "interval_minutes": 5}),
    ("session_levels", {"interval_minutes": 5}),
]


def _bars(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    stamps, o, h, low, c, v = zip(*rows, strict=True)
    idx = pd.DatetimeIndex(pd.to_datetime(list(stamps))).tz_localize("Asia/Taipei").tz_convert("UTC")
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "volume": v}, index=idx)


def test_opening_range_nan_until_window_complete():
    df = _bars(
        [
            ("2026-09-01 08:45", 100, 105, 99, 104, 10),
            ("2026-09-01 08:50", 104, 108, 103, 107, 10),
            ("2026-09-01 08:55", 107, 107, 96, 97, 10),  # closes 09:00 -> s_min_open 15
            ("2026-09-01 09:00", 97, 120, 90, 118, 10),  # after the window: must not move the OR
        ]
    )
    o = compute_opening_range(df, 15)
    assert o["s_or_high_15"].iloc[:2].isna().all()
    assert list(o["s_or_high_15"].iloc[2:]) == [108, 108]
    assert list(o["s_or_low_15"].iloc[2:]) == [96, 96]
    assert o["s_or_mid_15"].iloc[3] == 102 and o["s_or_range_15"].iloc[3] == 12


def test_opening_range_missing_session_start_has_no_range():
    df = _bars([("2026-09-01 09:30", 100, 101, 99, 100, 1), ("2026-09-01 09:35", 100, 101, 99, 100, 1)])
    assert compute_opening_range(df, 15).isna().all().all()


def test_opening_range_with_missing_bar_inside_window():
    df = _bars(
        [
            ("2026-09-01 08:45", 100, 105, 99, 104, 1),
            ("2026-09-01 08:55", 104, 106, 98, 100, 1),  # 08:50 missing; closes at 09:00
            ("2026-09-01 09:00", 100, 101, 99, 100, 1),
        ]
    )
    o = compute_opening_range(df, 15)
    assert np.isnan(o["s_or_high_15"].iloc[0])
    assert o["s_or_high_15"].iloc[1] == 106 and o["s_or_low_15"].iloc[1] == 98


def test_session_levels_running_and_previous_sessions():
    df = _bars(
        [
            ("2026-09-01 08:45", 100, 105, 99, 104, 10),
            ("2026-09-01 13:40", 104, 110, 100, 108, 30),
            ("2026-09-01 15:00", 108, 112, 107, 111, 0),
            ("2026-09-02 01:00", 111, 113, 95, 96, 10),
            ("2026-09-02 08:45", 96, 98, 94, 97, 10),
        ]
    )
    lv = compute_session_levels(df)
    # day 09-01
    assert list(lv["s_open"].iloc[:2]) == [100, 100]
    assert list(lv["s_high"].iloc[:2]) == [105, 110] and list(lv["s_low"].iloc[:2]) == [99, 99]
    vwap1 = ((105 + 99 + 104) / 3 * 10 + (110 + 100 + 108) / 3 * 30) / 40
    assert lv["s_vwap"].iloc[1] == pytest.approx(vwap1)
    assert lv[["s_prev_high", "s_prev_day_high"]].iloc[:2].isna().all().all()
    # night 09-01: zero volume first bar -> vwap NaN; previous session = the day session
    assert np.isnan(lv["s_vwap"].iloc[2])
    assert lv["s_vwap"].iloc[3] == pytest.approx((113 + 95 + 96) / 3)
    assert (lv["s_prev_high"].iloc[2], lv["s_prev_low"].iloc[2], lv["s_prev_close"].iloc[2]) == (110, 99, 108)
    assert lv["s_prev_day_close"].iloc[3] == 108
    # day 09-02: previous session = night, previous day session = day 09-01
    assert (lv["s_prev_high"].iloc[4], lv["s_prev_low"].iloc[4], lv["s_prev_close"].iloc[4]) == (113, 95, 96)
    assert (lv["s_prev_day_high"].iloc[4], lv["s_prev_day_low"].iloc[4], lv["s_prev_day_close"].iloc[4]) == (
        110,
        99,
        108,
    )


def test_out_of_session_bars_are_nan():
    df = _bars([("2026-09-01 08:45", 1, 1, 1, 1, 1), ("2026-09-01 13:45", 1, 1, 1, 1, 1)])
    lv = compute_session_levels(df)
    assert lv.iloc[1].isna().all()


def _synthetic_sessions(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = []
    for day in pd.date_range("2026-09-07", "2026-09-11", freq="D"):
        d = day.strftime("%Y-%m-%d")
        stamps += list(pd.date_range(f"{d} 08:45", f"{d} 13:45", freq="5min"))
        stamps += list(
            pd.date_range(f"{d} 15:00", f"{(day + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} 05:00", freq="5min")
        )
    keep = rng.random(len(stamps)) > 0.07  # random missing bars
    stamps = [s for s, k in zip(stamps, keep, strict=True) if k]
    n = len(stamps)
    close = 23000 + np.cumsum(rng.standard_normal(n) * 8)
    df = pd.DataFrame(
        {
            "open": close + rng.standard_normal(n),
            "high": close + np.abs(rng.standard_normal(n)) * 6,
            "low": close - np.abs(rng.standard_normal(n)) * 6,
            "close": close,
            "volume": rng.integers(0, 50, n).astype(float),
        },
        index=pd.DatetimeIndex(stamps).tz_localize("Asia/Taipei").tz_convert("UTC"),
    )
    return df


def _real_tx() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


def _assert_no_look_ahead(df: pd.DataFrame, step: int = 1) -> None:
    engine = FeatureEngine()
    full = engine.compute_from_df(df, SESSION_SPECS)
    new_cols = [c for c in full.columns if c not in df.columns]
    assert len(new_cols) == 6 + 3 * 4 + 10
    for k in range(0, len(df), step):
        trunc = engine.compute_from_df(df.iloc[: k + 1], SESSION_SPECS)
        a = trunc[new_cols].iloc[-1].to_numpy(dtype=float)
        b = full[new_cols].iloc[k].to_numpy(dtype=float)
        mismatch = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        assert not mismatch.any(), (k, [c for c, m in zip(new_cols, mismatch, strict=True) if m])
    # The columns are populated, not trivially NaN.
    assert full[new_cols].notna().any().all()


def test_no_look_ahead_synthetic():
    _assert_no_look_ahead(_synthetic_sessions())


def test_no_look_ahead_real_tx_two_weeks():
    _assert_no_look_ahead(_real_tx())
