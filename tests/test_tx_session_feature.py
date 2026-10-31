"""tx_session feature (DSL v2 slice 2): session model from bar close time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.features.session import compute_tx_session

FIXTURE = Path(__file__).parent / "fixtures" / "tx_5m_2026-09-01_2w.csv"


def _frame(local_stamps: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(local_stamps)).tz_localize("Asia/Taipei").tz_convert("UTC")
    n = len(idx)
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx[:n])


def _row(df: pd.DataFrame, stamp: str) -> pd.Series:
    key = pd.Timestamp(stamp, tz="Asia/Taipei").tz_convert("UTC")
    return df.loc[key]


def test_day_session_open_label_uses_close_time():
    df = _frame(["2026-09-01 08:40", "2026-09-01 08:45", "2026-09-01 08:50", "2026-09-01 13:40", "2026-09-01 13:45"])
    s = compute_tx_session(df, interval_minutes=5, bar_label="open")
    pre, first, second, last, stub = (s.iloc[i] for i in range(5))
    assert np.isnan(pre["s_id"])  # closes 08:45 -> before the open
    assert first["s_id"] == 20260901 * 10 + 1 and first["s_kind"] == 1
    assert first["s_bar"] == 0 and first["s_min_open"] == 5 and first["s_min_close"] == 295
    assert first["s_tod"] == 8 * 60 + 50
    assert second["s_bar"] == 1
    assert last["s_min_close"] == 0 and last["s_bar"] == 2
    # Thalassa's 13:45 stamp closes at 13:50 under the open-label convention -> outside the session.
    assert stub[list(s.columns)].isna().all()


def test_close_label_convention():
    df = _frame(["2026-09-01 08:45", "2026-09-01 08:50", "2026-09-01 13:45"])
    s = compute_tx_session(df, interval_minutes=5, bar_label="close")
    assert np.isnan(s.iloc[0]["s_id"])
    assert s.iloc[1]["s_min_open"] == 5 and s.iloc[1]["s_bar"] == 0
    assert s.iloc[2]["s_min_close"] == 0


def test_night_session_crosses_midnight_and_keeps_open_date():
    stamps = ["2026-09-01 15:00", "2026-09-01 23:55", "2026-09-02 00:00", "2026-09-02 04:55", "2026-09-02 05:00"]
    s = compute_tx_session(_frame(stamps), interval_minutes=5)
    night_id = 20260901 * 10 + 2
    assert list(s["s_id"].iloc[:4]) == [night_id] * 4
    assert list(s["s_bar"].iloc[:4]) == [0, 1, 2, 3]
    assert s.iloc[0]["s_min_open"] == 5
    assert s.iloc[1]["s_tod"] == 0  # closes at 00:00
    assert s.iloc[1]["s_min_close"] == 300
    assert s.iloc[3]["s_min_close"] == 0 and s.iloc[3]["s_min_open"] == 840
    assert np.isnan(s.iloc[4]["s_id"])  # closes 05:05


def test_friday_night_runs_into_saturday_and_monday_is_new_day():
    stamps = ["2026-09-04 15:00", "2026-09-05 01:00", "2026-09-05 04:55", "2026-09-07 08:45"]
    s = compute_tx_session(_frame(stamps), interval_minutes=5)
    assert list(s["s_id"].iloc[:3]) == [20260904 * 10 + 2] * 3
    assert s.iloc[3]["s_id"] == 20260907 * 10 + 1 and s.iloc[3]["s_bar"] == 0


def test_missing_bars_counted_by_actual_bars_and_minutes_by_clock():
    stamps = ["2026-09-01 08:45", "2026-09-01 09:30", "2026-09-01 09:35"]
    s = compute_tx_session(_frame(stamps), interval_minutes=5)
    assert list(s["s_bar"]) == [0, 1, 2]
    assert list(s["s_min_open"]) == [5, 50, 55]


def test_gap_between_sessions_and_holiday_session_absent():
    stamps = ["2026-09-01 13:40", "2026-09-01 14:00", "2026-09-03 08:45"]
    s = compute_tx_session(_frame(stamps), interval_minutes=5)
    assert s.iloc[0]["s_kind"] == 1
    assert np.isnan(s.iloc[1]["s_id"])  # 14:05 close: between sessions
    assert s.iloc[2]["s_id"] == 20260903 * 10 + 1


def test_15m_interval():
    s = compute_tx_session(_frame(["2026-09-01 08:45", "2026-09-01 13:30"]), interval_minutes=15)
    assert s.iloc[0]["s_min_open"] == 15
    assert s.iloc[1]["s_min_close"] == 0


def test_invalid_bar_label_rejected():
    with pytest.raises(ValueError):
        compute_tx_session(_frame(["2026-09-01 08:45"]), bar_label="mid")


def test_registered_through_feature_engine():
    df = _frame(["2026-09-01 08:45", "2026-09-01 08:50"])
    out = FeatureEngine().compute_from_df(df, [("tx_session", {"interval_minutes": 5})])
    assert {"s_id", "s_kind", "s_bar", "s_min_open", "s_min_close", "s_tod"} <= set(out.columns)


def _real_tx() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


def test_real_tx_bar_label_convention():
    """Thalassa TX 5m: stamps 08:45..13:45 and 15:00..05:00 (both ends present).

    Under the open-label convention the 13:45 and 05:00 stamps close after the
    session end, so they are the only real bars without a session.
    """
    df = _real_tx()
    local = df.index.tz_convert("Asia/Taipei").strftime("%H:%M")
    s = compute_tx_session(df, interval_minutes=5)
    outside = sorted(set(local[s["s_id"].isna().to_numpy()]))
    assert outside == ["05:00", "13:45"]
    day_first = df.index[(s["s_kind"] == 1) & (s["s_bar"] == 0)].tz_convert("Asia/Taipei").strftime("%H:%M")
    assert set(day_first) == {"08:45"}
    night_first = df.index[(s["s_kind"] == 2) & (s["s_bar"] == 0)].tz_convert("Asia/Taipei").strftime("%H:%M")
    # The fixture starts mid-night-session; every other night opens with the 15:00 stamp.
    assert set(night_first[1:]) == {"15:00"}
    per_session = s.dropna(subset=["s_id"]).groupby("s_id").size()
    assert per_session[per_session.index % 10 == 1].max() == 60
    assert per_session[per_session.index % 10 == 2].max() == 168
