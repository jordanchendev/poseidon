"""TAIFEX TX/MTX session model features (DSL v2).

Sessions are derived from each bar's *close time* in Asia/Taipei only; no
trading calendar is needed (a holiday simply has no bars, so its session does
not exist):

- day session   08:45-13:45, ``s_kind == 1``
- night session 15:00-05:00, ``s_kind == 2``; bars after midnight belong to
  the night session opened on the previous calendar day.

A bar belongs to a session when ``open < close_time <= end``. Bars outside
every session get NaN in all ``s_*`` columns, so no DSL rule can fire there.

Columns produced by ``tx_session``:

- ``s_id``        session open date ``YYYYMMDD * 10 + s_kind``
- ``s_kind``      1 = day, 2 = night
- ``s_bar``       0-based ordinal of the actual bar inside the session
- ``s_min_open``  minutes from the session open to the bar close
- ``s_min_close`` minutes from the bar close to the scheduled session end
- ``s_tod``       bar close time of day in minutes after local midnight

Every value only depends on the bar's own timestamp and earlier bars, so the
columns are causal (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature

SESSION_TZ = "Asia/Taipei"
DAY_OPEN_MIN = 8 * 60 + 45  # 08:45
DAY_CLOSE_MIN = 13 * 60 + 45  # 13:45
NIGHT_OPEN_MIN = 15 * 60  # 15:00
NIGHT_CLOSE_MIN = 5 * 60  # 05:00 (next calendar day)
MINUTES_PER_DAY = 24 * 60

SESSION_COLUMNS = ("s_id", "s_kind", "s_bar", "s_min_open", "s_min_close", "s_tod")
BAR_LABELS = ("open", "close")


def _time_index(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    """Bar timestamps as a tz-aware DatetimeIndex (naive input is taken as UTC)."""
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        idx = ohlcv.index
    elif "time" in ohlcv.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(ohlcv["time"]))
    else:
        raise ValueError("tx_session needs a DatetimeIndex or a 'time' column")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx


def compute_tx_session(
    ohlcv: pd.DataFrame,
    interval_minutes: int = 5,
    bar_label: str = "open",
) -> pd.DataFrame:
    """Return the ``s_*`` session columns aligned to ``ohlcv.index``.

    Args:
        ohlcv: OHLCV frame (DatetimeIndex or ``time`` column).
        interval_minutes: Bar length in minutes.
        bar_label: ``"open"`` when the timestamp labels the bar open (pandas
            resample default, used by Thalassa), ``"close"`` when it labels
            the bar close.
    """
    if bar_label not in BAR_LABELS:
        raise ValueError(f"bar_label must be one of {BAR_LABELS}, got {bar_label!r}")
    if int(interval_minutes) <= 0:
        raise ValueError("interval_minutes must be positive")

    n = len(ohlcv)
    out = pd.DataFrame({c: np.full(n, np.nan) for c in SESSION_COLUMNS}, index=ohlcv.index)
    if n == 0:
        return out

    local = _time_index(ohlcv).tz_convert(SESSION_TZ)
    if bar_label == "open":
        local = local + pd.Timedelta(minutes=int(interval_minutes))

    tod = (local.hour * 60 + local.minute + local.second / 60.0).to_numpy(dtype=float)
    ymd = (local.year * 10000 + local.month * 100 + local.day).to_numpy(dtype=np.int64)
    prev_ymd = local - pd.Timedelta(days=1)
    prev_ymd = (prev_ymd.year * 10000 + prev_ymd.month * 100 + prev_ymd.day).to_numpy(dtype=np.int64)

    day = (tod > DAY_OPEN_MIN) & (tod <= DAY_CLOSE_MIN)
    evening = tod > NIGHT_OPEN_MIN
    early = tod <= NIGHT_CLOSE_MIN
    night = evening | early
    in_session = day | night

    s_kind = np.where(day, 1.0, np.where(night, 2.0, np.nan))
    open_date = np.where(early, prev_ymd, ymd).astype(float)
    s_id = np.where(in_session, open_date * 10 + np.nan_to_num(s_kind), np.nan)

    min_open = np.where(
        day,
        tod - DAY_OPEN_MIN,
        np.where(evening, tod - NIGHT_OPEN_MIN, tod + MINUTES_PER_DAY - NIGHT_OPEN_MIN),
    )
    min_close = np.where(
        day,
        DAY_CLOSE_MIN - tod,
        np.where(evening, MINUTES_PER_DAY - tod + NIGHT_CLOSE_MIN, NIGHT_CLOSE_MIN - tod),
    )

    out["s_id"] = s_id
    out["s_kind"] = s_kind
    out["s_min_open"] = np.where(in_session, min_open, np.nan)
    out["s_min_close"] = np.where(in_session, min_close, np.nan)
    out["s_tod"] = np.where(in_session, tod, np.nan)
    s_bar = np.full(n, np.nan)
    if in_session.any():
        valid_ids = pd.Series(s_id[in_session])
        s_bar[in_session] = valid_ids.groupby(valid_ids).cumcount().to_numpy(dtype=float)
    out["s_bar"] = s_bar
    return out


@register_feature
class TxSession(BaseFeature):
    """TAIFEX day/night session columns (``s_*``), see module docstring."""

    name = "tx_session"
    description = "TX/MTX session id, kind, bar ordinal and minutes to open/close"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        interval_minutes: int = 5,
        bar_label: str = "open",
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(columns=list(SESSION_COLUMNS))
        return compute_tx_session(ohlcv, interval_minutes=interval_minutes, bar_label=bar_label)


OPENING_RANGE_FIELDS = ("high", "low", "mid", "range")
SESSION_LEVEL_COLUMNS = (
    "s_open",
    "s_high",
    "s_low",
    "s_vwap",
    "s_prev_high",
    "s_prev_low",
    "s_prev_close",
    "s_prev_day_high",
    "s_prev_day_low",
    "s_prev_day_close",
)


def opening_range_columns(minutes: int) -> list[str]:
    return [f"s_or_{f}_{int(minutes)}" for f in OPENING_RANGE_FIELDS]


def compute_opening_range(
    ohlcv: pd.DataFrame,
    minutes: int,
    interval_minutes: int = 5,
    bar_label: str = "open",
) -> pd.DataFrame:
    """Opening range of the first ``minutes`` of each session.

    NaN until the first bar whose ``s_min_open >= minutes``; from then on the
    max high / min low over the bars that close by ``open + minutes``. A
    session whose first observed bar is not the opening bar (data starts
    mid-session) has no opening range.
    """
    minutes = int(minutes)
    cols = opening_range_columns(minutes)
    out = pd.DataFrame({c: np.full(len(ohlcv), np.nan) for c in cols}, index=ohlcv.index)
    if len(ohlcv) == 0:
        return out
    s = compute_tx_session(ohlcv, interval_minutes=interval_minutes, bar_label=bar_label)
    sid = s["s_id"]
    valid = sid.notna()
    if not valid.any():
        return out

    first_min_open = s["s_min_open"].where(valid).groupby(sid).transform("first")
    starts_at_open = first_min_open <= int(interval_minutes)
    in_window = valid & (s["s_min_open"] <= minutes) & starts_at_open
    complete = valid & (s["s_min_open"] >= minutes) & starts_at_open

    hi = ohlcv["high"].astype(float).where(in_window).groupby(sid).cummax()
    lo = ohlcv["low"].astype(float).where(in_window).groupby(sid).cummin()
    hi = hi.groupby(sid).ffill().where(complete)
    lo = lo.groupby(sid).ffill().where(complete)

    out[cols[0]] = hi.reindex(ohlcv.index).to_numpy()
    out[cols[1]] = lo.reindex(ohlcv.index).to_numpy()
    out[cols[2]] = (out[cols[0]] + out[cols[1]]) / 2.0
    out[cols[3]] = out[cols[0]] - out[cols[1]]
    return out


def compute_session_levels(
    ohlcv: pd.DataFrame,
    interval_minutes: int = 5,
    bar_label: str = "open",
) -> pd.DataFrame:
    """Running levels of the current session plus completed previous sessions.

    ``s_open/s_high/s_low/s_vwap`` are running values of the current session
    up to and including the bar. ``s_prev_*`` are the final high/low/close of
    the previous session (either kind) and ``s_prev_day_*`` those of the most
    recent completed day session. Computed for both session kinds.
    """
    n = len(ohlcv)
    out = pd.DataFrame({c: np.full(n, np.nan) for c in SESSION_LEVEL_COLUMNS}, index=ohlcv.index)
    if n == 0:
        return out
    s = compute_tx_session(ohlcv, interval_minutes=interval_minutes, bar_label=bar_label)
    sid = s["s_id"]
    valid = sid.notna()
    if not valid.any():
        return out

    high = ohlcv["high"].astype(float).where(valid)
    low = ohlcv["low"].astype(float).where(valid)
    close = ohlcv["close"].astype(float).where(valid)
    volume = ohlcv["volume"].astype(float).where(valid)
    grouped_id = sid.where(valid)

    out["s_open"] = ohlcv["open"].astype(float).where(valid).groupby(grouped_id).transform("first").to_numpy()
    out["s_high"] = high.groupby(grouped_id).cummax().to_numpy()
    out["s_low"] = low.groupby(grouped_id).cummin().to_numpy()
    typical = (high + low + close) / 3.0
    cum_pv = (typical * volume).groupby(grouped_id).cumsum()
    cum_v = volume.groupby(grouped_id).cumsum()
    out["s_vwap"] = (cum_pv / cum_v.where(cum_v > 0)).to_numpy()

    per_session = pd.DataFrame(
        {
            "high": high.groupby(grouped_id).max(),
            "low": low.groupby(grouped_id).min(),
            "close": close.groupby(grouped_id).last(),
        }
    ).sort_index()
    prev = per_session.shift(1)
    is_day = (per_session.index.to_numpy() % 10) == 1
    prev_day = per_session[is_day].reindex(per_session.index).shift(1).ffill()

    for field in ("high", "low", "close"):
        out[f"s_prev_{field}"] = grouped_id.map(prev[field]).to_numpy()
        out[f"s_prev_day_{field}"] = grouped_id.map(prev_day[field]).to_numpy()
    return out


@register_feature
class OpeningRange(BaseFeature):
    """Opening range ``s_or_{high,low,mid,range}_{minutes}`` per session."""

    name = "opening_range"
    description = "Session opening range high/low/mid/range over the first N minutes"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        minutes: int = 30,
        interval_minutes: int = 5,
        bar_label: str = "open",
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(columns=opening_range_columns(minutes))
        return compute_opening_range(ohlcv, minutes, interval_minutes=interval_minutes, bar_label=bar_label)


@register_feature
class SessionLevels(BaseFeature):
    """Running session open/high/low/VWAP and previous session / day levels."""

    name = "session_levels"
    description = "Current session open/high/low/vwap and previous session/day high/low/close"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        interval_minutes: int = 5,
        bar_label: str = "open",
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(columns=list(SESSION_LEVEL_COLUMNS))
        return compute_session_levels(ohlcv, interval_minutes=interval_minutes, bar_label=bar_label)
