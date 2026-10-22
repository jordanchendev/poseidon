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
