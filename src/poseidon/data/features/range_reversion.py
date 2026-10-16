"""Features for the Range 50% Reversion strategy (TX 5m, Kane-style PO3 sweep).

All features are computed vectorised on the full OHLCV frame and are strictly
causal: a row only uses bars at or before its own timestamp. "Current" bar
levels (``*_cur_*``) include the row itself, which is safe because the backtest
runner only hands a row to the strategy once that bar has closed.

Session semantics (Taiwan futures):
  * The index is converted to ``tz`` (default Asia/Taipei).
  * A trading day starts at the night-session open: bars at or after
    ``day_rollover_minute`` (default 14:30 -> 870) belong to the NEXT business
    day, matching the exchange / Thalassa 1d convention (night + next day).
  * 1h and 4h bars are clock-aligned in the local timezone.

Columns are intentionally NaN-free after the genuine warmup (first day / first
range window) so ``BacktestRunner._find_warmup_end`` does not clip history when
an optional companion (SMT) is only available for a short window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature

DEFAULT_TZ = "Asia/Taipei"
DEFAULT_DAY_ROLLOVER_MINUTE = 14 * 60 + 30  # 14:30 local: night session -> next trading day


def _local_index(ohlcv: pd.DataFrame, tz: str) -> pd.DatetimeIndex:
    """Return the frame's timestamps as a tz-aware DatetimeIndex in ``tz``.

    Accepts a DatetimeIndex (naive treated as UTC) or a ``time`` column.
    """
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        idx = ohlcv.index
    elif "time" in ohlcv.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(ohlcv["time"]))
    else:
        raise ValueError("OHLCV needs a DatetimeIndex or a 'time' column")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)


def trading_day_ids(local_idx: pd.DatetimeIndex, day_rollover_minute: int) -> np.ndarray:
    """Map each local timestamp to a trading-day ordinal (days since epoch).

    Bars at/after ``day_rollover_minute`` roll to the next calendar day; any
    resulting Saturday/Sunday rolls forward to Monday so the Friday night
    session joins Monday's trading day.
    """
    minute_of_day = local_idx.hour * 60 + local_idx.minute
    day = local_idx.normalize().tz_localize(None)
    rolled = day + pd.to_timedelta((minute_of_day >= day_rollover_minute).astype(int), unit="D")
    weekday = rolled.dayofweek.to_numpy()
    shift = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
    rolled = rolled + pd.to_timedelta(shift, unit="D")
    return (rolled.astype("int64") // (86_400 * 10**9)).to_numpy()


def _group_levels(
    high: pd.Series, low: pd.Series, open_: pd.Series, key: np.ndarray, prefix: str
) -> dict[str, pd.Series]:
    """Running (cur) and previous-group (prev) high/low plus group open for a bar id."""
    grp_h = high.groupby(key)
    grp_l = low.groupby(key)
    cur_high = grp_h.cummax()
    cur_low = grp_l.cummin()
    cur_open = open_.groupby(key).transform("first")

    per_high = grp_h.max()  # index = sorted unique keys (time-monotonic ids)
    per_low = grp_l.min()
    prev_high = per_high.shift(1)
    prev_low = per_low.shift(1)
    key_series = pd.Series(key, index=high.index)
    return {
        f"{prefix}_id": key_series.astype("float64"),
        f"{prefix}_cur_high": cur_high,
        f"{prefix}_cur_low": cur_low,
        f"{prefix}_cur_open": cur_open,
        f"{prefix}_prev_high": key_series.map(prev_high),
        f"{prefix}_prev_low": key_series.map(prev_low),
    }


def compute_session_levels(
    ohlcv: pd.DataFrame,
    tz: str = DEFAULT_TZ,
    day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
) -> pd.DataFrame:
    """Pure helper shared by SessionLevels and SMTDivergence."""
    local_idx = _local_index(ohlcv, tz)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    open_ = ohlcv["open"].astype(float)

    day_id = trading_day_ids(local_idx, day_rollover_minute)
    epoch_ns = local_idx.tz_localize(None).astype("int64")
    h1_id = (epoch_ns // (3_600 * 10**9)).to_numpy()
    h4_id = (epoch_ns // (4 * 3_600 * 10**9)).to_numpy()

    cols: dict[str, pd.Series] = {
        "tpe_minute": pd.Series((local_idx.hour * 60 + local_idx.minute).astype("float64"), index=ohlcv.index),
        "trading_day": pd.Series(day_id.astype("float64"), index=ohlcv.index),
    }
    cols.update(_group_levels(high, low, open_, h1_id, "h1"))
    cols.update(_group_levels(high, low, open_, h4_id, "h4"))
    cols.update(_group_levels(high, low, open_, day_id, "d1"))
    return pd.DataFrame(cols, index=ohlcv.index)


@register_feature
class SessionLevels(BaseFeature):
    """Local-time session clock plus 1h / 4h / trading-day running and previous levels."""

    name = "session_levels"
    description = "Taipei session minute, trading day id, 1h/4h/1d current+previous high/low/open"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        tz: str = DEFAULT_TZ,
        day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float)
        return compute_session_levels(ohlcv, tz=tz, day_rollover_minute=day_rollover_minute)


@register_feature
class DealingRange(BaseFeature):
    """Daily dealing range over the previous ``lookback_days`` completed trading days.

    Columns (N = lookback_days):
      range_high_N / range_low_N : extremes of the window (current day excluded)
      range_eq_N                 : 50% level (equilibrium)
      range_bias_N               : -1 when the window low is more recent than the
                                   window high (lower-low structure -> bearish),
                                   +1 otherwise. NaN during warmup.
    """

    name = "dealing_range"
    description = "Prior-N-day high/low/50% and directional bias"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        lookback_days: int = 20,
        tz: str = DEFAULT_TZ,
        day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float)
        local_idx = _local_index(ohlcv, tz)
        day_id = trading_day_ids(local_idx, day_rollover_minute)
        high = ohlcv["high"].astype(float)
        low = ohlcv["low"].astype(float)

        day_high = high.groupby(day_id).max()
        day_low = low.groupby(day_id).min()
        # Window over COMPLETED days only -> shift(1) before mapping back.
        win_high = day_high.rolling(lookback_days).max().shift(1)
        win_low = day_low.rolling(lookback_days).min().shift(1)

        hi_vals = day_high.to_numpy()
        lo_vals = day_low.to_numpy()
        n = len(hi_vals)
        bias = np.full(n, np.nan)
        for i in range(lookback_days, n):
            w_hi = hi_vals[i - lookback_days : i]
            w_lo = lo_vals[i - lookback_days : i]
            # argmax/argmin of the window: later position == more recent extreme
            bias[i] = -1.0 if int(np.argmin(w_lo)) > int(np.argmax(w_hi)) else 1.0
        bias_s = pd.Series(bias, index=day_high.index)

        key = pd.Series(day_id, index=ohlcv.index)
        n_ = lookback_days
        rh = key.map(win_high)
        rl = key.map(win_low)
        return pd.DataFrame(
            {
                f"range_high_{n_}": rh,
                f"range_low_{n_}": rl,
                f"range_eq_{n_}": (rh + rl) / 2.0,
                f"range_bias_{n_}": key.map(bias_s),
            },
            index=ohlcv.index,
        )


@register_feature
class FairValueGap(BaseFeature):
    """Most recent 3-bar fair value gap (ICT FVG), carried forward for ``max_age_bars``.

    Bullish FVG at bar i: low[i] > high[i-2]  -> gap = [high[i-2], low[i]]
    Bearish FVG at bar i: high[i] < low[i-2]  -> gap = [high[i], low[i-2]]

    Columns: fvg_bull_low, fvg_bull_high, fvg_bull_age, fvg_bear_low,
    fvg_bear_high, fvg_bear_age. Age is bars since the gap formed (0 on the
    forming bar); -1 and 0.0 levels mean "no live gap" so the frame stays
    NaN-free.
    """

    name = "fair_value_gap"
    description = "Latest bullish/bearish fair value gap levels and age"

    def compute(self, ohlcv: pd.DataFrame, max_age_bars: int = 24, **kwargs) -> pd.DataFrame:
        if not self._validate(ohlcv, min_rows=3):
            return pd.DataFrame(dtype=float)
        high = ohlcv["high"].to_numpy(dtype=float)
        low = ohlcv["low"].to_numpy(dtype=float)
        n = len(high)

        out = {
            "fvg_bull_low": np.zeros(n),
            "fvg_bull_high": np.zeros(n),
            "fvg_bull_age": np.full(n, -1.0),
            "fvg_bear_low": np.zeros(n),
            "fvg_bear_high": np.zeros(n),
            "fvg_bear_age": np.full(n, -1.0),
        }
        bull = None  # (low, high, formed_at)
        bear = None
        for i in range(n):
            if i >= 2:
                if low[i] > high[i - 2]:
                    bull = (high[i - 2], low[i], i)
                if high[i] < low[i - 2]:
                    bear = (high[i], low[i - 2], i)
            if bull is not None:
                age = i - bull[2]
                if age > max_age_bars:
                    bull = None
                else:
                    out["fvg_bull_low"][i] = bull[0]
                    out["fvg_bull_high"][i] = bull[1]
                    out["fvg_bull_age"][i] = float(age)
            if bear is not None:
                age = i - bear[2]
                if age > max_age_bars:
                    bear = None
                else:
                    out["fvg_bear_low"][i] = bear[0]
                    out["fvg_bear_high"][i] = bear[1]
                    out["fvg_bear_age"][i] = float(age)
        return pd.DataFrame(out, index=ohlcv.index)


@register_feature
class SMTDivergence(BaseFeature):
    """Smart-money-technique divergence against a correlated companion (e.g. TX vs TE).

    On the 1h clock bar: a bearish divergence is flagged when exactly one of the
    two instruments has pushed above its previous-1h high (the other failed to
    confirm). Bullish divergence mirrors it on previous-1h lows.

    Columns: smt_bear_div, smt_bull_div (1.0 / 0.0) and smt_known (1.0 when
    companion data for this bar is fresh within ``max_staleness_minutes``,
    else 0.0 with both divergence flags forced to 0.0).

    Companion data is injected by FeatureEngine.compute_with_companions() /
    FeatureComputer.compute_with_data() through ``companion_data``.
    """

    name = "smt_divergence"
    description = "1h sweep divergence vs companion instrument"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        companion_data: pd.DataFrame | None = None,
        companion_symbol: str = "",
        companion_market: str = "",
        tz: str = DEFAULT_TZ,
        day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
        max_staleness_minutes: int = 30,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float)
        zeros = pd.Series(0.0, index=ohlcv.index)
        if companion_data is None or companion_data.empty:
            return pd.DataFrame({"smt_bear_div": zeros, "smt_bull_div": zeros, "smt_known": zeros}, index=ohlcv.index)

        primary = compute_session_levels(ohlcv, tz=tz, day_rollover_minute=day_rollover_minute)
        comp_levels = compute_session_levels(companion_data, tz=tz, day_rollover_minute=day_rollover_minute)

        p_idx = _local_index(ohlcv, tz)
        c_idx = _local_index(companion_data, tz)
        comp_levels = comp_levels.set_index(c_idx).sort_index()
        comp_levels = comp_levels[~comp_levels.index.duplicated(keep="last")]
        aligned = comp_levels.reindex(
            p_idx,
            method="ffill",
            tolerance=pd.Timedelta(minutes=max_staleness_minutes),
        )
        aligned.index = ohlcv.index

        # Only trust the companion when it sits in the same 1h clock bar.
        same_h1 = aligned["h1_id"].to_numpy() == primary["h1_id"].to_numpy()
        known = aligned["h1_prev_high"].notna().to_numpy() & aligned["h1_prev_low"].notna().to_numpy() & same_h1

        p_up = (primary["h1_cur_high"] > primary["h1_prev_high"]).to_numpy()
        c_up = (aligned["h1_cur_high"] > aligned["h1_prev_high"]).to_numpy()
        p_dn = (primary["h1_cur_low"] < primary["h1_prev_low"]).to_numpy()
        c_dn = (aligned["h1_cur_low"] < aligned["h1_prev_low"]).to_numpy()

        bear = known & (p_up != c_up)
        bull = known & (p_dn != c_dn)
        return pd.DataFrame(
            {
                "smt_bear_div": bear.astype(float),
                "smt_bull_div": bull.astype(float),
                "smt_known": known.astype(float),
            },
            index=ohlcv.index,
        )
