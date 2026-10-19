"""Features for the Trendline Ruler strategy (Marci "one line, one ruler", TX 5m).

Mechanical port of a discretionary NQ 5m method:

  * ``session_clock`` -- local minute-of-day and trading-day id (the
    ``session_levels`` columns the strategy needs, without its NaN warmup).
  * ``htf_bias`` -- daily direction from COMPLETED trading days only: +1 when
    the previous day's close is above SMA(N) of daily closes and that SMA is
    rising versus ``htf_slope_lookback`` days earlier, -1 for the mirror,
    0 otherwise (also during warmup and when ``htf_mode="off"``).
  * ``trendline_ruler`` -- pivot detection plus one trendline state machine per
    side. The short side (``tls_*``) draws a descending line through two
    consecutive known pivot highs (P1 higher than P2) and measures the "ruler":
    the distance from the pattern's lowest low (P1 .. current bar, wicks
    included) up to the line directly above that low. Target = lowest low -
    ruler. The long side (``tll_*``) is the exact mirror (rising pivot lows,
    highest high, target = highest high + ruler); it is computed by running the
    same machine on sign-flipped prices.

Causality: a pivot at bar j (strict max of high[j-L .. j+R], ties are not
pivots) only becomes known at bar j+R, and every column at row t uses bars
<= t only. The current bar is included, which is safe because the backtest
runner hands a row to the strategy only once that bar has closed.

Mechanical decisions where the original is subjective:
  * A line forms on the bar its second pivot becomes known, unless a close
    between P2 and that bar already sits above the line (born broken -> P2
    becomes the next P1 candidate instead).
  * A line retires on: close above the line (``tls_broken``), low at or
    below the PREVIOUS bar's target (``tls_done``; the current bar's own low
    would always sit above a target derived from it), or age since P2 above
    ``max_line_age_bars``. After retirement the next line needs two new pivots
    whose bars lie after the retirement bar; pivots found while a line is
    active are discarded.
  * Leg counter: each new line increments the side's pattern index; a close
    beyond the outer Bollinger band (upper for shorts, lower for longs) starts
    a new leg (``leg_reset="bb_upper"``; ``"off"`` never resets).
  * Setup zone: P2 must print at or beyond the BB middle at P2's bar (short:
    P2.high >= middle) when ``setup_zone="mid_or_beyond"``.

Columns are NaN-free (sentinels: 0.0 levels, -1 ids/indices, 0 flags) so
``BacktestRunner._find_warmup_end`` does not clip history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature
from poseidon.data.features.range_reversion import (
    DEFAULT_DAY_ROLLOVER_MINUTE,
    DEFAULT_TZ,
    _local_index,
    trading_day_ids,
)
from poseidon.data.features.technical import BollingerBands

_SHORT_EXTREME = ("low_min", "low_idx")
_LONG_EXTREME = ("high_max", "high_idx")


def compute_htf_bias(
    ohlcv: pd.DataFrame,
    htf_sma_period: int = 20,
    htf_slope_lookback: int = 5,
    htf_mode: str = "sma_slope",
    tz: str = DEFAULT_TZ,
    day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
) -> pd.Series:
    """Daily SMA-slope bias mapped onto the intraday bars (completed days only)."""
    if htf_mode == "off":
        return pd.Series(0.0, index=ohlcv.index, name="htf_bias")
    if htf_mode != "sma_slope":
        raise ValueError(f"htf_mode invalid: {htf_mode!r}")
    day_id = trading_day_ids(_local_index(ohlcv, tz), day_rollover_minute)
    day_close = ohlcv["close"].astype(float).groupby(day_id).last()
    sma = day_close.rolling(htf_sma_period).mean()
    rising = sma > sma.shift(htf_slope_lookback)
    falling = sma < sma.shift(htf_slope_lookback)
    bias = pd.Series(0.0, index=day_close.index)
    bias[(day_close > sma) & rising] = 1.0
    bias[(day_close < sma) & falling] = -1.0
    # Each trading day sees the bias of the previous COMPLETED day.
    prev_bias = bias.shift(1).fillna(0.0)
    return pd.Series(pd.Series(day_id).map(prev_bias).to_numpy(), index=ohlcv.index, name="htf_bias")


def _pivot_known(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Boolean array: True at bar t when bar t-right is a strict pivot high."""
    h = pd.Series(high)
    left_max = h.rolling(left).max().shift(1)  # max of high[j-left .. j-1] at row j
    right_max = h.rolling(right).max()  # max of high[t-right+1 .. t] at row t
    pivot_at_j = (h > left_max).to_numpy()
    known = np.zeros(len(high), dtype=bool)
    if len(high) > right:
        known[right:] = pivot_at_j[:-right] & (h.shift(right) > right_max).to_numpy()[right:]
    return known


def _line_machine(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bb_outer: np.ndarray,
    bb_middle: np.ndarray,
    *,
    pivot_left: int,
    pivot_right: int,
    max_line_age_bars: int,
    leg_reset: str,
    setup_zone: str,
) -> dict[str, np.ndarray]:
    """Short-side (descending line over pivot highs) state machine in one pass.

    The long side calls this with sign-flipped prices (high <-> -low).
    """
    n = len(high)
    out = {
        "pivot_new": np.zeros(n),
        "active": np.zeros(n),
        "id": np.full(n, -1.0),
        "line": np.zeros(n),
        "slope": np.zeros(n),
        "ext": np.zeros(n),
        "ext_idx": np.full(n, -1.0),
        "ruler": np.zeros(n),
        "target": np.zeros(n),
        "broken": np.zeros(n),
        "done": np.zeros(n),
        "pattern_index": np.zeros(n),
        "zone_ok": np.zeros(n),
    }
    pivots = _pivot_known(high, pivot_left, pivot_right)

    line_id = -1
    leg_count = 0
    retired_at = -1
    cand: int | None = None  # P1 candidate: bar index of the last usable pivot high
    active = False
    p1 = p2 = 0
    slope = 0.0
    ext = 0.0
    ext_idx = 0
    prev_target = 0.0
    pattern_index = 0
    zone_ok = False

    for t in range(n):
        if leg_reset == "bb_upper" and close[t] > bb_outer[t]:
            leg_count = 0

        retired_now = False
        if active:
            line_t = high[p1] + slope * (t - p1)
            if close[t] > line_t:
                out["broken"][t] = 1.0
                retired_now = True
            elif low[t] <= prev_target:
                out["done"][t] = 1.0
                retired_now = True
            elif t - p2 > max_line_age_bars:
                retired_now = True
            if low[t] < ext:
                ext, ext_idx = low[t], t

        if pivots[t]:
            out["pivot_new"][t] = 1.0
            j = t - pivot_right
            if not active and j > retired_at:
                if cand is not None and high[j] < high[cand]:
                    s = (high[j] - high[cand]) / (j - cand)
                    span = np.arange(j + 1, t + 1)
                    if np.all(close[j + 1 : t + 1] <= high[cand] + s * (span - cand)):
                        active = True
                        p1, p2, slope = cand, j, s
                        seg = low[p1 : t + 1]
                        k = int(np.argmin(seg))
                        ext, ext_idx = float(seg[k]), p1 + k
                        line_id += 1
                        leg_count += 1
                        pattern_index = leg_count
                        zone_ok = setup_zone == "off" or high[p2] >= bb_middle[p2]
                        cand = None
                    else:
                        cand = j
                else:
                    cand = j

        if active:
            line_t = high[p1] + slope * (t - p1)
            ruler = high[p1] + slope * (ext_idx - p1) - ext
            target = ext - ruler
            out["id"][t] = float(line_id)
            out["line"][t] = line_t
            out["slope"][t] = slope
            out["ext"][t] = ext
            out["ext_idx"][t] = float(ext_idx)
            out["ruler"][t] = ruler
            out["target"][t] = target
            out["pattern_index"][t] = float(pattern_index)
            out["zone_ok"][t] = 1.0 if zone_ok else 0.0
            out["active"][t] = 0.0 if retired_now else 1.0
            prev_target = target
            if retired_now:
                active = False
                retired_at = t
                cand = None
    return out


def compute_trendline_ruler(
    ohlcv: pd.DataFrame,
    pivot_left: int = 3,
    pivot_right: int = 3,
    max_line_age_bars: int = 60,
    leg_reset: str = "bb_upper",
    setup_zone: str = "mid_or_beyond",
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Both sides of the trendline/ruler state machine as a NaN-free frame."""
    if leg_reset not in ("bb_upper", "off"):
        raise ValueError(f"leg_reset invalid: {leg_reset!r}")
    if setup_zone not in ("mid_or_beyond", "off"):
        raise ValueError(f"setup_zone invalid: {setup_zone!r}")
    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    close = ohlcv["close"].to_numpy(dtype=float)
    bb = BollingerBands().compute(ohlcv, period=bb_period, num_std=bb_std)
    upper = bb[f"bb_upper_{bb_period}"].to_numpy(dtype=float)
    middle = bb[f"bb_middle_{bb_period}"].to_numpy(dtype=float)
    lower = bb[f"bb_lower_{bb_period}"].to_numpy(dtype=float)
    kw = {
        "pivot_left": pivot_left,
        "pivot_right": pivot_right,
        "max_line_age_bars": max_line_age_bars,
        "leg_reset": leg_reset,
        "setup_zone": setup_zone,
    }
    # NaN bands (warmup) compare False: no leg reset, zone not ok.
    short = _line_machine(high, low, close, upper, middle, **kw)
    long_ = _line_machine(-low, -high, -close, -lower, -middle, **kw)
    for key in ("line", "slope", "ext", "target"):
        long_[key] = -long_[key]

    cols: dict[str, np.ndarray] = {}
    for prefix, side, (ext_name, ext_idx_name) in (
        ("tls", short, _SHORT_EXTREME),
        ("tll", long_, _LONG_EXTREME),
    ):
        for key, values in side.items():
            name = {"ext": ext_name, "ext_idx": ext_idx_name}.get(key, key)
            cols[f"{prefix}_{name}"] = values
    return pd.DataFrame(cols, index=ohlcv.index)


@register_feature
class SessionClock(BaseFeature):
    """Local minute-of-day and trading-day id only (NaN-free, unlike session_levels)."""

    name = "session_clock"
    description = "Taipei minute-of-day and trading day id"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        tz: str = DEFAULT_TZ,
        day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float)
        local_idx = _local_index(ohlcv, tz)
        return pd.DataFrame(
            {
                "tpe_minute": (local_idx.hour * 60 + local_idx.minute).astype("float64"),
                "trading_day": trading_day_ids(local_idx, day_rollover_minute).astype("float64"),
            },
            index=ohlcv.index,
        )


@register_feature
class HtfBias(BaseFeature):
    """Daily SMA-slope direction from completed trading days (see module docstring)."""

    name = "htf_bias"
    description = "Prior-day close vs rising/falling daily SMA -> +1 / -1 / 0"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        htf_sma_period: int = 20,
        htf_slope_lookback: int = 5,
        htf_mode: str = "sma_slope",
        tz: str = DEFAULT_TZ,
        day_rollover_minute: int = DEFAULT_DAY_ROLLOVER_MINUTE,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float)
        bias = compute_htf_bias(
            ohlcv,
            htf_sma_period=htf_sma_period,
            htf_slope_lookback=htf_slope_lookback,
            htf_mode=htf_mode,
            tz=tz,
            day_rollover_minute=day_rollover_minute,
        )
        return bias.to_frame()


@register_feature
class TrendlineRuler(BaseFeature):
    """Pivot trendlines with ruler targets for both sides (see module docstring)."""

    name = "trendline_ruler"
    description = "Descending/ascending pivot trendline, ruler target, leg index and setup zone"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        pivot_left: int = 3,
        pivot_right: int = 3,
        max_line_age_bars: int = 60,
        leg_reset: str = "bb_upper",
        setup_zone: str = "mid_or_beyond",
        bb_period: int = 20,
        bb_std: float = 2.0,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv, min_rows=pivot_left + pivot_right + 1):
            return pd.DataFrame(dtype=float)
        return compute_trendline_ruler(
            ohlcv,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            max_line_age_bars=max_line_age_bars,
            leg_reset=leg_reset,
            setup_zone=setup_zone,
            bb_period=bb_period,
            bb_std=bb_std,
        )
