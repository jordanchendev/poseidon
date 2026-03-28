"""Cross-market return alignment with point-in-time semantics.

Handles TW/US/crypto timezone differences by aligning on calendar date
with forward-fill for non-trading days (per D-07). The T+1 lag for
non-overlapping sessions means TW Monday close aligns with US Monday close
even though US Monday close happens ~15h later.

All functions enforce ``as_of`` filtering to prevent look-ahead bias (D-08).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Market calendar metadata (D-06)
# ---------------------------------------------------------------------------
# freq: "B" = standard business day (weekends excluded), "D" = every calendar day
# tz: canonical timezone for the market's trading session
#
# NOTE: For now we use standard "B" frequency which only excludes weekends.
# Market-specific holidays can be added later via CustomBusinessDay with
# holiday lists.

MARKET_CALENDARS: dict[str, dict] = {
    "tw_stock": {"freq": "B", "tz": "Asia/Taipei"},
    "tw_futures": {"freq": "B", "tz": "Asia/Taipei"},
    "us_stock": {"freq": "B", "tz": "US/Eastern"},
    "crypto_spot": {"freq": "D", "tz": "UTC"},
}


def compute_returns(close_prices: pd.Series, *, as_of: datetime) -> pd.Series:
    """Compute log returns from close prices, respecting point-in-time.

    Parameters
    ----------
    close_prices:
        Series with a DatetimeIndex and close prices.
    as_of:
        Point-in-time cutoff. Only data on or before this timestamp is used.

    Returns
    -------
    pd.Series
        Log returns with NaN dropped (first observation removed).
    """
    # Point-in-time filter: no data after as_of
    # Normalize tz: strip timezone from as_of to match tz-naive indices
    # (OHLCV data is stored as tz-naive dates in TimescaleDB)
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is not None and close_prices.index.tz is None:
        as_of_ts = as_of_ts.tz_localize(None)
    elif as_of_ts.tzinfo is None and close_prices.index.tz is not None:
        as_of_ts = as_of_ts.tz_localize(close_prices.index.tz)
    filtered = close_prices[close_prices.index <= as_of_ts]
    # Log returns: ln(P_t / P_{t-1})
    log_ret = np.log(filtered / filtered.shift(1))
    # Drop NaN (first row has no previous price)
    return log_ret.dropna()


def align_returns(
    return_series: dict[str, pd.Series],
    *,
    as_of: datetime,
) -> pd.DataFrame:
    """Align return series across markets with forward-fill.

    Parameters
    ----------
    return_series:
        Mapping of asset name -> return Series. Each Series has a
        DatetimeIndex (timezone-naive or normalized to date level).
    as_of:
        Point-in-time cutoff (D-08). Only data on or before this date is kept.

    Returns
    -------
    pd.DataFrame
        Columns = asset names, index = aligned dates, forward-filled (D-07).
        Empty DataFrame if no overlapping data within the as_of window.
    """
    as_of_ts_raw = pd.Timestamp(as_of)

    # Point-in-time filter on each series
    filtered: dict[str, pd.Series] = {}
    for name, series in return_series.items():
        # Normalize tz awareness for comparison
        as_of_ts = as_of_ts_raw
        if as_of_ts.tzinfo is not None and series.index.tz is None:
            as_of_ts = as_of_ts.tz_localize(None)
        elif as_of_ts.tzinfo is None and series.index.tz is not None:
            as_of_ts = as_of_ts.tz_localize(series.index.tz)
        cut = series[series.index <= as_of_ts]
        if not cut.empty:
            filtered[name] = cut

    if not filtered:
        return pd.DataFrame()

    # Combine into DataFrame (auto-aligns on index union)
    df = pd.DataFrame(filtered)

    # Forward-fill missing values (D-07): a non-trading day inherits
    # the most recent close's return
    df = df.ffill()

    # Drop rows where ALL values are still NaN (before any asset's first obs)
    df = df.dropna(how="all")

    return df
