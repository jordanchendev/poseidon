"""Continuous futures series from per-contract daily quotes, rolled on the settlement calendar.

The position carried into trading day t is always valued on its own
contract: ``pnl_points[t] = prev_close[t] - close[t-1]`` where both prices
belong to ``prev_contract``. On a roll day the position moves to the next
contract at that day's close, so the price gap between the two contracts
(``roll_gap_points``) is never booked as PnL. Two back-adjusted prices end
at the last row's ``close``:

- ``close_adj`` (additive): shifted by the later roll gaps, so
  ``close_adj.diff() == pnl_points``. Over long histories it can reach zero
  or below (TX from 1998 with ``days_before=1`` does), so do not take ratios
  of it.
- ``close_adj_ratio`` (multiplicative): ``close_adj_ratio.pct_change() == ret``;
  use it for return-based signals (moving-average bands, volatility).

Roll rule ``days_before = k``: contract c is held through the close of the
trading day k days before its last trading day, then the next contract is
held. k = 1 is the design's ``roll.before_settlement {days_before: 1}``
(§5.5). k = 0 holds through the last trading day and exits at the published
final settlement price (``exit_at_fsp``); for contracts up to 2008-11 that
price was fixed the next morning.

Prices: day-session close, falling back to the daily settlement price when a
contract did not trade. ``open/high/low`` and ``night_high/night_low`` are
the carried contract's day session and the night session TAIFEX dates t
(which trades before the day session; see ``taifex`` module docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTINUOUS_COLUMNS = [
    "trade_date",
    "contract",
    "close",
    "close_adj",
    "close_adj_ratio",
    "prev_contract",
    "prev_close",
    "open",
    "high",
    "low",
    "night_high",
    "night_low",
    "pnl_points",
    "ret",
    "roll",
    "roll_gap_points",
    "exit_at_fsp",
]


class ContinuousSeriesError(ValueError):
    """The quotes cannot support a continuous series under the requested rule."""


def roll_dates(calendar: pd.DataFrame, trading_days: pd.DatetimeIndex, days_before: int) -> pd.Series:
    """Per contract month: the trading day at whose close the position leaves it."""

    out = {}
    for record in calendar.itertuples(index=False):
        ltd = record.last_trading_day
        if ltd in trading_days:
            position = trading_days.get_loc(ltd) - days_before
            out[record.contract_month] = trading_days[max(position, 0)]
        elif ltd > trading_days[-1]:  # projected: count business days back from the projected date
            out[record.contract_month] = ltd - pd.offsets.BDay(days_before)
        else:
            raise ContinuousSeriesError(
                f"last trading day {ltd.date()} of {record.contract_month} is not a trading day"
            )
    return pd.Series(out, dtype="datetime64[ns]")


def continuous_series(
    quotes: pd.DataFrame, calendar: pd.DataFrame, product: str, *, days_before: int = 1
) -> pd.DataFrame:
    if days_before < 0:
        raise ContinuousSeriesError("days_before must be >= 0")
    monthly = quotes[(quotes["product"] == product) & quotes["contract_month"].str.fullmatch(r"\d{6}")]
    day = monthly[monthly["session"] == "day"]
    night = monthly[monthly["session"] == "night"]
    if not len(day):
        raise ContinuousSeriesError(f"no {product} day-session quotes")
    price = day.pivot(index="trade_date", columns="contract_month", values="close")
    price = price.fillna(day.pivot(index="trade_date", columns="contract_month", values="settlement"))
    listed = day.groupby("trade_date")["contract_month"].agg(set)
    pivots = {
        name: day.pivot(index="trade_date", columns="contract_month", values=name) for name in ("open", "high", "low")
    }
    night_pivots = {
        name: night.pivot(index="trade_date", columns="contract_month", values=name).reindex(price.index)
        for name in ("high", "low")
    }
    dates = pd.DatetimeIndex(price.index)
    cal = calendar[calendar["product"] == product].sort_values("last_trading_day", kind="mergesort")
    exits = roll_dates(cal, dates, days_before)
    ltd = cal.set_index("contract_month")["last_trading_day"]
    fsp = cal.set_index("contract_month")["final_settlement_price"]
    order = list(cal["contract_month"])

    held = []
    pointer = 0
    for t in dates:
        while pointer < len(order) and exits[order[pointer]] <= t:
            pointer += 1
        contract = next((c for c in order[pointer:] if c in listed[t]), None)
        if contract is None:
            raise ContinuousSeriesError(f"no listed {product} contract to hold after {t.date()}")
        held.append(contract)
    held = pd.Series(held, index=dates)
    prev = held.shift(1).fillna(held.iloc[0])

    def at(frame: pd.DataFrame, contracts: pd.Series) -> np.ndarray:
        columns = frame.columns.get_indexer(contracts.to_numpy())
        rows = np.arange(len(frame))
        values = frame.to_numpy(dtype="float64")
        return np.where(columns >= 0, values[rows, np.clip(columns, 0, None)], np.nan)

    close = at(price, held)
    prev_close = at(price, prev)
    exit_at_fsp = np.zeros(len(dates), dtype=bool)
    if days_before == 0:
        for index, (t, contract) in enumerate(prev.items()):
            if t == ltd.get(contract) and pd.notna(fsp.get(contract)):
                prev_close[index] = fsp[contract]
                exit_at_fsp[index] = True
    if np.isnan(close).any() or np.isnan(prev_close).any():
        missing = dates[np.isnan(close) | np.isnan(prev_close)][:5]
        raise ContinuousSeriesError(f"no {product} price for the held contract on {list(missing.date)}")
    roll = (held != prev).to_numpy()
    gap = np.where(roll, close - prev_close, 0.0)
    pnl = prev_close - np.concatenate([[np.nan], close[:-1]])
    offset = np.concatenate([np.cumsum(gap[::-1])[::-1][1:], [0.0]])  # sum of the gaps after each row
    ret = pnl / np.concatenate([[np.nan], close[:-1]])
    growth = np.concatenate([np.cumprod((1.0 + ret[1:])[::-1])[::-1], [1.0]])  # product of (1 + ret) after each row
    out = pd.DataFrame(
        {
            "trade_date": dates,
            "contract": held.to_numpy(dtype=object),
            "close": close,
            "close_adj": close + offset,
            "close_adj_ratio": close[-1] / growth,
            "prev_contract": prev.to_numpy(dtype=object),
            "prev_close": prev_close,
            "open": at(pivots["open"], prev),
            "high": at(pivots["high"], prev),
            "low": at(pivots["low"], prev),
            "night_high": at(night_pivots["high"], prev),
            "night_low": at(night_pivots["low"], prev),
            "pnl_points": pnl,
            "ret": ret,
            "roll": roll,
            "roll_gap_points": gap,
            "exit_at_fsp": exit_at_fsp,
        },
        columns=CONTINUOUS_COLUMNS,
    )
    return out
