"""Gate 6 -- regime split.

``regime_states(daily)`` gives, per daily bar, the regime known after that bar
closed (trailing windows only):

* trend: close vs its ``regime_trend_sma_days`` SMA -> ``bull`` if close > SMA
  else ``bear``;
* vol: ATR% = mean true range over ``regime_atr_days`` / close, vs the rolling
  median of ATR% over ``regime_vol_median_days`` -> ``high_vol`` if strictly
  above the median else ``low_vol``.

A trade takes the state of the latest bar dated strictly **before** its entry
date (Asia/Taipei), so only completed prior days are used. The daily frame must
be indexed by the date each bar completes (TX day-session bars); a bar that
spans midnight must carry its closing date. Warm-up bars have no state and
their trades are reported as unlabelled.

Buckets with fewer than ``regime_min_trades_per_segment`` trades are excluded.
Pass iff evaluated buckets >= ``regime_min_segments`` and profitable buckets
(sum pnl_twd > 0) >= min(``regime_profitable_segments_min``, evaluated).
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from poseidon.pipeline.gates.base import GateResult, Trade, failed, invalid, is_finite, passed
from poseidon.pipeline.gates.daily import Basis, DailyResult

GATE = "regime_split"
TAIPEI = ZoneInfo("Asia/Taipei")
BUCKETS: tuple[str, ...] = ("bull_high_vol", "bull_low_vol", "bear_high_vol", "bear_low_vol")


@dataclass(frozen=True)
class RegimeInputs:
    trades: Sequence[Trade]
    states: Mapping[date, str | None]  # from regime_states(...).to_dict()


def _bar_date(value) -> date:
    return value if isinstance(value, date) and not isinstance(value, datetime) else pd.Timestamp(value).date()


def regime_states(daily: pd.DataFrame, *, sma_days: int = 60, atr_days: int = 20, median_days: int = 250) -> pd.Series:
    """Regime after each bar's close, indexed by bar date; ``None`` during warm-up."""

    frame = daily.sort_index()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1, skipna=False
    )
    sma = close.rolling(sma_days, min_periods=sma_days).mean()
    atr_pct = true_range.rolling(atr_days, min_periods=atr_days).mean() / close
    median = atr_pct.rolling(median_days, min_periods=median_days).median()

    known = sma.notna() & close.notna() & atr_pct.notna() & median.notna()
    trend = np.where(close > sma, "bull", "bear")
    vol = np.where(atr_pct > median, "high_vol", "low_vol")
    labels = [f"{t}_{v}" if ok else None for t, v, ok in zip(trend, vol, known, strict=True)]
    return pd.Series(labels, index=[_bar_date(value) for value in frame.index], dtype=object, name="regime")


def threshold_windows(thresholds: Mapping) -> dict:
    """``regime_states`` keyword arguments frozen in the run thresholds."""

    return {
        "sma_days": int(thresholds["regime_trend_sma_days"]),
        "atr_days": int(thresholds["regime_atr_days"]),
        "median_days": int(thresholds["regime_vol_median_days"]),
    }


def entry_date(entry_time: datetime) -> date:
    return (entry_time.astimezone(TAIPEI) if entry_time.tzinfo else entry_time).date()


def label_for_entry(entry_time: datetime, dates: Sequence[date], states: Mapping[date, str | None]) -> str | None:
    """State of the latest bar strictly before the entry date (``dates`` sorted)."""

    position = bisect.bisect_left(dates, entry_date(entry_time))
    return states[dates[position - 1]] if position else None


def evaluate(inputs: RegimeInputs, thresholds: Mapping) -> GateResult:
    min_trades = int(thresholds["regime_min_trades_per_segment"])
    min_segments = int(thresholds["regime_min_segments"])
    profitable_min = int(thresholds["regime_profitable_segments_min"])
    bad = [str(index) for index, trade in enumerate(inputs.trades) if not is_finite(trade.pnl_twd)]
    if bad:
        return invalid(GATE, "non-finite trade pnl at index " + ", ".join(bad))

    dates = sorted(inputs.states)
    buckets = {name: {"trades": 0, "pnl_twd": 0.0} for name in BUCKETS}
    unlabelled = 0
    for trade in inputs.trades:
        label = label_for_entry(trade.entry_time, dates, inputs.states)
        if label not in buckets:
            unlabelled += 1
            continue
        buckets[label]["trades"] += 1
        buckets[label]["pnl_twd"] += float(trade.pnl_twd)
    for bucket in buckets.values():
        bucket["evaluated"] = bucket["trades"] >= min_trades
        bucket["profitable"] = bucket["evaluated"] and bucket["pnl_twd"] > 0

    evaluated = [name for name, bucket in buckets.items() if bucket["evaluated"]]
    profitable = [name for name in evaluated if buckets[name]["profitable"]]
    required = min(profitable_min, len(evaluated))
    report = {
        "buckets": buckets,
        "trades": len(inputs.trades),
        "unlabelled_trades": unlabelled,
        "evaluated_buckets": len(evaluated),
        "profitable_buckets": len(profitable),
        "required_profitable_buckets": required,
        "min_trades_per_bucket": min_trades,
        "min_buckets": min_segments,
    }
    headline = f"{len(profitable)}/{len(evaluated)} regimes profitable"
    if len(evaluated) < min_segments:
        return failed(
            GATE,
            "insufficient_coverage",
            f"{len(evaluated)} buckets with >= {min_trades} trades < {min_segments}",
            f"{len(evaluated)}/4 regimes covered",
            report,
        )
    if len(profitable) < required:
        losers = ", ".join(name for name in evaluated if name not in profitable)
        return failed(
            GATE,
            "regime_concentration",
            f"{len(profitable)} profitable buckets < {required}; losers: {losers}",
            headline,
            report,
        )
    return passed(GATE, headline, report)


# --- daily classes (design 2026-10-24 §7) -----------------------------------------------------
#
# The same four buckets, but each train *day* is labelled (state of the latest underlying bar
# dated strictly before it, i.e. known at the previous close) and a bucket counts once it has
# >= ``regime_min_days_per_segment`` days. A bucket is profitable when the sum of its daily
# metric (daily returns for ``daily_equity``, TWD PnL for ``daily_fixed``) is > 0. Pass iff
# evaluated buckets >= ``regime_min_segments`` and profitable >= min(``regime_profitable_segments_min``,
# evaluated). ``states`` comes from :func:`regime_states` on the 1x underlying's daily bars.


@dataclass(frozen=True)
class RegimeDailyInputs:
    result: DailyResult  # the train backtest
    basis: Basis
    states: Mapping[date, str | None]


def evaluate_daily(inputs: RegimeDailyInputs, thresholds: Mapping) -> GateResult:
    min_days = int(thresholds["regime_min_days_per_segment"])
    min_segments = int(thresholds["regime_min_segments"])
    profitable_min = int(thresholds["regime_profitable_segments_min"])
    metric = inputs.result.metric(inputs.basis)
    pnl = inputs.result.pnl_twd
    bad = [day.isoformat() for day, value in zip(inputs.result.dates, metric, strict=True) if not is_finite(value)]
    if bad:
        return invalid(GATE, "non-finite daily metric on " + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))

    dates = sorted(inputs.states)
    buckets = {name: {"days": 0, "metric_sum": 0.0, "pnl_twd": 0.0} for name in BUCKETS}
    unlabelled = 0
    for day, value, day_pnl in zip(inputs.result.dates, metric, pnl, strict=True):
        position = bisect.bisect_left(dates, day)
        label = inputs.states[dates[position - 1]] if position else None
        if label not in buckets:
            unlabelled += 1
            continue
        bucket = buckets[label]
        bucket["days"] += 1
        bucket["metric_sum"] += float(value)
        bucket["pnl_twd"] += float(day_pnl)
    for bucket in buckets.values():
        bucket["evaluated"] = bucket["days"] >= min_days
        bucket["profitable"] = bucket["evaluated"] and bucket["metric_sum"] > 0

    evaluated = [name for name, bucket in buckets.items() if bucket["evaluated"]]
    profitable = [name for name in evaluated if buckets[name]["profitable"]]
    required = min(profitable_min, len(evaluated))
    report = {
        "basis": inputs.basis,
        "buckets": buckets,
        "days": inputs.result.n_days,
        "unlabelled_days": unlabelled,
        "evaluated_buckets": len(evaluated),
        "profitable_buckets": len(profitable),
        "required_profitable_buckets": required,
        "min_days_per_bucket": min_days,
        "min_buckets": min_segments,
    }
    headline = f"{len(profitable)}/{len(evaluated)} regimes profitable"
    if len(evaluated) < min_segments:
        return failed(
            GATE,
            "insufficient_coverage",
            f"{len(evaluated)} buckets with >= {min_days} days < {min_segments}",
            f"{len(evaluated)}/4 regimes covered",
            report,
        )
    if len(profitable) < required:
        losers = ", ".join(name for name in evaluated if name not in profitable)
        return failed(
            GATE,
            "regime_concentration",
            f"{len(profitable)} profitable buckets < {required}; losers: {losers}",
            headline,
            report,
        )
    return passed(GATE, headline, report)
