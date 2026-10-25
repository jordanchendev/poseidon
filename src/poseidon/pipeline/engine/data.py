"""Engine input: per-contract-month daily futures bars, the settlement calendar, auxiliary series.

``FuturesInput.bars`` is one row per (trading day, contract month):

========== ==================================================================================
column     meaning
========== ==================================================================================
date       trading day (anything ``pd.to_datetime`` accepts; normalised to midnight). TAIFEX
           convention: the after-hours session that starts at 15:00 belongs to the NEXT
           trading day, so the bar of day t covers the 15:00 session of t-1 plus the day
           session of t.
contract   contract month id, e.g. ``"202611"`` (any string; ordering comes from ``expiries``)
open       first trade of the trading day (the 15:00 after-hours open when that session
           exists, else the 08:45 day open). ``execution.next_open`` fills here.
high, low  extremes over the whole trading day (after-hours + day session); used for the
           intraday margin-risk check
close      day-session close (13:45; on the last trading day the final trade). Daily
           settlement, sizing and decisions use it.
settle     optional; fills a missing ``close`` (deferred months often have no trade)
========== ==================================================================================

A missing ``open``/``high``/``low`` is filled with the close (no trade that day).

``FuturesInput.expiries`` maps every contract in ``bars`` to its last trading day (final
settlement day). It and the trading ``calendar`` are ex-ante reference data: they are never
truncated and never inferred from the price rows inside the engine, because inferring "this
contract's last row is its expiry" from truncated data would roll differently at the right edge.

Roll (§5.5, ``roll.before_settlement {days_before: k}``): the contract held after the close of
day t is the one with the nearest expiry strictly after the trading day k sessions after t
(k = 1: roll at the close of the day before final settlement, as ``levtx.build_r1``; k = 0: roll
at the close of the expiry day itself). Each contract is settled at its own close, so the roll
gap is never booked as PnL.

``aux`` series (index levels, rates, ...) are DataFrames with a sorted, unique DatetimeIndex and
numeric columns. The strategy sees the rows dated on or before the decision day.

Stocks (``spec.kind == "stock"``) use the same table with one contract column held forever
(:func:`stock_input`). Their bars may carry per-day columns the futures bars leave out
(:data:`DAY_COLUMNS` defaults): ``traded`` (False = no session for this instrument that day,
the bar is a carry-forward: orders wait, nothing fills), and the corporate action effective on
that day, ``factor`` (reference / previous close: the causal adjusted series divides by it),
``unit_ratio`` (held units are multiplied by it: a 1 -> 4 split is 4) and ``cash_per_share``
(paid into cash). :func:`stock_bars` fills them from ``tw_corporate_actions`` rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from poseidon.pipeline.engine.specs import ContractSpec

FUTURES_BAR_COLUMNS = ("date", "contract", "open", "high", "low", "close")
# Optional per-day bar columns and their value when a bar leaves them out.
DAY_COLUMNS = {"traded": True, "factor": 1.0, "unit_ratio": 1.0, "cash_per_share": 0.0}
FOREVER = pd.Timestamp("2200-01-01")  # expiry of a stock's one "contract"


@dataclass(frozen=True)
class FuturesInput:
    """One continuous futures instrument built from per-contract-month bars."""

    spec: ContractSpec
    bars: pd.DataFrame
    expiries: Mapping[str, Any]
    roll_days_before: int = 1

    def truncated(self, cut: Any) -> FuturesInput:
        return replace(self, bars=self.bars[pd.to_datetime(self.bars["date"]).dt.normalize() <= pd.Timestamp(cut)])


def truncate_aux(aux: Mapping[str, pd.DataFrame], cut: Any) -> dict[str, pd.DataFrame]:
    return {name: frame.loc[frame.index <= pd.Timestamp(cut)] for name, frame in aux.items()}


@dataclass(frozen=True)
class ContractTable:
    """Dense (date x contract) price arrays plus the held-contract schedule for one instrument."""

    dates: pd.DatetimeIndex
    contracts: tuple[str, ...]  # sorted by expiry
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    held: np.ndarray  # contract column held after the close of each date
    traded: np.ndarray  # bool per date; False = carry-forward bar, no fills
    factor: np.ndarray  # corporate-action price factor effective that day (1.0 = none)
    unit_ratio: np.ndarray  # held units x this at the start of the day (1.0 = none)
    cash_per_share: np.ndarray  # TWD per share (per price point) paid at the start of the day

    def price(self, field: str, i: int, contract: int) -> float:
        value = float(getattr(self, field)[i, contract])
        if not np.isfinite(value):
            raise ValueError(f"no {field} for contract {self.contracts[contract]} on {self.dates[i].date()}")
        return value


def normalise_calendar(calendar: Sequence[Any]) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize()
    if not cal.is_monotonic_increasing or not cal.is_unique:
        raise ValueError("calendar must be strictly increasing")
    return cal


def build_contract_table(name: str, fut: FuturesInput, calendar: pd.DatetimeIndex) -> ContractTable:
    bars = fut.bars
    missing = [c for c in FUTURES_BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"{name}: bars missing columns {missing}")
    if fut.roll_days_before < 0:
        raise ValueError(f"{name}: roll_days_before must be >= 0")
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["contract"] = df["contract"].astype(str)
    if df.duplicated(["date", "contract"]).any():
        raise ValueError(f"{name}: duplicate (date, contract) rows")
    for col in ("open", "high", "low", "close", *(("settle",) if "settle" in df.columns else ())):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "settle" in df.columns:
        df["close"] = df["close"].fillna(df["settle"])
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df = df.dropna(subset=["close"])
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{name}: prices must be positive")

    expiries = {str(k): pd.Timestamp(v).normalize() for k, v in fut.expiries.items()}
    unknown = sorted(set(df["contract"]) - set(expiries))
    if unknown:
        raise ValueError(f"{name}: no expiry for contracts {unknown[:5]}")
    contracts = tuple(sorted(set(df["contract"]), key=lambda c: (expiries[c], c)))
    dates = pd.DatetimeIndex(sorted(set(df["date"])))
    if not dates.isin(calendar).all():
        raise ValueError(f"{name}: bar dates outside the trading calendar")

    def pivot(col: str) -> np.ndarray:
        return (
            df.pivot(index="date", columns="contract", values=col)
            .reindex(index=dates, columns=list(contracts))
            .to_numpy(float)
        )

    close = pivot("close")
    per_day = df.drop_duplicates("date").set_index("date").reindex(dates)
    day_columns = {}
    for col, default in DAY_COLUMNS.items():
        values = per_day[col].fillna(default) if col in per_day.columns else pd.Series(default, index=dates)
        day_columns[col] = values.to_numpy(bool if col == "traded" else float)
    if (day_columns["factor"] <= 0).any() or (day_columns["unit_ratio"] <= 0).any():
        raise ValueError(f"{name}: factor and unit_ratio must be positive")
    if (day_columns["cash_per_share"] < 0).any():
        raise ValueError(f"{name}: cash_per_share must be non-negative")
    expiry = np.array([expiries[c] for c in contracts], dtype="datetime64[ns]")
    cal_pos = calendar.get_indexer(dates)
    ref_pos = np.minimum(cal_pos + fut.roll_days_before, len(calendar) - 1)
    ref_dates = calendar.values[ref_pos]
    held = np.empty(len(dates), dtype=int)
    for i in range(len(dates)):
        candidates = np.flatnonzero(np.isfinite(close[i]) & (expiry > ref_dates[i]))
        if not len(candidates):
            raise ValueError(f"{name}: no contract to hold after the close of {dates[i].date()}")
        held[i] = candidates[0]  # columns are sorted by expiry
    return ContractTable(dates, contracts, pivot("open"), pivot("high"), pivot("low"), close, held, **day_columns)


def stock_bars(bars: pd.DataFrame, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Engine bars of one stock from daily OHLC rows (``date``, ``open``, ``high``, ``low``,
    ``close``, optional ``traded``) and its ``tw_corporate_actions`` rows (``date``,
    ``prev_close``, ``reference_price``, ``cash_dividend``, ``action_type``).

    An action lands on the first bar dated on or after its date (one dated before the first bar is
    history: nothing is held yet, and the adjusted series starts at the first close). The ledger keeps the position's
    value at the reference price: ``unit_ratio = (prev_close - cash) / reference_price`` with the
    cash paid out, so a cash dividend leaves the units alone, a split multiplies them, and a
    rights offering or cash capital reduction is booked as its total-return equivalent (the
    adjusted series makes the same assumption). A cash dividend row without an amount
    (old TWSE layouts) pays ``prev_close - reference_price``; other untyped rows pay nothing.
    """

    out = bars.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").reset_index(drop=True)
    for col, default in DAY_COLUMNS.items():
        if col not in out.columns:
            out[col] = default
    if actions is not None and len(actions):
        dates = out["date"].to_numpy()
        for action in actions.sort_values("date").to_dict("records"):
            when = np.datetime64(pd.Timestamp(action["date"]).normalize())
            i = int(np.searchsorted(dates, when))
            if i >= len(out) or when < dates[0]:
                continue
            prev, ref = float(action["prev_close"]), float(action["reference_price"])
            cash = action.get("cash_dividend")
            if cash is None or pd.isna(cash):
                cash = prev - ref if action["action_type"] == "cash_dividend" else 0.0
            out.loc[i, "factor"] *= ref / prev
            out.loc[i, "unit_ratio"] *= (prev - float(cash)) / ref
            out.loc[i, "cash_per_share"] += float(cash)
    return out


def stock_input(spec: ContractSpec, bars: pd.DataFrame, actions: pd.DataFrame | None = None) -> FuturesInput:
    """A stock as an engine input: one contract named after the symbol, never expiring or rolling."""

    if spec.policy.rolls:
        raise ValueError(f"{spec.symbol}: stock_input is for instrument kinds without a roll schedule")
    frame = stock_bars(bars, actions)
    frame["contract"] = spec.symbol
    return FuturesInput(spec=spec, bars=frame, expiries={spec.symbol: FOREVER}, roll_days_before=0)


def validate_aux(aux: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for name, frame in aux.items():
        index = pd.DatetimeIndex(frame.index)
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError(f"aux {name!r}: index must be sorted unique dates")
        out[name] = frame.set_axis(index.normalize()).astype(float)
    return out
