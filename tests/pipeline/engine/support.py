"""Shared fixtures for the engine tests: specs, scripted strategies, synthetic per-contract data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import BookView, ContractSpec, Decision, FuturesInput, MarketView, target_leverage_units

# §12 defaults: TMF 10 TWD/pt, 20 TWD/side, tax 2/100000 per side, 1 tick slippage, 7.0% / 5.4% margin.
TMF = ContractSpec(symbol="TMF", point_value=10.0, tick_size=1.0, commission_per_unit=20.0, tax_rate=2e-5)

FAR = "2100-01-01"


def day(n: int) -> pd.Timestamp:
    return pd.bdate_range("2026-01-05", periods=n + 1)[n]


def bars(rows: list[tuple]) -> pd.DataFrame:
    """rows: (day index, contract, open, high, low, close)."""

    return pd.DataFrame(
        [(day(i), c, o, h, lo, cl) for i, c, o, h, lo, cl in rows],
        columns=["date", "contract", "open", "high", "low", "close"],
    )


def flat_bars(closes: list[float], contract: str = "A") -> pd.DataFrame:
    return bars([(i, contract, c, c, c, c) for i, c in enumerate(closes)])


def tmf(
    frame: pd.DataFrame, expiries: Mapping[str, str] | None = None, spec: ContractSpec = TMF, roll_days_before: int = 1
) -> dict[str, FuturesInput]:
    expiries = expiries or {c: FAR for c in frame["contract"].unique()}
    return {"TMF": FuturesInput(spec=spec, bars=frame, expiries=expiries, roll_days_before=roll_days_before)}


class Scripted:
    """Targets by day index (into the decision days); days not listed keep the position."""

    def __init__(
        self, targets: Mapping[int, int], instrument: str = "TMF", leverage: Mapping[int, float] | None = None
    ):
        self.targets = dict(targets)
        self.leverage = dict(leverage or {})
        self.instrument = instrument
        self.calls = 0
        self.books: list[BookView] = []

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        n = self.calls
        self.calls += 1
        self.books.append(book)
        targets = {self.instrument: self.targets[n]} if n in self.targets else {}
        lev = {self.instrument: self.leverage[n]} if n in self.leverage else {}
        return Decision(targets, lev)


class MaBandLeverage:
    """levtx's "3x + 200DMA ±2%" rule as a target strategy.

    Regime (1 = on, 0 = off) is a hysteresis state machine over a signal series that starts at the
    first signal row dated on or after ``start``: on unless that row closes below its MA; then off
    when close < MA x (1 - band), back on when close > MA x (1 + band). The leverage used at a
    decision comes from the state of the signal row before the latest one visible (``lag=1``: data
    up to t-1, the levtx convention), so with ``signal_close`` execution the fills land where
    levtx books them. Units = round(equity x leverage / (close x point value)).
    """

    def __init__(
        self,
        start,
        *,
        instrument: str = "TMF",
        aux: str | None = None,
        ma_days: int = 200,
        band: float = 0.02,
        on: float = 3.0,
        off: float = 0.5,
        lag: int = 1,
        rounding: str = "nearest",
        spec: ContractSpec = TMF,
    ):
        self.start = np.datetime64(pd.Timestamp(start))
        self.instrument, self.aux = instrument, aux
        self.ma_days, self.band, self.on, self.off, self.lag = ma_days, band, on, off, lag
        self.rounding, self.spec = rounding, spec
        self.k = 0
        self.s: int | None = None
        self.closes: list[float] = []
        self.states: list[tuple[pd.Timestamp, int]] = []

    def _signal(self, view: MarketView) -> tuple[np.ndarray, np.ndarray]:
        if self.aux is not None:
            series = view.aux(self.aux)
            return series.dates, series["close"]
        series = view[self.instrument]
        return series.dates, series.adjusted_close

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        dates, closes = self._signal(view)
        while self.k < len(dates) - self.lag:
            close = float(closes[self.k])
            self.closes.append(close)
            ma = float(np.mean(self.closes[-self.ma_days :])) if len(self.closes) >= self.ma_days else np.nan
            if self.s is None:
                if dates[self.k] >= self.start:
                    self.s = 0 if (not np.isnan(ma) and close < ma) else 1
            elif not np.isnan(ma):
                if self.s == 1 and close < ma * (1 - self.band):
                    self.s = 0
                elif self.s == 0 and close > ma * (1 + self.band):
                    self.s = 1
            if dates[self.k] >= self.start:
                self.states.append((pd.Timestamp(dates[self.k]), self.s))
            self.k += 1
        state = 1 if self.s is None else self.s
        lev = self.on if state == 1 else self.off
        price = float(view[self.instrument].close[-1])
        units = target_leverage_units(book.equity, lev, price, self.spec.point_value, rounding=self.rounding)
        return Decision({self.instrument: units}, {self.instrument: lev})


def synthetic_contracts(n_days: int = 320, seed: int = 7) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], pd.DataFrame]:
    """Random-walk index, monthly contracts (expiry = 3rd Wednesday, or the next trading day),
    each listed 3 months ahead with a time-to-expiry basis. Returns (bars, expiries, index aux)."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    index = 18000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days)))
    months = pd.period_range(dates[0].to_period("M"), (dates[-1] + pd.DateOffset(months=3)).to_period("M"), freq="M")
    expiries = {}
    for m in months:
        third_wed = pd.date_range(m.start_time, m.end_time, freq="W-WED")[2]
        on_or_after = dates[dates >= third_wed]
        expiries[m.strftime("%Y%m")] = on_or_after[0] if len(on_or_after) else third_wed
    rows = []
    for i, t in enumerate(dates):
        for c, exp in expiries.items():
            months_out = (exp - t).days / 30.0
            if exp < t or months_out > 3.2:
                continue
            close = round(index[i] - 15 * months_out + rng.normal(0, 3))
            o = round(close + rng.normal(0, 40))
            hi = max(o, close) + abs(round(rng.normal(0, 60)))
            lo = min(o, close) - abs(round(rng.normal(0, 60)))
            rows.append((t, c, o, hi, lo, close))
    frame = pd.DataFrame(rows, columns=["date", "contract", "open", "high", "low", "close"])
    aux = pd.DataFrame({"close": np.round(index, 2)}, index=dates)
    return frame, expiries, aux
