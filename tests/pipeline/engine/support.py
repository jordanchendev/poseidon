"""Shared fixtures for the engine tests: specs, scripted strategies, synthetic per-contract data."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import BookView, ContractSpec, Decision, FuturesInput, MarketView
from poseidon.pipeline.v3.instruments import INSTRUMENTS

# The instrument-table row: 10 TWD/pt, 20 TWD/side, tax 2/100000 per side, 1 tick slippage, 7.0% / 5.4% margin.
TMF = ContractSpec.from_instrument(INSTRUMENTS["TMF"])
# The same contract without the listing date and TX price proxy (rows never flagged proxy).
TMF_PLAIN = dataclasses.replace(TMF, listed_since=None, price_proxy=None)

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
    frame: pd.DataFrame,
    expiries: Mapping[str, str] | None = None,
    spec: ContractSpec = TMF_PLAIN,
    roll_days_before: int = 1,
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
