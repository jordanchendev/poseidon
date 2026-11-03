"""Step 8 acceptance (1) on the real refdata: buy-and-hold 0050 and 00631L through the daily engine across
their unit splits (0050 1 -> 4 resumed 2025-06-18, 00631L 1 -> 22 resumed 2026-03-31) against the TWSE
TAI50 total-return index (0050) and the issuer's NAV with distributions reinvested (00631L), within the
tolerances the corporate-actions session used (resumption windows 0.5 pp, 126-session year window 1.0 pp).

Runs only with ``PV2_PAPER_REFDATA`` = a refdata root holding ``thalassa_tw_stock_daily``,
``tw_corporate_actions``, ``twse_tai50`` and ``yuanta_etf_nav``. The last two are frozen benchmark files: the
TWSE TAI50 build and the Yuanta NAV acceptance that wrote them were removed when the actions moved to Thalassa
(2026-11), and their rows through 2026-11-02 cover both windows. The ledger is the thing under test:
the strategy keeps every TWD it can in the ETF (odd lots, so no lot-size cash drag) and buys more when a
cash dividend arrives; the equity curve is compared the way the session compared adjusted closes.
``POSEIDON_V3_STOCKS_OUT`` keeps the window table as JSON.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline.engine import BookView, Decision, EngineConfig, MarketView, run
from poseidon.pipeline.engine.data import stock_input
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.tw_corporate_actions import load_actions
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.v3.instruments import contract_spec, declared

ROOT = os.environ.get("PV2_PAPER_REFDATA")
OUT = os.environ.get("POSEIDON_V3_STOCKS_OUT")
TAI50_DATASET = "twse_tai50"
NAV_DATASET = "yuanta_etf_nav"
NEEDED = (STOCKS, "tw_corporate_actions", TAI50_DATASET, NAV_DATASET)
pytestmark = pytest.mark.skipif(
    not ROOT or not all(RefdataStore(ROOT).exists(name) for name in NEEDED),
    reason="PV2_PAPER_REFDATA lacks the datasets",
)
SPLITS = {"0050": date(2025, 6, 18), "00631L": date(2026, 3, 31)}
DATA_FROM = {"0050": "2024-06-01", "00631L": "2025-04-01"}  # > 126 sessions before the split
CAPITAL = 10_000_000.0
SPLIT_TOLERANCE = 0.005  # |ours - benchmark| across the resumption (last close -> 5th session after)
YEAR_TOLERANCE = 0.010  # |ours - benchmark| over 126 sessions either side of the split
WINDOW_SESSIONS = 126
# Unit splits as the issuer announced them (Yuanta notices), independent of the TWSE reference price on purpose.
ISSUER_SPLIT_RATIOS = {("0050", date(2025, 6, 18)): 4.0, ("00631L", date(2026, 3, 31)): 22.0}


def _window_return(series: pd.Series, start, end) -> float:
    return float(series.loc[end] / series.loc[start] - 1)


def compare_windows(adjusted: pd.DataFrame, benchmark: pd.Series, split: date) -> list[dict]:
    """Buy-and-hold returns over the resumption and the surrounding year; ``adjusted`` holds ``adj_close`` (ours)
    and the raw ``close``, ``benchmark`` is indexed by session date."""
    dates = [d for d in adjusted.index if d in benchmark.index]
    before = [d for d in dates if d < split]
    after = [d for d in dates if d >= split]
    windows = {
        "resumption_1": (before[-1], after[0], SPLIT_TOLERANCE),
        "resumption_5": (before[-1], after[min(4, len(after) - 1)], SPLIT_TOLERANCE),
        "year": (
            before[-min(WINDOW_SESSIONS, len(before))],
            after[min(WINDOW_SESSIONS, len(after)) - 1],
            YEAR_TOLERANCE,
        ),
    }
    out = []
    for name, (start, end, tolerance) in windows.items():
        ours = _window_return(adjusted["adj_close"], start, end)
        bench = _window_return(benchmark, start, end)
        out.append(
            {
                "window": name,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "ours": ours,
                "benchmark": bench,
                "diff": ours - bench,
                "tolerance": tolerance,
                "pass": abs(ours - bench) <= tolerance,
                "raw_close": _window_return(adjusted["close"], start, end),
            }
        )
    return out


def nav_total_return(symbol: str, nav: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Issuer NAV with cash distributions reinvested on the ex-date, ``(NAV_t + D_t) / NAV_t-1``, and unit
    splits removed with the issuer's ratio unless the NAV history is already restated (read from the jump)."""
    series = nav["nav"].astype(float)
    cash = actions[actions["cash_dividend"].notna()].groupby("date")["cash_dividend"].sum()
    growth = (series + cash.reindex(series.index).fillna(0.0)) / series.shift(1)
    for (code, day), ratio in ISSUER_SPLIT_RATIOS.items():
        if code != symbol or day not in series.index:
            continue
        position = series.index.get_loc(day)
        jump = series.iloc[position] / series.iloc[position - 1]
        if abs(jump - 1) >= abs(jump * ratio - 1):  # not restated: the NAV fell by the ratio
            growth.iloc[position] *= ratio
    return growth.fillna(1.0).cumprod()


class FullyInvested:
    """Every TWD into the ETF, re-bought when cash (a dividend) affords another share."""

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        price = float(view["etf"].close[-1])
        return Decision({"etf": int(book.equity // price) if book.equity > 0 else 0}, {"etf": 1.0})


def buy_and_hold(store, symbol: str) -> tuple[pd.Series, pd.DataFrame]:
    bars = read_bars(store, symbol, start=DATA_FROM[symbol])
    instrument = declared({"symbol": symbol, "unit": "odd_lot"})
    etf = stock_input(contract_spec(instrument), bars, load_actions(store, symbol))
    calendar = pd.DatetimeIndex(pd.to_datetime(bars["date"]))
    config = EngineConfig(initial_capital=CAPITAL, execution="next_open", start=calendar[1])
    result = run(FullyInvested(), {"etf": etf}, config, calendar=calendar)
    equity = result.daily_pnl.set_index("date")["equity"].astype(float)
    equity.index = [pd.Timestamp(day).date() for day in equity.index]
    return equity, bars


def as_adjusted(equity: pd.Series, bars: pd.DataFrame) -> pd.DataFrame:
    """The frame ``compare_windows`` reads: our series as ``adj_close``, the raw close beside it."""

    closes = bars.set_index(pd.Index([pd.Timestamp(day).date() for day in bars["date"]]))["close"].astype(float)
    frame = pd.DataFrame({"close": closes})
    frame["adj_close"] = equity.reindex(frame.index)
    return frame.dropna(subset=["adj_close"])


def keep(name: str, rows: list[dict]) -> None:
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))


@pytest.mark.parametrize("symbol", sorted(SPLITS))
def test_buy_and_hold_across_the_split_matches_the_total_return_benchmark(symbol):
    store = RefdataStore(ROOT)
    equity, bars = buy_and_hold(store, symbol)
    adjusted = as_adjusted(equity, bars)
    if symbol == "0050":
        tai50 = store.read(TAI50_DATASET)
        benchmark = tai50.set_index(tai50["trade_date"].dt.date)["tai50_tr"].astype(float)
    else:
        nav = store.read(NAV_DATASET)
        nav = nav[nav["symbol"] == symbol]
        nav = nav.set_index(nav["trade_date"].dt.date).sort_index()
        benchmark = nav_total_return(symbol, nav, load_actions(store, symbol))
    rows = compare_windows(adjusted, benchmark, SPLITS[symbol])
    keep(f"acceptance1_{symbol}", rows)
    assert [row["window"] for row in rows] == ["resumption_1", "resumption_5", "year"]
    assert all(row["pass"] for row in rows), rows
    # the split itself: the raw close collapses, the ledger does not
    resumption = rows[0]
    assert resumption["raw_close"] < -0.5 and abs(resumption["ours"]) < 0.05, resumption
