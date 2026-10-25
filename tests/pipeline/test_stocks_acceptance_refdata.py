"""Step 8 acceptance (1) on the real refdata: buy-and-hold 0050 and 00631L through the daily engine across
their unit splits (0050 1 -> 4 resumed 2025-06-18, 00631L 1 -> 22 resumed 2026-03-31) against the TWSE
TAI50 total-return index (0050) and the issuer's NAV with distributions reinvested (00631L), within the
tolerances the corporate-actions session used (resumption windows 0.5 pp, 126-session year window 1.0 pp).

Runs only with ``PV2_PAPER_REFDATA`` = a refdata root holding ``thalassa_tw_stock_daily``,
``tw_corporate_actions``, ``twse_tai50`` and ``yuanta_etf_nav``. The ledger is the thing under test:
the strategy keeps every TWD it can in the ETF (odd lots, so no lot-size cash drag) and buys more when a
cash dividend arrives; the equity curve is compared the way the session compared adjusted closes.
``POSEIDON_V3_STOCKS_OUT`` keeps the window table as JSON.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import BookView, ContractSpec, Decision, EngineConfig, MarketView, run
from poseidon.pipeline.engine.data import stock_input
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.tw_corporate_actions import TAI50_DATASET, load_actions
from poseidon.pipeline.refdata.tw_corporate_actions_acceptance import NAV_DATASET, compare_windows, nav_total_return
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.v3.instruments import declared

ROOT = os.environ.get("PV2_PAPER_REFDATA")
OUT = os.environ.get("POSEIDON_V3_STOCKS_OUT")
NEEDED = (STOCKS, "tw_corporate_actions", TAI50_DATASET, NAV_DATASET)
pytestmark = pytest.mark.skipif(
    not ROOT or not all(RefdataStore(ROOT).exists(name) for name in NEEDED),
    reason="PV2_PAPER_REFDATA lacks the datasets",
)
SPLITS = {"0050": date(2025, 6, 18), "00631L": date(2026, 3, 31)}
DATA_FROM = {"0050": "2024-06-01", "00631L": "2025-04-01"}  # > 126 sessions before the split
CAPITAL = 10_000_000.0


class FullyInvested:
    """Every TWD into the ETF, re-bought when cash (a dividend) affords another share."""

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        price = float(view["etf"].close[-1])
        return Decision({"etf": int(book.equity // price) if book.equity > 0 else 0}, {"etf": 1.0})


def buy_and_hold(store, symbol: str) -> tuple[pd.Series, pd.DataFrame]:
    bars = read_bars(store, symbol, start=DATA_FROM[symbol])
    instrument = declared({"symbol": symbol, "unit": "odd_lot"})
    etf = stock_input(ContractSpec.from_instrument(instrument), bars, load_actions(store, symbol))
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
    frame["thalassa_adj_close"] = np.nan
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
        benchmark, _notes = nav_total_return(symbol, nav, load_actions(store, symbol))
    rows = compare_windows(adjusted, benchmark, SPLITS[symbol])
    keep(f"acceptance1_{symbol}", rows)
    assert [row["window"] for row in rows] == ["resumption_1", "resumption_5", "year"]
    assert all(row["pass"] for row in rows), rows
    # the split itself: the raw close collapses, the ledger does not
    resumption = rows[0]
    assert resumption["raw_close"] < -0.5 and abs(resumption["ours"]) < 0.05, resumption
