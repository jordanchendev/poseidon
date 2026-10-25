"""The Thalassa stock-bar dataset, its session filter, and stocks through the compiler on a synthetic store."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from poseidon.pipeline.refdata import tw_stocks
from poseidon.pipeline.refdata.datasets import FUTURES, MARKET_CALENDAR, TAIEX, TAIEX_TR
from poseidon.pipeline.refdata.engine_input import MAX_STOCK_GAP_DAYS, StockDataGap, align_stock_bars
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.refdata.tw_corporate_actions import ACTION_COLUMNS, TAI50_DATASET
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.v3.compiler import NotCompilable, compile_spec
from poseidon.pipeline.v3.config import REFDATA_DATASETS, STOCK_REFDATA_DATASETS, refdata_datasets, series_first_days
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050, tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.spec import substitute

STOCKS = tw_stocks.DATASET


def commit(store: RefdataStore, dataset: str, frame: pd.DataFrame, column: str) -> None:
    frame = frame.assign(**{column: pd.to_datetime(frame[column])})
    parts = {key: group.reset_index(drop=True) for key, group in frame.groupby(frame[column].dt.strftime("%Y-%m"))}
    store.commit(dataset, schema_version=1, date_column=column, partitions=parts, sources=[], code_version="test")


def tai50_frame(days: pd.DatetimeIndex, closures: list[pd.Timestamp] = ()) -> pd.DataFrame:
    """TAI50 rows on ``days``; ``closures`` carry the previous level (session False)."""

    level = pd.Series(np.linspace(100.0, 110.0, len(days)), index=days)
    for day in closures:
        level[day] = level[:day].iloc[-2]
    return pd.DataFrame({"trade_date": days, "tai50": level.to_numpy(), "tai50_tr": level.to_numpy() * 2}).assign(
        session=lambda f: f["tai50"].diff().ne(0)
    )


def thalassa_payload(symbol: str, days, closes) -> bytes:
    rows = [
        {
            "time": f"{pd.Timestamp(day).date()} 05:30:00+00:00",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
            "adj_close": None,
        }
        for day, close in zip(days, closes, strict=True)
    ]
    return json.dumps(
        {"data": rows, "symbol": symbol, "market": "tw_stock", "interval": "1d", "count": len(rows)}
    ).encode()


class FakeResponse:
    def __init__(self, body: bytes):
        self.status_code, self.content = 200, body


class FakeClient:
    def __init__(self, answers: dict[str, bytes]):
        self.answers, self.urls = answers, []

    def get(self, url):
        self.urls.append(url)
        symbol = url.split("symbol=")[1].split("&")[0]
        return FakeResponse(self.answers[symbol])


# ----------------------------------------------------------------------------- sessions


def test_keep_sessions_drops_closed_days_and_follows_the_schedule_after_tai50():
    tai50_days = pd.bdate_range("2026-01-05", "2026-01-16")
    open_days = set(tai50_days.date) - {date(2026, 1, 8)}
    schedule = MarketSchedule.from_frame(pd.DataFrame({"date": [pd.Timestamp("2026-01-21")], "closed": [True]}))
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2026-01-07", "2026-01-08", "2026-01-10", "2026-01-19", "2026-01-21", "2026-01-24"]
            ),
            "symbol": "0050",
        }
    )
    kept = tw_stocks.keep_sessions(frame, open_days, date(2026, 1, 16), schedule)
    assert kept["trade_date"].dt.date.tolist() == [date(2026, 1, 7), date(2026, 1, 19)]


def test_rows_frame_orders_columns_and_drops_bad_rows():
    days = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-06", "2026-01-07"]).date
    frame = pd.DataFrame(
        {"open": [1, 2, 2.5, 3], "high": [1, 2, 2.5, 3], "low": [1, 2, 2.5, 3], "close": [1, 2, 2.5, 0], "volume": 1.0},
        index=pd.Index(days, name="date"),
    )
    rows = tw_stocks.rows_frame({"0050": frame, "2330": frame.iloc[0:0]})
    assert list(rows.columns) == tw_stocks.COLUMNS
    assert rows["close"].tolist() == [1.0, 2.5] and rows["symbol"].tolist() == ["0050", "0050"]


# ----------------------------------------------------------------------------- build and read


@pytest.fixture
def world(tmp_path):
    """TAI50 sessions on 2026-01-05..02-27 with one closure, the schedule closing 2026-03-02, two symbols."""

    store = RefdataStore(tmp_path)
    days = pd.bdate_range("2026-01-05", "2026-02-27")
    commit(store, TAI50_DATASET, tai50_frame(days, closures=[pd.Timestamp("2026-01-20")]), "trade_date")
    commit(store, MARKET_CALENDAR, pd.DataFrame({"date": [pd.Timestamp("2026-03-02")], "closed": [True]}), "date")
    thalassa_days = [*days, pd.Timestamp("2026-01-24"), pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")]
    answers = {
        "0050": thalassa_payload("0050", thalassa_days, [100.0 + i for i in range(len(thalassa_days))]),
        "00631L": thalassa_payload("00631L", thalassa_days[5:], [40.0] * (len(thalassa_days) - 5)),
    }
    return store, FakeClient(answers), days


def test_build_commits_session_rows_and_is_idempotent(world):
    store, client, days = world
    result = tw_stocks.build(
        store, code_version="t", thalassa_url="http://thalassa/", api_key="k", client=client, names=["0050", "00631L"]
    )
    assert result.changed and client.urls[0].startswith("http://thalassa/api/v1/ohlcv?symbol=0050&market=tw_stock")
    rows = store.read(STOCKS)
    assert set(rows["symbol"]) == {"0050", "00631L"} and list(rows.columns) == tw_stocks.COLUMNS
    sessions = rows[rows["symbol"] == "0050"]["trade_date"]
    assert pd.Timestamp("2026-01-20") not in set(sessions)  # TAI50 closure
    assert pd.Timestamp("2026-01-24") not in set(sessions)  # Saturday
    assert pd.Timestamp("2026-03-02") not in set(sessions)  # scheduled holiday after the TAI50 history
    assert pd.Timestamp("2026-03-03") in set(sessions)  # scheduled open day after the TAI50 history
    assert len(sessions) == len(days) - 1 + 1

    bars = tw_stocks.read_bars(store, "00631L", start="2026-01-12", end="2026-01-16")
    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert bars["date"].dt.date.tolist() == [date(2026, 1, 12) + pd.Timedelta(days=k) for k in range(5)]
    assert tw_stocks.first_days(store) == {"0050": date(2026, 1, 5), "00631L": date(2026, 1, 12)}

    again = tw_stocks.build(
        store, code_version="t", thalassa_url="http://thalassa", api_key="k", client=client, names=["0050", "00631L"]
    )
    assert not again.changed and again.written == ()
    assert store.manifest(STOCKS)["sources"][0]["name"] == "thalassa_0050_1d.json"


def test_update_hook_skips_without_thalassa_settings_and_reports_otherwise(world, monkeypatch):
    from poseidon.core.config import settings

    store, client, _days = world
    monkeypatch.setattr(settings, "thalassa_base_url", "")
    assert tw_stocks.update(store, today=date(2026, 3, 3), code_version="t") == []
    monkeypatch.setattr(settings, "thalassa_base_url", "http://thalassa")
    monkeypatch.setattr(settings, "thalassa_api_key", "k")
    monkeypatch.setattr(tw_stocks, "thalassa_client", lambda _key: client)
    monkeypatch.setattr(tw_stocks, "symbols", lambda: ["0050", "00631L"])
    [report] = tw_stocks.update(store, today=date(2026, 3, 3), code_version="t")
    assert (report.dataset, report.changed, report.rows) == (STOCKS, True, 40 + 35)
    assert report.coverage == {"start": "2026-01-05", "end": "2026-03-03"}


# ----------------------------------------------------------------------------- alignment


def test_align_stock_bars_carries_gaps_forward_and_refuses_long_ones():
    days = pd.bdate_range("2026-01-05", periods=8)
    bars = pd.DataFrame(
        {"date": days[[0, 1, 2, 5, 6, 7]], "open": 1.0, "high": 2.0, "low": 0.5, "close": [10, 11, 12, 15, 16, 17.0]}
    )
    aligned = align_stock_bars(bars, days, "x")
    assert aligned["traded"].tolist() == [True, True, True, False, False, True, True, True]
    assert aligned["close"].tolist() == [10, 11, 12, 12, 12, 15, 16, 17]
    assert aligned.loc[3, ["open", "high", "low"]].tolist() == [12.0, 12.0, 12.0]
    with pytest.raises(StockDataGap, match="after the window's first day"):
        align_stock_bars(bars.iloc[1:], days, "x")
    saturday = pd.DataFrame(
        {"date": [days[0] + pd.Timedelta(days=5)], "open": 1.0, "high": 2.0, "low": 0.5, "close": 13.0}
    )
    with pytest.raises(StockDataGap, match="outside the trading calendar"):
        align_stock_bars(pd.concat([bars, saturday], ignore_index=True), days, "x")
    long_days = pd.bdate_range("2026-01-05", periods=MAX_STOCK_GAP_DAYS + 3)
    ended = bars.iloc[:2]
    with pytest.raises(StockDataGap, match=f"more than {MAX_STOCK_GAP_DAYS} sessions"):
        align_stock_bars(ended, long_days, "x")


# ----------------------------------------------------------------------------- compiled stock specs


def rotation(**values):
    doc = substitute(etf_rotation_00631l_0050(), {"ma_days": 100, "band": 0.02, **values})
    doc["params"] = {}
    return doc


@pytest.fixture(scope="module")
def stock_store(tmp_path_factory):
    """0050 and 00631L bars over 320 sessions (00631L missing one day), a 00631L 1 -> 4 split, both indices."""

    days = pd.bdate_range("2025-01-06", periods=320)
    rng = np.random.default_rng(3)
    base = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days))))
    lev = 40 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, len(days))))
    split_day = days[250]
    lev_raw = np.where(days >= split_day, lev / 4, lev)
    rows = []
    for symbol, closes in (("0050", base), ("00631L", lev_raw)):
        frame = pd.DataFrame({"trade_date": days, "symbol": symbol, "close": closes})
        frame["open"] = frame["close"] * 0.999
        frame["high"] = frame["close"] * 1.01
        frame["low"] = frame["close"] * 0.99
        frame["volume"] = 1000.0
        if symbol == "00631L":
            frame = frame[frame["trade_date"] != days[200]]  # one session without a bar
        rows.append(frame[tw_stocks.COLUMNS])
    store = RefdataStore(tmp_path_factory.mktemp("stocks"))
    commit(store, STOCKS, pd.concat(rows, ignore_index=True), "trade_date")
    prev = float(lev[249])
    action = {column: None for column in ACTION_COLUMNS}
    action.update(
        date=split_day,
        symbol="00631L",
        name="x",
        action_type="split",
        prev_close=prev,
        reference_price=prev / 4,
        factor=0.25,
        source="test",
    )
    commit(store, ACTIONS, pd.DataFrame([action]), "date")
    index = pd.DataFrame({"trade_date": days, "open": np.nan, "high": np.nan, "low": np.nan, "close": base * 100})
    commit(store, TAIEX, index, "trade_date")
    commit(store, TAIEX_TR, index[["trade_date", "close"]], "trade_date")
    return store, days, split_day


def test_compile_runs_a_stock_rotation_with_the_split_booked(stock_store):
    store, days, split_day = stock_store
    compiled = compile_spec(rotation(), start=days[120], store=store)
    assert set(compiled.inputs) == {TAIEX, TAIEX_TR, STOCKS, ACTIONS}
    assert compiled.config.execution == "next_open" and compiled.config.start == days[120]
    assert [note for note in compiled.notes if "carried forward" in note] == [
        "lev: 1 session(s) without a 00631L bar carried forward (no fills)"
    ]
    assert "lev: 1 corporate action day(s) of 00631L booked from tw_corporate_actions" in compiled.notes
    assert compiled.futures["lev"].spec.kind == "stock" and compiled.futures["base"].bars["traded"].all()
    lev_bars = compiled.futures["lev"].bars.set_index("date")
    assert lev_bars.loc[split_day, "unit_ratio"] == 4.0 and lev_bars.loc[split_day, "factor"] == 0.25
    assert not lev_bars.loc[days[200], "traded"]

    result = compiled.run()
    event = result.events[result.events["kind"] == "corporate_action"]
    assert len(event) <= 1
    if len(event):  # the split lands while 00631L is held
        assert event.iloc[0]["instrument"] == "lev" and event.iloc[0]["value"] == 4.0
        held = result.held_units("lev")
        assert held[split_day] == held[days[249]] * 4 or held[days[249]] == 0
    assert (result.daily_pnl["equity"] > 0).all()
    assert not result.events["kind"].isin(["margin_call", "forced_liquidation"]).any()


def test_stock_data_gaps_are_not_compilable(stock_store):
    store, days, _split = stock_store
    with pytest.raises(NotCompilable, match="no 2330 bars"):
        doc = rotation()
        doc["instruments"]["base"]["symbol"] = "2330"
        compile_spec(doc, start=days[120], store=store)


def test_refdata_datasets_and_first_days_cover_stocks_only_for_stock_specs(stock_store):
    store, days, _split = stock_store
    assert refdata_datasets(tmf_dynamic_leverage_3x()) == REFDATA_DATASETS
    assert refdata_datasets(rotation()) == REFDATA_DATASETS + STOCK_REFDATA_DATASETS
    quotes = pd.DataFrame(
        {
            "trade_date": [days[0]] * 2,
            "product": ["TX", "MTX"],
            "contract_month": ["202501", "202501"],
            "session": "day",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "settlement": 1.0,
        }
    )
    commit(store, FUTURES, quotes, "trade_date")
    first = series_first_days(store)
    assert first[f"{STOCKS}/0050"] == days[0].date() and first[f"{STOCKS}/00631L"] == days[0].date()
    assert f"{STOCKS}/2330" not in first and first["taifex_futures_daily/TX"] == days[0].date()
    assert first[TAIEX] == days[0].date()


def test_stock_fill_prices_follow_the_etf_tick(stock_store):
    store, days, _split = stock_store
    compiled = compile_spec(rotation(), start=days[120], store=store)
    fills = compiled.run().fills
    buys = fills[fills["units"] > 0]
    assert len(buys) and (buys["price"] - buys["ref_price"]).round(6).isin([0.01, 0.05]).all()
    assert (buys["tax"] == 0).all()
    assert buys["commission"].to_numpy() == approx((0.001425 * buys["units"] * buys["price"] * 1000).to_numpy())
