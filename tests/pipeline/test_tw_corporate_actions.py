"""tw_corporate_actions: Thalassa rows -> dataset rows, and how the engine books them (no network)."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline.engine.data import stock_bars
from poseidon.pipeline.refdata import tw_corporate_actions as ca
from poseidon.pipeline.refdata.store import RefdataStore, month_partitions
from poseidon.pipeline.refdata.tw_stocks import thalassa_bars

FIXTURES = Path(__file__).parent / "fixtures" / "tw_corporate_actions"
SPLIT_0050 = date(2025, 6, 18)
DIVIDEND_0050 = date(2025, 7, 21)


def thalassa_row(**fields) -> dict:
    """One row as Thalassa's /api/v1/tw/corporate-actions returns it."""
    row = {
        "date": "2026-07-15",
        "market": "twse",
        "symbol": "2330",
        "kind": "ex_rights",
        "prev_close": 100.0,
        "ref_price": 98.0,
        "detail": "息",
        "cash_dividend": 2.0,
        "ingested_at": None,
    }
    row.update(fields)
    row["factor"] = row["ref_price"] / row["prev_close"]
    return row


def payload(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


# --- the action_type mapping -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("market", "kind", "detail", "prev_close", "ref_price", "expected"),
    [
        ("twse", "ex_rights", "息", 100.0, 98.0, "cash_dividend"),
        ("twse", "ex_rights", "權", 100.0, 90.9, "ex_rights"),
        ("twse", "ex_rights", "權息", 100.0, 88.0, "ex_rights"),
        ("tpex", "ex_rights", "除息", 50.0, 49.0, "cash_dividend"),
        ("tpex", "ex_rights", "除權", 50.0, 45.5, "ex_rights"),
        ("tpex", "ex_rights", "除權息", 50.0, 44.0, "ex_rights"),
        ("twse", "capital_reduction", "彌補虧損", 20.0, 33.3, "capital_reduction"),
        ("twse", "capital_reduction", "退還股款", 40.0, 44.0, "capital_reduction"),
        ("tpex", "capital_reduction", "現金減資", 30.0, 33.0, "capital_reduction"),
        ("twse", "par_value_change", "換股率 10", 1000.0, 100.0, "par_value_change"),
        ("tpex", "par_value_change", "換股率 4", 400.0, 100.0, "par_value_change"),
        ("twse", "split", "分割 1:4", 188.65, 47.16, "split"),
        ("twse", "split", "反分割 7:1", 10.0, 70.0, "reverse_split"),
    ],
)
def test_every_thalassa_kind_and_label_maps_to_one_action_type(market, kind, detail, prev_close, ref_price, expected):
    row = thalassa_row(market=market, kind=kind, detail=detail, prev_close=prev_close, ref_price=ref_price)
    [action] = ca.actions_frame([row]).to_dict("records")
    assert action["action_type"] == expected
    assert action["factor"] == pytest.approx(ref_price / prev_close)
    assert (action["market"], action["source"]) == (market, f"thalassa:{market}:{kind}")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"market": "twse", "detail": "除息"}, "unknown ex_rights detail"),  # a TPEx label on a TWSE row
        ({"market": "tpex", "detail": "息"}, "unknown ex_rights detail"),
        ({"detail": None}, "unknown ex_rights detail"),
        ({"kind": "merger"}, "unknown Thalassa corporate-action kind"),
        ({"market": "otc"}, "unknown market"),
        ({"prev_close": 0.0}, "prev_close"),
        ({"ref_price": None}, "ref_price"),
        ({"cash_dividend": -1.0}, "cash_dividend"),
        ({"date": "115/07/15"}, "not an ISO date"),
    ],
)
def test_an_event_outside_the_mapping_stops_the_update(fields, message):
    row = {**thalassa_row(), **fields}
    with pytest.raises(ca.CorporateActionsError, match=message):
        ca.actions_frame([row])


def test_a_row_without_a_field_or_listed_twice_stops_the_update():
    row = thalassa_row()
    del row["cash_dividend"]
    with pytest.raises(ca.CorporateActionsError, match="lacks"):
        ca.actions_frame([row])
    with pytest.raises(ca.CorporateActionsError, match="twice"):
        ca.actions_frame([thalassa_row(), thalassa_row(ref_price=97.0)])


def test_the_same_symbol_on_both_markets_or_two_kinds_on_one_day_are_two_events():
    rows = [
        thalassa_row(),
        thalassa_row(market="tpex", detail="除息"),
        thalassa_row(kind="capital_reduction", detail="退還股款", ref_price=110.0, cash_dividend=None),
    ]
    frame = ca.actions_frame(rows)
    assert len(frame) == 3 and list(frame.columns) == ca.ACTION_COLUMNS


# --- cash_dividend: Thalassa's value, never a default --------------------------------------------------
def test_a_null_cash_stays_null_and_the_factor_comes_from_the_prices():
    rows = [
        thalassa_row(date="2004-07-20", detail="權息", prev_close=50.0, ref_price=44.0, cash_dividend=None),
        thalassa_row(date="2026-07-20", detail="權息", prev_close=50.0, ref_price=44.0, cash_dividend=1.5),
        thalassa_row(date="2026-08-20", detail="權", prev_close=50.0, ref_price=45.5, cash_dividend=0.0),
    ]
    frame = ca.actions_frame(rows)
    assert frame["action_type"].tolist() == ["ex_rights"] * 3
    assert frame["cash_dividend"].isna().tolist() == [True, False, False]
    assert frame["cash_dividend"].iloc[1:].tolist() == [1.5, 0.0]
    assert frame["factor"].tolist() == pytest.approx([44 / 50, 44 / 50, 45.5 / 50])


def test_the_engine_books_a_null_ex_rights_cash_in_units_and_a_null_cash_dividend_as_the_price_step():
    days = pd.bdate_range("2026-07-13", periods=6)
    bars = pd.DataFrame({"date": days, "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0})
    rows = [
        thalassa_row(date="2026-07-14", detail="權息", prev_close=50.0, ref_price=40.0, cash_dividend=None),
        thalassa_row(date="2026-07-16", detail="息", prev_close=50.0, ref_price=48.0, cash_dividend=None),
        thalassa_row(date="2026-07-17", detail="權息", prev_close=50.0, ref_price=40.0, cash_dividend=2.0),
    ]
    booked = stock_bars(bars, ca.actions_frame(rows)).set_index("date")
    null_rights, null_cash, known = (booked.loc[pd.Timestamp(d)] for d in ("2026-07-14", "2026-07-16", "2026-07-17"))
    assert (null_rights["cash_per_share"], null_rights["unit_ratio"]) == pytest.approx((0.0, 50 / 40))
    assert (null_cash["cash_per_share"], null_cash["unit_ratio"]) == pytest.approx((2.0, 1.0))
    assert (known["cash_per_share"], known["unit_ratio"]) == pytest.approx((2.0, 48 / 40))
    for day in (null_rights, null_cash, known):  # the position's value at the reference price is kept
        reference = 50.0 * day["factor"]
        assert day["unit_ratio"] * reference + day["cash_per_share"] == pytest.approx(50.0)


# --- 0050 2025 with Thalassa's own rows ------------------------------------------------------------------
def tai50_total_return() -> pd.Series:
    """TWSE 臺灣50報酬指數 for June-July 2025 (TAI50I files kept from the TWSE build, ROC dates)."""
    values = {}
    for month in ("202506", "202507"):
        for day, _price, total_return in payload(f"tai50i_{month}")["data"]:
            year, mm, dd = (int(part) for part in re.findall(r"\d+", day))
            values[date(year + 1911, mm, dd)] = float(total_return.replace(",", ""))
    return pd.Series(values).sort_index()


def booked_0050() -> pd.DataFrame:
    bars = thalassa_bars(payload("thalassa_0050_1d_2025")).reset_index()
    actions = ca.actions_frame(payload("thalassa_corporate_actions_0050_2025"))
    return stock_bars(bars, actions).set_index("date")


def test_thalassa_rows_book_the_0050_split_and_dividend():
    actions = ca.actions_frame(payload("thalassa_corporate_actions_0050_2025"))
    assert actions["action_type"].tolist() == ["split", "cash_dividend"]
    assert actions["cash_dividend"].isna().tolist() == [True, False] and actions["cash_dividend"].iloc[1] == 0.36
    booked = booked_0050()
    split, dividend = booked.loc[pd.Timestamp(SPLIT_0050)], booked.loc[pd.Timestamp(DIVIDEND_0050)]
    assert split["unit_ratio"] == pytest.approx(188.65 / 47.16) and split["cash_per_share"] == 0.0
    assert dividend["cash_per_share"] == pytest.approx(0.36) and dividend["unit_ratio"] == pytest.approx(1.0)


def test_buy_and_hold_0050_across_the_split_tracks_the_total_return_index():
    booked = booked_0050()
    units, cash, value = 1.0, 0.0, {}
    for day, bar in booked.iterrows():
        cash += units * bar["cash_per_share"]
        units *= bar["unit_ratio"]
        value[day.date()] = units * bar["close"] + cash
    ours = pd.Series(value)
    index = tai50_total_return()
    before, after = date(2025, 6, 10), SPLIT_0050  # halted 06-11 .. 06-17
    raw = booked.loc[pd.Timestamp(after), "close"] / booked.loc[pd.Timestamp(before), "close"] - 1
    assert raw < -0.7  # the raw price falls to a quarter
    assert abs((ours[after] / ours[before] - 1) - (index[after] / index[before] - 1)) < 0.005
    end = max(d for d in ours.index if d in index.index)
    assert abs((ours[end] / ours[before] - 1) - (index[end] / index[before] - 1)) < 0.01  # through the dividend


# --- the store round trip ---------------------------------------------------------------------------------
def test_actions_round_trip_through_month_partitions(tmp_path):
    store = RefdataStore(tmp_path)
    actions = ca.actions_frame(payload("thalassa_corporate_actions_0050_2025"))
    partitions = month_partitions(actions, "date")
    assert sorted(partitions) == ["2025-06", "2025-07"]

    def commit():
        return store.commit(
            ca.DATASET,
            schema_version=ca.SCHEMA_VERSION,
            date_column="date",
            partitions=partitions,
            sources=[],
            code_version="test",
            full=True,
        )

    assert commit().changed
    assert not commit().changed  # same rows: the manifest is left as it was
    loaded = ca.load_actions(store, "0050")
    assert loaded["date"].tolist() == [SPLIT_0050, DIVIDEND_0050]
    assert loaded["factor"].tolist() == pytest.approx(actions["factor"].tolist())
    assert loaded["cash_dividend"].isna().tolist() == [True, False]
