"""Corporate actions: parsers on real TWSE payloads, factors, adjusted series across the 0050
split and a cash dividend, the price-limit cross-check, raw fetching and the shared store."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline.refdata import tw_corporate_actions as ca
from poseidon.pipeline.refdata.fetch import PoliteFetcher, SourceBlockedError
from poseidon.pipeline.refdata.store import RefdataStore

FIXTURES = Path(__file__).parent / "fixtures" / "tw_corporate_actions"
SPLIT_0050 = date(2025, 6, 18)
DIVIDEND_0050 = date(2025, 7, 21)


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def bars_0050() -> pd.DataFrame:
    """Thalassa 0050 1d, 2025-05-26 .. 2025-07-31 (raw close; adj_close as Thalassa stores it)."""
    return ca.thalassa_bars(payload("thalassa_0050_1d_2025"))


def tai50() -> pd.DataFrame:
    return pd.DataFrame(ca.parse_tai50(payload("tai50i_202506")) + ca.parse_tai50(payload("tai50i_202507"))).set_index(
        "date"
    )


def actions_0050() -> pd.DataFrame:
    rows = ca.parse_ex_rights(payload("twt49u_2025_excerpt")) + ca.parse_etf_split(payload("twtcau"))
    frame = ca.actions_frame(rows)
    return frame[frame["symbol"] == "0050"]


# --- parsers -----------------------------------------------------------------------------------
def test_cash_dividend_row_gives_reference_over_prev_close():
    rows = {(r["symbol"], r["date"]): r for r in ca.parse_ex_rights(payload("twt49u_2025_excerpt"))}
    row = rows[("0050", DIVIDEND_0050)]
    assert row["action_type"] == "cash_dividend"
    assert (row["prev_close"], row["reference_price"], row["cash_dividend"]) == (51.45, 51.09, 0.36)
    assert row["factor"] == pytest.approx(51.09 / 51.45)
    assert (row["open_reference"], row["limit_up"], row["limit_down"]) == (51.10, 56.15, 45.99)


def test_ex_rights_rows_stay_unclassified_until_the_detail_is_read():
    kinds = {r["symbol"]: r for r in ca.parse_ex_rights(payload("twt49u_2010_excerpt"))}
    assert kinds["1715"]["action_type"] == "cash_dividend"
    assert kinds["3058"]["action_type"] == "ex_rights"
    assert kinds["3058"]["cash_dividend"] == 0.0
    assert kinds["6271"]["action_type"] == "ex_rights"
    assert kinds["6271"]["cash_dividend"] is None  # the 2010 layout states only 權值+息值
    assert kinds["6271"]["factor"] == pytest.approx(110.95 / 116.00)


def test_detail_separates_a_rights_offering_from_a_stock_dividend():
    cathay = ca.parse_ex_rights(payload("twt49u_2022_excerpt"))[0]
    typed = ca.classify_ex_rights(cathay, ca.parse_ex_rights_detail(payload("twt49udetail_2882_20221205")))
    assert typed["action_type"] == "rights_offering"  # 81.6 new shares per 1,000 at NT$35
    assert (typed["rights_per_1000"], typed["rights_price"], typed["stock_per_1000"]) == (81.61871609, 35.0, 0.0)
    assert typed["formula_reference"] == pytest.approx(cathay["reference_price"], rel=0.005)
    first = ca.parse_ex_rights(payload("twt49u_2015_excerpt"))[0]
    both = ca.classify_ex_rights(first, ca.parse_ex_rights_detail(payload("twt49udetail_2892_20150731")))
    assert both["action_type"] == "stock_dividend_and_rights_offering"
    assert (both["cash_dividend"], both["stock_per_1000"]) == (0.7, 65.0)
    assert both["formula_reference"] == pytest.approx(first["reference_price"], rel=0.005)
    assert ca.parse_ex_rights_detail(payload("twt49udetail_2885_20250723")) is None


def test_old_layout_reads_cash_from_its_own_column():
    rows = {r["symbol"]: r for r in ca.parse_ex_rights(payload("twt49u_2004_excerpt"))}
    tsmc = rows["2330"]  # 2004-06-14: 權值 6.57 + 息值 0.6037
    assert tsmc["action_type"] == "ex_rights"
    assert tsmc["cash_dividend"] == pytest.approx(0.6037)
    assert tsmc["factor"] == pytest.approx(43.33 / 50.50)


def test_etf_split_type_comes_from_the_price_ratio_even_without_a_label():
    rows = {r["symbol"]: r for r in ca.parse_etf_split(payload("twtcau"))}
    assert rows["0050"]["action_type"] == "split"
    assert rows["0050"]["factor"] == pytest.approx(47.16 / 188.65)
    assert 1 / rows["0050"]["factor"] == pytest.approx(4.0, rel=1e-3)
    assert rows["00631L"]["action_type"] == "split"  # 分割 column is blank on this row
    assert 1 / rows["00631L"]["factor"] == pytest.approx(22.0, rel=1e-3)
    assert rows["00632R"]["action_type"] == "reverse_split"
    assert rows["00632R"]["factor"] == pytest.approx(22.96 / 3.28)


def test_capital_reduction_with_cash_returned_raises_the_factor_above_one():
    rows = {r["symbol"]: r for r in ca.parse_capital_reduction(payload("twtauu_2011_excerpt"))}
    cht = rows["2412"]
    assert cht["action_type"] == "capital_reduction"
    assert cht["detail"] == "退還股款"
    assert cht["factor"] == pytest.approx(88.87 / 73.10)


def test_par_value_change_rows():
    rows = ca.parse_par_value_change(payload("twtb8u"))
    first = next(r for r in rows if r["symbol"] == "8070")
    assert first["date"] == date(2020, 8, 17)
    assert first["factor"] == pytest.approx(19.0 / 190.0)


def test_no_data_answer_is_empty_and_layout_changes_raise():
    assert ca.parse_ex_rights({"stat": "很抱歉，沒有符合條件的資料!"}) == []
    with pytest.raises(ca.TwseParseError):
        ca.parse_ex_rights({"stat": "查詢開始日期小於92年5月5日，請重新查詢!"})
    broken = payload("twt49u_2025_excerpt")
    broken["fields"] = [f.replace("除權息參考價", "參考價") for f in broken["fields"]]
    with pytest.raises(ca.TwseParseError, match="除權息參考價"):
        ca.parse_ex_rights(broken)


def test_tai50_rows_are_the_session_calendar():
    frame = tai50()
    assert date(2025, 6, 2) in frame.index
    assert date(2025, 6, 1) not in frame.index  # Sunday
    assert frame.loc[date(2025, 6, 2), "tai50_tr"] == pytest.approx(39334.51)


def test_published_limit_prices_match_the_limit_rule():
    rows = ca.parse_ex_rights(payload("twt49u_2025_excerpt")) + ca.parse_etf_split(payload("twtcau"))
    findings = ca.check_action_limits(ca.actions_frame(rows))
    assert [f for f in findings if f.severity == "error"] == []
    assert {f.symbol for f in findings if f.kind == "no_price_limit"} == {"00673R", "00706L", "00715L", "00674R"}
    wrong = ca.actions_frame(rows)
    wrong.loc[wrong["symbol"] == "00631L", "limit_up"] = 22.15  # a 10% band on a 2x ETF
    assert {(f.kind, f.symbol) for f in ca.check_action_limits(wrong) if f.severity == "error"} == {
        ("limit_rule_mismatch", "00631L")
    }
    assert ca.price_limits("00631L", date(2026, 3, 31)) == pytest.approx((0.20, 0.20))
    assert ca.price_limits("2330", date(2015, 5, 29)) == pytest.approx((0.07, 0.07))
    assert ca.price_limits("0050", date(2008, 10, 24)) == pytest.approx((0.07, 0.035))


# --- adjustment --------------------------------------------------------------------------------
def test_adjusted_series_has_no_split_crash_and_tracks_the_total_return_index():
    bars = bars_0050()
    adjusted = ca.adjust_prices(bars, actions_0050())
    before, after = date(2025, 6, 10), SPLIT_0050  # halted 06-11 .. 06-17
    raw = bars.loc[after, "close"] / bars.loc[before, "close"] - 1
    ours = adjusted.loc[after, "adj_close"] / adjusted.loc[before, "adj_close"] - 1
    index = tai50()["tai50_tr"]
    bench = index.loc[after] / index.loc[before] - 1
    assert raw < -0.7  # the fake crash
    assert abs(ours - bench) < 0.005
    assert adjusted["adj_close"].iloc[-1] == bars["close"].iloc[-1]  # latest price unchanged


def test_cash_dividend_day_return_is_measured_from_the_reference_price():
    bars = bars_0050()
    adjusted = ca.adjust_prices(bars, actions_0050())
    eve = max(d for d in bars.index if d < DIVIDEND_0050)
    ours = adjusted.loc[DIVIDEND_0050, "adj_close"] / adjusted.loc[eve, "adj_close"] - 1
    assert ours == pytest.approx(bars.loc[DIVIDEND_0050, "close"] / 51.09 - 1)
    assert adjusted.loc[eve, "adj_factor"] == pytest.approx(51.09 / 51.45)
    assert adjusted.loc[date(2025, 6, 10), "adj_factor"] == pytest.approx(51.09 / 51.45 * 47.16 / 188.65)


def test_an_action_without_prices_is_refused():
    actions = actions_0050().copy()
    actions.loc[actions.index[0], "factor"] = None
    with pytest.raises(ValueError, match="without a factor"):
        ca.adjust_prices(bars_0050(), actions)


# --- cross-check -------------------------------------------------------------------------------
def sessions() -> set[date]:
    return set(tai50().index)


def test_real_window_with_its_actions_is_clean():
    findings = ca.cross_check("0050", bars_0050(), actions_0050(), sessions())
    assert [f for f in findings if f.severity == "error"] == []
    halted = [f for f in findings if f.kind == "missing_sessions"]
    assert halted == []  # the 06-11 .. 06-17 halt is explained by the split


def test_a_dropped_split_is_an_unexplained_limit_breach():
    actions = actions_0050()
    findings = ca.cross_check("0050", bars_0050(), actions[actions["action_type"] != "split"], sessions())
    breaches = [f for f in findings if f.kind == "unexplained_limit_breach"]
    assert {f.date for f in breaches} == {SPLIT_0050}
    assert all(f.severity == "error" for f in breaches)


def synthetic(closes: list[float], start: date = date(2025, 6, 2)) -> pd.DataFrame:
    days = sorted(d for d in sessions() if d >= start)[: len(closes)]
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=pd.Index(days))


EMPTY = ca.actions_frame([])


def test_synthetic_unexplained_jump_is_flagged():
    findings = ca.cross_check("2330", synthetic([100, 101, 116, 117]), EMPTY, sessions())
    assert [(f.kind, f.detail["column"]) for f in findings if f.severity == "error"] == [
        ("unexplained_limit_breach", "close"),
        ("unexplained_limit_breach", "high"),
        ("unexplained_limit_breach", "low"),
    ]


def test_leveraged_etf_band_is_twice_as_wide():
    assert ca.cross_check("00631L", synthetic([100, 115, 98]), EMPTY, sessions()) == []
    flagged = ca.cross_check("00631L", synthetic([100, 121]), EMPTY, sessions())
    assert {f.kind for f in flagged} == {"unexplained_limit_breach"}


def test_missing_sessions_widen_the_band_and_are_reported():
    frame = synthetic([100, 101, 102, 103])
    gapped = frame.drop(frame.index[1:3]).copy()
    gapped.iloc[-1] = 118  # +18% over three sessions: inside 1.1**3 - 1
    findings = ca.cross_check("2330", gapped, EMPTY, sessions())
    assert [(f.severity, f.kind, f.detail["count"]) for f in findings] == [("warning", "missing_sessions", 2)]


def test_bar_on_a_non_session_is_an_error():
    frame = synthetic([100, 101])
    extra = pd.DataFrame({c: [101.0] for c in frame.columns}, index=pd.Index([date(2025, 6, 1)]))
    findings = ca.cross_check("2330", pd.concat([frame, extra]), EMPTY, sessions())
    assert [(f.kind, f.date) for f in findings] == [("non_session_bar", date(2025, 6, 1))]


def test_prev_close_disagreeing_with_the_stored_close_is_an_error():
    bars = bars_0050().copy()
    eve = max(d for d in bars.index if d < DIVIDEND_0050)
    bars.loc[eve, ["open", "high", "low", "close"]] = bars.loc[eve, "close"] * 1.02
    findings = ca.cross_check("0050", bars, actions_0050(), sessions())
    assert ("prev_close_mismatch", DIVIDEND_0050) in {(f.kind, f.date) for f in findings}


# --- fetching and the store ---------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code, self.content = status, body


class FakeClient:
    def __init__(self, answers):
        self.answers, self.urls = list(answers), []

    def get(self, url):
        self.urls.append(url)
        return self.answers.pop(0)


def raw_source(tmp_path, answers, dataset=ca.DATASET):
    client = FakeClient(answers)
    fetcher = PoliteFetcher(0.0, client=client, sleep=lambda _: None)
    return ca.RawSource(RefdataStore(tmp_path), dataset, fetcher), client


def test_raw_source_keeps_the_bytes_and_replays_them(tmp_path):
    body = json.dumps(payload("twtcau"), ensure_ascii=False).encode()
    source, client = raw_source(tmp_path, [FakeResponse(200, body)])
    params = {"startDate": "20030505", "endDate": "20261024", "response": "json"}
    first, record = source.get("TWTCAU.json", ca.ETF_SPLIT_URL, params)
    again, same = source.get("TWTCAU.json", ca.ETF_SPLIT_URL, params)
    assert len(client.urls) == 1 and client.urls[0].startswith(ca.ETF_SPLIT_URL + "?startDate=20030505")
    assert first == again and record == same
    assert record.url == ca.ETF_SPLIT_URL and record.params == params
    assert (tmp_path / ca.DATASET / "raw" / "TWTCAU.json").read_bytes() == body


def test_a_throttle_page_instead_of_json_stops_the_run(tmp_path):
    html = FakeResponse(200, b"<html>please wait</html>")
    source, client = raw_source(tmp_path, [html] * 4)
    with pytest.raises(SourceBlockedError):
        source.get("TWT49U_2025.json", ca.EX_RIGHTS_URL, {})
    assert len(client.urls) == 4
    assert not (tmp_path / ca.DATASET / "raw" / "TWT49U_2025.json").exists()


def test_actions_round_trip_through_month_partitions(tmp_path):
    store = RefdataStore(tmp_path)
    actions = actions_0050()
    partitions = ca.month_partitions(actions, "date")
    assert sorted(partitions) == ["2025-01", "2025-06", "2025-07"]

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
    assert loaded["date"].tolist() == actions["date"].tolist()
    assert loaded["factor"].tolist() == pytest.approx(actions["factor"].tolist())


def test_typhoon_carry_over_days_are_not_sessions():
    frame = ca.mark_sessions(pd.DataFrame(ca.parse_tai50(payload("tai50i_201609"))))
    closed = set(frame.loc[~frame["session"], "date"])
    assert closed == {date(2016, 9, 27), date(2016, 9, 28)}  # Typhoon Megi
