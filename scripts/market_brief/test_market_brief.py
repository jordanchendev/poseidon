"""Offline tests for scripts/market_brief (stdlib unittest; pytest also collects them).

Run on stormtrooper only:  cd <dir> && python3 -m unittest -v test_market_brief
Fixtures in fixtures/ are trimmed copies of real responses captured from stormtrooper on 2026-10-22.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.machinery
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import alert  # noqa: E402
import alert_seed  # noqa: E402
import collect  # noqa: E402
import deliver  # noqa: E402
import llm  # noqa: E402
import news  # noqa: E402
import render  # noqa: E402
import run  # noqa: E402
import tradingcal  # noqa: E402
import tw  # noqa: E402
import tw_credit  # noqa: E402
import tw_groups  # noqa: E402
import us  # noqa: E402
from netio import Fetcher, FetchError  # noqa: E402

FIX = HERE / "fixtures"
TPE = ZoneInfo("Asia/Taipei")
D21 = dt.date(2026, 10, 21)
D22 = dt.date(2026, 10, 22)
date = dt.date


def fx(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def cfg():
    c = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    c["thalassa"]["key_file"] = "/nonexistent/market-brief-tests/thalassa.env"  # never read a real key in tests
    return c


def U(x):
    return f'<span style="color:#d6453d">▲{x}</span>'


def D(x):
    return f'<span style="color:#2e9e6a">▼{x}</span>'


WATCH = json.loads((HERE / "watchlist.json").read_text(encoding="utf-8"))


class FakeTransport:
    """Routes by URL substring -> fixture file name, (status, bytes) tuple, or callable(url, body)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url))
        for needle, target in self.routes:
            if needle in url:
                if callable(target):
                    return target(url, body)
                if isinstance(target, tuple):
                    return target
                return 200, (FIX / target).read_bytes()
        return 404, b"not found"


def fetcher(routes):
    return Fetcher(transport=FakeTransport(routes), sleep=lambda s: None)


TW_ROUTES_21 = [
    ("MI_INDEX?date=20261021", "mi_index_20261021.json"),
    ("MI_INDEX?date=20260710", "mi_index_closed_20260710.json"),
    ("BFI82U?type=day&dayDate=20261021", "bfi82u_20261021.json"),
    ("BFI82U", "bfi82u_notyet.json"),
    ("FMTQIK", "fmtqik_202610.json"),
    ("highlight?date=2026%2F10%2F21", "tpex_highlight_20261021.json"),
    ("otc?date=2026%2F10%2F21", "tpex_otc_20261021.json"),
    ("inx?date=2026%2F10%2F21", "tpex_index_202610.json"),
    ("holidaySchedule", "holiday_2026.json"),
    ("TWT48U", "twt48u.json"),
    ("api.cnyes.com", "cnyes_tw.json"),
    ("cnbc.com/id/", "cnbc_markets.xml"),
    ("quote.cnbc.com", "cnbc_quotes.json"),
    ("futDailyMarketReport", "taifex_night_20261022.html"),
    ("mis.taifex", "mis_day.json"),
    ("finance.yahoo.com", (429, b"Too Many Requests")),
]


def calendar():
    return tradingcal.Calendar({2026: set(tw.parse_holidays(fx("holiday_2026.json")))}, cfg()["nyse_holidays"])


def collector(routes, now=None):
    now = now or dt.datetime(2026, 10, 22, 8, 30, tzinfo=dt.UTC)
    return collect.Collector(fetcher(routes), cfg(), WATCH, calendar(), now, log=lambda m: None)


# ================================================================================================ parsers
class TestTWParsers(unittest.TestCase):
    def test_mi_index(self):
        mi = tw.parse_mi_index(fx("mi_index_20261021.json"), D21)
        taiex = mi["indices"]["發行量加權股價指數"]
        self.assertEqual(taiex["close"], 46887.77)
        self.assertEqual(taiex["chg"], 594.84)
        self.assertEqual(taiex["pct"], 1.28)
        self.assertLess(mi["indices"]["金融類日報酬反向一倍指數"]["chg"], 0)  # sign from the html cell
        self.assertIn("塑膠類指數", mi["indices"])
        self.assertEqual(
            mi["breadth"]["stocks"],
            {"up": 600, "up_limit": 24, "down": 380, "down_limit": 1, "flat": 101, "untraded": 1, "nocompare": 2},
        )
        self.assertEqual(mi["breadth"]["all"]["up"], 10182)
        self.assertEqual(mi["stats"]["1.一般股票"]["turnover"], 813257346884)
        s = {x["code"]: x for x in mi["stocks"]}["2330"]
        self.assertEqual((s["close"], s["chg"]), (2420.0, 15.0))
        self.assertAlmostEqual(s["pct"], 15 / 2405 * 100)

    def test_mi_index_closed_and_wrong_date(self):
        with self.assertRaises(tw.MarketClosed):
            tw.parse_mi_index(fx("mi_index_closed_20260710.json"), dt.date(2026, 7, 10))
        with self.assertRaises(tw.NotReady):
            tw.parse_mi_index(fx("mi_index_20261021.json"), D22)

    def test_bfi82u(self):
        b = tw.parse_bfi82u(fx("bfi82u_20261021.json"), D21)
        self.assertEqual(b["total"]["net"], 1465743187)
        self.assertEqual(b["foreign"]["net"], 2238909816)
        self.assertEqual(b["dealer_hedge"]["net"], -7384233802)
        with self.assertRaises(tw.MarketClosed):  # "沒有符合條件" before 15:00 = not published yet
            tw.parse_bfi82u(fx("bfi82u_notyet.json"), D22)

    def test_fmtqik_twt48u_holidays(self):
        rows = tw.parse_fmtqik(fx("fmtqik_202610.json"))
        self.assertEqual(rows[-1], {"date": "2026-10-21", "turnover": 874614166736.0, "taiex": 46887.77, "chg": 594.84})
        ex = tw.parse_twt48u(fx("twt48u.json"), D22)
        self.assertEqual([(e["code"], e["cash"]) for e in ex], [("0056", 1.72)])
        closed = set(tw.parse_holidays(fx("holiday_2026.json")))
        self.assertIn("2026-10-26", closed)
        self.assertIn("2026-02-12", closed)  # 市場無交易，僅辦理結算交割作業
        self.assertNotIn("2026-01-02", closed)  # 國曆新年開始交易日 is a trading day
        self.assertNotIn("2026-02-11", closed)  # 農曆春節前最後交易日 is a trading day

    def test_tpex(self):
        h = tw.parse_tpex_highlight(fx("tpex_highlight_20261021.json"), D21)
        self.assertEqual((h["index_close"], h["index_chg"], h["up"], h["down"]), (395.42, 7.54, 520, 253))
        self.assertAlmostEqual(h["index_pct"], 7.54 / (395.42 - 7.54) * 100)
        self.assertEqual(h["turnover"], 214402e6)
        q = tw.parse_tpex_quotes(fx("tpex_otc_20261021.json"), D21)
        self.assertEqual(len(q), 20)
        self.assertTrue(all(r["close"] is not None for r in q))
        with self.assertRaises(tw.NotReady):
            tw.parse_tpex_highlight(fx("tpex_highlight_20261021.json"), D22)

    def test_taifex_night_report(self):
        html = (FIX / "taifex_night_20261022.html").read_bytes()
        tx = tw.parse_taifex_night_html(html, D22)
        self.assertEqual(
            (tx["symbol"], tx["last"], tx["chg"], tx["ref"], tx["pct"]), ("TX 202611", 47482.0, 469.0, 47013.0, 1.0)
        )
        self.assertEqual((tx["open"], tx["high"], tx["low"], tx["volume"]), (46991.0, 47555.0, 46983.0, 35753.0))
        with self.assertRaises(tw.NotReady):
            tw.parse_taifex_night_html(html, D21)
        with self.assertRaises(tw.NotReady):
            tw.parse_taifex_night_html("<p>查無資料</p>", D22)
        self.assertIn(b"queryDate=2026%2F10%2F22", tw.taifex_night_form(D22))

    def test_taifex_night_report_down(self):
        """11-06 night fell 7.03%: TAIFEX writes "▼-2946" / "▼-7.03%"; ref is the 11-05 day settlement 41,882."""
        html = (FIX / "taifex_night_20261106.html").read_bytes()
        tx = tw.parse_taifex_night_html(html, date(2026, 11, 6))
        self.assertEqual(
            (tx["symbol"], tx["last"], tx["chg"], tx["ref"], tx["pct"]), ("TX 202611", 38936.0, -2946.0, 41882.0, -7.03)
        )
        self.assertEqual((tx["open"], tx["high"], tx["low"], tx["volume"]), (41990.0, 42298.0, 38799.0, 89296.0))
        facts = collect.collect("tw-premarket", D22, collector(TW_ROUTES_21))
        facts["data"]["tx_night"] = tx
        md, _ = render.render(facts, None, None, cfg())
        night = md.split("## 台指期夜盤（近月）")[1].split("## 前一交易日台股")[0]
        self.assertIn(f"| 38,936 | {D('-2,946')} | {D('-7.03%')} |", night)
        self.assertIn("參考價 41,882", night)
        self.assertIn("台指期夜盤 38,936（-2,946）", md)

    def test_taifex_change_cell_sign(self):
        cases = {
            '<span class="green">▼-2,946</span>': -2946.0,
            '<span class="green">▼-7.03%</span>': -7.03,
            "<span>▼1.25%</span>": -1.25,
            '<span class="red">▲+580</span>': 580.0,
            '<span class="red">▲469</span>': 469.0,
            "-12": -12.0,
            "+12": 12.0,
            "0": 0.0,
            "0.00%": 0.0,
            "-": None,
        }
        for cell, want in cases.items():
            with self.subTest(cell=cell):
                self.assertEqual(tw._signed_arrow(cell), want)
        for cell in ("▲-5", "▼+5", "▲▼5"):
            with self.subTest(cell=cell), self.assertRaises(tw.NotReady):
                tw._signed_arrow(cell)

    def test_taifex_night_contradiction_falls_back_to_mis(self):
        """A report whose arrow and sign disagree is unusable: the brief takes MIS instead of guessing a sign."""
        bad = (FIX / "taifex_night_20261022.html").read_bytes().replace("▲469".encode(), "▲-469".encode())
        with self.assertRaises(tw.NotReady):
            tw.parse_taifex_night_html(bad, D22)
        routes = [("futDailyMarketReport", (200, bad)) if r[0] == "futDailyMarketReport" else r for r in TW_ROUTES_21]
        facts = collect.collect("tw-premarket", D22, collector(routes))
        self.assertEqual(facts["data"]["tx_night"]["source"], "TAIFEX MIS 盤後行情")
        self.assertEqual(facts["data"]["tx_night"]["chg"], -741.0)

    def test_mis(self):
        tx = tw.parse_mis_tx(fx("mis_day.json"))
        self.assertEqual((tx["symbol"], tx["last"], tx["ref"], tx["chg"]), ("TXFK6-F", 46272.0, 47013.0, -741.0))
        with self.assertRaises(tw.NotReady):
            tw.parse_mis_tx(fx("mis_night_reset.json"))


class TestUSParsers(unittest.TestCase):
    def test_cnbc(self):
        q = us.parse_cnbc(fx("cnbc_quotes.json"))
        self.assertEqual((q[".SPX"]["last"], q[".SPX"]["chg"], q[".SPX"]["pct"]), (7757.17, 44.6, 0.58))
        self.assertEqual(q[".SPX"]["session_date"], "2026-10-21")
        self.assertEqual(q["US10Y"]["session_date"], "2026-10-22")  # ISO with -0400 offset
        self.assertEqual(q[".N225"]["session_date"], "2026-10-22")
        self.assertIsNone(q[".VIX"]["chg"])  # UNCH is unknown, not flat (see test_no_silent_zero_changes)
        self.assertEqual(q["TSM"]["ext"]["type"], "POST_MKT_PREV")
        self.assertEqual(q["TSM"]["ext"]["last"], 426.5)
        self.assertEqual(q["TSM"]["ext"]["session_date"], "2026-10-21")
        self.assertNotIn("XLU", q)  # code 1 in this capture

    def test_no_silent_zero_changes(self):
        q = us.parse_cnbc(fx("cnbc_quotes.json"))
        vix = q[".VIX"]  # captured at 02:45 ET: change "UNCH", previous_day_closing reset to last
        self.assertEqual((vix["last"], vix["chg"], vix["pct"], vix["chg_missing"]), (16.68, None, None, True))
        self.assertIsNone(q["XLRE"]["ext"])  # after-hours "UNCH" -> no extended quote, renders "—"
        for sym, x in q.items():  # audit: no instrument may carry an exact-zero change as data
            self.assertFalse(x["last"] and x["chg"] == 0, sym)
            self.assertFalse(x["ext"] and x["ext"]["chg"] == 0, sym)
        self.assertEqual(us._real_change(18.47, 0.0, 0.0, 16.68)[2], False)
        self.assertAlmostEqual(us._real_change(18.47, 0.0, 0.0, 16.68)[0], 1.79)
        self.assertEqual(us._real_change(16.68, 0.0, 0.0, 16.68), (None, None, True))
        self.assertEqual(us._real_change(16.68, -0.5, -2.9, 16.68), (-0.5, -2.9, False))

    def test_session_date_check(self):
        c = collector(TW_ROUTES_21)
        got = us.parse_cnbc(fx("cnbc_quotes.json"))
        got = {"SPX": got[".SPX"], "N225": got[".N225"]}
        insts = [{"key": "SPX", "label": "S&P 500"}, {"key": "N225", "label": "日經 225"}]
        rows = c.us_rows(insts, got, expect_date="2026-10-21")
        self.assertEqual(rows[1]["note"], "2026-10-22 報價")
        self.assertEqual(rows[1]["chg"], -1481.65)
        rows = c.us_rows(insts, got, expect_date="2026-10-21", strict=True)
        self.assertEqual((rows[0]["chg"], rows[1]["chg"], rows[1]["pct"]), (44.6, None, None))
        self.assertTrue(any("日經 225 報價屬 2026-10-22" in g for g in c.gaps))
        labelled = render._quote_rows(c.us_rows(insts, got, expect_date="2026-10-21"))
        self.assertEqual(labelled[1][0], "日經 225（2026-10-22 報價）")
        self.assertEqual(render._quote_rows(rows)[1], ["日經 225（2026-10-22 報價）", "63,139.59", "未取得", "未取得"])

    def test_yahoo(self):
        q = us.parse_yahoo_chart(fx("yahoo_gspc.json"))
        self.assertIsNotNone(q["last"])
        self.assertEqual(q["source"], "Yahoo")

    def test_fetch_quotes_retry_and_breaker(self):
        f = fetcher([("quote.cnbc.com", "cnbc_quotes.json"), ("finance.yahoo.com", (429, b"Too Many Requests"))])
        insts = [
            {"key": "SPX", "cnbc": ".SPX", "yahoo": "^GSPC"},
            {"key": "XLU", "cnbc": "XLU", "yahoo": "XLU"},
            {"key": "XLF", "cnbc": "XLF", "yahoo": "XLF"},
            {"key": "ZZZ", "cnbc": "ZZZ", "yahoo": "ZZZ"},
        ]
        got, missing = us.fetch_quotes(f, insts)
        self.assertEqual(set(got), {"SPX", "XLF"})
        self.assertEqual(missing, ["XLU", "ZZZ"])
        calls = f.transport.calls
        self.assertEqual(sum("quote.cnbc.com" in u for _, u in calls), 2)  # one retry batch for the misses
        self.assertEqual(sum("yahoo" in u for _, u in calls), 1)  # breaker opens on the first 429


class TestFetcher(unittest.TestCase):
    def test_host_throttle_and_retry(self):
        slept, now = [], [0.0]
        t = FakeTransport(
            [
                ("twse.com.tw/a", "bfi82u_20261021.json"),
                ("twse.com.tw/b", (503, b"busy")),
                ("other.example", "cnyes_tw.json"),
            ]
        )
        f = Fetcher(transport=t, sleep=lambda s: slept.append(round(s, 2)), clock=lambda: now[0])
        f.get_json("https://www.twse.com.tw/a")
        f.get_json("https://other.example/x")  # unthrottled host
        f.get_json("https://www.twse.com.tw/a")
        self.assertEqual(slept, [2.5])
        with self.assertRaises(FetchError):
            f.request("https://www.twse.com.tw/b")  # 5xx: one retry, then FetchError
        self.assertEqual(sum("twse.com.tw/b" in u for _, u in t.calls), 2)
        self.assertEqual(f.log[-1]["status"], 503)


class TestNews(unittest.TestCase):
    def test_parse_and_select(self):
        rss = news.parse_rss((FIX / "cnbc_markets.xml").read_bytes(), "CNBC")
        self.assertEqual(len(rss), 3)
        self.assertTrue(all(i["link"].startswith("https://www.cnbc.com/") and "?" not in i["link"] for i in rss))
        cn = news.parse_cnyes(fx("cnyes_tw.json"), "鉅亨網")
        self.assertTrue(cn[0]["link"].startswith("https://news.cnyes.com/news/id/"))
        now = dt.datetime(2026, 10, 22, 7, 0, tzinfo=dt.UTC)
        picked = news.select([rss, cn, cn], now, 30, 2, 12)
        self.assertEqual(len(picked), 5)  # 2 per feed; the duplicate feed only adds the one cnyes item not yet picked
        self.assertEqual(len({i["title"] for i in picked}), 5)
        self.assertEqual(picked, sorted(picked, key=lambda i: i["published"], reverse=True))
        self.assertEqual(news.select([rss], now + dt.timedelta(days=5), 20, 4, 12), [])

    def test_clean_link(self):
        self.assertEqual(news.clean_link("https://x.com/a?token=abc#f"), "https://x.com/a")
        self.assertIsNone(news.clean_link("https://user:pw@x.com/a"))
        self.assertIsNone(news.clean_link("javascript:alert(1)"))


# ================================================================================================ calendar / slots
def tpe(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=TPE).astimezone(dt.UTC)


class TestCalendarAndSlots(unittest.TestCase):
    def setUp(self):
        self.cal = calendar()
        self.slots = cfg()["slots"]

    def due(self, now, state=None):
        return {
            (k, d.isoformat()): open_
            for k, d, _s, _a, open_ in tradingcal.due_slots(now, self.slots, self.cal, state or {}, 90)
        }

    def test_trading_days(self):
        self.assertTrue(self.cal.tw_open(D22))
        self.assertFalse(self.cal.tw_open(dt.date(2026, 10, 26)))  # 光復節補假 (Mon)
        self.assertFalse(self.cal.tw_open(dt.date(2026, 10, 24)))  # Saturday
        self.assertEqual(self.cal.prev_open("tw", dt.date(2026, 10, 27)), dt.date(2026, 10, 23))
        self.assertFalse(self.cal.us_open(dt.date(2026, 11, 26)))  # Thanksgiving
        self.assertTrue(self.cal.us_open(dt.date(2026, 11, 27)))
        self.assertFalse(self.cal.us_open(dt.date(2027, 3, 26)))  # Good Friday 2027
        self.assertFalse(self.cal.us_open(dt.date(2026, 7, 3)))  # Independence Day observed
        unknown = tradingcal.Calendar({}, {})
        self.assertTrue(unknown.tw_open(dt.date(2026, 10, 26)))  # no schedule -> weekdays open

    def test_tw_slots(self):
        self.assertEqual(self.due(tpe(2026, 10, 22, 8, 15)), {("tw-premarket", "2026-10-22"): True})
        self.assertEqual(self.due(tpe(2026, 10, 22, 8, 14)), {})
        self.assertEqual(self.due(tpe(2026, 10, 22, 19, 14)), {("tw-postmarket", "2026-10-22"): True})
        self.assertEqual(self.due(tpe(2026, 10, 22, 19, 15)), {})  # window is [slot, slot + 90 min)
        self.assertEqual(self.due(tpe(2026, 10, 26, 17, 50)), {("tw-postmarket", "2026-10-26"): False})  # holiday

    def test_us_slots_are_fixed_taipei_times_all_year(self):
        # same Taipei clock before and after US DST ends (2026-11-01) and starts (2027-03-14)
        self.assertEqual(self.due(tpe(2026, 10, 30, 20, 50)), {("us-premarket", "2026-10-30"): True})
        self.assertEqual(self.due(tpe(2026, 10, 31, 5, 20)), {("us-postmarket", "2026-10-30"): True})  # Sat in TPE
        self.assertEqual(self.due(tpe(2026, 11, 2, 20, 50)), {("us-premarket", "2026-11-02"): True})
        self.assertEqual(self.due(tpe(2026, 11, 2, 20, 44)), {})  # not before the slot (the old winter slot was 21:45)
        self.assertEqual(self.due(tpe(2026, 11, 3, 5, 20)), {("us-postmarket", "2026-11-02"): True})
        self.assertEqual(self.due(tpe(2026, 11, 3, 6, 20)), {("us-postmarket", "2026-11-02"): True})  # inside 90 min
        self.assertEqual(self.due(tpe(2026, 11, 3, 6, 46)), {})
        self.assertEqual(self.due(tpe(2027, 3, 12, 20, 46)), {("us-premarket", "2027-03-12"): True})
        self.assertEqual(self.due(tpe(2027, 3, 15, 20, 46)), {("us-premarket", "2027-03-15"): True})
        # NYSE holiday -> returned as closed (Thanksgiving Thu 11-26 NY: no pre, and Fri 05:15 reports the holiday)
        self.assertEqual(self.due(tpe(2026, 11, 26, 20, 50)), {("us-premarket", "2026-11-26"): False})
        self.assertEqual(self.due(tpe(2026, 11, 26, 5, 20)), {("us-postmarket", "2026-11-25"): True})  # Wed's close
        self.assertEqual(self.due(tpe(2026, 11, 27, 5, 20)), {("us-postmarket", "2026-11-26"): False})

    def test_us_session_mapping_summer_and_winter(self):
        ny = ZoneInfo("America/New_York")
        pre, post = self.slots["us-premarket"], self.slots["us-postmarket"]

        def both(t):  # the slot instant on the Taipei and the New York clock
            return t.astimezone(TPE).strftime("%a %H:%M"), t.astimezone(ny).strftime("%a %H:%M")

        # July (EDT): Wed 07-08 20:45 Taipei is 08:45 New York Wed; Thu 07-09 05:15 Taipei is 17:15 New York Wed
        a = tradingcal.slot_datetime(pre, dt.date(2026, 7, 8))
        self.assertEqual(both(a), ("Wed 20:45", "Wed 08:45"))
        b = tradingcal.slot_datetime(post, dt.date(2026, 7, 8))
        self.assertEqual(both(b), ("Thu 05:15", "Wed 17:15"))
        self.assertEqual(self.due(tpe(2026, 7, 8, 20, 45)), {("us-premarket", "2026-07-08"): True})
        self.assertEqual(self.due(tpe(2026, 7, 9, 5, 15)), {("us-postmarket", "2026-07-08"): True})
        # December (EST): the Taipei clock is the same but New York is an hour earlier: 07:45 and 16:15 (15 min after the close)
        a = tradingcal.slot_datetime(pre, dt.date(2026, 12, 7))
        self.assertEqual(both(a), ("Mon 20:45", "Mon 07:45"))
        b = tradingcal.slot_datetime(post, dt.date(2026, 12, 7))
        self.assertEqual(both(b), ("Tue 05:15", "Mon 16:15"))
        self.assertEqual(
            self.due(tpe(2026, 12, 7, 20, 45)), {("us-premarket", "2026-12-07"): True}
        )  # Taipei Monday = US Monday
        self.assertEqual(self.due(tpe(2026, 12, 8, 5, 15)), {("us-postmarket", "2026-12-07"): True})
        # weekends and holidays follow New York, not Taipei
        self.assertEqual(
            self.due(tpe(2026, 12, 12, 5, 15)), {("us-postmarket", "2026-12-11"): True}
        )  # Sat: Fri's close
        self.assertEqual(self.due(tpe(2026, 12, 13, 5, 15)), {("us-postmarket", "2026-12-12"): False})  # Sun: NY Sat
        self.assertEqual(self.due(tpe(2026, 12, 14, 5, 15)), {("us-postmarket", "2026-12-13"): False})  # Mon: NY Sun
        self.assertEqual(self.due(tpe(2026, 12, 14, 20, 45)), {("us-premarket", "2026-12-14"): True})  # Mon eve: NY Mon
        self.assertEqual(self.due(tpe(2026, 12, 25, 20, 45)), {("us-premarket", "2026-12-25"): False})  # Christmas
        self.assertEqual(self.due(tpe(2026, 7, 3, 20, 45)), {("us-premarket", "2026-07-03"): False})  # July 3 observed
        self.assertEqual(self.due(tpe(2026, 7, 4, 5, 15)), {("us-postmarket", "2026-07-03"): False})
        # the default date of a manual run is the session date too
        self.assertEqual(tpe(2026, 12, 8, 5, 20).astimezone(tradingcal.session_zone(post)).date(), dt.date(2026, 12, 7))
        self.assertEqual(tradingcal.session_zone(self.slots["tw-premarket"]).key, "Asia/Taipei")

    def test_sweep_uses_the_us_session_slot_instant(self):
        # a waiting us-postmarket 2026-12-07 (slot Tue 05:15 Taipei) expires 90 min after that instant, not after Mon 05:15
        state, outbox_notes = {"us-postmarket:2026-12-07": {"status": "waiting", "reason": "x"}}, []
        with tempfile.TemporaryDirectory() as tmp:
            for now, expired in ((tpe(2026, 12, 8, 6, 44), False), (tpe(2026, 12, 8, 6, 45), True)):
                state["us-postmarket:2026-12-07"] = {"status": "waiting", "reason": "x"}
                run.sweep_expired(state, Path(tmp), self.slots, 90, now, outbox_notes.append)
                self.assertEqual(state["us-postmarket:2026-12-07"]["status"] == "failed", expired)

    def test_state_makes_slots_idempotent(self):
        now = tpe(2026, 10, 22, 17, 50)
        for status in tradingcal.FINAL_STATUSES:
            self.assertEqual(self.due(now, {"tw-postmarket:2026-10-22": {"status": status}}), {})
        self.assertEqual(
            self.due(now, {"tw-postmarket:2026-10-22": {"status": "waiting"}}), {("tw-postmarket", "2026-10-22"): True}
        )

    def test_holiday_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = fetcher([("holidaySchedule", "holiday_2026.json")])
            closed = tradingcal.load_tw_holidays(f, tmp, 2026)
            self.assertIn("2026-10-26", closed)
            self.assertTrue((Path(tmp) / "tw-holidays-2026.json").is_file())
            f2 = fetcher([])  # everything 404: served from cache
            self.assertEqual(tradingcal.load_tw_holidays(f2, tmp, 2026), closed)
            self.assertIsNone(tradingcal.load_tw_holidays(f2, tmp, 2027))  # unknown year, no cache


# ================================================================================================ number check / LLM
class TestNumberCheck(unittest.TestCase):
    def setUp(self):
        self.facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        self.allowed = llm.allowed_numbers(self.facts)

    def test_numbers(self):
        u = llm.unknown_numbers
        self.assertEqual(u("加權指數收 46,887.77，漲 594.84 點（+1.28%），外資買超 22.39 億", self.allowed), [])
        self.assertEqual(u("收46887.77點", self.allowed), [])  # adjacent CJK characters
        self.assertEqual(u("指數約 46,888 點", self.allowed), [])  # rounding to the written digits is allowed
        self.assertEqual(u("成交 8,026.5 億", self.allowed), ["8,026.5"])
        self.assertEqual(u("收46999點", self.allowed), ["46999"])
        self.assertEqual(u("上漲 12 家，漲 0.5%，見 [3]", self.allowed), [])  # < 3 significant digits / citations
        self.assertEqual(u("合計 -1,465.74 億與 1465.74 億", [1465.74]), [])  # both separator forms accepted

    def test_add_separators(self):
        a = llm.add_separators
        self.assertEqual(a("加權指數收46041.43點，合計-1001.6億"), "加權指數收46,041.43點，合計-1,001.6億")
        self.assertEqual(
            a("2330 台積電 2026 年 46,041.43 點 663.72 億 0.62%"), "2330 台積電 2026 年 46,041.43 點 663.72 億 0.62%"
        )

    def test_validate(self):
        good = {
            "one_liner": "加權指數收 46,887.77。",
            "interpretation": "投信買超 127.99 億 [1]。",
            "watch_points": ["觀察 2330"],
        }
        sec, reasons = llm.validate(dict(good, one_liner="加權指數收 46887.77。"), self.facts)
        self.assertEqual(reasons, [])
        self.assertEqual(sec["one_liner"], "加權指數收 46,887.77。")
        bad = dict(good, interpretation="外資買超 99.99 億 [99]。")
        sec, reasons = llm.validate(bad, self.facts)
        self.assertIsNone(sec)
        self.assertTrue(any("[99]" in r for r in reasons))
        self.assertTrue(any("99.99" in r for r in reasons))
        sec, reasons = llm.validate({"one_liner": "x", "interpretation": "y", "watch_points": []}, self.facts)
        self.assertTrue(any("watch_points" in r for r in reasons))

    def _resp(self, obj, model="deepseek/deepseek-v4.1-flash-1017", content=True):
        body = {
            "model": model,
            "id": "gen-1",
            "usage": {"cost": 0.001},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(obj, ensure_ascii=False) if content else None},
                }
            ],
        }
        return 200, json.dumps(body).encode()

    def test_interpret_repair_and_refuse(self):
        good = {
            "one_liner": "加權指數收 46,887.77。",
            "interpretation": "投信買超 127.99 億。",
            "watch_points": ["觀察成交量"],
        }
        bad = dict(good, one_liner="加權指數收 47,000.12。")
        seen = []

        def transport_factory(responses):
            def t(method, url, headers, body, timeout):
                seen.append((headers, json.loads(body)))
                return responses.pop(0)

            return t

        lcfg = cfg()["llm"]
        sec, meta = llm.interpret(
            self.facts, "sk-test-SECRET", lcfg, transport=transport_factory([self._resp(bad), self._resp(good)])
        )
        self.assertEqual(meta["status"], "ok_after_repair")
        self.assertEqual(sec["one_liner"], good["one_liner"])
        self.assertEqual(meta["returned_model"], "deepseek/deepseek-v4.1-flash-1017")
        self.assertAlmostEqual(meta["cost"], 0.002)
        headers, body = seen[0]
        self.assertEqual(headers["X-Title"], "market-brief")
        self.assertIn("market-brief", headers["HTTP-Referer"])
        self.assertEqual(body["user"], "market-brief")
        self.assertEqual(body["model"], "deepseek/deepseek-v4.1-flash-1017")
        self.assertNotIn("sk-test-SECRET", json.dumps(meta))
        sec, meta = llm.interpret(
            self.facts, "k", lcfg, transport=transport_factory([self._resp(bad), self._resp(bad)])
        )
        self.assertIsNone(sec)
        self.assertEqual(meta["status"], "refused")
        md, _ = render.render(self.facts, None, meta, cfg())
        self.assertIn("AI 解讀未通過資料核對", md)
        self.assertIn('model: "none"', md)
        sec, meta = llm.interpret(self.facts, "k", lcfg, transport=transport_factory([self._resp(good, content=False)]))
        self.assertEqual(meta["status"], "error")
        self.assertIn("no content", meta["reasons"][0])


class TestKeyFile(unittest.TestCase):
    def test_fail_closed(self):
        var = "RESEARCH_DRAFT_OPENROUTER_KEY"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "k.env"
            with self.assertRaises(llm.MissingKeyError) as e:
                llm.load_key(p, var)
            self.assertIn(var, str(e.exception))
            p.write_text("export OTHER_KEY=abc\n")
            os.chmod(p, 0o600)
            with (
                mock.patch.dict(os.environ, {var: "from-env-must-not-be-used"}),
                self.assertRaises(llm.MissingKeyError),
            ):
                llm.load_key(p, var)
            p.write_text(f'export {var}="sk-or-test"\n')
            os.chmod(p, 0o644)
            with self.assertRaises(llm.MissingKeyError) as e:
                llm.load_key(p, var)
            self.assertIn("600", str(e.exception))
            os.chmod(p, 0o600)
            self.assertEqual(llm.load_key(p, var), "sk-or-test")


# ================================================================================================ collect / render
NEXUS_SENSITIVE_CI = [  # copied from ~/bin/nexus_pull.py (MacBook) so a report never gets quarantined
    r"(^|[\s,{])([a-z0-9]+[_-])*(password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|auth[_-]?token|token|cookie|dsn|database[_-]?url|redis[_-]?url)\s*[=:]\s*[^\]\s,}]+",
    r"[a-z][a-z0-9+.-]*://[^\s/@]*(:[^\s/@]*)?@",
    r"(authorization|bearer)\s*[:= ]\s*[^\]\s,}]+",
    r"(password|passwd|secret|token|api[_-]?key|authorization)\s*[=:]\s*[^\s]{6,}",
    r"x-api-key:\s*[^\s]{16,}",
    r"key[^A-Za-z0-9_]{0,3}[A-Za-z0-9_-]{43}([^A-Za-z0-9_]|$)",
]


class TestCollectRender(unittest.TestCase):
    def test_tw_postmarket(self):
        facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        self.assertEqual(facts["status"], "ok")
        d = facts["data"]
        self.assertEqual(d["institutions_yi"]["foreign"], 22.39)
        self.assertEqual(d["institutions_yi"]["dealer_total"], -135.72)
        self.assertEqual(d["top_turnover"][0]["code"], "2330")
        self.assertEqual(next(i for i in d["indices"] if i["name"] == "櫃買指數")["close"], 395.42)
        self.assertEqual(d["breadth"]["tpex"]["up"], 520)
        self.assertEqual(d["sectors"]["leaders"][0]["name"], "塑膠")
        self.assertEqual(facts["gaps"], [])
        md, summary = render.render(facts, None, None, cfg())
        self.assertTrue(md.startswith("---\ntype: market_daily\n"))
        for needle in (
            "# 台股盤後日報 2026-10-21（三）",
            "## 一句話",
            "## 大盤",
            "## 漲跌結構",
            "## 類股表現",
            "## 三大法人",
            "## 成交金額前 20 大",
            "## Watchlist",
            "## 解讀",
            f"| 加權指數 | 46,887.77 | {U('+594.84')} | {U('+1.28%')} |",
            f"| 外資及陸資（不含外資自營商） | 3,470.69 | 3,448.30 | {U('+22.39')} |",
            f"| 自營商（避險） | 317.91 | 391.75 | {D('-73.84')} |",
            f"| 合計 | 4,180.86 | 4,166.21 | {U('+14.66')} |",
            f"| 2330 | 台積電 | 2,420 | {U('+15')} | {U('+0.62%')} | 655.16 | Phase 101 pilot |",
            "| 上市股票 | 600 | 380 | 101 | 24 | 1 |",
            "|---|---:|---:|---:|",
            "<!-- tg: 收盤=Close; 漲跌=Chg; 漲跌%=Chg%!; 指數; @only=加權指數,櫃買指數,臺灣50 -->",
            "### 上櫃跌幅（成交 ≥ 5 億）",
        ):
            self.assertIn(needle, md)
        self.assertLessEqual(len(summary), 580)
        self.assertIn("46,887.77", summary)
        for pat in NEXUS_SENSITIVE_CI:
            self.assertIsNone(re.search(pat, md, re.I | re.M), pat)

    def test_telegram_text(self):
        facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        facts["headlines"] = [
            {
                "n": 1,
                "title": "台股[盤後]收漲",
                "link": "https://news.cnyes.com/news/id/1",
                "source": "鉅亨網",
                "published": "2026-10-21T06:31Z",
            }
        ]
        md, summary = render.render(
            facts,
            {"one_liner": "一句", "interpretation": "第一段。\n\n第二段 [1]。", "watch_points": ["甲", "乙"]},
            {"returned_model": "m"},
            cfg(),
        )
        sources_section = md.split("## 資料來源", 1)[1].split("\n\n_", 1)[0]
        self.assertNotIn("：http", sources_section)  # sources are link labels, the URL is not shown
        self.assertRegex(sources_section, r"\n- \[[^\]]+\]\(https?://[^)]+\)")
        tg = render.telegram_text(md)
        self.assertFalse(tg.startswith("---"))
        self.assertNotIn("promote: nexus", tg)
        self.assertNotIn("資料來源", tg)
        self.assertNotIn("api.cnyes.com", tg)  # source URLs only lived in 資料來源
        self.assertNotIn("\n#", "\n" + tg)
        self.assertNotIn("|", tg)
        self.assertTrue(tg.startswith("**台股盤後日報 2026-10-21（三）**"))
        for needle in (
            "**大盤**",
            "**一句話**",
            "一句",
            "第一段。\n\n第二段 [1]。",
            "觀察重點：\n- 甲\n- 乙",
            "```\n    Close      Chg    Chg% 指數\n46,887.77  +594.84  +1.28% 加權指數 🔴\n",
            "2330   2,420  +0.62% 台積電 🔴\n",
            " #  Code     Amt    Chg% 名稱\n 1  2330  655.16  +0.62% 台積電 🔴\n",
            "   Up  Down  Flat  LimUp  LimDn 市場\n  600   380   101     24      1 上市股票\n",  # no change column
            "    Net       Buy      Sell 法人\n +22.39  3,470.69  3,448.30 外資及陸資（不含外資自營商） 🔴\n",
            " -73.84    317.91    391.75 自營商（避險） 🟢\n",
            "-2.15% 通信網路 🟢\n",
            "1. [台股［盤後］收漲](https://news.cnyes.com/news/id/1) — 鉅亨網\n",
            "_本報告由程式彙整",
        ):
            self.assertIn(needle, tg)
        self.assertNotIn("2026-10-21T06:31Z", tg)
        self.assertNotIn("<!--", tg)
        self.assertIn(" 國巨* 🔴\n", tg)  # inside a code block: no markdown escaping
        self.assertNotIn("▲", tg.split("**解讀")[0].split("**大盤**")[1].split("```")[1])  # ASCII inside code blocks
        for block in tg.split("```")[1::2]:
            for ln in block.strip("\n").split("\n"):
                cut = next((k for k, ch in enumerate(ln) if ord(ch) > 127), len(ln))
                head = ln[:cut]
                self.assertTrue(cut == len(ln) or head.endswith(" "), ln)  # non-ASCII only in the last column
                self.assertLessEqual(len(head), 34, ln)
        self.assertLess(len(tg), render.TELEGRAM_TEXT_CAP)
        entry = run.outbox_entry(facts, md, summary)
        self.assertEqual(entry["telegram_md"], tg)
        # US table rows and the 資料未齊 banner
        facts = collect.collect("us-postmarket", D21, collector(TW_ROUTES_21))
        tg = render.telegram_text(render.render(facts, None, None, cfg())[0])
        tsm = next(ln for ln in tg.split("\n") if ln.startswith("TSM "))
        self.assertEqual(tsm.split(), ["TSM", "431.89", "+1.60%", "-1.25%", "🔴"])
        self.assertEqual(next(ln for ln in tg.split("\n") if ln.startswith("NVDA")).split(), ["NVDA", "-", "-", "-"])
        self.assertIn("Sym", tg)
        self.assertIn("\n**資料未齊**：", tg)
        self.assertIn("**類股 ETF（依漲跌排序）**", tg)  # 10 ETFs <= 2 x 5: shown in full, heading unchanged
        long_md = "---\na: b\n---\n" + "x" * 30000
        self.assertLessEqual(len(render.telegram_text(long_md)), render.TELEGRAM_TEXT_CAP)

    def test_frontmatter(self):
        facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        md, _ = render.render(
            facts,
            {"one_liner": "x", "interpretation": "y", "watch_points": ["z"]},
            {"returned_model": "deepseek/deepseek-v4.1-flash-1017"},
            cfg(),
        )
        m = re.match(r"---\n(.*?)\n---\n?(.*)", md, re.S)  # nexus_pull's frontmatter regex
        self.assertIsNotNone(m)
        self.assertRegex(m.group(1), r"(?m)^promote:\s*\"?nexus\"?\s*$")
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        fm = yaml.safe_load(m.group(1))
        self.assertEqual(fm["type"], "market_daily")
        self.assertEqual(fm["tags"], ["market_daily", "tw_stock", "postmarket"])
        self.assertEqual(fm["title"], "台股盤後日報 2026-10-21（三）")
        self.assertEqual(fm["model"], "deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(fm["promote"], "nexus")
        self.assertEqual(str(fm["session_date"]), "2026-10-21")
        self.assertTrue(any("MI_INDEX" in s for s in fm["sources"]))
        self.assertIn("as_of", fm)
        self.assertEqual(render.filename("tw-postmarket", "2026-10-21"), "2026-10-21-tw-postmarket.md")

    def test_gaps_closed_and_must_missing(self):
        routes = [r for r in TW_ROUTES_21 if not r[0].startswith("BFI82U?type=day&dayDate=20261021")]
        facts = collect.collect("tw-postmarket", D21, collector(routes))
        self.assertEqual(facts["status"], "ok")
        self.assertIn("三大法人買賣金額（BFI82U）尚未公布", facts["retry_gaps"])
        md, _ = render.render(facts, None, None, cfg())
        self.assertIn("> **資料未齊**", md)
        self.assertIn("data_gaps:", md)
        routes = [r for r in TW_ROUTES_21 if not r[0].startswith("highlight")]
        facts = collect.collect("tw-postmarket", D21, collector(routes))  # TPEx highlight late -> fallback
        otc = next(i for i in facts["data"]["indices"] if i["name"] == "櫃買指數")
        self.assertEqual((otc["close"], otc["chg"]), (395.42, 7.54))
        comp = facts["data"]["breadth"]["tpex_computed"]
        self.assertEqual(
            sum(comp.values()),
            sum(1 for r in fx("tpex_otc_20261021.json")["tables"][0]["data"] if re.fullmatch(r"[1-9]\d{3}", r[0])),
        )
        self.assertIn("上櫃普通股（依收盤行情計算", render.render(facts, None, None, cfg())[0])
        facts = collect.collect("tw-postmarket", dt.date(2026, 7, 10), collector(TW_ROUTES_21))
        self.assertEqual(facts["status"], "closed")
        facts = collect.collect("us-postmarket", D22, collector(TW_ROUTES_21))  # quotes are for 10-21
        self.assertEqual(facts["status"], "must_missing")

    def test_us_and_tw_premarket(self):
        facts = collect.collect("us-postmarket", D21, collector(TW_ROUTES_21))
        self.assertEqual(facts["status"], "ok")
        d = facts["data"]
        self.assertEqual(d["adr"]["tw_close"], 2420.0)
        a = d["adr"]
        self.assertAlmostEqual(a["equiv_twd"], round(431.89 * a["usdtwd"] / 5, 2), delta=0.05)
        tsm = next(r for r in d["watchlist"] if r["key"] == "TSM")
        self.assertEqual(tsm["ext"]["type"], "POST_MKT_PREV")
        self.assertIn("XLU", " ".join(facts["gaps"]))  # missing from CNBC and Yahoo 429 -> reported, not faked
        vix = next(r for r in d["indices"] if r["key"] == "VIX")
        self.assertEqual((vix["chg"], vix["pct"]), (None, None))
        self.assertIn("VIX 漲跌未取得（資料源回報不變或缺值）", facts["gaps"])
        md, _ = render.render(facts, None, None, cfg())
        self.assertIn("## 類股 ETF", md)
        self.assertIn("| VIX | 16.68 | 未取得 | 未取得 |", md)
        self.assertNotIn("0.00%", md.split("## 指數")[1].split("## 利率")[0])
        self.assertIn(f"| SMH | 半導體 | 556.70 | {U('+10.34')} | {U('+1.89%')} |", md)
        self.assertIn(f"| TSM | 431.89 | {U('+6.81')} | {U('+1.60%')} | 426.50 | {D('-1.25%')} |", md)  # 盤後% too
        self.assertNotIn("breadth", md.lower().split("---", 2)[2])
        # tw-premarket for 10-22 (previous TW session 10-21); MIS day fixture has CDate 20261022
        facts = collect.collect("tw-premarket", D22, collector(TW_ROUTES_21))
        self.assertEqual(facts["status"], "ok")
        self.assertEqual(facts["data"]["prev_tw"]["date"], "2026-10-21")
        self.assertEqual(facts["data"]["tx_night"]["last"], 47482.0)  # TAIFEX report (primary)
        self.assertEqual(facts["data"]["tx_night"]["basis_vs_taiex"], round(47482.0 - 46887.77, 2))
        routes = [r for r in TW_ROUTES_21 if r[0] != "futDailyMarketReport"]
        f2 = collect.collect("tw-premarket", D22, collector(routes))  # MIS fallback (fixture CDate 20261022)
        self.assertEqual(f2["data"]["tx_night"]["last"], 46272.0)
        self.assertEqual(facts["data"]["exdiv_today"][0]["code"], "0056")
        md, _ = render.render(facts, None, None, cfg())
        self.assertIn("## 台指期夜盤（近月）", md)
        self.assertNotIn("，0.00（0.00%）", md)
        self.assertIn("| 0056 | 元大高股息 | 息 | 1.72 | — |", md)


# ================================================================================================ delivery
class TestDeliver(unittest.TestCase):
    def entry(self, kind="tw-postmarket", date="2026-10-21", **kw):
        e = {"kind": kind, "date": date, "filename": f"{date}-{kind}.md", "summary": "s", "report_md": "---\nx\n---\n"}
        e.update(kw)
        return e

    def test_telegram_md_and_resend(self):
        with tempfile.TemporaryDirectory() as tmp:
            ob = Path(tmp)
            (ob / "a.json").write_text(json.dumps(self.entry(telegram_md="**全文**")))
            seen = []

            def runner(cmd, input, **kw):
                seen.append((cmd, input))
                return mock.Mock(returncode=0, stdout="{}", stderr="")

            self.assertEqual(deliver.deliver_outbox(ob, "ssh research-bridge", log=lambda m: None, runner=runner), 0)
            self.assertEqual(json.loads(seen[0][1])["telegram_md"], "**全文**")
            self.assertTrue(
                deliver.resend(
                    "tw-postmarket", "2026-10-22", "全文", "ssh research-bridge", log=lambda m: None, runner=runner
                )
            )
            self.assertEqual(seen[1][0], ["ssh", "research-bridge", "brief-resend", "tw-postmarket", "2026-10-22"])
            self.assertEqual(json.loads(seen[1][1]), {"telegram_md": "全文"})
            with self.assertRaises(ValueError):
                deliver.resend("tw-x", "2026-10-22", "t", "ssh research-bridge", runner=runner)

    def test_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ob = Path(tmp)
            for name, e in (
                ("a.json", self.entry()),
                ("b.json", self.entry(kind="us-premarket")),
                ("c.json", self.entry(date="2026-10-22")),
                ("d.json", self.entry(filename="../x.md")),
                (
                    "e.failed.json",
                    {
                        "verb": "brief-failed",
                        "kind": "tw-postmarket",
                        "date": "2026-10-20",
                        "reason": "required data\nmissing",
                    },
                ),
                ("f.failed.json", {"verb": "rm", "kind": "tw-postmarket", "date": "2026-10-20", "reason": "x"}),
            ):
                (ob / name).write_text(json.dumps(e))
            calls = []

            def runner(cmd, input, **kw):
                calls.append((cmd, json.loads(input)))
                rc = {"tw-postmarket": 0, "us-premarket": 2}.get(cmd[-2], 1)
                if cmd[-1] == "2026-10-22":
                    rc = 1
                return mock.Mock(returncode=rc, stdout="{}", stderr="nope")

            def runner_any(cmd, input, **kw):
                if cmd[2] == "brief-failed":
                    calls.append((cmd, input))
                    return mock.Mock(returncode=0, stdout="{}", stderr="")
                return runner(cmd, input, **kw)

            failed = deliver.deliver_outbox(ob, "ssh research-bridge", log=lambda m: None, runner=runner_any)
            self.assertEqual(failed, 4)  # refused + retryable + 2 malformed
            self.assertEqual(calls[0][0], ["ssh", "research-bridge", "brief", "tw-postmarket", "2026-10-21"])
            self.assertEqual(set(calls[0][1]), {"filename", "summary", "report_md"})
            fail_call = next(c for c in calls if c[0][2] == "brief-failed")
            self.assertEqual(
                fail_call,
                (["ssh", "research-bridge", "brief-failed", "tw-postmarket", "2026-10-20"], "required data missing"),
            )
            self.assertTrue((ob / "sent" / "e.failed.json").is_file())
            self.assertTrue((ob / "f.failed.json").is_file())  # unknown verb: never sent
            self.assertTrue((ob / "sent" / "a.json").is_file())
            self.assertTrue((ob / "refused" / "b.json").is_file())
            self.assertTrue((ob / "c.json").is_file())  # kept for retry
            self.assertTrue((ob / "d.json").is_file())  # malformed: never sent
            self.assertEqual(len(calls), 4)


class TestBridgePatch(unittest.TestCase):
    def load(self, tmp):
        path = HERE / "bridge_brief_patch.py.txt"
        import ast

        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
        loader = importlib.machinery.SourceFileLoader("bridge_patch_under_test", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        mod.BRIEFS = str(Path(tmp) / "media" / "market-briefs")
        mod.NEXUS = str(Path(tmp) / "Nexus")
        mod.MIRROR = str(Path(tmp) / "mirror")
        mod.LOG = str(Path(tmp) / "bridge.log")
        os.makedirs(mod.NEXUS)
        (Path(mod.NEXUS) / "AGENTS.md").write_text("# Nexus\n")
        self.sent = []
        mod.telegram = lambda text, media=None: self.sent.append((text, media)) or self.ok
        self.ok = True
        return mod

    def call(self, mod, command, payload):
        stdin = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        out = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"SSH_ORIGINAL_COMMAND": command}),
            mock.patch.object(sys, "stdin", io.StringIO(stdin)),
            mock.patch.object(sys, "stdout", out),
            mock.patch.object(sys, "stderr", io.StringIO()),
        ):
            try:
                mod.main()
                code = 0
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def test_brief_verb(self):
        md = "---\ntype: market_daily\npromote: nexus\n---\n# 台股盤後日報\n"
        p = {"filename": "2026-10-21-tw-postmarket.md", "summary": "台股盤後\n加權 +1.28%", "report_md": md}
        with tempfile.TemporaryDirectory() as tmp:
            mod = self.load(tmp)
            code, out = self.call(mod, "brief tw-postmarket 2026-10-21", p)
            self.assertEqual(code, 0, out)
            media = Path(mod.BRIEFS) / "2026-10-21-tw-postmarket.md"
            self.assertEqual(media.read_text(encoding="utf-8"), md)
            self.assertEqual(oct(media.stat().st_mode & 0o777), "0o600")
            raw = (Path(mod.NEXUS) / "_kb/raw/market-daily/2026-10-21-tw-postmarket.md").read_text(encoding="utf-8")
            self.assertTrue(raw.startswith("---\ntype: source\n"))
            self.assertTrue(raw.endswith("# 台股盤後日報\n"))
            self.assertEqual(self.sent[-1][1], str(media))
            self.assertIn("完整報告見附件", self.sent[-1][0])
            code, out = self.call(
                mod, "brief tw-postmarket 2026-10-21", p
            )  # same content again -> duplicate, no resend
            self.assertEqual((code, len(self.sent)), (0, 1))
            self.assertIn("duplicate", out)
            code, _ = self.call(mod, "brief tw-postmarket 2026-10-21", dict(p, report_md=md + "changed\n"))
            self.assertEqual(code, 2)
            for cmd, payload in (
                ("brief tw-midday 2026-10-21", p),
                ("brief tw-postmarket 2026-13-01", p),
                ("brief tw-postmarket 2026-10-21 extra", p),
                ("brief us-premarket 2026-10-21", p),
                (
                    "brief us-premarket 2026-10-22",
                    dict(p, filename="2026-10-22-us-premarket.md", report_md="no frontmatter"),
                ),
                ("brief us-premarket 2026-10-22", "not json"),
            ):
                self.assertEqual(self.call(mod, cmd, payload)[0], 2, cmd)
            self.ok = False  # Telegram down: saved, exit 1 (retryable), retry resends
            p2 = dict(p, filename="2026-10-22-us-premarket.md")
            self.assertEqual(self.call(mod, "brief us-premarket 2026-10-22", p2)[0], 1)
            self.ok = True
            self.assertEqual(self.call(mod, "brief us-premarket 2026-10-22", p2)[0], 0)
            self.assertEqual(len(self.sent), 3)

    def test_brief_with_telegram_md(self):
        md = "---\ntype: market_daily\npromote: nexus\n---\n# 台股盤後日報\n"
        p = {
            "filename": "2026-10-22-tw-postmarket.md",
            "summary": "s",
            "report_md": md,
            "telegram_md": "**台股盤後日報**\n\n**大盤**\n- 加權\x07",
        }
        with tempfile.TemporaryDirectory() as tmp:
            mod = self.load(tmp)
            self.ok = False  # text fails, document goes out -> exit 1
            mod.telegram = lambda text, media=None: self.sent.append((text, media)) or (media is not None)
            self.assertEqual(self.call(mod, "brief tw-postmarket 2026-10-22", p)[0], 1)
            self.assertEqual([m is None for _, m in self.sent], [True, False])
            self.assertEqual(self.sent[1][0], "完整報告 md：2026-10-22-tw-postmarket.md")
            mod.telegram = lambda text, media=None: self.sent.append((text, media)) or True
            code, out = self.call(mod, "brief tw-postmarket 2026-10-22", p)  # retry: only the text
            self.assertEqual(code, 0, out)
            self.assertEqual(len(self.sent), 3)
            self.assertEqual(self.sent[2], ("**台股盤後日報**\n\n**大盤**\n- 加權", None))  # cleaned, newlines kept
            code, out = self.call(mod, "brief tw-postmarket 2026-10-22", p)
            self.assertEqual((code, len(self.sent)), (0, 3))
            self.assertIn("duplicate", out)
            self.assertEqual(
                self.call(
                    mod,
                    "brief us-premarket 2026-10-22",
                    dict(p, filename="2026-10-22-us-premarket.md", telegram_md="x" * 20001),
                )[0],
                2,
            )
            self.assertEqual(
                self.call(
                    mod, "brief us-premarket 2026-10-22", dict(p, filename="2026-10-22-us-premarket.md", telegram_md=5)
                )[0],
                2,
            )

    def test_brief_resend_verb(self):
        md = "---\ntype: market_daily\npromote: nexus\n---\n# x\n"
        with tempfile.TemporaryDirectory() as tmp:
            mod = self.load(tmp)
            self.assertEqual(
                self.call(mod, "brief-resend tw-postmarket 2026-10-22", {"telegram_md": "全文"})[0], 2
            )  # not delivered yet
            old = {
                "filename": "2026-10-22-tw-postmarket.md",
                "summary": "s",
                "report_md": md,
            }  # delivered in the old format
            self.assertEqual(self.call(mod, "brief tw-postmarket 2026-10-22", old)[0], 0)
            for _ in range(2):  # no markers: the owner may resend more than once
                self.assertEqual(
                    self.call(mod, "brief-resend tw-postmarket 2026-10-22", {"telegram_md": "全文\n第二行"})[0], 0
                )
            self.assertEqual(self.sent[-1], ("全文\n第二行", None))
            self.assertEqual(len(self.sent), 3)
            self.assertEqual(self.call(mod, "brief-resend tw-postmarket 2026-10-22", "not json")[0], 2)
            self.assertEqual(self.call(mod, "brief-resend tw-postmarket 2026-10-22", {"telegram_md": ""})[0], 2)
            self.assertEqual(self.call(mod, "brief-resend tw-postmarket 2026-10-22 x", {"telegram_md": "t"})[0], 2)
            # old-format delivery + new client payload with telegram_md -> only the text is sent
            self.assertEqual(self.call(mod, "brief tw-postmarket 2026-10-22", dict(old, telegram_md="新格式"))[0], 0)
            self.assertEqual(self.sent[-1], ("新格式", None))
            self.assertEqual(len(self.sent), 4)

    def test_brief_failed_verb(self):
        with tempfile.TemporaryDirectory() as tmp:
            mod = self.load(tmp)
            code, out = self.call(
                mod, "brief-failed us-postmarket 2026-10-22", "required data missing:\x07 S&P 500 close\n" + "x" * 900
            )
            self.assertEqual(code, 0, out)
            text = self.sent[-1][0]
            self.assertTrue(
                text.startswith("市場報告未產出 us-postmarket 2026-10-22：required data missing: S&P 500 close")
            )
            self.assertNotIn("\x07", text)
            self.assertLessEqual(len(text), len("市場報告未產出 us-postmarket 2026-10-22：") + 500)
            self.assertIsNone(self.sent[-1][1])  # no attachment
            code, out = self.call(mod, "brief-failed us-postmarket 2026-10-22", "again")
            self.assertEqual((code, len(self.sent)), (0, 1))  # idempotent per kind+date
            self.assertIn("duplicate", out)
            self.assertIn("brief-failed us-postmarket 2026-10-22", Path(mod.LOG).read_text())
            for cmd in (
                "brief-failed us-postmarket 2026-10-32",
                "brief-failed nope 2026-10-22",
                "brief-failed us-postmarket",
            ):
                self.assertEqual(self.call(mod, cmd, "r")[0], 2, cmd)
            self.assertEqual(self.call(mod, "brief-failed tw-premarket 2026-10-22", "x" * 5000)[0], 2)  # stdin cap
            self.ok = False
            self.assertEqual(self.call(mod, "brief-failed tw-premarket 2026-10-22", "r")[0], 1)  # retryable
            self.ok = True
            self.assertEqual(self.call(mod, "brief-failed tw-premarket 2026-10-22", "r")[0], 0)
            self.assertEqual(len(self.sent), 3)


# ================================================================================================ runner gating
class TestDueRunner(unittest.TestCase):
    def test_wait_then_deliver_with_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = cfg()
            c["state_dir"], c["outbox_dir"], c["watchlist_file"] = f"{tmp}/state", f"{tmp}/outbox", f"{tmp}/none.json"
            c["bridge"] = "/bin/false"  # never reach the real bridge from tests
            cpath = Path(tmp) / "config.json"
            cpath.write_text(json.dumps(c, ensure_ascii=False))
            routes = [r for r in TW_ROUTES_21 if not r[0].startswith("BFI82U?type=day&dayDate=20261021")]
            delivered = []
            with (
                mock.patch.object(run, "Fetcher", lambda: fetcher(routes)),
                mock.patch.object(run.deliver, "deliver_outbox", lambda ob, br, log=None: delivered.append(ob) or 0),
            ):
                rc = run.main(["--config", str(cpath), "due", "--now", "2026-10-21T17:50:00+08:00", "--no-llm"])
                self.assertEqual(rc, 0)
                st = json.loads((Path(tmp) / "state" / "state.json").read_text())
                self.assertEqual(st["tw-postmarket:2026-10-21"]["status"], "waiting")
                self.assertFalse(list((Path(tmp) / "outbox").glob("*.json")))
                rc = run.main(["--config", str(cpath), "due", "--now", "2026-10-21T18:50:00+08:00", "--no-llm"])
                st = json.loads((Path(tmp) / "state" / "state.json").read_text())
                self.assertEqual(st["tw-postmarket:2026-10-21"]["status"], "queued")
                entry = json.loads((Path(tmp) / "outbox" / "2026-10-21-tw-postmarket.json").read_text())
                self.assertEqual(entry["filename"], "2026-10-21-tw-postmarket.md")
                self.assertIn("資料未齊", entry["report_md"])
                n = len(list((Path(tmp) / "outbox").glob("*.json")))
                run.main(["--config", str(cpath), "due", "--now", "2026-10-21T18:55:00+08:00", "--no-llm"])
                self.assertEqual(len(list((Path(tmp) / "outbox").glob("*.json"))), n)  # idempotent
                # holiday: recorded once as closed, no fetch of report data, and no failure notice
                run.main(["--config", str(cpath), "due", "--now", "2026-10-26T17:50:00+08:00", "--no-llm"])
                st = json.loads((Path(tmp) / "state" / "state.json").read_text())
                self.assertEqual(st["tw-postmarket:2026-10-26"]["status"], "closed")
                self.assertFalse(list((Path(tmp) / "outbox").glob("*.failed.json")))

    def _cfg(self, tmp):
        c = cfg()
        c["state_dir"], c["outbox_dir"], c["watchlist_file"] = f"{tmp}/state", f"{tmp}/outbox", f"{tmp}/none.json"
        c["bridge"] = "/bin/false"  # never reach the real bridge from tests
        cpath = Path(tmp) / "config.json"
        cpath.write_text(json.dumps(c, ensure_ascii=False))
        return cpath

    def _due(self, cpath, now, routes=TW_ROUTES_21, build=None):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(run, "Fetcher", lambda: fetcher(routes)))
            stack.enter_context(mock.patch.object(run.deliver, "deliver_outbox", lambda ob, br, log=None: 0))
            if build:
                stack.enter_context(mock.patch.object(run, "build", build))
            return run.main(["--config", str(cpath), "due", "--now", now, "--no-llm"])

    def _state(self, tmp):
        return json.loads((Path(tmp) / "state" / "state.json").read_text())

    def test_give_up_queues_one_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._cfg(tmp)
            ob = Path(tmp) / "outbox"
            # us-postmarket 2026-10-22: slot 05:15 TPE on 10-23; quotes are still 10-21 -> required data missing
            self.assertEqual(self._due(cpath, "2026-10-23T05:20:00+08:00"), 0)
            self.assertEqual(self._state(tmp)["us-postmarket:2026-10-22"]["status"], "waiting")
            self.assertFalse(list(ob.glob("*.failed.json")))
            self.assertEqual(self._due(cpath, "2026-10-23T06:40:00+08:00"), 1)  # last tick of the window
            st = self._state(tmp)["us-postmarket:2026-10-22"]
            self.assertEqual((st["status"], st["notice_queued"]), ("failed", True))
            notice = json.loads((ob / "2026-10-22-us-postmarket.failed.json").read_text())
            self.assertEqual(notice["verb"], "brief-failed")
            self.assertIn("S&P 500", notice["reason"])
            (ob / "2026-10-22-us-postmarket.failed.json").unlink()  # pretend delivered
            self._due(cpath, "2026-10-23T06:44:00+08:00")
            self.assertFalse(list(ob.glob("*.failed.json")))  # never twice

    def test_internal_error_and_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._cfg(tmp)
            ob = Path(tmp) / "outbox"
            (Path(tmp) / "state").mkdir()
            (Path(tmp) / "state" / "state.json").write_text(
                json.dumps({"tw-postmarket:2026-10-20": {"status": "waiting", "reason": "BFI82U"}})
            )

            def boom(*a, **kw):
                raise RuntimeError("bug")

            self.assertEqual(self._due(cpath, "2026-10-21T17:50:00+08:00", build=boom), 1)
            st = self._state(tmp)
            self.assertEqual(
                (st["tw-postmarket:2026-10-21"]["status"], st["tw-postmarket:2026-10-21"]["notice_queued"]),
                ("failed", True),
            )
            self.assertIn(
                "internal error RuntimeError: bug",
                json.loads((ob / "2026-10-21-tw-postmarket.failed.json").read_text())["reason"],
            )
            self.assertEqual(st["tw-postmarket:2026-10-20"]["status"], "failed")  # swept: window long over
            self.assertIn(
                "window ended", json.loads((ob / "2026-10-20-tw-postmarket.failed.json").read_text())["reason"]
            )
            n = len(list(ob.glob("*.failed.json")))
            self._due(cpath, "2026-10-21T17:55:00+08:00", build=boom)
            self.assertEqual(len(list(ob.glob("*.failed.json"))), n)

    def test_exchange_closed_no_notice(self):  # typhoon-style closure on a calendar trading day
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._cfg(tmp)
            routes = [("MI_INDEX", "mi_index_closed_20260710.json"), *TW_ROUTES_21]
            self._due(cpath, "2026-10-21T17:50:00+08:00", routes=routes)
            self.assertEqual(self._state(tmp)["tw-postmarket:2026-10-21"]["status"], "closed")
            self.assertFalse(list((Path(tmp) / "outbox").glob("*.json")))

    def test_resend_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._cfg(tmp)
            facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
            md, _summary = render.render(facts, None, None, cfg())
            sent = Path(tmp) / "outbox" / "sent"
            sent.mkdir(parents=True)
            (sent / "2026-10-21-tw-postmarket.json").write_text(json.dumps({"report_md": md}, ensure_ascii=False))
            out = io.StringIO()
            with mock.patch.object(sys, "stdout", out):
                self.assertEqual(
                    run.main(["--config", str(cpath), "resend", "tw-postmarket", "--date", "2026-10-21", "--dry-run"]),
                    0,
                )
            self.assertEqual(out.getvalue().rstrip("\n"), render.telegram_text(md).rstrip("\n"))
            calls = []
            with mock.patch.object(run.deliver, "resend", lambda *a, **kw: calls.append(a) or True):
                self.assertEqual(
                    run.main(["--config", str(cpath), "resend", "tw-postmarket", "--date", "2026-10-21"]), 0
                )
            self.assertEqual(calls[0][:3], ("tw-postmarket", "2026-10-21", render.telegram_text(md)))

    def test_missing_key_exits_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = cfg()
            c["state_dir"], c["outbox_dir"] = f"{tmp}/state", f"{tmp}/outbox"
            c["bridge"] = "/bin/false"
            c["llm"]["key_file"] = f"{tmp}/absent.env"
            cpath = Path(tmp) / "config.json"
            cpath.write_text(json.dumps(c, ensure_ascii=False))
            with (
                mock.patch.object(run, "Fetcher", lambda: fetcher(TW_ROUTES_21)),
                mock.patch.object(run.deliver, "deliver_outbox", lambda ob, br, log=None: 0),
            ):
                rc = run.main(["--config", str(cpath), "due", "--now", "2026-10-21T17:50:00+08:00"])
            self.assertEqual(rc, 3)
            ob = Path(tmp) / "outbox"
            self.assertEqual([p.name for p in ob.glob("*.json")], ["2026-10-21-tw-postmarket.failed.json"])
            self.assertIn(
                "RESEARCH_DRAFT_OPENROUTER_KEY",
                json.loads((ob / "2026-10-21-tw-postmarket.failed.json").read_text())["reason"],
            )


def cnbc_unch(*symbols):
    """CNBC route that reports the given symbols as UNCH with previous close == last (a change that is unknown)."""

    def route(url, body):
        doc = fx("cnbc_quotes.json")
        for q in doc["FormattedQuoteResult"]["FormattedQuote"]:
            if q["symbol"] in symbols:
                q.update(change="UNCH", change_pct="UNCH", previous_day_closing=q["last"])
        return 200, json.dumps(doc).encode()

    return [("quote.cnbc.com", route), *[r for r in TW_ROUTES_21 if r[0] != "quote.cnbc.com"]]


class TestCriticalAndOptionalData(unittest.TestCase):
    """Critical data holds a brief back (retry_gaps); optional data is reported and renders 未取得 but never waits."""

    def test_optional_fx_change_does_not_hold_tw_premarket(self):
        facts = collect.collect("tw-premarket", D22, collector(cnbc_unch(".DXY")))
        self.assertEqual(facts["status"], "ok")
        self.assertTrue(any("美元指數 漲跌未取得" in g for g in facts["gaps"]))
        self.assertEqual(facts["retry_gaps"], [])
        md, _ = render.render(facts, None, None, cfg())
        self.assertIn("| 美元指數 | 101.527 | 未取得 | 未取得 |", md)
        self.assertIn("> **資料未齊**：", md)

    def test_critical_change_holds_tw_premarket(self):
        # SOX is part of the 一句話 (S&P 500 + 費半), so its missing change is waited for; DXY next to it is not
        facts = collect.collect("tw-premarket", D22, collector(cnbc_unch(".SOX", ".DXY")))
        self.assertEqual(facts["retry_gaps"], ["費城半導體 漲跌未取得（資料源回報不變或缺值）"])
        self.assertTrue(any("美元指數 漲跌未取得" in g for g in facts["gaps"]))

    def test_tw_premarket_critical_and_optional_sources(self):
        no_night = [r for r in TW_ROUTES_21 if r[0] not in ("futDailyMarketReport", "mis.taifex")]
        facts = collect.collect("tw-premarket", D22, collector(no_night))
        self.assertIn("台指期夜盤（TAIFEX）未取得", facts["retry_gaps"])  # TX night session: critical
        no_exdiv = [r for r in TW_ROUTES_21 if r[0] != "TWT48U"]
        facts = collect.collect("tw-premarket", D22, collector(no_exdiv))
        self.assertIn("除權息預告（TWT48U）未取得", facts["gaps"])  # ex-dividend calendar: optional
        self.assertEqual(facts["retry_gaps"], [])

    def test_us_postmarket_index_closes_critical_others_optional(self):
        facts = collect.collect("us-postmarket", D21, collector(TW_ROUTES_21))
        self.assertIn("VIX 漲跌未取得（資料源回報不變或缺值）", facts["gaps"])  # fixture: VIX UNCH
        self.assertIn("XLU", " ".join(facts["gaps"]))
        self.assertEqual(facts["retry_gaps"], [])
        facts = collect.collect("us-postmarket", D21, collector(cnbc_unch(".DJI")))
        self.assertEqual(facts["retry_gaps"], ["道瓊工業 漲跌未取得（資料源回報不變或缺值）"])

    def test_missing_quote_classes(self):
        c = collector(TW_ROUTES_21)
        c.critical = collect.CRITICAL_QUOTES["us-premarket"]
        c.gap_missing_quotes("缺：", ["YM", "ES", "N225"])
        self.assertEqual(c.gaps, ["缺：ES", "缺：YM、N225"])
        self.assertEqual(c.retry_gaps, ["缺：ES"])
        self.assertEqual(collect.CRITICAL_QUOTES["tw-postmarket"], frozenset())
        self.assertEqual(set(collect.CRITICAL_QUOTES), set(tradingcal.KINDS))

    def test_postmarket_tables_name_the_gap_in_the_row(self):
        rows = [
            {"label": "美 10 年期公債殖利率", "last": 4.1, "kind": "yield", "chg_bp": None},
            {"label": "WTI 原油", "last": 80.0, "chg": None, "pct": None},
        ]
        self.assertEqual(
            render._quote_rows(rows),
            [["美 10 年期公債殖利率", "4.100%", "未取得", "—"], ["WTI 原油", "80.00", "未取得", "未取得"]],
        )

    def test_due_tick_delivers_on_time_with_optional_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = TestDueRunner()
            cpath = helper._cfg(tmp)
            self.assertEqual(helper._due(cpath, "2026-10-22T08:20:00+08:00", routes=cnbc_unch(".DXY")), 0)
            self.assertEqual(helper._state(tmp)["tw-premarket:2026-10-22"]["status"], "queued")
            md = json.loads((Path(tmp) / "outbox" / "2026-10-22-tw-premarket.json").read_text())["report_md"]
            self.assertIn("| 美元指數 | 101.527 | 未取得 | 未取得 |", md)

    def test_due_tick_still_waits_for_critical_data_then_delivers(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = TestDueRunner()
            cpath = helper._cfg(tmp)
            routes = cnbc_unch(".SOX")
            helper._due(cpath, "2026-10-22T08:20:00+08:00", routes=routes)
            self.assertEqual(helper._state(tmp)["tw-premarket:2026-10-22"]["status"], "waiting")
            self.assertFalse(list((Path(tmp) / "outbox").glob("*.json")))
            helper._due(cpath, "2026-10-22T09:16:00+08:00", routes=routes)  # complete_wait_minutes (60) is over
            self.assertEqual(helper._state(tmp)["tw-premarket:2026-10-22"]["status"], "queued")


class TestLLMCache(unittest.TestCase):
    GOOD = {
        "one_liner": "加權指數收 46,887.77。",
        "interpretation": "投信買超 127.99 億。",
        "watch_points": ["觀察成交量"],
    }

    def setUp(self):
        self.calls = []

    def transport(self, method, url, headers, body, timeout):
        self.calls.append(1)
        resp = {
            "model": "m",
            "id": "g",
            "usage": {"cost": 0.002},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.GOOD, ensure_ascii=False)}}],
        }
        return 200, json.dumps(resp).encode()

    def facts(self, routes=TW_ROUTES_21):
        return collect.collect("tw-postmarket", D21, collector(routes))

    def test_same_data_reuses_the_section_and_changed_data_calls_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts, lcfg = self.facts(), cfg()["llm"]
            sec, meta = llm.interpret_cached(facts, "k", lcfg, tmp, transport=self.transport)
            self.assertEqual((len(self.calls), meta["cost"], meta.get("cached")), (1, 0.002, None))
            later = dict(facts, as_of="later tick, same data", generated_at="2026-10-21T18:05:00+08:00")
            sec2, meta2 = llm.interpret_cached(later, "k", lcfg, tmp, transport=self.transport)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual((sec2, meta2["cached"], meta2["cost"], meta2["cached_cost"]), (sec, True, 0.0, 0.002))
            self.assertEqual(meta2["status"], "ok")
            llm.interpret_cached(
                dict(facts, gaps=[*facts["gaps"], "new gap"]), "k", lcfg, tmp, transport=self.transport
            )
            self.assertEqual(len(self.calls), 2)  # a data change misses
            llm.interpret_cached(
                dict(facts, gaps=[*facts["gaps"], "new gap"]), "k", lcfg, tmp, transport=self.transport
            )
            self.assertEqual(len(self.calls), 2)
            other = dict(lcfg, model="other/model")
            llm.interpret_cached(facts, "k", other, tmp, transport=self.transport)
            self.assertEqual(len(self.calls), 3)  # a model change misses

    def test_refused_output_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = dict(self.GOOD, one_liner="加權指數收 47,000.12。")

            def t(*a):
                return self.transport(*a)[0], json.dumps(
                    {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(bad)}}]}
                ).encode()

            sec, _meta = llm.interpret_cached(self.facts(), "k", cfg()["llm"], tmp, transport=t)
            self.assertIsNone(sec)
            self.assertEqual(list(Path(tmp).glob("*.json")), [])

    def test_old_cache_files_are_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "2026-10-01-tw-postmarket.json"
            old.write_text("{}")
            os.utime(old, (1, 1))
            llm.interpret_cached(self.facts(), "k", cfg()["llm"], tmp, transport=self.transport)
            self.assertFalse(old.exists())
            self.assertTrue((Path(tmp) / "2026-10-21-tw-postmarket.json").exists())

    def test_build_does_not_call_the_model_while_waiting_for_critical_data(self):
        routes = cnbc_unch(".SOX")
        kw = dict(cfg=cfg(), calendar=calendar(), watchlist=WATCH, key="k", log=lambda m: None)
        now = dt.datetime(2026, 10, 22, 0, 20, tzinfo=dt.UTC)
        with tempfile.TemporaryDirectory() as tmp:
            args = dict(fetcher=fetcher(routes), now_utc=now, llm_transport=self.transport, cache_dir=tmp)
            facts, md, _s, meta = run.build("tw-premarket", D22, may_wait=True, **args, **kw)
            self.assertEqual((facts["status"], md, meta, len(self.calls)), ("ok", None, None, 0))
            # the wait is over: the same build now renders, with the model
            facts, md, _s, meta = run.build("tw-premarket", D22, may_wait=False, **args, **kw)
            self.assertGreaterEqual(len(self.calls), 1)
            self.assertIsNotNone(md)


class TestLLMContentRules(unittest.TestCase):
    def setUp(self):
        self.facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))

    def _resp(self, obj):
        body = {
            "model": "m",
            "usage": {"cost": 0.0},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(obj, ensure_ascii=False)}}],
        }
        return 200, json.dumps(body).encode()

    def test_tech_terms_and_normalize(self):
        self.assertEqual(
            llm.tech_terms("加權指數跌破季線，46,887.77 點附近有支撐；MA5、KD 黃金交叉"),
            ["KD", "MA5", "季線", "支撐", "黃金交叉"],
        )
        self.assertEqual(llm.tech_terms("賣壓沉重、電子族群壓力大、相對有撐"), [])  # ordinary words are fine
        self.assertEqual(
            llm.normalize("台積電收 2,420.0 元、漲 0.6%，指數 46887.77、跌 -1.5%"),
            "台積電收 2,420 元、漲 0.60%，指數 46,887.77、跌 -1.50%",
        )
        sec, reasons = llm.validate(
            {"one_liner": "加權指數收 46,887.77。", "interpretation": "跌破季線。", "watch_points": ["x"]}, self.facts
        )
        self.assertIsNone(sec)
        self.assertTrue(any("季線" in r for r in reasons))

    def test_repair_then_drop(self):
        bad = {
            "one_liner": "加權指數收 46,887.77。",
            "interpretation": "投信買超 127.99 億。加權指數站上月線。\n\n成交量擴大。",
            "watch_points": ["觀察 46,887.77 點支撐", "觀察投信動向"],
        }
        calls = []

        def t(method, url, headers, body, timeout):
            calls.append(1)
            return self._resp(bad)

        sec, meta = llm.interpret(self.facts, "k", cfg()["llm"], transport=t)
        self.assertEqual(len(calls), 2)  # one repair round
        self.assertEqual(meta["status"], "ok_terms_dropped")
        self.assertEqual(sec["interpretation"], "投信買超 127.99 億。\n\n成交量擴大。")
        self.assertEqual(sec["watch_points"], ["觀察投信動向"])
        self.assertEqual(len(meta["dropped"]), 2)
        md, _ = render.render(self.facts, sec, meta, cfg())
        self.assertIn("_（已移除 2 句資料中沒有依據的技術面描述）_", md)
        self.assertNotIn("月線", md)
        good = dict(bad, interpretation="投信買超 127.99 億。", watch_points=["觀察投信動向"])
        sec, meta = llm.interpret(self.facts, "k", cfg()["llm"], transport=lambda *a: self._resp(good))
        self.assertEqual(meta["status"], "ok")

    def test_code_table(self):
        lines = render.code_table(
            ["代號", "名稱", "收盤", "漲跌%"],
            [["2330", "台積電", "2,420", "▲+0.62%"], ["0050", "元大台灣50", "資料未取得", "—"]],
            render._tg_spec("代號=Code; 收盤=Close; 漲跌%=Chg%; 名稱"),
        )
        self.assertEqual(
            lines,
            ["```", "Code  Close    Chg% 名稱", "2330  2,420  +0.62% 台積電", "0050      -       - 元大台灣50", "```"],
        )

    def test_code_table_ascii_last_column_has_no_double_sign(self):
        lines = render.code_table(
            ["代號", "收盤", "盤後%"],
            [["TSM", "398.63", "▲+0.53%"], ["AAPL", "338.32", "▼-2.58%"]],
            render._tg_spec("代號=Sym; 收盤=Close; 盤後%=AH%"),
        )
        self.assertEqual(lines[1:-1], ["Sym    Close AH%", "TSM   398.63 +0.53%", "AAPL  338.32 -2.58%"])


def _slim_md(extra_tg="", rows=12):
    body = "\n".join(f"| 項{n:02d} | {100 - n}.00 | {'▲+' if n < 3 else '▼-'}{n}.00% |" for n in range(rows))
    return (
        "# T\n\n## 一句話\n甲 [3]\n\n## 類股表現（12 類，依漲跌排序）\n\n"
        f"<!-- tg: 收盤=Close; 漲跌%=Chg%!; 名稱{extra_tg} -->\n| 名稱 | 收盤 | 漲跌% |\n|---|---:|---:|\n{body}\n\n"
        f"{render.MD_ONLY}只給 md 看的說明。\n\n## 解讀（非投資建議）\n乙 [1][1]，丙 [2](https://x.test)。\n\n"
        "## 新聞標題（解讀引用編號）\n"
        "1. [一](https://n.test/1) — 甲報\n2. [二](https://n.test/2) — 乙報\n3. [三](https://n.test/3) — 丙報\n"
        "4. [四](https://n.test/4) — 丁報\n\n## 資料來源\n- [S](https://s.test)\n\n" + render.DISCLAIMER + "\n"
    )


class TestPhoneSlimming(unittest.TestCase):
    def test_edge_head_only_and_title(self):
        tg = render.telegram_text(_slim_md("; @edge=3; @title=類股表現（漲跌前後各 3）"))
        self.assertIn("**類股表現（漲跌前後各 3）**", tg)
        self.assertNotIn("依漲跌排序", tg)
        block = tg.split("```")[1].strip("\n").split("\n")
        self.assertEqual(len(block), 1 + 3 + 1 + 3)  # header, top 3, gap marker, bottom 3
        self.assertIn("... 6 omitted", block[4])
        self.assertTrue(block[1].endswith("項00 🔴") and block[-1].endswith("項11 🟢"))
        head = render.telegram_text(_slim_md("; @head=4")).split("```")[1].strip("\n").split("\n")
        self.assertEqual([r.split()[-2] for r in head[1:]], ["項00", "項01", "項02", "項03"])
        only = render.telegram_text(_slim_md("; @only=項05,項02,不存在")).split("```")[1].strip("\n").split("\n")
        self.assertEqual([r.split()[-2] for r in only[1:]], ["項05", "項02"])  # in the order given
        self.assertEqual(len(render.telegram_text(_slim_md()).split("```")[1].strip("\n").split("\n")), 13)  # no opts

    def test_unmatched_only_keeps_all_rows(self):
        tg = render.telegram_text(_slim_md("; @only=沒有"))
        self.assertEqual(len(tg.split("```")[1].strip("\n").split("\n")), 13)

    def test_md_only_lines_and_sources_stay_off_the_phone(self):
        md = _slim_md()
        self.assertIn("只給 md 看的說明", md)
        self.assertIn("## 資料來源", md)
        tg = render.telegram_text(md)
        self.assertNotIn("只給 md 看的說明", tg)
        self.assertNotIn("資料來源", tg)
        self.assertNotIn("s.test", tg)

    def test_news_limited_to_cited_headlines_with_stable_numbers(self):
        md = _slim_md()
        tg = render.telegram_text(md)
        self.assertEqual(
            re.findall(r"^(\d+)\. \[", tg, re.M), ["1", "3"]
        )  # [3] from 一句話, [1] from 解讀; [2](url) is a link
        self.assertIn("1. [一](https://n.test/1) — 甲報", tg)
        self.assertNotIn("https://n.test/2", tg)
        self.assertEqual(len(re.findall(r"^\d+\. \[", md, re.M)), 4)  # the md keeps every headline

    def test_no_citation_drops_the_news_section_from_the_phone_only(self):
        md = _slim_md().replace("[3]", "").replace("[1][1]", "")
        tg = render.telegram_text(md)
        self.assertNotIn("新聞標題", tg)
        self.assertNotIn("n.test", tg)
        self.assertIn("## 新聞標題", md)

    def test_llm_prompt_asks_for_the_short_form(self):
        facts = {"kind": "k", "date": "d", "title": "t", "as_of": "a", "gaps": [], "data": {}, "headlines": []}
        msgs = llm.build_messages(facts)
        self.assertIn("<= 350 characters", msgs[0]["content"])
        self.assertIn("Cite sparingly", msgs[0]["content"])
        long = {"one_liner": "x", "interpretation": "字" * 701, "watch_points": ["y"]}
        _sec, reasons = llm.validate(long, {"headlines": [], "data": {}, "date": "d"})
        self.assertTrue(any("longer than 700" in r for r in reasons))


class TestBridgeNexus(unittest.TestCase):
    """Nexus raw archive written by the bridge `brief` / `brief-archive` verbs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b = TestBridgePatch()
        self.mod = self.b.load(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def md(self):
        facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        facts["gaps"] = ["三大法人未取得"]
        return render.render(facts, None, None, cfg())[0]

    def test_raw_schema_matches_lint(self):
        raw = self.mod.to_nexus_raw("2026-10-21-tw-postmarket.md", self.md(), today="2026-10-22")
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        fm = yaml.safe_load(raw.split("---\n")[1])
        # nexus_pull to_nexus_raw() schema + lint REQUIRED / enums / dates / tags list
        for k in ("type", "title", "status", "created", "updated", "tags", "origin"):
            self.assertIn(k, fm)
        self.assertEqual((fm["type"], fm["status"], fm["origin"]), ("source", "stable", "agent"))
        self.assertEqual(fm["created"], "2026-10-22")
        self.assertEqual(fm["updated"], "2026-10-22")
        self.assertEqual(fm["tags"], ["market_daily", "tw_stock", "postmarket", "market-daily"])
        self.assertEqual(fm["source_system"], "market-brief")
        self.assertEqual(fm["source_kind"], "market_daily")
        self.assertEqual(fm["source_path"], "stormtrooper:market-brief/2026-10-21-tw-postmarket.md")
        self.assertIs(fm["compiled"], False)
        self.assertEqual(fm["compiled_into"], [])
        self.assertTrue(any("MI_INDEX" in x for x in fm["sources"]))
        self.assertEqual((fm["market"], fm["session"], fm["session_date"]), ("tw", "postmarket", "2026-10-21"))
        self.assertEqual(fm["data_gaps"], ["三大法人未取得"])
        self.assertIn("as_of", fm)
        self.assertIn("## 大盤", raw)

    def test_add_only_duplicate_sensitive_and_archive_verb(self):
        mod, name = self.mod, "2026-10-21-tw-postmarket.md"
        md = self.md()
        self.assertEqual(mod.nexus_archive(name, md), "written")
        self.assertEqual(mod.nexus_archive(name, md), "duplicate")
        dest = Path(mod.NEXUS) / "_kb/raw/market-daily" / name
        dest.write_text(dest.read_text(encoding="utf-8").replace("compiled: false", "compiled: true"), encoding="utf-8")
        self.assertEqual(mod.nexus_archive(name, md), "duplicate")  # compile may flip frontmatter; body unchanged
        with self.assertRaises(SystemExit) as e:
            mod.nexus_archive(name, md.replace("## 大盤", "## 大盤（改）"))
        self.assertEqual(e.exception.code, 2)
        self.assertEqual(mod.nexus_archive("2026-10-22-tw-postmarket.md", md + "\napi_key=abcdefghijk\n"), "sensitive")
        self.assertFalse((Path(mod.NEXUS) / "_kb/raw/market-daily/2026-10-22-tw-postmarket.md").exists())
        os.remove(Path(mod.NEXUS) / "AGENTS.md")
        self.assertEqual(mod.nexus_archive("2026-10-23-tw-postmarket.md", md), "no-vault")
        (Path(mod.NEXUS) / "AGENTS.md").write_text("x")
        # brief-archive: only for a brief already delivered to market-briefs/
        self.assertEqual(self.b.call(mod, "brief-archive tw-premarket 2026-10-22", "")[0], 2)
        os.makedirs(mod.BRIEFS, exist_ok=True)
        (Path(mod.BRIEFS) / "2026-10-22-tw-premarket.md").write_text(md, encoding="utf-8")
        code, out = self.b.call(mod, "brief-archive tw-premarket 2026-10-22", "")
        self.assertEqual((code, json.loads(out)["nexus"]), (0, "written"))
        self.assertEqual(self.b.sent, [])  # no Telegram

    def test_brief_writes_nexus_before_telegram(self):
        mod = self.mod
        md = '---\ntype: market_daily\ntitle: "t"\npromote: nexus\n---\n# 台股\n'
        self.b.ok = False  # Telegram down: the archive is still written
        p = {"filename": "2026-10-22-us-premarket.md", "summary": "s", "report_md": md}
        self.assertEqual(self.b.call(mod, "brief us-premarket 2026-10-22", p)[0], 1)
        self.assertTrue((Path(mod.NEXUS) / "_kb/raw/market-daily/2026-10-22-us-premarket.md").is_file())
        self.b.ok = True
        code, out = self.b.call(mod, "brief us-premarket 2026-10-22", p)
        self.assertEqual((code, json.loads(out)["nexus"]), (0, "duplicate"))


class TestBriefServer(unittest.TestCase):
    def setUp(self):
        import ast
        import threading

        import brief_server

        ast.parse((HERE / "brief_server.py").read_text(encoding="utf-8"), feature_version=(3, 9))
        self.bs = brief_server
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "market-daily"
        root.mkdir()
        b = TestBridgePatch()
        mod = b.load(self.tmp.name + "/bridge")
        facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        facts["gaps"] = ["三大法人未取得"]
        md = render.render(
            facts,
            {"one_liner": "加權指數收 46,887.77。", "interpretation": "x", "watch_points": ["y"]},
            {"returned_model": "m"},
            cfg(),
        )[0]
        (root / "2026-10-21-tw-postmarket.md").write_text(
            mod.to_nexus_raw("2026-10-21-tw-postmarket.md", md, today="2026-10-22"), encoding="utf-8"
        )
        legacy = "---\ntype: source\ntitle: 台股盤勢日報 2026-10-21（三）\ncreated: '2026-10-21'\ntags:\n- tw_stock\ncompiled: false\n---\n\n# x\n\n## 一句話\n加權指數收 46,887.77。\n"
        (root / "2026-10-21-tw-market.md").write_text(legacy, encoding="utf-8")
        (root / "2026-10-22-us-premarket.md").write_text(
            '---\ntitle: "美股盤前快報"\nsession_date: 2026-10-22\ndata_gaps: []\n---\n# y\n', encoding="utf-8"
        )
        (root / "notes.txt").write_text("secret")
        (Path(self.tmp.name) / "outside.md").write_text("outside")
        os.symlink(Path(self.tmp.name) / "outside.md", root / "2026-10-20-tw-postmarket.md")
        self.root = root
        self.srv = brief_server.make_server(str(root), ["127.0.0.1"], "127.0.0.1", 0)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def req(self, path, method="GET"):
        import http.client

        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), body

    def test_index_and_files(self):
        st, ct, body = self.req("/index.json")
        self.assertEqual((st, ct), (200, "application/json; charset=utf-8"))
        idx = json.loads(body)
        self.assertEqual(set(idx), {"generated_at", "briefs"})
        self.assertEqual(
            [(b["date"], b["kind"]) for b in idx["briefs"]],
            [("2026-10-22", "us-premarket"), ("2026-10-21", "tw-postmarket"), ("2026-10-21", "tw-postmarket-legacy")],
        )
        tw = idx["briefs"][1]
        self.assertEqual(tw["title"], "台股盤後日報 2026-10-21（三）")
        self.assertEqual(tw["summary"], "加權指數收 46,887.77。")
        self.assertEqual(tw["session_date"], "2026-10-21")
        self.assertEqual(tw["data_gaps"], ["三大法人未取得"])
        self.assertEqual(tw["file"], "2026-10-21-tw-postmarket.md")
        self.assertEqual(tw["bytes"], (self.root / tw["file"]).stat().st_size)
        self.assertEqual(set(tw), {"date", "kind", "title", "file", "summary", "bytes", "session_date", "data_gaps"})
        self.assertEqual(idx["briefs"][2]["title"], "台股盤勢日報 2026-10-21（三）")
        st, ct, body = self.req("/2026-10-21-tw-postmarket.md")
        self.assertEqual((st, ct), (200, "text/markdown; charset=utf-8"))
        self.assertTrue(body.decode("utf-8").startswith("---\ntype: source"))
        st, _, body = self.req("/2026-10-21-tw-postmarket.md", "HEAD")
        self.assertEqual((st, body), (200, b""))

    def test_rejections(self):
        for path in (
            "/",
            "/notes.txt",
            "/../outside.md",
            "/%2e%2e/outside.md",
            "/x/2026-10-21-tw-postmarket.md",
            "/2026-10-20-tw-postmarket.md",
            "/2026-10-23-tw-postmarket.md",
            "/index.html",
            "/2026-10-21-TW.md",
            "/..%2Foutside.md",
        ):
            self.assertEqual(self.req(path)[0], 404, path)
        self.assertEqual(self.req("/index.json", "POST")[0], 405)
        self.assertEqual(self.req("/index.json", "DELETE")[0], 405)
        self.srv.RequestHandlerClass.allow = frozenset({"192.168.31.181"})
        self.assertEqual(self.req("/index.json")[0], 403)
        self.assertEqual(self.req("/2026-10-21-tw-postmarket.md")[0], 403)

    def test_frontmatter_parser(self):
        fm, body = self.bs.parse_frontmatter(
            "---\na: \"x: y\"\nb: 'it''s'\nc: [p, q]\nd:\n- \"1\"\n- two\ne: []\nf: false\n---\nbody"
        )
        self.assertEqual(fm, {"a": "x: y", "b": "it's", "c": ["p", "q"], "d": ["1", "two"], "e": [], "f": False})
        self.assertEqual(body, "body")
        self.assertEqual(self.bs.parse_frontmatter("no frontmatter"), ({}, "no frontmatter"))


class TestMirror(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b = TestBridgePatch()
        self.mod = self.b.load(self.tmp.name)
        facts = collect.collect("tw-postmarket", D21, collector(TW_ROUTES_21))
        self.md = render.render(facts, None, None, cfg())[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_mirrors_exact_bytes(self):
        mod, name = self.mod, "2026-10-21-tw-postmarket.md"
        self.assertEqual(mod.nexus_archive(name, self.md), "written")
        src = Path(mod.NEXUS) / "_kb/raw/market-daily" / name
        dst = Path(mod.MIRROR) / name
        self.assertEqual(dst.read_bytes(), src.read_bytes())
        self.assertEqual(oct(Path(mod.MIRROR).stat().st_mode & 0o777), "0o700")
        self.assertEqual(oct(dst.stat().st_mode & 0o777), "0o600")
        dst.unlink()
        self.assertEqual(mod.nexus_archive(name, self.md), "duplicate")  # identical note -> mirrored again
        self.assertTrue(dst.is_file())
        self.assertEqual(
            mod.nexus_archive("2026-10-22-tw-postmarket.md", self.md + "\ntoken=abcdefghij\n"), "sensitive"
        )
        self.assertFalse((Path(mod.MIRROR) / "2026-10-22-tw-postmarket.md").exists())

    def test_mirror_sync(self):
        mod = self.mod
        src_dir = Path(mod.NEXUS) / "_kb/raw/market-daily"
        src_dir.mkdir(parents=True)
        (src_dir / "2026-10-21-tw-market.md").write_text("legacy", encoding="utf-8")  # put there by nexus_pull
        (src_dir / "2026-10-22-tw-postmarket.md").write_text("v1", encoding="utf-8")
        (src_dir / "notes.txt").write_text("x")
        out = json.loads(self.b.call(mod, "brief-mirror-sync", "")[1])
        self.assertEqual(out, {"copied": 2, "deleted": 0, "unchanged": 0})
        self.assertEqual(sorted(os.listdir(mod.MIRROR)), ["2026-10-21-tw-market.md", "2026-10-22-tw-postmarket.md"])
        self.assertEqual(
            json.loads(self.b.call(mod, "brief-mirror-sync", "")[1]), {"copied": 0, "deleted": 0, "unchanged": 2}
        )
        (src_dir / "2026-10-22-tw-postmarket.md").write_text("v2-longer", encoding="utf-8")
        (src_dir / "2026-10-21-tw-market.md").unlink()
        out = json.loads(self.b.call(mod, "brief-mirror-sync", "")[1])
        self.assertEqual(out, {"copied": 1, "deleted": 1, "unchanged": 0})
        self.assertEqual((Path(mod.MIRROR) / "2026-10-22-tw-postmarket.md").read_text(), "v2-longer")
        self.assertEqual(self.b.call(mod, "brief-mirror-sync extra", "")[0], 2)
        self.assertIn("brief-mirror-sync copied=1 deleted=1", Path(mod.LOG).read_text())

    def test_client_mirror_sync_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir, outbox = Path(tmp) / "state", Path(tmp) / "outbox"
            state_dir.mkdir()
            calls = []
            t0 = dt.datetime(2026, 10, 22, 10, 0, tzinfo=dt.UTC)
            with mock.patch.object(run.deliver, "mirror_sync", lambda bridge, log=None: calls.append(bridge) or True):
                run.maybe_mirror_sync(cfg(), state_dir, outbox, 0, t0, print)  # never synced -> sync
                run.maybe_mirror_sync(cfg(), state_dir, outbox, 0, t0 + dt.timedelta(minutes=30), print)  # < 1 h
                (outbox / "sent").mkdir(parents=True)
                (outbox / "sent" / "x.json").write_text("{}")
                run.maybe_mirror_sync(cfg(), state_dir, outbox, 0, t0 + dt.timedelta(minutes=35), print)  # delivered
                run.maybe_mirror_sync(cfg(), state_dir, outbox, 1, t0 + dt.timedelta(minutes=95), print)  # 1 h since
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads((state_dir / "state.json").read_text())["_mirror_sync"]["ok"], True)
        runner_calls = []

        def runner(cmd, **kw):
            runner_calls.append(cmd)
            return mock.Mock(returncode=255, stdout="", stderr="ssh: connect failed")

        logs = []
        self.assertFalse(run.deliver.mirror_sync("ssh research-bridge", log=logs.append, runner=runner))
        self.assertEqual(runner_calls[0], ["ssh", "research-bridge", "brief-mirror-sync"])
        self.assertIn("exit 255", logs[0])


# ================================================================================================ group strength
GROUP_KEY = "tg-test-key-0123456789abcdef"


def group_reply(d=D21, **over):
    def g(name, rank, r20, r60, share, chg, n=4, valid=4, missing=()):
        return {
            "group": name, "n_members": n, "n_valid": valid, "missing": list(missing), "members": [],
            "ret20": r20, "ret60": r60, "rel20": None, "rel60": None, "turnover_share": share,
            "turnover_share_change": chg, "score": None, "rank": rank, "rank_ex_flow": rank,
        }  # fmt: skip

    resp = {
        "date": d.isoformat(),
        "table_version": "2026-10-30-v1",
        "table_sha": "abc123abc123",
        "min_members": 2,
        "taiex": {"ret20": -0.034488, "ret60": 0.161144},
        "groups": [
            g("PCB 材料", 1, 0.102931, 0.904889, 0.051935, 0.312588),
            g("散熱", 2, -0.075877, 0.724185, 0.028786, 0.013613, 5, 4, ["8996"]),
            g("電源", 3, -0.135707, 0.110137, 0.022431, -0.176795, 3, 3),
            g("網通", None, None, None, 0.008628, 0.146662, 2, 0, ["2345", "5388"]),
        ],
    }
    resp.update(over)
    return resp


class GroupTransport:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.calls = status, body, []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers)))
        if "/api/v1/tw/groups/ranking" in url:
            raw = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()
            return self.status, raw
        return 404, b"not found"


class TestGroupRanking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keyfile = Path(self.tmp.name) / "thalassa.env"
        self.keyfile.write_text(f"export MARKET_BRIEF_THALASSA_API_KEY={GROUP_KEY}\n")
        os.chmod(self.keyfile, 0o600)
        self.cfg = cfg()
        self.cfg["thalassa"]["key_file"] = str(self.keyfile)
        self.logs = []

    def fetch(self, transport, d=D21):
        f = Fetcher(transport=transport, sleep=lambda s: None)
        out = tw_groups.fetch(f, self.cfg, d, self.logs.append)
        self.assertNotIn(GROUP_KEY, json.dumps(f.log) + "\n".join(self.logs))  # the key is never logged
        return out

    def test_fetch_ok_sends_key_header_and_shapes_percent(self):
        t = GroupTransport(200, group_reply())
        data, status = self.fetch(t)
        self.assertEqual(status, "ok")
        ((method, url, headers),) = t.calls
        self.assertEqual((method, url), ("GET", "http://192.168.31.241:8001/api/v1/tw/groups/ranking?date=2026-10-21"))
        self.assertEqual(headers["X-API-Key"], GROUP_KEY)
        self.assertEqual(data["taiex"], {"ret20": -3.4, "ret60": 16.1})
        first = data["groups"][0]
        self.assertEqual((first["name"], first["rank"], first["ret20"], first["ret60"]), ("PCB 材料", 1, 10.3, 90.5))
        self.assertEqual((first["share"], first["share_chg"]), (5.2, 31.3))
        self.assertIsNone(data["groups"][3]["rank"])

    def test_every_failure_skips_the_section_and_logs_why(self):
        cases = {}
        cases["no key file"] = lambda: (self.keyfile.unlink(), GroupTransport(200, group_reply()))[1]
        cases["wrong variable"] = lambda: (
            self.keyfile.write_text(f"export OTHER_VAR={GROUP_KEY}\n"),
            GroupTransport(200, group_reply()),
        )[1]
        cases["open permissions"] = lambda: (os.chmod(self.keyfile, 0o644), GroupTransport(200, group_reply()))[1]
        cases["http 500"] = lambda: GroupTransport(500, b"boom")
        cases["http 401"] = lambda: GroupTransport(401, b'{"detail":"Invalid API key"}')
        cases["http 404"] = lambda: GroupTransport(404, b'{"detail":"x"}')
        cases["not json"] = lambda: GroupTransport(200, b"<html>")
        cases["wrong date"] = lambda: GroupTransport(200, group_reply(date(2026, 10, 20)))
        cases["no groups"] = lambda: GroupTransport(200, group_reply(groups=[]))
        cases["missing field"] = lambda: GroupTransport(200, group_reply(groups=[{"group": "A", "rank": 1}]))
        cases["no taiex"] = lambda: GroupTransport(200, group_reply(taiex={"ret20": None, "ret60": None}))
        for name, make in cases.items():
            self.setUp()
            with self.subTest(name):
                data, status = self.fetch(make())
                self.assertIsNone(data)
                self.assertNotEqual(status, "ok")
                self.assertTrue(any("section skipped" in m for m in self.logs), self.logs)

    def test_no_fallback_to_environment(self):
        self.keyfile.unlink()
        env = {"MARKET_BRIEF_THALASSA_API_KEY": "from-env", "RESEARCH_DRAFT_OPENROUTER_KEY": "other"}
        t = GroupTransport(200, group_reply())
        with mock.patch.dict(os.environ, env):
            data, _ = self.fetch(t)
        self.assertIsNone(data)
        self.assertEqual(t.calls, [])  # no request without the file's key

    def _collect(self, transport):
        routes = [("groups/ranking", lambda url, body: transport("GET", url, {}, body, 1)), *TW_ROUTES_21]
        c = collect.Collector(
            fetcher(routes), self.cfg, WATCH, calendar(), dt.datetime(2026, 10, 22, 8, 30, tzinfo=dt.UTC),
            log=self.logs.append,
        )  # fmt: skip
        return collect.collect("tw-postmarket", D21, c)

    def test_section_in_md_and_telegram(self):
        facts = self._collect(GroupTransport(200, group_reply()))
        self.assertEqual(facts["status"], "ok")
        self.assertEqual(facts["group_ranking_status"], "ok")
        self.assertNotIn("group_ranking", json.dumps(facts["data"]))  # the model never sees it
        md, _ = render.render(facts, None, None, cfg())
        for needle in (
            "## 族群強弱（4 族群，等權、已還原除權息）",
            "加權指數 20 日 -3.4%、60 日 +16.1%",
            "| # | 族群 | 檔數 | 20日 | 60日 | 占比% | 占比變化 |",
            f"| 1 | PCB 材料 | 4 | {U('+10.3%')} | {U('+90.5%')} | 5.2 | {U('+31.3%')} |",
            f"| 3 | 電源 | 3 | {D('-13.6%')} | {U('+11.0%')} | 2.2 | {D('-17.7%')} |",
            "| — | 網通 | 2 | — | — | 0.9 | " + U("+14.7%") + " |",
            "<!-- tg: #=#; 20日=R20!; 60日=R60; 占比%=Shr; 占比變化=ShrChg; 族群 -->",
            "個股資料缺漏（未計入該族群）：散熱 8996；網通 2345、5388",
            "不是買進訊號",
        ):
            self.assertIn(needle, md)
        self.assertLess(md.index("## 類股表現"), md.index("## 族群強弱"))
        self.assertLess(md.index("## 族群強弱"), md.index("## 三大法人"))
        self.assertNotIn(GROUP_KEY, md)
        for pat in NEXUS_SENSITIVE_CI:
            self.assertIsNone(re.search(pat, md, re.I | re.M), pat)
        tg = render.telegram_text(md)
        block = tg.split("**族群強弱（4 族群，等權、已還原除權息）**")[1].split("**三大法人")[0]
        self.assertIn("不是買進訊號", block)
        table = block.split("```")[1].strip("\n").split("\n")
        self.assertEqual(len(table), 5)  # header + 4 groups
        self.assertRegex(table[1], r"^1\s+\+10\.3%\s+\+90\.5%\s+5\.2\s+\+31\.3% PCB 材料 🔴$")
        self.assertRegex(table[0], r"^#\s+R20\s+R60\s+Shr\s+ShrChg 族群$")
        self.assertTrue(table[3].endswith("電源 🟢"), table[3])
        self.assertLessEqual(max(len(r) for r in table), 40)

    def test_brief_goes_out_without_section_when_endpoint_fails(self):
        facts = self._collect(GroupTransport(500, b"boom"))
        self.assertEqual(facts["status"], "ok")
        self.assertIsNone(facts["group_ranking"])
        self.assertIn("HTTP 500", facts["group_ranking_status"])
        self.assertEqual(facts["gaps"], [])
        md, _ = render.render(facts, None, None, self.cfg)
        self.assertNotIn("## 族群強弱", md)
        self.assertIn("族群強弱：本期未取得", md)
        self.assertNotIn("HTTP 500", md)  # the reason (URL, key path) never reaches the reader
        self.assertIn("族群強弱：本期未取得", render.telegram_text(md))
        self.assertIn("## 三大法人", md)
        rec = run.run_record(facts, None, Fetcher())
        self.assertIn("HTTP 500", rec["group_ranking"])
        self.assertTrue(any("section skipped" in m for m in self.logs))

    def _assert_brief_without_section(self, facts):
        self.assertEqual(facts["status"], "ok")
        self.assertIsNone(facts["group_ranking"])
        self.assertEqual(facts["gaps"], [])  # an optional section never delays the brief
        md, _ = render.render(facts, None, None, self.cfg)
        self.assertIn("族群強弱：本期未取得", md)
        self.assertIn("族群強弱：本期未取得", render.telegram_text(md))
        self.assertIn("## 三大法人", md)
        for secret in (GROUP_KEY, str(self.keyfile), "192.168.31.241"):
            self.assertNotIn(secret, md)
        self.assertTrue(any("section skipped" in m for m in self.logs), self.logs)

    def test_unreadable_key_file_still_sends_the_brief(self):
        real = Path.read_text

        def deny_key_file(path, *a, **kw):  # the key file passes the mode check but cannot be read
            if path == self.keyfile:
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, *a, **kw)

        with mock.patch.object(Path, "read_text", deny_key_file):
            self._assert_brief_without_section(self._collect(GroupTransport(200, group_reply())))

    def test_non_utf8_key_file_still_sends_the_brief(self):
        self.keyfile.write_bytes(b"export MARKET_BRIEF_THALASSA_API_KEY=\xff\xfe\x80\n")
        os.chmod(self.keyfile, 0o600)
        self._assert_brief_without_section(self._collect(GroupTransport(200, group_reply())))

    def test_config_missing_base_url_still_sends_the_brief(self):
        del self.cfg["thalassa"]["base_url"]
        self._assert_brief_without_section(self._collect(GroupTransport(200, group_reply())))

    def test_non_dict_group_in_reply_still_sends_the_brief(self):
        reply = group_reply()
        reply["groups"].append("oops")
        self._assert_brief_without_section(self._collect(GroupTransport(200, reply)))

    def test_no_thalassa_config_shows_no_skip_line(self):
        facts = self._collect(GroupTransport(200, group_reply()))
        facts["group_ranking"] = None
        md, _ = render.render(facts, None, None, {k: v for k, v in self.cfg.items() if k != "thalassa"})
        self.assertNotIn("族群強弱", md)

    def test_groups_are_ordered_by_rank_whatever_the_reply_order(self):
        reply = group_reply()
        reply["groups"].reverse()  # unranked first, rank 3, 2, 1
        data, status = self.fetch(GroupTransport(200, reply))
        self.assertEqual(status, "ok")
        self.assertEqual([g["name"] for g in data["groups"]], ["PCB 材料", "散熱", "電源", "網通"])
        md, _ = render.render(self._collect(GroupTransport(200, reply)), None, None, self.cfg)
        self.assertLess(md.index("| 1 | PCB 材料"), md.index("| 2 | 散熱"))
        self.assertLess(md.index("| 3 | 電源"), md.index("| — | 網通"))

    def test_other_slots_do_not_call_thalassa(self):
        t = GroupTransport(200, group_reply())
        routes = [("groups/ranking", lambda url, body: t("GET", url, {}, body, 1)), *TW_ROUTES_21]
        c = collect.Collector(
            fetcher(routes), self.cfg, WATCH, calendar(), dt.datetime(2026, 10, 22, 8, 30, tzinfo=dt.UTC)
        )
        collect.collect("tw-premarket", D22, c)
        self.assertEqual(t.calls, [])


# ================================================================================================ credit (信用交易)
CREDIT_KEY = "tg-credit-key-0123456789abcdef"


def credit_row(day, market, amount, **over):
    r = {
        "date": day, "market": market, "margin_balance": 9555729, "margin_balance_change": -33027,
        "margin_amount": amount, "margin_amount_change": -2_300_000_000, "short_balance": 700000,
        "short_balance_change": 1234, "margin_market_value": amount * 1.7175, "priced_lot_share": 0.99999,
        "maintenance_ratio_estimate_pct": 171.75 if market == "twse" else 167.0,
    }  # fmt: skip
    r.update(over)
    return r


def credit_reply(last="2026-10-21", **over_last):
    days = ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", last]
    rows = []
    for i, day in enumerate(days):
        o = over_last if day == last else {}
        rows.append(credit_row(day, "tpex", 80_000_000_000 + i * 10**9, **o))
        rows.append(credit_row(day, "twse", 620_000_000_000 + i * 10**9 + (-532_114_000 if day == last else 0), **o))
    return rows


class CreditTransport:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.calls = status, body, []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers)))
        if "/api/v1/tw/credit/market" in url:
            raw = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()
            return self.status, raw
        return 404, b"not found"


class TestCreditSection(unittest.TestCase):
    SKIP = "信用交易：本期未取得"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keyfile = Path(self.tmp.name) / "thalassa.env"
        self.keyfile.write_text(f"export MARKET_BRIEF_THALASSA_API_KEY={CREDIT_KEY}\n")
        os.chmod(self.keyfile, 0o600)
        self.cfg = cfg()
        self.cfg["thalassa"]["key_file"] = str(self.keyfile)
        self.logs = []

    def fetch(self, transport):
        f = Fetcher(transport=transport, sleep=lambda s: None)
        out = tw_credit.fetch(f, self.cfg, D21, self.logs.append)
        self.assertNotIn(CREDIT_KEY, json.dumps(f.log) + "\n".join(self.logs))
        return out

    def collect_pre(self, transport):
        routes = [("credit/market", lambda url, body: transport("GET", url, {}, body, 1)), *TW_ROUTES_21]
        c = collect.Collector(
            fetcher(routes), self.cfg, WATCH, calendar(), dt.datetime(2026, 10, 22, 8, 30, tzinfo=dt.UTC),
            log=self.logs.append,
        )  # fmt: skip
        return collect.collect("tw-premarket", D22, c)

    def test_fetch_ok_requests_a_range_with_the_key_and_shapes_units(self):
        t = CreditTransport(200, credit_reply())
        data, status, public = self.fetch(t)
        self.assertEqual((status, public), ("ok", None))
        ((method, url, headers),) = t.calls
        self.assertEqual(method, "GET")
        self.assertEqual(url, "http://192.168.31.241:8001/api/v1/tw/credit/market?start=2026-10-07&end=2026-10-21")
        self.assertEqual(headers["X-API-Key"], CREDIT_KEY)
        self.assertEqual(data["date"], "2026-10-21")
        self.assertEqual(
            data["latest"]["twse"],
            {"amount_yi": 6234.7, "amount_chg_yi": -23.0, "balance_chg": -33027, "short_chg": 1234, "ratio": 171.8},
        )
        self.assertEqual(data["latest"]["tpex"]["ratio"], 167.0)
        self.assertEqual(
            [r["date"] for r in data["trend"]],
            ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21"],
        )

    def test_every_failure_gives_one_public_reason_and_a_logged_detail(self):
        stale = credit_reply("2026-10-20")
        one_market = [r for r in credit_reply() if not (r["date"] == "2026-10-21" and r["market"] == "tpex")]
        no_ratio = credit_reply(maintenance_ratio_estimate_pct=None)
        no_field = credit_reply()
        del no_field[-1]["short_balance_change"]
        string_amt = credit_reply(margin_amount="623467886000")
        cases = {
            "no key file": (lambda: self.keyfile.unlink(), CreditTransport(200, credit_reply()), "金鑰未就緒"),
            "http 500": (None, CreditTransport(500, b"boom"), "Thalassa 讀取失敗"),
            "http 401": (None, CreditTransport(401, b'{"detail":"Invalid API key"}'), "Thalassa 讀取失敗"),
            "not deployed (404)": (None, CreditTransport(404, b'{"detail":"Not Found"}'), "端點尚未上線"),
            "not json": (None, CreditTransport(200, b"<html>"), "Thalassa 讀取失敗"),
            "empty": (None, CreditTransport(200, []), "無資料"),
            "not a list": (None, CreditTransport(200, {"rows": []}), "回應格式異常"),
            "stale": (None, CreditTransport(200, stale), "資料尚未更新（最新 2026-10-20）"),
            "one market missing": (None, CreditTransport(200, one_market), "欄位缺漏"),
            "estimate null": (None, CreditTransport(200, no_ratio), "欄位缺漏"),
            "missing field": (None, CreditTransport(200, no_field), "欄位缺漏"),
            "string amount": (None, CreditTransport(200, string_amt), "欄位缺漏"),
            "junk row": (None, CreditTransport(200, ["oops"]), "回應格式異常"),
        }
        for name, (prep, transport, public_want) in cases.items():
            self.setUp()
            if prep:
                prep()
            with self.subTest(name):
                data, status, public = self.fetch(transport)
                self.assertIsNone(data)
                self.assertNotEqual(status, "ok")
                self.assertEqual(public, public_want)
                self.assertTrue(any("section skipped" in m for m in self.logs), self.logs)

    def test_no_fallback_to_environment(self):
        self.keyfile.unlink()
        t = CreditTransport(200, credit_reply())
        with mock.patch.dict(os.environ, {"MARKET_BRIEF_THALASSA_API_KEY": "from-env"}):
            data, _, _ = self.fetch(t)
        self.assertIsNone(data)
        self.assertEqual(t.calls, [])

    def test_golden_render_md_and_telegram(self):
        facts = self.collect_pre(CreditTransport(200, credit_reply()))
        self.assertEqual(facts["status"], "ok")
        self.assertEqual(facts["credit_status"], "ok")
        self.assertFalse([g for g in facts["gaps"] if "信用" in g or "融資" in g])
        self.assertNotIn("credit", json.dumps(facts["data"]))  # the model never sees it
        md, _ = render.render(facts, None, None, self.cfg)
        sec = md.split("## 信用交易")[1].split("\n## ")[0]
        self.assertEqual(
            sec.rstrip("\n"),
            "（2026-10-21，融資融券）\n"
            "\n"
            "<!-- tg: 融資金額(億)=Amt; 金額增減(億)=AmtChg; 融資餘額增減(張)=BalChg; 融券增減(張)=ShChg; 估算維持率=Est; 市場 -->\n"
            "| 市場 | 融資金額(億) | 金額增減(億) | 融資餘額增減(張) | 融券增減(張) | 估算維持率 |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            "| 上市 | 6,234.7 | -23.0 | -33,027 | +1,234 | 171.8% |\n"
            "| 上櫃 | 840.0 | -23.0 | -33,027 | +1,234 | 167.0% |\n"
            "\n"
            "估算維持率＝融資股票以當日收盤計的市值 ÷ 融資金額，是市場平均的估算值，不是交易所公布的數字。\n"
            "\n"
            "<!-- md-only -->券商是對個別帳戶以整戶維持率 130% 追繳，所以這個數字不是斷頭線，看水位與變化即可。\n"
            "\n"
            "近 5 個交易日（融資金額單位：億元）\n"
            "\n"
            "<!-- tg: 上市融資=TWSE; 上市估算=TWEst; 上櫃融資=TPEx; 上櫃估算=TPEst; 日期 -->\n"
            "| 日期 | 上市融資 | 上市估算 | 上櫃融資 | 上櫃估算 |\n"
            "|---|---:|---:|---:|---:|\n"
            "| 10-15 | 6,200.0 | 171.8% | 800.0 | 167.0% |\n"
            "| 10-16 | 6,210.0 | 171.8% | 810.0 | 167.0% |\n"
            "| 10-19 | 6,220.0 | 171.8% | 820.0 | 167.0% |\n"
            "| 10-20 | 6,230.0 | 171.8% | 830.0 | 167.0% |\n"
            "| 10-21 | 6,234.7 | 171.8% | 840.0 | 167.0% |",
        )
        self.assertNotIn("<span", sec)  # no red/green judgement on margin changes
        self.assertLess(md.index("### Watchlist（前一交易日收盤）"), md.index("## 信用交易"))
        self.assertLess(md.index("## 信用交易"), md.index("## 今日除權息"))
        for secret in (CREDIT_KEY, str(self.keyfile), "192.168.31.241"):
            self.assertNotIn(secret, md)
        for pat in NEXUS_SENSITIVE_CI:
            self.assertIsNone(re.search(pat, md, re.I | re.M), pat)
        tg = render.telegram_text(md)
        block = tg.split("**信用交易（2026-10-21，融資融券）**")[1].split("**今日除權息")[0]
        t1, t2 = block.split("```")[1].strip("\n").split("\n"), block.split("```")[3].strip("\n").split("\n")
        self.assertRegex(t1[0], r"^\s*Amt\s+AmtChg\s+BalChg\s+ShChg\s+Est 市場$")
        self.assertRegex(t1[1], r"^6,234\.7\s+-23\.0\s+-33,027\s+\+1,234\s+171\.8% 上市$")
        self.assertEqual(len(t2), 6)
        self.assertLessEqual(max(len(r) for r in t1 + t2), 40)
        self.assertIn("不是交易所公布的數字", block)
        self.assertNotIn("不是斷頭線", tg)  # the md keeps the full caution, the phone the one-line definition

    def test_failure_paths_still_render_the_brief_with_one_line(self):
        cases = {
            "http 500": (CreditTransport(500, b"boom"), "Thalassa 讀取失敗"),
            "not deployed": (CreditTransport(404, b'{"detail":"Not Found"}'), "端點尚未上線"),
            "stale": (CreditTransport(200, credit_reply("2026-10-20")), "資料尚未更新（最新 2026-10-20）"),
            "missing field": (CreditTransport(200, [credit_row("2026-10-21", "twse", 1)]), "欄位缺漏"),
        }
        for name, (t, public) in cases.items():
            self.setUp()
            with self.subTest(name):
                facts = self.collect_pre(t)
                self.assertEqual(facts["status"], "ok")
                self.assertFalse([g for g in facts["gaps"] if "信用" in g or "融資" in g])  # never delays the brief
                self.assertEqual(facts["credit"], {"skipped": public})
                self.assertNotEqual(facts["credit_status"], "ok")
                md, _ = render.render(facts, None, None, self.cfg)
                self.assertIn(f"{self.SKIP}（{public}）", md)
                self.assertNotIn("## 信用交易", md)
                self.assertIn(f"{self.SKIP}（{public}）", render.telegram_text(md))
                self.assertIn("## 今日除權息", md)
                self.assertIn("## 前一交易日台股", md)
                for secret in (CREDIT_KEY, str(self.keyfile), "192.168.31.241", "HTTP 5", "HTTP 4"):
                    self.assertNotIn(secret, md)
                self.assertIn(facts["credit_status"], json.dumps(run.run_record(facts, None, Fetcher())))

    def test_unreadable_key_file_still_sends_the_brief(self):
        real = Path.read_text

        def deny(path, *a, **kw):
            if path == self.keyfile:
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, *a, **kw)

        with mock.patch.object(Path, "read_text", deny):
            facts = self.collect_pre(CreditTransport(200, credit_reply()))
        self.assertEqual(facts["status"], "ok")
        self.assertEqual(facts["credit"], {"skipped": "回應格式異常"})

    def test_no_thalassa_config_shows_nothing(self):
        del self.cfg["thalassa"]
        facts = self.collect_pre(CreditTransport(200, credit_reply()))
        self.assertIsNone(facts["credit"])
        md, _ = render.render(facts, None, None, self.cfg)
        self.assertNotIn("信用交易", md)

    def test_a_single_session_has_no_trend_table(self):
        one = [r for r in credit_reply() if r["date"] == "2026-10-21"]
        facts = self.collect_pre(CreditTransport(200, one))
        md, _ = render.render(facts, None, None, self.cfg)
        self.assertIn("## 信用交易", md)
        self.assertNotIn("近 1 個交易日", md)

    def test_other_slots_do_not_call_the_credit_endpoint(self):
        t = CreditTransport(200, credit_reply())
        routes = [("credit/market", lambda url, body: t("GET", url, {}, body, 1)), *TW_ROUTES_21]
        c = collect.Collector(
            fetcher(routes), self.cfg, WATCH, calendar(), dt.datetime(2026, 10, 22, 8, 30, tzinfo=dt.UTC)
        )
        collect.collect("tw-postmarket", D21, c)
        self.assertEqual(t.calls, [])


class TestColours(unittest.TestCase):
    def _contrast(self, a, b):
        def lum(h):
            c = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

        la, lb = lum(a), lum(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    def test_contrast_on_obsidian_backgrounds(self):
        for fg in (render.UP_COLOR, render.DOWN_COLOR):
            for bg in ("#ffffff", "#1e1e1e"):
                self.assertGreaterEqual(self._contrast(fg, bg), 3.0, (fg, bg))
        self.assertAlmostEqual(self._contrast("#d6453d", "#1e1e1e"), 3.79, places=2)
        self.assertAlmostEqual(self._contrast("#2e9e6a", "#ffffff"), 3.38, places=2)

    def _all_md(self):
        out = []
        for kind, d in (("tw-postmarket", D21), ("us-postmarket", D21), ("tw-premarket", D22)):
            facts = collect.collect(kind, d, collector(TW_ROUTES_21))
            out.append(render.render(facts, None, None, cfg())[0])
        return out

    def test_spans_wrap_every_arrow_and_tables_stay_valid(self):
        for md in self._all_md():
            body = md.split("\n---\n", 1)[1]
            plain = re.sub(
                r'<span style="color:#d6453d">▲[^<|]*</span>|<span style="color:#2e9e6a">▼[^<|]*</span>', "", body
            )
            self.assertNotIn("▲", plain)  # every arrow is inside its colour span
            self.assertNotIn("▼", plain)
            self.assertNotIn("<span", plain)
            self.assertNotIn('#d6453d">▼', body)
            self.assertNotIn('#2e9e6a">▲', body)
            block = []
            for line in [*body.splitlines(), ""]:
                if line.startswith("|"):
                    block.append(line)
                    continue
                if block:  # GFM: header, separator and every row have the same column count
                    counts = {len(b.strip().strip("|").split("|")) for b in block}
                    self.assertEqual(len(counts), 1, block[:3])
                    block = []
            for pat in NEXUS_SENSITIVE_CI:
                self.assertIsNone(re.search(pat, md, re.I | re.M), pat)

    def test_telegram_has_no_spans_and_marks_only_at_line_end(self):
        for md in self._all_md():
            tg = render.telegram_text(md)
            self.assertNotIn("<span", tg)
            self.assertNotIn("</span>", tg)
            for i, block in enumerate(tg.split("```")):
                for ln in block.split("\n"):
                    for mark in ("🔴", "🟢"):
                        if mark in ln:
                            self.assertEqual(i % 2, 1, ln)  # only inside code blocks
                            self.assertTrue(ln.endswith(" " + mark), ln)
                            self.assertEqual(ln.count(mark), 1, ln)
            for block in tg.split("```")[1::2]:
                rows = [r for r in block.strip("\n").split("\n")]
                self.assertFalse(rows[0].endswith(("🔴", "🟢")), rows[0])  # header row


# ================================================================================================ 市場警示
def _weekdays(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


ALERT_DAYS = _weekdays(date(2026, 10, 5), 25)  # D = ALERT_DAYS[-1] = 2026-11-06 (Fri)
ALERT_D = ALERT_DAYS[-1]


def taiex_rows(last=(920.0, 965.0, 910.0)):
    """Flat 1000, then 990 .. 960 over the four sessions before D, D = (close, high, low)."""
    closes = [1000.0] * 20 + [990.0, 980.0, 970.0, 960.0]
    rows = [
        {"trade_date": d, "open": c, "high": c, "low": c, "close": c}
        for d, c in zip(ALERT_DAYS[:-1], closes, strict=True)
    ]
    c, h, low = last
    rows.append({"trade_date": ALERT_D, "open": 960.0, "high": h, "low": low, "close": c})
    return rows


def tx_rows(vol_d, prev=100000, saturday=None):
    rows = []
    for d in ALERT_DAYS[-6:]:
        v = vol_d if d == ALERT_D else prev
        rows += [
            {"trade_date": d, "product": "TX", "contract_month": "202611", "session": "day", "volume": v - 1000},
            {"trade_date": d, "product": "TX", "contract_month": "202612", "session": "day", "volume": 1000},
            {"trade_date": d, "product": "TX", "contract_month": "202611W2", "session": "day", "volume": 77777},
            {"trade_date": d, "product": "TX", "contract_month": "202611", "session": "night", "volume": 55555},
        ]
    if saturday:
        rows.append(
            {"trade_date": saturday, "product": "TX", "contract_month": "202611", "session": "day", "volume": 10}
        )
    return rows


def credit_rows(chg_prev_yi):
    prev = ALERT_DAYS[-2]
    return [
        {"date": prev, "market": "twse", "margin_amount_change": int(chg_prev_yi * 1e8)},
        {"date": prev, "market": "tpex", "margin_amount_change": -1},
        {"date": ALERT_DAYS[-3], "market": "twse", "margin_amount_change": -5_000_000_000},
    ]


def tw_facts(close=920.0, pct=-4.17, up=58, down=1000, foreign=-1100.0, total=-1300.0):
    return {
        "data": {
            "indices": [{"name": "加權指數", "close": close, "chg": None, "pct": pct}],
            "breadth": {"twse_stocks": {"up": up, "down": down}},
            "institutions_yi": {"foreign": foreign, "total": total},
        }
    }


def write_seed(tmp, tw_rows, us_rows):
    seed = Path(tmp) / "seed"
    seed.mkdir()
    lines = ["date,up,down,foreign_yi,total_yi,margin_chg_yi", *tw_rows]
    (seed / "tw.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (seed / "us.csv").write_text("\n".join(["date,spx_pct,ixic_pct,sox_pct", *us_rows]) + "\n", encoding="utf-8")
    return seed


class TestAlert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        key = t / "thalassa.env"
        key.write_text("MARKET_BRIEF_THALASSA_API_KEY=test-key\n")
        key.chmod(0o600)
        self.cfg = cfg()
        self.cfg["thalassa"]["key_file"] = str(key)
        self.state = t / "state"
        # seed: breadth/三大法人 for the sessions before D (one earlier sell-off day), margin for D-3
        tw_seed = [f"{d},500,400,10,20," for d in ALERT_DAYS[:-1]]
        tw_seed[3] = f"{ALERT_DAYS[3]},50,900,-1000,-1200,"
        us_seed = ["2026-11-02,-0.5,-1.0,-4.5", "2026-11-03,0.2,0.3,0.4"]
        self.seed = write_seed(self.tmp.name, tw_seed, us_seed)
        p = mock.patch.object(alert, "SEED_DIR", self.seed)
        p.start()
        self.addCleanup(p.stop)
        self.logs = []

    def routes(self, taiex=None, tx=None, credit=None):
        out = []
        for needle, doc in (("refdata/index-daily", taiex), ("taifex-futures-daily", tx), ("tw/credit/market", credit)):
            if doc is not None:
                out.append((needle, (200, json.dumps(doc).encode())))
        return out

    def evaluate(self, market, facts, routes, d=ALERT_D, persist=True):
        f = fetcher(routes)
        res = alert.evaluate(
            market, d, facts, cfg=self.cfg, fetcher=f, state_dir=self.state, log=self.logs.append, persist=persist
        )
        return res, f

    def test_tw_fired_lines_value_threshold_rarity(self):
        res, f = self.evaluate("tw", tw_facts(), self.routes(taiex_rows(), tx_rows(150000), credit_rows(-250)))
        by = {r.key: r for r in res.readings}
        self.assertEqual(
            sorted(r.key for r in res.fired),
            ["breadth_ratio", "foreign_yi", "margin_chg_yi", "taiex_5d_pct", "taiex_low_pct", "taiex_pct", "total_yi"],
        )
        self.assertAlmostEqual(by["taiex_pct"].value, -4.1667, places=3)
        self.assertAlmostEqual(by["taiex_5d_pct"].value, -8.0, places=6)  # 920 vs 1000 five sessions earlier
        self.assertAlmostEqual(by["taiex_low_pct"].value, (910 / 960 - 1) * 100)
        self.assertEqual((by["taiex_pct"].days, by["taiex_pct"].hits, by["taiex_pct"].first), (24, 1, ALERT_DAYS[1]))
        # breadth: today's facts plus the seed; the seeded sell-off day counts as an earlier hit
        self.assertEqual((by["breadth_ratio"].days, by["breadth_ratio"].hits), (25, 2))
        self.assertEqual((by["foreign_yi"].hits, by["total_yi"].hits), (2, 2))
        # margin posts after the brief: alert D reads the session before D
        self.assertEqual((by["margin_chg_yi"].at, by["margin_chg_yi"].value), (ALERT_DAYS[-2], -250.0))
        text = alert.render(res, self.cfg)
        lines = text.splitlines()
        self.assertEqual(lines[0], f"**市場警示 {ALERT_D}**")
        self.assertEqual(lines[1], f"台股 {ALERT_D}（五）收盤，觸發 7 項條件：")
        self.assertIn("- 加權指數收盤漲跌 -4.17%（門檻 ≤ -3%）：2026 年以來 24 個交易日中第 1 次", text)
        self.assertIn("上市股票下跌家數／上漲家數 17.2 倍（門檻 ≥ 10 倍；1,000 家跌、58 家漲）", text)
        self.assertIn(f"上市融資金額單日增減（{ALERT_DAYS[-2]}） -250.0 億元（門檻 ≤ -200 億元）", text)
        self.assertIn("三大法人合計買賣超 -1,300.0 億元（門檻 ≤ -1,170 億元）", text)
        self.assertEqual(lines[-1], "僅描述市場數據，不含任何買賣建議。")
        self.assertNotIn("投降日", text)  # closed in the lower part of the range
        self.assertNotIn("買進", text)
        self.assertNotIn("賣出", text)
        # the key goes in the header, never in the URL
        self.assertTrue(all("test-key" not in url for _m, url in f.transport.calls))
        hist = json.loads((self.state / "alert-history.json").read_text())
        self.assertEqual(hist["up"][ALERT_D], 58.0)
        self.assertEqual(hist["margin_chg_yi"][ALERT_DAYS[-2]], -250.0)

    def test_capitulation_line_uses_seed007_shape(self):
        rows = taiex_rows(last=(945.0, 960.0, 920.0))  # low -4.17%, close at 62.5% of the range
        res, _ = self.evaluate("tw", tw_facts(close=945.0), self.routes(rows, tx_rows(150000), credit_rows(0)))
        self.assertTrue(res.cap["holds"])
        self.assertAlmostEqual(res.cap["vr"], 1.5)  # weekly and night contracts are not counted
        text = alert.render(res, self.cfg)
        self.assertIn(
            "投降日型態成立（盤中最低較前收跌 4.17%、收在當日區間 62% 處、台指期日盤量為前 5 日均量 1.50 倍", text
        )
        self.assertIn("此型態已檢驗並否決（SEED-007 H1／H2 否決：歷史上此型態之後的報酬沒有比較好）", text)
        # weekend sessions never enter the volume baseline: with the Saturday counted vr would be 1.375
        res, _ = self.evaluate(
            "tw", tw_facts(close=945.0), self.routes(rows, tx_rows(110000, saturday="2026-10-31"), credit_rows(0))
        )
        self.assertAlmostEqual(res.cap["vr"], 1.1)
        self.assertFalse(res.cap["holds"])
        self.assertNotIn("投降日", alert.render(res, self.cfg))
        # no TX volume: the line says it cannot tell
        res, _ = self.evaluate("tw", tw_facts(close=945.0), self.routes(rows, [], credit_rows(0)))
        self.assertIn("台指期量能未取得，無法判斷是否符合 SEED-007 型態", alert.render(res, self.cfg))

    def test_quiet_day_sends_nothing(self):
        flat = taiex_rows(last=(958.0, 962.0, 955.0))
        res, _ = self.evaluate(
            "tw", tw_facts(close=958.0, up=500, down=400, foreign=5, total=8), self.routes(flat, [], credit_rows(-10))
        )
        self.assertEqual(res.fired, [])
        self.assertIsNone(alert.render(res, self.cfg))

    def test_thalassa_down_still_reads_the_brief_values(self):
        res, _ = self.evaluate("tw", tw_facts(), [])  # every Thalassa read fails (404)
        by = {r.key: r for r in res.readings}
        self.assertEqual((by["taiex_pct"].value, by["taiex_pct"].fired), (-4.17, True))  # the brief's own change
        self.assertTrue(by["breadth_ratio"].fired)
        self.assertEqual(by["margin_chg_yi"].note, "前一交易日融資資料未取得")
        text = alert.render(res, self.cfg)
        self.assertIn("- 加權指數收盤漲跌 -4.17%（門檻 ≤ -3%）：歷史資料未取得（Thalassa 讀取失敗）", text)
        self.assertIn("無法判斷：加權指數近 5 個交易日漲跌（歷史資料未取得（Thalassa 讀取失敗））", text)
        self.assertTrue(any("TAIEX history" in p for p in res.problems))
        self.assertNotIn("test-key", alert.summary(res))

    def test_today_close_from_the_brief_when_thalassa_lags(self):
        rows = taiex_rows()[:-1]  # Thalassa has not ingested D yet
        res, _ = self.evaluate("tw", tw_facts(close=920.0), self.routes(rows, [], credit_rows(0)))
        by = {r.key: r for r in res.readings}
        self.assertTrue(by["taiex_pct"].fired)
        self.assertIsNone(by["taiex_low_pct"].value)  # no OHLC from the brief
        self.assertIsNone(res.cap)

    def test_missing_tw_sessions_after_the_seed_are_fetched_once(self):
        # the seed ends three sessions before D: the two sessions in between come from TWSE and are kept
        tw_seed = [f"{d},500,400,10,20," for d in ALERT_DAYS[:-3]]
        (self.seed / "tw.csv").write_text(
            "date,up,down,foreign_yi,total_yi,margin_chg_yi\n" + "\n".join(tw_seed) + "\n"
        )

        def ms(url, body):
            ymd = re.search(r"date=(\d{8})", url).group(1)
            doc = {
                "stat": "OK",
                "date": ymd,
                "tables": [
                    {
                        "title": "漲跌證券數合計",
                        "fields": [],
                        "data": [["上漲(漲停)", "1(0)", "10(0)"], ["下跌(跌停)", "1(0)", "20(0)"]],
                    }
                ],
            }
            return 200, json.dumps(doc).encode()

        def bfi(url, body):
            ymd = re.search(r"dayDate=(\d{8})", url).group(1)
            doc = {
                "stat": "OK",
                "date": ymd,
                "data": [
                    ["外資及陸資(不含外資自營商)", "1", "2", "-30,000,000,000"],
                    ["合計", "1", "2", "-40,000,000,000"],
                ],
            }
            return 200, json.dumps(doc).encode()

        routes = [("type=MS", ms), ("BFI82U", bfi), *self.routes(taiex_rows(), [], credit_rows(0))]
        self.evaluate("tw", tw_facts(), routes)
        hist = json.loads((self.state / "alert-history.json").read_text())
        self.assertEqual(hist["down"][ALERT_DAYS[-2]], 20.0)
        self.assertEqual(hist["foreign_yi"][ALERT_DAYS[-3]], -300.0)
        _res, f = self.evaluate("tw", tw_facts(), routes)
        self.assertFalse([u for _m, u in f.transport.calls if "twse.com.tw" in u])  # filled: not fetched again

    def test_dry_run_writes_no_history(self):
        self.evaluate("tw", tw_facts(), self.routes(taiex_rows(), [], credit_rows(0)), persist=False)
        self.assertFalse((self.state / "alert-history.json").exists())

    def test_us(self):
        facts = {
            "data": {
                "indices": [
                    {"key": "SPX", "label": "S&P 500", "pct": -1.2},
                    {"key": "IXIC", "label": "那斯達克", "pct": -3.1},
                    {"key": "SOX", "label": "費城半導體", "missing": True},
                ],
                "adr": {"adr_pct": -5.12},
            }
        }
        res, _ = self.evaluate("us", facts, [], d="2026-11-04")
        text = alert.render(res, self.cfg)
        self.assertEqual(
            text.splitlines()[:2], ["**市場警示 2026-11-04**", "美股 2026-11-04（三）收盤，觸發 1 項條件："]
        )
        self.assertIn("- 那斯達克收盤漲跌 -3.10%（門檻 ≤ -3%）：2026 年以來 3 個交易日中第 1 次", text)
        self.assertIn("未觸發：S&P 500 收盤漲跌 -1.20%（門檻 ≤ -2.5%）", text)
        self.assertIn("無法判斷：費城半導體收盤漲跌（當日資料未取得）", text)
        self.assertIn("台積電 ADR 收盤 -5.12%（不設門檻）。", text)

    def test_rarity_text(self):
        r = alert.Reading(
            key="taiex_pct",
            label="x",
            op="<=",
            threshold=-3,
            unit="%",
            at="2026-11-03",
            value=-3.59,
            fired=True,
            days=6358,
            hits=67,
            first="2001-01-02",
            streak=1,
        )
        self.assertEqual(alert.rarity_text(r), "2001 年以來 6,358 個交易日中第 67 次，約每年 2.6 次")
        self.assertIn("；連續第 3 個交易日", alert.rarity_text(dataclasses.replace(r, streak=3)))
        self.assertEqual(alert.fmt_value(dataclasses.replace(r, unit="倍", value=float("inf"))), "無上漲家數")

    def test_breadth_parser_on_a_real_ms_body(self):
        b = tw.parse_breadth(fx("mi_index_ms_20261103.json"), date(2026, 11, 3))
        self.assertEqual((b["up"], b["down"], b["down_limit"]), (58, 1000, 33))
        with self.assertRaises(tw.NotReady):
            tw.parse_breadth(fx("mi_index_ms_20261103.json"), date(2026, 11, 4))

    def test_seed_parsers(self):
        doc = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1793716200, 1793802600, 1793889000],
                        "indicators": {"quote": [{"close": [100.0, None, 97.0]}]},
                    }
                ]
            }
        }
        # 2026-11-03, -04 (null close: skipped, the next change is against the last close), -05 14:30Z = New York dates
        self.assertEqual(alert_seed.parse_yahoo(doc, "2001-01-02"), {"2026-11-05": -3.0})


class TestAlertRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.outbox, self.state_dir = self.t / "outbox", self.t / "state"
        self.outbox.mkdir()
        self.logs = []

    def fired(self, market="tw", d="2026-11-03"):
        r = alert.Reading(
            key="taiex_pct",
            label="加權指數收盤漲跌",
            op="<=",
            threshold=-3,
            unit="%",
            at=d,
            value=-3.59,
            fired=True,
            days=10,
            hits=2,
            first="2026-10-01",
            streak=1,
        )
        return alert.Result(market, d, readings=[r])

    def run_alerts(self, built, state, evaluate):
        with mock.patch.object(run.alert, "evaluate", evaluate):
            return run.run_alerts(
                built,
                state,
                cfg=cfg(),
                fetcher=None,
                state_dir=self.state_dir,
                outbox=self.outbox,
                now=dt.datetime(2026, 11, 3, 9, 50, tzinfo=dt.UTC),
                log=self.logs.append,
            )

    def test_once_per_market_and_session(self):
        state, calls = {}, []

        def evaluate(market, d, facts, **kw):
            calls.append((market, d))
            return self.fired(market, d)

        built = [("tw-postmarket", date(2026, 11, 3), {}), ("tw-premarket", date(2026, 11, 3), {})]
        self.assertTrue(self.run_alerts(built, state, evaluate))
        self.assertEqual(calls, [("tw", "2026-11-03")])  # only the post-market brief triggers
        self.assertEqual(state["alert:tw:2026-11-03"]["status"], "queued")
        entry = json.loads((self.outbox / "2026-11-03-alert-tw.json").read_text())
        self.assertEqual((entry["verb"], entry["market"], entry["date"]), ("brief-alert", "tw", "2026-11-03"))
        self.assertTrue(entry["text"].startswith("**市場警示 2026-11-03**\n"))
        self.assertTrue(deliver.valid(entry))
        (self.outbox / "2026-11-03-alert-tw.json").unlink()  # delivered
        self.assertFalse(self.run_alerts(built, state, evaluate))  # a rerun of the tick: nothing
        self.assertEqual(len(calls), 1)
        self.assertFalse(list(self.outbox.glob("*.json")))

    def test_error_is_recorded_not_raised(self):
        state = {}

        def boom(*a, **kw):
            raise RuntimeError("bug")

        self.assertTrue(self.run_alerts([("us-postmarket", date(2026, 11, 4), {})], state, boom))
        self.assertEqual(state["alert:us:2026-11-04"]["status"], "error")
        self.assertFalse(list(self.outbox.glob("*.json")))
        self.assertTrue(any("internal error RuntimeError" in m for m in self.logs))

    def test_quiet(self):
        state = {}
        quiet = alert.Result("tw", "2026-11-03", readings=[])
        self.run_alerts([("tw-postmarket", date(2026, 11, 3), {})], state, lambda *a, **kw: quiet)
        self.assertEqual(state["alert:tw:2026-11-03"]["status"], "quiet")
        self.assertFalse(list(self.outbox.glob("*.json")))

    def test_due_tick_queues_brief_then_alert_after_delivery(self):
        c = cfg()
        c["state_dir"], c["outbox_dir"], c["watchlist_file"] = str(self.state_dir), str(self.outbox), f"{self.t}/x.json"
        c["bridge"] = "/bin/false"
        cpath = self.t / "config.json"
        cpath.write_text(json.dumps(c, ensure_ascii=False))
        order = []

        def deliver_outbox(ob, br, log=None):
            order.append(sorted(p.name for p in Path(ob).glob("*.json")))
            return 0

        def evaluate(market, d, facts, **kw):
            order.append(("evaluate", market, d, facts["kind"]))
            return self.fired(market, d)

        with (
            mock.patch.object(run, "Fetcher", lambda: fetcher(TW_ROUTES_21)),
            mock.patch.object(run.deliver, "deliver_outbox", deliver_outbox),
            mock.patch.object(run.alert, "evaluate", evaluate),
        ):
            self.assertEqual(
                run.main(["--config", str(cpath), "due", "--now", "2026-10-21T18:50:00+08:00", "--no-llm"]), 0
            )
            run.main(["--config", str(cpath), "due", "--now", "2026-10-21T18:55:00+08:00", "--no-llm"])
        # deliver (nothing), build + queue the brief, deliver it, then the alert, then deliver the alert
        self.assertEqual(order[1], ["2026-10-21-tw-postmarket.json"])
        self.assertEqual(order[2], ("evaluate", "tw", "2026-10-21", "tw-postmarket"))
        self.assertIn("2026-10-21-alert-tw.json", order[3])
        self.assertEqual(sum(1 for x in order if isinstance(x, tuple)), 1)  # the next tick does not re-evaluate
        st = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(st["alert:tw:2026-10-21"]["status"], "queued")

    def test_alert_command_dry_run(self):
        c = cfg()
        c["state_dir"], c["outbox_dir"] = str(self.state_dir), str(self.outbox)
        c["bridge"] = "/bin/false"
        cpath = self.t / "config.json"
        cpath.write_text(json.dumps(c, ensure_ascii=False))
        facts = {"kind": "tw-postmarket", "status": "ok", "must_missing": [], "data": {}}
        out = io.StringIO()
        with (
            mock.patch.object(run, "Fetcher", lambda: fetcher(TW_ROUTES_21)),
            mock.patch.object(run, "build", lambda *a, **kw: (facts, None, None, None)),
            mock.patch.object(run.alert, "evaluate", lambda *a, **kw: self.fired("tw", "2026-11-03")),
            mock.patch.object(sys, "stdout", out),
        ):
            rc = run.main(
                [
                    "--config",
                    str(cpath),
                    "alert",
                    "tw",
                    "--date",
                    "2026-11-03",
                    "--dry-run",
                    "--out",
                    str(self.t / "dry"),
                ]
            )
        self.assertEqual(rc, 0)
        self.assertTrue(out.getvalue().startswith("**市場警示 2026-11-03**"))
        self.assertEqual((self.t / "dry" / "2026-11-03-alert-tw.txt").read_text(), out.getvalue().rstrip("\n"))
        self.assertFalse(list(self.outbox.glob("*.json")))
        self.assertNotIn("alert:tw:2026-11-03", run.load_state(self.state_dir))


class TestDeliverAlert(unittest.TestCase):
    def test_brief_alert_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            ob = Path(tmp)
            text = "**市場警示 2026-11-03**\n台股 ..."
            (ob / "2026-11-03-alert-tw.json").write_text(json.dumps(alert.outbox_entry("tw", "2026-11-03", text)))
            bad = alert.outbox_entry("tw", "2026-11-03", "**市場警示 2026-11-04**")  # title for another day
            (ob / "2026-11-03-alert-us.json").write_text(json.dumps({**bad, "market": "us"}))
            seen = []

            def runner(cmd, input, **kw):
                seen.append((cmd, input))
                return mock.Mock(returncode=0, stdout="{}", stderr="")

            self.assertEqual(deliver.deliver_outbox(ob, "ssh research-bridge", log=lambda m: None, runner=runner), 1)
            self.assertEqual(seen, [(["ssh", "research-bridge", "brief-alert", "tw", "2026-11-03"], text)])
            self.assertTrue((ob / "sent" / "2026-11-03-alert-tw.json").is_file())
            self.assertTrue((ob / "2026-11-03-alert-us.json").is_file())  # malformed: never sent
            self.assertFalse(deliver.valid({**alert.outbox_entry("jp", "2026-11-03", text)}))


class TestAlertReading(unittest.TestCase):
    def test_counts_and_streak_through_the_value_day_only(self):
        cond = {"key": "taiex_pct", "label": "x", "op": "<=", "threshold": -3}
        s = {
            "2000-12-29": -9.0,
            "2001-01-02": -5.0,
            "2001-01-03": -1.0,
            "2001-01-04": -4.0,
            "2001-01-05": -6.0,
            "2001-01-08": -7.0,
        }
        r = alert.reading(cond, s, "2001-01-05", "2001-01-01")
        self.assertEqual((r.value, r.fired, r.days, r.hits, r.first, r.streak), (-6.0, True, 4, 3, "2001-01-02", 2))
        self.assertIsNone(alert.reading(cond, s, "2001-01-09", "2001-01-01").value)
        self.assertEqual(
            alert.breadth_ratio({"a": 0, "b": 0, "c": 4}, {"a": 5, "b": 0, "c": 40}),
            {"a": float("inf"), "b": 0.0, "c": 10.0},
        )


if __name__ == "__main__":
    unittest.main()
