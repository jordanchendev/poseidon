"""Offline tests for scripts/market_brief (stdlib unittest; pytest also collects them).

Run on stormtrooper only:  cd <dir> && python3 -m unittest -v test_market_brief
Fixtures in fixtures/ are trimmed copies of real responses captured from stormtrooper on 2026-10-22.
"""

from __future__ import annotations

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

import collect  # noqa: E402
import deliver  # noqa: E402
import llm  # noqa: E402
import news  # noqa: E402
import render  # noqa: E402
import run  # noqa: E402
import tradingcal  # noqa: E402
import tw  # noqa: E402
import us  # noqa: E402
from netio import Fetcher, FetchError  # noqa: E402

FIX = HERE / "fixtures"
TPE = ZoneInfo("Asia/Taipei")
D21 = dt.date(2026, 10, 21)
D22 = dt.date(2026, 10, 22)


def fx(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def cfg():
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


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
        self.assertEqual(tw._signed_arrow("<span>▼1.25%</span>"), -1.25)
        self.assertIn(b"queryDate=2026%2F10%2F22", tw.taifex_night_form(D22))

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
        self.assertIn(
            "日經 225（2026-10-22 報價）",
            render._chg_line("日經 225", c.us_rows(insts, got, expect_date="2026-10-21")[1]),
        )
        self.assertEqual(render._chg_line("日經 225", rows[1]), "- 日經 225（2026-10-22 報價）：63,139.59，—（—）")

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
        self.assertEqual(self.due(tpe(2026, 10, 22, 17, 59)), {("tw-postmarket", "2026-10-22"): True})
        self.assertEqual(self.due(tpe(2026, 10, 22, 18, 0)), {})  # window is [slot, slot + 90 min)
        self.assertEqual(self.due(tpe(2026, 10, 26, 16, 35)), {("tw-postmarket", "2026-10-26"): False})  # holiday

    def test_us_slots_follow_new_york_dst(self):
        # EDT (UTC-4): pre 08:45 ET = 20:45 TPE, post 17:15 ET = 05:15 TPE next day
        self.assertEqual(self.due(tpe(2026, 10, 30, 20, 50)), {("us-premarket", "2026-10-30"): True})
        self.assertEqual(self.due(tpe(2026, 10, 31, 5, 20)), {("us-postmarket", "2026-10-30"): True})  # Sat in TPE
        # DST ends 2026-11-01: EST (UTC-5): pre = 21:45 TPE, post = 06:15 TPE
        self.assertEqual(self.due(tpe(2026, 11, 2, 20, 50)), {})
        self.assertEqual(self.due(tpe(2026, 11, 2, 21, 50)), {("us-premarket", "2026-11-02"): True})
        self.assertEqual(self.due(tpe(2026, 11, 3, 5, 20)), {})
        self.assertEqual(self.due(tpe(2026, 11, 3, 6, 20)), {("us-postmarket", "2026-11-02"): True})
        # DST starts 2027-03-14: Fri 03-12 still EST (21:45 TPE), Mon 03-15 EDT (20:45 TPE)
        self.assertEqual(self.due(tpe(2027, 3, 12, 21, 46)), {("us-premarket", "2027-03-12"): True})
        self.assertEqual(self.due(tpe(2027, 3, 15, 20, 46)), {("us-premarket", "2027-03-15"): True})
        # NYSE holiday -> returned as closed
        self.assertEqual(self.due(tpe(2026, 11, 26, 21, 50)), {("us-premarket", "2026-11-26"): False})

    def test_state_makes_slots_idempotent(self):
        now = tpe(2026, 10, 22, 16, 40)
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
            "- 加權指數：46,887.77，+594.84（+1.28%）",
            "- 外資及陸資（不含外資自營商）：+22.39",
            "| 2330 | 台積電 | 2,420 | +15 | +0.62% |",
        ):
            self.assertIn(needle, md)
        self.assertLessEqual(len(summary), 580)
        self.assertIn("46,887.77", summary)
        for pat in NEXUS_SENSITIVE_CI:
            self.assertIsNone(re.search(pat, md, re.I | re.M), pat)

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
        self.assertIn("- VIX：16.68，—（—）", md)
        self.assertNotIn("，0.00（0.00%）", md)
        self.assertIn("- 半導體（SMH）：", md)
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
        self.assertIn("0056 元大高股息（息）現金股利 1.72 元", md)


# ================================================================================================ delivery
class TestDeliver(unittest.TestCase):
    def entry(self, kind="tw-postmarket", date="2026-10-21", **kw):
        e = {"kind": kind, "date": date, "filename": f"{date}-{kind}.md", "summary": "s", "report_md": "---\nx\n---\n"}
        e.update(kw)
        return e

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
        mod.SISYPHUS_OUT = str(Path(tmp) / "Sisyphus" / "output")
        mod.LOG = str(Path(tmp) / "bridge.log")
        os.makedirs(mod.SISYPHUS_OUT)
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
            self.assertEqual((Path(mod.SISYPHUS_OUT) / "2026-10-21-tw-postmarket.md").read_text(encoding="utf-8"), md)
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
            cpath = Path(tmp) / "config.json"
            cpath.write_text(json.dumps(c, ensure_ascii=False))
            routes = [r for r in TW_ROUTES_21 if not r[0].startswith("BFI82U?type=day&dayDate=20261021")]
            delivered = []
            with (
                mock.patch.object(run, "Fetcher", lambda: fetcher(routes)),
                mock.patch.object(run.deliver, "deliver_outbox", lambda ob, br, log=None: delivered.append(ob) or 0),
            ):
                rc = run.main(["--config", str(cpath), "due", "--now", "2026-10-21T16:35:00+08:00", "--no-llm"])
                self.assertEqual(rc, 0)
                st = json.loads((Path(tmp) / "state" / "state.json").read_text())
                self.assertEqual(st["tw-postmarket:2026-10-21"]["status"], "waiting")
                self.assertFalse(list((Path(tmp) / "outbox").glob("*.json")))
                rc = run.main(["--config", str(cpath), "due", "--now", "2026-10-21T17:35:00+08:00", "--no-llm"])
                st = json.loads((Path(tmp) / "state" / "state.json").read_text())
                self.assertEqual(st["tw-postmarket:2026-10-21"]["status"], "queued")
                entry = json.loads((Path(tmp) / "outbox" / "2026-10-21-tw-postmarket.json").read_text())
                self.assertEqual(entry["filename"], "2026-10-21-tw-postmarket.md")
                self.assertIn("資料未齊", entry["report_md"])
                n = len(list((Path(tmp) / "outbox").glob("*.json")))
                run.main(["--config", str(cpath), "due", "--now", "2026-10-21T17:40:00+08:00", "--no-llm"])
                self.assertEqual(len(list((Path(tmp) / "outbox").glob("*.json"))), n)  # idempotent
                # holiday: recorded once as closed, no fetch of report data, and no failure notice
                run.main(["--config", str(cpath), "due", "--now", "2026-10-26T16:35:00+08:00", "--no-llm"])
                st = json.loads((Path(tmp) / "state" / "state.json").read_text())
                self.assertEqual(st["tw-postmarket:2026-10-26"]["status"], "closed")
                self.assertFalse(list((Path(tmp) / "outbox").glob("*.failed.json")))

    def _cfg(self, tmp):
        c = cfg()
        c["state_dir"], c["outbox_dir"], c["watchlist_file"] = f"{tmp}/state", f"{tmp}/outbox", f"{tmp}/none.json"
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

            self.assertEqual(self._due(cpath, "2026-10-21T16:35:00+08:00", build=boom), 1)
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
            self._due(cpath, "2026-10-21T16:40:00+08:00", build=boom)
            self.assertEqual(len(list(ob.glob("*.failed.json"))), n)

    def test_exchange_closed_no_notice(self):  # typhoon-style closure on a calendar trading day
        with tempfile.TemporaryDirectory() as tmp:
            cpath = self._cfg(tmp)
            routes = [("MI_INDEX", "mi_index_closed_20260710.json"), *TW_ROUTES_21]
            self._due(cpath, "2026-10-21T16:35:00+08:00", routes=routes)
            self.assertEqual(self._state(tmp)["tw-postmarket:2026-10-21"]["status"], "closed")
            self.assertFalse(list((Path(tmp) / "outbox").glob("*.json")))

    def test_missing_key_exits_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = cfg()
            c["state_dir"], c["outbox_dir"] = f"{tmp}/state", f"{tmp}/outbox"
            c["llm"]["key_file"] = f"{tmp}/absent.env"
            cpath = Path(tmp) / "config.json"
            cpath.write_text(json.dumps(c, ensure_ascii=False))
            with (
                mock.patch.object(run, "Fetcher", lambda: fetcher(TW_ROUTES_21)),
                mock.patch.object(run.deliver, "deliver_outbox", lambda ob, br, log=None: 0),
            ):
                rc = run.main(["--config", str(cpath), "due", "--now", "2026-10-21T16:35:00+08:00"])
            self.assertEqual(rc, 3)
            ob = Path(tmp) / "outbox"
            self.assertEqual([p.name for p in ob.glob("*.json")], ["2026-10-21-tw-postmarket.failed.json"])
            self.assertIn(
                "RESEARCH_DRAFT_OPENROUTER_KEY",
                json.loads((ob / "2026-10-21-tw-postmarket.failed.json").read_text())["reason"],
            )


if __name__ == "__main__":
    unittest.main()
