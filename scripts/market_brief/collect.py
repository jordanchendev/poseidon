"""Deterministic data layer: fetch + shape the facts for one report kind and session date.

Every number in a report comes from the dict returned here (`facts["data"]`), rounded once at collection so the renderer,
the model payload and the number check all see the same values. Missing optional datasets are listed in
`facts["gaps"]`; a missing required dataset sets status "must_missing"; an exchange "no data" reply for the session
date sets status "closed".
"""

from __future__ import annotations

import contextlib
import datetime as dt
import re
from zoneinfo import ZoneInfo

import news
import tw
import tw_credit
import tw_groups
import us
from netio import FetchError
from numfmt import weekday_zh

TPE = ZoneInfo("Asia/Taipei")
ET = ZoneInfo("America/New_York")

TITLES = {
    "tw-premarket": "台股盤前快報",
    "tw-postmarket": "台股盤後日報",
    "us-premarket": "美股盤前快報",
    "us-postmarket": "美股盤後日報",
}
TW_INDEX_PICK = [
    ("發行量加權股價指數", "加權指數"),
    ("寶島股價指數", "寶島指數"),
    ("臺灣50指數", "臺灣50"),
    ("臺灣中型100指數", "臺灣中型100"),
    ("未含金融指數", "未含金融"),
    ("未含電子指數", "未含電子"),
]
SECTOR_AGGREGATES = {"水泥窯製類指數", "塑膠化工類指數", "機電類指數", "化學生技醫療類指數"}
COMMON_STOCK = re.compile(r"^[1-9]\d{3}$")

# Quote keys whose absence (or missing change) makes a brief wrong rather than thinner: the run waits for them
# (retry_gaps, up to complete_wait_minutes). Every other missing value is optional: it is listed in `gaps`, renders
# 「未取得」 in its row and never delays delivery. The 一句話 and the model's interpretation are built from these.
CRITICAL_QUOTES = {
    "tw-premarket": frozenset({"SPX", "IXIC", "DJI", "SOX", "TSM", "USDTWD"}),  # US closes, TSMC ADR and its FX leg
    "us-postmarket": frozenset({"SPX", "IXIC", "DJI", "SOX"}),  # the US closing prints
    "us-premarket": frozenset({"ES", "NQ"}),  # index futures
    "tw-postmarket": frozenset(),
}


def r2(x, nd=2):
    return None if x is None else round(float(x), nd)


def yi2(x):
    return None if x is None else round(x / 1e8, 2)


class Collector:
    def __init__(self, fetcher, cfg, watchlist, calendar, now_utc, log=print):
        self.f = fetcher
        self.cfg = cfg
        self.watchlist = watchlist
        self.cal = calendar
        self.now = now_utc
        self.log = log
        self.sources = []
        self.gaps = []
        self.retry_gaps = []  # gaps worth waiting for (data may still be published within the window)
        self.must_missing = []
        self.critical = frozenset()  # quote keys worth waiting for, set per kind by collect()
        self._mi_cache = {}
        self.bfi_detail = {}
        self._tpex_q_cache = {}

    # -------------------------------------------------------------------------------------------- helpers
    def src(self, name, url):
        if not any(s["url"] == url for s in self.sources):
            self.sources.append({"name": name, "url": url})

    def gap(self, msg, retry=True):  # retry=False: optional data, listed but never waited for
        if msg not in self.gaps:
            self.gaps.append(msg)
            if retry:
                self.retry_gaps.append(msg)

    def mi_index(self, d):
        """Parsed MI_INDEX for d (cached). Raises tw.MarketClosed / tw.NotReady / FetchError."""
        if d not in self._mi_cache:
            url = tw.mi_index_url(d)
            try:
                self._mi_cache[d] = tw.parse_mi_index(self.f.get_json(url), d)
            except (tw.MarketClosed, tw.NotReady, FetchError) as e:
                self._mi_cache[d] = e
            if not isinstance(self._mi_cache[d], Exception):
                self.src(f"TWSE MI_INDEX {d.isoformat()}", url)
        v = self._mi_cache[d]
        if isinstance(v, Exception):
            raise v
        return v

    def tpex_quotes(self, d):
        if d not in self._tpex_q_cache:
            url = tw.tpex_quotes_url(d)
            try:
                self._tpex_q_cache[d] = tw.parse_tpex_quotes(self.f.get_json(url), d)
                self.src(f"TPEx 上櫃收盤行情 {d.isoformat()}", url)
            except (tw.NotReady, FetchError) as e:
                self._tpex_q_cache[d] = e
        v = self._tpex_q_cache[d]
        if isinstance(v, Exception):
            raise v
        return v

    def latest_tw_session(self, on_or_before):
        """(date, MI_INDEX) of the latest TW session on or before the given date, walking back past closures."""
        d = on_or_before
        if not self.cal.tw_open(d):
            d = self.cal.prev_open("tw", d)
        for _ in range(8):
            if d is None:
                return None, None
            try:
                return d, self.mi_index(d)
            except tw.MarketClosed:
                d = self.cal.prev_open("tw", d)
            except (tw.NotReady, FetchError) as e:
                self.log(f"MI_INDEX {d} unavailable: {e}")
                return None, None
        return None, None

    def bfi(self, d):
        url = tw.bfi82u_url(d)
        try:
            out = tw.parse_bfi82u(self.f.get_json(url), d)
        except (tw.MarketClosed, tw.NotReady, FetchError) as e:
            self.log(f"BFI82U {d} unavailable: {e}")
            return None
        self.src(f"TWSE BFI82U {d.isoformat()}", url)
        nets = {k: yi2(v["net"]) for k, v in out.items()}
        if out.get("dealer_self") and out.get("dealer_hedge"):
            nets["dealer_total"] = yi2(out["dealer_self"]["net"] + out["dealer_hedge"]["net"])
        self.bfi_detail[d] = {
            k: {"buy": yi2(v["buy"]), "sell": yi2(v["sell"]), "net": yi2(v["net"])} for k, v in out.items()
        }
        return nets

    def quotes(self, instruments):
        got, missing = us.fetch_quotes(self.f, instruments, log=self.log)
        if any(q["source"] == "CNBC" for q in got.values()):
            self.src("CNBC quote service", us.CNBC_URL)
        if any(q["source"] == "Yahoo" for q in got.values()):
            self.src("Yahoo Finance chart API (fallback)", us.YAHOO_URL)
        return got, missing

    def tw_watch_rows(self, d, mi):
        stocks = {s["code"]: s for s in mi["stocks"]}
        rows, otc_needed = [], []
        for w in self.watchlist.get("tw", []):
            s = stocks.get(w["code"])
            if s is None:
                otc_needed.append(w["code"])
            rows.append((w, s))
        otc = {}
        if otc_needed:
            try:
                otc = {s["code"]: s for s in self.tpex_quotes(d)}
            except (tw.NotReady, FetchError) as e:
                self.log(f"TPEx quotes {d} unavailable: {e}")
        out = []
        for w, s in rows:
            s = s or otc.get(w["code"])
            if s is None:
                out.append({"code": w["code"], "name": None, "note": w.get("note"), "missing": True})
                continue
            out.append(
                {
                    "code": s["code"],
                    "name": s["name"],
                    "note": w.get("note"),
                    "close": r2(s["close"]),
                    "chg": r2(s["chg"]),
                    "pct": r2(s["pct"]),
                    "turnover_yi": yi2(s["turnover"]),
                }
            )
        return out

    def tw_summary(self, d, mi):
        """Compact TW session summary used by tw-premarket and us-premarket."""
        idx = mi["indices"].get("發行量加權股價指數") or {}
        total = (mi["stats"].get("總計(1~15)") or {}).get("turnover")
        b = mi["breadth"].get("stocks") or {}
        out = {
            "date": d.isoformat(),
            "taiex": {"close": r2(idx.get("close")), "chg": r2(idx.get("chg")), "pct": r2(idx.get("pct"))},
            "turnover_total_yi": yi2(total),
            "breadth_stocks": {"up": b.get("up"), "down": b.get("down"), "flat": b.get("flat")},
            "institutions_yi": self.bfi(d),
            "watchlist": self.tw_watch_rows(d, mi),
        }
        if out["institutions_yi"] is None:
            self.gap(f"{d.isoformat()} 三大法人（BFI82U）未取得", retry=False)
        return out

    def us_rows(self, insts, got, *, with_ext=False, expect_date=None, strict=False):
        """Quote rows. A missing change stays None (renders "—") and is listed in gaps. expect_date: session date the
        section is about (only for instruments whose CNBC date is reliable, i.e. not stocks); strict drops the change
        of a row from another session, otherwise the row is annotated with its own session date."""
        rows = []
        for i in insts:
            q = got.get(i["key"])
            row = {"key": i["key"], "label": i.get("label", i["key"])}
            if q is None:
                row["missing"] = True
            else:
                nd = 3 if i.get("kind") == "yield" else i.get("nd", 2)
                chg, pct = q["chg"], q["pct"]
                note = None
                if expect_date and q["session_date"] and q["session_date"] != expect_date:
                    note = f"{q['session_date']} 報價"
                    if strict:
                        chg = pct = None
                        self.gap(
                            f"{row['label']} 報價屬 {q['session_date']} 交易時段，非 {expect_date}，漲跌略過",
                            retry=i["key"] in self.critical,
                        )
                if chg is None and q.get("chg_missing"):
                    self.gap(f"{row['label']} 漲跌未取得（資料源回報不變或缺值）", retry=i["key"] in self.critical)
                row.update(
                    last=r2(q["last"], nd),
                    chg=r2(chg, nd),
                    pct=None if i.get("kind") == "yield" else r2(pct),
                    session_date=q["session_date"],
                    source=q["source"],
                    nd=nd,
                )
                if note:
                    row["note"] = note
                if i.get("kind") == "yield":
                    row["kind"] = "yield"
                    row["chg_bp"] = None if q["chg"] is None else round(q["chg"] * 100, 1)
                if with_ext:
                    row["ext"] = (
                        None
                        if not q.get("ext")
                        else {
                            "type": q["ext"]["type"],
                            "last": r2(q["ext"]["last"]),
                            "chg": r2(q["ext"]["chg"]),
                            "pct": r2(q["ext"]["pct"]),
                            "session_date": q["ext"]["session_date"],
                        }
                    )
            rows.append(row)
        return rows

    def gap_missing_quotes(self, prefix, missing):
        """One gap per class so only the critical keys hold the run back (the optional ones are just reported)."""
        for retry in (True, False):
            keys = [k for k in missing if (k in self.critical) == retry]
            if keys:
                self.gap(prefix + "、".join(keys), retry=retry)

    def adr(self, got, tw_close, tw_date):
        a = self.cfg["adr"]
        tsm, fx = got.get(a["us"]), got.get(a["fx"])
        if not tsm or not fx or tw_close is None:
            return None
        equiv = tsm["last"] * fx["last"] / a["shares_per_adr"]
        return {
            "adr": a["us"],
            "adr_close": r2(tsm["last"]),
            "adr_pct": r2(tsm["pct"]),
            "usdtwd": r2(fx["last"], 3),
            "shares_per_adr": a["shares_per_adr"],
            "equiv_twd": r2(equiv),
            "tw_code": a["tw"],
            "tw_close": r2(tw_close),
            "tw_date": tw_date,
            "premium_pct": r2((equiv / tw_close - 1) * 100),
        }

    def headlines(self, market):
        try:
            items, status = news.fetch_news(self.f, self.cfg["news"], market, self.now, log=self.log)
        except Exception as e:  # news is optional; never fail a report on it
            self.log(f"news failed: {type(e).__name__}: {e}")
            items, status = [], []
        for s in status:
            if s["ok"]:
                feed = next(f for f in self.cfg["news"]["feeds"] if f["id"] == s["id"])
                self.src(f"新聞：{feed['name']}", feed["url"])
        if not items:
            self.gap("新聞標題未取得", retry=False)
        return items

    # -------------------------------------------------------------------------------------------- kinds
    def tw_postmarket(self, d):
        tcfg = self.cfg["tw"]
        try:
            mi = self.mi_index(d)
        except tw.MarketClosed:
            return {"status": "closed"}
        except (tw.NotReady, FetchError) as e:
            self.must_missing.append(f"TWSE MI_INDEX {d}: {e}")
            return {"status": "must_missing"}
        idx = [
            {"name": label, **{k: r2(v) for k, v in mi["indices"][name].items()}}
            for name, label in TW_INDEX_PICK
            if name in mi["indices"]
        ]
        otc = None
        url = tw.tpex_highlight_url(d)
        try:
            otc = tw.parse_tpex_highlight(self.f.get_json(url), d)
            self.src(f"TPEx 上櫃股票當日彙總 {d.isoformat()}", url)
        except (tw.NotReady, FetchError) as e:
            self.log(f"TPEx highlight {d} unavailable: {e}; falling back to index + quotes")
            self.gap(
                "TPEx 上櫃當日彙總尚未公布（櫃買指數取自指數表，上櫃漲跌家數依收盤行情普通股計算）", retry=False
            )  # the fallback below computes what the report needs
        stats = mi["stats"]
        turnover = {
            "stocks_yi": yi2((stats.get("1.一般股票") or {}).get("turnover")),
            "total_yi": yi2((stats.get("總計(1~15)") or {}).get("turnover")),
            "otc_yi": yi2(otc["turnover"]) if otc else None,
        }
        fm_url = tw.fmtqik_url(d)
        try:
            rows = tw.parse_fmtqik(self.f.get_json(fm_url))
            prior = [r for r in rows if r["date"] < d.isoformat()]
            if len(prior) < 5:
                prev_month = d.replace(day=1) - dt.timedelta(days=1)
                rows2 = tw.parse_fmtqik(self.f.get_json(tw.fmtqik_url(prev_month)))
                prior = sorted(rows2 + prior, key=lambda r: r["date"])
            prior = prior[-5:]
            if len(prior) == 5 and turnover["total_yi"]:
                avg = sum(r["turnover"] for r in prior) / 5
                turnover["avg5_total_yi"] = yi2(avg)
                turnover["vs_avg5_pct"] = r2((turnover["total_yi"] * 1e8 / avg - 1) * 100)
            self.src(f"TWSE FMTQIK {d:%Y-%m}", fm_url)
        except (tw.NotReady, FetchError) as e:
            self.log(f"FMTQIK unavailable: {e}")
        b = mi["breadth"]
        breadth = {"twse_stocks": b.get("stocks"), "twse_all": b.get("all")}
        otc_index = otc
        if otc:
            breadth["tpex"] = {k: otc[k] for k in ("up", "up_limit", "down", "down_limit", "flat")}
        else:
            try:
                otc_index = tw.parse_tpex_index(self.f.get_json(tw.tpex_index_url(d)), d)
                self.src(f"TPEx 櫃買指數 {d:%Y-%m}", tw.tpex_index_url(d))
            except (tw.NotReady, FetchError) as e:
                self.log(f"TPEx index {d} unavailable: {e}")
            with contextlib.suppress(tw.NotReady, FetchError):
                breadth["tpex_computed"] = tw.breadth_from_quotes(self.tpex_quotes(d))
        if otc_index:
            idx.append(
                {
                    "name": "櫃買指數",
                    "close": r2(otc_index["index_close"]),
                    "chg": r2(otc_index["index_chg"]),
                    "pct": r2(otc_index["index_pct"]),
                }
            )
        sectors = [
            {"name": n.replace("類指數", ""), "close": r2(v["close"]), "pct": r2(v["pct"])}
            for n, v in mi["indices"].items()
            if n.endswith("類指數") and n not in SECTOR_AGGREGATES and v.get("pct") is not None
        ]
        sectors.sort(key=lambda s: s["pct"], reverse=True)
        n = tcfg["sectors_n"]
        sector_data = {
            "leaders": sectors[:n],
            "laggards": list(reversed(sectors[-n:])) if len(sectors) > n else [],
            "all": sectors,  # full sorted list for the sector chart
        }
        inst = self.bfi(d)
        if inst is None:
            self.gap("三大法人買賣金額（BFI82U）尚未公布")
        common = [s for s in mi["stocks"] if COMMON_STOCK.match(s["code"]) and s["turnover"]]

        def row(s):
            return {
                "code": s["code"],
                "name": s["name"],
                "turnover_yi": yi2(s["turnover"]),
                "close": r2(s["close"]),
                "chg": r2(s["chg"]),
                "pct": r2(s["pct"]),
            }

        top = [row(s) for s in sorted(common, key=lambda s: s["turnover"], reverse=True)[: tcfg["top_turnover_n"]]]
        big = [s for s in common if s["pct"] is not None and s["turnover"] >= tcfg["big_turnover_yi"] * 1e8]
        mv = tcfg["movers_n"]
        movers = {
            "threshold_yi": tcfg["big_turnover_yi"],
            "gainers": [row(s) for s in sorted(big, key=lambda s: s["pct"], reverse=True)[:mv] if s["pct"] > 0],
            "losers": [row(s) for s in sorted(big, key=lambda s: s["pct"])[:mv] if s["pct"] < 0],
        }
        otc_movers = None
        try:
            oq = [
                s
                for s in self.tpex_quotes(d)
                if COMMON_STOCK.match(s["code"]) and s["pct"] is not None and s["turnover"]
            ]
            obig = [s for s in oq if s["turnover"] >= tcfg["otc_big_turnover_yi"] * 1e8]
            otc_movers = {
                "threshold_yi": tcfg["otc_big_turnover_yi"],
                "gainers": [row(s) for s in sorted(obig, key=lambda s: s["pct"], reverse=True)[:mv] if s["pct"] > 0],
                "losers": [row(s) for s in sorted(obig, key=lambda s: s["pct"])[:mv] if s["pct"] < 0],
            }
        except (tw.NotReady, FetchError) as e:
            self.log(f"TPEx quotes {d} unavailable: {e}")
            self.gap("上櫃個股行情（TPEx）未取得", retry=False)
        group_ranking, group_status = tw_groups.fetch(self.f, self.cfg, d, self.log)
        data = {
            "session_date": d.isoformat(),
            "indices": idx,
            "turnover": turnover,
            "breadth": breadth,
            "sectors": sector_data,
            "institutions_yi": inst,
            "institutions_detail_yi": self.bfi_detail.get(d),
            "top_turnover": top,
            "movers": movers,
            "otc_movers": otc_movers,
            "watchlist": self.tw_watch_rows(d, mi),
        }
        return {
            "status": "ok",
            "data": data,
            "headlines": self.headlines("tw"),
            "as_of": f"{d.isoformat()} 收盤（Asia/Taipei）",
            # Kept out of "data": the model never sees it and the number check does not cover it.
            "group_ranking": group_ranking,
            "group_ranking_status": group_status,
        }

    def tw_premarket(self, d):
        ins = self.cfg["us_instruments"]
        insts = (
            ins["indices"]
            + ins["rates_fx"]
            + [{"key": w["symbol"], "cnbc": w["symbol"], "yahoo": w["symbol"]} for w in self.watchlist.get("us", [])]
        )
        got, missing = self.quotes(insts)
        if "SPX" not in got:
            self.must_missing.append("US index quotes (S&P 500)")
            return {"status": "must_missing"}
        us_date = got["SPX"]["session_date"]
        prev_d, prev_mi = self.latest_tw_session(self.cal.prev_open("tw", d) or d - dt.timedelta(days=1))
        prev = None
        if prev_mi:
            prev = self.tw_summary(prev_d, prev_mi)
        else:
            self.gap("前一交易日台股資料未取得")
        tw_close = None
        if prev:
            tw_close = next((w.get("close") for w in prev["watchlist"] if w["code"] == self.cfg["adr"]["tw"]), None)
            if tw_close is None and prev_mi:
                tw_close = next((s["close"] for s in prev_mi["stocks"] if s["code"] == self.cfg["adr"]["tw"]), None)
        night = None
        try:
            raw = self.f.request(
                tw.TAIFEX_REPORT_URL,
                method="POST",
                body=tw.taifex_night_form(d),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            night = tw.parse_taifex_night_html(raw, d)
            self.src(f"TAIFEX 期貨每日交易行情 盤後交易時段 {d.isoformat()}", tw.TAIFEX_REPORT_URL)
        except (tw.NotReady, FetchError) as e:
            self.log(f"TAIFEX night report unavailable: {e}; trying MIS")
            try:
                night = tw.parse_mis_tx(self.f.post_json(tw.MIS_URL, tw.mis_payload(1)))
                night["source"] = "TAIFEX MIS 盤後行情"
                self.src("TAIFEX MIS 盤後（夜盤）行情", tw.MIS_URL)
                if night.get("cdate") != ymd_s(d):
                    self.log(f"MIS night CDate {night.get('cdate')} is not {d}; dropped")
                    night = None
            except (tw.NotReady, FetchError) as e2:
                self.log(f"TAIFEX MIS night session unavailable: {e2}")
        if night is None:
            self.gap("台指期夜盤（TAIFEX）未取得")
        else:
            night = {k: (r2(v) if isinstance(v, float) else v) for k, v in night.items()}
            taiex_prev = prev["taiex"]["close"] if prev else None
            night["basis_vs_taiex"] = None if taiex_prev is None else r2(night["last"] - taiex_prev)
        exdiv = None
        try:
            exdiv = tw.parse_twt48u(self.f.get_json(tw.twt48u_url()), d)
            self.src("TWSE TWT48U 除權除息預告表", tw.twt48u_url())
        except (tw.NotReady, FetchError) as e:
            self.log(f"TWT48U unavailable: {e}")
            self.gap("除權息預告（TWT48U）未取得", retry=False)
        credit, credit_status, credit_skip = tw_credit.fetch(
            self.f, self.cfg, prev_d or self.cal.prev_open("tw", d) or d - dt.timedelta(days=1), self.log
        )
        watch_keys = [w["symbol"] for w in self.watchlist.get("us", [])]
        data = {
            "session_date": d.isoformat(),
            "us_session_date": us_date,
            "us_indices": self.us_rows(ins["indices"], got, expect_date=us_date),
            "rates_fx": self.us_rows(ins["rates_fx"], got),
            "adr": self.adr(got, tw_close, prev["date"] if prev else None),
            "tx_night": night,
            "prev_tw": prev,
            "exdiv_today": exdiv,
            "us_watchlist": self.us_rows([{"key": k, "label": k} for k in watch_keys], got, with_ext=True),
        }
        self.gap_missing_quotes("部分美股報價未取得：", missing)
        return {
            "status": "ok",
            "data": data,
            "headlines": self.headlines("tw"),
            "as_of": f"美股 {us_date} 收盤；台指期夜盤；{self.now.astimezone(TPE):%Y-%m-%d %H:%M} Asia/Taipei",
            # Kept out of "data" like the group ranking: the model never sees it and the number check does not cover it.
            "credit": credit if credit else ({"skipped": credit_skip} if credit_skip else None),
            "credit_status": credit_status,
        }

    def us_premarket(self, d):
        ins = self.cfg["us_instruments"]
        watch = [{"key": w["symbol"], "cnbc": w["symbol"], "yahoo": w["symbol"]} for w in self.watchlist.get("us", [])]
        vix = [i for i in ins["indices"] if i["key"] == "VIX"]
        got, missing = self.quotes(ins["futures"] + vix + ins["rates_fx"] + ins["asia"] + watch)
        if not any(k in got for k in ("ES", "NQ")):
            self.must_missing.append("US index futures")
            return {"status": "must_missing"}
        tw_d, tw_mi = self.latest_tw_session(d)
        tw_sum = self.tw_summary(tw_d, tw_mi) if tw_mi else None
        if tw_sum is None:
            self.gap("台股收盤資料未取得", retry=False)
        rows = self.us_rows([{"key": w["key"], "label": w["key"]} for w in watch], got, with_ext=True)
        live_pre = [
            r
            for r in rows
            if (r.get("ext") or {}).get("type") == "PRE_MKT" and r["ext"].get("session_date") == d.isoformat()
        ]
        for r in rows:
            if r.get("ext") and not (
                r["ext"].get("type") == "PRE_MKT" and r["ext"].get("session_date") == d.isoformat()
            ):
                r["ext"] = None  # stale after-hours quote from the previous session: not a pre-market quote
        if not live_pre:
            self.gap("盤前個股報價（CNBC extended hours）未取得", retry=False)
        tw_close = next(
            (w.get("close") for w in (tw_sum or {}).get("watchlist", []) if w["code"] == self.cfg["adr"]["tw"]), None
        )
        data = {
            "session_date": d.isoformat(),
            "futures": self.us_rows(ins["futures"], got),
            "vix_prev": self.us_rows(vix, got),
            "rates_fx": self.us_rows(ins["rates_fx"], got),
            "asia": self.us_rows(ins["asia"], got),
            "tw_today": tw_sum,
            "adr_ref": {
                "tw_code": self.cfg["adr"]["tw"],
                "tw_close": tw_close,
                "tw_date": tw_sum["date"] if tw_sum else None,
            },
            "us_watchlist_pre": rows,
        }
        self.gap_missing_quotes("部分報價未取得：", missing)
        return {
            "status": "ok",
            "data": data,
            "headlines": self.headlines("us"),
            "as_of": f"{self.now.astimezone(ET):%Y-%m-%d %H:%M} America/New_York（盤前）",
        }

    def us_postmarket(self, d):
        ins = self.cfg["us_instruments"]
        watch = [{"key": w["symbol"], "cnbc": w["symbol"], "yahoo": w["symbol"]} for w in self.watchlist.get("us", [])]
        sectors = [{"key": s["key"], "label": s["label"], "cnbc": s["key"], "yahoo": s["key"]} for s in ins["sectors"]]
        got, missing = self.quotes(ins["indices"] + ins["rates_fx"] + sectors + watch)
        spx = got.get("SPX")
        if not spx or spx["session_date"] != d.isoformat():
            self.must_missing.append(f"S&P 500 close for {d} (got {spx and spx['session_date']})")
            return {"status": "must_missing"}
        srows = [r for r in self.us_rows(sectors, got) if not r.get("missing")]
        srows.sort(key=lambda r: r["pct"] if r["pct"] is not None else -999, reverse=True)
        wrows = self.us_rows([{"key": w["key"], "label": w["key"]} for w in watch], got, with_ext=True)
        for r in wrows:
            e = r.get("ext")
            if e and not (str(e.get("type", "")).startswith("POST_MKT") and e.get("session_date") == d.isoformat()):
                r["ext"] = None
        tw_d, tw_mi = self.latest_tw_session(d)
        tw_close = None
        if tw_mi:
            tw_close = next((s["close"] for s in tw_mi["stocks"] if s["code"] == self.cfg["adr"]["tw"]), None)
        data = {
            "session_date": d.isoformat(),
            "indices": self.us_rows(ins["indices"], got, expect_date=d.isoformat(), strict=True),
            "rates_fx": self.us_rows(ins["rates_fx"], got),
            "sectors": srows,
            "watchlist": wrows,
            "adr": self.adr(got, tw_close, tw_d.isoformat() if tw_d else None),
            "breadth": None,
        }
        self.gap_missing_quotes("部分報價未取得：", missing)
        return {
            "status": "ok",
            "data": data,
            "headlines": self.headlines("us"),
            "as_of": f"{d.isoformat()} 收盤；盤後報價至 {self.now.astimezone(ET):%H:%M} America/New_York",
        }


def ymd_s(d):
    return d.strftime("%Y%m%d")


def collect(kind, d, collector):
    """-> facts dict {kind, date, market, session, title, status, data, headlines, sources, gaps, must_missing, as_of}."""
    fn = {
        "tw-postmarket": collector.tw_postmarket,
        "tw-premarket": collector.tw_premarket,
        "us-premarket": collector.us_premarket,
        "us-postmarket": collector.us_postmarket,
    }[kind]
    collector.critical = CRITICAL_QUOTES[kind]
    res = fn(d)
    market, session = kind.split("-")
    return {
        "kind": kind,
        "date": d.isoformat(),
        "market": market,
        "session": session,
        "title": f"{TITLES[kind]} {d.isoformat()}（{weekday_zh(d)}）",
        "status": res["status"],
        "data": res.get("data"),
        "headlines": res.get("headlines") or [],
        "as_of": res.get("as_of"),
        "sources": collector.sources,
        "gaps": collector.gaps,
        "retry_gaps": collector.retry_gaps,
        "group_ranking": res.get("group_ranking"),
        "group_ranking_status": res.get("group_ranking_status"),
        "credit": res.get("credit"),
        "credit_status": res.get("credit_status"),
        "must_missing": collector.must_missing,
        "generated_at": collector.now.astimezone(TPE).isoformat(timespec="seconds"),
    }
