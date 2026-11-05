"""Taiwan market sources: TWSE, TPEx, TAIFEX MIS. Fetch helpers return raw JSON; parse_* are pure functions.

Endpoints (all public, no key):
  TWSE MI_INDEX   https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=YYYYMMDD&type=ALLBUT0999&response=json
                  (價格指數 incl. 類股指數, 大盤統計, 漲跌證券數, 每日收盤行情; type=IND carries the same index tables,
                  so one ALLBUT0999 call is enough; type=MS carries only 大盤統計 and 漲跌證券數, for the alert history)
  TWSE BFI82U     https://www.twse.com.tw/rwd/zh/fund/BFI82U?type=day&dayDate=YYYYMMDD&response=json
  TWSE FMTQIK     https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=YYYYMMDD&response=json   (month table)
  TWSE TWT48U     https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=json                    (除權除息預告表)
  TWSE holidays   https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule?response=json&queryYear=<ROC year>
  TPEx highlight  https://www.tpex.org.tw/www/zh-tw/afterTrading/highlight?date=YYYY/MM/DD&response=json
  TPEx quotes     https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date=YYYY/MM/DD&type=EW&response=json
  TPEx index      https://www.tpex.org.tw/www/zh-tw/indexInfo/inx?date=YYYY/MM/DD&response=json  (month table; fallback
                  when the highlight is late: on 2026-10-22 the highlight was still empty at 15:26 while quotes and
                  index were up by 14:40)
  TAIFEX report   POST https://www.taifex.com.tw/cht/3/futDailyMarketReport  (form: queryType=2, marketCode=1 = 盤後
                  交易時段, commodity_id=TX, queryDate=YYYY/MM/DD). The after-hours session is filed under the trading
                  date it belongs to: queryDate=D is the night from D-1 15:00 to D 05:00, so it is complete at 08:15 on D.
  TAIFEX MIS      POST https://mis.taifex.com.tw/futures/api/getQuoteList  (MarketType 1 = 盤後/夜盤, 0 = 一般); fallback
                  only: it resets for the next night session around 14:50, and its 08:15 behaviour is not verified yet.

A TWSE body whose stat contains "沒有符合條件" for a past date means the market did not trade (holiday / typhoon).
"""

from __future__ import annotations

import re

from numfmt import html_sign, num, pct_from_change, roc_to_iso, strip_html

TWSE = "https://www.twse.com.tw/rwd/zh"
TPEX = "https://www.tpex.org.tw/www/zh-tw"
MIS_URL = "https://mis.taifex.com.tw/futures/api/getQuoteList"
TAIFEX_REPORT_URL = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
NO_DATA = "沒有符合條件"


class MarketClosed(Exception):
    """The exchange reports no data for that date (non-trading day)."""


class NotReady(Exception):
    """The dataset for that date is not published yet (or the body is unusable)."""


def ymd(d):
    return d.strftime("%Y%m%d")


# ------------------------------------------------------------------------------------------------ URLs
def mi_index_url(d):
    return f"{TWSE}/afterTrading/MI_INDEX?date={ymd(d)}&type=ALLBUT0999&response=json"


def bfi82u_url(d):
    return f"{TWSE}/fund/BFI82U?type=day&dayDate={ymd(d)}&response=json"


def fmtqik_url(d):
    return f"{TWSE}/afterTrading/FMTQIK?date={ymd(d)}&response=json"


def twt48u_url():
    return f"{TWSE}/exRight/TWT48U?response=json"


def holiday_url(year):
    return f"{TWSE}/holidaySchedule/holidaySchedule?response=json&queryYear={year - 1911}"


def tpex_highlight_url(d):
    return f"{TPEX}/afterTrading/highlight?date={d:%Y}%2F{d:%m}%2F{d:%d}&response=json"


def tpex_quotes_url(d):
    return f"{TPEX}/afterTrading/otc?date={d:%Y}%2F{d:%m}%2F{d:%d}&type=EW&response=json"


def tpex_index_url(d):
    return f"{TPEX}/indexInfo/inx?date={d:%Y}%2F{d:%m}%2F{d:%d}&response=json"


def mis_payload(market_type):
    return {
        "MarketType": str(market_type),
        "SymbolType": "F",
        "KindID": "1",
        "CID": "TXF",
        "ExpireMonth": "",
        "RowSize": "全部",
        "PageNo": "",
        "SortColumn": "",
        "AscDesc": "A",
    }


# ------------------------------------------------------------------------------------------------ TWSE
def _check_twse(doc, d):
    stat = str(doc.get("stat") or "")
    if NO_DATA in stat:
        raise MarketClosed(stat)
    if stat.upper() != "OK":
        raise NotReady(f"stat={stat[:40]!r}")
    if doc.get("date") and doc["date"] != ymd(d):
        raise NotReady(f"date {doc['date']} != {ymd(d)}")


def _table(doc, needle):
    for t in doc.get("tables") or []:
        if needle in str(t.get("title") or ""):
            return t
    return None


def _breadth_cell(s):
    m = re.match(r"^\s*([\d,]+)\s*(?:\(([\d,]+)\))?", strip_html(s))
    if not m:
        return None, None
    return int(m.group(1).replace(",", "")), (int(m.group(2).replace(",", "")) if m.group(2) else None)


BREADTH_ROWS = {"上漲(漲停)": "up", "下跌(跌停)": "down", "持平": "flat", "未成交": "untraded", "無比價": "nocompare"}


def _breadth(doc):
    """漲跌證券數合計 -> {"all": {...}, "stocks": {...}} (columns 整體市場 / 股票); {} when the table is absent."""
    breadth = {}
    t = _table(doc, "漲跌證券數合計")
    for row in (t or {}).get("data") or []:
        key = BREADTH_ROWS.get(strip_html(row[0]))
        if not key:
            continue
        for col, market in ((1, "all"), (2, "stocks")):
            n, limit = _breadth_cell(row[col])
            breadth.setdefault(market, {})[key] = n
            if key in ("up", "down"):
                breadth[market][key + "_limit"] = limit
    return breadth


def mi_index_ms_url(d):
    """MI_INDEX type=MS: 大盤統計 and 漲跌證券數 only (about 2 KB; the market alert's history)."""
    return f"{TWSE}/afterTrading/MI_INDEX?date={ymd(d)}&type=MS&response=json"


def parse_breadth(doc, d):
    """MI_INDEX (any type carrying 漲跌證券數合計) -> the 股票 column {up, down, ...}. Raises MarketClosed / NotReady."""
    _check_twse(doc, d)
    stocks = _breadth(doc).get("stocks") or {}
    if stocks.get("up") is None or stocks.get("down") is None:
        raise NotReady("MI_INDEX without 漲跌證券數合計")
    return stocks


def parse_mi_index(doc, d):
    """MI_INDEX ALLBUT0999 -> {indices, stats, breadth, stocks}. Raises MarketClosed / NotReady."""
    _check_twse(doc, d)
    indices = {}
    for t in doc.get("tables") or []:
        if "價格指數" not in str(t.get("title") or ""):
            continue
        for row in t.get("data") or []:
            name, close, sign, pts, pct = row[0], num(row[1]), html_sign(row[2]), num(row[3]), num(row[4])
            chg = None if pts is None or sign is None else sign * abs(pts)
            indices[strip_html(name)] = {"close": close, "chg": chg, "pct": pct}
    stats = {}
    t = _table(doc, "大盤統計資訊")
    for row in (t or {}).get("data") or []:
        stats[strip_html(row[0])] = {"turnover": num(row[1]), "shares": num(row[2]), "trades": num(row[3])}
    breadth = _breadth(doc)
    stocks = []
    t = _table(doc, "每日收盤行情")
    if t is not None:
        f = [strip_html(x) for x in t.get("fields") or []]
        ix = {name: f.index(name) for name in f}
        for row in t.get("data") or []:
            close = num(row[ix["收盤價"]])
            sign = html_sign(row[ix["漲跌(+/-)"]])
            diff = num(row[ix["漲跌價差"]])
            chg = None if diff is None or sign is None else sign * abs(diff)
            stocks.append(
                {
                    "code": strip_html(row[ix["證券代號"]]),
                    "name": strip_html(row[ix["證券名稱"]]),
                    "turnover": num(row[ix["成交金額"]]),
                    "close": close,
                    "chg": chg,
                    "pct": pct_from_change(close, chg),
                }
            )
    if not indices or not stocks:
        raise NotReady("MI_INDEX without index or quote tables")
    return {"indices": indices, "stats": stats, "breadth": breadth, "stocks": stocks}


BFI_KEYS = {
    "自營商(自行買賣)": "dealer_self",
    "自營商(避險)": "dealer_hedge",
    "投信": "trust",
    "外資及陸資(不含外資自營商)": "foreign",
    "外資自營商": "foreign_dealer",
    "合計": "total",
}


def parse_bfi82u(doc, d):
    """BFI82U -> {key: {buy, sell, net}} in 元."""
    _check_twse(doc, d)
    out = {}
    for row in doc.get("data") or []:
        key = BFI_KEYS.get(strip_html(row[0]))
        if key:
            out[key] = {"buy": num(row[1]), "sell": num(row[2]), "net": num(row[3])}
    if "total" not in out:
        raise NotReady("BFI82U without 合計 row")
    return out


def parse_fmtqik(doc):
    """FMTQIK month table -> [{date, turnover, taiex, chg}] (ascending)."""
    stat = str(doc.get("stat") or "")
    if stat.upper() != "OK":
        if NO_DATA in stat:
            return []
        raise NotReady(f"FMTQIK stat={stat[:40]!r}")
    rows = []
    for r in doc.get("data") or []:
        iso = roc_to_iso(r[0])
        if iso:
            rows.append({"date": iso, "turnover": num(r[2]), "taiex": num(r[4]), "chg": num(r[5])})
    return sorted(rows, key=lambda x: x["date"])


def parse_twt48u(doc, d):
    """除權除息預告表 rows whose 除權除息日期 is d."""
    if str(doc.get("stat") or "").upper() != "OK":
        raise NotReady("TWT48U stat not OK")
    iso = d.isoformat()
    out = []
    for r in doc.get("data") or []:
        if roc_to_iso(r[0]) != iso:
            continue
        out.append(
            {
                "code": strip_html(r[1]),
                "name": strip_html(r[2]),
                "kind": strip_html(r[3]),
                "stock_ratio": num(r[4]),
                "cash": num(r[7]),
                "cash_text": None if num(r[7]) is not None else strip_html(r[7])[:30],
            }
        )
    return out


def parse_holidays(doc):
    """holidaySchedule -> sorted list of ISO dates on which the market does not trade.

    Rows named "...開始交易日" / "...最後交易日" are trading days and are excluded; "市場無交易，僅辦理結算交割作業" and
    ordinary holidays are closed days.
    """
    if str(doc.get("stat") or "").lower() != "ok":
        raise NotReady("holidaySchedule stat not ok")
    closed = set()
    for r in doc.get("data") or []:
        date, name = str(r[0]).strip(), strip_html(r[1])
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue
        if "交易日" in name:
            continue
        closed.add(date)
    return sorted(closed)


# ------------------------------------------------------------------------------------------------ TPEx
def _tpex_date_ok(table_date, d):
    return roc_to_iso(table_date) == d.isoformat()


def parse_tpex_highlight(doc, d):
    t = (doc.get("tables") or [None])[0]
    if not t or not t.get("data"):
        raise NotReady("TPEx highlight empty")
    if not _tpex_date_ok(t.get("date"), d):
        raise NotReady(f"TPEx highlight date {t.get('date')}")
    f = [strip_html(x) for x in t["fields"]]
    row = dict(zip(f, t["data"][0], strict=False))
    close, chg = num(row.get("收市指數")), num(row.get("指數漲跌"))
    return {
        "index_close": close,
        "index_chg": chg,
        "index_pct": pct_from_change(close, chg),
        "turnover": None if num(row.get("本日總成交值(佰萬元)")) is None else num(row["本日總成交值(佰萬元)"]) * 1e6,
        "up": num(row.get("上漲家數")),
        "up_limit": num(row.get("漲停家數")),
        "down": num(row.get("下跌家數")),
        "down_limit": num(row.get("跌停家數")),
        "flat": num(row.get("平盤家數")),
        "listed": num(row.get("上櫃家數")),
    }


def parse_tpex_quotes(doc, d):
    t = (doc.get("tables") or [None])[0]
    if not t or not t.get("data"):
        raise NotReady("TPEx quotes empty")
    if not _tpex_date_ok(t.get("date"), d):
        raise NotReady(f"TPEx quotes date {t.get('date')}")
    f = [strip_html(x).strip() for x in t["fields"]]
    ix = {name: i for i, name in enumerate(f)}
    out = []
    for r in t["data"]:
        close, chg = num(r[ix["收盤"]]), num(r[ix["漲跌"]])
        out.append(
            {
                "code": strip_html(r[ix["代號"]]),
                "name": strip_html(r[ix["名稱"]]),
                "close": close,
                "chg": chg,
                "pct": pct_from_change(close, chg),
                "turnover": num(r[ix["成交金額(元)"]]),
            }
        )
    return out


def parse_tpex_index(doc, d):
    """櫃買指數 month table -> {index_close, index_chg, index_pct} for d."""
    t = (doc.get("tables") or [None])[0] or {}
    for r in t.get("data") or []:
        if str(r[0]).replace("/", "-") == d.isoformat():
            close, chg = num(r[4]), num(r[5])
            if close is not None:
                return {"index_close": close, "index_chg": chg, "index_pct": pct_from_change(close, chg)}
    raise NotReady(f"TPEx index has no row for {d}")


def breadth_from_quotes(rows):
    """Up/down/flat counts over common stocks (4-digit codes) of a quotes table."""
    out = {"up": 0, "down": 0, "flat": 0}
    for r in rows:
        if not re.fullmatch(r"[1-9]\d{3}", r["code"]) or r["chg"] is None:
            continue
        out["up" if r["chg"] > 0 else "down" if r["chg"] < 0 else "flat"] += 1
    return out


# ------------------------------------------------------------------------------------------------ TAIFEX MIS
_TX_CONTRACT = re.compile(r"^TXF[A-L]\d-[MF]$")


def parse_mis_tx(doc):
    """getQuoteList -> nearest TX contract {symbol, name, last, ref, chg, pct, open, high, low, volume, cdate, ctime}.

    Raises NotReady when the session has no trades yet (CLastPrice empty), e.g. after the 15:00 reset.
    """
    if str(doc.get("RtCode")) != "0":
        raise NotReady(f"MIS RtCode={doc.get('RtCode')}")
    quotes = ((doc.get("RtData") or {}).get("QuoteList")) or []
    contracts = [q for q in quotes if _TX_CONTRACT.match(str(q.get("SymbolID") or ""))]
    spot = next((q for q in quotes if str(q.get("SymbolID", "")).startswith("TXF-")), None)
    if not contracts:
        raise NotReady("MIS has no TX contract")
    q = contracts[0]
    last = num(q.get("CLastPrice"))
    if last is None:
        raise NotReady("MIS TX has no last price yet")
    ref = num(q.get("CRefPrice"))
    chg = None if ref is None else last - ref
    return {
        "symbol": q.get("SymbolID"),
        "name": strip_html(q.get("DispCName")),
        "last": last,
        "ref": ref,
        "chg": chg,
        "pct": None if not ref else chg / ref * 100.0,
        "open": num(q.get("COpenPrice")),
        "high": num(q.get("CHighPrice")),
        "low": num(q.get("CLowPrice")),
        "volume": num(q.get("CTotalVolume")),
        "cdate": q.get("CDate"),
        "ctime": q.get("CTime"),
        "spot_ref": None if spot is None else num(spot.get("CRefPrice")),
    }


def taifex_night_form(d):
    """URL-encoded form body for the TX after-hours (盤後) daily report of trading date d."""
    import urllib.parse

    return urllib.parse.urlencode(
        {
            "queryType": "2",
            "marketCode": "1",
            "MarketCode": "1",
            "dateaddcnt": "",
            "commodity_id": "TX",
            "commodity_idt": "TX",
            "commodity_id2": "",
            "commodity_id2t": "",
            "commodity_id2t2": "",
            "queryDate": d.strftime("%Y/%m/%d"),
        }
    ).encode()


def _signed_arrow(s):
    t = strip_html(s).replace(",", "").replace("%", "").strip()
    sign = -1 if "▼" in t else 1
    t = t.replace("▲", "").replace("▼", "").strip()
    v = num(t)
    return None if v is None else sign * v


def parse_taifex_night_html(html, d):
    """futDailyMarketReport (marketCode=1) HTML -> nearest TX contract of the after-hours session filed under d."""
    if isinstance(html, bytes):
        html = html.decode("utf-8", "replace")
    m = re.search(r"日期：\s*(\d{4}/\d{2}/\d{2})", html)
    if "查無資料" in html or not m:
        raise NotReady("TAIFEX report: no data")
    if m.group(1) != d.strftime("%Y/%m/%d"):
        raise NotReady(f"TAIFEX report date {m.group(1)}")
    body = html[m.end() :]
    end = body.find("</table>")
    body = body if end < 0 else body[:end]
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [strip_html(c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        if len(cells) < 9 or cells[0] != "TX" or not re.fullmatch(r"\d{6}", cells[1]):
            continue
        last, chg = num(cells[5]), _signed_arrow(cells[6])
        if last is None:
            continue
        ref = None if chg is None else last - chg
        return {
            "symbol": f"TX {cells[1]}",
            "name": f"臺股期貨 {cells[1]}（盤後時段）",
            "last": last,
            "ref": ref,
            "chg": chg,
            "pct": _signed_arrow(cells[7]),
            "open": num(cells[2]),
            "high": num(cells[3]),
            "low": num(cells[4]),
            "volume": num(cells[8]),
            "cdate": d.strftime("%Y%m%d"),
            "ctime": None,
            "source": "TAIFEX 期貨每日交易行情（盤後交易時段）",
        }
    raise NotReady("TAIFEX report: no TX contract row")
