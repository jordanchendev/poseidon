"""US / global quotes: CNBC quote service (primary, one batched request) with a Yahoo v8 chart fallback per symbol.

CNBC  https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?symbols=A|B|C&requestMethod=itv
      &noform=1&partnerId=2&fund=1&exthrs=1&output=json&events=1
      Indices/futures/FX/yields/stocks in one call; `ExtendedMktQuote` carries pre-market (PRE_MKT) and after-hours
      (POST_MKT / POST_MKT_PREV) quotes. Caveat seen 2026-10-22: for stocks `last_time` lags one day (regular close of
      10/21 labelled 2026-10-20) while indices are labelled correctly, so session dates come from index quotes and from
      the extended quote's last_time, never from a stock's regular last_time. Individual symbols intermittently return
      code 1 (no data); they are retried once in a second batch.
Yahoo https://query1.finance.yahoo.com/v8/finance/chart/<sym>?range=1d&interval=1d  (fallback only; stormtrooper got
      HTTP 429 on most calls on 2026-10-22, so the first 429 opens a circuit breaker for the rest of the run).
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.parse
from zoneinfo import ZoneInfo

from netio import FetchError
from numfmt import num

CNBC_URL = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
ET = ZoneInfo("America/New_York")


def cnbc_url(symbols):
    q = urllib.parse.urlencode(
        {
            "symbols": "|".join(symbols),
            "requestMethod": "itv",
            "noform": "1",
            "partnerId": "2",
            "fund": "1",
            "exthrs": "1",
            "output": "json",
            "events": "1",
        }
    )
    return f"{CNBC_URL}?{q}"


def _session_date(last_time):
    """CNBC last_time ('2026-10-21' or '2026-10-22T02:43:00.000-0400') -> ET ISO date or None."""
    if not last_time:
        return None
    s = str(last_time)
    if "T" not in s:
        return s[:10] if len(s) >= 10 else None
    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", s)
    if not m:
        return s[:10]
    tz = (m.group(2) or "+0000").replace("Z", "+0000").replace(":", "")
    t = dt.datetime.strptime(m.group(1) + tz, "%Y-%m-%dT%H:%M:%S%z")
    return t.astimezone(ET).date().isoformat()


def _real_change(last, chg, pct, prev):
    """(chg, pct, missing). CNBC reports UNCH / 0 when it has rolled its session (seen for VIX at 02:45 ET and
    European indices), with previous_day_closing reset to last, so an exact zero is "unknown", not "flat". Recompute
    from prev_close when that still differs from last; otherwise return None so the report renders "—"."""
    if chg is None or chg != 0 or not last:
        return chg, pct, chg is None
    if prev and prev != last:
        return last - prev, (last - prev) / prev * 100.0, False
    return None, None, True


def parse_cnbc(doc):
    """-> {cnbc_symbol: quote} for symbols with code 0. Quote: last, chg, pct, prev_close, last_time, session_date,
    name, type, ext {type, last, chg, pct, last_time, session_date} or None."""
    out = {}
    res = (doc.get("FormattedQuoteResult") or {}).get("FormattedQuote") or []
    if isinstance(res, dict):
        res = [res]
    for q in res:
        if str(q.get("code")) != "0" or num(q.get("last")) is None:
            continue
        e = q.get("ExtendedMktQuote") or None
        ext = None
        if e and num(e.get("last")) is not None and num(e.get("change")) not in (None, 0.0):
            # an extended quote reported UNCH means no extended-hours trade worth showing: dropped, renders "—"
            ext = {
                "type": e.get("type"),
                "last": num(e.get("last")),
                "chg": num(e.get("change")),
                "pct": num(e.get("change_pct")),
                "last_time": e.get("last_time"),
                "session_date": _session_date(e.get("last_time")),
            }
        last, prev = num(q.get("last")), num(q.get("previous_day_closing"))
        chg, pct, missing = _real_change(last, num(q.get("change")), num(q.get("change_pct")), prev)
        out[q["symbol"]] = {
            "name": q.get("name"),
            "type": q.get("type"),
            "last": last,
            "chg": chg,
            "pct": pct,
            "chg_missing": missing,
            "prev_close": prev,
            "last_time": q.get("last_time"),
            "session_date": _session_date(q.get("last_time")),
            "ext": ext,
            "source": "CNBC",
        }
    return out


def parse_yahoo_chart(doc):
    r = ((doc.get("chart") or {}).get("result") or [None])[0]
    if not r:
        return None
    m = r.get("meta") or {}
    last, prev = m.get("regularMarketPrice"), m.get("chartPreviousClose") or m.get("previousClose")
    if last is None:
        return None
    chg = None if prev is None or prev == last else last - prev  # exact zero treated as unknown, as for CNBC
    t = m.get("regularMarketTime")
    sd = dt.datetime.fromtimestamp(t, dt.UTC).astimezone(ET).date().isoformat() if t else None
    return {
        "name": m.get("shortName") or m.get("symbol"),
        "type": m.get("instrumentType"),
        "last": float(last),
        "chg": chg,
        "pct": None if chg is None else chg / prev * 100.0,
        "chg_missing": chg is None,
        "prev_close": prev,
        "last_time": t,
        "session_date": sd,
        "ext": None,
        "source": "Yahoo",
    }


def fetch_quotes(fetcher, instruments, log=None):
    """instruments: [{key, cnbc, yahoo?}] -> ({key: quote}, [keys missing]). One CNBC batch, one retry batch for
    misses, then Yahoo per remaining symbol until the first 429."""
    by_cnbc = {i["cnbc"]: i for i in instruments if i.get("cnbc")}
    got = {}
    for attempt in range(2):
        want = [s for s in by_cnbc if by_cnbc[s]["key"] not in got]
        if not want:
            break
        for i in range(0, len(want), 40):
            chunk = want[i : i + 40]
            try:
                parsed = parse_cnbc(fetcher.get_json(cnbc_url(chunk)))
            except FetchError as e:
                if log:
                    log(f"cnbc batch failed: {e}")
                continue
            for sym, q in parsed.items():
                if sym in by_cnbc:
                    got[by_cnbc[sym]["key"]] = q
        if attempt == 0 and any(by_cnbc[s]["key"] not in got for s in by_cnbc):
            fetcher.sleep(2.0)
    yahoo_open = True
    for inst in instruments:
        if inst["key"] in got or not inst.get("yahoo") or not yahoo_open:
            continue
        url = YAHOO_URL + urllib.parse.quote(inst["yahoo"]) + "?range=1d&interval=1d"
        try:
            q = parse_yahoo_chart(fetcher.get_json(url, retries=0))
        except FetchError as e:
            if e.status == 429:
                yahoo_open = False
                if log:
                    log("yahoo fallback rate-limited (429); breaker open for this run")
            continue
        if q:
            got[inst["key"]] = q
        fetcher.sleep(1.0)
    missing = [i["key"] for i in instruments if i["key"] not in got]
    return got, missing
