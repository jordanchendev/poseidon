"""市場警示: a descriptive market-move alert, sent as its own Telegram message after the TW and US post-market briefs.

Deterministic and descriptive only: no LLM, no advice, no forecast. The only interpretive line is the SEED-007 one
(capitulation shape tested and rejected); any other interpretive line would have to carry 「未驗證」, and there is none.

When: run.py `due`, after the tw-postmarket (market tw) or us-postmarket (market us) brief of session D was queued and
delivered in that tick, so the alert never delays or blocks a brief. Every config `alert.<market>` condition is evaluated
from the brief's facts plus history; any one that fires queues ONE outbox entry <D>-alert-<market>.json (bridge verb
`brief-alert <market> <D>`, same deliver.py path as the briefs). The runner's state key alert:<market>:<D> makes it once
per (market, D); the bridge keeps a sent marker per (market, D) as well. A failure here is logged, never raised.

Each fired line gives the value, the threshold and the rarity: days in the history with a value, from the later of
config alert.since and the series' first day, through the value's day, and how many of them met the condition.

History
  TAIEX OHLC      Thalassa GET /api/v1/refdata/index-daily?index_name=TAIEX (1999-01 on), read at every evaluation.
  TX volume       Thalassa GET /api/v1/refdata/taifex-futures-daily?product=TX, last TX_LOOKBACK_DAYS (SEED-007 shape).
  上市融資金額     Thalassa GET /api/v1/tw/credit/market, last CREDIT_LOOKBACK_DAYS, over the seed (2018-01 on). Margin
                  posts around 22:00 Taipei, after the 17:45 brief, so alert D uses the session before D.
  breadth, 三大法人, S&P 500 / Nasdaq / SOX: no Thalassa endpoint serves them (/nonprice/institutional is per stock and
                  in shares, /tw/market-daily has no previous close to count 漲跌 with, and Thalassa holds no US index).
                  The seed alert_seed/{tw,us}.csv (built by alert_seed.py from TWSE MI_INDEX type=MS + BFI82U for every
                  session 2021-01-04..2026-11-04 and Yahoo closes from 2001) plus what the runner appends to
                  <state>/alert-history.json. TW sessions after the seed that are still missing are fetched from TWSE
                  (at most FILL_MAX per evaluation); US days cannot be refetched (Yahoo throttles stormtrooper), so a
                  missed US day only lowers N.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import llm
import tw
from netio import FetchError
from numfmt import fmt_pct, fmt_signed, weekday_zh

HERE = Path(__file__).resolve().parent
SEED_DIR = HERE / "alert_seed"
TRIGGERS = {"tw-postmarket": "tw", "us-postmarket": "us"}  # brief kind -> alert market
MARKET_ZH = {"tw": "台股", "us": "美股"}
TAIEX_PATH = "/api/v1/refdata/index-daily"
TX_PATH = "/api/v1/refdata/taifex-futures-daily"
CREDIT_PATH = "/api/v1/tw/credit/market"
TAIEX_FROM = "1999-01-01"
TX_LOOKBACK_DAYS = 30
CREDIT_LOOKBACK_DAYS = 14
CREDIT_MAX_DAYS = 2000  # Thalassa /tw/credit/market range limit
FILL_MAX = 10
MONTHLY = re.compile(r"^\d{6}$")  # TX monthly contract (weekly ones are YYYYMMWn)
TW_SEED = ("up", "down", "foreign_yi", "total_yi", "margin_chg_yi")
US_SEED = ("spx_pct", "ixic_pct", "sox_pct")
US_QUOTE = {"spx_pct": "SPX", "ixic_pct": "IXIC", "sox_pct": "SOX"}
UNITS = {
    "taiex_pct": "%",
    "taiex_5d_pct": "%",
    "taiex_low_pct": "%",
    "breadth_ratio": "倍",
    "margin_chg_yi": "億",
    "foreign_yi": "億",
    "total_yi": "億",
    "spx_pct": "%",
    "ixic_pct": "%",
    "sox_pct": "%",
}
LAGGED = frozenset({"margin_chg_yi"})  # published the evening after its session: alert D reads the session before D
TAIEX_KEYS = frozenset({"taiex_pct", "taiex_5d_pct", "taiex_low_pct"})
SEED007 = "SEED-007 H1／H2 否決：歷史上此型態之後的報酬沒有比較好"


class HistoryUnavailable(ValueError):
    """A Thalassa read failed; str() is the detail for the log (never the key or its path)."""


# ------------------------------------------------------------------------------------------------ history store
class History:
    """{series: {iso date: float}}: the committed seed with the runner's appends (state) on top."""

    def __init__(self, seed, extra):
        self.seed = seed
        self.extra = extra

    @classmethod
    def load(cls, seed_dir, state_path):
        seed = {}
        for name, fields in (("tw.csv", TW_SEED), ("us.csv", US_SEED)):
            with open(Path(seed_dir) / name, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    for f in fields:
                        if row.get(f) not in (None, ""):
                            seed.setdefault(f, {})[row["date"]] = float(row[f])
        extra = {}
        if state_path is not None:
            try:
                extra = json.loads(Path(state_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                extra = {}
        return cls(seed, extra)

    def series(self, name):
        return {**self.seed.get(name, {}), **self.extra.get(name, {})}

    def seed_last(self, name):
        return max(self.seed.get(name) or {"": 0})

    def add(self, name, date, value):
        if value is not None:
            self.extra.setdefault(name, {})[date] = float(value)

    def save(self, state_path):
        p = Path(state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.extra, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)


def thalassa_get(fetcher, cfg, path, params):
    """GET a Thalassa JSON list with the brief's key (config thalassa.key_file / key_var only). Raises HistoryUnavailable."""
    t = cfg.get("thalassa")
    if not t:
        raise HistoryUnavailable("no thalassa config")
    try:
        key = llm.load_key(t["key_file"], t["key_var"])
        resp = fetcher.get_json(t["base_url"].rstrip("/") + path, params=params, headers={"X-API-Key": key})
    except llm.MissingKeyError as e:  # SystemExit subclass: must be named
        raise HistoryUnavailable(f"key file: {e}") from None
    except FetchError as e:
        raise HistoryUnavailable(f"fetch: {e}") from None
    if not isinstance(resp, list):
        raise HistoryUnavailable(f"{path} reply is {type(resp).__name__}, not a list")
    return resp


def fetch_taiex(fetcher, cfg, end):
    rows = thalassa_get(fetcher, cfg, TAIEX_PATH, {"index_name": "TAIEX", "start": TAIEX_FROM, "end": end})
    try:
        return sorted(
            (
                {"date": str(r["trade_date"]), "high": r.get("high"), "low": r.get("low"), "close": float(r["close"])}
                for r in rows
            ),
            key=lambda r: r["date"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise HistoryUnavailable(f"TAIEX row malformed: {e}") from None


def fetch_tx_volume(fetcher, cfg, start, end):
    """{iso date: TX day-session volume summed over monthly contracts} (SEED-007 §3.1 V)."""
    rows = thalassa_get(fetcher, cfg, TX_PATH, {"product": "TX", "start": start, "end": end})
    out = {}
    try:
        for r in rows:
            if r["session"] == "day" and MONTHLY.match(str(r["contract_month"])):
                d = str(r["trade_date"])
                out[d] = out.get(d, 0) + int(r["volume"])
    except (KeyError, TypeError, ValueError) as e:
        raise HistoryUnavailable(f"TX row malformed: {e}") from None
    return out


def fetch_margin(fetcher, cfg, start, end):
    """{iso date: 上市 (twse) 融資金額 change in 億元}, in ranges of at most CREDIT_MAX_DAYS."""
    out = {}
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while a <= b:
        z = min(b, a + dt.timedelta(days=CREDIT_MAX_DAYS))
        for r in thalassa_get(fetcher, cfg, CREDIT_PATH, {"start": a.isoformat(), "end": z.isoformat()}):
            if isinstance(r, dict) and r.get("market") == "twse" and isinstance(r.get("margin_amount_change"), int):
                out[str(r["date"])] = round(r["margin_amount_change"] / 1e8, 2)
        a = z + dt.timedelta(days=1)
    return out


def fill_tw(fetcher, hist, days, log):
    """Fetch breadth + 三大法人 from TWSE for `days` (TW sessions missing in hist); adds what it gets."""
    for d in days:
        day = dt.date.fromisoformat(d)
        try:
            b = tw.parse_breadth(fetcher.get_json(tw.mi_index_ms_url(day)), day)
            n = tw.parse_bfi82u(fetcher.get_json(tw.bfi82u_url(day)), day)
        except tw.MarketClosed:
            continue
        except (tw.NotReady, FetchError) as e:
            log(f"alert history fill {d}: {e}")
            continue
        hist.add("up", d, b["up"])
        hist.add("down", d, b["down"])
        hist.add("foreign_yi", d, round(n["foreign"]["net"] / 1e8, 2) if "foreign" in n else None)
        hist.add("total_yi", d, round(n["total"]["net"] / 1e8, 2))


# ------------------------------------------------------------------------------------------------ series (pure)
def taiex_series(rows):
    """Sorted TAIEX rows -> {taiex_pct, taiex_5d_pct, taiex_low_pct}: {iso: %} (close vs previous close, close vs the
    close 5 sessions earlier, low vs previous close)."""
    out = {k: {} for k in TAIEX_KEYS}
    for i in range(1, len(rows)):
        d, c, prev = rows[i]["date"], rows[i]["close"], rows[i - 1]["close"]
        out["taiex_pct"][d] = (c / prev - 1) * 100
        if rows[i].get("low") is not None:
            out["taiex_low_pct"][d] = (rows[i]["low"] / prev - 1) * 100
        if i >= 5:
            out["taiex_5d_pct"][d] = (c / rows[i - 5]["close"] - 1) * 100
    return out


def breadth_ratio(up, down):
    """{iso: decliners / advancers}; no advancer with decliners -> inf."""
    out = {}
    for d, u in up.items():
        n = down.get(d)
        if n is None:
            continue
        out[d] = n / u if u > 0 else (math.inf if n > 0 else 0.0)
    return out


def merge_today_taiex(rows, d, close):
    """Rows with session d appended from the brief's close when Thalassa does not have it yet (no OHLC then)."""
    if close is None or any(r["date"] == d for r in rows):
        return rows
    return [*(r for r in rows if r["date"] < d), {"date": d, "high": None, "low": None, "close": float(close)}]


def meets(op, v, threshold):
    return v <= threshold if op == "<=" else v >= threshold


@dataclass(frozen=True)
class Reading:
    key: str
    label: str
    op: str
    threshold: float
    unit: str
    at: str | None  # the value's session (D, or the session before D for LAGGED keys)
    value: float | None
    fired: bool = False
    days: int = 0  # sessions with a value from `first` through `at`
    hits: int = 0  # of those, the sessions that met the condition
    first: str | None = None
    streak: int = 0  # consecutive sessions meeting it, ending at `at`
    note: str = ""  # detail shown with the value, or why there is no value
    no_history: str = ""  # why the rarity is missing


def reading(cond, series, at, since, *, note="", no_history=""):
    """One condition read at session `at` against its daily series."""
    key = cond["key"]
    base = {"key": key, "label": cond["label"], "op": cond["op"], "threshold": float(cond["threshold"])}
    base["unit"] = UNITS[key]
    if at is None or at not in series:
        return Reading(**base, at=at, value=None, note=note or "當日資料未取得")
    dates = sorted(k for k in series if since <= k <= at)
    flags = [meets(cond["op"], series[k], base["threshold"]) for k in dates]
    streak = 0
    for f in reversed(flags):
        if not f:
            break
        streak += 1
    return Reading(
        **base,
        at=at,
        value=series[at],
        fired=flags[-1],
        days=len(dates),
        hits=sum(flags),
        first=dates[0],
        streak=streak,
        note=note,
        no_history=no_history,
    )


def capitulation(rows, volume, d, cap):
    """SEED-007 §3.1 shape on session d from TAIEX OHLC and TX day volume: {dd_i, clv, vr, holds}; None without D's
    OHLC. vr: V(D) over the mean V of the 5 sessions before D that are not weekend sessions (None if any is 0/missing)."""
    i = next((k for k, r in enumerate(rows) if r["date"] == d), None)
    if not i or rows[i]["low"] is None or rows[i]["high"] is None:
        return None
    r, prev = rows[i], rows[i - 1]["close"]
    dd_i = 1 - r["low"] / prev
    clv = (r["close"] - r["low"]) / (r["high"] - r["low"]) if r["high"] > r["low"] else None
    before = [k for k in sorted(volume) if k < d and dt.date.fromisoformat(k).weekday() < 5][-5:]
    vr = None
    if volume.get(d) and len(before) == 5 and all(volume[k] > 0 for k in before):
        vr = volume[d] / (sum(volume[k] for k in before) / 5)
    shape = dd_i >= cap["dd_i_min"] and clv is not None and clv >= cap["clv_min"]
    return {
        "dd_i": dd_i,
        "clv": clv,
        "vr": vr,
        "shape": shape,
        "holds": shape and vr is not None and vr >= cap["vr_min"],
    }


# ------------------------------------------------------------------------------------------------ evaluation
@dataclass
class Result:
    market: str
    date: str
    readings: list = field(default_factory=list)
    cap: dict | None = None
    adr_pct: float | None = None
    problems: list = field(default_factory=list)  # data that could not be read (log + run record)

    @property
    def fired(self):
        return [r for r in self.readings if r.fired]


def _session_before(days, d):
    return max((x for x in days if x < d), default=None)


def tw_today(facts):
    """(TAIEX row {close, pct}, breadth {up, down}, 三大法人 {foreign, total} in 億) from tw-postmarket facts."""
    data = facts.get("data") or {}
    tx = next((i for i in data.get("indices") or [] if i.get("name") == "加權指數"), {})
    breadth = (data.get("breadth") or {}).get("twse_stocks") or {}
    return tx, breadth, data.get("institutions_yi") or {}


def evaluate_tw(d, facts, *, cfg, fetcher, hist, log):
    acfg = cfg["alert"]
    res = Result("tw", d)
    taiex, breadth, inst = tw_today(facts)
    no_hist = ""
    try:
        rows = fetch_taiex(fetcher, cfg, d)
    except HistoryUnavailable as e:
        rows, no_hist = [], "歷史資料未取得（Thalassa 讀取失敗）"
        res.problems.append(f"TAIEX history: {e}")
    rows = merge_today_taiex(rows, d, taiex.get("close"))
    series = taiex_series(rows)
    if no_hist and taiex.get("pct") is not None:  # no history: the brief still has the day's change
        series["taiex_pct"] = {d: taiex["pct"]}
    hist.add("up", d, breadth.get("up"))
    hist.add("down", d, breadth.get("down"))
    hist.add("foreign_yi", d, inst.get("foreign"))
    hist.add("total_yi", d, inst.get("total"))
    days = sorted({r["date"] for r in rows} | set(hist.series("up")))
    after_seed = min(hist.seed_last("up"), hist.seed_last("foreign_yi"))
    gaps = [x for x in days if after_seed < x < d and not all(x in hist.series(s) for s in ("up", "foreign_yi"))]
    if gaps:
        log(f"alert tw {d}: filling {len(gaps[-FILL_MAX:])} of {len(gaps)} missing session(s) from TWSE")
        fill_tw(fetcher, hist, gaps[-FILL_MAX:], log)
    try:
        start = (dt.date.fromisoformat(d) - dt.timedelta(days=CREDIT_LOOKBACK_DAYS)).isoformat()
        for k, v in fetch_margin(fetcher, cfg, start, d).items():
            hist.add("margin_chg_yi", k, v)
    except HistoryUnavailable as e:
        res.problems.append(f"margin: {e}")
    series["breadth_ratio"] = breadth_ratio(hist.series("up"), hist.series("down"))
    for s in ("foreign_yi", "total_yi", "margin_chg_yi"):
        series[s] = hist.series(s)
    prev = _session_before(days, d)
    for cond in acfg["tw"]:
        key, note = cond["key"], ""
        if key == "breadth_ratio" and None not in (breadth.get("up"), breadth.get("down")):
            note = f"{breadth['down']:,} 家跌、{breadth['up']:,} 家漲"
        at = prev if key in LAGGED else d
        if key == "margin_chg_yi" and at not in series[key]:
            note = "前一交易日融資資料未取得"
        if key in TAIEX_KEYS and no_hist and d not in series[key]:
            note = no_hist
        res.readings.append(
            reading(cond, series[key], at, acfg["since"], note=note, no_history=no_hist if key in TAIEX_KEYS else "")
        )
    if rows and not no_hist:
        try:
            start = (dt.date.fromisoformat(d) - dt.timedelta(days=TX_LOOKBACK_DAYS)).isoformat()
            res.cap = capitulation(rows, fetch_tx_volume(fetcher, cfg, start, d), d, acfg["capitulation"])
        except HistoryUnavailable as e:
            res.problems.append(f"TX volume: {e}")
            res.cap = capitulation(rows, {}, d, acfg["capitulation"])
    return res


def evaluate_us(d, facts, *, cfg, hist):
    acfg = cfg["alert"]
    res = Result("us", d)
    data = facts.get("data") or {}
    quotes = {r.get("key"): r for r in data.get("indices") or []}
    for s, k in US_QUOTE.items():
        hist.add(s, d, (quotes.get(k) or {}).get("pct"))
    for cond in acfg["us"]:
        res.readings.append(reading(cond, hist.series(cond["key"]), d, acfg["since"]))
    res.adr_pct = (data.get("adr") or {}).get("adr_pct")
    return res


def evaluate(market, d, facts, *, cfg, fetcher, state_dir, log, persist=True):
    """Evaluate the alert of (market, session d) from the brief facts. persist=False (dry runs) keeps the history
    appends in memory only."""
    path = Path(state_dir) / "alert-history.json"
    hist = History.load(SEED_DIR, path)
    if market == "tw":
        res = evaluate_tw(d, facts, cfg=cfg, fetcher=fetcher, hist=hist, log=log)
    else:
        res = evaluate_us(d, facts, cfg=cfg, hist=hist)
    if persist:
        hist.save(path)
    return res


# ------------------------------------------------------------------------------------------------ message
def _g(x):
    return f"{x:g}"


def fmt_value(r):
    if r.unit == "%":
        return fmt_pct(r.value)
    if r.unit == "億":
        return f"{fmt_signed(r.value, 1)} 億元"
    return "無上漲家數" if math.isinf(r.value) else f"{r.value:.1f} 倍"


def fmt_threshold(r):
    unit = {"%": "%", "億": " 億元", "倍": " 倍"}[r.unit]
    return f"門檻 {'≤' if r.op == '<=' else '≥'} {_g(r.threshold)}{unit}"


def _label(r, d):
    return r.label if r.at in (None, d) else f"{r.label}（{r.at}）"


def rarity_text(r):
    if r.no_history:
        return r.no_history
    span = (dt.date.fromisoformat(r.at) - dt.date.fromisoformat(r.first)).days / 365.25
    per_year = f"，約每年 {r.hits / span:.1f} 次" if span >= 1 else ""
    s = f"{r.first[:4]} 年以來 {r.days:,} 個交易日中第 {r.hits:,} 次{per_year}"
    return s + (f"；連續第 {r.streak} 個交易日" if r.streak >= 2 else "")


def cap_line(cap, cfg_cap):
    """The SEED-007 line: only when the price shape holds (with volume: the verdict; without: cannot tell)."""
    if not cap or not cap["shape"]:
        return None
    shape = f"盤中最低較前收跌 {cap['dd_i'] * 100:.2f}%、收在當日區間 {cap['clv'] * 100:.0f}% 處"
    if cap["vr"] is None:
        return f"投降日型態：{shape}，但台指期量能未取得，無法判斷是否符合 SEED-007 型態。"
    if not cap["holds"]:
        return None
    gate = (
        f"SEED-007 門檻：跌幅 ≥ {cfg_cap['dd_i_min'] * 100:.2f}%、位置 ≥ {cfg_cap['clv_min'] * 100:.0f}%、"
        f"量 ≥ {cfg_cap['vr_min']:.2f} 倍"
    )
    return f"投降日型態成立（{shape}、台指期日盤量為前 5 日均量 {cap['vr']:.2f} 倍；{gate}）：此型態已檢驗並否決（{SEED007}）。"


def render(res, cfg):
    """-> Telegram text, or None when nothing fired."""
    fired = res.fired
    if not fired:
        return None
    d = res.date
    day = dt.date.fromisoformat(d)
    lines = [
        f"**市場警示 {d}**",
        f"{MARKET_ZH[res.market]} {d}（{weekday_zh(day)}）收盤，觸發 {len(fired)} 項條件：",
    ]
    for r in fired:
        note = f"；{r.note}" if r.note else ""
        lines.append(f"- {_label(r, d)} {fmt_value(r)}（{fmt_threshold(r)}{note}）：{rarity_text(r)}")
    quiet = [r for r in res.readings if not r.fired and r.value is not None]
    if quiet:
        lines.append("未觸發：" + "；".join(f"{_label(r, d)} {fmt_value(r)}（{fmt_threshold(r)}）" for r in quiet))
    missing = [r for r in res.readings if r.value is None]
    if missing:
        lines.append("無法判斷：" + "；".join(f"{_label(r, d)}（{r.note}）" for r in missing))
    if res.market == "us" and res.adr_pct is not None:
        lines.append(f"台積電 ADR 收盤 {fmt_pct(res.adr_pct)}（不設門檻）。")
    cap = cap_line(res.cap, cfg["alert"]["capitulation"]) if res.market == "tw" else None
    if cap:
        lines.append(cap)
    lines.append("僅描述市場數據，不含任何買賣建議。")
    return "\n".join(lines)


def outbox_name(market, d):
    return f"{d}-alert-{market}.json"


def outbox_entry(market, d, text):
    return {"verb": "brief-alert", "market": market, "date": d, "text": text}


def summary(res):
    """One line for the log and the state: fired keys and the data problems."""
    fired = ",".join(r.key for r in res.fired) or "none"
    missing = ",".join(r.key for r in res.readings if r.value is None)
    s = f"fired {fired}" + (f"; no value {missing}" if missing else "")
    return s + (f"; {'; '.join(res.problems)}" if res.problems else "")


def record(res):
    """The evaluation as JSON-able data (dry runs and the replay)."""
    return {
        "market": res.market,
        "date": res.date,
        "readings": [asdict(r) for r in res.readings],
        "capitulation": res.cap,
        "adr_pct": res.adr_pct,
        "problems": res.problems,
    }
