"""Consensus and surprise evidence (gap B1) for research_theme.py: consensus:<symbol> from a manual JSON snapshot or
the best-effort Yahoo Finance probe, surprise:<symbol> (algorithm surprise-v1) from the consensus and the reported
actuals, and the plain-text renderings the claim precheck reads. Moved out of research_theme.py unchanged; see its
module docstring for the contract.
"""

import datetime
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from guidance import _UNIT_RE, GUIDANCE_BASES, _num_text, quantity_text
from refusal import ResearchRefusal

from poseidon.core.canonical import content_sha256
from poseidon.core.timeutil import iso_time

# ------------------------------------------------------------------------------------------------ consensus + surprise (gap B1)
CONSENSUS_TIER = "third-party aggregate"


SURPRISE_ALGO = "surprise-v1"


_SCALE = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


_CUR_CODE = {"usd": "USD", "us$": "USD", "ntd": "TWD", "twd": "TWD", "nt$": "TWD", "jpy": "JPY"}


_PERIOD_OK = re.compile(r"^(?:[1-4]Q\d{2}|[12]H\d{2}|FY\d{2})$")


def norm_period(p):
    """'3Q26' | 'Q3 2026' | '2026Q3' | '3Q2026' | "Q3'26" -> '3Q26'; 'FY2026' | 'FY26' -> 'FY26'; '1H26' kept.
    Anything else -> upper-cased, whitespace removed (so 'Q4' still matches 'Q4')."""
    s = re.sub(r"[\s'’]", "", str(p or "")).upper()
    for pat, f in (
        (r"([1-4])Q(\d{2}|\d{4})", lambda q, y: "%sQ%s" % (q, y[-2:])),
        (r"Q([1-4])(\d{2}|\d{4})", lambda q, y: "%sQ%s" % (q, y[-2:])),
        (r"(\d{4})Q([1-4])", lambda y, q: "%sQ%s" % (q, y[-2:])),
        (r"([12])H(\d{2}|\d{4})", lambda h, y: "%sH%s" % (h, y[-2:])),
        (r"FY(\d{2}|\d{4})", lambda y: "FY%s" % y[-2:]),
    ):
        m = re.fullmatch(pat, s)
        if m:
            return f(*m.groups())
    return s


def metric_key(m):
    return " ".join(str(m).replace("_", " ").lower().split())


def unit_base(unit):
    """(class, multiplier): money units -> ('money:TWD', 1e9) etc; '%' -> ('%', 1); anything else is its own count class."""
    u = (unit or "").strip()
    m = _UNIT_RE.match(u)
    if m:
        return "money:" + _CUR_CODE[m.group(1).lower()], _SCALE[(m.group(2) or "").lower()]
    if u in ("%", "percent"):
        return "%", 1.0
    if u == "pp":
        return "pp", 1.0
    return "count:" + u.lower(), 1.0


def convert(value, from_unit, to_unit):
    """value in from_unit expressed in to_unit, or None when the unit classes differ (no FX conversion, ever)."""
    (fc, fm), (tc, tm) = unit_base(from_unit), unit_base(to_unit)
    return None if fc != tc else float(value) * fm / tm


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_consensus_items(items, what):
    if not isinstance(items, list) or not items:
        raise ResearchRefusal("%s.items must be a non-empty list" % what)
    out = []
    for n, it in enumerate(items):
        w = "%s.items[%d]" % (what, n)
        if not isinstance(it, dict):
            raise ResearchRefusal("%s must be an object" % w)
        for k in ("metric", "period", "unit", "basis"):
            if not isinstance(it.get(k), str) or not it[k].strip():
                raise ResearchRefusal("%s.%s must be a non-empty string" % (w, k))
        if it["basis"] not in GUIDANCE_BASES:
            raise ResearchRefusal("%s.basis must be one of %s, got %r" % (w, "|".join(GUIDANCE_BASES), it["basis"]))
        period = norm_period(it["period"])
        if not _PERIOD_OK.match(period):
            raise ResearchRefusal("%s.period %r does not normalise to 3Q26 / 1H26 / FY26" % (w, it["period"]))
        for k in ("mean", "low", "high"):
            if not _is_num(it.get(k)):
                raise ResearchRefusal("%s.%s must be a number" % (w, k))
        if "median" in it and it["median"] is not None and not _is_num(it["median"]):
            raise ResearchRefusal("%s.median must be a number when present" % w)
        if not it["low"] <= it["mean"] <= it["high"]:
            raise ResearchRefusal("%s needs low <= mean <= high" % w)
        if isinstance(it.get("n_analysts"), bool) or not isinstance(it.get("n_analysts"), int) or it["n_analysts"] < 1:
            raise ResearchRefusal("%s.n_analysts must be an integer >= 1" % w)
        cls, _ = unit_base(it["unit"])
        cur = it.get("currency")
        if cls.startswith("money:") and cur != cls[6:]:
            raise ResearchRefusal("%s.currency %r does not match unit %r" % (w, cur, it["unit"]))
        if not cls.startswith("money:") and cur not in (None, ""):
            raise ResearchRefusal("%s.currency must be null for a non-money unit %r" % (w, it["unit"]))
        out.append(dict(it, period=period, currency=cur or None))
    return out


def consensus_evidence(data, as_of, source="consensus-json"):
    """consensus:<symbol> from the manual/offline JSON (see README 'Consensus JSON') or a live fetch result.
    The captured raw file must hash to raw_sha256 (like documents' raw_files). Returns (item, note); item None + note when
    the snapshot was observed after as_of (not point-in-time)."""
    what = "%s %s" % (source, data.get("symbol"))
    for k in ("symbol", "provider", "as_of", "source_uri", "raw_sha256", "fetched_at"):
        if not isinstance(data.get(k), str) or not data[k].strip():
            raise ResearchRefusal("%s: %s must be a non-empty string" % (what, k))
    raw = data.get("raw_path")
    if not raw or not Path(raw).is_file():
        raise ResearchRefusal("%s: raw_path must point at the captured raw file (got %r)" % (what, raw))
    digest = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    if digest != data["raw_sha256"]:
        raise ResearchRefusal(
            "%s: raw_path sha256 %s != declared raw_sha256 %s; refusing to freeze"
            % (what, digest[:12], data["raw_sha256"][:12])
        )
    observed = iso_time(data["as_of"], "consensus.as_of")
    if pd.Timestamp(observed) > pd.Timestamp(as_of):
        return None, "consensus for %s observed %s, after as_of %s: not point-in-time, skipped" % (
            data["symbol"],
            observed,
            as_of,
        )
    items = validate_consensus_items(data.get("items"), what)
    payload = {
        "symbol": data["symbol"],
        "provider": data["provider"],
        "as_of": observed,
        "items": items,
        "source_uri": data["source_uri"],
        "raw_files": {Path(raw).name: digest},
        "raw_sha256": digest,
        "fetched_at": data["fetched_at"],
        "source_tier": CONSENSUS_TIER,
        **({"instrument": data["instrument"]} if data.get("instrument") else {}),
        **({"note": data["note"]} if data.get("note") else {}),
    }
    return {
        "id": "consensus:%s" % data["symbol"],
        "kind": "consensus",
        "source_uri": data["source_uri"],
        "event_time": observed,
        "available_at": observed,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }, None


# -- best-effort live fetch (Yahoo Finance; US tickers / ADRs). Never raises; returns (data|None, status dict).
YAHOO_ANALYSIS_URL = "https://finance.yahoo.com/quote/{t}/analysis/"


YAHOO_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{t}?modules=earningsTrend"


_UA = "aquarium-research/1.0 (offline research tooling)"


def _http_get(url, opener, timeout=20):
    import http.client
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/html,application/json"})
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:  # network down, TLS, timeout
        return None, repr(e).encode()


def _period_from_end(end, kind):
    d = datetime.date.fromisoformat(str(end)[:10])
    return ("%dQ%02d" % ((d.month - 1) // 3 + 1, d.year % 100)) if kind == "q" else "FY%02d" % (d.year % 100)


def yahoo_items(summary_result, adr):
    """quoteSummary result -> consensus items. EPS from an ADR quote is per ADR (metric eps_per_adr) so it never matches a
    per-share actual by accident. Periods: calendar quarter/year of trend endDate (assumes calendar fiscal year)."""
    raw = lambda v: v.get("raw") if isinstance(v, dict) else v
    items = []
    for t in (summary_result.get("earningsTrend") or {}).get("trend", []):
        per = str(t.get("period", ""))
        if not re.fullmatch(r"[0+]\d?[qy]", per) or not t.get("endDate"):
            continue
        period = _period_from_end(t["endDate"], per[-1])
        for key, metric in (("earningsEstimate", "eps_per_adr" if adr else "eps"), ("revenueEstimate", "revenue")):
            e = {k: raw(v) for k, v in (t.get(key) or {}).items()}
            if (
                not all(_is_num(e.get(k)) for k in ("avg", "low", "high"))
                or not isinstance(e.get("numberOfAnalysts"), int)
                or e["numberOfAnalysts"] < 1
            ):
                continue
            cur = e.get("earningsCurrency") or e.get("revenueCurrency")
            cur = {"NTD": "TWD"}.get(cur, cur)
            if cur not in ("USD", "TWD", "JPY"):
                continue
            items.append(
                {
                    "metric": metric,
                    "period": period,
                    "mean": e["avg"],
                    "low": e["low"],
                    "high": e["high"],
                    "n_analysts": e["numberOfAnalysts"],
                    "unit": cur,
                    "currency": cur,
                    "basis": "n/a",
                    "provider_period": per,
                }
            )
    return items


def fetch_yahoo_consensus(symbol, ticker, raw_dir, adr=True, opener=None):
    """Probe 1: the public analysis page (server-rendered quoteSummary JSON); probe 2: the quoteSummary API with a crumb.
    Cloudflare challenge pages are refused. Every attempt is recorded {url, status, sha256}; crumbs are stripped."""
    import html as _html
    import http.cookiejar
    import urllib.request

    opener = opener or urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    attempts, now = [], datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def attempt(url):
        st, body = _http_get(url, opener)
        attempts.append({"url": url, "status": st, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        return st, body

    def done(result, body, url, reason=None):
        if result is None:
            return None, {
                "provider": "yahoo",
                "ticker": ticker,
                "status": "unavailable",
                "reason": reason,
                "attempts": attempts,
                "at": now,
            }
        items = yahoo_items(result, adr)
        if not items:
            return None, {
                "provider": "yahoo",
                "ticker": ticker,
                "status": "unavailable",
                "reason": "earningsTrend present but no usable estimates",
                "attempts": attempts,
                "at": now,
            }
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        raw = Path(raw_dir) / (
            "yahoo-%s-%s.%s" % (ticker, now.replace(":", ""), "html" if url.endswith("/analysis/") else "json")
        )
        raw.write_bytes(body)
        data = {
            "symbol": symbol,
            "provider": "yahoo-finance",
            "instrument": "%s (%s)" % (ticker, "ADR" if adr else "listing"),
            "as_of": now,
            "source_uri": url,
            "raw_path": str(raw),
            "raw_sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": now,
            "items": items,
            "note": "Yahoo earningsTrend; EPS per %s; periods = calendar quarter/year of endDate"
            % ("ADR" if adr else "share"),
        }
        return data, {
            "provider": "yahoo",
            "ticker": ticker,
            "status": "ok",
            "items": len(items),
            "attempts": attempts,
            "at": now,
        }

    url = YAHOO_ANALYSIS_URL.format(t=ticker)
    st, body = attempt(url)
    text = body.decode("utf-8", "replace")
    if st == 200 and ("cf-chl" in text or "Just a moment..." in text):
        return done(None, None, url, "Cloudflare challenge page; refused")
    if st == 200:
        for m in re.finditer(
            r'<script type="application/json" data-sveltekit-fetched data-url="([^"]+)"[^>]*>(.*?)</script>', text, re.S
        ):
            if "quoteSummary" not in m.group(1) or "earningsTrend" not in _html.unescape(m.group(1)):
                continue
            try:
                res = json.loads(json.loads(m.group(2))["body"])["quoteSummary"]["result"][0]
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            return done(res, body, url)
    # probe 2: API (needs the A3 cookie from fc.yahoo.com and a crumb)
    attempt("https://fc.yahoo.com/")
    cst, crumb = _http_get("https://query2.finance.yahoo.com/v1/test/getcrumb", opener)
    attempts.append(
        {
            "url": "https://query2.finance.yahoo.com/v1/test/getcrumb",
            "status": cst,
            "bytes": len(crumb),
            "sha256": "(not recorded: session crumb)",
        }
    )
    if cst != 200 or not crumb.strip():
        return done(
            None,
            None,
            url,
            "analysis page had no server-rendered earningsTrend (HTTP %s); getcrumb HTTP %s" % (st, cst),
        )
    from urllib.parse import quote

    api = YAHOO_SUMMARY_URL.format(t=ticker)
    st2, body2 = _http_get(api + "&crumb=" + quote(crumb.decode().strip()), opener)
    attempts.append({"url": api, "status": st2, "bytes": len(body2), "sha256": hashlib.sha256(body2).hexdigest()})
    try:
        res = json.loads(body2)["quoteSummary"]["result"][0] if st2 == 200 else None
    except (ValueError, KeyError, IndexError, TypeError):
        res = None
    return done(res, body2, api, None if res else "quoteSummary API HTTP %s" % st2)


def adr_ticker_for(cfg, symbol):
    t = cfg.get("adr_ticker")
    if isinstance(t, dict):
        return t.get(symbol)
    if isinstance(t, str) and symbol == cfg.get("symbol"):
        return t
    return None if symbol.isdigit() else symbol


def consensus_text(payload):
    """consensus payload as sentences for claim_precheck (claims citing consensus:<symbol>)."""
    return "\n".join(
        "%s %s consensus mean %s, low %s, high %s, %d analysts."
        % (
            it["period"],
            it["metric"].replace("_", " "),
            quantity_text(it["mean"], it["unit"]),
            quantity_text(it["low"], it["unit"]),
            quantity_text(it["high"], it["unit"]),
            it["n_analysts"],
        )
        for it in payload["items"]
    )


def _actual_inputs(doc_items, narrative_acts, evidence_by_id):
    """Actuals usable for surprise: document actuals (payload, hashed) + narrative claim actuals, each with the time it
    became public (the cited document's available_at)."""
    out = []
    for e in doc_items:
        for a in e["payload"].get("actuals") or []:
            out.append(
                {
                    "metric": a["metric"],
                    "period": a["period"],
                    "value": a["value"],
                    "unit": a["unit"],
                    "source": e["id"],
                    "source_content_sha256": e["content_sha256"],
                    "published_at": e["available_at"],
                }
            )
    for a in narrative_acts:
        cited = [
            evidence_by_id[i]
            for i in a.get("evidence_ids", [])
            if i in evidence_by_id and evidence_by_id[i]["kind"] not in ("consensus", "surprise")
        ]
        out.append(
            {
                "metric": a["metric"],
                "period": a["period"],
                "value": a["value"],
                "unit": a["unit"],
                "source": a["source"],
                "cited": [e["id"] for e in cited],
                "published_at": max((e["available_at"] for e in cited), default=None),
            }
        )
    return out


def surprise_rows(consensus_payload, actuals):
    """surprise-v1: for every consensus item with a matching actual (metric_key + norm_period):
    surprise = actual - consensus mean in the consensus unit (actual converted by unit scale only, never FX),
    surprise_pct = surprise / |mean| * 100 (2 dp), vs_range = below | within | above the [low, high] analyst range.
    Only a consensus observed at or before the actual was published is a baseline; otherwise status 'post-event'."""
    rows = []
    for it in consensus_payload["items"]:
        act = next(
            (
                a
                for a in actuals
                if metric_key(a["metric"]) == metric_key(it["metric"]) and norm_period(a["period"]) == it["period"]
            ),
            None,
        )
        if act is None:
            continue
        row = {
            "metric": it["metric"],
            "period": it["period"],
            "unit": it["unit"],
            "basis": it["basis"],
            "consensus": {k: it[k] for k in ("mean", "median", "low", "high", "n_analysts") if it.get(k) is not None},
            "actual": {k: act[k] for k in ("value", "unit", "source", "published_at") if k in act},
            "surprise_abs": None,
            "surprise_pct": None,
            "vs_range": None,
        }
        val = convert(act["value"], act["unit"], it["unit"])
        if val is None:
            row["status"] = "n/a (unit differs: actual %s vs consensus %s)" % (act["unit"], it["unit"])
        elif not act.get("published_at"):
            row["status"] = "n/a (actual publication time unknown)"
        elif pd.Timestamp(consensus_payload["as_of"]) > pd.Timestamp(act["published_at"]):
            row["status"] = "post-event (consensus observed %s, after the actual was published %s)" % (
                consensus_payload["as_of"],
                act["published_at"],
            )
        else:
            s = val - it["mean"]
            row.update(
                status="ok",
                surprise_abs=round(s, 6),
                surprise_pct=(round(s / abs(it["mean"]) * 100, 2) if it["mean"] else None),
                vs_range="above" if val > it["high"] else "below" if val < it["low"] else "within",
                actual_in_consensus_unit=round(val, 6),
            )
        rows.append(row)
    return rows


def surprise_evidence(consensus_item, actuals, as_of):
    """Derived surprise:<symbol>; inputs are listed in the payload so it replays from the manifest (replay_surprise)."""
    rows = surprise_rows(consensus_item["payload"], actuals)
    if not rows:
        return None
    used = [
        a
        for a in actuals
        if any(
            metric_key(a["metric"]) == metric_key(r["metric"]) and norm_period(a["period"]) == r["period"] for r in rows
        )
    ]
    sym = consensus_item["payload"]["symbol"]
    payload = {
        "symbol": sym,
        "algorithm": SURPRISE_ALGO,
        "consensus_evidence_id": consensus_item["id"],
        "consensus_content_sha256": consensus_item["content_sha256"],
        "actual_inputs": used,
        "rows": rows,
    }
    times = [consensus_item["available_at"]] + [a["published_at"] for a in used if a.get("published_at")]
    t = max(times, key=pd.Timestamp)
    return {
        "id": "surprise:%s" % sym,
        "kind": "surprise",
        "source_uri": "derived://%s/%s" % (SURPRISE_ALGO, consensus_item["id"]),
        "event_time": t,
        "available_at": t,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def replay_surprise(manifest_payload):
    """Recompute every surprise:<symbol> from the manifest alone; returns {id: bool}. Document-sourced inputs must equal
    the frozen document actuals; claim-sourced inputs are taken from the payload (they also live in research_json claims)."""
    ev = {e["id"]: e for e in manifest_payload.get("evidence", [])}
    out = {}
    for e in ev.values():
        if e["kind"] != "surprise":
            continue
        p = e["payload"]
        c = ev[p["consensus_evidence_id"]]
        ok = c["content_sha256"] == p["consensus_content_sha256"] == content_sha256(c["payload"])
        for a in p["actual_inputs"]:
            if a["source"] in ev:
                d = ev[a["source"]]
                ok = ok and any(
                    all(x.get(k) == a[k] for k in ("metric", "period", "value", "unit"))
                    for x in d["payload"].get("actuals", [])
                )
        out[e["id"]] = ok and surprise_rows(c["payload"], p["actual_inputs"]) == p["rows"]
    return out


def surprise_text(payload):
    return "\n".join(
        "%s %s actual %s versus consensus mean %s, surprise %s (%s%%)."
        % (
            r["period"],
            r["metric"].replace("_", " "),
            quantity_text(r["actual_in_consensus_unit"], r["unit"]),
            quantity_text(r["consensus"]["mean"], r["unit"]),
            quantity_text(r["surprise_abs"], r["unit"]),
            _num_text(r["surprise_pct"]) if r["surprise_pct"] is not None else "n/a",
        )
        for r in payload["rows"]
        if r.get("status") == "ok"
    )
