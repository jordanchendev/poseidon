"""KPI series evidence (gap B3) for research_theme.py: TWSE OpenAPI t187ap05_L monthly revenue rows -> kpi_series:<symbol>
(last 12 months visible at as_of) and its plain-text rendering for the claim precheck. Moved out of research_theme.py
unchanged; see its module docstring for the contract.
"""

import datetime
import json
import re
from pathlib import Path

import pandas as pd
from guidance import _num_text

from poseidon.decision_loop.manifest import content_sha256, iso_time

# ------------------------------------------------------------------------------------------------ KPI series (gap B3)
TWSE_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"


KPI_POINTS = 12


def _roc_ym(s):
    s = str(s or "").strip()
    return "%04d-%s" % (int(s[:-2]) + 1911, s[-2:]) if re.fullmatch(r"\d{4,5}", s) and 1 <= int(s[-2:]) <= 12 else None


def _roc_date_end_utc(s):
    """ROC 'yyyMMdd' -> end of that Asia/Taipei day in UTC (conservative availability), or None."""
    s = str(s or "").strip()
    if not re.fullmatch(r"\d{6,7}", s):
        return None
    try:
        d = datetime.date(int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:]))
    except ValueError:
        return None
    return datetime.datetime.combine(d, datetime.time(15, 59, 59), tzinfo=datetime.UTC)


def _f(v):
    try:
        x = float(str(v).replace(",", "").strip())
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def load_kpi_rows(path=None):
    """Return (rows, source_uri, fetched_at). File form: a list of raw t187ap05_L rows (fetched_at unknown -> None) or
    {"rows": [...], "source_uri": ..., "fetched_at": ...} (e.g. accumulated monthly snapshots). No path -> live fetch."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data, TWSE_REVENUE_URL, None
        return data["rows"], data.get("source_uri", TWSE_REVENUE_URL), data.get("fetched_at")
    import urllib.request

    req = urllib.request.Request(
        TWSE_REVENUE_URL, headers={"User-Agent": "aquarium-research", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.loads(r.read().decode("utf-8-sig"))
    return rows, TWSE_REVENUE_URL, datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prev_month(ym):
    y, m = int(ym[:4]), int(ym[5:])
    return "%04d-%02d" % ((y, m - 1) if m > 1 else (y - 1, 12))


def kpi_series_evidence(symbol, rows, source_uri, fetched_at, as_of):
    """kpi_series:<symbol> from TWSE t187ap05_L rows (unit: NTD thousand). Point-in-time: a row counts only when its
    出表日期 (end of that Taipei day) <= as_of; per 資料年月 the latest such row wins. Returns None if no row."""
    cutoff = pd.Timestamp(as_of).to_pydatetime()
    best = {}
    for raw in rows:
        r = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
        if str(r.get("公司代號", "")).strip() != symbol:
            continue
        ym, pub = _roc_ym(r.get("資料年月")), _roc_date_end_utc(r.get("出表日期"))
        if not ym or not pub or pub > cutoff or _f(r.get("營業收入-當月營收")) is None:
            continue
        if ym not in best or pub > best[ym][0]:
            best[ym] = (pub, r)
    if not best:
        return None
    used = [best[k] for k in sorted(best)][-KPI_POINTS:]
    r2 = lambda x: None if x is None else round(x, 2)
    points = [
        {
            "period": _roc_ym(r["資料年月"]),
            "value": _f(r["營業收入-當月營收"]),
            "yoy_pct": r2(_f(r.get("營業收入-去年同月增減(%)"))),
            "mom_pct": r2(_f(r.get("營業收入-上月比較增減(%)"))),
            "prior_year_value": _f(r.get("營業收入-去年當月營收")),
        }
        for _, r in used
    ]
    last3 = points[-3:]
    consecutive = (
        len(last3) == 3
        and _prev_month(last3[2]["period"]) == last3[1]["period"]
        and _prev_month(last3[1]["period"]) == last3[0]["period"]
    )
    t3 = None
    if consecutive and all(p["prior_year_value"] for p in last3):
        t3 = round((sum(p["value"] for p in last3) / sum(p["prior_year_value"] for p in last3) - 1) * 100, 2)
    raw_used = [r for _, r in used]
    payload = {
        "symbol": symbol,
        "metric": "monthly_revenue",
        "unit": "NTD thousand",
        "points": points,
        "trailing_3m_yoy_pct": t3,
        "source_uri": source_uri,
        "fetched_at": fetched_at,
        "content_sha256": content_sha256(raw_used),
    }
    avail = iso_time(max(p for p, _ in used), "kpi.available_at")
    return {
        "id": "kpi_series:%s" % symbol,
        "kind": "kpi_series",
        "source_uri": source_uri,
        "event_time": avail,
        "available_at": avail,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def kpi_text(payload):
    """kpi_series payload as plain sentences for claim_precheck.check_numbers."""
    out = []
    for p in payload["points"]:
        s = "%s monthly revenue NT$%s thousand" % (p["period"], _num_text(p["value"]))
        if p.get("yoy_pct") is not None:
            s += ", %s %s%% year-over-year" % ("up" if p["yoy_pct"] >= 0 else "down", _num_text(abs(p["yoy_pct"])))
        if p.get("mom_pct") is not None:
            s += "; %s %s%% month-over-month" % ("up" if p["mom_pct"] >= 0 else "down", _num_text(abs(p["mom_pct"])))
        out.append(s + ".")
    t3 = payload.get("trailing_3m_yoy_pct")
    if t3 is not None:
        out.append(
            "Trailing three-month revenue %s %s%% year-over-year." % ("up" if t3 >= 0 else "down", _num_text(abs(t3)))
        )
    return "\n".join(out)
