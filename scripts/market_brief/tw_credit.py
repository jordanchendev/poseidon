"""Market credit (信用交易) for the TW pre-market brief: one range GET to Thalassa, shaped for the renderer (stdlib only).

`GET /api/v1/tw/credit/market?start&end` returns one row per (date, market in {twse, tpex}). Thalassa ingests margin data
around 22:00 Taipei, so only the pre-market brief can show the previous session; the post-market brief cannot.
`maintenance_ratio_estimate_pct` is an ESTIMATE (see thalassa src/thalassa/data/tw_public/credit.py): the renderer labels it.

Same contract as tw_groups: the key comes from config thalassa.key_file / thalassa.key_var only (llm.load_key), and any
failure (key, network, HTTP status incl. 404 when the endpoint is not deployed yet, malformed, stale, or missing fields)
returns (None, detail, public_reason). The brief then shows one line with the public reason; the detail goes to the log
and the run record. Nothing here ever logs or returns the key, URL or key path in the public reason.
"""

from __future__ import annotations

import datetime as dt

import llm
from netio import FetchError

PATH = "/api/v1/tw/credit/market"
MARKETS = ("twse", "tpex")
TREND_SESSIONS = 5
LOOKBACK_DAYS = 14  # enough calendar days for 5 sessions around a long holiday; fewer sessions just shorten the trend
REQUIRED = (
    "margin_amount",
    "margin_amount_change",
    "margin_balance_change",
    "short_balance_change",
    "maintenance_ratio_estimate_pct",
)


class CreditUnavailable(ValueError):
    """``public`` is safe to print in the brief; ``str(self)`` is the detail for the log."""

    def __init__(self, detail, public):
        super().__init__(detail)
        self.public = public


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(row, field):
    v = row.get(field)
    if not _is_num(v):
        raise CreditUnavailable(f"{row.get('market')} {row.get('date')} lacks a numeric {field}", "欄位缺漏")
    return v


def _yi(twd):
    return round(twd / 1e8, 1)


def shape(resp, d):
    """Validate a reply whose newest session must be ``d`` (the previous trading session) for both markets and round
    it once into display units (億元, lots, percent). Raises CreditUnavailable."""
    if not isinstance(resp, list):
        raise CreditUnavailable(f"reply is {type(resp).__name__}, not a list", "回應格式異常")
    by_date = {}
    for r in resp:
        if not isinstance(r, dict) or not isinstance(r.get("date"), str) or r.get("market") not in MARKETS:
            raise CreditUnavailable("reply has a row without date or a known market", "回應格式異常")
        by_date.setdefault(r["date"], {})[r["market"]] = r
    if not by_date:
        raise CreditUnavailable("reply has no rows", "無資料")
    newest = max(by_date)
    if newest != d.isoformat():
        raise CreditUnavailable(f"newest row is {newest}, wanted {d.isoformat()}", f"資料尚未更新（最新 {newest}）")
    last = by_date[newest]
    if set(last) != set(MARKETS):
        raise CreditUnavailable(f"{newest} has only {sorted(last)}", "欄位缺漏")
    latest = {}
    for m in MARKETS:
        r = last[m]
        vals = {f: _num(r, f) for f in REQUIRED}
        latest[m] = {
            "amount_yi": _yi(vals["margin_amount"]),
            "amount_chg_yi": _yi(vals["margin_amount_change"]),
            "balance_chg": vals["margin_balance_change"],
            "short_chg": vals["short_balance_change"],
            "ratio": round(vals["maintenance_ratio_estimate_pct"], 1),
        }
    trend = []
    for day in sorted(by_date)[-TREND_SESSIONS:]:
        row = {"date": day}
        for m in MARKETS:
            r = by_date[day].get(m) or {}
            amt, ratio = r.get("margin_amount"), r.get("maintenance_ratio_estimate_pct")
            row[m] = {
                "amount_yi": _yi(amt) if _is_num(amt) else None,
                "ratio": round(ratio, 1) if _is_num(ratio) else None,
            }
        trend.append(row)
    return {"date": newest, "latest": latest, "trend": trend}


def fetch(fetcher, cfg, d, log):
    """-> (shaped or None, detail text, public reason or None). The section is optional: nothing raised inside escapes."""
    tcfg = cfg.get("thalassa")
    if not tcfg:
        return None, "disabled (no thalassa config)", None
    try:
        key = llm.load_key(tcfg["key_file"], tcfg["key_var"])
        url = tcfg["base_url"].rstrip("/") + PATH
        params = {"start": (d - dt.timedelta(days=LOOKBACK_DAYS)).isoformat(), "end": d.isoformat()}
        resp = fetcher.get_json(url, params=params, headers={"X-API-Key": key})
        return shape(resp, d), "ok", None
    except CreditUnavailable as e:
        return _fail(log, d, str(e), e.public)
    except llm.MissingKeyError as e:  # SystemExit subclass: must be named, `except Exception` would not catch it
        return _fail(log, d, f"key file: {e}", "金鑰未就緒")
    except FetchError as e:
        public = "端點尚未上線" if e.status == 404 else "Thalassa 讀取失敗"
        return _fail(log, d, f"fetch: {e}", public)
    except Exception as e:  # an optional section must never fail the brief
        return _fail(log, d, f"{type(e).__name__} {e}", "回應格式異常")


def _fail(log, d, detail, public):
    detail = " ".join(str(detail).split())[:300]
    log(f"credit {d}: section skipped ({detail})")
    return None, detail, public
