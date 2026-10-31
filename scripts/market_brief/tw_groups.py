"""Group strength data for the TW post-market brief: one GET to Thalassa, shaped for the renderer (stdlib only).

The group table and the ranking live in Thalassa (`GET /api/v1/tw/groups/ranking?date=`); this module only fetches and
validates what it returns. The key comes from config thalassa.key_file / thalassa.key_var, read from that 600 env file
only (llm.load_key): no fallback to another variable or to os.environ. Any failure (key file, network, HTTP status,
malformed or wrong-date reply, anything unexpected) returns (None, reason): the brief goes out without the section and the reason is logged
and kept in the run record. Nothing here ever logs or returns the key.
"""

from __future__ import annotations

import llm
from netio import FetchError

PATH = "/api/v1/tw/groups/ranking"
GROUP_FIELDS = (
    "group",
    "n_members",
    "n_valid",
    "missing",
    "ret20",
    "ret60",
    "turnover_share",
    "turnover_share_change",
    "rank",
)


def _pct(x, nd=1):
    return None if x is None else round(x * 100.0, nd)


def shape(resp, d):
    """Validate a ranking reply for session ``d`` and round it once into display units (percent).

    Raises ValueError with a short reason when the reply cannot be used."""
    if not isinstance(resp, dict) or not isinstance(resp.get("groups"), list) or not resp["groups"]:
        raise ValueError("reply has no groups")
    if resp.get("date") != d.isoformat():
        raise ValueError(f"reply is for {resp.get('date')}, wanted {d.isoformat()}")
    taiex = resp.get("taiex") or {}
    if taiex.get("ret20") is None or taiex.get("ret60") is None:
        raise ValueError("reply has no TAIEX returns")
    groups = []
    for g in resp["groups"]:
        if not isinstance(g, dict):
            raise ValueError(f"group entry is {type(g).__name__}, not an object")
        miss = [f for f in GROUP_FIELDS if f not in g]
        if miss:
            raise ValueError(f"group {g.get('group')!r} lacks {','.join(miss)}")
        groups.append(
            {
                "name": g["group"],
                "n": g["n_members"],
                "n_valid": g["n_valid"],
                "missing": list(g["missing"]),
                "ret20": _pct(g["ret20"]),
                "ret60": _pct(g["ret60"]),
                "share": _pct(g["turnover_share"]),
                "share_chg": _pct(g["turnover_share_change"]),
                "rank": g["rank"],
            }
        )
    groups.sort(key=lambda x: (x["rank"] is None, x["rank"] if x["rank"] is not None else 0))
    return {
        "date": resp["date"],
        "table_version": resp.get("table_version"),
        "taiex": {"ret20": _pct(taiex["ret20"]), "ret60": _pct(taiex["ret60"])},
        "groups": groups,
    }


def fetch(fetcher, cfg, d, log):
    """-> (shaped ranking or None, status text). The section is optional: nothing raised inside it escapes."""
    tcfg = cfg.get("thalassa")
    if not tcfg:
        return None, "disabled (no thalassa config)"
    try:
        key = llm.load_key(tcfg["key_file"], tcfg["key_var"])
        url = tcfg["base_url"].rstrip("/") + PATH
        resp = fetcher.get_json(url, params={"date": d.isoformat()}, headers={"X-API-Key": key})
        return shape(resp, d), "ok"
    except llm.MissingKeyError as e:  # SystemExit subclass: must be named, `except Exception` would not catch it
        return _fail(log, d, f"key file: {e}")
    except FetchError as e:
        return _fail(log, d, f"fetch: {e}")
    except Exception as e:  # an optional section must never fail the brief
        return _fail(log, d, f"{type(e).__name__} {e}")


def _fail(log, d, reason):
    reason = " ".join(str(reason).split())[:300]
    log(f"group ranking {d}: section skipped ({reason})")
    return None, reason
