#!/usr/bin/env python3
"""Seed and replay for the 市場警示 (alert.py); stdlib only. Run on stormtrooper (it reads the brief's Thalassa key).

alert_seed.py build --twse DIR --yahoo DIR --end YYYY-MM-DD [--config PATH] [--out DIR]
    Writes <out>/tw.csv and <out>/us.csv (default alert_seed/ next to this file) from saved raw responses:
      DIR/bfi82u/YYYYMMDD.json   TWSE BFI82U type=day      -> foreign_yi, total_yi (億元, 外資 = 不含外資自營商)
      DIR/mi_ms/YYYYMMDD.json    TWSE MI_INDEX type=MS     -> up, down (漲跌證券數合計, 股票 column)
      DIR/{GSPC,IXIC,SOX}.json   Yahoo v8 chart, interval 1d -> spx_pct, ixic_pct, sox_pct (close vs previous close)
    and 上市融資金額 change (margin_chg_yi) from Thalassa /api/v1/tw/credit/market for MARGIN_FROM..end.
    A body TWSE answers "no data" for (holiday) is skipped; an unusable one is listed and skipped.
alert_seed.py replay --start YYYY-MM-DD --end YYYY-MM-DD [--config PATH]
    Every TW session (Thalassa TAIEX) and US session (seed) in the range through the alert conditions of the config,
    with the seed as history: days fired per condition, alert days per month, and the 三大法人 / 外資 percentiles
    the thresholds were derived from. No message, no state, no outbox.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import alert
import tw
from netio import Fetcher

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
MARGIN_FROM = "2018-01-01"
US_FROM = "2001-01-02"
YAHOO = {"GSPC": "spx_pct", "IXIC": "ixic_pct", "SOX": "sox_pct"}
PERCENTILES = (0.5, 1, 2, 5)


def _day(path):
    s = path.stem
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def parse_twse(twse_dir, log):
    """{iso: {up, down, foreign_yi, total_yi}} from the saved BFI82U and MI_INDEX type=MS bodies."""
    rows = {}
    for path in sorted((Path(twse_dir) / "bfi82u").glob("*.json")):
        d = _day(path)
        try:
            n = tw.parse_bfi82u(json.loads(path.read_text(encoding="utf-8")), d)
        except tw.MarketClosed:
            continue
        except (tw.NotReady, ValueError) as e:
            log(f"BFI82U {d}: {e}")
            continue
        row = rows.setdefault(d.isoformat(), {})
        row["total_yi"] = round(n["total"]["net"] / 1e8, 2)
        if "foreign" in n:
            row["foreign_yi"] = round(n["foreign"]["net"] / 1e8, 2)
    for path in sorted((Path(twse_dir) / "mi_ms").glob("*.json")):
        d = _day(path)
        try:
            b = tw.parse_breadth(json.loads(path.read_text(encoding="utf-8")), d)
        except tw.MarketClosed:
            continue
        except (tw.NotReady, ValueError) as e:
            log(f"MI_INDEX MS {d}: {e}")
            continue
        rows.setdefault(d.isoformat(), {}).update(up=b["up"], down=b["down"])
    return rows


def parse_yahoo(doc, start):
    """Yahoo v8 chart -> {New York iso date: % change of the close vs the previous close}, dates >= start."""
    res = doc["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    out, prev = {}, None
    for ts, c in zip(res["timestamp"], closes, strict=True):
        if c is None:
            continue
        d = dt.datetime.fromtimestamp(ts, dt.UTC).astimezone(ET).date().isoformat()
        if prev is not None and d >= start:
            out[d] = round((c / prev - 1) * 100, 4)
        prev = c
    return out


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", *fields])
        for d in sorted(rows):
            w.writerow([d, *(rows[d].get(f, "") for f in fields)])


def build(a, cfg, log):
    out = Path(a.out) if a.out else alert.SEED_DIR
    rows = parse_twse(a.twse, log)
    for d, v in alert.fetch_margin(Fetcher(), cfg, MARGIN_FROM, a.end).items():
        rows.setdefault(d, {})["margin_chg_yi"] = v
    rows = {d: r for d, r in rows.items() if d <= a.end}
    write_csv(out / "tw.csv", alert.TW_SEED, rows)
    us = {}
    for sym, key in YAHOO.items():
        for d, v in parse_yahoo(
            json.loads((Path(a.yahoo) / f"{sym}.json").read_text(encoding="utf-8")), US_FROM
        ).items():
            if d <= a.end:
                us.setdefault(d, {})[key] = v
    write_csv(out / "us.csv", alert.US_SEED, us)
    for f in alert.TW_SEED:
        have = sorted(d for d, r in rows.items() if f in r)
        log(f"tw {f}: {len(have)} sessions {have[0] if have else '-'}..{have[-1] if have else '-'}")
    log(f"us: {len(us)} sessions {min(us)}..{max(us)}")
    return 0


def _flags(cond, series):
    return {d: alert.meets(cond["op"], v, float(cond["threshold"])) for d, v in series.items()}


def _percentile(values, p):
    return statistics.quantiles(values, n=1000, method="inclusive")[round(p * 10) - 1]


def replay(a, cfg, log):
    acfg = cfg["alert"]
    hist = alert.History.load(alert.SEED_DIR, None)
    rows = alert.fetch_taiex(Fetcher(), cfg, a.end)
    series = alert.taiex_series(rows)
    series["breadth_ratio"] = alert.breadth_ratio(hist.series("up"), hist.series("down"))
    for s in ("foreign_yi", "total_yi", "margin_chg_yi", *alert.US_SEED):
        series[s] = hist.series(s)
    tw_all = sorted(series["taiex_pct"])
    prev = dict(zip(tw_all[1:], tw_all, strict=False))
    markets = {
        "tw": [d for d in tw_all if a.start <= d <= a.end],
        "us": sorted(d for d in series["spx_pct"] if a.start <= d <= a.end),
    }
    print(f"replay {a.start}..{a.end}; TAIEX from Thalassa through {tw_all[-1]}; seed {alert.SEED_DIR}")
    months = Counter()
    for market, days in markets.items():
        fired_days = Counter()
        print(f"\n{market}: {len(days)} sessions")
        print("| condition | threshold | sessions with a value | days fired | per year |")
        print("|---|---:|---:|---:|---:|")
        years = (dt.date.fromisoformat(days[-1]) - dt.date.fromisoformat(days[0])).days / 365.25
        for cond in acfg[market]:
            flags = _flags(cond, series[cond["key"]])
            at = {d: (prev.get(d) if cond["key"] in alert.LAGGED else d) for d in days}
            hit = [d for d in days if flags.get(at[d])]
            n = sum(1 for d in days if at[d] in flags)
            fired_days.update(hit)
            print(
                f"| {cond['key']} {cond['label']} | {cond['op']} {cond['threshold']:g} | {n} | {len(hit)} | {len(hit) / years:.1f} |"
            )
        nmon = len({d[:7] for d in days})
        print(f"| any (alert days) | | {len(days)} | {len(fired_days)} | {len(fired_days) / years:.1f} |")
        print(f"{market}: {len(fired_days)} alert days over {nmon} months = {len(fired_days) / nmon:.2f} per month")
        for d in fired_days:
            months[(d[:7], market)] += 1
    print("\n| month | tw | us | total |")
    print("|---|---:|---:|---:|")
    for m in sorted({k[0] for k in months}):
        t, u = months[(m, "tw")], months[(m, "us")]
        print(f"| {m} | {t} | {u} | {t + u} |")
    allm = sorted({d[:7] for days in markets.values() for d in days})
    totals = [months[(m, "tw")] + months[(m, "us")] for m in allm]
    print(
        f"\nmonths {len(allm)}; alert messages per month mean {statistics.mean(totals):.2f}, median "
        f"{statistics.median(totals)}, max {max(totals)}; months above 3: {sum(t > 3 for t in totals)}"
    )
    for s in ("foreign_yi", "total_yi"):
        vals = [v for d, v in series[s].items() if a.start <= d <= a.end]
        ps = ", ".join(f"p{p:g} {_percentile(vals, p):,.1f}" for p in PERCENTILES)
        print(f"{s} over {len(vals)} sessions {a.start}..{a.end} (億元): {ps}; min {min(vals):,.1f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--twse", required=True)
    b.add_argument("--yahoo", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--out")
    r = sub.add_parser("replay")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))

    def log(m):
        print(m, file=sys.stderr)

    return build(a, cfg, log) if a.cmd == "build" else replay(a, cfg, log)


if __name__ == "__main__":
    sys.exit(main())
