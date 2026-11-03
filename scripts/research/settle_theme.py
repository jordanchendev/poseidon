#!/usr/bin/env python3
"""Settlement / prediction ledger for completed research revisions (gap B6). Offline, SQLite only.

Reads completed `research_revisions` (+ their `data_manifests`) from the ledger research_theme.py writes and keeps a
separate table `research_settlements` in the same file. See SETTLEMENT.md for the rules; summary:

  settle   --ledger L --csv BARS [--as-of T] [--force] [--benchmark 0050]
           settles every completed revision whose horizon (leading ISO time of next_review_at) <= as-of.
  flag     --ledger L --revision ID --note TEXT [--clear]     human invalidation flag (never set automatically).
  report   --ledger L [--as-of T] [--out FILE]                markdown summary + per-revision table.
  pending  --ledger L [--as-of T]                             revisions awaiting their horizon / due but unsettled.

Supersede: a revision is superseded when a LATER completed revision in the same ledger carries supersedes_revision_id (or
`supersedes`) naming it; its row gets outcome `superseded`, keeps outcome_before_supersede and superseded_by, and is
excluded from hit counts, hit rate and mean rel.

Return math is NOT re-implemented: it calls research_theme._event_returns (the price-facts-v2 "since" window:
base = last close with bar time <= decided_as_of, end = last close with bar time <= horizon) and computes
rel = round((sym - bench) * 100, 2) on unrounded fractions, exactly like research_theme.stats().
Needs the Poseidon worktree on PYTHONPATH because research_theme imports Poseidon.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger_schema
from research_theme import _event_returns, content_sha256, load_bars

SETTLE_ALGO = "settle-v1"
FLAT_BAND_PP = 1.0  # |rel_return_pct| <= band is "inside the band"
BASE_TOLERANCE = 1e-6  # CSV base close vs frozen manifest base close
DEFAULT_BENCHMARK = "0050"
# Stance -> expected direction of the RELATIVE return. Existing revisions (2026-10-20) only use "watch".
STANCE_DIRECTION = {
    "long": 1,
    "bullish": 1,
    "buy": 1,
    "overweight": 1,
    "accumulate": 1,
    "short": -1,
    "bearish": -1,
    "sell": -1,
    "underweight": -1,
    "avoid": -1,
    "reduce": -1,
    "watch": 0,
    "hold": 0,
    "neutral": 0,
}
OUTCOMES = ("hit", "miss", "flat", "unsettled", "superseded")

COLUMNS = (
    "revision_id",
    "scope_key",
    "symbol",
    "benchmark",
    "stance",
    "direction",
    "decided_as_of",
    "horizon_time",
    "horizon_bar_time",
    "settle_as_of",
    "settled_at",
    "settle_algorithm",
    "bars_sha256",
    "abs_return_pct",
    "bench_return_pct",
    "rel_return_pct",
    "outcome",
    "detail",
    "invalidation_flag",
    "note",
    "flagged_at",
    "outcome_before_supersede",
    "superseded_by",
    "content_sha256",
)
# Added 2026-10-21; omitted from the row hash while NULL so rows written before them keep their content_sha256.
SUPERSEDE_COLUMNS = ("outcome_before_supersede", "superseded_by")
_ISO_PREFIX = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(Z|[+-]\d{2}:?\d{2})?)?")


def utcnow():
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0)


def iso(ts):
    return None if ts is None else pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def parse_ts(text):
    ts = pd.Timestamp(text)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def parse_horizon(next_review_at):
    """Leading ISO time of the free-text next_review_at. Date-only -> end of that UTC day; no offset -> UTC. None if absent."""
    m = _ISO_PREFIX.match(str(next_review_at or ""))
    if not m:
        return None
    day, clock, tz = m.groups()
    if clock is None:
        return parse_ts(day + "T23:59:59Z")
    return parse_ts("%sT%s%s" % (day, clock, tz or "Z"))


def direction_of(stance):
    return STANCE_DIRECTION.get(str(stance or "").strip().lower())


def classify(direction, rel_pct, band=FLAT_BAND_PP):
    """settle-v1 outcome rules (SETTLEMENT.md):
    directional (+1/-1): d*rel > band -> hit; d*rel < -band -> miss; otherwise flat.
    neutral (0, e.g. watch): |rel| <= band -> hit (no relative move the stance sat out); |rel| > band -> miss."""
    if direction == 0:
        return "hit" if abs(rel_pct) <= band else "miss"
    signed = direction * rel_pct
    return "hit" if signed > band else "miss" if signed < -band else "flat"


# ---------------------------------------------------------------- ledger access
def connect(path):
    return ledger_schema.connect(path)


def _rid(x):
    return str(x).replace("-", "").lower() if x else None


def completed_revisions(db):
    q = """SELECT r.id, r.scope_key, r.research_json, r.created_at, m.payload_json FROM research_revisions r
           JOIN data_manifests m ON m.id = r.manifest_id WHERE r.status = 'completed' ORDER BY r.created_at, r.id"""
    out = []
    for row in db.execute(q):
        research = json.loads(row["research_json"])
        manifest = json.loads(row["payload_json"])
        out.append(
            {
                "id": row["id"],
                "scope_key": row["scope_key"],
                "research": research,
                "manifest": manifest,
                "decided_as_of": research.get("as_of") or manifest.get("as_of"),
                "horizon": parse_horizon(research.get("next_review_at")),
                "stance": research.get("stance"),
                "supersedes": _rid(research.get("supersedes_revision_id") or research.get("supersedes")),
                "created_at": str(row["created_at"]),
            }
        )
    return out


def supersede_map(revs):
    """{superseded revision id: id of the first completed revision (in ledger order) that names it and was not created
    before it}. created_at ties are allowed: SQLite CURRENT_TIMESTAMP has 1 s resolution, and research_theme.py only accepts
    a pointer to a revision that was already completed. Pointers to unknown or earlier-created revisions are ignored."""
    by_id = {r["id"]: r for r in revs}
    out = {}
    for r in revs:
        old = r["supersedes"]
        if old in by_id and old != r["id"] and old not in out and by_id[old]["created_at"] <= r["created_at"]:
            out[old] = r["id"]
    return out


def subject(rev, benchmark_default):
    """(symbols, benchmark). Single-symbol scope = request symbol; otherwise the equal-weight basket of every frozen
    ohlcv symbol except the benchmark. Benchmark = price_facts payload benchmark if frozen, else the CLI default."""
    ev = rev["manifest"].get("evidence", [])
    bench = (
        next(
            (e["payload"].get("benchmark") for e in ev if e["kind"] == "price_facts" and e["payload"].get("benchmark")),
            None,
        )
        or benchmark_default
    )
    sym = rev["manifest"].get("symbol")
    if sym:
        return [sym], bench
    return [e["payload"]["symbol"] for e in ev if e["kind"] == "ohlcv" and e["payload"]["symbol"] != bench], bench


def frozen_close(rev, symbol, at):
    """Close of the frozen manifest ohlcv bar at time `at` (None if that symbol/bar was not frozen)."""
    for e in rev["manifest"].get("evidence", []):
        if e["kind"] == "ohlcv" and e["payload"]["symbol"] == symbol:
            for b in e["payload"]["bars"]:
                if parse_ts(b["time"]) == at:
                    return float(b["close"])
    return None


def row_hash(row):
    return content_sha256(
        {
            k: row.get(k)
            for k in COLUMNS
            if k != "content_sha256" and not (k in SUPERSEDE_COLUMNS and row.get(k) is None)
        }
    )


def get_row(db, revision_id):
    r = db.execute("SELECT * FROM research_settlements WHERE revision_id = ?", (revision_id,)).fetchone()
    return dict(r) if r else None


def put_row(db, row):
    row["content_sha256"] = row_hash(row)
    db.execute(
        "INSERT OR REPLACE INTO research_settlements (%s) VALUES (%s)"
        % (",".join(COLUMNS), ",".join("?" * len(COLUMNS))),
        [row.get(k) for k in COLUMNS],
    )


# ---------------------------------------------------------------- settle
def compute(rev, bars, settle_as_of, benchmark_default):
    """Returns the measured part of a settlement row; outcome 'unsettled' + detail when it cannot be measured."""
    symbols, bench = subject(rev, benchmark_default)
    direction = direction_of(rev["stance"])
    res = {
        "symbol": ",".join(symbols) or None,
        "benchmark": bench,
        "direction": direction,
        "horizon_bar_time": None,
        "bars_sha256": None,
        "abs_return_pct": None,
        "bench_return_pct": None,
        "rel_return_pct": None,
    }
    unsettled = lambda why: dict(res, outcome="unsettled", detail=why)
    if direction is None:
        return unsettled("unknown stance %r (not in STANCE_DIRECTION)" % rev["stance"])
    if not symbols:
        return unsettled("no subject symbol (no request symbol and no non-benchmark ohlcv in the manifest)")
    decided, horizon = parse_ts(rev["decided_as_of"]), rev["horizon"]
    fractions, used, horizon_bar = {}, {}, None
    for s in symbols + [bench]:
        if s not in bars:
            return unsettled("no bars for %s in the csv" % s)
        c = bars[s]["close"]
        c = c[c.index <= settle_as_of]
        before, window = c[c.index <= decided], c[(c.index > decided) & (c.index <= horizon)]
        if before.empty:
            return unsettled("%s: no bar at or before decided_as_of" % s)
        if window.empty:
            return unsettled("%s: no bar after decided_as_of up to the horizon" % s)
        last = window.index[-1]
        if not (c.index > horizon).any() and last.date() != horizon.date():
            return unsettled("%s: csv does not reach the horizon (last bar %s)" % (s, iso(c.index[-1])))
        base_t = before.index[-1]
        fz = frozen_close(rev, s, base_t)
        if fz is not None and abs(fz - float(before.iloc[-1])) > BASE_TOLERANCE:
            return unsettled(
                "%s: csv base close %s != frozen manifest close %s at %s (adjusted series?)"
                % (s, float(before.iloc[-1]), fz, iso(base_t))
            )
        if fz is None:
            res.setdefault("_unverified", []).append(s)
        fractions[s] = _event_returns(c[c.index <= horizon], rev["decided_as_of"])["since"]
        used[s] = [[iso(t), float(v)] for t, v in c[(c.index >= base_t) & (c.index <= horizon)].items()]
        horizon_bar = max(horizon_bar, last) if horizon_bar is not None else last
    sym_frac = sum(fractions[s] for s in symbols) / len(symbols)
    rel = round((sym_frac - fractions[bench]) * 100, 2)
    unverified = res.pop("_unverified", [])
    detail = "base close not cross-checked (not frozen): %s" % ",".join(unverified) if unverified else None
    if len(symbols) > 1:
        detail = "; ".join(x for x in ("equal-weight basket of %d symbols" % len(symbols), detail) if x)
    return dict(
        res,
        horizon_bar_time=iso(horizon_bar),
        bars_sha256=content_sha256(used),
        abs_return_pct=round(sym_frac * 100, 2),
        bench_return_pct=round(fractions[bench] * 100, 2),
        rel_return_pct=rel,
        outcome=classify(direction, rel),
        detail=detail,
    )


def _supersede(row, by):
    """Mark a row superseded: outcome_before_supersede keeps the measured outcome (None if never measured)."""
    if by:
        if row["outcome"] != "superseded":
            row["outcome_before_supersede"] = row["outcome"] if row.get("settle_as_of") else None
        row.update(outcome="superseded", superseded_by=by)
    return row


def cmd_settle(a):
    settle_as_of = parse_ts(a.as_of) if a.as_of else pd.Timestamp(utcnow())
    bars = load_bars(a.csv)
    db = connect(a.ledger)
    log = []
    revs = completed_revisions(db)
    sup_by = supersede_map(revs)
    for rev in revs:
        rid, by = rev["id"], sup_by.get(rev["id"])
        due = rev["horizon"] is not None and rev["horizon"] <= settle_as_of
        old = get_row(db, rid)
        if not due and not by:
            log.append(
                (
                    rid,
                    "skip: next_review_at has no leading ISO time"
                    if rev["horizon"] is None
                    else "pending until %s" % iso(rev["horizon"]),
                )
            )
            continue
        if old and not a.force:
            if by and old["outcome"] not in ("unsettled", "superseded"):
                put_row(db, _supersede(old, by))
                log.append((rid, "superseded by %s (was %s)" % (by, old["outcome_before_supersede"])))
                continue
            if old["outcome"] not in ("unsettled", "superseded") or (
                old["outcome"] == "superseded"
                and (not due or old["outcome_before_supersede"] not in (None, "unsettled"))
            ):
                log.append((rid, "no-op: already %s (use --force)" % old["outcome"]))
                continue
        if due:
            m = compute(rev, bars, settle_as_of, a.benchmark)
            measured = True
        else:
            symbols, bench = subject(rev, a.benchmark)
            measured = False
            m = {
                "symbol": ",".join(symbols) or None,
                "benchmark": bench,
                "direction": direction_of(rev["stance"]),
                "horizon_bar_time": None,
                "bars_sha256": None,
                "abs_return_pct": None,
                "bench_return_pct": None,
                "rel_return_pct": None,
                "outcome": "unsettled",
                "detail": "superseded before its horizon %s" % iso(rev["horizon"])
                if rev["horizon"] is not None
                else "superseded; next_review_at has no leading ISO time",
            }
        row = {
            "revision_id": rid,
            "scope_key": rev["scope_key"],
            "stance": rev["stance"],
            "decided_as_of": rev["decided_as_of"],
            "horizon_time": iso(rev["horizon"]),
            "settle_as_of": iso(settle_as_of) if measured else None,
            "settled_at": iso(utcnow()),
            "settle_algorithm": SETTLE_ALGO,
            **m,
            # the human flag survives re-settlement, including --force
            "invalidation_flag": old["invalidation_flag"] if old else 0,
            "note": old["note"] if old else None,
            "flagged_at": old["flagged_at"] if old else None,
            "outcome_before_supersede": None,
            "superseded_by": None,
        }
        put_row(db, _supersede(row, by))
        log.append(
            (
                rid,
                "%s%s rel=%s%s%s"
                % (
                    "re-settled " if old else "",
                    row["outcome"],
                    row["rel_return_pct"],
                    (" (was %s, superseded by %s)" % (row["outcome_before_supersede"], by)) if by else "",
                    (" (%s)" % row["detail"]) if row["detail"] else "",
                ),
            )
        )
    db.commit()
    for rid, msg in log:
        print("%s  %s" % (rid, msg))


# ---------------------------------------------------------------- flag
def cmd_flag(a):
    a.revision = a.revision.replace("-", "").lower()  # accept the dashed form printed in revision JSON
    db = connect(a.ledger)
    rev = next((r for r in completed_revisions(db) if r["id"] == a.revision), None)
    if rev is None:
        raise SystemExit("revision %s is not a completed revision in %s" % (a.revision, a.ledger))
    if not a.note.strip():
        raise SystemExit("--note must be non-empty")
    row = get_row(db, a.revision) or {
        "revision_id": rev["id"],
        "scope_key": rev["scope_key"],
        "symbol": None,
        "benchmark": None,
        "stance": rev["stance"],
        "direction": direction_of(rev["stance"]),
        "decided_as_of": rev["decided_as_of"],
        "horizon_time": iso(rev["horizon"]),
        "horizon_bar_time": None,
        "settle_as_of": None,
        "settled_at": None,
        "settle_algorithm": SETTLE_ALGO,
        "bars_sha256": None,
        "abs_return_pct": None,
        "bench_return_pct": None,
        "rel_return_pct": None,
        "outcome": "unsettled",
        "detail": "flagged before settlement",
        "outcome_before_supersede": None,
        "superseded_by": None,
    }
    row.update(invalidation_flag=0 if a.clear else 1, note=a.note.strip(), flagged_at=iso(utcnow()))
    put_row(db, row)
    db.commit()
    print(
        "%s  invalidation_flag=%d outcome=%s content_sha256=%s"
        % (a.revision, row["invalidation_flag"], row["outcome"], row["content_sha256"])
    )


# ---------------------------------------------------------------- report / pending
def _view(db, as_of):
    rows = {r["revision_id"]: dict(r) for r in db.execute("SELECT * FROM research_settlements")}
    revs = completed_revisions(db)
    sup_by = supersede_map(revs)
    out = []
    for rev in revs:
        r = rows.get(rev["id"])
        if r is None and sup_by.get(rev["id"]):
            r = {
                "revision_id": rev["id"],
                "scope_key": rev["scope_key"],
                "symbol": rev["manifest"].get("symbol"),
                "stance": rev["stance"],
                "decided_as_of": rev["decided_as_of"],
                "horizon_time": iso(rev["horizon"]),
                "outcome": "superseded",
                "outcome_before_supersede": None,
                "superseded_by": sup_by[rev["id"]],
                "rel_return_pct": None,
                "abs_return_pct": None,
                "invalidation_flag": 0,
                "note": None,
                "detail": "superseded; not settled yet (run settle)",
            }
        elif r is None:
            state = "pending" if rev["horizon"] is None or rev["horizon"] > as_of else "due"
            r = {
                "revision_id": rev["id"],
                "scope_key": rev["scope_key"],
                "symbol": rev["manifest"].get("symbol"),
                "stance": rev["stance"],
                "decided_as_of": rev["decided_as_of"],
                "horizon_time": iso(rev["horizon"]),
                "outcome": "unsettled",
                "rel_return_pct": None,
                "abs_return_pct": None,
                "invalidation_flag": 0,
                "note": None,
                "detail": {"pending": "awaiting horizon", "due": "horizon passed; run settle"}[state]
                if rev["horizon"] is not None
                else "next_review_at has no leading ISO time",
            }
        out.append(r)
    return out


def _fmt(x):
    return "" if x is None else "%+.2f" % x


def _group_table(rows, key, title):
    groups = {}
    for r in rows:
        groups.setdefault(r[key] or "-", []).append(r)
    lines = [
        "### %s" % title,
        "",
        "| %s | n | hit | miss | flat | unsettled | flagged | mean rel %% (settled) | superseded | hit rate (excl. superseded) |"
        % key,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g, rs in sorted(groups.items()):
        c = {o: sum(1 for r in rs if r["outcome"] == o) for o in OUTCOMES}
        rel = [
            r["rel_return_pct"]
            for r in rs
            if r["outcome"] not in ("unsettled", "superseded") and r["rel_return_pct"] is not None
        ]
        scored = c["hit"] + c["miss"] + c["flat"]
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %d | %s | %d | %s |"
            % (
                g,
                len(rs),
                c["hit"],
                c["miss"],
                c["flat"],
                c["unsettled"],
                sum(1 for r in rs if r["invalidation_flag"]),
                _fmt(sum(rel) / len(rel)) if rel else "n/a",
                c["superseded"],
                "%d/%d" % (c["hit"], scored) if scored else "n/a",
            )
        )
    return lines


def render_report(db, as_of):
    rows = _view(db, as_of)
    lines = [
        "# Research settlement report",
        "",
        "- as_of: %s" % iso(as_of),
        "- algorithm: %s (flat band ±%.1f pp relative)" % (SETTLE_ALGO, FLAT_BAND_PP),
        "- revisions: %d completed, %d settled, %d superseded (excluded from hit counts, hit rate and mean rel)"
        % (
            len(rows),
            sum(1 for r in rows if r["outcome"] not in ("unsettled", "superseded")),
            sum(1 for r in rows if r["outcome"] == "superseded"),
        ),
        "",
    ]
    lines += _group_table(rows, "stance", "By stance") + [""] + _group_table(rows, "scope_key", "By scope") + [""]
    lines += [
        "### Per revision",
        "",
        "| revision | scope_key | symbol | stance | decided_as_of | horizon | abs % | rel % | outcome | flag | detail / note | superseded_by |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        txt = "; ".join(x for x in (r.get("detail"), ("flag: " + r["note"]) if r.get("note") else None) if x)
        outcome = r["outcome"] + (
            " (was %s)" % r["outcome_before_supersede"]
            if r["outcome"] == "superseded" and r.get("outcome_before_supersede")
            else ""
        )
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r["revision_id"][:8],
                r["scope_key"],
                r.get("symbol") or "",
                r["stance"],
                r["decided_as_of"],
                r.get("horizon_time") or "",
                _fmt(r.get("abs_return_pct")),
                _fmt(r["rel_return_pct"]),
                outcome,
                "yes" if r["invalidation_flag"] else "",
                txt.replace("|", "/"),
                (r.get("superseded_by") or "")[:8],
            )
        )
    return "\n".join(lines) + "\n"


def cmd_report(a):
    md = render_report(connect(a.ledger), parse_ts(a.as_of) if a.as_of else pd.Timestamp(utcnow()))
    if a.out:
        Path(a.out).write_text(md)
        print("wrote %s" % a.out)
    else:
        sys.stdout.write(md)


def cmd_pending(a):
    as_of = parse_ts(a.as_of) if a.as_of else pd.Timestamp(utcnow())
    rows = _view(connect(a.ledger), as_of)
    waiting = [
        r for r in rows if r["outcome"] == "unsettled" and r.get("horizon_time") and parse_ts(r["horizon_time"]) > as_of
    ]
    due = [r for r in rows if r["outcome"] == "unsettled" and r not in waiting]
    print("# as_of %s" % iso(as_of))
    print("## awaiting horizon (%d)" % len(waiting))
    for r in sorted(waiting, key=lambda r: r["horizon_time"]):
        print("%s  %s  stance=%s  horizon=%s" % (r["revision_id"], r["scope_key"], r["stance"], r["horizon_time"]))
    print("## due or not settleable yet (%d)" % len(due))
    for r in due:
        print(
            "%s  %s  stance=%s  horizon=%s  %s"
            % (r["revision_id"], r["scope_key"], r["stance"], r.get("horizon_time"), r.get("detail") or "")
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("settle")
    s.add_argument("--ledger", required=True)
    s.add_argument("--csv", required=True)
    s.add_argument("--as-of")
    s.add_argument("--force", action="store_true")
    s.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    f = sub.add_parser("flag")
    f.add_argument("--ledger", required=True)
    f.add_argument("--revision", required=True)
    f.add_argument("--note", required=True)
    f.add_argument("--clear", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--ledger", required=True)
    r.add_argument("--as-of")
    r.add_argument("--out")
    p = sub.add_parser("pending")
    p.add_argument("--ledger", required=True)
    p.add_argument("--as-of")
    a = ap.parse_args(argv)
    {"settle": cmd_settle, "flag": cmd_flag, "report": cmd_report, "pending": cmd_pending}[a.cmd](a)


if __name__ == "__main__":
    main()
