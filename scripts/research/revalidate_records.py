#!/usr/bin/env python3
"""Re-check the narrative records already in a ledger under today's record rules (owner decision 2026-10-31).

Read-only: the ledger is opened with SQLite mode=ro, nothing is written, deleted or re-hashed. Each record is run
through record_rules.record_problems as narrative_ledger.add_record runs it today (check_period=False; a default
resolves_on, set by the program, counts as none) and every record that breaks a rule is printed, one tab-separated
line each:

  narrative_id  origin_kind  origin_ref  status  tags  messages

Records still held for a pending proposal (held_records, status held) are checked the same way and printed with
origin_kind `held`: add_record runs these rules on them when they are released.
A summary line follows on stderr. The owner decides what to do with a flagged record; this script never does.
Run it against a COPY of the runner ledger, never the live file:

  cp ~/research-runner/state/ledger.sqlite /tmp/ledger-copy.sqlite
  PYTHONPATH=src python scripts/research/revalidate_records.py --ledger /tmp/ledger-copy.sqlite [--origin revision]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import record_rules

STATUS_KINDS = ("created", "status", "override")


def flagged(db: sqlite3.Connection, origin: str | None = None) -> list[dict]:
    """[{narrative_id, origin_kind, origin_ref, status, problems}] for every record that breaks a record rule."""
    sql = "SELECT * FROM narrative_records" + (" WHERE origin_kind = ?" if origin else "") + " ORDER BY narrative_id"
    out = []
    for r in db.execute(sql, (origin,) if origin else ()):
        resolves_on = None if r["resolves_on_default"] else r["resolves_on"]
        problems = record_rules.record_problems(
            r["claim"], r["quote"], r["claim_type"], resolves_on, check_period=False
        )
        if not problems:
            continue
        last = db.execute(
            "SELECT to_status FROM narrative_events WHERE narrative_id = ? AND kind IN (?, ?, ?) ORDER BY seq DESC "
            "LIMIT 1",
            (r["narrative_id"], *STATUS_KINDS),
        ).fetchone()
        out.append(
            {
                "narrative_id": r["narrative_id"],
                "origin_kind": r["origin_kind"],
                "origin_ref": r["origin_ref"],
                "status": last["to_status"] if last else None,
                "problems": problems,
            }
        )
    return out


def flagged_held(db: sqlite3.Connection) -> list[dict]:
    """The held records (status held) that add_record would refuse on release, in flagged()'s shape."""
    out = []
    for r in db.execute(
        "SELECT narrative_id, record_json FROM held_records WHERE status = 'held' ORDER BY narrative_id"
    ):
        rec = json.loads(r["record_json"])
        problems = record_rules.record_problems(
            rec.get("claim") or "",
            rec.get("quote") or "",
            rec.get("claim_type"),
            rec.get("resolves_on"),
            check_period=False,
        )
        if problems:
            out.append(
                {
                    "narrative_id": r["narrative_id"],
                    "origin_kind": "held",
                    "origin_ref": rec.get("origin_ref"),
                    "status": "held",
                    "problems": problems,
                }
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Flag ledger narrative records that break today's record rules (read-only)."
    )
    ap.add_argument("--ledger", required=True, help="a COPY of the runner's ledger.sqlite")
    ap.add_argument("--origin", choices=("material", "revision", "owner"), help="only records of this origin")
    a = ap.parse_args(argv)
    path = Path(a.ledger).resolve()
    if not path.is_file():
        sys.exit(f"no ledger at {path}")
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        total = db.execute(
            "SELECT count(*) FROM narrative_records" + (" WHERE origin_kind = ?" if a.origin else ""),
            (a.origin,) if a.origin else (),
        ).fetchone()[0]
        rows = flagged(db, a.origin)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        held = flagged_held(db) if a.origin is None and "held_records" in tables else []
    finally:
        db.close()
    for x in rows + held:
        tags = ",".join(sorted({p.tag for p in x["problems"]}))
        messages = " | ".join(p.message for p in x["problems"])
        print(
            "\t".join(
                str(v) for v in (x["narrative_id"], x["origin_kind"], x["origin_ref"], x["status"], tags, messages)
            )
        )
    print(
        f"{len(rows)} of {total} records and {len(held)} held records flagged (read-only; nothing changed)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
