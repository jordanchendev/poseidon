"""Build or update the refdata datasets (idempotent).

    python -m poseidon.pipeline.refdata update [--root DIR] [--group futures|twse|cboe] [--full]
                                               [--seed-taifex DIR] [--seed-twse DIR] [--refresh-months N]
    python -m poseidon.pipeline.refdata verify [--root DIR]
    python -m poseidon.pipeline.refdata daily [--root DIR]      # update + freshness report (one JSON)

``update`` fetches every month not yet stored, re-fetches the last
``--refresh-months`` (default 2: current and previous month), rebuilds the
partitions whose raw bytes changed and the derived calendar/roll tables, and
prints one JSON line per dataset. Running it twice in a row changes nothing.

The daily job (schedule, freshness check, hooks for later datasets) is
:func:`poseidon.pipeline.refdata.daily.run_daily`, also available as
``python -m poseidon.pipeline.refdata daily``. A ``SourceBlockedError`` stops
it without retries (the next day's run fills the gap).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from poseidon.core.config import settings
from poseidon.pipeline.refdata.datasets import GROUPS, update_all
from poseidon.pipeline.refdata.store import RefdataError, RefdataStore


def _verify(store: RefdataStore) -> int:
    failures = 0
    for name in [name for group in GROUPS.values() for name in group]:
        if not store.exists(name):
            print(json.dumps({"dataset": name, "status": "missing"}))
            failures += 1
            continue
        try:
            rows = len(store.read(name))
        except RefdataError as error:
            print(json.dumps({"dataset": name, "status": "corrupt", "error": str(error)}))
            failures += 1
            continue
        manifest = store.manifest(name)
        print(
            json.dumps(
                {
                    "dataset": name,
                    "status": "ok",
                    "rows": rows,
                    "coverage": manifest["coverage"],
                    "content_sha256": manifest["content_sha256"],
                }
            )
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poseidon.pipeline.refdata")
    parser.add_argument("command", choices=["update", "verify", "daily"])
    parser.add_argument("--root", default=None, help="refdata root (default: settings.pipeline_refdata_root)")
    parser.add_argument("--group", action="append", choices=sorted(GROUPS), help="limit to a group (repeatable)")
    parser.add_argument("--full", action="store_true", help="rebuild every partition from the stored raw files")
    parser.add_argument("--refresh-months", type=int, default=2)
    parser.add_argument("--seed-taifex", type=Path, help="directory of previously fetched taifex_<P>_<YYYYMM>.csv")
    parser.add_argument("--seed-twse", type=Path, help="directory of previously fetched twse_<KIND>_<YYYYMM>.json")
    parser.add_argument("--code-version", default=settings.pipeline_code_version or "unknown")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = RefdataStore(args.root)
    if args.command == "verify":
        return _verify(store)
    if args.command == "daily":
        from poseidon.pipeline.refdata.daily import run_daily

        print(json.dumps(run_daily(store, code_version=args.code_version), sort_keys=True, default=str))
        return 0
    seeds = {key: value for key, value in (("taifex", args.seed_taifex), ("twse", args.seed_twse)) if value}
    reports = update_all(
        store,
        groups=args.group or list(GROUPS),
        refresh_months=args.refresh_months,
        full=args.full,
        code_version=args.code_version,
        seed_dirs=seeds,
    )
    for report in reports:
        print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
