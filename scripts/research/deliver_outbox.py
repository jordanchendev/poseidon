#!/usr/bin/env python3
"""Deliver runner.py --outbox reports to the agenda bridge (host side; stdlib only, Python 3.8+).

For each <outbox>/<agenda_id>.json ({verb: done|needs-review, args: [...], stdin}) run `<bridge> <verb> <args...>`
with stdin. Exit 0 -> file removed; exit 2 (refused) -> moved to <outbox>/refused/ with the reason logged; anything
else -> kept for the next run. Ids are re-validated here; nothing else is passed to the bridge.

Usage: deliver_outbox.py --outbox ~/research-runner/state/outbox [--bridge "ssh research-bridge"]
Exit code 0 when everything was delivered, 4 otherwise.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

AGENDA_ID = re.compile(r"^\d{8}-m?\d{1,3}$")
REVISION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def valid(entry):
    verb, args = entry.get("verb"), entry.get("args") or []
    if verb == "done":
        return len(args) == 2 and AGENDA_ID.match(args[0]) and REVISION_ID.match(args[1])
    if verb == "needs-review":
        return len(args) == 1 and AGENDA_ID.match(args[0])
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outbox", required=True)
    ap.add_argument("--bridge", default="ssh research-bridge")
    a = ap.parse_args(argv)
    outbox = Path(a.outbox).expanduser()
    failed = 0
    for path in sorted(outbox.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            entry = {}
        if not valid(entry):
            print(f"skipping malformed outbox entry {path.name}", file=sys.stderr)
            failed += 1
            continue
        cmd = shlex.split(a.bridge) + [entry["verb"]] + entry["args"]
        try:
            p = subprocess.run(cmd, input=entry.get("stdin", ""), capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(
                "bridge {} {} failed to run: {}".format(entry["verb"], entry["args"][0], type(e).__name__),
                file=sys.stderr,
            )
            failed += 1
            continue
        if p.returncode == 0:
            path.unlink()
            print("delivered {} {}".format(entry["verb"], " ".join(entry["args"])))
        elif p.returncode == 2:
            (outbox / "refused").mkdir(exist_ok=True)
            path.replace(outbox / "refused" / path.name)
            print(
                "bridge refused {} {}: {}".format(entry["verb"], entry["args"][0], p.stderr.strip()[:300]),
                file=sys.stderr,
            )
            failed += 1
        else:
            print(
                f"bridge {entry['verb']} {entry['args'][0]} exit {p.returncode}: {p.stderr.strip()[:300]}",
                file=sys.stderr,
            )
            failed += 1
    return 4 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
