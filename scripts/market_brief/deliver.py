#!/usr/bin/env python3
"""Deliver queued briefs to macminim4 through the research bridge verb `brief <kind> <YYYY-MM-DD>` (stdlib only).

Each <outbox>/<date>-<kind>.json is {kind, date, filename, summary, report_md}. The bridge gets only the two validated
arguments on its command line; {filename, summary, report_md} goes on stdin as JSON.
A failure notice <outbox>/<date>-<kind>.failed.json is {verb: "brief-failed", kind, date, reason} and is sent as
`brief-failed <kind> <date>` with the one-line reason on stdin. A 市場警示 <outbox>/<date>-alert-<tw|us>.json is
{verb: "brief-alert", market, date, text} and is sent as `brief-alert <market> <date>` with the text on stdin.
Exit 0 -> entry moved to <outbox>/sent/; exit 2 (refused) -> <outbox>/refused/ with the reason logged;
anything else -> kept for the next run.

Usage: deliver.py --outbox ~/market-brief/outbox [--bridge "ssh research-bridge"]
Exit code 0 when everything was delivered (or nothing queued), 4 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

KINDS = ("tw-premarket", "tw-postmarket", "us-premarket", "us-postmarket")
ALERT_MARKETS = ("tw", "us")
DATE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
STDIN_CAP = 450000


def valid(entry):
    kind, date = entry.get("kind"), entry.get("date")
    if not isinstance(date, str) or not DATE.match(date):
        return False
    if entry.get("verb") == "brief-alert":
        text = entry.get("text")
        return (
            entry.get("market") in ALERT_MARKETS and isinstance(text, str) and text.startswith(f"**市場警示 {date}**")
        )
    if kind not in KINDS:
        return False
    if entry.get("verb") == "brief-failed":
        return isinstance(entry.get("reason"), str)
    if entry.get("verb") not in (None, "brief"):
        return False
    if entry.get("filename") != f"{date}-{kind}.md":
        return False
    if entry.get("telegram_md") is not None and not isinstance(entry["telegram_md"], str):
        return False
    return isinstance(entry.get("report_md"), str) and isinstance(entry.get("summary"), str)


def deliver_outbox(outbox, bridge, log=print, runner=subprocess.run):
    outbox = Path(outbox).expanduser()
    failed = 0
    for path in sorted(outbox.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            entry = {}
        if not valid(entry):
            log(f"skipping malformed outbox entry {path.name}")
            failed += 1
            continue
        verb = entry.get("verb") or "brief"
        target = entry["market"] if verb == "brief-alert" else entry["kind"]
        if verb == "brief-failed":
            stdin = " ".join(entry["reason"].split())[:500]
        elif verb == "brief-alert":
            stdin = entry["text"]
        else:
            payload = {"filename": entry["filename"], "summary": entry["summary"], "report_md": entry["report_md"]}
            if entry.get("telegram_md"):
                payload["telegram_md"] = entry["telegram_md"]
            stdin = json.dumps(payload, ensure_ascii=False)
        if len(stdin.encode("utf-8")) > STDIN_CAP:
            log(f"outbox entry {path.name} exceeds the bridge stdin cap; not sent")
            failed += 1
            continue
        cmd = [*shlex.split(bridge), verb, target, entry["date"]]
        try:
            p = runner(cmd, input=stdin, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as e:
            log(f"bridge {verb} {target} {entry['date']} failed to run: {type(e).__name__}")
            failed += 1
            continue
        if p.returncode == 0:
            (outbox / "sent").mkdir(exist_ok=True)
            path.replace(outbox / "sent" / path.name)
            log(f"delivered {verb} {target} {entry['date']}: {p.stdout.strip()[:200]}")
        elif p.returncode == 2:
            (outbox / "refused").mkdir(exist_ok=True)
            path.replace(outbox / "refused" / path.name)
            log(f"bridge refused {verb} {target} {entry['date']}: {p.stderr.strip()[:300]}")
            failed += 1
        else:
            log(f"bridge {verb} {target} {entry['date']} exit {p.returncode}: {p.stderr.strip()[:300]}")
            failed += 1
    return failed


def resend(kind, date, telegram_md, bridge, log=print, runner=subprocess.run):
    """`brief-resend <kind> <date>` with {telegram_md} on stdin: text only, for a brief the bridge already holds."""
    if kind not in KINDS or not DATE.match(date):
        raise ValueError("bad kind/date")
    cmd = [*shlex.split(bridge), "brief-resend", kind, date]
    stdin = json.dumps({"telegram_md": telegram_md}, ensure_ascii=False)
    try:
        p = runner(cmd, input=stdin, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"bridge brief-resend {kind} {date} failed to run: {type(e).__name__}")
        return False
    if p.returncode != 0:
        log(f"bridge brief-resend {kind} {date} exit {p.returncode}: {p.stderr.strip()[:300]}")
        return False
    log(f"resent brief text {kind} {date}: {p.stdout.strip()[:200]}")
    return True


def mirror_sync(bridge, log=print, runner=subprocess.run):
    """`brief-mirror-sync` on the bridge (Nexus market-daily -> mirror for the brief server). Never raises."""
    try:
        p = runner([*shlex.split(bridge), "brief-mirror-sync"], input="", capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"bridge brief-mirror-sync failed to run: {type(e).__name__}")
        return False
    if p.returncode != 0:
        log(f"bridge brief-mirror-sync exit {p.returncode}: {p.stderr.strip()[:300]}")
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outbox", required=True)
    ap.add_argument("--bridge", default="ssh research-bridge")
    a = ap.parse_args(argv)
    return 4 if deliver_outbox(a.outbox, a.bridge, log=lambda m: print(m, file=sys.stderr)) else 0


if __name__ == "__main__":
    sys.exit(main())
