"""The research runner's agenda bridge: the forced-command verbs on macminim4 (`ssh research-bridge`), the ids and caps
they accept, and the outbox variant for containers without an ssh client. Moved out of runner.py unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
from pathlib import Path

import intake as intake_mod
import narrative_decisions
from textcap import cap, one_line

log = logging.getLogger("research-runner")

REPORT_CAP = 400_000  # chars of rendered Markdown attached to the done report
AGENDA_ID = re.compile(r"^\d{8}-m?\d{1,3}$")
MATERIAL_ID = intake_mod.MATERIAL_ID
DECISION_ID = narrative_decisions.DECISION_ID
NOTICE_CAP = 1000  # narrative-ack message; the bridge keeps 1000 characters
REVISION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REASON_CAP = 580
SUMMARY_CAP = 300


# ------------------------------------------------------------------------------------------------ bridge
class BridgeError(Exception):
    pass


class Bridge:
    """Forced-command agenda bridge: `<cmd> pending`, `<cmd> done <id> <rev>` (tl_dr on stdin),
    `<cmd> needs-review <id>` (reason on stdin); materials: `material-pending`, `material-done <id>` (JSON on stdin),
    `material-needs-review <id>` (reason on stdin); owner decisions: `narrative-pending`,
    `narrative-ack <decision_id> applied|refused` (message on stdin). Exit code 2 = refused (stderr has the reason)."""

    def __init__(self, cmd, timeout=60):
        self.argv = shlex.split(cmd)
        self.timeout = timeout

    def _run(self, args, stdin=""):
        try:
            p = subprocess.run(
                [*self.argv, *args], input=stdin, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise BridgeError(f"bridge {args[0]} failed to run: {type(e).__name__}") from None
        if p.returncode == 2:
            raise BridgeError(f"bridge {args[0]} refused: {p.stderr.strip()[:300]}")
        if p.returncode != 0:
            raise BridgeError(f"bridge {args[0]} exit {p.returncode}: {p.stderr.strip()[:300]}")
        return p.stdout

    def _list(self, verb):
        out = self._run([verb])
        try:
            rows = json.loads(out)
        except ValueError:
            raise BridgeError(f"bridge {verb} returned non-JSON output") from None
        if not isinstance(rows, list):
            raise BridgeError(f"bridge {verb} returned {type(rows).__name__}, not a list")
        return rows

    def pending(self):
        return self._list("pending")

    def material_pending(self):
        return self._list("material-pending")

    def narrative_pending(self):
        return self._list("narrative-pending")

    def done(self, agenda_id, revision_id, summary, report_md=None, suggestions=(), theme_proposals=()):
        """stdin is the one-line summary, or JSON {summary, report_md, suggestions, theme_proposals} when a rendered
        report is attached, narrative_apply confirm has suggestions or the revision proposed themes (the bridge saves the
        report, sends it to Telegram as a document and appends one line per suggestion {id, narrative_id, line} and per
        theme proposal {id, line} to the notice)."""
        if not AGENDA_ID.match(agenda_id) or not REVISION_ID.match(revision_id):
            raise BridgeError(f"refusing to report malformed ids {agenda_id!r} / {revision_id!r}")
        line = one_line(summary, SUMMARY_CAP)
        if report_md and len(report_md) > REPORT_CAP:
            log.warning(
                "report for %s is %d chars (> %d); sent without attachment", agenda_id, len(report_md), REPORT_CAP
            )
            report_md = None
        if report_md or suggestions or theme_proposals:
            stdin = json.dumps(
                {
                    "summary": line,
                    **({"report_md": report_md} if report_md else {}),
                    **({"suggestions": list(suggestions)} if suggestions else {}),
                    **({"theme_proposals": list(theme_proposals)} if theme_proposals else {}),
                },
                ensure_ascii=False,
            )
        else:
            stdin = line + "\n"
        self._run(["done", agenda_id, revision_id], stdin)

    def material_done(self, material_id, payload):
        if not MATERIAL_ID.match(material_id):
            raise BridgeError(f"refusing to report malformed material id {material_id!r}")
        self._run(["material-done", material_id], json.dumps(payload, ensure_ascii=False))

    def material_needs_review(self, material_id, reason):
        if not MATERIAL_ID.match(material_id):
            raise BridgeError(f"refusing to report malformed material id {material_id!r}")
        self._run(["material-needs-review", material_id], cap(reason, REASON_CAP) + "\n")

    def narrative_ack(self, decision_id, result, message):
        if not DECISION_ID.match(decision_id) or result not in ("applied", "refused"):
            raise BridgeError(f"refusing to ack malformed decision {decision_id!r} {result!r}")
        self._run(["narrative-ack", decision_id, result], cap(message, NOTICE_CAP) + "\n")

    def needs_review(self, agenda_id, reason):
        if not AGENDA_ID.match(agenda_id):
            raise BridgeError(f"refusing to report malformed agenda id {agenda_id!r}")
        self._run(["needs-review", agenda_id], cap(reason, REASON_CAP) + "\n")


class OutboxBridge(Bridge):
    """For containers without an ssh client: every report verb is queued as <outbox>/<id>.json ({verb, args, stdin}) and
    delivered by deliver_outbox.py on the host. The list verbs still run the bridge command (or the runner reads the
    host's --agenda-json / --materials-json / --decisions-json). One file per agenda, material or decision id, so a
    re-report replaces the queued one."""

    LIST_VERBS = ("pending", "material-pending", "narrative-pending")

    def __init__(self, cmd, outbox):
        super().__init__(cmd)
        self.outbox = Path(outbox)

    def _run(self, args, stdin=""):
        if args[0] in self.LIST_VERBS:
            return super()._run(args, stdin)
        try:
            self.outbox.mkdir(parents=True, exist_ok=True)
            tmp = self.outbox / f".{args[1]}.json.tmp"
            tmp.write_text(json.dumps({"verb": args[0], "args": args[1:], "stdin": stdin}, ensure_ascii=False))
            tmp.replace(self.outbox / f"{args[1]}.json")
        except OSError as e:
            raise BridgeError(f"outbox write failed: {e}") from None
        return ""
