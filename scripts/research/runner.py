#!/usr/bin/env python3
"""Server-side research runner (stormtrooper): one accepted agenda topic per invocation, no human in the loop.

Pipeline for the chosen row (design: aquarium local_dev/design/2026-10-21-research-runner-stormtrooper.md):
  assemble (assemble.py: source fetch, Thalassa OHLCV, TWSE kpi_series -> config.json)
  -> freeze/prepare (research_theme.py without --narrative: manifest frozen, revision prepared in the ledger)
  -> draft (draft.py: OpenRouter, RESEARCH_DRAFT_OPENROUTER_KEY)
  -> event_status (pre_event | post_event, decided by assemble from dates, never by the model) is written into the
     narrative with event_date
  -> gates (schema incl. Poseidon validate_research, narrative extras, quote-in-cited-evidence, claim_precheck strict
     numbers, lint_atomic (inference: multi-quantity allowed), GAAP basis, inference basis marker, pre_event catalysts,
     narrative_check.gates: [evidence] [theme] [check] [proposal]);
     refused -> ONE repair round with the refusal reasons -> gates again
  -> JEV jev-1.13 on every FACT claim (JEV_RESEARCH_OPENROUTER_KEY); every label must be `supports`.
     Inference claims are not sent to JEV (UMC 3de55971 precedent: a source never states an inference, so JEV would
     answer says_nothing); they stay gated by claim_precheck (scale/sign) and the verbatim-quote check.
  -> complete (research_theme.py --narrative: re-runs its own prechecks, completes, replay + immutability check)
  -> narrative_apply (narrative_check.apply, only with a narrative_snapshot): JEV judges each checked record against
     the cited fact claims and sweeps every shown record against all official excerpts; the rule table R1-R9 decides;
     narrative_checks rows always, status events only with --narrative-apply on. A failure here is recorded in
     run.json and never undoes the completed revision.
     With --narrative-apply confirm the rule's events become suggestions the done notice lists for the owner.
  -> bridge `done <agenda_id> <revision_id>` with the tl_dr on stdin.
Before the agenda topic, each invocation (not a dry run) applies the owner's narrative decisions (`narrative-pending`,
acked with `narrative-ack`) and takes in at most one material (`material-pending`, intake.py; reported with
`material-done` / `material-needs-review`). Intake writes narrative records and theme proposals, never a revision.
Any failure -> the revision stays prepared, bridge `needs-review <agenda_id>` with a short reason on stdin.
No escalation to another model.

State (all under --state-dir): ledger.sqlite (research_theme tables + runner_runs), runs/<agenda_id>/{config.json,
narrative.json, revision.json, run.json, bars.csv, sources/}, runs/material-<id>/{sources/material.txt,
intake.draft<n>.json, intake.json, run.json}, runner.log, runner.lock.

Exit codes: 0 done / needs_review handled / nothing to do / lock busy; 3 key file error; 4 a bridge report failed
(recorded, retried on the next run); 5 a bridge list call (`pending`, `material-pending`, `narrative-pending`) failed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import logging
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble as assemble_mod  # noqa: E402
import draft as draft_mod  # noqa: E402
import intake as intake_mod  # noqa: E402
import narrative_check  # noqa: E402
import narrative_ledger as nl  # noqa: E402
from claim_precheck import has_inference_basis  # noqa: E402

JEV_KEY_VAR = "JEV_RESEARCH_OPENROUTER_KEY"
JEV_MODEL = "typesafe/jev-1.13"
JEV_BASE = "https://openrouter.ai/api"
JEV_PRIMARY, JEV_FALLBACK = "/v1/systemone", "/alpha/decisions"
JEV_SOURCE_CAP = 12000
REPORT_CAP = 400_000  # chars of rendered Markdown attached to the done report
AGENDA_ID = re.compile(r"^\d{8}-m?\d{1,3}$")
MATERIAL_ID = intake_mod.MATERIAL_ID
DECISION_ID = narrative_check.DECISION_ID
NOTICE_CAP = 600
REVISION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REASON_CAP = 580
SUMMARY_CAP = 300
DEFAULT_BRIDGE = "ssh research-bridge"
EXIT_KEY, EXIT_BRIDGE_REPORT, EXIT_BRIDGE_PENDING = 3, 4, 5

log = logging.getLogger("research-runner")


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

    def done(self, agenda_id, revision_id, summary, report_md=None, suggestions=()):
        """stdin is the one-line summary, or JSON {summary, report_md, suggestions} when a rendered report is attached
        or narrative_apply confirm has suggestions (the bridge saves the report, sends it to Telegram as a document and
        appends one line per suggestion {id, narrative_id, line} to the notice)."""
        if not AGENDA_ID.match(agenda_id) or not REVISION_ID.match(revision_id):
            raise BridgeError(f"refusing to report malformed ids {agenda_id!r} / {revision_id!r}")
        line = one_line(summary, SUMMARY_CAP)
        if report_md and len(report_md) > REPORT_CAP:
            log.warning(
                "report for %s is %d chars (> %d); sent without attachment", agenda_id, len(report_md), REPORT_CAP
            )
            report_md = None
        if report_md or suggestions:
            stdin = json.dumps(
                {
                    "summary": line,
                    **({"report_md": report_md} if report_md else {}),
                    **({"suggestions": list(suggestions)} if suggestions else {}),
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


def one_line(text, n):
    return cap(" ".join(str(text or "").split()), n)


def cap(text, n):
    text = str(text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


# ------------------------------------------------------------------------------------------------ runner ledger
RUNS_DDL = """CREATE TABLE IF NOT EXISTS runner_runs (
  agenda_id TEXT PRIMARY KEY, status TEXT NOT NULL, revision_id TEXT, scope_key TEXT, reason TEXT,
  summary TEXT, reported INTEGER NOT NULL DEFAULT 0, run_dir TEXT, started_at TEXT, updated_at TEXT)"""


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(RUNS_DDL)
        self.db.commit()

    def get(self, agenda_id):
        r = self.db.execute("SELECT * FROM runner_runs WHERE agenda_id = ?", (agenda_id,)).fetchone()
        return dict(r) if r else None

    def put(self, agenda_id, **fields):
        fields["updated_at"] = now_iso()
        if self.get(agenda_id) is None:
            cols = ["agenda_id", *fields]
            self.db.execute(
                "INSERT INTO runner_runs ({}) VALUES ({})".format(", ".join(cols), ", ".join("?" * len(cols))),
                (agenda_id, *fields.values()),
            )
        else:
            self.db.execute(
                "UPDATE runner_runs SET {} WHERE agenda_id = ?".format(", ".join(f"{k} = ?" for k in fields)),
                (*fields.values(), agenda_id),
            )
        self.db.commit()

    def manifest_payload(self, manifest_sha256):
        r = self.db.execute(
            "SELECT payload_json FROM data_manifests WHERE content_sha256 = ?", (manifest_sha256,)
        ).fetchone()
        if r is None:
            raise RuntimeError(f"manifest {manifest_sha256[:12]} not found in ledger")
        v = r[0]
        return json.loads(v) if isinstance(v, (str, bytes)) else v

    def close(self):
        self.db.close()


def now_iso():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------ research_theme calls
class StageError(Exception):
    """A pipeline stage failed; `stage` and the message become the needs_review reason."""

    def __init__(self, stage, reason, details=None):
        super().__init__(reason)
        self.stage, self.reason, self.details = stage, reason, details or []


def research_theme(config_path, cfg, ledger_path, narrative_path=None, out_path=None, timeout=600):
    args = [
        sys.executable,
        str(HERE / "research_theme.py"),
        "--config",
        str(config_path),
        "--csv",
        cfg["runner_inputs"]["csv"],
        "--ledger",
        str(ledger_path),
    ]
    if cfg["runner_inputs"].get("kpi_json"):
        args += ["--kpi-json", cfg["runner_inputs"]["kpi_json"]]
    if narrative_path:
        args += ["--narrative", str(narrative_path), "--out", str(out_path)]
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if p.returncode != 0:
        lines = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
        msg = "\n".join(lines[-6:]) if lines else f"exit {p.returncode}"
        raise RuntimeError(msg)
    return json.loads(p.stdout)


# ------------------------------------------------------------------------------------------------ gates
NARRATIVE_LISTS = ("risks", "invalidation_conditions", "unknowns")
NARRATIVE_TEXTS = ("tl_dr", "thesis", "stance", "change_summary", "next_review_at")


def apply_event(narrative, cfg):
    """Write the runner's event_status / event_date into the drafted narrative (the model never decides them)."""
    event = cfg.get("event")
    if event and isinstance(narrative, dict):
        narrative["event_status"] = event["status"]
        if event.get("event_date"):
            narrative["event_date"] = event["event_date"]
    return narrative


def run_gates(narrative, payload, cfg):
    """All local checks; returns (reasons, report). An empty reasons list means the narrative may be completed."""
    import research_theme as rt

    from poseidon.decision_loop.research import validate_research

    reasons, report = [], {}
    # schema
    for k in NARRATIVE_TEXTS:
        if not isinstance(narrative.get(k), str) or not narrative[k].strip():
            reasons.append(f"[schema] {k} must be a non-empty string")
    if narrative.get("stance") not in draft_mod.STANCES:
        reasons.append("[schema] stance must be one of {}".format("|".join(draft_mod.STANCES)))
    for k in NARRATIVE_LISTS:
        v = narrative.get(k)
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            reasons.append(f"[schema] {k} must be a non-empty list of strings")
    claims = narrative.get("claims")
    if not isinstance(claims, list) or not all(isinstance(c, dict) for c in claims):
        reasons.append("[schema] claims must be a list of objects")
        return reasons, report
    body = dict(narrative, scope_key=cfg["scope_key"], as_of=payload["as_of"])
    if cfg.get("symbol"):
        body["symbol"] = cfg["symbol"]
    try:
        validate_research(body, types.SimpleNamespace(scope_key=cfg["scope_key"]), payload)
    except Exception as e:  # ValidationError and malformed shapes
        reasons.append(f"[schema] {e}")
    for fn in (
        lambda: rt.validate_narrative_extras(narrative, payload["as_of"]),
        lambda: rt.validate_variant_perception(narrative, payload["evidence"]),
    ):
        try:
            fn()
        except SystemExit as e:
            reasons.append(f"[schema] {e}")
        except Exception as e:
            reasons.append(f"[schema] {type(e).__name__}: {e}")
    # quote-in-cited-evidence (the document excerpts themselves were verified against the source text at freeze)
    by_id = {e["id"]: e for e in payload["evidence"]}
    quotes = []
    for n, c in enumerate(claims):
        ids = c.get("evidence_ids") if isinstance(c.get("evidence_ids"), list) else []
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            reasons.append(f"[evidence] claims[{n}] cites unknown evidence ids {unknown}")
        cited = [by_id[i] for i in ids if i in by_id]
        q = c.get("quote")
        ok = isinstance(q, str) and draft_mod.quote_in_evidence(q, cited)
        quotes.append(ok)
        if not ok:
            reasons.append(
                f"[quote] claims[{n}] quote is missing or not a verbatim substring of the cited evidence {ids}"
            )
    report["quote_ok"] = sum(quotes)
    # narrative ledger (design 2026-10-29 §4.4): [evidence] [theme] [check] [proposal]
    reasons += narrative_check.gates(narrative, payload, cfg)
    # inference claims must state their basis (owner decision 2026-10-21)
    for n, c in enumerate(claims):
        if c.get("kind") == "inference" and not has_inference_basis(c.get("text", "")):
            reasons.append(
                f"[basis] claims[{n}] is an inference but its text states no basis: say which numbers/evidence it "
                "rests on (依…推算 / 依…判斷 / 由 A 與 B 推算) or that it is an expectation (預期 / 可能), or make it a fact"
            )
    # pre_event: the event date (when known and not past) must be on the catalyst calendar
    event = cfg.get("event") or {}
    if narrative.get("event_status") == "pre_event" and (event.get("event_date") or "") >= payload["as_of"][:10]:
        dates = [c.get("date") for c in narrative.get("catalysts") or [] if isinstance(c, dict)]
        if event["event_date"] not in dates:
            reasons.append(
                "[event] pre_event research: catalysts must include the event date {}".format(event["event_date"])
            )
    # claim_precheck strict (numbers: fact any mismatch, inference scale/sign), lint_atomic, GAAP basis
    try:
        report["claim_precheck"] = rt.precheck_claims(narrative, payload["evidence"], allow_non_atomic=False)
    except SystemExit as e:
        for line in str(e).splitlines()[1:]:
            line = line.strip()
            tag = "atomic" if "non-atomic" in line else "gaap_basis" if "GAAP" in line else "precheck"
            reasons.append(f"[{tag}] {line}")
    except Exception as e:
        reasons.append(f"[precheck] {type(e).__name__}: {e}")
    return reasons, report


# ------------------------------------------------------------------------------------------------ JEV
class Jev:
    """JEV calls on one endpoint: the primary path, switched once to the alpha path when this client's first call gets a
    404. Sums usage over its calls. Used for the fact claims and for narrative_apply (design 2026-10-29 §4.5)."""

    SOURCE_CAP = JEV_SOURCE_CAP

    def __init__(self, key, model=JEV_MODEL, transport=None, path=JEV_PRIMARY):
        self.key, self.model, self.transport, self.path = key, model, transport, path
        self.calls, self.usage_total = 0, {}

    @staticmethod
    def source_text(claim, cited):
        """The claim's quote, then the full texts of its cited evidence, capped at SOURCE_CAP."""
        parts = []
        if isinstance(claim.get("quote"), str) and claim["quote"].strip():
            parts.append("Quoted: " + claim["quote"].strip())
        for item in cited:
            parts.append("[{}]\n{}".format(item["id"], "\n".join(draft_mod.evidence_texts(item))))
        return cap("\n\n".join(parts), JEV_SOURCE_CAP)

    def judge(self, claim, source_text, locator, symbol, as_of):
        """One citation_relation judgement -> {label, confidence, probabilities, latency_ms}. Raises draft.PostError."""
        from jev_eval_control_set import build_request

        case = {"symbol": symbol, "claim": claim, "source": {"text": source_text, "locator": locator}}
        req = build_request(case, self.model, as_of)
        kw = {"transport": self.transport} if self.transport else {}
        t0 = time.monotonic()
        try:
            resp = draft_mod.post_json(JEV_BASE + self.path, self.key, req, timeout=60, **kw)
        except draft_mod.PostError as e:
            if self.calls or e.status != 404 or self.path != JEV_PRIMARY:
                raise
            self.path = JEV_FALLBACK
            resp = draft_mod.post_json(JEV_BASE + self.path, self.key, req, timeout=60, **kw)
        self.calls += 1
        ans = (resp.get("answers") or {}).get("citation_relation") or {}
        for k, v in (resp.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                self.usage_total[k] = self.usage_total.get(k, 0) + v
        return {
            "label": ans.get("choice"),
            "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities"),
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }


def run_symbol(cfg):
    return cfg.get("symbol") or ",".join(cfg["symbols"])


def jev_check(narrative, payload, cfg, jev):
    """Judge every fact claim with JEV; returns jev_results (UMC revision shape). Raises StageError on a call failure."""
    by_id = {e["id"]: e for e in payload["evidence"]}
    per = []
    for n, c in enumerate(narrative["claims"]):
        if c.get("kind") != "fact":
            continue
        cited = [by_id[i] for i in c["evidence_ids"] if i in by_id]
        try:
            ans = jev.judge(
                c["text"], jev.source_text(c, cited), ", ".join(c["evidence_ids"]), run_symbol(cfg), payload["as_of"]
            )
        except draft_mod.PostError as e:
            raise StageError("jev", f"JEV call failed on claims[{n}]: {e}") from None
        per.append(
            {"claim_index": n, "claim": c["text"], "kind": c.get("kind"), "evidence_ids": c["evidence_ids"], **ans}
        )
    labels = [p["label"] for p in per]
    return {
        "model": jev.model,
        "endpoint": jev.path,
        "n": len(per),
        "supports": labels.count("supports"),
        "contradicts": labels.count("contradicts"),
        "says_nothing": labels.count("says_nothing"),
        "run_at": now_iso(),
        "scope": "fact claims only (inference claims gated by precheck + quote check); source = claim quote + cited evidence texts",
        "usage_total": dict(jev.usage_total),
        "per_claim": per,
    }


# ------------------------------------------------------------------------------------------------ one topic
def validate_row(row):
    problems = []
    if not isinstance(row.get("topic"), str) or not row["topic"].strip():
        problems.append("topic missing")
    if not isinstance(row.get("symbols"), list) or not row["symbols"]:
        problems.append("symbols missing")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(row.get("event_date", ""))):
        problems.append("event_date not YYYY-MM-DD")
    if not str(row.get("primary_source_url") or "").startswith(("http://", "https://")):
        problems.append("primary_source_url missing")
    return problems


class Run:
    """State of one agenda topic; written to run.json whatever happens."""

    def __init__(self, row, run_dir):
        self.row, self.dir = row, Path(run_dir)
        self.data = {
            "agenda_id": row["id"],
            "agenda_row": row,
            "status": "running",
            "reasons": [],
            "started_at": now_iso(),
            "timings_s": {},
            "models": {},
            "draft_calls": [],
            "usage": {},
            "costs": {},
        }

    @contextlib.contextmanager
    def stage(self, name):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.data["timings_s"][name] = round(time.monotonic() - t0, 2)

    def write(self):
        self.data["finished_at"] = now_iso()
        (self.dir / "run.json").write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def process_row(row, *, state_dir, ledger, keys, opts):
    """Run the pipeline for one row; returns the Run (status completed | needs_review | dry_run)."""
    run_dir = Path(state_dir) / "runs" / row["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    run = Run(row, run_dir)
    ledger_path = ledger.path
    revision = None
    try:
        problems = validate_row(row)
        if problems:
            raise StageError("intake", "invalid agenda row: {}".format(", ".join(problems)))
        # assemble
        with run.stage("assemble"):
            try:
                cfg = assemble_mod.assemble(
                    row,
                    run_dir,
                    as_of_dt=opts["as_of"],
                    draft=opts.get("draft"),
                    ledger_path=opts.get("narrative_ledger"),
                    **opts.get("assemble_kw", {}),
                )
            except assemble_mod.AssembleError as e:
                raise StageError("assemble", str(e)) from None
        run.data["assemble_notes"] = cfg["assemble_notes"]
        run.data["event"] = cfg.get("event")
        run.data["models"]["draft_configured"] = cfg["draft"]["model"]
        config_path = run_dir / "config.json"
        # freeze + prepare
        with run.stage("prepare"):
            try:
                info = research_theme(config_path, cfg, ledger_path)
                payload = ledger.manifest_payload(info["manifest_sha256"])
            except Exception as e:
                raise StageError("prepare", f"freeze/prepare refused: {e}") from None
        revision = {"identity": info, "research": None, "manifest_payload": payload}
        run.data.update(revision_id=info["revision_id"], manifest_sha256=info["manifest_sha256"])
        if opts.get("dry_run"):
            messages = draft_mod.build_messages(cfg, payload, info.get("previous_revision_id"))
            write_json(run_dir / "prompt.json", messages)
            run.data["status"] = "dry_run"
            run.data["prompt_sha256"] = draft_mod.sha256_json(messages)
            return run
        ledger.put(row["id"], revision_id=info["revision_id"], scope_key=cfg["scope_key"])
        # draft + gates (+ one repair round)
        transport = opts.get("transport")
        attempt, reasons = None, []
        for round_no in (1, 2):
            with run.stage(f"draft{round_no}"):
                try:
                    if round_no == 1:
                        attempt = draft_mod.draft(
                            cfg,
                            payload,
                            keys["draft"],
                            previous_revision_id=info.get("previous_revision_id"),
                            transport=transport,
                        )
                    else:
                        attempt = draft_mod.repair(cfg, attempt, reasons, keys["draft"], transport=transport)
                except draft_mod.DraftError as e:
                    run.data["draft_calls"].append({"round": round_no, **e.record, "error": str(e)})
                    raise StageError("draft", str(e)) from None
            run.data["draft_calls"].append({"round": round_no, **attempt.record})
            apply_event(attempt.narrative, cfg)
            if round_no == 2:
                (run_dir / "narrative.json").replace(run_dir / "narrative.draft1.json")
            write_json(run_dir / "narrative.json", attempt.narrative)
            with run.stage(f"gates{round_no}"):
                reasons, report = run_gates(attempt.narrative, payload, cfg)
            run.data.setdefault("gates", []).append({"round": round_no, "reasons": reasons, **report})
            if not reasons:
                break
        if reasons:
            raise StageError("gates", "validator refused after one repair round", reasons)
        narrative = attempt.narrative
        # JEV
        jev_client = Jev(keys["jev"], opts.get("jev_model", JEV_MODEL), transport)
        with run.stage("jev"):
            jev = jev_check(narrative, payload, cfg, jev_client)
        revision["jev_results"] = jev
        run.data["models"]["jev"] = jev["model"]
        run.data["usage"]["jev"] = jev["usage_total"]
        if isinstance(jev["usage_total"].get("cost"), (int, float)):
            run.data["costs"]["jev_usd"] = round(jev["usage_total"]["cost"], 6)
        bad = [p for p in jev["per_claim"] if p["label"] != "supports"]
        if bad:
            raise StageError(
                "jev",
                f"JEV did not support {len(bad)}/{jev['n']} claims",
                [f"claims[{p['claim_index']}] {p['label']}: {cap(p['claim'], 60)}" for p in bad],
            )
        # complete
        with run.stage("complete"):
            try:
                research_theme(config_path, cfg, ledger_path, run_dir / "narrative.json", run_dir / "revision.json")
            except Exception as e:
                raise StageError("complete", f"research_theme refused completion: {e}") from None
        revision = json.loads((run_dir / "revision.json").read_text(encoding="utf-8"))
        revision["jev_results"] = jev
        run.data["status"] = "completed"
        # narrative_apply (design 2026-10-29 §4.5): a check that fails is recorded; the revision stays completed
        with run.stage("narrative"):
            narr_jev = Jev(keys["jev"], jev_client.model, transport, path=jev_client.path)
            mode = opts.get("narrative_apply", "off")
            try:
                results = narrative_check.apply(ledger_path, revision, payload, cfg, narr_jev, mode=mode)
            except Exception as e:
                log.exception("narrative_apply failed for %s", row.get("id"))
                run.data["narrative_error"] = f"{type(e).__name__}: {cap(str(e), 300)}"
                results = None
            if narr_jev.calls:
                run.data["usage"]["jev_narrative"] = narr_jev.usage_total
                if isinstance(narr_jev.usage_total.get("cost"), (int, float)):
                    run.data["costs"]["jev_narrative_usd"] = round(narr_jev.usage_total["cost"], 6)
        if results is not None:
            revision["narrative_results"] = results
            run.data["narrative_apply"] = mode
        run.data["content_sha256"] = revision["identity"].get("content_sha256")
        run.data["summary"] = one_line(narrative.get("tl_dr") or narrative.get("thesis"), SUMMARY_CAP)
        return run
    except StageError as e:
        run.data["status"] = "needs_review"
        run.data["failed_stage"] = e.stage
        run.data["reasons"] = [e.reason, *e.details]
        return run
    except Exception as e:  # a bug must not leave the topic silently half-done
        log.exception("internal error while processing %s", row.get("id"))
        run.data["status"] = "needs_review"
        run.data["failed_stage"] = "internal"
        run.data["reasons"] = [f"{type(e).__name__}: {e}"]
        return run
    finally:
        draft_calls = run.data["draft_calls"]
        run.data["models"]["draft_returned"] = [c.get("returned_model") for c in draft_calls]
        run.data["usage"]["draft"] = [c.get("usage") for c in draft_calls]
        costs = [c.get("cost") for c in draft_calls if isinstance(c.get("cost"), (int, float))]
        if costs:
            run.data["costs"]["draft_usd"] = round(sum(costs), 6)
        if revision is not None:
            if run.data["status"] != "completed":
                revision.setdefault("identity", {})["status"] = "prepared"
                revision["needs_review"] = run.data["reasons"]
            write_json(run_dir / "revision.json", revision)
        run.write()


def review_reason(run):
    stage = run.data.get("failed_stage", "run")
    return cap("{}: {}".format(stage, "; ".join(run.data["reasons"])), REASON_CAP)


# ------------------------------------------------------------------------------------------------ main loop
def render_report(run_dir):
    """Markdown of <run_dir>/revision.json via render_theme.py (same rendering as the vault note), or None."""
    rev = Path(run_dir or "") / "revision.json"
    if not run_dir or not rev.is_file():
        return None
    try:
        p = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("render_theme.py")), str(rev)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("render_theme failed for %s: %s", rev, type(e).__name__)
        return None
    if p.returncode != 0 or not p.stdout.strip():
        log.warning("render_theme exit %s for %s: %s", p.returncode, rev, p.stderr.strip()[:200])
        return None
    return p.stdout


def suggestions_of(run_dir):
    """narrative_apply confirm suggestions still open in <run_dir>/revision.json, as the done payload lists them."""
    rev = Path(run_dir or "") / "revision.json"
    if not run_dir or not rev.is_file():
        return []
    records = (json.loads(rev.read_text(encoding="utf-8")).get("narrative_results") or {}).get("records") or []
    return [
        {"id": x["suggestion"]["id"], "narrative_id": x["narrative_id"], "line": x["suggestion"]["line"]}
        for x in records
        if (x.get("suggestion") or {}).get("status") == "suggested"
    ]


def report(bridge, ledger, agenda_id, rec):
    """Tell the bridge the ledger outcome; records the ack. Returns False on a bridge failure."""
    try:
        if rec["status"] == "completed":
            run_dir = rec.get("run_dir")
            bridge.done(
                agenda_id, rec["revision_id"], rec.get("summary") or "", render_report(run_dir), suggestions_of(run_dir)
            )
        else:
            bridge.needs_review(agenda_id, rec.get("reason") or "needs review")
    except BridgeError as e:
        log.error("bridge report for %s failed: %s", agenda_id, e)
        return False
    ledger.put(agenda_id, reported=1)
    log.info("reported %s %s to the bridge", agenda_id, rec["status"])
    return True


# ------------------------------------------------------------------------------------------------ owner inputs
def owner_rows(args, bridge, verb):
    """(rows, rc) for material-pending / narrative-pending. In host mode (--agenda-json) the host passes each list it
    read as a file; a list it did not pass is empty, so a container without ssh never calls the bridge."""
    path, fetch = {
        "material-pending": (args.materials_json, bridge.material_pending),
        "narrative-pending": (args.decisions_json, bridge.narrative_pending),
    }[verb]
    if path:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return (rows if isinstance(rows, list) else []), 0
    if args.agenda_json:
        return [], 0
    try:
        return fetch(), 0
    except BridgeError as e:
        log.error("%s", e)
        return [], EXIT_BRIDGE_PENDING


def run_decisions(bridge, rows, ledger_path):
    """Apply each owner decision (confirm / reject a suggestion, narrative-set) and ack it with the owner's message."""
    rc = 0
    for d in rows:
        did = str(d.get("decision_id", "")) if isinstance(d, dict) else ""
        if not DECISION_ID.match(did):
            log.error("skipping decision row with malformed id %r", did)
            continue
        result, message = narrative_check.apply_decision(ledger_path, d)
        log.info("decision %s %s: %s", did, result, message)
        try:
            bridge.narrative_ack(did, result, message)
        except BridgeError as e:
            log.error("bridge narrative-ack for %s failed: %s", did, e)
            rc = EXIT_BRIDGE_REPORT
    return rc


def material_report(bridge, material_id, data):
    """Tell the bridge a processed material's outcome (from its run.json). Returns False on a bridge failure."""
    try:
        if data["status"] == "intaken":
            bridge.material_done(material_id, data["report"])
        else:
            reason = "{}: {}".format(data.get("failed_stage", "intake"), "; ".join(data["reasons"]))
            bridge.material_needs_review(material_id, reason)
    except BridgeError as e:
        log.error("bridge report for material %s failed: %s", material_id, e)
        return False
    log.info("reported material %s %s to the bridge", material_id, data["status"])
    return True


def process_material_row(row, run_dir, ledger_path, keys, opts):
    """Intake of one bridge row; run.json is written whatever happens. -> run.json data (status intaken|needs_review)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "material_id": row["id"],
        "status": "running",
        "reasons": [],
        "started_at": now_iso(),
        "models": {"draft_configured": opts["draft"]["model"]},
        "draft_calls": [],
        "gates": [],
        "usage": {},
        "costs": {},
    }
    try:
        m = intake_mod.parse_row(row)
        data["report"] = intake_mod.process_material(
            m, run_dir=run_dir, ledger_path=ledger_path, keys=keys, opts=opts, data=data
        )
        data["status"] = "intaken"
    except intake_mod.IntakeError as e:
        data.update(status="needs_review", failed_stage=e.stage, reasons=[e.reason, *e.details])
    except Exception as e:  # a bug must not leave the material silently half-done
        log.exception("internal error while taking in material %s", row.get("id"))
        data.update(status="needs_review", failed_stage="internal", reasons=[f"{type(e).__name__}: {e}"])
    finally:
        calls = data["draft_calls"]
        data["models"]["draft_returned"] = [c.get("returned_model") for c in calls]
        data["usage"]["draft"] = [c.get("usage") for c in calls]
        costs = [c.get("cost") for c in calls if isinstance(c.get("cost"), (int, float))]
        if costs:
            data["costs"]["draft_usd"] = round(sum(costs), 6)
        if data["status"] != "intaken":
            db = nl.connect(ledger_path)
            try:
                if nl._one(db, "SELECT 1 FROM materials WHERE material_id = ?", row["id"]):
                    nl.set_material_status(
                        db, row["id"], "needs_review", note=cap("; ".join(data["reasons"]), REASON_CAP)
                    )
            finally:
                db.close()
        data["finished_at"] = now_iso()
        write_json(run_dir / "run.json", data)
    return data


def run_materials(bridge, rows, base, keys, opts):
    """At most one new material per invocation. A material the bridge still lists after it was taken in (or sent to
    needs_review) is reported again from its run.json, without a new model call."""
    rc, chosen = 0, None
    for row in rows:
        mid = str(row.get("id", "")) if isinstance(row, dict) else ""
        if not MATERIAL_ID.match(mid):
            log.error("skipping material row with malformed id %r", mid)
            continue
        run_json = base / "runs" / f"material-{mid}" / "run.json"
        prior = json.loads(run_json.read_text(encoding="utf-8")) if run_json.is_file() else None
        if prior and prior.get("status") in ("intaken", "needs_review"):
            if not material_report(bridge, mid, prior):
                rc = EXIT_BRIDGE_REPORT
            continue
        if chosen is None:
            chosen = row
    if chosen is not None:
        mid = chosen["id"]
        log.info("taking in material %s (%s %s)", mid, chosen.get("tier"), chosen.get("author"))
        data = process_material_row(chosen, base / "runs" / f"material-{mid}", base / "ledger.sqlite", keys, opts)
        log.info("material %s -> %s %s", mid, data["status"], "; ".join(data["reasons"])[:300])
        if not material_report(bridge, mid, data):
            rc = EXIT_BRIDGE_REPORT
    return rc


def run_owner_inputs(args, bridge, base, keys, *, transport, as_of):
    """Owner decisions first (they may settle a suggestion the agenda run would otherwise show again), then one material."""
    decisions, rc1 = owner_rows(args, bridge, "narrative-pending")
    rc2 = run_decisions(bridge, decisions, base / "ledger.sqlite") if decisions else 0
    materials, rc3 = owner_rows(args, bridge, "material-pending")
    opts = {
        "as_of": as_of or datetime.datetime.now(datetime.UTC).replace(second=0, microsecond=0),
        "transport": transport,
        "draft": {"model": args.draft_model, "reasoning": not args.no_reasoning, "max_tokens": args.max_tokens},
        "jev_factory": lambda: Jev(keys["jev"], args.jev_model, transport),
    }
    rc4 = run_materials(bridge, materials, base, keys, opts) if materials else 0
    return rc1 or rc2 or rc3 or rc4


def run_once(args, keys, *, transport=None, assemble_kw=None, as_of=None):
    state = Path(args.state_dir).expanduser()
    base = state / "dry-run" if args.dry_run else state
    base.mkdir(parents=True, exist_ok=True)
    bridge = OutboxBridge(args.bridge, args.outbox) if args.outbox else Bridge(args.bridge)
    side_rc = 0 if args.dry_run else run_owner_inputs(args, bridge, base, keys, transport=transport, as_of=as_of)
    return (
        run_agenda(args, keys, bridge, state, base, transport=transport, assemble_kw=assemble_kw, as_of=as_of)
        or side_rc
    )


def run_agenda(args, keys, bridge, state, base, *, transport, assemble_kw, as_of):
    if args.agenda_json:
        rows = json.loads(Path(args.agenda_json).read_text(encoding="utf-8"))
    else:
        try:
            rows = bridge.pending()
        except BridgeError as e:
            log.error("%s", e)
            return EXIT_BRIDGE_PENDING
    rows = [r for r in rows if isinstance(r, dict) and r.get("status", "accepted") == "accepted"]
    ledger = Ledger(base / "ledger.sqlite")
    rc = 0
    try:
        chosen = None
        for row in rows:
            aid = str(row.get("id", ""))
            if not AGENDA_ID.match(aid):
                log.error("skipping agenda row with malformed id %r", aid)
                continue
            if args.agenda_id and aid != args.agenda_id:
                continue
            rec = ledger.get(aid)
            if rec and rec["status"] in ("completed", "needs_review"):
                # the bridge still lists it as accepted, so the earlier report did not land: report again, no rerun
                if not args.dry_run and not report(bridge, ledger, aid, rec):
                    rc = EXIT_BRIDGE_REPORT
                continue
            if chosen is None:
                chosen = row
        if chosen is None:
            log.info("nothing to do (%d accepted rows)", len(rows))
            return rc
        aid = chosen["id"]
        log.info("processing %s: %s", aid, chosen.get("topic"))
        if not args.dry_run:
            ledger.put(aid, status="running", started_at=now_iso(), run_dir=str(base / "runs" / aid), reported=0)
        opts = {
            "as_of": as_of or datetime.datetime.now(datetime.UTC).replace(second=0, microsecond=0),
            "dry_run": args.dry_run,
            "transport": transport,
            "jev_model": args.jev_model,
            "draft": {"model": args.draft_model, "reasoning": not args.no_reasoning, "max_tokens": args.max_tokens},
            "assemble_kw": assemble_kw or {},
            "narrative_ledger": state / "ledger.sqlite",  # read for the snapshot, also by a dry run
            "narrative_apply": args.narrative_apply,
        }
        run = process_row(chosen, state_dir=base, ledger=ledger, keys=keys, opts=opts)
        status = run.data["status"]
        log.info("%s -> %s %s", aid, status, "; ".join(run.data["reasons"])[:300])
        if args.dry_run:
            return rc
        if status == "completed":
            ledger.put(
                aid, status="completed", revision_id=run.data["revision_id"], summary=run.data["summary"], reason=None
            )
        else:
            ledger.put(aid, status="needs_review", reason=review_reason(run))
        if not report(bridge, ledger, aid, ledger.get(aid)):
            rc = EXIT_BRIDGE_REPORT
        return rc
    finally:
        ledger.close()


def setup_logging(state_dir):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(Path(state_dir) / "runner.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Research runner: one accepted agenda topic per invocation.")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--draft-env", help=f"600 env file with {draft_mod.DRAFT_KEY_VAR}")
    ap.add_argument("--jev-env", help=f"600 env file with {JEV_KEY_VAR}")
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE, help="agenda bridge command (default: %(default)s)")
    ap.add_argument("--agenda-json", help="read pending rows from this file instead of `<bridge> pending`")
    ap.add_argument(
        "--outbox", help="queue done / needs-review here for deliver_outbox.py instead of calling the bridge"
    )
    ap.add_argument(
        "--materials-json",
        help="read material-pending rows from this file (host mode: with --agenda-json, no file = none)",
    )
    ap.add_argument(
        "--decisions-json",
        help="read narrative-pending rows from this file (host mode: with --agenda-json, no file = none)",
    )
    ap.add_argument("--agenda-id", help="only consider this agenda id")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble + prepare + prompt only, in <state>/dry-run; no model, JEV or bridge writes, no keys needed",
    )
    ap.add_argument("--draft-model", default=draft_mod.DEFAULT_MODEL)
    ap.add_argument("--no-reasoning", action="store_true", help="send reasoning.enabled=false to the draft model")
    ap.add_argument("--max-tokens", type=int, default=draft_mod.DEFAULT_MAX_TOKENS)
    ap.add_argument("--jev-model", default=JEV_MODEL)
    ap.add_argument(
        "--narrative-apply",
        choices=narrative_check.APPLY_MODES,
        default="off",
        help="write narrative status events from the rule table (default off until the JEV control set passes, §9 "
        "step 7); off still writes one narrative_checks row per shown record; confirm (owner decision 2026-10-29) "
        "lists each change the rule would write in the done notice for the owner to confirm or reject in Telegram",
    )
    return ap.parse_args(argv)


def main(argv=None, *, transport=None, assemble_kw=None, as_of=None):
    args = parse_args(argv)
    state = Path(args.state_dir).expanduser()
    state.mkdir(parents=True, exist_ok=True)
    setup_logging(state)
    keys = {}
    if not args.dry_run:
        try:
            keys["draft"] = draft_mod.load_key(args.draft_env, draft_mod.DRAFT_KEY_VAR)
            keys["jev"] = draft_mod.load_key(args.jev_env, JEV_KEY_VAR)
        except draft_mod.MissingKeyError as e:
            log.error("key error: %s", e)
            return EXIT_KEY
    lock = open(state / "runner.lock", "w")  # noqa: SIM115 - held for the whole run
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("another runner holds %s; exiting", state / "runner.lock")
        return 0
    try:
        lock.write(str(os.getpid()))
        lock.flush()
        return run_once(args, keys, transport=transport, assemble_kw=assemble_kw, as_of=as_of)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()


if __name__ == "__main__":
    sys.exit(main())
