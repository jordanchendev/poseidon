#!/usr/bin/env python3
"""Telegram <-> strategy pipeline bridge for stormtrooper host cron (stdlib only, host python3).

Commands
  bridge_sync.py sync [--config FILE] [--dry-run]
      Cron entry (every 2 minutes), under an exclusive lock (busy -> exit 0):
      pull  `ssh research-bridge strategy-pending` -> POST /api/v1/pipelines {idea_text, source: "telegram"} with
            Idempotency-Key "telegram-idea-<idea id>" (default stage_plan = full pipeline) -> `strategy-ack <id>
            <run_no> <run_id>`. A crash between POST and ack is harmless: the next tick POSTs the same key and the
            API replays the same run.
      push  for every run this bridge created: on first sight `strategy-notify <id> start` ("pipeline #N 開始：..."),
            and once the run is terminal (completed/failed/unsupported/error/aborted) one summary message
            `strategy-notify <id> <status>`. The bridge keeps a marker per idea+event and answers a repeat with
            "duplicate" without sending, so a crash after a send (or a re-run of the ssh command) never notifies twice.
            Local state only saves calls; the bridge marker is the authority.
      --dry-run reads pending ideas and run status, logs what it would do, writes nothing and sends nothing.
  bridge_sync.py preview RUN_ID [--config FILE]
      Print the terminal summary message for any run (nothing is sent).

Config (JSON, merged over DEFAULTS; default file ~/strategy-pipeline/bridge.json when present):
  api_base, key_file (researcher key, 600, never logged), bridge, state_dir, log_file, kairos_base_url (empty -> the
  Kairos link is omitted), watch_days (how long unsupported/error/aborted runs are watched for a rerun),
  hold_windows_utc (no new runs inside these UTC windows; default 06:00-08:40 for the Phase 101 soak).
State: <state_dir>/ideas/<idea id>.json, <state_dir>/runs/<run id>.json, <state_dir>/sync.lock.
Exit codes: 0 ok / nothing to do / lock busy, 1 some step failed (retried next tick), 3 key file problem.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULTS = {
    "api_base": "http://127.0.0.1:8011",
    "key_file": "~/strategy-pipeline/secrets/strategy-bridge-api.key",
    "bridge": "ssh research-bridge",
    "state_dir": "~/strategy-pipeline/bridge-state",
    "log_file": "~/strategy-pipeline/bridge.log",
    "kairos_base_url": "",
    "watch_days": 14,
    # New runs are not created inside these UTC windows (Phase 101 soak: no heavy CPU 06:00-08:40Z); ideas wait
    # on the bridge and are picked up after the window. Notifications are not held. [] disables the hold.
    "hold_windows_utc": [["06:00", "08:40"]],
}
DEFAULT_CONFIG = "~/strategy-pipeline/bridge.json"
IDEA_ID_RE = re.compile(r"^\d{8}-s\d{1,4}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
IDEA_MAX_CHARS = 2000
TERMINAL = ("completed", "failed", "unsupported", "error", "aborted")
ALL_GATES_PASSED = "passed all validation gates"  # poseidon.pipeline.plan.ALL_GATES_PASSED (run.reason)
FINAL = ("completed", "failed")  # never rerun (owner decision: a gate fail is final)
STATUS_LINE = {
    "completed": "✅ 通過全部驗證關卡",
    "failed": "❌ 未通過",
    "unsupported": "⛔ 不支援",
    "error": "⚠️ 執行錯誤",
    "aborted": "⏹ 已中止",
}
VERDICT_MARK = {"pass": "✅", "fail": "❌", "unsupported": "⛔", "error": "⚠️"}
MAX_ASSUMPTIONS = 5
MESSAGE_CAP = 3800
PERMANENT_HTTP = (400, 401, 403, 404, 405, 413, 415, 422)


def expand(path) -> Path:
    return Path(os.path.expanduser(str(path)))


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def load_config(path=None) -> dict:
    cfg = dict(DEFAULTS)
    candidate = expand(path or DEFAULT_CONFIG)
    if path or candidate.is_file():
        cfg.update(json.loads(candidate.read_text(encoding="utf-8")))
    return cfg


class Logger:
    def __init__(self, path=None, echo=True):
        self.path = expand(path) if path else None
        self.echo = echo
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        line = f"{utcnow():%Y-%m-%dT%H:%M:%SZ} {msg}"
        self.lines.append(line)
        if self.echo:
            print(line, file=sys.stderr)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class KeyFileError(Exception):
    pass


def read_key(path) -> str:
    """The researcher key; refused unless the file is a regular file readable by the owner only."""

    p = expand(path)
    try:
        st = p.stat()
    except OSError as error:
        raise KeyFileError(f"key file {p} not readable: {type(error).__name__}") from None
    if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
        raise KeyFileError(f"key file {p} must be a regular file with mode 600")
    key = p.read_text(encoding="utf-8").strip()
    if not key or any(ch.isspace() for ch in key):
        raise KeyFileError(f"key file {p} is empty or malformed")
    return key


# ------------------------------------------------------------------ I/O


def urllib_request(method, url, body, headers, timeout=30):
    """-> (http status, parsed JSON or None). Network errors raise OSError."""

    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, None


class Api:
    def __init__(self, base, key, request=urllib_request):
        self.base = base.rstrip("/")
        self._key = key
        self.request = request

    def _headers(self, extra=None):
        headers = {"X-API-Key": self._key, "Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def get(self, path):
        return self.request("GET", self.base + path, None, self._headers())

    def create(self, idea_id, text):
        headers = self._headers({"Content-Type": "application/json", "Idempotency-Key": f"telegram-idea-{idea_id}"})
        body = {"idea_text": text, "source": "telegram"}  # no stage_plan: the server default (full pipeline)
        return self.request("POST", self.base + "/api/v1/pipelines", body, headers)


class Bridge:
    """`ssh research-bridge <verb> <args>`; free text only on stdin."""

    def __init__(self, command, runner=subprocess.run, timeout=180):
        self.command = shlex.split(command)
        self.runner = runner
        self.timeout = timeout

    def call(self, *args, stdin=""):
        try:
            p = self.runner([*self.command, *args], input=stdin, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            return 255, "", type(error).__name__
        return p.returncode, p.stdout, p.stderr


class State:
    def __init__(self, root):
        self.root = expand(root)
        self.ideas = self.root / "ideas"
        self.runs = self.root / "runs"

    def ensure(self):
        for d in (self.root, self.ideas, self.runs):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    @staticmethod
    def _write(path, data):
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def idea(self, idea_id):
        return self._read(self.ideas / f"{idea_id}.json")

    def save_idea(self, rec):
        self._write(self.ideas / f"{rec['idea_id']}.json", rec)

    def run(self, run_id):
        return self._read(self.runs / f"{run_id}.json")

    def save_run(self, rec):
        self._write(self.runs / f"{rec['run_id']}.json", rec)

    def open_runs(self):
        out = []
        for path in sorted(self.runs.glob("*.json")):
            rec = self._read(path)
            if rec and not rec.get("closed"):
                out.append(rec)
        return sorted(out, key=lambda r: r.get("run_no") or 0)


# ------------------------------------------------------------------ formatting


def _one_line(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def start_message(run) -> str:
    return f"pipeline #{run['run_no']} 開始：{_one_line(run.get('idea_text'), 60)}"


def _num(value, fmt):
    return format(value, fmt) if isinstance(value, (int, float)) and not isinstance(value, bool) else "?"


def _parse_iso(value):
    try:
        return dt.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def wall_time(run) -> str | None:
    start, end = _parse_iso(run.get("created_at")), _parse_iso(run.get("finished_at"))
    if not start or not end:
        return None
    seconds = max(0, int((end - start).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分"
    if minutes:
        return f"{minutes} 分 {secs} 秒" if minutes < 10 else f"{minutes} 分"
    return f"{secs} 秒"


def _report(attempt):
    artifact = (attempt or {}).get("artifact_json") or {}
    report = artifact.get("report") if isinstance(artifact, dict) else None
    return report if isinstance(report, dict) else {}


def _params_text(params: dict) -> str:
    return ", ".join(f"`{name}={value}`" for name, value in params.items())


def stage_line(stage, latest, report) -> str:
    if latest is None:
        return f"{stage} ⏭"
    status, verdict = latest.get("status"), latest.get("verdict")
    if status == "aborted":
        mark = "⏹"
    elif status != "completed":
        mark = "⏳"
    else:
        mark = VERDICT_MARK.get(verdict, "?")
    if isinstance(report.get("summary"), str) and report["summary"].strip():
        return _one_line(report["summary"], 160)
    detail = ""
    if verdict == "pass" and stage == "backtest" and isinstance(report.get("metrics"), dict):
        m = report["metrics"]
        detail = (
            f"交易 {_num(m.get('n_trades'), 'd')} 筆，PnL {_num(m.get('total_pnl_twd'), '+,.0f')} TWD，"
            f"Sharpe(日) {_num(m.get('sharpe_daily'), '.3f')}"
        )
    elif verdict == "pass" and stage == "optimize" and isinstance(report.get("best"), dict):
        best = report["best"]
        metrics = best.get("metrics") or {}
        detail = (
            f"{_num(report.get('n_trials'), 'd')} trials，最佳 #{_num(best.get('trial_no'), 'd')} "
            f"Sharpe(日) {_num(metrics.get('sharpe_daily'), '.3f')}"
        )
    elif latest.get("reason"):
        detail = _one_line(latest["reason"], 120)
    return f"{stage} {mark} {detail}".rstrip()


def terminal_message(run, stages, reports, kairos_base_url="") -> str:
    """``stages`` = the API's stage list; ``reports`` = {stage: artifact report} for stages with a full attempt."""

    status = run.get("status")
    head = STATUS_LINE.get(status, status)
    if status == "completed" and run.get("reason") != ALL_GATES_PASSED:
        head = "✅ 完成"  # a shortened stage_plan: no validation gates ran
    lines = [f"pipeline #{run.get('run_no')} {head}：{_one_line(run.get('idea_text'), 60)}"]
    not_run = []
    for entry in stages:
        stage, latest = entry.get("stage"), entry.get("latest")
        if latest is None:
            not_run.append(stage)
            continue
        lines.append(stage_line(stage, latest, reports.get(stage) or {}))
    if not_run:
        lines.append("未執行：" + ", ".join(not_run))

    spec_generate = reports.get("spec_generate") or {}
    missing = spec_generate.get("missing_capabilities") or (reports.get("spec_validate") or {}).get(
        "missing_capabilities"
    )
    if isinstance(missing, list) and missing:
        lines.append("缺少能力：" + ", ".join(_one_line(m, 40) for m in missing[:8]))
    if status in ("unsupported", "error", "aborted") and run.get("reason"):
        lines.append("原因：" + _one_line(run["reason"], 200))

    assumptions = [a for a in spec_generate.get("assumptions") or [] if isinstance(a, str) and a.strip()]
    if assumptions:
        lines.append("假設：")
        lines += [f"• {_one_line(a, 160)}" for a in assumptions[:MAX_ASSUMPTIONS]]
        if len(assumptions) > MAX_ASSUMPTIONS:
            lines.append(f"• …另有 {len(assumptions) - MAX_ASSUMPTIONS} 條")

    best = (reports.get("optimize") or {}).get("best")
    validated = (reports.get("spec_validate") or {}).get("params")
    if isinstance(best, dict) and isinstance(best.get("params"), dict) and best["params"]:
        lines.append("參數（optimize 最佳）：" + _params_text(best["params"]))
    elif isinstance(validated, dict) and validated:
        defaults = {
            name: (spec or {}).get("default", "?") for name, spec in validated.items() if isinstance(spec, dict)
        }
        lines.append("參數（預設值）：" + _params_text(defaults))
    elif isinstance(spec_generate.get("params"), list) and spec_generate["params"]:
        lines.append("參數：" + ", ".join(f"`{p}`" for p in spec_generate["params"]))

    elapsed = wall_time(run)
    if elapsed:
        lines.append(f"耗時 {elapsed}")
    if kairos_base_url:
        lines.append(f"{kairos_base_url.rstrip('/')}/pipelines?run={run.get('id')}")
    text = "\n".join(lines)
    return text if len(text) <= MESSAGE_CAP else text[: MESSAGE_CAP - 1] + "…"


def fetch_terminal_detail(api, run_id):
    """-> (stages list, {stage: report}) or raises RuntimeError."""

    status, body = api.get(f"/api/v1/pipelines/{run_id}/stages")
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"stages http {status}")
    stages = body.get("stages") or []
    reports = {}
    for entry in stages:
        latest = entry.get("latest")
        if not latest or latest.get("status") != "completed":
            continue
        status, attempt = api.get(
            f"/api/v1/pipelines/{run_id}/stages/{entry['stage']}/attempts/{int(latest['attempt'])}"
        )
        if status != 200:
            raise RuntimeError(f"attempt {entry['stage']}#{latest['attempt']} http {status}")
        reports[entry["stage"]] = _report(attempt)
    return stages, reports


# ------------------------------------------------------------------ sync


def _detail(body):
    if isinstance(body, dict) and "detail" in body:
        return _one_line(json.dumps(body["detail"], ensure_ascii=False), 200)
    return ""


def held_until(windows, now) -> str | None:
    """'HH:MM' end of the UTC hold window containing ``now`` (an aware datetime), else None."""

    minute = now.astimezone(dt.UTC).hour * 60 + now.astimezone(dt.UTC).minute
    for start, end in windows or []:
        a, b = (int(x[:2]) * 60 + int(x[3:5]) for x in (start, end))
        if (a <= minute < b) if a <= b else (minute >= a or minute < b):
            return end
    return None


def pull(api, bridge, state, log, dry_run=False, hold=None, now=None) -> int:
    rc, out, err = bridge.call("strategy-pending")
    if rc != 0:
        log(f"bridge strategy-pending exit {rc}: {_one_line(err, 200)}")
        return 1
    try:
        ideas = json.loads(out)
        assert isinstance(ideas, list)
    except (ValueError, AssertionError):
        log("bridge strategy-pending returned malformed JSON")
        return 1
    failures = 0
    for idea in ideas:
        idea_id = idea.get("id") if isinstance(idea, dict) else None
        text = idea.get("text") if isinstance(idea, dict) else None
        if not isinstance(idea_id, str) or not IDEA_ID_RE.match(idea_id):
            log(f"skipping pending idea with a bad id: {_one_line(idea_id, 40)}")
            failures += 1
            continue
        if not isinstance(text, str) or not text.strip() or len(text) > IDEA_MAX_CHARS:
            log(f"skipping pending idea {idea_id}: text missing or longer than {IDEA_MAX_CHARS} chars")
            failures += 1
            continue
        rec = state.idea(idea_id)
        if rec and rec.get("rejected"):
            continue  # logged once when it was rejected
        until = held_until(hold, now or utcnow()) if rec is None else None
        if until:
            marker = state.root / "held" / idea_id
            if not marker.exists():
                marker.parent.mkdir(exist_ok=True)
                marker.touch()
                log(f"idea {idea_id} held: no new runs until {until}Z (hold_windows_utc)")
            continue
        if dry_run:
            log(f"dry-run: would {'re-ack' if rec else 'create a run for'} idea {idea_id} ({len(text)} chars)")
            continue
        if rec is None:
            try:
                status, body = api.create(idea_id, text)
            except OSError as error:
                log(f"create for idea {idea_id} failed: {type(error).__name__}")
                failures += 1
                continue
            if status in PERMANENT_HTTP:
                state.save_idea(
                    {"idea_id": idea_id, "rejected": True, "http_status": status, "at": utcnow().isoformat()}
                )
                log(f"create for idea {idea_id} rejected http {status} {_detail(body)}; not retried")
                failures += 1
                continue
            run_id = body.get("id") if isinstance(body, dict) else None
            run_no = body.get("run_no") if isinstance(body, dict) else None
            if (
                status not in (200, 201)
                or not isinstance(run_id, str)
                or not UUID_RE.match(run_id)
                or not isinstance(run_no, int)
            ):
                log(f"create for idea {idea_id} http {status} {_detail(body)}; retry next tick")
                failures += 1
                continue
            rec = {"idea_id": idea_id, "run_id": run_id, "run_no": run_no, "created_at": utcnow().isoformat()}
            if state.run(run_id) is None:
                state.save_run({**rec, "notified": [], "closed": False})
            state.save_idea(rec)
            log(f"idea {idea_id} -> pipeline run #{run_no} {run_id}")
        rc, out, err = bridge.call("strategy-ack", idea_id, str(rec["run_no"]), rec["run_id"])
        if rc == 0:
            log(f"acked idea {idea_id} as run #{rec['run_no']}")
        else:
            log(f"bridge strategy-ack {idea_id} exit {rc}: {_one_line(err, 200)}")
            failures += 1
    return failures


def _notify(bridge, rec, event, text, state, log, dry_run) -> bool:
    if dry_run:
        log(f"dry-run: would send {event} for run #{rec['run_no']} ({len(text)} chars)")
        return True
    rc, out, err = bridge.call("strategy-notify", rec["idea_id"], event, stdin=text)
    if rc != 0:
        log(f"bridge strategy-notify {rec['idea_id']} {event} exit {rc}: {_one_line(err, 200)}")
        return False
    try:
        duplicate = bool(json.loads(out).get("duplicate"))
    except (ValueError, AttributeError):
        duplicate = False
    rec["notified"] = sorted(set(rec.get("notified") or []) | {event})
    state.save_run(rec)
    log(f"notified run #{rec['run_no']} {event}{' (bridge: duplicate, not resent)' if duplicate else ''}")
    return True


def push(api, bridge, state, cfg, log, dry_run=False) -> int:
    failures = 0
    watch = dt.timedelta(days=float(cfg.get("watch_days") or 14))
    for rec in state.open_runs():
        try:
            status, run = api.get(f"/api/v1/pipelines/{rec['run_id']}")
        except OSError as error:
            log(f"run #{rec['run_no']} status read failed: {type(error).__name__}")
            failures += 1
            continue
        if status != 200 or not isinstance(run, dict):
            log(f"run #{rec['run_no']} status http {status}")
            failures += 1
            continue
        notified = set(rec.get("notified") or [])
        if "start" not in notified and not _notify(bridge, rec, "start", start_message(run), state, log, dry_run):
            failures += 1
            continue  # the summary never goes out before the start message
        run_status = run.get("status")
        if run_status in TERMINAL and run_status not in notified:
            try:
                stages, reports = fetch_terminal_detail(api, rec["run_id"])
            except (OSError, RuntimeError) as error:
                log(
                    f"run #{rec['run_no']} detail read failed: {error if isinstance(error, RuntimeError) else type(error).__name__}"
                )
                failures += 1
                continue
            text = terminal_message(run, stages, reports, cfg.get("kairos_base_url") or "")
            if not _notify(bridge, rec, run_status, text, state, log, dry_run):
                failures += 1
                continue
        if dry_run or run_status not in TERMINAL or run_status not in set(rec.get("notified") or []):
            continue
        finished = _parse_iso(run.get("finished_at"))
        if run_status in FINAL or (finished and utcnow() - finished > watch):
            rec["closed"] = True
            state.save_run(rec)
    return failures


class LockBusy(Exception):
    pass


class Lock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            raise LockBusy() from None
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def sync(cfg, log, dry_run=False, request=urllib_request, runner=subprocess.run, now=None) -> int:
    try:
        key = read_key(cfg["key_file"])
    except KeyFileError as error:
        log(str(error))
        return 3
    state = State(cfg["state_dir"])
    state.ensure()
    try:
        with Lock(state.root / "sync.lock"):
            api = Api(cfg["api_base"], key, request=request)
            bridge = Bridge(cfg["bridge"], runner=runner)
            failures = pull(api, bridge, state, log, dry_run=dry_run, hold=cfg.get("hold_windows_utc"), now=now)
            failures += push(api, bridge, state, cfg, log, dry_run=dry_run)
    except LockBusy:
        return 0
    return 1 if failures else 0


def preview(cfg, run_id, request=urllib_request) -> str:
    if not UUID_RE.match(run_id):
        raise SystemExit("run id must be a UUID")
    api = Api(cfg["api_base"], read_key(cfg["key_file"]), request=request)
    status, run = api.get(f"/api/v1/pipelines/{run_id}")
    if status != 200:
        raise SystemExit(f"run http {status}")
    stages, reports = fetch_terminal_detail(api, run_id)
    return start_message(run) + "\n----\n" + terminal_message(run, stages, reports, cfg.get("kairos_base_url") or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--config")
    s.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("preview")
    p.add_argument("run_id")
    p.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.command == "preview":
        try:
            print(preview(cfg, a.run_id))
        except KeyFileError as error:
            print(error, file=sys.stderr)
            return 3
        return 0
    log = Logger(cfg["log_file"], echo=False)
    return sync(cfg, log, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
