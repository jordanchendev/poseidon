#!/usr/bin/env python3
"""Daily market briefs (TW + US, pre/post market) for stormtrooper host cron. Stdlib only, host python3.

Commands
  run.py due [--now ISO] [--no-llm] [--no-deliver] [--dry-run]
      Cron entry (every 5 min). Under a lock: deliver queued briefs, build every slot that is due (slot <= now <
      slot + window_minutes, trading day, not final in state), queue it in the outbox, deliver again.
      Data not complete -> retried on later ticks until complete_wait_minutes after the slot, then built with a
      「資料未齊」 note. Required data missing at the end of the window -> state failed (logged, nothing sent).
      Closed market (calendar or exchange "no data") -> state closed (logged once, no notice). Giving up (window over
      with required data missing, an unhandled error while building, or a missing LLM key) queues ONE failure notice
      per kind/date (bridge `brief-failed`), recorded as notice_queued in state. --dry-run only lists due slots.
  run.py run KIND [--date YYYY-MM-DD] [--no-llm] [--dry-run] [--out DIR]
      One report now, ignoring the slot window. --dry-run writes <out>/<file>.md, the outbox JSON and run record to
      --out (default <state>/dry-run) and touches neither state nor outbox; without it the brief is queued.
  run.py deliver
      Deliver queued briefs only.
  run.py archive-backfill
      Copy delivered briefs (outbox/sent/*.json) into the public archive (archive_dir) and its index.json.
Archive (Kairos): every queued brief is also written to <archive_dir>/<YYYY>/<date>-<kind>.md (644, dirs 755) and
<archive_dir>/index.json is upserted (newest first), right after the md is final, independent of Telegram.
  run.py resend KIND --date YYYY-MM-DD [--dry-run]
      Re-send the phone-format text of an already delivered brief (bridge `brief-resend`, text only). Source: the
      delivered markdown in outbox/sent/<date>-<kind>.json, else a fresh no-LLM build. --dry-run prints the text.

Files (config.json; paths may use ~): state_dir/state.json, state_dir/market-brief.log, state_dir/runs/<date>-<kind>.json,
state_dir/cache/tw-holidays-<year>.json, outbox_dir/<date>-<kind>.json (+ sent/, refused/).
LLM key: config llm.key_file / llm.key_var, read from that file only (see llm.py); unless --no-llm a missing key exits
3 before any report data is fetched or queued.
Exit codes: 0 ok / nothing due / lock busy, 1 report failed, 3 key file, 4 delivery failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import sys
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect  # noqa: E402
import deliver  # noqa: E402
import llm  # noqa: E402
import render  # noqa: E402
import tradingcal  # noqa: E402
from netio import Fetcher  # noqa: E402

TPE = ZoneInfo("Asia/Taipei")


def expand(p):
    return Path(os.path.expanduser(str(p)))


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg


def load_watchlist(cfg):
    p = expand(cfg.get("watchlist_file") or "")
    src = p if p.is_file() else HERE / "watchlist.json"
    return json.loads(src.read_text(encoding="utf-8")), str(src)


class Logger:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, msg):
        line = f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ} {msg}"
        print(line, file=sys.stderr)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def load_state(state_dir):
    try:
        return json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def build_calendar(fetcher, cfg, state_dir, years, log):
    tw_closed = {y: tradingcal.load_tw_holidays(fetcher, state_dir / "cache", y, log) for y in sorted(years)}
    cal = tradingcal.Calendar(tw_closed, cfg["nyse_holidays"])
    for y in years:
        if not cal.nyse_known(y):
            log(f"NYSE holidays for {y} are not in config; weekdays assumed open")
    return cal


def get_key(cfg, no_llm):
    if no_llm:
        return None
    return llm.load_key(cfg["llm"]["key_file"], cfg["llm"]["key_var"])


def build(kind, d, *, cfg, fetcher, calendar, watchlist, now_utc, key, log, llm_transport=None):
    """-> (facts, md or None, summary or None, llm_meta or None)."""
    c = collect.Collector(fetcher, cfg, watchlist, calendar, now_utc, log=log)
    facts = collect.collect(kind, d, c)
    if facts["status"] != "ok":
        return facts, None, None, None
    section, meta = None, None
    if key:
        section, meta = llm.interpret(facts, key, cfg["llm"], transport=llm_transport)
        log(
            f"{kind} {d} llm {meta['status']} model={meta['returned_model']} cost={meta['cost']} {'; '.join(meta['reasons'])[:300]}"
        )
    md, summary = render.render(facts, section, meta, cfg)
    return facts, md, summary, meta


def run_record(facts, meta, fetcher):
    return {
        "kind": facts["kind"],
        "date": facts["date"],
        "status": facts["status"],
        "generated_at": facts["generated_at"],
        "gaps": facts["gaps"],
        "must_missing": facts["must_missing"],
        "llm": meta,
        "fetches": fetcher.log,
    }


def outbox_entry(facts, md, summary):
    return {
        "kind": facts["kind"],
        "date": facts["date"],
        "filename": render.filename(facts["kind"], facts["date"]),
        "summary": summary,
        "report_md": md,
        "telegram_md": render.telegram_text(md),
    }


def queue_failure(state, outbox, kind, d, reason, now, log):
    """Mark kind/date failed and queue one `brief-failed` notice. Never more than one notice per kind/date."""
    k = f"{kind}:{d}"
    prev = state.get(k) or {}
    reason = " ".join(str(reason).split())[:400] or "unknown"
    state[k] = {**prev, "status": "failed", "reason": reason, "at": now.isoformat(timespec="seconds")}
    if prev.get("notice_queued"):
        return False
    save_json(
        outbox / f"{d}-{kind}.failed.json", {"verb": "brief-failed", "kind": kind, "date": str(d), "reason": reason}
    )
    state[k]["notice_queued"] = True
    log(f"{kind} {d}: gave up ({reason}); failure notice queued")
    return True


def sweep_expired(state, outbox, slots, window_minutes, now, log):
    """Slots left `waiting` after their window closed (e.g. the last tick was skipped) -> give up with a notice."""
    for k, v in list(state.items()):
        if v.get("status") != "waiting":
            continue
        kind, _, ds = k.partition(":")
        if kind not in slots:
            continue
        end = tradingcal.slot_datetime(slots[kind], dt.date.fromisoformat(ds)) + dt.timedelta(minutes=window_minutes)
        if now >= end:
            queue_failure(
                state, outbox, kind, ds, f"window ended without a report; last: {v.get('reason', '')}", now, log
            )


# ------------------------------------------------------------------------------------------------ archive (Kairos)
def parse_frontmatter(md):
    """The subset render.frontmatter writes: `key: value` (JSON-quoted strings or bare) and `key:` + `  - "item"` lists."""
    m = re.match(r"---\n(.*?)\n---", md, re.S)
    out, key = {}, None
    for line in (m.group(1) if m else "").splitlines():
        if line.startswith("  - ") and key:
            v = line[4:].strip()
            out.setdefault(key, []).append(json.loads(v) if v.startswith('"') else v)
            continue
        k, sep, v = line.partition(":")
        if not sep or line.startswith(" "):
            continue
        key, v = k.strip(), v.strip()
        if v:
            try:
                out[key] = json.loads(v) if v.startswith('"') else v
            except ValueError:
                out[key] = v
        else:
            out[key] = []
    return out


def _write_public(path, data):
    """Atomic write, mode 644 (the archive is public market data served read-only by Caddy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for d in (path.parent, path.parent.parent):
        if d.exists():
            os.chmod(d, 0o755)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def archive_brief(archive_dir, kind, date, md, summary, log=None):
    """Write <archive>/<YYYY>/<date>-<kind>.md and upsert archive/index.json (newest first). Runs as soon as the md is
    final, independent of Telegram delivery."""
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(archive_dir, 0o755)
    name = render.filename(kind, date)
    rel = f"{date[:4]}/{name}"
    data = md.encode("utf-8")
    _write_public(archive_dir / rel, data)
    fm = parse_frontmatter(md)
    item = {
        "date": date,
        "kind": kind,
        "market": kind.split("-")[0],
        "session": kind.split("-")[1],
        "title": fm.get("title") or name,
        "file": rel,
        "summary": summary,
        "bytes": len(data),
        "session_date": str(fm.get("session_date") or date),
        "generated_at": fm.get("generated_at"),
        "data_gaps": fm.get("data_gaps") or [],
    }
    index_path = archive_dir / "index.json"
    try:
        briefs = json.loads(index_path.read_text(encoding="utf-8")).get("briefs", [])
    except (OSError, ValueError):
        briefs = []
    briefs = [b for b in briefs if not (b.get("date") == date and b.get("kind") == kind)] + [item]
    briefs.sort(key=lambda b: (b.get("date") or "", b.get("generated_at") or ""), reverse=True)
    index = {"generated_at": dt.datetime.now(TPE).isoformat(timespec="seconds"), "briefs": briefs}
    _write_public(index_path, json.dumps(index, ensure_ascii=False, indent=1).encode("utf-8"))
    if log:
        log(f"{kind} {date}: archived {rel}")
    return item


def try_archive(cfg, entry, log):
    try:
        archive_brief(
            expand(cfg.get("archive_dir") or "~/market-brief/archive"),
            entry["kind"],
            entry["date"],
            entry["report_md"],
            entry["summary"],
            log,
        )
    except Exception as e:  # never block delivery on the archive
        log(f"{entry['kind']} {entry['date']}: archive failed {type(e).__name__}: {e}")


def cmd_archive_backfill(a, cfg, state_dir, outbox, log):
    """Import delivered briefs from outbox/sent/*.json into the archive (brief entries only)."""
    n = 0
    for p in sorted((outbox / "sent").glob("*.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if e.get("verb") not in (None, "brief") or not deliver.valid(e):
            continue
        archive_brief(
            expand(cfg.get("archive_dir") or "~/market-brief/archive"),
            e["kind"],
            e["date"],
            e["report_md"],
            e["summary"],
            log,
        )
        n += 1
    log(f"archive-backfill: {n} brief(s) imported")
    return 0


class Lock:
    def __init__(self, path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            self.fh = None
        return self

    @property
    def acquired(self):
        return self.fh is not None

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


def parse_now(s):
    if not s:
        return dt.datetime.now(dt.UTC)
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=TPE)
    return t.astimezone(dt.UTC)


def cmd_due(a, cfg, state_dir, outbox, log):
    now = parse_now(a.now)
    fetcher = Fetcher()
    years = {now.astimezone(TPE).year, (now.astimezone(TPE) - dt.timedelta(days=1)).year}
    with Lock(state_dir / "run.lock") as lock:
        if not lock.acquired:
            return 0
        rc = 0
        if not a.no_deliver and not a.dry_run and deliver.deliver_outbox(outbox, cfg["bridge"], log=log):
            rc = 4
        state = load_state(state_dir)
        cal = build_calendar(fetcher, cfg, state_dir, years, log)
        due = tradingcal.due_slots(now, cfg["slots"], cal, state, cfg["window_minutes"])
        if a.dry_run:
            for kind, d, sdt, age, open_ in due:
                print(f"due: {kind} {d} slot {sdt.isoformat()} age {age:.0f} min market_open={open_}")
            return 0
        work = [x for x in due if x[4]]
        for kind, d, _sdt, _age, open_ in due:
            if not open_:
                state[f"{kind}:{d}"] = {
                    "status": "closed",
                    "reason": "calendar holiday",
                    "at": now.isoformat(timespec="seconds"),
                }
                log(f"{kind} {d}: market closed (calendar); no report")
        try:
            key = get_key(cfg, a.no_llm) if work else None  # fail closed before any report data is fetched
        except llm.MissingKeyError as e:
            for kind, d, *_ in work:
                queue_failure(state, outbox, kind, d, f"LLM key file error: {e}", now, log)
            save_json(state_dir / "state.json", state)
            if not a.no_deliver:
                deliver.deliver_outbox(outbox, cfg["bridge"], log=log)
            raise
        watchlist, _ = load_watchlist(cfg)
        for kind, d, _sdt, age, _open in work:
            k = f"{kind}:{d}"
            entry = state.get(k) or {}
            attempts = entry.get("attempts", 0) + 1
            last_tick = age >= cfg["window_minutes"] - 6
            try:
                facts, md, summary, meta = build(
                    kind, d, cfg=cfg, fetcher=fetcher, calendar=cal, watchlist=watchlist, now_utc=now, key=key, log=log
                )
            except (
                Exception
            ) as e:  # a bug, not a data delay (fetch errors are handled inside): give up now, notify once
                log(f"{kind} {d}: internal error {type(e).__name__}: {e}\n{traceback.format_exc()}")
                state[k] = {**entry, "attempts": attempts}
                queue_failure(state, outbox, kind, d, f"internal error {type(e).__name__}: {e}"[:300], now, log)
                rc = rc or 1
                continue
            if facts["status"] == "closed":
                state[k] = {
                    "status": "closed",
                    "reason": "exchange reports no data",
                    "at": now.isoformat(timespec="seconds"),
                }
                log(f"{kind} {d}: market closed (exchange reports no data); no report")
                continue
            if facts["status"] == "must_missing":
                reason = "required data missing: " + "; ".join(facts["must_missing"])[:300]
                state[k] = {**entry, "status": "waiting", "attempts": attempts, "reason": reason}
                if last_tick:
                    queue_failure(state, outbox, kind, d, reason, now, log)
                    rc = rc or 1
                else:
                    log(f"{kind} {d}: {reason}; retry next tick")
                continue
            if facts["retry_gaps"] and age < cfg["complete_wait_minutes"]:
                state[k] = {
                    **entry,
                    "status": "waiting",
                    "attempts": attempts,
                    "reason": "; ".join(facts["retry_gaps"])[:300],
                }
                log(f"{kind} {d}: data incomplete ({state[k]['reason']}); retry next tick")
                continue
            save_json(state_dir / "runs" / f"{d}-{kind}.json", run_record(facts, meta, fetcher))
            entry = outbox_entry(facts, md, summary)
            save_json(outbox / f"{d}-{kind}.json", entry)
            try_archive(cfg, entry, log)
            state[k] = {
                "status": "queued",
                "attempts": attempts,
                "gaps": facts["gaps"],
                "at": now.isoformat(timespec="seconds"),
            }
            log(f"{kind} {d}: queued ({len(md)} chars, gaps={len(facts['gaps'])})")
        sweep_expired(state, outbox, cfg["slots"], cfg["window_minutes"], now, log)
        save_json(state_dir / "state.json", state)
        if not a.no_deliver and deliver.deliver_outbox(outbox, cfg["bridge"], log=log):
            rc = 4
        return rc


def cmd_run(a, cfg, state_dir, outbox, log):
    now = parse_now(a.now)
    slot = cfg["slots"][a.kind]
    d = dt.date.fromisoformat(a.date) if a.date else now.astimezone(ZoneInfo(slot["tz"])).date()
    fetcher = Fetcher()
    cal = build_calendar(fetcher, cfg, state_dir, {d.year, now.astimezone(TPE).year}, log)
    if not cal.is_open(slot["market"], d):
        log(f"{a.kind} {d}: market closed per calendar; building anyway (manual run)")
    key = get_key(cfg, a.no_llm)
    watchlist, wsrc = load_watchlist(cfg)
    facts, md, summary, meta = build(
        a.kind, d, cfg=cfg, fetcher=fetcher, calendar=cal, watchlist=watchlist, now_utc=now, key=key, log=log
    )
    rec = run_record(facts, meta, fetcher)
    rec["watchlist_file"] = wsrc
    if facts["status"] != "ok":
        log(f"{a.kind} {d}: {facts['status']} {facts['must_missing']}")
        if a.dry_run:
            out = expand(a.out) if a.out else state_dir / "dry-run"
            save_json(out / f"{d}-{a.kind}.run.json", rec)
        return 1
    entry = outbox_entry(facts, md, summary)
    if a.dry_run:
        out = expand(a.out) if a.out else state_dir / "dry-run"
        out.mkdir(parents=True, exist_ok=True)
        (out / entry["filename"]).write_text(md, encoding="utf-8")
        save_json(out / f"{d}-{a.kind}.outbox.json", entry)
        save_json(out / f"{d}-{a.kind}.run.json", rec)
        log(f"{a.kind} {d}: dry-run written to {out / entry['filename']} (gaps={facts['gaps']})")
        return 0
    save_json(state_dir / "runs" / f"{d}-{a.kind}.json", rec)
    save_json(outbox / f"{d}-{a.kind}.json", entry)
    try_archive(cfg, entry, log)
    state = load_state(state_dir)
    state[f"{a.kind}:{d}"] = {
        "status": "queued",
        "manual": True,
        "gaps": facts["gaps"],
        "at": now.isoformat(timespec="seconds"),
    }
    save_json(state_dir / "state.json", state)
    log(f"{a.kind} {d}: queued (manual)")
    return 0


def cmd_resend(a, cfg, state_dir, outbox, log):
    d = dt.date.fromisoformat(a.date)
    sent = outbox / "sent" / f"{d}-{a.kind}.json"
    md = None
    try:
        md = json.loads(sent.read_text(encoding="utf-8"))["report_md"]
        log(f"resend {a.kind} {d}: using delivered markdown {sent}")
    except (OSError, ValueError, KeyError):
        now = parse_now(a.now)
        fetcher = Fetcher()
        cal = build_calendar(fetcher, cfg, state_dir, {d.year}, log)
        watchlist, _ = load_watchlist(cfg)
        facts, md, _summary, _meta = build(
            a.kind, d, cfg=cfg, fetcher=fetcher, calendar=cal, watchlist=watchlist, now_utc=now, key=None, log=log
        )
        if facts["status"] != "ok":
            log(f"resend {a.kind} {d}: no delivered markdown and rebuild failed ({facts['status']})")
            return 1
        log(f"resend {a.kind} {d}: no delivered markdown; rebuilt without LLM")
    text = render.telegram_text(md)
    if a.dry_run:
        print(text)
        return 0
    ok = deliver.resend(a.kind, str(d), text, cfg["bridge"], log=log)
    return 0 if ok else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily market briefs (TW + US).")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_due = sub.add_parser("due")
    p_run = sub.add_parser("run")
    p_run.add_argument("kind", choices=tradingcal.KINDS)
    p_run.add_argument("--date")
    p_run.add_argument("--out")
    sub.add_parser("deliver")
    sub.add_parser("archive-backfill")
    p_resend = sub.add_parser("resend")
    p_resend.add_argument("kind", choices=tradingcal.KINDS)
    p_resend.add_argument("--date", required=True)
    p_resend.add_argument("--dry-run", action="store_true")
    p_resend.add_argument("--now", help=argparse.SUPPRESS)
    for p in (p_due, p_run):
        p.add_argument("--now", help="ISO time (tests/backfill); naive = Asia/Taipei")
        p.add_argument("--no-llm", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    p_due.add_argument("--no-deliver", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    state_dir = expand(cfg["state_dir"])
    outbox = expand(cfg["outbox_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(state_dir / "market-brief.log")
    try:
        if a.cmd == "due":
            return cmd_due(a, cfg, state_dir, outbox, log)
        if a.cmd == "run":
            return cmd_run(a, cfg, state_dir, outbox, log)
        if a.cmd == "archive-backfill":
            return cmd_archive_backfill(a, cfg, state_dir, outbox, log)
        if a.cmd == "resend":
            return cmd_resend(a, cfg, state_dir, outbox, log)
        with Lock(state_dir / "run.lock") as lock:
            if not lock.acquired:
                return 0
            return 4 if deliver.deliver_outbox(outbox, cfg["bridge"], log=log) else 0
    except llm.MissingKeyError as e:
        log(f"key file error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
