#!/usr/bin/env python3
"""bridge_sync tests (stdlib unittest; run with host python3 on stormtrooper, no Poseidon imports).

    python3 scripts/strategy_bridge/test_bridge_sync.py -v

FakeApi mimics /api/v1/pipelines (idempotent create keyed by actor + Idempotency-Key, run/stages/attempt reads);
FakeBridge mimics the macminim4 strategy-* verbs, including the per idea+event notify marker.
"""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bridge_sync as bs  # noqa: E402

KEY = "sentinel-key-" + "x" * 30
IDEA = "台指期夜盤前一日高低點反轉：價格突破前一日日盤高點後跌回高點下方做空，跌破低點後站回做多，停損 30 點"
NOON = datetime.datetime(2026, 10, 23, 12, 0, tzinfo=datetime.UTC)  # outside the default hold window
PLAN = [
    "spec_generate",
    "spec_validate",
    "backtest",
    "optimize",
    "param_stability",
    "dsr",
    "cost_stress",
    "regime_split",
    "wfe",
    "holdout",
]


class FakeApi:
    def __init__(self):
        self.runs: dict[str, dict] = {}
        self.by_key: dict[str, str] = {}
        self.stages: dict[str, list] = {}
        self.attempts: dict[tuple, dict] = {}
        self.posts = 0
        self.seen_headers = []
        self.fail_next_get = False
        self.plan = PLAN  # PAPER_PLAN for runs created after the shadow-paper branch merged

    def request(self, method, url, body, headers):
        self.seen_headers.append(dict(headers))
        assert headers["X-API-Key"] == KEY
        path = url.split("8011", 1)[1]
        if method == "POST":
            assert path == "/api/v1/pipelines"
            self.posts += 1
            assert set(body) == {"idea_text", "source"} and body["source"] == "telegram"
            key = headers["Idempotency-Key"]
            if key in self.by_key:
                return 201, dict(self.runs[self.by_key[key]], status="queued")  # replayed create response
            run_id = str(uuid.uuid4())
            run = {
                "id": run_id,
                "run_no": len(self.runs) + 10,
                "idea_text": body["idea_text"],
                "status": "queued",
                "created_at": "2026-10-23T09:00:00+00:00",
                "finished_at": None,
                "reason": None,
                "current_stage": self.plan[0],
                "config_json": {"stage_plan": list(self.plan), "thresholds": {"paper_min_trading_days": 20}},
            }
            self.runs[run_id] = run
            self.by_key[key] = run_id
            self.stages[run_id] = [
                {"stage": s, "seq": i, "latest": None, "attempts": []} for i, s in enumerate(self.plan)
            ]
            return 201, dict(run)
        if self.fail_next_get:
            self.fail_next_get = False
            raise OSError("connection refused")
        parts = path.strip("/").split("/")
        run_id = parts[3]
        if run_id not in self.runs:
            return 404, {"detail": "not found"}
        if len(parts) == 4:
            return 200, dict(self.runs[run_id])
        if parts[4] == "stages" and len(parts) == 5:
            return 200, {"run_id": run_id, "stages": self.stages[run_id]}
        return 200, self.attempts[(run_id, parts[5], int(parts[7]))]

    def advance(self, run_id, status, current_stage, results, reason=None):
        """A non-terminal state: results as in finish(); verdict None + status "waiting" = a held/waiting attempt."""
        self.runs[run_id].update(status=status, current_stage=current_stage, reason=reason)
        for entry in self.stages[run_id]:
            if entry["stage"] in results:
                stage_status, verdict, report, why = results[entry["stage"]]
                latest = {
                    "stage": entry["stage"],
                    "attempt": 1,
                    "status": stage_status,
                    "verdict": verdict,
                    "reason": why,
                }
                entry["latest"] = latest
                self.attempts[(run_id, entry["stage"], 1)] = dict(latest, artifact_json={"report": report, "files": []})

    def finish(self, run_id, status, results, reason=None):
        """results: {stage: (verdict|None for aborted, report, reason)}"""
        run = self.runs[run_id]
        run.update(status=status, finished_at="2026-10-23T09:42:30+00:00", reason=reason)
        for entry in self.stages[run_id]:
            if entry["stage"] in results:
                verdict, report, why = results[entry["stage"]]
                latest = {
                    "stage": entry["stage"],
                    "attempt": 1,
                    "status": "completed" if verdict else "aborted",
                    "verdict": verdict,
                    "reason": why,
                }
                entry["latest"] = latest
                self.attempts[(run_id, entry["stage"], 1)] = dict(latest, artifact_json={"report": report, "files": []})


class FakeBridge:
    """Implements the macminim4 verbs over an in-memory ideas table."""

    def __init__(self):
        self.ideas: dict[str, dict] = {}
        self.sent: list[tuple[str, str, str]] = []
        self.fail = {}  # verb -> number of upcoming calls that fail with exit 1
        self.calls = []

    def add(self, idea_id, text=IDEA):
        self.ideas[idea_id] = {
            "id": idea_id,
            "text": text,
            "status": "accepted",
            "source": "telegram",
            "created_at": "2026-10-23T17:00:00+08:00",
            "notified": {},
        }

    def __call__(self, cmd, input="", capture_output=True, text=True, timeout=None):
        assert cmd[:2] == ["ssh", "research-bridge"], cmd
        verb, args = cmd[2], cmd[3:]
        self.calls.append((verb, *args))
        if self.fail.get(verb):
            self.fail[verb] -= 1
            return types.SimpleNamespace(returncode=1, stdout="", stderr="telegram send failed")
        if verb == "strategy-pending":
            keep = ("id", "text", "source", "created_at")
            rows = [{k: r[k] for k in keep} for r in self.ideas.values() if r["status"] == "accepted"]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows, ensure_ascii=False), stderr="")
        row = self.ideas.get(args[0])
        if row is None:
            return types.SimpleNamespace(returncode=2, stdout="", stderr="strategy idea not found")
        if verb == "strategy-ack":
            run_no, run_id = int(args[1]), args[2]
            if row["status"] == "submitted":
                if (row["run_no"], row["run_id"]) != (run_no, run_id):
                    return types.SimpleNamespace(returncode=2, stdout="", stderr="already submitted")
                return types.SimpleNamespace(returncode=0, stdout=json.dumps({"duplicate": True}), stderr="")
            row.update(status="submitted", run_no=run_no, run_id=run_id)
            return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")
        if verb == "strategy-notify":
            event = args[1]
            if row["status"] != "submitted":
                return types.SimpleNamespace(returncode=2, stdout="", stderr="not submitted")
            if event in row["notified"]:
                return types.SimpleNamespace(returncode=0, stdout=json.dumps({"duplicate": True}), stderr="")
            row["notified"][event] = True
            self.sent.append((args[0], event, input))
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({"telegram_sent": True}), stderr="")
        raise AssertionError(verb)


def gate(name, ok, headline):
    return {"summary": f"{name} {'✅' if ok else '❌'} {headline}", "gate": name}


PAPER_PLAN = [*PLAN, "shadow_paper", "awaiting_release"]
PAPER_WAITING = {
    "summary": "shadow_paper ⏳ day 3/20 · 1/10 trades · replay match 100% · paper PnL +1,200 TWD",
    "trading_days": 3,
    "paper": {"n_trades": 1, "total_pnl_twd": 1200.0},
}
PAPER_PASSED = {
    "summary": "shadow_paper ✅ 23 trading days · 14 trades · replay match 93% · paper PnL +12,300 TWD",
    "trading_days": 23,
    "paper": {"n_trades": 14, "total_pnl_twd": 12300.0, "wins": 9},
    "consistency": {"matched": 13, "ratio": 0.9286},
    "snapshot": {"replay_match_ratio": 0.9286},
}
HELD = {"summary": "awaiting_release ⏸ passed shadow paper; awaiting human release (Step 9, after the soak gate)"}
FULL_PASS = {
    "spec_generate": (
        "pass",
        {"assumptions": [f"assumption {i}" for i in range(7)], "params": ["sl_buf", "tp_r"]},
        None,
    ),
    "spec_validate": ("pass", {"params": {"sl_buf": {"default": 0}, "tp_r": {"default": 2.0}}}, None),
    "backtest": ("pass", {"metrics": {"n_trades": 312, "total_pnl_twd": 123400.0, "sharpe_daily": 0.0412}}, None),
    "optimize": (
        "pass",
        {
            "n_trials": 45,
            "best": {"trial_no": 17, "params": {"sl_buf": 5, "tp_r": 2.5}, "metrics": {"sharpe_daily": 0.0731}},
        },
        "45 trials",
    ),
    **{g: ("pass", gate(g, True, "ok"), None) for g in PLAN[4:]},
}


class Harness:
    def __init__(self, tc):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        key = root / "api.key"
        key.write_text(KEY + "\n")
        key.chmod(0o600)
        self.cfg = dict(
            bs.DEFAULTS, key_file=str(key), state_dir=str(root / "state"), log_file=str(root / "bridge.log")
        )
        self.api, self.bridge = FakeApi(), FakeBridge()
        self.log = bs.Logger(self.cfg["log_file"], echo=False)
        tc.addCleanup(self.tmp.cleanup)

    def sync(self, **kw):
        kw.setdefault("now", NOON)
        return bs.sync(self.cfg, self.log, request=self.api.request, runner=self.bridge, **kw)

    def run_id(self, idea_id):
        return self.bridge.ideas[idea_id]["run_id"]

    def events(self):
        return [(i, e) for i, e, _ in self.bridge.sent]


class PullTests(unittest.TestCase):
    def test_new_idea_creates_one_run_acks_and_sends_start_once(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.api.posts, 1)
        row = h.bridge.ideas["20261023-s1"]
        self.assertEqual((row["status"], row["run_no"]), ("submitted", 10))
        self.assertEqual(h.events(), [("20261023-s1", "start")])
        self.assertEqual(h.bridge.sent[0][2], "pipeline #10 開始：" + IDEA)  # 52 chars: not truncated
        for _ in range(3):
            self.assertEqual(h.sync(), 0)
        self.assertEqual((h.api.posts, len(h.bridge.sent)), (1, 1))
        self.assertEqual(h.api.seen_headers[0]["Idempotency-Key"], "telegram-idea-20261023-s1")

    def test_crash_between_post_and_ack_creates_no_duplicate_run(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        h.bridge.fail["strategy-ack"] = 1
        self.assertEqual(h.sync(), 1)  # POST ok, ack failed; start notify refused (not submitted)
        self.assertEqual(h.bridge.ideas["20261023-s1"]["status"], "accepted")
        # Simulate a crash before the local state was written: the next tick has no memory of the POST.
        for p in Path(h.cfg["state_dir"]).rglob("*.json"):
            p.unlink()
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.api.posts, 2)
        self.assertEqual(len(h.api.runs), 1)  # the API replayed the same run for the same Idempotency-Key
        self.assertEqual(h.bridge.ideas["20261023-s1"]["status"], "submitted")
        self.assertEqual(h.events(), [("20261023-s1", "start")])

    def test_ack_failure_is_retried_without_a_second_post(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        h.bridge.fail["strategy-ack"] = 1
        h.sync()
        h.sync()
        self.assertEqual(h.api.posts, 1)
        self.assertEqual(h.bridge.ideas["20261023-s1"]["status"], "submitted")
        self.assertEqual(h.events(), [("20261023-s1", "start")])

    def test_permanent_rejection_is_not_retried(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        h.api.request = lambda m, u, b, hd: (422, {"detail": "idea_text too long"}) if m == "POST" else (404, None)
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.sync(), 0)
        self.assertEqual(sum("rejected http 422" in line for line in h.log.lines), 1)

    def test_malformed_pending_rows_are_skipped(self):
        h = Harness(self)
        h.bridge.ideas["bad"] = {
            "id": "../x",
            "text": "t",
            "status": "accepted",
            "source": "telegram",
            "created_at": "",
        }
        h.bridge.add("20261023-s2", text="多" * 2001)
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.api.posts, 0)

    def test_dry_run_writes_nothing(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        self.assertEqual(h.sync(dry_run=True), 0)
        self.assertEqual((h.api.posts, h.bridge.sent), (0, []))
        self.assertEqual(list(Path(h.cfg["state_dir"]).rglob("*.json")), [])


class HoldTests(unittest.TestCase):
    def test_no_new_runs_inside_the_hold_window_but_notifications_continue(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        h.sync()
        h.api.finish(h.run_id("20261023-s1"), "completed", FULL_PASS)
        h.bridge.add("20261023-s2", text="另一個 idea")
        inside = datetime.datetime(2026, 10, 23, 7, 0, tzinfo=datetime.UTC)
        for _ in range(3):
            self.assertEqual(h.sync(now=inside), 0)
        self.assertEqual(h.api.posts, 1)
        self.assertEqual(h.bridge.ideas["20261023-s2"]["status"], "accepted")
        self.assertEqual(sum("held: no new runs until 08:40Z" in line for line in h.log.lines), 1)
        self.assertIn(("20261023-s1", "completed"), h.events())  # push is not held
        h.sync(now=datetime.datetime(2026, 10, 23, 8, 40, tzinfo=datetime.UTC))
        self.assertEqual((h.api.posts, h.bridge.ideas["20261023-s2"]["status"]), (2, "submitted"))

    def test_held_until(self):
        at = lambda hh, mm: datetime.datetime(2026, 10, 23, hh, mm, tzinfo=datetime.UTC)  # noqa: E731
        self.assertEqual(bs.held_until([["06:00", "08:40"]], at(6, 0)), "08:40")
        self.assertIsNone(bs.held_until([["06:00", "08:40"]], at(5, 59)))
        self.assertIsNone(bs.held_until([], at(7, 0)))
        self.assertEqual(bs.held_until([["23:00", "01:00"]], at(0, 30)), "01:00")


def completed(results):
    return {stage: ("completed", verdict, report, why) for stage, (verdict, report, why) in results.items()}


class MilestoneTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(self)
        self.h.api.plan = PAPER_PLAN
        self.h.bridge.add("20261023-s1")
        self.h.sync()
        self.run_id = self.h.run_id("20261023-s1")

    def enter_paper(self):
        self.h.api.advance(
            self.run_id,
            "waiting",
            "shadow_paper",
            {**completed(FULL_PASS), "shadow_paper": ("waiting", None, PAPER_WAITING, PAPER_WAITING["summary"])},
        )

    def test_gates_passed_starts_shadow_paper_once_with_gate_lines(self):
        h = self.h
        h.api.advance(self.run_id, "running", "wfe", completed({k: FULL_PASS[k] for k in PLAN[:8]}))
        h.sync()
        self.assertEqual([e for _, e in h.events()], ["start"])  # holdout not yet passed
        self.enter_paper()
        for _ in range(3):
            self.assertEqual(h.sync(), 0)
        self.assertEqual([e for _, e in h.events()], ["start", "shadow_paper"])
        lines = h.bridge.sent[1][2].splitlines()
        self.assertEqual(lines[0], "pipeline #10 通過全部驗證關卡，開始 shadow paper（至少 20 個交易日）")
        self.assertEqual(lines[2:8], [f"{g} ✅ ok" for g in PLAN[4:]])
        self.assertEqual(lines[8], "參數（optimize 最佳）：`sl_buf=5`, `tp_r=2.5`")
        self.assertNotIn("shadow_paper ⏳", h.bridge.sent[1][2])

    def test_awaiting_release_once_with_paper_summary(self):
        h = self.h
        self.enter_paper()
        h.sync()
        held = {**completed(FULL_PASS), "shadow_paper": ("completed", "pass", PAPER_PASSED, PAPER_PASSED["summary"])}
        held["awaiting_release"] = ("waiting", None, HELD, "passed shadow paper; awaiting human release")
        h.api.advance(self.run_id, "waiting", "awaiting_release", held, reason="passed shadow paper; awaiting ...")
        for _ in range(3):
            self.assertEqual(h.sync(), 0)
        self.assertEqual([e for _, e in h.events()], ["start", "shadow_paper", "awaiting_release"])
        lines = h.bridge.sent[2][2].splitlines()
        self.assertEqual(lines[0], "pipeline #10 通過 shadow paper，等待人工 release")
        self.assertEqual(lines[2], PAPER_PASSED["summary"])
        self.assertEqual(lines[3], "交易日 23，交易 14 筆，paper PnL +12,300 TWD，replay match 93%")
        rec = json.loads((Path(h.cfg["state_dir"]) / "runs" / f"{self.run_id}.json").read_text())
        self.assertFalse(rec["closed"])  # still watched: a human abort or Step 9 release ends it later

    def test_milestone_dedupe_survives_lost_local_state(self):
        h = self.h
        self.enter_paper()
        h.sync()
        path = Path(h.cfg["state_dir"]) / "runs" / f"{self.run_id}.json"
        rec = json.loads(path.read_text())
        rec["notified"] = ["start"]
        path.write_text(json.dumps(rec))
        h.sync()
        self.assertEqual([e for _, e in h.events()], ["start", "shadow_paper"])
        self.assertTrue(any("shadow_paper (bridge: duplicate" in line for line in h.log.lines))

    def test_both_milestones_reached_between_ticks_go_out_in_order(self):
        h = self.h
        held = {**completed(FULL_PASS), "shadow_paper": ("completed", "pass", PAPER_PASSED, None)}
        held["awaiting_release"] = ("waiting", None, HELD, None)
        h.api.advance(self.run_id, "waiting", "awaiting_release", held)
        h.sync()
        self.assertEqual([e for _, e in h.events()], ["start", "shadow_paper", "awaiting_release"])

    def test_shadow_paper_failure_is_in_the_terminal_summary(self):
        h = self.h
        self.enter_paper()
        h.sync()
        fail = dict(PAPER_PASSED, summary="shadow_paper ❌ replay match 71% < 90% after 22 trading days")
        h.api.finish(self.run_id, "failed", {**FULL_PASS, "shadow_paper": ("fail", fail, "replay match 71% < 90%")})
        h.sync()
        self.assertEqual([e for _, e in h.events()], ["start", "shadow_paper", "failed"])
        text = h.bridge.sent[2][2]
        self.assertIn("shadow_paper ❌ replay match 71% < 90% after 22 trading days", text)
        self.assertIn("未執行：awaiting_release", text)

    def test_plan_without_paper_stages_never_reads_stages_before_terminal(self):
        h = Harness(self)  # integration before the shadow branch: plan ends at holdout
        h.bridge.add("20261023-s1")
        h.sync()
        run_id = h.run_id("20261023-s1")
        h.api.advance(run_id, "running", "holdout", completed({k: FULL_PASS[k] for k in PLAN[:9]}))
        before = len(h.api.seen_headers)
        h.sync()
        self.assertEqual(len(h.api.seen_headers) - before, 1)  # only the run status read
        self.assertEqual([e for _, e in h.events()], ["start"])

    def test_milestone_reached_is_generic(self):
        stages = [
            {"stage": "a", "latest": {"status": "completed", "verdict": "pass"}},
            {"stage": "shadow_paper", "latest": {"status": "running", "verdict": None}},
            {"stage": "awaiting_release", "latest": None},
        ]
        self.assertTrue(bs.milestone_reached("shadow_paper", stages))
        self.assertFalse(bs.milestone_reached("awaiting_release", stages))
        self.assertFalse(bs.milestone_reached("missing", stages))
        stages[0]["latest"]["verdict"] = "fail"
        self.assertFalse(bs.milestone_reached("shadow_paper", stages))


class PushTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(self)
        self.h.bridge.add("20261023-s1")
        self.h.sync()
        self.run_id = self.h.run_id("20261023-s1")

    def test_terminal_summary_once_across_syncs_and_lost_state(self):
        h = self.h
        h.api.finish(self.run_id, "completed", FULL_PASS, reason="passed all validation gates")
        h.sync()
        h.sync()
        self.assertEqual(h.events(), [("20261023-s1", "start"), ("20261023-s1", "completed")])
        # Crash after the bridge sent but before the local state was saved: the bridge marker stops a resend.
        rec = json.loads((Path(h.cfg["state_dir"]) / "runs" / f"{self.run_id}.json").read_text())
        rec.update(notified=["start"], closed=False)
        (Path(h.cfg["state_dir"]) / "runs" / f"{self.run_id}.json").write_text(json.dumps(rec))
        h.sync()
        self.assertEqual(len(h.bridge.sent), 2)
        self.assertTrue(any("bridge: duplicate" in line for line in h.log.lines))
        rec = json.loads((Path(h.cfg["state_dir"]) / "runs" / f"{self.run_id}.json").read_text())
        self.assertTrue(rec["closed"])
        calls = len(h.bridge.calls)
        h.sync()  # closed runs are no longer polled or notified
        self.assertEqual([c[0] for c in h.bridge.calls[calls:]], ["strategy-pending"])

    def test_notify_failure_is_retried_next_tick(self):
        h = self.h
        h.api.finish(self.run_id, "failed", {"spec_generate": FULL_PASS["spec_generate"]})
        h.bridge.fail["strategy-notify"] = 1
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.events(), [("20261023-s1", "start"), ("20261023-s1", "failed")])

    def test_rerunnable_status_stays_watched_and_a_new_status_is_notified(self):
        h = self.h
        h.api.finish(self.run_id, "error", {"spec_generate": ("error", {}, "llm request failed")}, reason="llm down")
        h.sync()
        rec = json.loads((Path(h.cfg["state_dir"]) / "runs" / f"{self.run_id}.json").read_text())
        self.assertFalse(rec["closed"])
        h.api.finish(self.run_id, "failed", {"spec_generate": FULL_PASS["spec_generate"]})
        h.sync()
        h.sync()
        self.assertEqual([e for _, e in h.events()], ["start", "error", "failed"])

    def test_start_is_sent_before_the_summary_even_when_the_run_is_already_terminal(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        h.bridge.fail["strategy-notify"] = 1  # start fails on the first tick
        h.sync()
        run_id = h.run_id("20261023-s1")
        h.api.finish(run_id, "unsupported", {"spec_generate": ("unsupported", {"missing_capabilities": ["htf"]}, "x")})
        h.sync()
        self.assertEqual([e for _, e in h.events()], ["start", "unsupported"])

    def test_status_read_failure_is_counted_and_retried(self):
        h = self.h
        h.api.finish(self.run_id, "completed", FULL_PASS)
        h.api.fail_next_get = True
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.sync(), 0)
        self.assertEqual([e for _, e in h.events()], ["start", "completed"])


class FormatTests(unittest.TestCase):
    def run_and_stages(self, status, results, reason=None):
        api = FakeApi()
        _, run = api.request(
            "POST",
            "http://x:8011/api/v1/pipelines",
            {"idea_text": IDEA, "source": "telegram"},
            {"X-API-Key": KEY, "Idempotency-Key": "k"},
        )
        api.finish(run["id"], status, results, reason=reason)
        stages, reports = bs.fetch_terminal_detail(bs.Api("http://x:8011", KEY, request=api.request), run["id"])
        return api.runs[run["id"]], stages, reports

    def test_failed_gate_summary(self):
        results = {k: v for k, v in FULL_PASS.items() if k in PLAN[:5]}
        results["param_stability"] = (
            "fail",
            gate("param_stability", False, "neighbourhood profitable 40% < 60%"),
            "neighbourhood profitable 40% < 60%",
        )
        run, stages, reports = self.run_and_stages("failed", results, reason="param_stability failed")
        text = bs.terminal_message(run, stages, reports)
        lines = text.splitlines()
        self.assertEqual(lines[0], "pipeline #10 ❌ 未通過：" + IDEA)
        self.assertEqual(
            bs.start_message({"run_no": 1, "idea_text": "多\n" * 80}), "pipeline #1 開始：" + "多 " * 29 + "多…"
        )
        self.assertEqual(
            lines[1:5],
            [
                "spec_generate ✅",
                "spec_validate ✅",
                "backtest ✅ 交易 312 筆，PnL +123,400 TWD，Sharpe(日) 0.041",
                "optimize ✅ 45 trials，最佳 #17 Sharpe(日) 0.073",
            ],
        )
        self.assertEqual(lines[5], "param_stability ❌ neighbourhood profitable 40% < 60%")
        self.assertEqual(lines[6], "未執行：dsr, cost_stress, regime_split, wfe, holdout")
        self.assertEqual(lines[7], "假設：")
        self.assertEqual(lines[8:14], [f"• assumption {i}" for i in range(5)] + ["• …另有 2 條"])
        self.assertEqual(lines[14], "參數（optimize 最佳）：`sl_buf=5`, `tp_r=2.5`")
        self.assertEqual(lines[15], "耗時 42 分")
        self.assertNotIn("http", text)  # no kairos_base_url -> no link
        self.assertNotIn("原因", text)

    def test_completed_with_link(self):
        run, stages, reports = self.run_and_stages("completed", FULL_PASS, reason="passed all validation gates")
        text = bs.terminal_message(run, stages, reports, "https://kairos.example/")
        self.assertIn("✅ 通過全部驗證關卡", text.splitlines()[0])
        run["reason"] = "all stages passed"  # shortened stage_plan
        self.assertIn("✅ 完成：", bs.terminal_message(run, stages, reports).splitlines()[0])
        self.assertEqual(sum(line.endswith("✅ ok") for line in text.splitlines()), 6)
        self.assertEqual(text.splitlines()[-1], f"https://kairos.example/pipelines?run={run['id']}")
        self.assertNotIn("未執行", text)

    def test_unsupported_shows_missing_capabilities_and_reason(self):
        results = {
            "spec_generate": (
                "unsupported",
                {"missing_capabilities": ["htf", "order_book"], "assumptions": []},
                "unsupported: needs 1h bars",
            )
        }
        run, stages, reports = self.run_and_stages("unsupported", results, reason="unsupported: needs 1h bars")
        text = bs.terminal_message(run, stages, reports)
        self.assertIn("spec_generate ⛔ unsupported: needs 1h bars", text)
        self.assertIn("缺少能力：htf, order_book", text)
        self.assertIn("原因：unsupported: needs 1h bars", text)
        self.assertNotIn("假設", text)

    def test_params_fall_back_to_defaults_then_names(self):
        results = {
            "spec_generate": FULL_PASS["spec_generate"],
            "spec_validate": FULL_PASS["spec_validate"],
            "backtest": ("error", {}, "thalassa timeout"),
        }
        run, stages, reports = self.run_and_stages("error", results, reason="backtest error")
        text = bs.terminal_message(run, stages, reports)
        self.assertIn("參數（預設值）：`sl_buf=0`, `tp_r=2.0`", text)
        self.assertIn("backtest ⚠️ thalassa timeout", text)
        reports.pop("spec_validate")
        self.assertIn("參數：`sl_buf`, `tp_r`", bs.terminal_message(run, stages, reports))

    def test_aborted_stage_and_wall_time_formats(self):
        results = {"spec_generate": (None, {}, "abort requested")}
        run, stages, reports = self.run_and_stages("aborted", results, reason="owner abort")
        text = bs.terminal_message(run, stages, reports)
        self.assertIn("spec_generate ⏹ abort requested", text)
        self.assertIn("⏹ 已中止", text.splitlines()[0])
        self.assertEqual(
            bs.wall_time({"created_at": "2026-10-23T09:00:00+00:00", "finished_at": "2026-10-23T10:03:00+00:00"}),
            "1 小時 3 分",
        )
        self.assertEqual(
            bs.wall_time({"created_at": "2026-10-23T09:00:00+00:00", "finished_at": "2026-10-23T09:00:35+00:00"}),
            "35 秒",
        )
        self.assertEqual(
            bs.wall_time({"created_at": "2026-10-23T09:00:00+00:00", "finished_at": "2026-10-23T09:04:05+00:00"}),
            "4 分 5 秒",
        )
        self.assertIsNone(bs.wall_time({"created_at": "x"}))

    def test_message_is_capped(self):
        results = {"spec_generate": ("error", {}, "x" * 50)}
        run, stages, reports = self.run_and_stages("error", results, reason="y")
        reports["spec_generate"] = {"assumptions": ["a" * 300] * 50}
        self.assertLessEqual(len(bs.terminal_message(run, stages, reports)), bs.MESSAGE_CAP)


class SafetyTests(unittest.TestCase):
    def test_key_never_logged_and_permissions_enforced(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        h.sync()
        h.api.finish(h.run_id("20261023-s1"), "completed", FULL_PASS)
        h.sync()
        self.assertNotIn(KEY, Path(h.cfg["log_file"]).read_text())
        self.assertNotIn(KEY, json.dumps([c for c in h.bridge.calls], ensure_ascii=False))
        Path(h.cfg["key_file"]).chmod(0o644)
        self.assertEqual(h.sync(), 3)
        self.assertIn("mode 600", h.log.lines[-1])
        self.assertNotIn(KEY, h.log.lines[-1])

    def test_lock_busy_exits_zero_without_work(self):
        h = Harness(self)
        h.bridge.add("20261023-s1")
        state = bs.State(h.cfg["state_dir"])
        state.ensure()
        with bs.Lock(state.root / "sync.lock"):
            self.assertEqual(h.sync(), 0)
        self.assertEqual(h.api.posts, 0)

    def test_config_file_merges_over_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bridge.json"
            p.write_text(json.dumps({"kairos_base_url": "https://k"}))
            cfg = bs.load_config(str(p))
        self.assertEqual((cfg["kairos_base_url"], cfg["api_base"]), ("https://k", "http://127.0.0.1:8011"))


if __name__ == "__main__":
    unittest.main()
