#!/usr/bin/env python3
"""bridge_sync tests (stdlib unittest; run with host python3 on stormtrooper, no Poseidon imports).

    python3 scripts/strategy_bridge/test_bridge_sync.py -v

FakeApi mimics /api/v1/pipelines (idempotent create keyed by actor + Idempotency-Key, run/stages/attempt reads);
FakeBridge mimics the macminim4 strategy-* verbs, including the per idea+event notify marker.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "strategy_scout"))  # repo layout
sys.path.insert(0, str(HERE.parent / "scout-bin"))  # deployed: ~/strategy-pipeline/{bridge,scout-bin}

import bridge_sync as bs  # noqa: E402
import test_scout  # noqa: E402  (ScoutFixture: the real scout.propose with fake LLM / API)

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
        self.bodies = []
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
            assert set(body) - {"original"} == {"idea_text", "source"} and body["source"] in ("telegram", "scout")
            self.bodies.append(body)
            key = headers["Idempotency-Key"]
            assert body["source"] != "scout" or key.startswith("scout-proposal-"), key
            if key in self.by_key:
                return 201, dict(self.runs[self.by_key[key]], status="queued")  # replayed create response
            run_id = str(uuid.uuid4())
            run = {
                "id": run_id,
                "run_no": len(self.runs) + 10,
                "idea_text": body["idea_text"],
                "source": body["source"],
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
        self.proposals: dict[str, dict] = {}  # batch -> {"items", "text", "sent"}
        self.sent: list[tuple[str, str, str]] = []
        self.fail = {}  # verb -> number of upcoming calls that fail with exit 1
        self.calls = []
        self.capability = None  # (sha256, brief) last stored by strategy-capability

    def add(self, idea_id, text=IDEA):
        self.ideas[idea_id] = {
            "id": idea_id,
            "text": text,
            "status": "accepted",
            "source": "telegram",
            "created_at": "2026-10-23T17:00:00+08:00",
            "notified": {},
        }

    def accept(self, batch, *numbers, text=None):
        """What strategy_ideas.py accept does on macminim4: queue the proposal's idea_text under its id."""
        for n in numbers:
            item = self.proposals[batch]["items"][n - 1]
            self.add(item["id"], text=item["idea_text"] if text is None else text)
            self.ideas[item["id"]]["source"] = "scout"

    def __call__(self, cmd, input="", capture_output=True, text=True, timeout=None):
        assert cmd[:2] == ["ssh", "research-bridge"], cmd
        verb, args = cmd[2], cmd[3:]
        self.calls.append((verb, *args))
        if self.fail.get(verb):
            self.fail[verb] -= 1
            return types.SimpleNamespace(returncode=1, stdout="", stderr="telegram send failed")
        if verb == "strategy-proposals":
            batch, payload = args[0], json.loads(input)
            assert payload["batch"] == batch and isinstance(payload["text"], str)
            for n in range(1, len(payload["proposals"]) + 1):
                if f"\n{n}. " not in "\n" + payload["text"]:
                    return types.SimpleNamespace(returncode=2, stdout="", stderr=f"message does not show number {n}")
            known = self.proposals.get(batch)
            if known:
                if known["items"] != payload["proposals"]:
                    return types.SimpleNamespace(
                        returncode=2, stdout="", stderr="batch recorded with different proposals"
                    )
                return types.SimpleNamespace(returncode=0, stdout=json.dumps({"duplicate": True}), stderr="")
            self.proposals[batch] = {"items": payload["proposals"], "text": payload["text"]}
            self.sent.append((batch, "proposals", payload["text"]))
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({"telegram_sent": True}), stderr="")
        if verb == "strategy-capability":
            digest = args[0]
            if hashlib.sha256(input.encode("utf-8")).hexdigest() != digest:
                return types.SimpleNamespace(returncode=2, stdout="", stderr="sha256 does not match the brief")
            state = "unchanged" if self.capability and self.capability[0] == digest else "updated"
            self.capability = (digest, input)
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({"state": state}), stderr="")
        if verb == "strategy-pending":
            keep = ("id", "text", "source", "created_at", "original", "original_error")
            rows = [{k: r[k] for k in keep if k in r} for r in self.ideas.values() if r["status"] == "accepted"]
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
        self.scout = test_scout.ScoutFixture(root / "scout-host")
        self.cfg = dict(
            bs.DEFAULTS,
            key_file=str(key),
            state_dir=str(root / "state"),
            log_file=str(root / "bridge.log"),
            scout_dir=self.scout.cfg["state_dir"],
            capability_file=str(test_scout.scout.CAPABILITY_FILE),
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

    def test_drafted_idea_posts_the_original_beside_the_draft(self):
        h = Harness(self)
        h.bridge.add("20261026-s1")
        original = {"text": "240MA 附近是估值目標，左側分批買", "sha256": "a" * 64, "drafted_by": "sisyphus"}
        h.bridge.ideas["20261026-s1"]["original"] = original
        h.bridge.add("20261026-s2")
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.api.bodies[0], {"idea_text": IDEA, "source": "telegram", "original": original})
        self.assertEqual(h.api.bodies[1], {"idea_text": IDEA, "source": "telegram"})  # owner's own idea: unchanged

    def test_drafted_idea_without_its_original_gets_no_run(self):
        h = Harness(self)
        h.bridge.add("20261026-s1")
        h.bridge.ideas["20261026-s1"]["original_error"] = "original does not match original_sha256"
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.sync(), 0)
        self.assertEqual((h.api.posts, h.bridge.ideas["20261026-s1"]["status"]), (0, "accepted"))
        refused = [line for line in h.log.lines if "refused: original unavailable" in line]
        self.assertEqual(len(refused), 1, h.log.lines)

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
            bs.start_message({"run_no": 1, "idea_text": "多\n" * 80}), "pipeline #1 開始：" + "多\n" * 79 + "多"
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

    def test_long_idea_is_shortened_before_the_link(self):
        run, stages, reports = self.run_and_stages("completed", FULL_PASS, reason="passed all validation gates")
        run["idea_text"] = "多" * bs.IDEA_MAX_CHARS
        reports["spec_generate"] = {**reports["spec_generate"], "assumptions": ["a" * 160] * 5}
        base = "https://kairos.example"
        for text in (
            bs.terminal_message(run, stages, reports, base),
            bs.milestone_message("shadow_paper", run, stages, reports, base),
        ):
            self.assertLessEqual(len(text), bs.MESSAGE_CAP)
            self.assertEqual(text.splitlines()[-1], f"{base}/pipelines?run={run['id']}")
        full = bs.milestone_message("shadow_paper", run, stages, reports)
        self.assertIn("多" * bs.IDEA_MAX_CHARS, full)  # fits: full idea, no marker
        self.assertNotIn(bs.IDEA_OVERFLOW, full)
        run["idea_text"] = "多" * 3790
        shortened = bs.terminal_message(run, stages, reports, base)
        self.assertIn(bs.IDEA_OVERFLOW, shortened)
        self.assertEqual(len(shortened), bs.MESSAGE_CAP)


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


SCOUT_NOW = datetime.datetime(2026, 10, 23, 7, 30, tzinfo=test_scout.scout.TZ)  # batch 20261023


class ScoutBridgeTests(unittest.TestCase):
    """scout.propose (real, fake LLM) -> bridge deliver -> 「接受策略 N」 on macminim4 (FakeBridge) -> runs."""

    def setUp(self):
        self.h = Harness(self)
        self.assertEqual(self.h.scout.propose(SCOUT_NOW), 0, self.h.scout.lines)
        self.ids = [r["id"] for r in self.h.scout.proposals()]

    def deliveries(self):
        return [b for b, e, _ in self.h.bridge.sent if e == "proposals"]

    def test_batch_is_delivered_exactly_once(self):
        h = self.h
        for _ in range(3):
            self.assertEqual(h.sync(), 0)
        self.assertEqual(self.deliveries(), ["20261023"])
        delivered = h.bridge.proposals["20261023"]
        self.assertEqual([i["id"] for i in delivered["items"]], self.ids)
        self.assertEqual(delivered["items"][0]["idea_text"], test_scout.IDEAS[0]["idea_text"])
        self.assertTrue(delivered["text"].startswith("策略 scout 提案 2026-10-23（3 則）"))
        self.assertEqual(sum(c[0] == "strategy-proposals" for c in h.bridge.calls), 1)  # local marker saves calls
        marker = json.loads((Path(h.cfg["state_dir"]) / "delivered" / "20261023").read_text())
        self.assertEqual(marker["ids"], self.ids)
        # Lost local marker (crash after the send): the bridge answers duplicate and nothing is resent.
        (Path(h.cfg["state_dir"]) / "delivered" / "20261023").unlink()
        self.assertEqual(h.sync(), 0)
        self.assertEqual(self.deliveries(), ["20261023"])
        self.assertTrue(any("duplicate, not resent" in line for line in h.log.lines))
        self.assertEqual(h.api.posts, 0)  # delivery alone never creates a run

    def test_failed_send_is_retried_and_a_refusal_is_not(self):
        h = self.h
        h.bridge.fail["strategy-proposals"] = 1
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.sync(), 0)
        self.assertEqual(self.deliveries(), ["20261023"])
        h2 = Harness(self)
        h2.scout.propose(SCOUT_NOW)
        outbox = Path(h2.cfg["scout_dir"]) / "outbox" / "20261023.json"
        batch = json.loads(outbox.read_text())
        outbox.write_text(json.dumps(dict(batch, text="no numbers"), ensure_ascii=False))
        self.assertEqual(h2.sync(), 1)
        self.assertEqual(h2.sync(), 0)
        self.assertEqual(sum(c[0] == "strategy-proposals" for c in h2.bridge.calls), 1)
        self.assertEqual(sum("refused by the bridge" in line for line in h2.log.lines), 1)

    def test_accepted_proposals_become_scout_runs_and_are_marked_in_proposals_jsonl(self):
        h = self.h
        h.sync()
        h.bridge.accept("20261023", 1, 3)
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.api.posts, 2)
        runs = sorted(h.api.runs.values(), key=lambda r: r["run_no"])
        self.assertEqual([r["source"] for r in runs], ["scout", "scout"])
        self.assertEqual(
            [r["idea_text"] for r in runs], [test_scout.IDEAS[0]["idea_text"], test_scout.IDEAS[2]["idea_text"]]
        )
        self.assertEqual(
            sorted(hd["Idempotency-Key"] for hd in h.api.seen_headers if "Idempotency-Key" in hd),
            [f"scout-proposal-{self.ids[0]}", f"scout-proposal-{self.ids[2]}"],
        )
        rows = {r["id"]: r for r in h.scout.proposals()}
        for pid in (self.ids[0], self.ids[2]):
            self.assertEqual(rows[pid]["status"], "accepted")
            self.assertEqual(rows[pid]["run_id"], h.run_id(pid))
            self.assertEqual(rows[pid]["run_no"], h.bridge.ideas[pid]["run_no"])
        self.assertEqual(rows[self.ids[1]]["status"], "proposed")
        self.assertEqual(h.bridge.ideas[self.ids[0]]["status"], "submitted")
        starts = [t for i, e, t in h.bridge.sent if e == "start"]
        self.assertEqual(len(starts), 2)
        self.assertTrue(starts[0].startswith(f"pipeline #10 開始（策略提案 {self.ids[0]}）："), starts[0])
        for _ in range(2):
            h.sync()
        self.assertEqual(h.api.posts, 2)
        self.assertEqual(sum(e == "start" for _, e, _ in h.bridge.sent), 2)
        # proposals.jsonl is marked before the ack: a failed ack is retried without a second POST.
        h.bridge.accept("20261023", 2)
        h.bridge.fail["strategy-ack"] = 1
        h.sync()
        self.assertEqual({r["id"]: r for r in h.scout.proposals()}[self.ids[1]]["status"], "accepted")
        h.sync()
        self.assertEqual(h.bridge.ideas[self.ids[1]]["status"], "submitted")
        self.assertEqual(h.api.posts, 3)

    def test_proposal_runs_off_waits_and_logs_once_then_creates_when_flipped(self):
        h = self.h
        h.cfg["proposal_runs"] = False
        h.sync()
        h.bridge.accept("20261023", 2)
        h.bridge.add("20261023-s1")  # Telegram ideas are not affected by the switch
        for _ in range(3):
            self.assertEqual(h.sync(), 0)
        self.assertEqual(h.api.posts, 1)
        self.assertEqual(h.bridge.ideas[self.ids[1]]["status"], "accepted")
        self.assertEqual(sum("proposal_runs is false" in line for line in h.log.lines), 1)
        self.assertEqual({r["id"]: r for r in h.scout.proposals()}[self.ids[1]]["status"], "proposed")
        h.cfg["proposal_runs"] = True
        self.assertEqual(h.sync(), 0)
        self.assertEqual((h.api.posts, h.bridge.ideas[self.ids[1]]["status"]), (2, "submitted"))

    def test_dry_run_reports_acceptance_without_creating_anything(self):
        h = self.h
        self.assertEqual(h.sync(dry_run=True), 0)
        self.assertEqual(self.deliveries(), [])
        self.assertTrue(any("dry-run: would deliver scout batch 20261023 (3 proposals" in line for line in h.log.lines))
        h.sync()
        h.bridge.accept("20261023", 1)
        self.assertEqual(h.sync(dry_run=True), 0)
        self.assertEqual(h.api.posts, 0)
        self.assertTrue(any(f"dry-run: would create a run for idea {self.ids[0]}" in line for line in h.log.lines))
        self.assertEqual({r["id"]: r for r in h.scout.proposals()}[self.ids[0]]["status"], "proposed")

    def test_unknown_or_altered_proposals_are_refused(self):
        h = self.h
        h.sync()
        h.bridge.add("20261023-p9")  # not a proposal of this host
        h.bridge.accept("20261023", 1, text="完全不同的策略文字，不是 scout 寫的那一則提案")
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.sync(), 0)  # refused once, not retried
        self.assertEqual(h.api.posts, 0)
        self.assertEqual(sum("refused: not in proposals.jsonl" in line for line in h.log.lines), 1)
        self.assertEqual(sum("refused: text differs" in line for line in h.log.lines), 1)

    def test_whitespace_differences_are_not_a_text_change(self):
        h = self.h
        h.sync()
        h.bridge.accept("20261023", 1, text="  " + test_scout.IDEAS[0]["idea_text"].replace("：", "： ") + "\n")
        self.assertEqual(h.sync(), 0)
        (run,) = h.api.runs.values()
        self.assertEqual(run["idea_text"], test_scout.IDEAS[0]["idea_text"])  # this host's text is used

    def test_hold_window_applies_to_proposals(self):
        h = self.h
        h.sync()
        h.bridge.accept("20261023", 1)
        inside = datetime.datetime(2026, 10, 24, 7, 0, tzinfo=datetime.UTC)
        self.assertEqual(h.sync(now=inside), 0)
        self.assertEqual(h.api.posts, 0)
        h.sync(now=datetime.datetime(2026, 10, 24, 8, 41, tzinfo=datetime.UTC))
        self.assertEqual(h.api.posts, 1)

    def test_old_batches_are_not_delivered_and_partial_json_is_retried(self):
        h = self.h
        late = datetime.datetime(2026, 10, 31, 12, 0, tzinfo=datetime.UTC)  # 8 days after the batch date
        self.assertEqual(h.sync(now=late), 0)
        self.assertEqual(self.deliveries(), [])
        self.assertEqual(sum("8 days old, not delivered" in line for line in h.log.lines), 1)
        h2 = Harness(self)
        outbox = Path(h2.cfg["scout_dir"]) / "outbox"
        outbox.mkdir(parents=True)
        (outbox / "20261023.json").write_text('{"date": "20261023", "ids": [')
        self.assertEqual(h2.sync(), 1)
        self.assertEqual(h2.bridge.calls.count(("strategy-proposals", "20261023")), 0)
        self.assertFalse((Path(h2.cfg["state_dir"]) / "delivered" / "20261023").exists())


class CapabilityTests(unittest.TestCase):
    """capability_brief.md (the file next to scout.py) -> macminim4 (strategy-capability) for Sisyphus co-authoring."""

    def test_brief_is_pushed_once_and_again_only_when_it_changes(self):
        h = Harness(self)
        brief = test_scout.scout.CAPABILITY_FILE.read_text(encoding="utf-8")
        self.assertIn("## A. Intraday", brief)
        self.assertIn("## B. Daily", brief)  # the whole file goes to Sisyphus, not just scout's part A
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.bridge.capability, (hashlib.sha256(brief.encode("utf-8")).hexdigest(), brief))
        for _ in range(2):
            self.assertEqual(h.sync(), 0)
        self.assertEqual(sum(c[0] == "strategy-capability" for c in h.bridge.calls), 1)
        changed = Path(h.tmp.name) / "capability_brief.md"
        changed.write_text("TX only, 5m.\n", encoding="utf-8")
        h.cfg["capability_file"] = str(changed)
        self.assertEqual(h.sync(), 0)
        self.assertEqual(h.bridge.capability[1], "TX only, 5m.\n")
        self.assertEqual(sum(c[0] == "strategy-capability" for c in h.bridge.calls), 2)
        self.assertEqual(h.bridge.sent, [])  # nothing goes to the owner's Telegram

    def test_brief_is_the_file_text_and_blank_is_unreadable(self):
        h = Harness(self)
        path = Path(h.tmp.name) / "brief.md"
        path.write_text("brief text\n", encoding="utf-8")
        self.assertEqual(bs.capability_brief(path), "brief text\n")
        path.write_text(" \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            bs.capability_brief(path)
        h.cfg["capability_file"] = str(path)
        self.assertEqual(h.sync(), 1)
        self.assertIsNone(h.bridge.capability)
        self.assertIn("capability brief unreadable", h.log.lines[-1])

    def test_failures_are_retried_and_dry_run_sends_nothing(self):
        h = Harness(self)
        self.assertEqual(h.sync(dry_run=True), 0)
        self.assertIsNone(h.bridge.capability)
        self.assertTrue(any("dry-run: would push capability brief" in line for line in h.log.lines))
        h.bridge.fail["strategy-capability"] = 1
        self.assertEqual(h.sync(), 1)
        self.assertFalse((Path(h.cfg["state_dir"]) / "capability.sha256").exists())
        self.assertEqual(h.sync(), 0)
        self.assertIsNotNone(h.bridge.capability)
        h.cfg["capability_file"] = str(Path(h.tmp.name) / "missing.md")
        self.assertEqual(h.sync(), 1)
        self.assertIn("capability brief unreadable", h.log.lines[-1])

    def test_ideas_still_flow_when_the_brief_cannot_be_pushed(self):
        h = Harness(self)
        h.cfg["capability_file"] = str(Path(h.tmp.name) / "missing.md")
        h.bridge.add("20261023-s1")
        self.assertEqual(h.sync(), 1)
        self.assertEqual(h.bridge.ideas["20261023-s1"]["status"], "submitted")
        self.assertEqual(h.events(), [("20261023-s1", "start")])


if __name__ == "__main__":
    unittest.main()
