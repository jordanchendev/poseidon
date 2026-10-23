#!/usr/bin/env python3
"""scout.py tests (stdlib unittest; run with host python3 on stormtrooper, never on the MacBook, no network).

    python3 scripts/strategy_scout/test_scout.py -v

FakeOpenRouter answers GET /key (label = the pin) and /chat/completions with fixed ideas; FakePipelineApi answers
the run-history GETs. test_bridge_sync.py reuses ``ScoutFixture`` for the scout -> bridge round trip.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scout  # noqa: E402

PIN = "strategy-pipeline-test-label"
IDEAS = [
    {
        "idea_text": "台指期日盤開盤 30 分鐘區間突破做多：收盤站上開盤區間高點進場，停損放在區間低點，收盤前 15 分鐘平倉",
        "summary": "日盤 30 分鐘開盤區間突破",
        "rationale": "開盤區間突破代表當日方向確立",
        "behavior": "早盤突破後趨勢延續",
        "session": "day",
    },
    {
        "idea_text": "台指期夜盤跳空回補：夜盤開盤相對日盤收盤跳空超過 40 點時反向進場，停損 30 點，回補到日盤收盤價出場",
        "summary": "夜盤跳空回補",
        "rationale": "夜盤開盤跳空常被回補",
        "behavior": "跳空後價格回歸前收",
        "session": "both",
    },
    {
        "idea_text": "台指期日盤 ATR(14) 低於其 50 根 SMA 的 0.7 倍時，收盤突破前 20 根最高點做多，2 倍 ATR 停損",
        "summary": "ATR 壓縮後 20 根通道突破",
        "rationale": "波動壓縮後常出現擴張",
        "behavior": "低波動後的突破延續",
        "session": "day",
    },
]


class FakeOpenRouter:
    def __init__(self, ideas=IDEAS, label=PIN):
        self.ideas, self.label, self.calls = ideas, label, []

    def __call__(self, method, url, headers, body=None, timeout=30):
        self.calls.append((method, url))
        assert headers["Authorization"].startswith("Bearer ")
        if url.endswith("/key"):
            return 200, {"data": {"label": self.label, "limit_remaining": 5.0, "usage": 0.1, "limit": 10}}
        assert url.endswith("/chat/completions") and method == "POST"
        content = json.dumps({"ideas": self.ideas}, ensure_ascii=False)
        return 200, {"model": "fake", "choices": [{"message": {"content": content}}], "usage": {"cost": 0.001}}


class FakePipelineApi:
    def __init__(self, items=()):
        self.items, self.methods = list(items), []

    def __call__(self, method, url, headers, body=None, timeout=30):
        self.methods.append(method)
        assert method == "GET" and headers["X-API-Key"] == "pipeline-key"
        return 200, {"items": self.items}


class ScoutFixture:
    """A scout state dir with key files; ``propose()`` runs scout.propose with fakes at ``now``."""

    def __init__(self, root: Path):
        self.root = root
        secrets = root / "secrets"
        secrets.mkdir(parents=True, exist_ok=True)
        (secrets / "or.env").write_text(f"{scout.KEY_ENV}=sk-test-{'x' * 20}\n")
        (secrets / "api.key").write_text("pipeline-key\n")
        (root / "pipeline.env").write_text(f'{scout.PIN_ENV}="{PIN}"\n')
        self.cfg = dict(
            scout.DEFAULTS,
            state_dir=str(root / "scout"),
            openrouter_key_file=str(secrets / "or.env"),
            api_key_file=str(secrets / "api.key"),
            pipeline_env_file=str(root / "pipeline.env"),
        )
        self.state = root / "scout"
        self.lines: list[str] = []

    def propose(self, now, ideas=IDEAS, force=False):
        return scout.propose(
            self.cfg,
            log=self.lines.append,
            api_request=FakePipelineApi(),
            llm_request=FakeOpenRouter(ideas),
            now=now,
            force=force,
        )

    def proposals(self):
        return scout.read_jsonl(self.state / "proposals.jsonl")


NOW = dt.datetime(2026, 10, 26, 7, 30, tzinfo=scout.TZ)


class ProposeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fx = ScoutFixture(Path(tmp.name))

    def test_propose_writes_proposals_and_a_numbered_outbox_batch(self):
        self.assertEqual(self.fx.propose(NOW), 0, self.fx.lines)
        rows = self.fx.proposals()
        self.assertEqual([r["id"] for r in rows], ["20261026-p1", "20261026-p2", "20261026-p3"])
        self.assertEqual({r["status"] for r in rows}, {"proposed"})
        batch = json.loads((self.fx.state / "outbox" / "20261026.json").read_text())
        self.assertEqual(batch["ids"], [r["id"] for r in rows])
        text = batch["text"]
        self.assertEqual((self.fx.state / "outbox" / "20261026.txt").read_text(), text + "\n")
        for n, row in enumerate(rows, 1):
            self.assertRegex(text, rf"(^|\n){n}\. .*（{row['id']}）")
            self.assertIn(row["idea_text"], text)  # the full idea text is what the owner accepts
        self.assertIn("接受策略 N", text)
        self.assertIn("7 天內有效", text)
        self.assertTrue((self.fx.state / ".proposals.lock").exists())
        self.assertEqual(list((self.fx.state / "outbox").glob(".*.tmp")), [])

    def test_second_run_same_day_is_a_no_op_and_force_adds_a_new_batch(self):
        self.fx.propose(NOW)
        self.assertEqual(self.fx.propose(NOW), 0)
        self.assertEqual(len(self.fx.proposals()), 3)
        other = [dict(i, idea_text=i["idea_text"].replace("台指期", "台指期近月"), session="night") for i in IDEAS[:1]]
        other[0]["idea_text"] = "台指期夜盤 RSI(14) 低於 25 後回升穿越 30 做多，停損 1.5 倍 ATR，持有最多 24 根"
        self.assertEqual(self.fx.propose(NOW, ideas=other, force=True), 0, self.fx.lines)
        self.assertEqual(self.fx.proposals()[-1]["id"], "20261026-p4")
        batch = json.loads((self.fx.state / "outbox" / "20261026-from-p4.json").read_text())
        self.assertEqual(batch["ids"], ["20261026-p4"])
        self.assertTrue(batch["text"].splitlines()[2].startswith("1. "))  # numbers restart in every message

    def test_message_fits_the_delivery_cap(self):
        long = [dict(i, idea_text=i["idea_text"] + "。" + "補充說明" * 70) for i in IDEAS]
        rows = [dict(r, id=f"20261026-p{n}") for n, r in enumerate(long, 1)]
        self.assertLess(len(scout.telegram_text("20261026", rows * 2)), 8000)

    def test_scout_never_writes_the_pipeline(self):
        api = FakePipelineApi()
        scout.propose(self.fx.cfg, log=self.fx.lines.append, api_request=api, llm_request=FakeOpenRouter(), now=NOW)
        self.assertEqual(set(api.methods), {"GET"})


class CapabilityBriefTests(unittest.TestCase):
    """The brief must match DSL v2.1 as the pipeline validator accepts it (spec_llm prompt v6)."""

    def test_v21_capabilities_are_offered(self):
        brief = scout.CAPABILITY_BRIEF
        self.assertIn("DSL v2.1", brief)
        self.assertRegex(brief, r"Donchian")
        self.assertRegex(brief, r"SMA or EMA \(2-400 bars\)")
        self.assertIn("0.9-1.1", brief)
        can, cannot = brief.split("It CANNOT express")
        self.assertIn("channel", can)
        self.assertNotRegex(cannot, r"Donchian|rolling N-bar|indicators of indicators")
        for word in ("higher timeframes", "smoothing a smoothed series", "standard deviation", "volume", "options"):
            self.assertIn(word, cannot)

    def test_channel_family_dedupes_channel_ideas(self):
        self.assertEqual(scout.signature("日盤收盤突破前 20 根最高點做多")[0], "channel")
        self.assertEqual(scout.signature("唐奇安通道突破")[0], "channel")


if __name__ == "__main__":
    unittest.main()
