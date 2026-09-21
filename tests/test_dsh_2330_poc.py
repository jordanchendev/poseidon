"""Contract tests for the bounded 2330 dsh research PoC."""

from __future__ import annotations

import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dsh_2330_poc import (
    POLICY_VERSION,
    RevisionLedger,
    ValidationError,
    select_snapshot,
    validate_research,
)
from dsh_poc_tools import previous_for_session, serve


def bundle() -> dict:
    return {
        "symbol": "2330",
        "snapshots": [
            {
                "id": "2026-q1",
                "as_of": "2026-04-17T00:00:00Z",
                "evidence": [
                    {
                        "id": "q1-release",
                        "published_at": "2026-04-16T08:00:00Z",
                        "source_url": "https://example.test/q1",
                        "title": "Q1 release",
                        "locator": "p. 2",
                        "text": "營收與展望",
                        "facts": {"revenue": "1"},
                    },
                    {
                        "id": "future-release",
                        "published_at": "2026-04-18T08:00:00Z",
                        "source_url": "https://example.test/future",
                        "title": "future",
                        "locator": "p. 1",
                        "text": "不應出現",
                        "facts": {},
                    },
                ],
            }
        ],
    }


def research() -> dict:
    return {
        "symbol": "2330",
        "as_of": "2026-04-17T00:00:00Z",
        "thesis": "中性",
        "stance": "觀察",
        "claims": [
            {"kind": "fact", "text": "基本面資料", "evidence_ids": ["q1-release"]},
            {"kind": "fact", "text": "技術面資料", "evidence_ids": ["q1-release"]},
            {"kind": "inference", "text": "推論", "evidence_ids": ["q1-release"]},
        ],
        "risks": ["風險"],
        "invalidation_conditions": ["失效條件"],
        "unknowns": ["未知"],
        "change_summary": "首次研究",
        "next_review_at": "下一次財報公布日或價格跌破 20 日均線時",
    }


class Dsh2330PocTests(unittest.TestCase):
    def test_cutoff_excludes_future_evidence_and_rejects_its_citation(self) -> None:
        snapshot = select_snapshot(bundle(), "2026-q1")
        self.assertEqual([item["id"] for item in snapshot["evidence"]], ["q1-release"])
        invalid = research()
        invalid["claims"][0]["evidence_ids"] = ["future-release"]
        with self.assertRaisesRegex(ValidationError, "citation"):
            validate_research(invalid, snapshot)

    def test_ledger_reuses_duplicate_and_links_new_revision_immutably(self) -> None:
        with self.subTest("database"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                snapshot = select_snapshot(bundle(), "2026-q1")
                ledger = RevisionLedger(Path(directory) / "ledger.sqlite")
                first, created = ledger.persist(research(), snapshot, POLICY_VERSION, "openai", "model-a")
                duplicate, duplicate_created = ledger.persist(research(), snapshot, POLICY_VERSION, "openai", "model-a")
                changed = research()
                changed["change_summary"] = "修訂"
                changed["claims"][2]["text"] = "不同推論"
                changed["as_of"] = "2026-04-18T00:00:00Z"
                later_snapshot = dict(snapshot, id="2026-q2", as_of=changed["as_of"])
                second, second_created = ledger.persist(
                    changed,
                    later_snapshot,
                    POLICY_VERSION,
                    "openai",
                    "model-b",
                    first["id"],
                )

                self.assertTrue(created)
                self.assertFalse(duplicate_created)
                self.assertEqual(first["id"], duplicate["id"])
                self.assertTrue(second_created)
                self.assertEqual(second["previous_revision_id"], first["id"])
                self.assertEqual(ledger.read_previous_revision(second["id"])["id"], first["id"])
                self.assertEqual(ledger.latest_before(later_snapshot["as_of"])["id"], first["id"])
                self.assertIsNone(ledger.latest_before(snapshot["as_of"]))

    def test_research_cites_technical_evidence_when_snapshot_has_it(self) -> None:
        data = bundle()
        data["snapshots"][0]["evidence"].append(
            {
                "id": "price",
                "published_at": "2026-04-16T08:00:00Z",
                "source_url": "https://example.test/price",
                "title": "price",
                "locator": "row 1",
                "text": "price",
                "facts": {"sma20_twd": 1},
            }
        )
        snapshot = select_snapshot(data, "2026-q1")
        with self.assertRaisesRegex(ValidationError, "technical"):
            validate_research(research(), snapshot)

    def test_mcp_rejects_future_previous_revision(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            snapshot = select_snapshot(bundle(), "2026-q1")
            ledger = RevisionLedger(Path(directory) / "ledger.sqlite")
            future = research()
            future["as_of"] = "2026-04-18T00:00:00Z"
            future_snapshot = dict(snapshot, id="future", as_of=future["as_of"])
            future_revision, _ = ledger.persist(future, future_snapshot, POLICY_VERSION, "openai", "model")
            with self.assertRaisesRegex(ValidationError, "earlier"):
                previous_for_session(snapshot, ledger, future_revision["id"])

    def test_mcp_serves_only_declared_tools(self) -> None:
        snapshot = select_snapshot(bundle(), "2026-q1")
        incoming = StringIO(
            '{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
            '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"read_snapshot","arguments":{}}}\n'
        )
        outgoing = StringIO()
        old_stdin, old_stdout = sys.stdin, sys.stdout
        try:
            sys.stdin, sys.stdout = incoming, outgoing
            serve(snapshot, None)
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
        replies = [__import__("json").loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3])
        self.assertEqual(
            [tool["name"] for tool in replies[1]["result"]["tools"]],
            ["read_snapshot", "read_previous_revision"],
        )
        self.assertIn("q1-release", replies[2]["result"]["content"][0]["text"])
