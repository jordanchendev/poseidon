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
    load_decision_context,
    report,
    select_snapshot,
    snapshot_digest,
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

    def test_decision_artifact_matches_snapshot_and_filters_generator_evidence(self) -> None:
        from tempfile import TemporaryDirectory

        from dsh_decision_poc import build_jev_request, consume_answers, packet_sha256

        snapshot = select_snapshot(bundle(), "2026-q1")
        packet = {
            "id": "q1-decisions",
            "snapshot_id": snapshot["id"],
            "snapshot_digest": snapshot_digest(snapshot),
            "state": {"research_goal": "bounded", "snapshot": snapshot, "previous_snapshot": None},
            "questions": {
                "evidence_0": {
                    "type": "choice",
                    "instructions": "relevance",
                    "criteria": {"relevant": "yes", "irrelevant": "no", "uncertain": "unknown"},
                },
                "revision_route": {
                    "type": "choice",
                    "instructions": "route",
                    "criteria": {"create": "new", "update": "changed", "no_change": "same", "review": "unclear"},
                },
                "research_sufficiency": {
                    "type": "choice",
                    "instructions": "enough",
                    "criteria": {"bounded": "yes", "insufficient": "no"},
                },
                "valuation_sufficiency": {
                    "type": "choice",
                    "instructions": "valuation",
                    "criteria": {"sufficient": "yes", "insufficient": "no"},
                },
            },
            "evidence_question_ids": {"evidence_0": "q1-release"},
        }
        answers = {
            "evidence_0": {
                "type": "choice",
                "choice": "relevant",
                "confidence": 1.0,
                "probabilities": {"relevant": 1.0, "irrelevant": 0.0, "uncertain": 0.0},
            },
            "revision_route": {
                "type": "choice",
                "choice": "create",
                "confidence": 1.0,
                "probabilities": {"create": 1.0, "update": 0.0, "no_change": 0.0, "review": 0.0},
            },
            "research_sufficiency": {
                "type": "choice",
                "choice": "bounded",
                "confidence": 1.0,
                "probabilities": {"bounded": 1.0, "insufficient": 0.0},
            },
            "valuation_sufficiency": {
                "type": "choice",
                "choice": "insufficient",
                "confidence": 1.0,
                "probabilities": {"sufficient": 0.0, "insufficient": 1.0},
            },
        }
        record = {
            "status": "success",
            "packet_id": "q1-decisions",
            "packet_sha256": packet_sha256(packet),
            "provider": "jev",
            "snapshot_id": snapshot["id"],
            "snapshot_digest": snapshot_digest(snapshot),
            "request": build_jev_request(packet),
            "response": {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}},
            "decision": dict(consume_answers(packet, answers), route="review", requires_review=True),
        }
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "records.json"
            artifact.write_text(__import__("json").dumps({"packet": packet, "records": [record]}), encoding="utf-8")
            context, filtered = load_decision_context(artifact, snapshot)
        self.assertEqual(context["route"], "create")
        self.assertEqual([item["id"] for item in filtered["evidence"]], ["q1-release"])

    def test_report_shows_initial_and_resolved_review_decision_provenance(self) -> None:
        snapshot = select_snapshot(bundle(), "2026-q1")
        reviewed = research()
        reviewed["decision_provenance"] = {
            "initial": {"provider": "jev", "model": "jev-1.13.0", "policy": "decision-v1", "context_sha256": "initial"},
            "review": {
                "provider": "terra",
                "model": "terra-reviewer",
                "policy": "decision-v1",
                "context_sha256": "reviewed",
            },
        }
        rendered = report(reviewed, snapshot, "terra", "terra-writer")
        self.assertIn("決策：terra/terra-reviewer", rendered)
        self.assertIn("初始決策：jev/jev-1.13.0", rendered)
