"""Behavioral contract for the default JEV research workflow."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dsh_2330_poc import RevisionLedger
from dsh_decision_poc import JEV_MODEL
from dsh_research import ValidationError, complete, prepare


def bundle() -> dict:
    return {
        "symbol": "2330",
        "snapshots": [
            {
                "id": "2026-q1",
                "as_of": "2026-04-17T00:00:00Z",
                "evidence": [
                    {
                        "id": "q1-fundamental",
                        "published_at": "2026-04-16T08:00:00Z",
                        "source_url": "https://example.test/q1/fundamental",
                        "title": "Q1 fundamental",
                        "locator": "p. 1",
                        "text": "fundamental",
                        "facts": {"revenue_twd_billion": 100},
                    },
                    {
                        "id": "q1-technical",
                        "published_at": "2026-04-16T08:00:00Z",
                        "source_url": "https://example.test/q1/technical",
                        "title": "Q1 technical",
                        "locator": "row 1",
                        "text": "technical",
                        "facts": {"close_twd": 900},
                    },
                    {
                        "id": "q1-future",
                        "published_at": "2026-04-18T08:00:00Z",
                        "source_url": "https://example.test/q1/future",
                        "title": "future",
                        "locator": "p. 1",
                        "text": "future",
                        "facts": {"revenue_twd_billion": 101},
                    },
                ],
            },
            {
                "id": "2026-q2",
                "as_of": "2026-07-17T00:00:00Z",
                "evidence": [
                    {
                        "id": "q2-fundamental",
                        "published_at": "2026-07-16T08:00:00Z",
                        "source_url": "https://example.test/q2/fundamental",
                        "title": "Q2 fundamental",
                        "locator": "p. 1",
                        "text": "fundamental",
                        "facts": {"revenue_twd_billion": 110},
                    },
                    {
                        "id": "q2-technical",
                        "published_at": "2026-07-16T08:00:00Z",
                        "source_url": "https://example.test/q2/technical",
                        "title": "Q2 technical",
                        "locator": "row 1",
                        "text": "technical",
                        "facts": {"close_twd": 950},
                    },
                ],
            },
        ],
    }


def answers(
    packet: dict, *, route: str = "create", confidence: float = 0.9, choices: dict[str, str] | None = None
) -> dict:
    result = {}
    for question_id, question in packet["questions"].items():
        options = question["criteria"]
        choice = (choices or {}).get(
            question_id,
            {
                "revision_route": route,
                "research_sufficiency": "bounded",
                "valuation_sufficiency": "insufficient",
                "evidence_consistency": "consistent",
            }.get(question_id, "relevant"),
        )
        result[question_id] = {
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": {
                option: confidence if option == choice else (1 - confidence) / (len(options) - 1) for option in options
            },
        }
    return result


def research(snapshot: dict) -> dict:
    evidence_ids = [item["id"] for item in snapshot["evidence"]]
    return {
        "symbol": "2330",
        "as_of": snapshot["as_of"],
        "thesis": "中性",
        "stance": "觀察",
        "claims": [
            {"kind": "fact", "text": "基本面", "evidence_ids": [evidence_ids[0]]},
            {"kind": "fact", "text": "技術面", "evidence_ids": [evidence_ids[1]]},
            {"kind": "inference", "text": "推論", "evidence_ids": evidence_ids[:2]},
        ],
        "risks": ["風險"],
        "invalidation_conditions": ["失效條件"],
        "unknowns": ["估值資料不足"],
        "change_summary": "首次研究",
        "next_review_at": "下次財報公布日",
    }


def rebind_request(value: dict) -> dict:
    value.pop("request_digest", None)
    value["request_digest"] = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value


class DshResearchTests(unittest.TestCase):
    def test_prepare_defaults_to_jev_filters_future_evidence_and_reuses_attempt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            request = root / "q1-request.json"
            calls: list[dict] = []

            def call(provider: str, packet: dict) -> dict:
                calls.append({"provider": provider, "packet": packet})
                return {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }

            with patch("dsh_research.call_provider", side_effect=call):
                first = prepare(source, "2026-q1", ledger, request)
                second = prepare(source, "2026-q1", ledger, request)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["provider"], "jev")
            self.assertEqual(first, second)
            self.assertEqual(first["type"], "generation")
            self.assertEqual(first["provider"], "jev")
            self.assertEqual([item["id"] for item in first["snapshot"]["evidence"]], ["q1-fundamental", "q1-technical"])
            self.assertEqual(first["request_digest"], json.loads(request.read_text(encoding="utf-8"))["request_digest"])

    def test_complete_persists_once_and_rejects_mismatched_reply_digest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            request_path = root / "request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                request = prepare(source, "2026-q1", ledger, request_path)
            reply_path = root / "reply.json"
            reply_path.write_text(
                json.dumps(
                    {
                        "request_digest": request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )

            first = complete(source, ledger, request_path, reply_path, root / "reports")
            reply_path.write_text(
                json.dumps(
                    {
                        "request_digest": request["request_digest"],
                        "provider": "terra-second",
                        "model": "terra-second",
                        "research": research(request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            second = complete(source, ledger, request_path, reply_path, root / "reports")
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["revision"]["id"], second["revision"]["id"])

            reply_path.write_text(
                json.dumps({"request_digest": "0" * 64, "research": research(request["snapshot"])}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValidationError, "digest"):
                complete(source, ledger, request_path, reply_path, root / "reports")

    def test_low_probability_and_provider_error_export_blocked_review_without_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            request_path = root / "request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, confidence=0.79),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                request = prepare(source, "2026-q1", ledger, request_path)
            self.assertEqual(request["type"], "review")
            self.assertIn("low_probability", request["reasons"])
            reply = root / "reply.json"
            reply.write_text(
                json.dumps({"request_digest": request["request_digest"], "research": research(request["snapshot"])}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "review"):
                complete(source, ledger, request_path, reply, root / "reports")

            with patch("dsh_research.call_provider", side_effect=RuntimeError("provider unavailable")):
                failed = prepare(source, "2026-q2", ledger, root / "q2-request.json")
            self.assertEqual(failed["type"], "review")
            self.assertEqual(failed["reasons"], ["provider_error"])
            self.assertEqual(failed["provider"], "jev")

    def test_completion_rejects_a_stale_actual_predecessor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger_path = root / "ledger.sqlite"
            q1_request_path = root / "q1-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q1_request = prepare(source, "2026-q1", ledger_path, q1_request_path)
            q1_reply = root / "q1-reply.json"
            q1_reply.write_text(
                json.dumps(
                    {
                        "request_digest": q1_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q1_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            q1 = complete(source, ledger_path, q1_request_path, q1_reply, root / "reports")

            q2_request_path = root / "q2-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, route="update"),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q2_request = prepare(source, "2026-q2", ledger_path, q2_request_path)

            ledger = RevisionLedger(ledger_path)
            changed = research(q1_request["snapshot"])
            changed["change_summary"] = "較晚的 Q1 修訂"
            ledger.persist(changed, q1_request["snapshot"], "manual-revision", "review", "human", q1["revision"]["id"])
            q2_reply = root / "q2-reply.json"
            q2_reply.write_text(
                json.dumps(
                    {
                        "request_digest": q2_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q2_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "previous revision changed"):
                complete(source, ledger_path, q2_request_path, q2_reply, root / "reports")

    def test_resolved_review_can_persist_with_terra_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            request_path = root / "request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, confidence=0.79),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                request = prepare(source, "2026-q1", ledger, request_path)
            reply = root / "reply.json"
            reply.write_text(
                json.dumps(
                    {
                        "request_digest": request["request_digest"],
                        "resolution": "resolved",
                        "provider": "terra",
                        "model": "terra-reviewer",
                        "answers": answers(request["decision_packet"]),
                        "research": research(request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            result = complete(source, ledger, request_path, reply, root / "reports")
            self.assertTrue(result["created"])
            self.assertEqual(result["revision"]["decision_provenance"]["initial"]["provider"], "jev")
            self.assertEqual(result["revision"]["decision_provenance"]["review"]["provider"], "terra")
            self.assertEqual(
                result["revision"]["generation_provenance"], {"provider": "terra", "model": "terra-reviewer"}
            )

    def test_no_change_returns_the_actual_prior_revision_without_a_generation_request(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            q1_request_path = root / "q1-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q1_request = prepare(source, "2026-q1", ledger, q1_request_path)
            q1_reply = root / "q1-reply.json"
            q1_reply.write_text(
                json.dumps(
                    {
                        "request_digest": q1_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q1_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            q1 = complete(source, ledger, q1_request_path, q1_reply, root / "reports")
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, route="no_change"),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                unchanged = prepare(source, "2026-q2", ledger, root / "q2-request.json")
            self.assertEqual(unchanged["type"], "no_change")
            self.assertFalse(unchanged["created"])
            self.assertEqual(unchanged["revision"]["id"], q1["revision"]["id"])

    def test_conflicting_or_missing_required_evidence_cannot_silently_reach_generation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, choices={"evidence_consistency": "conflict"}),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                conflict = prepare(source, "2026-q1", ledger, root / "conflict.json")
            self.assertEqual(conflict["type"], "review")
            self.assertIn("evidence_conflict", conflict["reasons"])

            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, choices={"evidence_1": "irrelevant"}),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                filtered = prepare(source, "2026-q1", root / "other.sqlite", root / "filtered.json")
            self.assertEqual(filtered["type"], "review")
            self.assertIn("required_evidence_missing", filtered["reasons"])

    def test_no_change_rechecks_current_snapshot_and_returns_current_predecessor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            q1_request_path = root / "q1-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q1_request = prepare(source, "2026-q1", ledger, q1_request_path)
            q1_reply = root / "q1-reply.json"
            q1_reply.write_text(
                json.dumps(
                    {
                        "request_digest": q1_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q1_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            complete(source, ledger, q1_request_path, q1_reply, root / "reports")
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, route="no_change"),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                prepare(source, "2026-q2", ledger, root / "q2-request.json")
            changed = bundle()
            changed["snapshots"][1]["evidence"][0]["facts"]["revenue_twd_billion"] = 999
            source.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "snapshot"):
                complete(source, ledger, root / "q2-request.json", root / "unused.json", root / "reports")

    def test_completion_rechecks_predecessor_source_and_full_request_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            q1_request_path = root / "q1-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q1_request = prepare(source, "2026-q1", ledger, q1_request_path)
            q1_reply = root / "q1-reply.json"
            q1_reply.write_text(
                json.dumps(
                    {
                        "request_digest": q1_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q1_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            complete(source, ledger, q1_request_path, q1_reply, root / "reports")
            q2_request_path = root / "q2-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, route="update"),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q2_request = prepare(source, "2026-q2", ledger, q2_request_path)
            reply = root / "q2-reply.json"
            reply.write_text(
                json.dumps(
                    {
                        "request_digest": q2_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q2_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            changed = bundle()
            changed["snapshots"][0]["evidence"][0]["facts"]["revenue_twd_billion"] = 999
            source.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "predecessor"):
                complete(source, ledger, q2_request_path, reply, root / "reports")

            source.write_text(json.dumps(bundle()), encoding="utf-8")
            tampered = json.loads(q2_request_path.read_text(encoding="utf-8"))
            tampered["snapshot"]["evidence"][0]["facts"]["revenue_twd_billion"] = 999
            q2_request_path.write_text(json.dumps(rebind_request(tampered)), encoding="utf-8")
            reply.write_text(
                json.dumps(
                    {
                        "request_digest": tampered["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q2_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "canonical prepared"):
                complete(source, ledger, q2_request_path, reply, root / "reports")

    def test_completion_rejects_a_review_rewritten_as_generation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            request_path = root / "request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, confidence=0.79),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                request = prepare(source, "2026-q1", ledger, request_path)
            self.assertEqual(request["type"], "review")
            tampered = json.loads(request_path.read_text(encoding="utf-8"))
            tampered["type"] = "generation"
            request_path.write_text(json.dumps(rebind_request(tampered)), encoding="utf-8")
            reply = root / "reply.json"
            reply.write_text(
                json.dumps(
                    {
                        "request_digest": tampered["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(tampered["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "prepared"):
                complete(source, ledger, request_path, reply, root / "reports")

    def test_resolved_review_can_return_the_actual_predecessor_as_no_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bundle.json"
            source.write_text(json.dumps(bundle()), encoding="utf-8")
            ledger = root / "ledger.sqlite"
            q1_request_path = root / "q1-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                q1_request = prepare(source, "2026-q1", ledger, q1_request_path)
            q1_reply = root / "q1-reply.json"
            q1_reply.write_text(
                json.dumps(
                    {
                        "request_digest": q1_request["request_digest"],
                        "provider": "terra",
                        "model": "terra",
                        "research": research(q1_request["snapshot"]),
                    }
                ),
                encoding="utf-8",
            )
            q1 = complete(source, ledger, q1_request_path, q1_reply, root / "reports")
            q2_request_path = root / "q2-request.json"
            with patch(
                "dsh_research.call_provider",
                side_effect=lambda provider, packet: {
                    "model": JEV_MODEL,
                    "answers": answers(packet, route="update", confidence=0.79),
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ):
                review = prepare(source, "2026-q2", ledger, q2_request_path)
            self.assertEqual(review["type"], "review")
            reply = root / "q2-reply.json"
            reply.write_text(
                json.dumps(
                    {
                        "request_digest": review["request_digest"],
                        "resolution": "resolved",
                        "provider": "terra",
                        "model": "terra-reviewer",
                        "answers": answers(review["decision_packet"], route="no_change"),
                    }
                ),
                encoding="utf-8",
            )
            result = complete(source, ledger, q2_request_path, reply, root / "reports")
            self.assertEqual(result["type"], "no_change")
            self.assertFalse(result["created"])
            self.assertEqual(result["revision"]["id"], q1["revision"]["id"])
            self.assertEqual(result["decision_provenance"]["review"]["model"], "terra-reviewer")


if __name__ == "__main__":
    unittest.main()
