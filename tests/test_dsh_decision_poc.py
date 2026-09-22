"""Behavioral contract for the bounded DSH decision replacement."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dsh_decision_poc import (
    JEV_MODEL,
    OPENAI_MODEL,
    ValidationError,
    build_jev_request,
    build_openai_request,
    consume_answers,
    decision_context_from_records,
    packet_sha256,
    run,
    validate_packet,
    validate_response,
)


def packet() -> dict:
    value = {
        "id": "q1-decisions",
        "snapshot_id": "q1",
        "snapshot_digest": "",
        "state": {"research_goal": "bounded note", "snapshot": {"id": "q1"}, "previous_snapshot": None},
        "questions": {
            "evidence_0": {
                "type": "choice",
                "instructions": "Is evidence relevant?",
                "criteria": {"relevant": "yes", "irrelevant": "no", "uncertain": "unknown"},
            },
            "revision_route": {
                "type": "choice",
                "instructions": "Choose the document action.",
                "criteria": {"create": "new", "update": "changed", "no_change": "same", "review": "unclear"},
            },
            "research_sufficiency": {
                "type": "choice",
                "instructions": "Is this enough?",
                "criteria": {"bounded": "enough", "insufficient": "not enough"},
            },
            "valuation_sufficiency": {
                "type": "choice",
                "instructions": "Is valuation supported?",
                "criteria": {"sufficient": "yes", "insufficient": "no"},
            },
        },
        "evidence_question_ids": {"evidence_0": "evidence-a"},
    }
    value["snapshot_digest"] = hashlib.sha256(
        json.dumps(value["state"]["snapshot"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return value


def answers() -> dict:
    return {
        "evidence_0": {
            "type": "choice",
            "choice": "relevant",
            "confidence": 0.9,
            "probabilities": {"relevant": 0.9, "irrelevant": 0.05, "uncertain": 0.05},
        },
        "revision_route": {
            "type": "choice",
            "choice": "create",
            "confidence": 0.9,
            "probabilities": {"create": 0.9, "update": 0.05, "no_change": 0.03, "review": 0.02},
        },
        "research_sufficiency": {
            "type": "choice",
            "choice": "bounded",
            "confidence": 0.9,
            "probabilities": {"bounded": 0.9, "insufficient": 0.1},
        },
        "valuation_sufficiency": {
            "type": "choice",
            "choice": "insufficient",
            "confidence": 0.9,
            "probabilities": {"sufficient": 0.1, "insufficient": 0.9},
        },
    }


class DshDecisionPocTests(unittest.TestCase):
    def test_both_provider_requests_carry_the_same_packet_state_and_questions(self) -> None:
        source = packet()
        jev = build_jev_request(source)
        openai = build_openai_request(source)
        self.assertEqual(jev["model"], JEV_MODEL)
        self.assertEqual(openai["model"], OPENAI_MODEL)
        self.assertEqual(jev["state"], source["state"])
        self.assertEqual(jev["questions"], source["questions"])
        self.assertEqual(
            json.loads(openai["messages"][1]["content"]),
            {"state": source["state"], "questions": source["questions"]},
        )

    def test_response_requires_normalized_probability_argmax_and_numeric_usage(self) -> None:
        response = {"model": JEV_MODEL, "answers": answers(), "usage": {"input_tokens": 12, "output_tokens": 3}}
        checked = validate_response("jev", response, packet())
        self.assertEqual(checked["usage"], {"input_tokens": 12, "output_tokens": 3, "cached_input_tokens": 0})
        response["answers"]["evidence_0"]["probabilities"]["irrelevant"] = 0.2
        with self.assertRaisesRegex(ValidationError, "probabilities"):
            validate_response("jev", response, packet())

    def test_host_consumes_evidence_and_routes_from_defined_answers(self) -> None:
        decision = consume_answers(packet(), answers())
        self.assertEqual(decision["route"], "create")
        self.assertEqual(decision["keep_evidence_ids"], ["evidence-a"])
        self.assertEqual(decision["review_evidence_ids"], [])
        self.assertEqual(decision["valuation_sufficiency"], "insufficient")
        self.assertEqual(decision["packet_sha256"], packet_sha256(packet()))

    def test_run_records_failed_request_without_zero_cost_success(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packets = root / "packets.json"
            packets.write_text(json.dumps([packet()]), encoding="utf-8")
            args = type(
                "Args", (), {"provider": "jev", "packets": str(packets), "out_dir": str(root / "out"), "repeats": 1}
            )()
            with (
                patch.dict(os.environ, {"JEV_DSH_ANALYSIS_API_KEY": "test"}),
                patch("dsh_decision_poc.post_json", side_effect=RuntimeError("HTTP 429 insufficient_quota")),
            ):
                records = run(args)
            self.assertEqual(records[0]["status"], "error")
            self.assertNotIn("cost_usd", records[0])
            self.assertIn("error", records[0])

    def test_packet_rejects_undeclared_evidence_mapping(self) -> None:
        invalid = packet()
        invalid["evidence_question_ids"] = {"evidence_0": "evidence-a", "wrong": "evidence-b"}
        with self.assertRaisesRegex(ValidationError, "evidence_question_ids"):
            validate_packet(invalid)

    def test_context_export_requires_consistent_successful_non_review_decisions(self) -> None:
        source = packet()
        decision = consume_answers(source, answers())
        response = {"model": JEV_MODEL, "answers": answers(), "usage": {"input_tokens": 12, "output_tokens": 3}}
        records = [
            {
                "status": "success",
                "packet_id": source["id"],
                "snapshot_id": source["snapshot_id"],
                "snapshot_digest": source["snapshot_digest"],
                "packet_sha256": packet_sha256(source),
                "provider": "jev",
                "request": build_jev_request(source),
                "response": response,
                "decision": dict(decision, route="review", requires_review=True),
            }
            for _ in range(2)
        ]
        context = decision_context_from_records(records, "q1", source["snapshot_digest"], source)
        self.assertEqual(context["route"], "create")
        self.assertEqual(context["keep_evidence_ids"], ["evidence-a"])
        records[1]["request"] = {}
        with self.assertRaisesRegex(ValidationError, "request"):
            decision_context_from_records(records, "q1", source["snapshot_digest"], source)


if __name__ == "__main__":
    unittest.main()
