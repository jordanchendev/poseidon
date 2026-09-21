from __future__ import annotations

import os
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from jev_2330_poc import Cache, ValidationError, build_request, run, validate_case, validate_response

SNAPSHOT = {
    "id": "q1",
    "as_of": "2026-04-16T23:59:59Z",
    "evidence": [
        {
            "id": "e1",
            "published_at": "2026-04-16",
            "source_url": "https://x",
            "title": "x",
            "locator": "p1",
            "text": "margin was 66.2%",
            "facts": {},
        }
    ],
}
CASE = {
    "id": "c1",
    "claim": "margin was 66.2%",
    "evidence_id": "e1",
    "quote": "margin was 66.2%",
    "expected_label": "supports",
}


class JevPocTests(unittest.TestCase):
    def test_rejects_quote_not_present_in_selected_evidence(self) -> None:
        bad = dict(CASE, quote="invented")
        with self.assertRaisesRegex(ValidationError, "quote"):
            validate_case(bad, SNAPSHOT)

    def test_rejects_invalid_choice_probability_schema(self) -> None:
        response = {
            "model": "jev-1.13.0",
            "answers": {
                "citation_relation": {
                    "type": "choice",
                    "choice": "supports",
                    "confidence": 0.9,
                    "probabilities": {"supports": 0.9, "contradicts": 0.2, "says_nothing": 0.0},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
        with self.assertRaisesRegex(ValidationError, "probabilities"):
            validate_response(response)

    def test_cache_reuses_exact_request_hash(self) -> None:
        with TemporaryDirectory() as directory:
            request = build_request(CASE, SNAPSHOT)
            cache = Cache(Path(directory) / "jev.sqlite")
            first = cache.put(
                request, {"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}
            )
            second = cache.get(request)
            self.assertEqual(first["request_hash"], second["request_hash"])

    def test_run_links_chronological_revisions_and_replays_without_api(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = {
                "symbol": "2330",
                "snapshots": [
                    dict(SNAPSHOT, id="q1"),
                    dict(
                        SNAPSHOT,
                        id="q2",
                        as_of="2026-07-16T23:59:59Z",
                        evidence=[dict(SNAPSHOT["evidence"][0], id="e2", published_at="2026-07-16")],
                    ),
                ],
            }
            cases = {"q1": [CASE], "q2": [dict(CASE, id="c2", evidence_id="e2")]}
            bundle_path, cases_path = root / "bundle.json", root / "cases.json"
            bundle_path.write_text(__import__("json").dumps(bundle), encoding="utf-8")
            cases_path.write_text(__import__("json").dumps(cases), encoding="utf-8")

            def args(snapshot: str) -> Namespace:
                return Namespace(
                    bundle=str(bundle_path), snapshot=snapshot, cases=str(cases_path), out_dir=str(root / "out")
                )

            response = {
                "model": "jev-1.13.0",
                "answers": {
                    "citation_relation": {
                        "type": "choice",
                        "choice": "supports",
                        "confidence": 0.9,
                        "probabilities": {"supports": 0.9, "contradicts": 0.05, "says_nothing": 0.05},
                    }
                },
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
            with (
                patch.dict(os.environ, {"JEV_DSH_ANALYSIS_API_KEY": "test"}),
                patch("jev_2330_poc.post", return_value=response) as post,
            ):
                q1 = run(args("q1"))
                run(args("q2"))
                self.assertEqual(post.call_count, 2)
            import sqlite3

            with sqlite3.connect(root / "out" / "jev.sqlite") as conn:
                rows = conn.execute("SELECT id, previous_id FROM revisions ORDER BY as_of_utc").fetchall()
            self.assertIsNone(rows[0][1])
            self.assertEqual(rows[1][1], rows[0][0])
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("jev_2330_poc.post", side_effect=AssertionError("replay must not call API")),
            ):
                self.assertEqual(run(args("q1")), q1)

    def test_rejects_invalid_expected_label(self) -> None:
        with self.assertRaisesRegex(ValidationError, "expected_label"):
            from jev_2330_poc import validate_case_id

            validate_case_id(dict(CASE, expected_label="buy"))
