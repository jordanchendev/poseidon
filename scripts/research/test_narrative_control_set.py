"""Design 2026-10-29 §9 step 7: the narrative JEV control set is runnable as frozen evidence.

Every excerpt is a verbatim substring of its committed source file, each file still has the sha256 recorded for it,
and jev_eval_control_set.build_request sends only the claim, excerpt, locator and symbol of each case, never its label
or note. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_control_set.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

from jev_eval_control_set import LABELS, build_request  # noqa: E402

CS = json.loads((HERE / "narrative_control_set.json").read_text(encoding="utf-8"))


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def test_excerpts_are_verbatim_in_sources_with_recorded_sha256():
    docs = CS["meta"]["source_documents"]
    texts = {}
    for did, d in docs.items():
        raw = (HERE / d["path"]).read_bytes()
        assert sha256(raw) == d["file_sha256"], f"{did}: {d['path']} changed since it was frozen"
        texts[did] = raw.decode("utf-8")
    for c in CS["cases"]:
        src = c["source"]
        assert src["text"] in texts[src["document_id"]], f"{c['id']}: excerpt is not verbatim in {src['document_id']}"
        assert sha256(src["text"].encode("utf-8")) == src["text_sha256"], c["id"]
    assert {c["source"]["document_id"] for c in CS["cases"]} == set(docs)


def test_scored_label_is_the_owner_label_and_meta_counts_match():
    cases = CS["cases"]
    assert all(c["expected_label"] == c["owner_label"] and c["label_source"].startswith("owner-") for c in cases)
    assert CS["meta"]["counts"] == {lab: sum(c["expected_label"] == lab for c in cases) for lab in LABELS}


def test_every_case_builds_a_jev_request_without_its_label():
    assert 20 <= len(CS["cases"]) <= 30
    assert len({c["id"] for c in CS["cases"]}) == len(CS["cases"])
    for c in CS["cases"]:
        assert c["expected_label"] in LABELS, c["id"]
        state = build_request(c, "typesafe/jev-1.13", CS["meta"]["created"])["state"]
        assert state == {
            "symbol": c["symbol"],
            "as_of": CS["meta"]["created"],
            "claim": c["claim"],
            "source_context": {"text": c["source"]["text"], "facts": {}, "locator": c["source"]["locator"]},
        }, c["id"]
