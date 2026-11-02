#!/usr/bin/env python3
"""Run the narrative JEV control set through the production judgement (design 2026-10-29 §9 step 7, D10).

Each case goes through narrative_check.judge_record, the function narrative_apply uses for its judgement and sweep:
the period and media prechecks, then JEV_VOTES JEV calls and the majority rule. The key is EVAL_OPENROUTER_KEY
only (poseidon.core.eval_key): from the environment (docker --env-file ~/secrets/eval-openrouter.env) or --env-file.
It is never printed. Labels and notes are never sent.

D10 passes when no case labelled says_nothing ends supports or contradicts.
  python scripts/research/narrative_eval_control_set.py --out <report.json>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jev_client  # noqa: E402
import jev_eval_control_set  # noqa: E402
import narrative_check  # noqa: E402

LABELS = narrative_check.LABELS


class Recording:
    """jev_client.Jev that keeps each raw answer of the current case, for the report."""

    def __init__(self, jev):
        self.jev, self.answers = jev, []

    def judge(self, *args):
        ans = self.jev.judge(*args)
        self.answers.append({k: ans[k] for k in ("label", "probabilities", "latency_ms")})
        return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-set", default=str(HERE / "narrative_control_set.json"))
    ap.add_argument("--env-file", help="env file with EVAL_OPENROUTER_KEY; default: the process environment")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cs = json.loads(Path(a.control_set).read_text(encoding="utf-8"))
    jev = jev_client.Jev(jev_eval_control_set.eval_key_or_exit(a.env_file))
    rec = Recording(jev)
    rows = []
    for c in cs["cases"]:
        rec.answers, cost0, t0 = [], jev.usage_total.get("cost", 0.0), time.monotonic()
        r = narrative_check.judge_record(
            rec, c["claim"], c["source"]["text"], c["source"]["locator"], c["symbol"], cs["meta"]["created"]
        )
        rows.append(
            {
                "id": c["id"],
                "expected": c["expected_label"],
                **r,
                "answers": rec.answers,
                "latency_ms": round((time.monotonic() - t0) * 1000),
                "cost": round(jev.usage_total.get("cost", 0.0) - cost0, 8),
            }
        )
        print(c["id"], c["expected_label"], "->", r["label"], r["precheck"], r["votes"], r["p_label"], flush=True)

    conf = {e: {p: sum(r["expected"] == e and r["label"] == p for r in rows) for p in LABELS} for e in LABELS}
    d10 = [r["id"] for r in rows if r["expected"] == "says_nothing" and r["label"] != "says_nothing"]
    voted = [r for r in rows if r["precheck"] == "none"]
    report = {
        "model": jev.model,
        "endpoint": jev.path,
        "jev_votes": narrative_check.JEV_VOTES,
        "min_win_probability": narrative_check.MIN_WIN_PROBABILITY,
        "n": len(rows),
        "correct": sum(r["expected"] == r["label"] for r in rows),
        "d10": {"pass": not d10, "says_nothing_cases_judged_otherwise": d10},
        "wrong": [r["id"] for r in rows if r["expected"] != r["label"]],
        "decided_by": {g: [r["id"] for r in rows if r["precheck"] == g] for g in ("period", "media", "none")},
        "confusion_matrix": {"rows": "expected", "cols": "final", "matrix": conf},
        "jev_calls": jev.calls,
        "usage_total": jev.usage_total,
        "median_case_latency_ms_voted": statistics.median(r["latency_ms"] for r in voted) if voted else None,
        "median_call_latency_ms": statistics.median(x["latency_ms"] for r in rows for x in r["answers"])
        if jev.calls
        else None,
        "cases": rows,
    }
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("correct", "n", "d10", "wrong", "jev_calls", "usage_total")}))


if __name__ == "__main__":
    main()
