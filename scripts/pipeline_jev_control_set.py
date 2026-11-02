"""Pipeline JEV control set: render statements, evaluate faithfulness (design 2026-10-24 §4, §12 #12).

The control set is a JSONL of clause / spec-variant pairs (``id, clause, spec_id, statement,
proposed_label, label, ...``) plus a JSON of spec v3 variants. ``label`` is the owner's; until the owner
fills it, ``--labels proposed_label`` evaluates against Claude's draft labels, and every output says so.

  render: fill each row's ``statement`` with ``v3.describe.describe_spec`` of its variant (no network).
  eval:   ask JEV the stage's faithfulness question for every pair (the same ``jev.FAITHFULNESS_QUESTION``,
          key and identity check as ``spec_faithfulness``) and run the strict precheck; report agreement,
          per-label precision / recall and the confusion matrix for "stage" (precheck decides a multiple /
          sign mismatch, JEV the rest) and "jev_only". Labels are never sent to JEV.

Key: only ``EVAL_OPENROUTER_KEY`` (poseidon.core.eval_key; the process environment via docker --env-file
~/secrets/eval-openrouter.env, or ``--env-file``), never printed; pin: ``EVAL_OPENROUTER_KEY_LABEL`` from the
environment (docker --env-file ~/secrets/eval-openrouter-label.env). The pipeline's own key and pin are not used here.

Usage (stormtrooper throwaway container; both env files are docker run options, the key file 600, one line each):
  python scripts/pipeline_jev_control_set.py render CS.jsonl SPECS.json
  docker run --rm --env-file ~/secrets/eval-openrouter.env --env-file ~/secrets/eval-openrouter-label.env ... \
      python scripts/pipeline_jev_control_set.py eval CS.jsonl SPECS.json --labels proposed_label --out /out/run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from poseidon.core.eval_key import EVAL_KEY_VAR, EVAL_LABEL_VAR, EvalKeyError, read_eval_key, read_eval_label
from poseidon.pipeline import jev
from poseidon.pipeline.clause_precheck import precheck
from poseidon.pipeline.plan import DEFAULT_SPEC_FAITHFULNESS
from poseidon.pipeline.v3.describe import describe_spec
from poseidon.pipeline.v3.validation import validate_spec_v3

LABELS = list(jev.FAITHFULNESS_LABELS)
MAX_PAIRS = 100


def load(cs_path: Path, specs_path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = [json.loads(line) for line in cs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    variants = json.loads(specs_path.read_text(encoding="utf-8"))["variants"]
    templates = {}
    for name, entry in variants.items():
        result = validate_spec_v3(entry["spec"])
        if not result.ok:
            problems = [issue.text() for issue in (*result.errors, *result.unsupported)]
            raise SystemExit(f"variant {name} does not validate: {problems}")
        templates[name] = {"template": result.template, "statement": describe_spec(result.template, result.params)}
    missing = sorted({row["spec_id"] for row in rows} - set(templates))
    if missing:
        raise SystemExit(f"rows reference unknown variants {missing}")
    return rows, templates


def render(args) -> None:
    rows, templates = load(args.control_set, args.specs)
    for row in rows:
        row["statement"] = templates[row["spec_id"]]["statement"]
    args.control_set.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"rendered {len(rows)} statements")


def metrics(rows: list[dict], field: str) -> dict:
    confusion = {e: dict.fromkeys(LABELS, 0) for e in LABELS}
    for row in rows:
        confusion[row["expected"]][row[field]] += 1
    n = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[e][label] for e in LABELS)
        actual = sum(confusion[label].values())
        per_label[label] = {
            "precision": tp / predicted if predicted else None,
            "recall": tp / actual if actual else None,
            "support": actual,
        }
    return {
        "n": n,
        "correct": correct,
        "agreement": correct / n if n else None,
        "per_label": per_label,
        "confusion_matrix": {"rows": "expected", "cols": "predicted", "labels": LABELS, "matrix": confusion},
        "qualifies": bool(n)
        and correct / n >= DEFAULT_SPEC_FAITHFULNESS["qualification"]["agreement_min"]
        and (per_label["contradicts"]["recall"] or 0.0)
        >= DEFAULT_SPEC_FAITHFULNESS["qualification"]["contradicts_recall_min"],
    }


def evaluate(args) -> None:
    rows, templates = load(args.control_set, args.specs)
    if len(rows) > MAX_PAIRS:
        raise SystemExit(f"more than {MAX_PAIRS} pairs")
    labelled = [row for row in rows if row.get(args.labels) in LABELS]
    if len(labelled) != len(rows):
        raise SystemExit(f"{len(rows) - len(labelled)} rows have no valid {args.labels}")
    try:
        key, pin = read_eval_key(env_file=args.env_file), read_eval_label()
    except EvalKeyError as error:
        raise SystemExit(str(error)) from None
    config = {**DEFAULT_SPEC_FAITHFULNESS, "max_calls_per_attempt": len(rows), "max_cost_usd_per_attempt": args.budget}
    session = jev.JevSession(config, pin=pin, environ={EVAL_KEY_VAR: key}, key_var=EVAL_KEY_VAR, pin_var=EVAL_LABEL_VAR)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with session:
        for row in rows:
            variant = templates[row["spec_id"]]
            checked = precheck(row["clause"], variant["template"], mode="strict")
            # labels never leave this process: the state is the clause and the statement only
            answer = session.ask(
                jev.FAITHFULNESS_QUESTION, {"clause": row["clause"], "spec_statement": variant["statement"]}
            )
            call = session.calls[answer.call]
            stage = checked["decided_label"] if checked["decides"] else answer.choice
            results.append(
                {
                    "id": row["id"],
                    "spec_id": row["spec_id"],
                    "category": row.get("category"),
                    "clause": row["clause"],
                    "expected": row[args.labels],
                    "jev": answer.choice,
                    "stage": stage,
                    "decided_by": "precheck" if checked["decides"] else "jev",
                    "confidence": answer.confidence,
                    "probabilities": answer.probabilities,
                    "precheck": {k: checked[k] for k in ("decides", "deciding_kinds", "unmatched")},
                    "latency_ms": call.get("latency_ms"),
                    "note": row.get("note"),
                }
            )
            print(row["id"], row[args.labels], "->", answer.choice, "| stage", stage, flush=True)
    report = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels_field": args.labels,
        "labels_status": "UNCONFIRMED (Claude's proposed labels; the owner has not labelled yet)"
        if args.labels == "proposed_label"
        else "owner labels",
        "model": config["model"],
        "questions_sha256": jev.questions_sha256(),
        "stage": metrics(results, "stage"),
        "jev_only": metrics(results, "jev"),
        "precheck_decided": [r["id"] for r in results if r["decided_by"] == "precheck"],
        "median_latency_ms": statistics.median(r["latency_ms"] or 0 for r in results),
        "jev": session.usage(),
        "cases": results,
    }
    (out / "report.json").write_text(json.dumps(session.scrub(report), ensure_ascii=False, indent=1), encoding="utf-8")
    for name, data in session.files().items():
        (out / name).write_bytes(data)
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


def markdown(report: dict) -> str:
    def f(x):
        return "-" if x is None else f"{x:.2f}"

    lines = [
        "# Pipeline JEV control set evaluation",
        "",
        f"- labels: `{report['labels_field']}` -- **{report['labels_status']}**",
        f"- model `{report['model']}`, {report['jev']['calls']} calls, ${report['jev']['cost_usd']:.5f}, "
        f"median {report['median_latency_ms']} ms",
        f"- precheck decided: {report['precheck_decided']}",
    ]
    for mode in ("stage", "jev_only"):
        m = report[mode]
        recall = m["per_label"]["contradicts"]["recall"]
        lines += [
            "",
            f"## {mode}: agreement {m['correct']}/{m['n']} = {m['agreement']:.1%}, contradicts recall {f(recall)}"
            f" (qualifies: {m['qualifies']})",
            "",
            "| label | precision | recall | support |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {label} | {f(v['precision'])} | {f(v['recall'])} | {v['support']} |"
            for label, v in m["per_label"].items()
        ]
        matrix = m["confusion_matrix"]["matrix"]
        lines += ["", "| expected \\ predicted | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|"]
        lines += [f"| {e} | " + " | ".join(str(matrix[e][p]) for p in LABELS) + " |" for e in LABELS]
    lines += ["", "## Disagreements (stage mode)", ""]
    wrong = [r for r in report["cases"] if r["stage"] != r["expected"]]
    lines += [
        f"- `{r['id']}` [{r['spec_id']}, {r['decided_by']}] {r['clause']} -- expected {r['expected']}, "
        f"stage {r['stage']} (jev {r['jev']}, conf {f(r['confidence'])})"
        for r in wrong
    ] or ["None."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "eval"):
        cmd = sub.add_parser(name)
        cmd.add_argument("control_set", type=Path)
        cmd.add_argument("specs", type=Path)
        if name == "eval":
            cmd.add_argument(
                "--env-file", type=Path, help="env file with EVAL_OPENROUTER_KEY; default: the environment"
            )
            cmd.add_argument("--labels", default="label", choices=["label", "proposed_label"])
            cmd.add_argument("--out", type=Path, required=True)
            cmd.add_argument("--budget", type=float, default=0.10, help="max USD for the whole run")
    args = parser.parse_args(argv)
    {"render": render, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
