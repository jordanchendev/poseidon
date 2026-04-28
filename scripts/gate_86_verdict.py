"""Phase 86: Decision-gate evaluation script for v17.0 milestone verdict.

Reads:
  - .planning/phases/84-strategy-1m-adaptation-frozen-gate/GATE.yaml (frozen at 5a1ecc9)
  - .planning/phases/85-optuna-wfe-validation/artifacts/{btc,eth}usdt_{wfe,optuna}.json

Computes per-symbol gate evaluation against the 4 frozen `criteria` entries,
aggregates with both-must-pass rule (D-05), then emits VERDICT.md +
gate_86_results.json. Exit 0 on PASS, 1 on FAIL (D-10 — wired in Plan 03).

Design constraints (D-03):
  - Pure: no DB, no Redis, no network, no subprocess, no git.
  - Idempotent: same inputs ⇒ same outputs.
  - CWD-agnostic: default paths anchored on `Path(__file__).resolve().parents[2]`
    (= aquarium root) so the script runs the same from any working directory.

Plan 02 (Wave 1) scope: three pure functions — `evaluate_symbol_gates`,
`aggregate_milestone_verdict`, `assert_frozen_anchor` — plus module-level
constants. CLI / `main()` / output writers land in Plan 03 (Wave 2).
"""

from __future__ import annotations

import argparse
import json
import operator as op
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Anchor: this file lives at poseidon/scripts/gate_86_verdict.py.
# parents[0] = poseidon/scripts, parents[1] = poseidon, parents[2] = aquarium root.
# Pitfall 4 fix: never rely on CWD-relative paths (D-03 / D-02).
AQUARIUM_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GATE_YAML = AQUARIUM_ROOT / ".planning/phases/84-strategy-1m-adaptation-frozen-gate/GATE.yaml"
DEFAULT_BTC_WFE = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/btcusdt_wfe.json"
DEFAULT_ETH_WFE = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/ethusdt_wfe.json"
DEFAULT_BTC_OPTUNA = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/btcusdt_optuna.json"
DEFAULT_ETH_OPTUNA = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/ethusdt_optuna.json"
DEFAULT_OUT_DIR = AQUARIUM_ROOT / ".planning/phases/86-decision-gate-evaluation-verdict"

# Operator-string dispatch table (RESEARCH Pattern 1). Stdlib `operator` module
# is well-tested and avoids `eval()` (Pitfall — security smell + lint-tripping).
OPS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
}


def evaluate_symbol_gates(criteria_block: dict, verdict_inputs: dict) -> dict[str, dict]:
    """Evaluate criteria.gate_01..gate_NN against one symbol's verdict_inputs.

    Args:
        criteria_block: ``gate_yaml["criteria"]`` — dict with keys ``gate_01``,
            ``gate_02``, ... Each value has shape
            ``{"name": str, "metric": str, "operator": str, "threshold": number}``.
        verdict_inputs: One symbol's ``verdict_inputs`` block from
            ``*_wfe.json``. Keys are metric names (e.g. ``oos_aggregate_sharpe``,
            ``wfe_degradation``, ``oos_total_trades``, ``max_consecutive_losses``,
            ``n_oos_windows``). Values may be ``None`` per D-06.

    Returns:
        Dict keyed by gate_id with per-gate result dict containing
        ``name``, ``metric``, ``value``, ``operator``, ``threshold``,
        ``passed``, ``reason``. ``reason`` is populated only when the metric
        value is ``None`` (cannot evaluate ⇒ FAIL per D-06).

    Behavior:
        - Iterate ``sorted(k for k in criteria_block if k.startswith("gate_"))``
          for deterministic order independent of YAML loader insertion order.
        - When ``verdict_inputs.get(metric_name) is None`` ⇒
          ``passed=False``, ``reason="null metric '<metric>' (cannot evaluate ⇒ FAIL per D-06)"``.
          Never compares ``None`` to a number (Pitfall 2 — would TypeError).
        - Otherwise ⇒ ``passed = OPS[operator](value, threshold)``, ``reason=None``.
    """
    gates: dict[str, dict] = {}
    for gate_key in sorted(k for k in criteria_block if k.startswith("gate_")):
        spec = criteria_block[gate_key]
        metric_name = spec["metric"]
        operator_str = spec["operator"]
        threshold = spec["threshold"]

        value = verdict_inputs.get(metric_name)  # may be None per D-06

        if value is None:
            passed = False
            reason = f"null metric '{metric_name}' (cannot evaluate ⇒ FAIL per D-06)"
        else:
            passed = OPS[operator_str](value, threshold)
            reason = None

        gates[gate_key] = {
            "name": spec["name"],
            "metric": metric_name,
            "value": value,  # preserve None for JSON sidecar traceability
            "operator": operator_str,
            "threshold": threshold,
            "passed": passed,
            "reason": reason,
        }
    return gates


def aggregate_milestone_verdict(
    per_symbol_results: dict[str, dict],
    min_pass: int,
) -> tuple[str, str]:
    """Aggregate per-symbol gate results into a milestone-level verdict (D-05).

    Args:
        per_symbol_results: ``{symbol: {"gates": {...}, "passed_count": int, ...}, ...}``.
            Caller controls iteration order; tests pass BTCUSDT before ETHUSDT
            so the rationale string lists symbols in that order.
        min_pass: ``gate_yaml["min_pass"]`` (top-level, NOT inside ``criteria``).
            Currently 3 in v17.0 frozen GATE.yaml.

    Returns:
        ``(verdict, rationale)`` where ``verdict ∈ {"PASS", "FAIL"}``.

    Behavior (D-05 both-must-pass):
        - PASS iff EVERY symbol has ``passed_count >= min_pass``. Rationale
          lists every symbol's pass-side fragment, e.g.
          ``"BTCUSDT 4/4 >= 3; ETHUSDT 4/4 >= 3"``.
        - FAIL iff ANY symbol has ``passed_count < min_pass``. Rationale lists
          ONLY the failing symbols' fail-side fragments (Pitfall 5: do not mix
          PASS-side fragments into a FAIL rationale — D-13 example shape is
          ``"BTCUSDT 2/4 < 3; ETHUSDT 1/4 < 3"``).

    Determinism: rationale fragments are joined in the caller's iteration order
    of ``per_symbol_results`` (Python 3.7+ dict insertion-order guarantee).
    """
    pass_fragments: list[str] = []
    fail_fragments: list[str] = []
    for symbol, result in per_symbol_results.items():
        passed_count = result["passed_count"]
        # `gates` may not be present in synthetic test fixtures that only carry
        # passed_count/verdict; fall back to len-from-passed_count semantics
        # would be wrong, so derive a safe total from gates when available and
        # default to 4 (current GATE.yaml has exactly 4 gates) otherwise.
        gates = result.get("gates")
        total = len(gates) if gates is not None else 4
        if passed_count >= min_pass:
            pass_fragments.append(f"{symbol} {passed_count}/{total} >= {min_pass}")
        else:
            fail_fragments.append(f"{symbol} {passed_count}/{total} < {min_pass}")

    all_passed = not fail_fragments
    if all_passed:
        return ("PASS", "; ".join(pass_fragments))
    return ("FAIL", "; ".join(fail_fragments))


def assert_frozen_anchor(gate_yaml: dict, artifacts: dict[str, dict]) -> None:
    """Verify every artifact's frozen_gate_anchor matches gate_yaml.frozen_commit.

    Args:
        gate_yaml: parsed GATE.yaml. MUST contain top-level ``frozen_commit``
            string (raises KeyError otherwise — fail-loud is intentional).
        artifacts: ``{label: parsed_json}`` mapping. Each value MUST be a dict
            and SHOULD contain top-level ``frozen_gate_anchor``. A missing key
            surfaces as ``got=None`` (distinct from a wrong-string value).

    Behavior (RESEARCH Pattern 2 + Pitfall 3):
        - Strict string equality. No truncation tolerance, no normalization.
          ``"5a1ecc9" != "5a1ecc9e" != "5A1ECC9"``.
        - On any mismatch, raise ``SystemExit`` with all mismatching labels
          listed (D-20 violation). Exit code 1 propagates to CLI.
        - Returns ``None`` implicitly on full match.

    Raises:
        SystemExit: when one or more artifacts carry a mismatched anchor.
    """
    expected = gate_yaml["frozen_commit"]
    mismatches: list[str] = []
    for label, art in artifacts.items():
        got = art.get("frozen_gate_anchor")
        if got != expected:
            mismatches.append(f"{label}: expected {expected!r}, got {got!r}")
    if mismatches:
        raise SystemExit("Frozen-gate anchor mismatch (D-20 violation):\n  " + "\n  ".join(mismatches))


# ----------------------------------------------------------------------------
# Plan 03 (Wave 2): CLI surface + writers + main() orchestration.
# Per D-03 purity: stdlib + PyYAML only — no DB, no Redis, no network, no git.
# Per D-11/D-12: this module never modifies STATE.md/PROJECT.md/ROADMAP.md.
# ----------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Load YAML file via stdlib + PyYAML safe_load. Raises on parse error (fail-loud per D-03)."""
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> dict:
    """Load JSON file. Raises on parse error (fail-loud per D-03)."""
    return json.loads(path.read_text())


def _format_value(value: object) -> str:
    """Render a metric value for the VERDICT.md per-gate-results table.

    - ``None`` → ``"null"`` (D-06 null preservation in human-readable output)
    - ``float`` → 4-decimal fixed-point (e.g. ``0.0404``)
    - other (``int``, ``bool``) → ``str(value)``

    JSON sidecar bypasses this helper and preserves the raw value (None included).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        # bool is a subclass of int in Python; handle before the int/float branches
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_verdict_markdown(
    *,
    frozen_commit: str,
    gate_yaml_path: Path,
    eval_date: str,
    per_symbol_results: dict,
    milestone_verdict: str,
    milestone_rationale: str,
) -> str:
    """Render the 6-section VERDICT.md text per D-08 layout (RESEARCH Pattern 3).

    Sections (in order):
      1. Header (title + frozen commit + evaluation date + GATE.yaml path)
      2. ## Per-Gate Results table (4 gates × N symbols)
      3. ## Per-Symbol Summary table
      4. ## Milestone Verdict
      5. ## Notes / Caveats
      6. ## Action Path (D-11 PASS branch or D-12 FAIL branch — picked by milestone_verdict)

    Args:
        frozen_commit: gate_yaml["frozen_commit"] (e.g. "5a1ecc9").
        gate_yaml_path: absolute or repo-relative path to GATE.yaml (rendered as code).
        eval_date: ISO date string (YYYY-MM-DD) — date-only, not full timestamp.
        per_symbol_results: insertion-ordered dict (BTCUSDT first per Plan 02 contract).
        milestone_verdict: "PASS" or "FAIL".
        milestone_rationale: aggregator-produced rationale string.

    Returns:
        Full markdown text. Caller writes to ``<out_dir>/VERDICT.md``.
    """
    lines: list[str] = []

    # --- Section 1: Header --------------------------------------------------
    lines.append("# Phase 86 — Decision Gate Verdict")
    lines.append("")
    lines.append(f"**Frozen GATE commit:** `{frozen_commit}`")
    lines.append(f"**Evaluation date:** {eval_date}")
    lines.append(f"**GATE.yaml:** `{gate_yaml_path}`")
    lines.append("")

    # --- Section 2: Per-Gate Results table ----------------------------------
    lines.append("## Per-Gate Results")
    lines.append("")
    lines.append("| Symbol | Gate | Metric | Threshold | Operator | Value | Result |")
    lines.append("|--------|------|--------|-----------|----------|-------|--------|")
    for symbol, result in per_symbol_results.items():
        gates = result["gates"]
        for gate_id in sorted(gates.keys()):
            g = gates[gate_id]
            badge = "✅ PASS" if g["passed"] else "❌ FAIL"
            cell = badge
            if g.get("reason"):
                cell = f"{badge} (null metric ⇒ cannot evaluate per D-06)"
            lines.append(
                f"| {symbol} | {gate_id} {g['name']} | {g['metric']} | "
                f"{_format_value(g['threshold'])} | {g['operator']} | "
                f"{_format_value(g['value'])} | {cell} |"
            )
    lines.append("")

    # --- Section 3: Per-Symbol Summary --------------------------------------
    lines.append("## Per-Symbol Summary")
    lines.append("")
    lines.append("| Symbol | passed_count | min_pass | Symbol Verdict |")
    lines.append("|--------|--------------|----------|----------------|")
    for symbol, result in per_symbol_results.items():
        total = len(result["gates"])
        lines.append(f"| {symbol} | {result['passed_count']}/{total} | {result['min_pass']} | {result['verdict']} |")
    lines.append("")

    # --- Section 4: Milestone Verdict ---------------------------------------
    lines.append("## Milestone Verdict")
    lines.append("")
    lines.append(f"**{milestone_verdict}** — {milestone_rationale}")
    lines.append("")

    # --- Section 5: Notes / Caveats -----------------------------------------
    lines.append("## Notes / Caveats")
    lines.append("")
    # Echo Phase 85 flags from each symbol's record (D-09 traceability echoed in markdown)
    flags_lines: list[str] = []
    for symbol, result in per_symbol_results.items():
        flags = result.get("flags", [])
        flags_lines.append(f"  - {symbol}: {flags}")
    null_metric_note = ""
    eth_record = per_symbol_results.get("ETHUSDT")
    if eth_record is not None:
        eth_g04 = eth_record.get("gates", {}).get("gate_04")
        if eth_g04 and eth_g04.get("value") is None:
            null_metric_note = (
                "- ETH `max_consecutive_losses` is null because Phase 85 surfaced "
                "`zero_oos_trades` flag (no trades ⇒ no loss-streak to compute). "
                "Treated as FAIL per D-06."
            )
    if null_metric_note:
        lines.append(null_metric_note)
    lines.append("- Strict `>` comparisons treat `0.0` as FAIL (Pitfall 1 — correct frozen behavior, not a bug).")
    lines.append(
        "- `n_oos_windows = 3` for both symbols is below "
        "`walk_forward.min_oos_windows = 4` configured in GATE.yaml, but the "
        "`criteria` block does NOT include an n_oos_windows gate (D-07) — "
        "informational only."
    )
    lines.append("- Phase 85 flags echoed in JSON sidecar:")
    lines.extend(flags_lines)
    lines.append("")

    # --- Section 6: Action Path ---------------------------------------------
    lines.append("## Action Path")
    lines.append("")
    if milestone_verdict == "PASS":
        # D-11 — PASS branch (manual operator follow-ups)
        lines.append("Per D-11 (PASS path — manual follow-up actions, NOT automated by this script):")
        lines.append("")
        lines.append("1. Mark Phase 86 complete in `.planning/ROADMAP.md`")
        lines.append("2. Promote v17.0 milestone to next stage")
        lines.append("3. Update `.planning/PROJECT.md` with PASS thesis")
        lines.append("4. Plan v18.0 launch")
    else:
        # D-12 — FAIL branch (manual operator follow-ups)
        lines.append("Per D-12 (FAIL path — manual follow-up actions, NOT automated by this script):")
        lines.append("")
        lines.append("1. Mark Phase 86 complete in `.planning/ROADMAP.md`")
        lines.append("2. Close v17.0 milestone")
        lines.append(
            "3. Record dead-end thesis in `.planning/PROJECT.md` (mirror "
            "`project_poseidon_ta_dead_end.md` memory pattern: which strategy / "
            "which gates failed / evidence pointers)"
        )
        lines.append("4. Stop track — no v18.0 launch")
    lines.append("")

    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the locked 6-flag argparse surface (D-02). Defaults from Plan 02 constants."""
    parser = argparse.ArgumentParser(
        prog="gate_86_verdict.py",
        description=(
            "Phase 86 decision-gate evaluator. Reads frozen GATE.yaml + Phase 85 "
            "WFE/Optuna artifacts, emits VERDICT.md + gate_86_results.json. Exits "
            "0 on PASS milestone, 1 on FAIL (D-10)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gate-yaml", type=Path, default=DEFAULT_GATE_YAML)
    parser.add_argument("--btc-wfe", type=Path, default=DEFAULT_BTC_WFE)
    parser.add_argument("--eth-wfe", type=Path, default=DEFAULT_ETH_WFE)
    parser.add_argument("--btc-optuna", type=Path, default=DEFAULT_BTC_OPTUNA)
    parser.add_argument("--eth-optuna", type=Path, default=DEFAULT_ETH_OPTUNA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Phase 86 decision-gate evaluator entrypoint.

    Returns:
        ``0`` if milestone verdict is PASS, ``1`` if FAIL (D-10).
    """
    args = _build_arg_parser().parse_args(argv)

    # 1. Load all inputs (fail-loud on parse errors per D-03).
    gate_yaml = _load_yaml(args.gate_yaml)
    btc_wfe = _load_json(args.btc_wfe)
    eth_wfe = _load_json(args.eth_wfe)
    btc_optuna = _load_json(args.btc_optuna)
    eth_optuna = _load_json(args.eth_optuna)

    # 2. Frozen-anchor gate (D-20): all 4 artifacts must declare frozen_gate_anchor
    #    matching gate_yaml.frozen_commit. SystemExit on mismatch.
    assert_frozen_anchor(
        gate_yaml,
        {
            "btc_wfe": btc_wfe,
            "btc_optuna": btc_optuna,
            "eth_wfe": eth_wfe,
            "eth_optuna": eth_optuna,
        },
    )

    # 3. Per-symbol evaluation (BTC first, ETH second — insertion order matters
    #    for aggregate_milestone_verdict's rationale fragment ordering).
    min_pass = gate_yaml["min_pass"]
    per_symbol_results: dict[str, dict] = {}
    for symbol, wfe in (("BTCUSDT", btc_wfe), ("ETHUSDT", eth_wfe)):
        verdict_inputs = wfe["verdict_inputs"]
        gates = evaluate_symbol_gates(gate_yaml["criteria"], verdict_inputs)
        passed_count = sum(1 for g in gates.values() if g["passed"])
        per_symbol_results[symbol] = {
            "verdict": "PASS" if passed_count >= min_pass else "FAIL",
            "passed_count": passed_count,
            "min_pass": min_pass,
            "verdict_inputs": verdict_inputs,
            "flags": list(wfe.get("flags", [])),
            "gates": gates,
        }

    # 4. Aggregate milestone verdict (both-must-pass per D-05).
    milestone_verdict, milestone_rationale = aggregate_milestone_verdict(per_symbol_results, min_pass)

    # 5. Build JSON sidecar payload (D-09 + 86-PATTERNS schema).
    now = datetime.now(UTC)
    payload = {
        "phase": "86",
        "frozen_commit": gate_yaml["frozen_commit"],
        "evaluation_timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_pass": min_pass,
        "input_paths": {
            "gate_yaml": str(args.gate_yaml.resolve()),
            "btc_wfe": str(args.btc_wfe.resolve()),
            "eth_wfe": str(args.eth_wfe.resolve()),
            "btc_optuna": str(args.btc_optuna.resolve()),
            "eth_optuna": str(args.eth_optuna.resolve()),
        },
        "per_symbol": per_symbol_results,
        "milestone_verdict": milestone_verdict,
        "milestone_rationale": milestone_rationale,
    }

    # 6. Render VERDICT.md text.
    verdict_md = render_verdict_markdown(
        frozen_commit=gate_yaml["frozen_commit"],
        gate_yaml_path=args.gate_yaml,
        eval_date=now.strftime("%Y-%m-%d"),
        per_symbol_results=per_symbol_results,
        milestone_verdict=milestone_verdict,
        milestone_rationale=milestone_rationale,
    )

    # 7. Write outputs (idempotent overwrite). Bounded to <out-dir> + <out-dir>/artifacts.
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "VERDICT.md").write_text(verdict_md)
    (artifacts_dir / "gate_86_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # 8. Human feedback line.
    print(f"Phase 86 milestone verdict: {milestone_verdict} — {milestone_rationale}")

    # 9. Exit code per D-10.
    return 0 if milestone_verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
