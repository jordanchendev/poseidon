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

import operator as op
from pathlib import Path

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
