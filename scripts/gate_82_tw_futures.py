#!/usr/bin/env python3
"""Phase 82: Decision gate evaluation for TrendFollowing TX daily.

Reads:
  - config/v82_decision_gate.yaml (frozen gate criteria)
  - scripts/output/wfe_82_results.json (WFE validation results)
Evaluates 4 gate criteria (D-13 through D-16), requires min_pass=3 (D-17).
Produces PASS/FAIL verdict saved to scripts/output/gate_82_results.json.

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/gate_82_tw_futures.py

Expected runtime: < 1 second (pure JSON/YAML read + evaluation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

GATE_YAML = Path("config/v82_decision_gate.yaml")
WFE_JSON = Path("scripts/output/wfe_82_results.json")
OUTPUT_JSON = Path("scripts/output/gate_82_results.json")


def evaluate_gate(gate_criteria: dict, wfe_data: dict) -> dict:
    """Evaluate decision gate criteria against WFE results.

    Args:
        gate_criteria: Parsed YAML gate criteria (top-level dict).
        wfe_data: Parsed JSON WFE results (top-level dict).

    Returns:
        Dict with gates, passed_count, min_pass, overall verdict.
    """
    criteria = gate_criteria["criteria"]
    min_pass = criteria["min_pass"]

    # Extract metrics from WFE JSON
    # Handle both nested (wfe_data["wfe_results"]) and flat structures
    wfe_results = wfe_data.get("wfe_results", wfe_data)
    oos_sharpe = wfe_results.get("oos_aggregate_sharpe", 0.0)
    avg_wfe = wfe_results.get("avg_wfe", 0.0)
    oos_max_dd = wfe_results.get("oos_aggregate_max_drawdown", 1.0)
    bh_oos_sharpe = wfe_data.get("bh_oos_sharpe", 0.0)

    # Evaluate each gate
    gates: dict[str, dict] = {}

    # GATE-01 (D-13): OOS Sharpe > 0
    gate_01 = criteria["gate_01"]
    gate_01_passed = oos_sharpe > gate_01["threshold"]
    gates["gate_01"] = {
        "name": gate_01["name"],
        "value": oos_sharpe,
        "threshold": gate_01["threshold"],
        "operator": gate_01["operator"],
        "passed": gate_01_passed,
    }

    # GATE-02 (D-14): WFE >= 50%
    gate_02 = criteria["gate_02"]
    gate_02_passed = avg_wfe >= gate_02["threshold"]
    gates["gate_02"] = {
        "name": gate_02["name"],
        "value": avg_wfe,
        "threshold": gate_02["threshold"],
        "operator": gate_02["operator"],
        "passed": gate_02_passed,
    }

    # GATE-03 (D-15): OOS Sharpe beats B&H
    gate_03 = criteria["gate_03"]
    gate_03_passed = oos_sharpe > bh_oos_sharpe
    gates["gate_03"] = {
        "name": gate_03["name"],
        "value": oos_sharpe,
        "comparator_value": bh_oos_sharpe,
        "operator": gate_03["operator"],
        "passed": gate_03_passed,
    }

    # GATE-04 (D-16): MaxDD < 50%
    gate_04 = criteria["gate_04"]
    gate_04_passed = oos_max_dd < gate_04["threshold"]
    gates["gate_04"] = {
        "name": gate_04["name"],
        "value": oos_max_dd,
        "threshold": gate_04["threshold"],
        "operator": gate_04["operator"],
        "passed": gate_04_passed,
    }

    passed_count = sum(1 for g in gates.values() if g["passed"])
    overall = "PASS" if passed_count >= min_pass else "FAIL"

    return {
        "gates": gates,
        "passed_count": passed_count,
        "min_pass": min_pass,
        "overall": overall,
    }


def main() -> int:
    """Run Phase 82 decision gate evaluation."""
    print("Phase 82: Decision Gate Evaluation")
    print("=" * 60)

    # Check input files exist
    if not GATE_YAML.exists():
        print(f"ERROR: {GATE_YAML} not found.")
        return 1
    if not WFE_JSON.exists():
        print(f"ERROR: {WFE_JSON} not found. Run wfe_82_tw_futures.py first.")
        return 1

    # Load gate criteria and WFE results
    with open(GATE_YAML) as f:
        gate_criteria = yaml.safe_load(f)

    with open(WFE_JSON) as f:
        wfe_data = json.load(f)

    # Evaluate gate
    verdict = evaluate_gate(gate_criteria, wfe_data)

    # Extract input metrics for display
    wfe_results = wfe_data.get("wfe_results", wfe_data)
    oos_sharpe = wfe_results.get("oos_aggregate_sharpe", 0.0)
    avg_wfe = wfe_results.get("avg_wfe", 0.0)
    bh_oos_sharpe = wfe_data.get("bh_oos_sharpe", 0.0)
    oos_max_dd = wfe_results.get("oos_aggregate_max_drawdown", 1.0)

    # Print verdict table
    print()
    print("=" * 60)
    print("Phase 82 Decision Gate Results")
    print("=" * 60)

    for gate_key in ["gate_01", "gate_02", "gate_03", "gate_04"]:
        g = verdict["gates"][gate_key]
        label = gate_key.upper().replace("_", "-")
        status = "PASS" if g["passed"] else "FAIL"

        if "comparator_value" in g:
            detail = f"value: {g['value']:.4f} vs B&H: {g['comparator_value']:.4f}"
        else:
            detail = f"value: {g['value']:.4f} vs threshold: {g['threshold']}"

        print(f"  {label}: {g['name']} -> {status} ({detail})")

    print()
    overall = verdict["overall"]
    print(f"  Overall: {overall} ({verdict['passed_count']}/{verdict['min_pass']} required)")
    print()

    # Build output JSON
    output = {
        "phase": "82",
        "script": "gate_82_tw_futures.py",
        "gate_yaml_path": str(GATE_YAML),
        "wfe_json_path": str(WFE_JSON),
        "verdict": verdict,
        "input_metrics": {
            "oos_aggregate_sharpe": oos_sharpe,
            "avg_wfe": avg_wfe,
            "bh_oos_sharpe": bh_oos_sharpe,
            "oos_aggregate_max_drawdown": oos_max_dd,
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {OUTPUT_JSON}")

    # Exit code: 0 = PASS, 1 = FAIL
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
