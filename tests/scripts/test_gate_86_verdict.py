"""Tests for Phase 86 decision-gate verdict script.

Covers (RESEARCH Validation Architecture → Test Map):
- evaluate_symbol_gates returns correct PASS/FAIL per criterion (BTC actual values, D-04, D-13)
- evaluate_symbol_gates returns correct PASS/FAIL per criterion (ETH actual values incl. null, D-04, D-06, D-13)
- null metric → gate FAIL with reason (D-06)
- Both-must-pass milestone aggregation truth table (D-05)
- frozen_gate_anchor mismatch raises SystemExit (D-20 guard / RESEARCH Pattern 2)
- CLI smoke against real Phase 85 artifacts → exit code 1 (D-13 expected FAIL)
- Output files written with required sections/keys (D-08, D-09)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Locate gate_86_verdict.py — lives at poseidon/scripts/gate_86_verdict.py.
# parents[0]=tests/scripts, parents[1]=tests, parents[2]=poseidon.
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gate_86_verdict.py"

# Aquarium root for locating real Phase 85 artifacts (parents[3] = aquarium/).
AQUARIUM_ROOT = Path(__file__).resolve().parents[3]
REAL_BTC_WFE = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/btcusdt_wfe.json"
REAL_ETH_WFE = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/ethusdt_wfe.json"
REAL_BTC_OPTUNA = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/btcusdt_optuna.json"
REAL_ETH_OPTUNA = AQUARIUM_ROOT / ".planning/phases/85-optuna-wfe-validation/artifacts/ethusdt_optuna.json"
REAL_GATE_YAML = AQUARIUM_ROOT / ".planning/phases/84-strategy-1m-adaptation-frozen-gate/GATE.yaml"


def _import_script():
    """Load gate_86_verdict.py as a module (script lives outside PYTHONPATH)."""
    spec = importlib.util.spec_from_file_location("gate_86_verdict", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot locate {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEvaluateSymbolGates:
    def test_btc_actual_inputs(self, fixture_gate_yaml, fixture_btc_wfe_actual):
        """D-04 + D-13: BTC actual verdict_inputs → gate_01 PASS, gate_02 FAIL, gate_03 FAIL, gate_04 PASS (passed_count=2, symbol FAIL)."""
        mod = _import_script()
        gates = mod.evaluate_symbol_gates(
            fixture_gate_yaml["criteria"],
            fixture_btc_wfe_actual["verdict_inputs"],
        )
        assert gates["gate_01"]["passed"] is True  # 0.0404 > 0.0
        assert gates["gate_02"]["passed"] is False  # 10.26 not < 0.40
        assert gates["gate_03"]["passed"] is False  # 6 not >= 100
        assert gates["gate_04"]["passed"] is True  # 0 <= 8
        assert sum(1 for g in gates.values() if g["passed"]) == 2

    def test_eth_actual_inputs_with_null(self, fixture_gate_yaml, fixture_eth_wfe_actual):
        """D-04 + D-06 + D-13: ETH actual verdict_inputs → gate_01 FAIL (0.0 not > 0), gate_02 PASS (0.0 < 0.40), gate_03 FAIL (0 < 100), gate_04 FAIL (null per D-06). passed_count=1."""
        mod = _import_script()
        gates = mod.evaluate_symbol_gates(
            fixture_gate_yaml["criteria"],
            fixture_eth_wfe_actual["verdict_inputs"],
        )
        assert gates["gate_01"]["passed"] is False  # 0.0 > 0.0 is False (Pitfall 1)
        assert gates["gate_02"]["passed"] is True  # 0.0 < 0.40
        assert gates["gate_03"]["passed"] is False  # 0 < 100
        assert gates["gate_04"]["passed"] is False  # null → FAIL per D-06
        assert gates["gate_04"]["reason"] is not None  # null reason populated
        assert "null" in gates["gate_04"]["reason"].lower()
        assert sum(1 for g in gates.values() if g["passed"]) == 1

    def test_null_metric_fails_with_reason(self, fixture_gate_yaml, fixture_eth_wfe_null):
        """D-06: synthetic isolation — null metric on gate_04 alone FAILs while gates 01-03 PASS."""
        mod = _import_script()
        gates = mod.evaluate_symbol_gates(
            fixture_gate_yaml["criteria"],
            fixture_eth_wfe_null["verdict_inputs"],
        )
        assert gates["gate_01"]["passed"] is True
        assert gates["gate_02"]["passed"] is True
        assert gates["gate_03"]["passed"] is True
        assert gates["gate_04"]["passed"] is False
        assert gates["gate_04"]["value"] is None  # value preserved as None
        assert gates["gate_04"]["reason"] is not None
        assert "max_consecutive_losses" in gates["gate_04"]["reason"]


class TestMilestoneAggregation:
    def test_milestone_aggregation(self):
        """D-05: both-must-pass — milestone PASS only when BOTH symbols passed_count >= min_pass.
        Truth table: PP=PASS, PF=FAIL, FP=FAIL, FF=FAIL.
        """
        mod = _import_script()
        min_pass = 3
        # Simulated per-symbol results (only verdict + passed_count matter for aggregation)
        pp = {"BTCUSDT": {"verdict": "PASS", "passed_count": 4}, "ETHUSDT": {"verdict": "PASS", "passed_count": 3}}
        pf = {"BTCUSDT": {"verdict": "PASS", "passed_count": 3}, "ETHUSDT": {"verdict": "FAIL", "passed_count": 2}}
        fp = {"BTCUSDT": {"verdict": "FAIL", "passed_count": 1}, "ETHUSDT": {"verdict": "PASS", "passed_count": 4}}
        ff = {"BTCUSDT": {"verdict": "FAIL", "passed_count": 2}, "ETHUSDT": {"verdict": "FAIL", "passed_count": 1}}
        v_pp, _ = mod.aggregate_milestone_verdict(pp, min_pass)
        v_pf, _ = mod.aggregate_milestone_verdict(pf, min_pass)
        v_fp, _ = mod.aggregate_milestone_verdict(fp, min_pass)
        v_ff, _ = mod.aggregate_milestone_verdict(ff, min_pass)
        assert v_pp == "PASS"
        assert v_pf == "FAIL"
        assert v_fp == "FAIL"
        assert v_ff == "FAIL"


class TestFrozenAnchor:
    def test_anchor_mismatch_fails_loud(self, fixture_gate_yaml, fixture_anchor_mismatch):
        """RESEARCH Pattern 2 + Pitfall 3: artifact carrying anchor != gate_yaml['frozen_commit'] must SystemExit."""
        mod = _import_script()
        with pytest.raises(SystemExit) as excinfo:
            mod.assert_frozen_anchor(fixture_gate_yaml, fixture_anchor_mismatch)
        # Mismatch label "eth_wfe" and the bad value "DEADBEEF" should be mentioned
        err = str(excinfo.value)
        assert "eth_wfe" in err
        assert "DEADBEEF" in err


class TestCLI:
    def test_cli_actual_artifacts(self, tmp_path):
        """D-13: invoking the script with real Phase 85 artifacts must exit 1 (FAIL milestone)."""
        if not SCRIPT_PATH.exists():
            pytest.skip(f"{SCRIPT_PATH} not yet implemented")
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--gate-yaml",
                str(REAL_GATE_YAML),
                "--btc-wfe",
                str(REAL_BTC_WFE),
                "--eth-wfe",
                str(REAL_ETH_WFE),
                "--btc-optuna",
                str(REAL_BTC_OPTUNA),
                "--eth-optuna",
                str(REAL_ETH_OPTUNA),
                "--out-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1, (
            f"Expected exit 1 (FAIL), got {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    def test_outputs_written(self, tmp_path):
        """D-08 + D-09: VERDICT.md and artifacts/gate_86_results.json must be created with required sections/keys."""
        if not SCRIPT_PATH.exists():
            pytest.skip(f"{SCRIPT_PATH} not yet implemented")
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--gate-yaml",
                str(REAL_GATE_YAML),
                "--btc-wfe",
                str(REAL_BTC_WFE),
                "--eth-wfe",
                str(REAL_ETH_WFE),
                "--btc-optuna",
                str(REAL_BTC_OPTUNA),
                "--eth-optuna",
                str(REAL_ETH_OPTUNA),
                "--out-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        verdict_md = tmp_path / "VERDICT.md"
        results_json = tmp_path / "artifacts" / "gate_86_results.json"
        assert verdict_md.exists()
        assert results_json.exists()
        md_text = verdict_md.read_text()
        # All 6 D-08 sections must be present (header line + 5 ## sections)
        assert "Frozen GATE commit" in md_text or "frozen GATE commit" in md_text.lower()
        assert "## Per-Gate Results" in md_text or "## Per-Symbol" in md_text
        assert "## Milestone Verdict" in md_text
        assert "## Notes" in md_text or "## Caveats" in md_text or "Caveats" in md_text
        assert "## Action Path" in md_text
        # JSON sidecar required keys
        payload = json.loads(results_json.read_text())
        assert payload["phase"] == "86"
        assert payload["frozen_commit"] == "5a1ecc9"
        assert payload["min_pass"] == 3
        assert "BTCUSDT" in payload["per_symbol"]
        assert "ETHUSDT" in payload["per_symbol"]
        assert payload["milestone_verdict"] == "FAIL"
        assert "rationale" in payload["milestone_rationale"].lower() or len(payload["milestone_rationale"]) > 0
