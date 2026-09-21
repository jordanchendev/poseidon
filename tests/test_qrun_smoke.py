"""ACTIVATE-02 — qrun YAML pipeline smoke.

Two distinct gates:

  1. ``test_qrun_yaml_loads`` — runs on Mac (no STORMTROOPER guard). Asserts
     ``qrun_configs/v18/tx_basis_vol.yml`` parses cleanly and every ``class:``
     entry resolves through ``poseidon.qlib.allowlist`` (Pattern P8 / RCE boundary).
     Defends against allowlist drift catching the issue before
     the stormtrooper smoke runs.

  2. ``test_qrun_smoke`` — stormtrooper-only (Pattern S4). Drives the full
     qrun workflow end-to-end via ``scripts.run_qrun_basis_vol.run_qrun_basis_vol``
     and applies the parity check. Status OK iff sign agreement
     AND magnitude ratio ∈ [0.5, 2.0]. PARTIAL with structured root cause is
     acceptable (does not block the phase).

Pattern P9: ``import qlib`` only via ``pytest.importorskip`` inside test
bodies. Module-top imports are stdlib-only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

STORMTROOPER_GATE = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)

# Budget: 600s (10 min) for the qrun smoke (model train + record chain).
_BUDGET_SEC = 600.0


def _smoke_dir(prong: str) -> Path:
    """Resolve the smoke output directory for the given prong.

    Inside the qlib-research container the bind-mount maps
    aquarium/poseidon/tests → /app/tests, so ``parents[2]`` is "/" rather than
    the real aquarium root. Detect this and fall back to /app/local_dev which
    IS bind-mounted — keeps smoke artifacts host-visible. Mirrors the carry-
    forward helper from test_alpha158_eval.py.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/phase95_smoke") / prong
    else:
        out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_qrun_yaml_loads() -> None:
    """RCE boundary: every YAML ``class:`` resolves via allowlist.

    No STORMTROOPER gate — runs on Mac so allowlist drift surfaces in CI
    before the stormtrooper smoke. Pattern P8: handler/model class names are
    free-form strings in YAML; they MUST round-trip through
    ``resolve_handler``/``resolve_model`` and the resolved FQN must agree with
    the YAML's stated ``module_path`` (defense in depth).
    """
    import yaml

    from poseidon.qlib.allowlist import resolve_handler, resolve_model

    cfg_path = Path(__file__).parents[1] / "qrun_configs" / "v18" / "tx_basis_vol.yml"
    assert cfg_path.exists(), f"YAML missing at {cfg_path}"

    cfg = yaml.safe_load(cfg_path.read_text())

    # Model class allowlisted.
    model_class = cfg["task"]["model"]["class"]
    model_fqn = resolve_model(model_class)
    assert model_fqn.startswith("qlib.contrib.model."), f"unexpected model FQN {model_fqn}"

    # Handler class allowlisted.
    handler_class = cfg["task"]["dataset"]["kwargs"]["handler"]["class"]
    handler_fqn = resolve_handler(handler_class)
    assert "data_handler_qrun" in handler_fqn, f"unexpected handler FQN {handler_fqn}"

    # Module paths in YAML must agree with allowlist FQNs (defense in depth).
    yaml_model_module = cfg["task"]["model"]["module_path"]
    yaml_handler_module = cfg["task"]["dataset"]["kwargs"]["handler"]["module_path"]
    assert model_fqn.startswith(yaml_model_module + "."), "YAML model module_path inconsistent with allowlist FQN"
    assert handler_fqn.startswith(yaml_handler_module + "."), "YAML handler module_path inconsistent with allowlist FQN"
    records = cfg["task"]["record"]
    basis_record = next(record for record in records if record["class"] == "BasisRuleRecord")
    assert basis_record["module_path"] == "poseidon.qlib.basis_rule_record"


@STORMTROOPER_GATE
def test_qrun_smoke(tmp_path: Path) -> None:
    """A fresh provider-backed run must emit the complete PortAnaRecord set."""
    pytest.importorskip("qlib")
    provider_uri = os.environ.get("PHASE95_QLIB_PROVIDER_URI")
    assert provider_uri, "set PHASE95_QLIB_PROVIDER_URI to a fresh TX+0050 Qlib dump"

    from scripts.run_qrun_basis_vol import run_qrun_basis_vol
    from scripts.run_qrun_parity_check import parity_check

    t0 = time.time()
    artifacts_dir = run_qrun_basis_vol(uri_folder=tmp_path / "mlruns", provider_uri=provider_uri)
    parity = parity_check(
        artifacts_dir=artifacts_dir,
        expected_returns_path=Path("/app/scripts/output/tx_walkforward_v2_basis_b_returns.parquet"),
    )
    elapsed = time.time() - t0
    (_smoke_dir("ACTIVATE-02") / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "ACTIVATE-02",
                "status": parity["status"],
                "elapsed_sec": elapsed,
                "parity": parity,
                "artifacts_dir": str(artifacts_dir),
            },
            indent=2,
            default=str,
        )
    )

    assert (artifacts_dir / "portfolio_analysis/port_analysis_1day.pkl").exists()
    assert (artifacts_dir / "basis_rule/returns.pkl").exists()
    assert (artifacts_dir / "basis_rule/engaged.pkl").exists()
    assert (artifacts_dir / "basis_rule/metrics.pkl").exists()
    assert parity["status"] == "OK", parity
    assert elapsed < _BUDGET_SEC, f"qrun smoke exceeded {_BUDGET_SEC}s budget: {elapsed:.1f}s"
