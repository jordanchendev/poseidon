"""Phase 95 ACTIVATE-02 — qrun YAML pipeline smoke (Wave 2).

Two distinct gates per Plan 95-03:

  1. ``test_qrun_yaml_loads`` — runs on Mac (no STORMTROOPER guard). Asserts
     ``qrun_configs/v18/tx_basis_vol.yml`` parses cleanly and every ``class:``
     entry resolves through ``poseidon.qlib.allowlist`` (Pattern P8 / T-95-01
     RCE boundary). Defends against allowlist drift catching the issue before
     the stormtrooper smoke runs.

  2. ``test_qrun_smoke`` — stormtrooper-only (Pattern S4). Drives the full
     qrun workflow end-to-end via ``scripts.run_qrun_basis_vol.run_qrun_basis_vol``
     and applies the D-11 amended parity check. Status OK iff sign agreement
     AND magnitude ratio ∈ [0.5, 2.0]. PARTIAL with structured root cause is
     acceptable per D-12 (does not block the phase).

Pattern P9: ``import qlib`` only via ``pytest.importorskip`` inside test
bodies. Module-top imports are stdlib-only.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import pytest

STORMTROOPER_GATE = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)

# D-31 budget: 600s (10 min) for the qrun smoke (model train + record chain).
_BUDGET_SEC = 600.0


def _smoke_dir(prong: str) -> Path:
    """Resolve .planning/phases/95-*/smoke/{prong}/ from this file's path.

    Inside the qlib-research container the bind-mount maps
    aquarium/poseidon/tests → /app/tests, so ``parents[2]`` is "/" rather than
    the real aquarium root. Detect this and fall back to /app/local_dev which
    IS bind-mounted — keeps smoke artifacts host-visible. Mirrors the carry-
    forward helper from test_alpha158_eval.py (Plan 95-02).
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
    """T-95-01 RCE boundary: every YAML ``class:`` resolves via allowlist.

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


@STORMTROOPER_GATE
def test_qrun_smoke() -> None:
    """ACTIVATE-02 smoke: drive qrun YAML end-to-end + run D-11 parity check.

    Runs ``run_qrun_basis_vol()`` inside the qlib-research container. After the
    workflow completes, asserts the recorder directory + mlruns/ exist (Wave 5
    consumer contract) and runs the parity check. PARTIAL is acceptable per
    D-12; only an empty recorder dir or a hard exception fails the test.
    """
    pytest.importorskip("qlib")

    smoke_dir = _smoke_dir("ACTIVATE-02")
    recorder_dir = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")

    status: str = "OK"
    error: str | None = None
    parity: dict | None = None
    t0 = time.time()
    try:
        from scripts.run_qrun_basis_vol import run_qrun_basis_vol
        from scripts.run_qrun_parity_check import parity_check

        run_qrun_basis_vol()

        # D-11 amended parity check — never raises, returns dict.
        v18_json = Path("/app/scripts/output/tx_walkforward_v2.json")
        parity = parity_check(recorder_dir=recorder_dir, v18_json_path=v18_json)
        if parity.get("status") != "OK":
            status = "PARTIAL"

        # Persist parity.json next to mlruns/ so Wave 5 + verifier can find it.
        parity_path = recorder_dir / "parity.json"
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        parity_path.write_text(json.dumps(parity, indent=2, default=str))

    except Exception:
        status = "PARTIAL"
        error = traceback.format_exc()
    elapsed = time.time() - t0

    # Pattern P3 — persist machine-readable per-prong summary.
    (smoke_dir / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "ACTIVATE-02",
                "status": status,
                "elapsed_sec": round(elapsed, 2),
                "parity": parity,
                "recorder_dir": str(recorder_dir),
                "error": error,
            },
            indent=2,
            default=str,
        )
    )

    # Hard assertions: PortAnaRecord pickles must exist (Wave 5 dependency).
    # An empty recorder dir means qrun never ran — that IS a phase blocker.
    assert (recorder_dir / "mlruns").exists(), (
        f"mlruns directory missing at {recorder_dir / 'mlruns'} — qrun did not produce output"
    )

    # PARTIAL is acceptable per D-12 (parity failure is not a phase blocker).
    # Hard exceptions outside parity_check are still fatal because they mean
    # the smoke could not even complete the workflow.
    if error is not None:
        # If the failure is purely the parity check signalling PARTIAL, error
        # would still be None (parity_check never raises). A non-None error
        # means the workflow itself or another path raised — fail.
        pytest.fail(f"qrun smoke raised: {error}")

    assert status in ("OK", "PARTIAL"), f"unexpected status {status}"
    assert elapsed < _BUDGET_SEC, f"qrun smoke exceeded {_BUDGET_SEC}s budget: {elapsed:.1f}s"
