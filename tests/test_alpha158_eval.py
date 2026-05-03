"""Phase 95 ACTIVATE-01 — Alpha158 production-signal smoke (stormtrooper-only).

Wave 1 implementation. Pattern S4 (STORMTROOPER gate) + Pattern P9
(``pytest.importorskip`` for qlib inside the test body) keep Mac-side
``pytest --collect-only`` healthy while the actual smoke runs in the
qlib-research container.

What this test exercises:

* ``scripts.run_alpha158_eval.run_alpha158_eval`` end-to-end against the
  Wave 0 ``make_synthetic_basis_arb_panel`` fixture (400 daily bars, single
  instrument "TX").
* Asserts the D-07 artifact triplet (features.parquet, performance.json,
  summary.json) is persisted to ``.planning/phases/95-*/smoke/ACTIVATE-01/``.
* Asserts D-08 summary keys: ``n_features >= 158``, ``n_trigger_days > 30``
  (synthetic 400-day mask consistently yields ~50 triggers).
* Wall-clock <300s budget (D-31).

Pattern P3 artifact triplet — ``output_summary.json`` mirrors the prong
status / elapsed shape used by the Phase 94 zoo smoke and the eventual
Phase 95 Wave 6 cross-prong roll-up.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


# Per-prong wall-clock budget per CONTEXT D-31 (5 minutes / 300 seconds).
_BUDGET_SEC = 300.0


def _smoke_dir(prong: str) -> Path:
    """Resolve .planning/phases/95-*/smoke/{prong}/ from this file's path.

    poseidon/tests/test_alpha158_eval.py → parents[0]=tests/, parents[1]=poseidon/,
    parents[2]=aquarium root (where .planning/ lives). NEVER take a path from
    user input — D-13 + Pitfall 6.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_alpha158_eval_smoke() -> None:
    """ACTIVATE-01 smoke: run Alpha158 eval against synthetic basis arb panel."""
    pytest.importorskip("qlib")

    from tests.conftest import make_synthetic_basis_arb_panel

    out_dir = _smoke_dir("ACTIVATE-01")
    panel = make_synthetic_basis_arb_panel(n_days=400)

    status = "OK"
    error: str | None = None
    summary: dict = {}
    t0 = time.time()
    try:
        from scripts.run_alpha158_eval import run_alpha158_eval

        summary = run_alpha158_eval(panel=panel, out_dir=out_dir)
    except Exception:
        status = "PARTIAL"
        error = traceback.format_exc()
    elapsed = time.time() - t0

    # Pattern P3 — persist machine-readable per-prong summary.
    (out_dir / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "ACTIVATE-01",
                "status": status,
                "elapsed_sec": round(elapsed, 2),
                "n_features": summary.get("n_features") if status == "OK" else None,
                "n_trigger_days": summary.get("n_trigger_days") if status == "OK" else None,
                "n_evaluated_features": summary.get("n_evaluated_features") if status == "OK" else None,
                "top_5_features_by_abs_ic": summary.get("top_5_features_by_abs_ic") if status == "OK" else None,
                "anchor_signal": summary.get("anchor_signal") if status == "OK" else None,
                "error": error,
            },
            indent=2,
            default=str,
        )
    )

    # Hard assertions — fail fast on regression.
    assert status == "OK", f"alpha158 smoke {status}: {error}"

    features_path = out_dir / "features.parquet"
    assert features_path.exists(), "features.parquet missing"
    assert features_path.stat().st_size > 0, "features.parquet is empty"

    assert (out_dir / "performance.json").exists(), "performance.json missing"
    assert (out_dir / "summary.json").exists(), "summary.json missing"

    # VALIDATION.md / CONTEXT D-08: ≥158 Alpha158 features.
    assert summary["n_features"] >= 158, f"expected ≥158 features, got {summary['n_features']}"

    # Synthetic 400-day panel + basis_z<-1 mask consistently yields ~50 triggers.
    assert summary["n_trigger_days"] > 30, (
        f"expected >30 trigger days from synthetic panel, got {summary['n_trigger_days']}"
    )

    # n_evaluated_features should equal n_features (every feature attempted).
    assert summary["n_evaluated_features"] >= 158, (
        f"expected ≥158 evaluated features, got {summary['n_evaluated_features']}"
    )

    # D-31 wall-clock budget.
    assert elapsed < _BUDGET_SEC, f"alpha158 smoke exceeded {_BUDGET_SEC}s budget: {elapsed:.1f}s"
