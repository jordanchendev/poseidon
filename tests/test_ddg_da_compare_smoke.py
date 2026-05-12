"""DDG-DA comparison stormtrooper end-to-end smoke.

Module-level pytestmark: skip on Mac collect; only runs inside qlib-research
container with STORMTROOPER=1 env var (Pattern S4).

What this exercises:
  - poseidon.autoresearch.ddg_da_compare.run_comparison(...) end-to-end with
    smoke=True (last 2 walk-forward folds only)
  - PoseidonDDGDA wrapper actually loads + emits YAML +
    Rolling.run() against the qlib_data tree at
    /root/.qlib/qlib_data/poseidon_tw_futures/
  - poseidon.autoresearch.ddg_da_verdict.choose_verdict + write_verdict_md
    consume the smoke output

Pass conditions:
  - run_dir/with_ddg_da/, run_dir/without_ddg_da/, comparison_summary.parquet,
    verdict.md all exist
  - Per-window Sharpe values in comparison_summary are non-NaN
  - verdict.md contains one of {rescue, partial-help, no-effect,
    no-effect (insufficient power)} keyword
  - Wall-clock <= 45 minutes (per Pitfall 5 / feedback
    latency budget; smoke uses last-2-fold so should be far under)
  - Smoke verdict can be ANY of the four legal keywords (explicitly
    does NOT require rescue — pipeline-pass is the goal)
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


# Per-prong wall-clock budget — Pitfall 5 / feedback
# latency.
_BUDGET_SEC = 45 * 60.0  # 45 minutes


def _smoke_dir(prong: str) -> Path:
    """Resolve the smoke output directory for the given prong.

    poseidon/tests/test_ddg_da_compare_smoke.py → parents[0]=tests/,
    parents[1]=poseidon/, parents[2]=aquarium root (where the planning tree lives).
    NEVER take a path from user input — Pitfall 6.

    Inside the qlib-research container the bind-mount maps
    aquarium/poseidon/tests → /app/tests, so parents[2] is "/" rather than the
    real aquarium root. Detect this and fall back to /app/local_dev which IS
    bind-mounted — keeps smoke artifacts host-visible for the verifier.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/phase92_smoke") / prong
    else:
        out = aquarium_root / ".planning" / "phases" / "92-ddg-da-domain-adaptation" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def _runs_dir() -> Path:
    """Resolve local_dev/ddg-da/runs/ host-visible bind-mounted location.

    Per the persistence layout. Inside qlib-research container this
    is /app/local_dev/ddg-da/runs/ (bind-mounted to host
    aquarium/local_dev/ddg-da/runs/).
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/ddg-da/runs")
    else:
        out = aquarium_root / "local_dev" / "ddg-da" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.mark.stormtrooper
def test_ddg_da_compare_smoke():
    """End-to-end with last 2 walk-forward folds only."""
    pytest.importorskip("qlib")

    smoke_dir = _smoke_dir("DDGDA-SMOKE")
    runs_dir = _runs_dir()

    # Full UUID4 for path-traversal safety; never accept caller-
    # supplied id.
    run_id = str(uuid.uuid4())
    run_dir = runs_dir / run_id

    status = "OK"
    error: str | None = None
    summary: dict = {}
    t0 = time.time()
    try:
        from poseidon.autoresearch.ddg_da_compare import run_comparison
        from poseidon.autoresearch.ddg_da_verdict import (
            choose_verdict,
            write_verdict_md,
        )

        # Option B segments — train 33mo / test 28-fold
        # (2024-01-01..2026-04-30). Smoke trims to last 2 folds via
        # smoke=True flag in run_comparison.
        # BUG-7 fix: trimmed test end_time from 2026-05-04 to
        # 2026-04-30 to stay inside the TX qlib_data tree boundary
        # (ingest produced data through 2026-05-01 per
        # instruments/all.txt: "TX 2021-03-22 2026-05-01"). The previous
        # 2026-05-04 end caused the last walk-forward fold to slice past
        # the data tail → empty dataset → lightgbm refuses to train with
        # ValueError "Empty data from dataset".
        segments = {
            "train": ("2021-03-22", "2023-08-31"),
            "valid": ("2023-09-01", "2023-12-31"),
            "test": ("2024-01-01", "2026-04-30"),
        }
        result = run_comparison(
            thesis_name="tx_gap_intraday",
            model_class="LGBModel",
            segments=segments,
            run_dir=run_dir,
            smoke=True,
        )

        # Read the comparison summary back; choose a verdict (smoke verdict
        # acceptable as ANY of the legal keywords).
        df = pd.read_parquet(result["comparison_summary_parquet"])
        bs = result["bootstrap_result"]
        v = choose_verdict(df, bs)
        verdict_path = write_verdict_md(
            v,
            df,
            bs,
            run_dir / "verdict.md",
            thesis_name="tx_gap_intraday",
            model_class="LGBModel",
        )
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "n_folds": result["n_folds"],
            "verdict": v["verdict"],
            "verdict_path": str(verdict_path),
            "bootstrap": bs,
        }
    except Exception:
        status = "PARTIAL"
        error = traceback.format_exc()
    elapsed = time.time() - t0

    # Persist machine-readable per-prong summary (Pattern P3 mirror).
    (smoke_dir / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "DDGDA-SMOKE",
                "status": status,
                "elapsed_sec": round(elapsed, 2),
                "summary": summary if status == "OK" else None,
                "error": error,
            },
            indent=2,
            default=str,
        )
    )

    # Hard assertions.
    assert status == "OK", f"DDG-DA smoke {status}: {error}"

    # Artifact existence.
    assert (run_dir / "with_ddg_da").is_dir(), "with_ddg_da/ missing"
    assert (run_dir / "without_ddg_da").is_dir(), "without_ddg_da/ missing"
    assert (run_dir / "comparison_summary.parquet").exists(), "comparison_summary.parquet missing"
    assert (run_dir / "verdict.md").exists(), "verdict.md missing"

    # Non-NaN windows-Sharpe.
    df = pd.read_parquet(run_dir / "comparison_summary.parquet")
    assert len(df) >= 1, f"comparison_summary.parquet empty (rows={len(df)})"
    assert not df["sharpe_with"].isna().all(), "all sharpe_with NaN"
    assert not df["sharpe_without"].isna().all(), "all sharpe_without NaN"

    # Verdict keyword check.
    body = (run_dir / "verdict.md").read_text()
    assert any(f"# Verdict: {kw}" in body for kw in ("rescue", "partial-help", "no-effect")), (
        "no verdict keyword in verdict.md heading"
    )

    # Wall-clock budget (Pitfall 5 / feedback latency).
    assert elapsed < _BUDGET_SEC, f"smoke exceeded {_BUDGET_SEC}s budget: {elapsed:.1f}s"
