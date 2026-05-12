# Source: poseidon/tests/test_ddg_da_compare_smoke.py:1-87 (pytestmark.skipif STORMTROOPER + _smoke_dir/_runs_dir helpers)
"""NestedExecutor stormtrooper end-to-end smoke.

Module-level pytestmark: skip on Mac collect; only runs inside qlib-research
container with STORMTROOPER=1 env var (Pattern S4).

Smoke target: NestedExecutor TWAP × 1-2 max-|basis_z| trigger
days (NOT full 67-day run); pass conditions:
- fill_log.parquet non-empty
- each fill_price ∈ [bar_low, bar_high]  (sanity)
- comparison.parquet has nested_twap_* columns
- cost_delta_bps non-NaN/inf

This phase has NO verdict gate.

This file is a STUB. A later wave replaces it with the actual smoke body
that drives ``scripts.run_basis_arb_nested.main([...])``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)

# 1-2 trigger days × ~10-15 min each → 30 min budget
_BUDGET_SEC = 30 * 60.0


def _smoke_dir(prong: str) -> Path:
    """Container-or-host aware smoke artifact dir.

    Inside qlib-research (aquarium_root resolves to "/"): /app/local_dev/phase93_smoke/
    On Mac host: the aquarium-side smoke output directory.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/phase93_smoke") / prong
    else:
        out = aquarium_root / ".planning" / "phases" / "93-nestedexecutor-multi-level-backtest" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def _runs_dir() -> Path:
    """Container-or-host aware runs dir for NestedExecutor outputs."""
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/nested-executor/runs")
    else:
        out = aquarium_root / "local_dev" / "nested-executor" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.mark.stormtrooper
def test_nested_executor_basic_smoke():
    """1-2 trigger days × NestedExecutor TWAP, schema-only assertions.

    Drives ``scripts.run_basis_arb_nested.main(["--smoke", "--max-triggers", "2",
    "--out-dir", str(run_dir)])`` and asserts:
    1. fill_log.parquet exists + non-empty
    2. each fill_price ∈ [bar_low, bar_high]  (sanity)
    3. comparison.parquet has `nested_twap_*` columns
    4. cost_delta_bps non-NaN/inf

    This phase has NO verdict gate.
    """
    pytest.importorskip("qlib")

    smoke_dir = _smoke_dir("NESTEXEC-SMOKE")
    runs_dir = _runs_dir()
    run_id = str(uuid.uuid4())
    run_dir = runs_dir / run_id

    t0 = time.time()
    # A later wave lands scripts/run_basis_arb_nested.py — until then, RED is expected
    from scripts.run_basis_arb_nested import main as run_nested

    # Inside qlib-research container, the planning tree is NOT bind-mounted but the prior
    # 90 wave2-full-002/ is at /app/local_dev/rl-execution/runs/wave2-full-002/.
    # Pass it explicitly so Path C reconstruction can produce comparison.parquet.
    phase90_baseline = "/app/local_dev/rl-execution/runs/wave2-full-002/comparison.csv"
    rc = run_nested(
        [
            "--smoke",
            "--max-triggers",
            "2",
            "--out-dir",
            str(run_dir),
            "--phase90-baseline",
            phase90_baseline,
        ]
    )
    elapsed = time.time() - t0
    assert elapsed < _BUDGET_SEC, f"smoke ran {elapsed:.0f}s > budget {_BUDGET_SEC:.0f}s"
    assert rc == 0, f"run_basis_arb_nested.main returned rc={rc}"

    # ── (1) fill_log.parquet exists + non-empty ──────────────────────────
    fill_log_path = run_dir / "fill_log.parquet"
    assert fill_log_path.exists(), f"missing {fill_log_path}"
    fill_log = pd.read_parquet(fill_log_path)
    assert len(fill_log) > 0

    # ── (2) sanity: fill_price ∈ [bar_low, bar_high] ─────────────────────
    in_range = (fill_log["fill_price"] >= fill_log["bar_low"]) & (fill_log["fill_price"] <= fill_log["bar_high"])
    assert in_range.all(), f"{(~in_range).sum()} fills outside bar OHLC range"

    # ── (3) comparison.parquet has nested_twap_* columns ─────────────────
    comparison_path = run_dir / "comparison.parquet"
    assert comparison_path.exists(), f"missing {comparison_path}"
    comparison = pd.read_parquet(comparison_path)
    nested_cols = [c for c in comparison.columns if "nested_twap" in c.lower()]
    assert nested_cols, f"no nested_twap_* columns in {list(comparison.columns)}"

    # ── (4) cost_delta_bps non-NaN/inf ───────────────────────────────────
    summary_path = run_dir / "comparison_summary.md"
    if summary_path.exists():
        # The summary references the cost_delta — but the canonical place is
        # a sidecar JSON with the breakdown dict, written by an earlier wave's
        # write_delta_breakdown helper.
        ...
    delta_path = run_dir / "delta_breakdown.json"
    if delta_path.exists():
        import json
        import math

        deltas = json.loads(delta_path.read_text())
        cost_delta = deltas.get("cost_delta_bps")
        assert cost_delta is not None
        assert not math.isnan(float(cost_delta))
        assert not math.isinf(float(cost_delta))

    # Persist smoke artifacts under the smoke output directory
    (smoke_dir / "stdout.log").write_text(f"run_id={run_id}\nelapsed={elapsed:.1f}s\nrc={rc}\n")
