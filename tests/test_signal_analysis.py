"""ACTIVATE-03 — Signal Analysis / IC report smoke (stormtrooper-only).

Pattern S4 (STORMTROOPER gate) + Pattern P9 (`importorskip` inside the body)
+ Pitfall 3 / Pattern P10 (MultiIndex via ``make_synthetic_anchor_signal``).

Persists smoke artifacts to
the smoke output directory plus the production output
sink at ``local_dev/qlib-activations/signal-analysis/basis_arb/`` via the
``run_signal_analysis`` driver.
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

# 5-prong smoke budget = 30 min total → ≤6 min per prong; signal analysis
# is the cheapest of the 5 (no model train, no data fetch). 300 s gives 50x
# headroom over expected wall-clock (~5 s for 400-day synthetic panel).
_BUDGET_SEC = 300.0


def _smoke_dir(prong: str) -> Path:
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_signal_analysis_smoke() -> None:
    """ACTIVATE-03 smoke: synthetic anchor → run_signal_analysis → 4 outputs.

    Asserts:
    * All three output files persisted (ic.json + ic_decay.parquet +
      group_analysis.parquet).
    * comparison_vs_v18.json present (v18 baseline may be null).
    * A one-instrument panel records cross-sectional metrics as not applicable.
    * IC decay parquet has columns ``[lag, ic_mean, n]`` and no fabricated rows.
    * Wall-clock < ``_BUDGET_SEC``.
    """
    pytest.importorskip("qlib")
    from tests.conftest import make_synthetic_anchor_signal

    out_dir = _smoke_dir("ACTIVATE-03")
    pred, label = make_synthetic_anchor_signal(n_days=400)

    status, error, summary = "OK", None, None
    t0 = time.time()
    try:
        from scripts.run_signal_analysis import run_signal_analysis

        summary = run_signal_analysis(pred=pred, label=label, out_dir=out_dir)
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0

    (out_dir / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "ACTIVATE-03",
                "status": status,
                "elapsed_sec": elapsed,
                "ic_mean": summary.get("ic_mean") if summary else None,
                "icir": summary.get("icir") if summary else None,
                "rank_ic_mean": summary.get("rank_ic_mean") if summary else None,
                "ann_long_short_sharpe": (summary.get("ann_long_short_sharpe") if summary else None),
                "ann_long_avg_sharpe": (summary.get("ann_long_avg_sharpe") if summary else None),
                "n_dates": summary.get("n_dates") if summary else None,
                "error": error,
            },
            indent=2,
        )
    )

    # === 3 standard signal-analysis outputs ===
    assert status == "OK", f"signal analysis smoke {status}: {error}"
    assert (out_dir / "ic.json").exists(), "ic.json missing"
    assert (out_dir / "ic_decay.parquet").exists(), "ic_decay.parquet missing"
    assert (out_dir / "group_analysis.parquet").exists(), "group_analysis.parquet missing"
    # === Comparison vs v18 perf() ===
    assert (out_dir / "comparison_vs_v18.json").exists(), "comparison_vs_v18.json missing"

    # === IC summary sanity ===
    ic_summary = json.loads((out_dir / "ic.json").read_text())
    for key in ("ic_mean", "ic_std", "icir", "rank_ic_mean", "rank_icir", "n_dates"):
        assert key in ic_summary, f"ic.json missing key {key}"
    assert ic_summary["n_dates"] > 30, (
        f"insufficient sample (n_dates={ic_summary['n_dates']}) — synthetic fixture default n_days=400"
    )
    assert ic_summary["cross_sectional"]["status"] == "NOT_APPLICABLE"
    assert ic_summary["ic_mean"] is None
    assert ic_summary["rank_ic_mean"] is None
    assert "NaN" not in (out_dir / "ic.json").read_text()

    # === IC decay table sanity ===
    import pandas as pd

    decay = pd.read_parquet(out_dir / "ic_decay.parquet")
    assert {"lag", "ic_mean", "n"}.issubset(decay.columns), (
        f"ic_decay missing required columns; got {list(decay.columns)}"
    )
    assert decay.empty, "single-instrument IC decay must not fabricate cross-sectional values"

    # === Group analysis sanity ===
    group = pd.read_parquet(out_dir / "group_analysis.parquet")
    assert {"long_short", "long_avg"}.issubset(group.columns), (
        f"group_analysis missing required columns; got {list(group.columns)}"
    )
    assert summary["long_short"]["status"] == "NOT_APPLICABLE"
    assert summary["ann_long_short_sharpe"] is None
    assert summary["ann_long_avg_return"] is not None

    # === Comparison shape ===
    comp = json.loads((out_dir / "comparison_vs_v18.json").read_text())
    assert "qlib_signal_analysis" in comp, "comparison_vs_v18 missing qlib_signal_analysis block"
    assert "v18_perf_full" in comp, "comparison_vs_v18 missing v18_perf_full block"
    assert "NaN" not in (out_dir / "comparison_vs_v18.json").read_text()
    # v18_perf_full may be null on Mac path; on stormtrooper baseline may be absent
    # (A prior plan generated it once but it's not in scripts/output by default).

    assert elapsed < _BUDGET_SEC, f"wall-clock {elapsed:.1f}s >= {_BUDGET_SEC}s budget"
