"""Phase 95 ACTIVATE-03 — Signal Analysis / IC report smoke (stormtrooper-only).

Pattern S4 (STORMTROOPER gate) + Pattern P9 (`importorskip` inside the body)
+ Pitfall 3 / Pattern P10 (MultiIndex via ``make_synthetic_anchor_signal``).

Persists smoke artifacts to
``.planning/phases/95-*/smoke/ACTIVATE-03/`` plus the production output
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

# D-31 5-prong smoke budget = 30 min total → ≤6 min per prong; signal analysis
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
    * All three D-15 output files persisted (ic.json + ic_decay.parquet +
      group_analysis.parquet).
    * D-16 comparison_vs_v18.json present (v18 baseline may be null).
    * IC summary has expected keys; n_dates > 30 (sample sanity).
    * IC decay parquet has columns ``[lag, ic_mean, n]`` and at least one row.
    * Wall-clock < ``_BUDGET_SEC`` (D-31).
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

    # === D-15: 3 standard signal-analysis outputs ===
    assert status == "OK", f"signal analysis smoke {status}: {error}"
    assert (out_dir / "ic.json").exists(), "D-15.1 ic.json missing"
    assert (out_dir / "ic_decay.parquet").exists(), "D-15.2 ic_decay.parquet missing"
    assert (out_dir / "group_analysis.parquet").exists(), "D-15.3 group_analysis.parquet missing"
    # === D-16: comparison vs v18 perf() ===
    assert (out_dir / "comparison_vs_v18.json").exists(), "D-16 comparison_vs_v18.json missing"

    # === IC summary sanity (synthetic fixture has 0.15 correlation built-in) ===
    ic_summary = json.loads((out_dir / "ic.json").read_text())
    for key in ("ic_mean", "ic_std", "icir", "rank_ic_mean", "rank_icir", "n_dates"):
        assert key in ic_summary, f"ic.json missing key {key}"
    assert ic_summary["n_dates"] > 30, (
        f"insufficient sample (n_dates={ic_summary['n_dates']}) — synthetic fixture default n_days=400"
    )

    # === IC decay table sanity ===
    import pandas as pd

    decay = pd.read_parquet(out_dir / "ic_decay.parquet")
    assert len(decay) >= 1, "ic_decay.parquet must have ≥1 lag row"
    assert {"lag", "ic_mean", "n"}.issubset(decay.columns), (
        f"ic_decay missing required columns; got {list(decay.columns)}"
    )

    # === Group analysis sanity ===
    group = pd.read_parquet(out_dir / "group_analysis.parquet")
    assert {"long_short", "long_avg"}.issubset(group.columns), (
        f"group_analysis missing required columns; got {list(group.columns)}"
    )

    # === D-16 comparison shape ===
    comp = json.loads((out_dir / "comparison_vs_v18.json").read_text())
    assert "qlib_signal_analysis" in comp, "comparison_vs_v18 missing qlib_signal_analysis block"
    assert "v18_perf_full" in comp, "comparison_vs_v18 missing v18_perf_full block"
    # v18_perf_full may be null on Mac path; on stormtrooper baseline may be absent
    # (Plan 95-03 generated it once but it's not in scripts/output by default).

    assert elapsed < _BUDGET_SEC, f"wall-clock {elapsed:.1f}s >= {_BUDGET_SEC}s budget (D-31)"
