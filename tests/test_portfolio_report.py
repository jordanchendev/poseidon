"""Phase 95 ACTIVATE-05 — Portfolio attribution / plotly report smoke.

Stormtrooper-only (Pattern S4). Loads the Wave 1 (Plan 95-03) qrun
SignalRecord pickles (pred.pkl + label.pkl) and renders the qlib + plotly
graphs that don't require PortAnaRecord — see run_portfolio_report.py
docstring for the full Wave 5 contract amendment.

Skip-with-reason if Wave 1 mlruns directory missing — that's acceptable
(test isolation per CONTEXT D-33).
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


_BUDGET_SEC = 300.0
_RECORDER_DIR = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")


def _smoke_dir(prong: str) -> Path:
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_portfolio_report_smoke() -> None:
    """End-to-end driver smoke: loads Wave 1 pickles, emits HTMLs, asserts size.

    Per the Wave 5 contract amendment (95-03 SUMMARY Deviation #4), this only
    asserts on SignalRecord-derived graphs (score_ic_graph,
    model_performance_graph). PortAnaRecord-dependent graphs are SKIPPED in
    the driver — counted in summary.n_skipped, not asserted on.
    """
    pytest.importorskip("qlib")
    pytest.importorskip("plotly")

    out_dir = _smoke_dir("ACTIVATE-05")
    recorder_dir = _RECORDER_DIR

    # Wave 1 dependency — skip-with-reason if pickles missing (Pitfall 5)
    mlruns = recorder_dir / "mlruns"
    if not mlruns.exists():
        pytest.skip(f"Wave 1 qrun mlruns missing at {mlruns} — run scripts/run_qrun_basis_vol.py first (Plan 95-03)")

    backtest_out = Path("/app/local_dev/backtests/phase95_basis_vol/reports")
    backtest_out.mkdir(parents=True, exist_ok=True)

    status: str = "OK"
    error: str | None = None
    summary: dict | None = None
    t0 = time.time()
    try:
        from scripts.run_portfolio_report import run_portfolio_report

        summary = run_portfolio_report(recorder_dir=recorder_dir, out_dir=backtest_out)
    except Exception:
        status = "PARTIAL"
        error = traceback.format_exc()
    elapsed = time.time() - t0

    (out_dir / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "ACTIVATE-05",
                "status": status,
                "elapsed_sec": elapsed,
                "summary": summary,
                "error": error,
            },
            indent=2,
            default=str,
        )
    )

    assert status == "OK", f"portfolio_report smoke {status}: {error}"
    assert summary is not None

    # ROADMAP success criterion 5: "qlib's Portfolio Attribution / Graphical
    # Reports generated alongside at least one backtest run output". Drives the
    # ≥1 HTML assertion.
    htmls = sorted(backtest_out.glob("*.html"))
    assert len(htmls) >= 1, f"expected ≥1 HTML files, got {len(htmls)}: {htmls}"
    for html in htmls:
        size = html.stat().st_size
        assert size > 1024, f"{html.name} size {size}B is below 1KB threshold"

    # Wave 5 contract amendment: PARTIAL count (PortAnaRecord-dependent SKIPS)
    # ≤ 4 of GRAPH_NAME_LIST (6 entries). The four SKIPPED graphs are
    # cumulative_return / risk_analysis / report / rank_label.
    assert summary["n_skipped"] <= 4, (
        f"too many SKIPPED graphs: {summary['n_skipped']} (expected ≤4 — "
        f"only PortAnaRecord-dependent graphs should skip)"
    )

    # At least 1 OK graph (sufficient for ROADMAP success criterion 5)
    assert summary["n_ok"] >= 1, f"no OK graphs emitted — ROADMAP criterion 5 not met. summary={summary}"

    # Specifically expect score_ic_graph to OK (simplest/most reliable)
    ok_graphs = summary["graph_name_list_coverage"]["ok"]
    assert "analysis_position.score_ic_graph" in ok_graphs, (
        f"score_ic_graph should be OK on pred+label data; got OK list: {ok_graphs}"
    )

    assert elapsed < _BUDGET_SEC, f"elapsed {elapsed:.1f}s exceeds budget {_BUDGET_SEC}s"
