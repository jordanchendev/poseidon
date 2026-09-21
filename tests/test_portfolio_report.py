"""ACTIVATE-05 — Portfolio attribution / plotly report smoke.

Stormtrooper-only (Pattern S4). Loads the qrun
SignalRecord pickles (pred.pkl + label.pkl) and renders the qlib + plotly
graphs that don't require PortAnaRecord — see run_portfolio_report.py
docstring for the full contract amendment.

Skip-with-reason if the qrun mlruns directory missing — that's acceptable
(test isolation).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


_BUDGET_SEC = 300.0
_RECORDER_DIR = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")


def test_portana_label_uses_instrument_then_datetime_index() -> None:
    """PortAna slices its label data on MultiIndex level 1 (datetime)."""
    import pandas as pd

    from scripts.run_portfolio_report import _label_for_position_graphs

    index = pd.MultiIndex.from_tuples([("2025-01-03", "TX"), ("2025-01-02", "TX")], names=["datetime", "instrument"])
    label = pd.DataFrame({"label": [0.2, 0.1]}, index=index)

    actual = _label_for_position_graphs(label)

    assert actual.index.names == ["instrument", "datetime"]
    assert actual.index.get_level_values("datetime").dtype.kind == "M"
    assert actual.index.is_monotonic_increasing


def _smoke_dir(prong: str) -> Path:
    """Resolve the smoke output directory for the given prong.

    Inside the qlib-research container the bind-mount maps
    aquarium/poseidon/tests → /app/tests, so parents[2] is "/" rather than the
    real aquarium root. Detect this and fall back to /app/local_dev/phase95_smoke
    which IS bind-mounted (matches Pattern S4 carry-over from test_alpha158_eval.py).
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/phase95_smoke") / prong
    else:
        out = aquarium_root / ".planning" / "phases" / "95-activate-underutilised-qlib-surface" / "smoke" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_portfolio_report_smoke(tmp_path: Path) -> None:
    """Fresh qrun artifacts drive all six Qlib report graph families."""
    pytest.importorskip("qlib")
    pytest.importorskip("plotly")

    provider_uri = os.environ.get("PHASE95_QLIB_PROVIDER_URI")
    assert provider_uri, "set PHASE95_QLIB_PROVIDER_URI to a fresh TX+0050 Qlib dump"
    out_dir = _smoke_dir("ACTIVATE-05")
    backtest_out = tmp_path / "reports"
    t0 = time.time()
    from scripts.run_portfolio_report import run_portfolio_report
    from scripts.run_qrun_basis_vol import run_qrun_basis_vol

    artifacts_dir = run_qrun_basis_vol(uri_folder=tmp_path / "mlruns", provider_uri=provider_uri)
    summary = run_portfolio_report(out_dir=backtest_out, artifacts_dir=artifacts_dir)
    elapsed = time.time() - t0

    (out_dir / "output_summary.json").write_text(
        json.dumps(
            {
                "prong": "ACTIVATE-05",
                "status": "OK",
                "elapsed_sec": elapsed,
                "summary": summary,
                "artifacts_dir": str(artifacts_dir),
            },
            indent=2,
            default=str,
        )
    )

    # ROADMAP success criterion 5: "qlib's Portfolio Attribution / Graphical
    # Reports generated alongside at least one backtest run output". Drives the
    # ≥1 HTML assertion.
    htmls = sorted(backtest_out.glob("*.html"))
    assert len(htmls) >= 1, f"expected ≥1 HTML files, got {len(htmls)}: {htmls}"
    for html in htmls:
        size = html.stat().st_size
        assert size > 1024, f"{html.name} size {size}B is below 1KB threshold"

    assert summary["n_skipped"] == 0, summary
    assert summary["n_partial"] == 0, summary
    assert summary["n_not_applicable"] == 1, summary
    ok_graphs = summary["graph_name_list_coverage"]["ok"]
    assert len(ok_graphs) == 5, ok_graphs

    assert elapsed < _BUDGET_SEC, f"elapsed {elapsed:.1f}s exceeds budget {_BUDGET_SEC}s"
