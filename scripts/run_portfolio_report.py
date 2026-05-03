#!/usr/bin/env python3
"""ACTIVATE-05: emit qlib graphical reports from anchor signal qrun output.

Per Phase 95 CONTEXT D-26..D-29. Reuses Wave 1 (Plan 95-03) qrun
SignalRecord pickles emitted at::

    local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol/mlruns/

**Wave 1 amendment (95-03 SUMMARY):** PortAnaRecord pickles are deliberately
absent — qlib's ``PortfolioMetrics.init_bench`` requires ``qlib.data.D``
provider data which Pitfall 1 bypasses (``provider_uri=""`` so the
``PoseidonDataHandlerForQrun.StaticDataLoader`` path can serve the basis arb
panel without a qlib ``.bin`` calendar). The four PortAnaRecord-dependent
graphs in qlib's ``GRAPH_NAME_LIST`` (``cumulative_return_graph``,
``risk_analysis_graph``, ``report_graph``, ``rank_label_graph``) therefore
SKIP with a structured PARTIAL note. The two SignalRecord-only graphs
(``score_ic_graph`` and ``model_performance_graph``) run end-to-end and
emit HTML — sufficient for ROADMAP success criterion 5 ("at least one
backtest run output").

Run on stormtrooper::

    docker compose exec -T qlib-research uv run python scripts/run_portfolio_report.py

The Wave 5 pytest smoke (``tests/test_portfolio_report.py``) imports
``run_portfolio_report`` and invokes it programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Bind-mounted at /app/local_dev → host poseidon/local_dev/.
# Wave 1 (Plan 95-03) writes recorder pickles under
# {DEFAULT_RECORDER_DIR}/mlruns/{exp_id}/{run_id}/artifacts/.
DEFAULT_RECORDER_DIR = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")
DEFAULT_OUT_DIR = Path("/app/local_dev/backtests/phase95_basis_vol/reports")
EXPERIMENT_NAME = "phase95_tx_basis_vol"

# qlib's GRAPH_NAME_LIST in pyqlib v0.9.7 — used to compute SKIPPED count for
# the SUMMARY / smoke output. The four below require PortAnaRecord pickles
# (position / report_normal / port_analysis) and are skipped per the Wave 1
# amendment described in the module docstring.
PORTANARECORD_DEPENDENT_GRAPHS = (
    "analysis_position.cumulative_return_graph",
    "analysis_position.risk_analysis_graph",
    "analysis_position.report_graph",
    "analysis_position.rank_label_graph",
)
SIGNALRECORD_ONLY_GRAPHS = (
    "analysis_position.score_ic_graph",
    "analysis_model.model_performance_graph",
)


def _load_signal_record(recorder_dir: Path) -> dict:
    """Load pred.pkl + label.pkl from the Wave 1 qrun mlruns directory.

    Walks the mlruns tree to find a recorder dir that contains BOTH pred.pkl
    and label.pkl. The qrun SignalRecord stage runs before PortAnaRecord, so
    pred + label are guaranteed even when PortAnaRecord aborts (95-03 SUMMARY
    Deviation #4).

    Raises FileNotFoundError naming the missing pickle if no recorder has both.
    """
    mlruns = recorder_dir / "mlruns"
    if not mlruns.exists():
        raise FileNotFoundError(
            f"Wave 1 mlruns directory missing at {mlruns}; run scripts/run_qrun_basis_vol.py first (Plan 95-03)"
        )

    # Walk the mlruns tree explicitly so we don't depend on a working
    # qlib.workflow.R URI (which can hit Pitfall 1 / Postgres MLflow leak when
    # the tracking URI env var isn't pinned). pred.pkl + label.pkl are plain
    # pickles — load them directly.
    import pickle

    candidate_dirs = sorted(mlruns.glob("*/*/artifacts"))
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No recorder artifacts/ subdir under {mlruns} — "
            f"Wave 1 qrun has not run yet (no '<exp_id>/<run_id>/artifacts/')"
        )

    last_error: str | None = None
    for art_dir in reversed(candidate_dirs):  # newest first
        pred_path = art_dir / "pred.pkl"
        label_path = art_dir / "label.pkl"
        if pred_path.exists() and label_path.exists():
            with open(pred_path, "rb") as f:
                pred = pickle.load(f)
            with open(label_path, "rb") as f:
                label = pickle.load(f)
            logger.info("Loaded SignalRecord pickles from %s", art_dir)
            return {"pred": pred, "label": label, "artifacts_dir": art_dir}
        else:
            missing = []
            if not pred_path.exists():
                missing.append("pred.pkl")
            if not label_path.exists():
                missing.append("label.pkl")
            last_error = f"{art_dir} missing {missing}"

    raise FileNotFoundError(
        f"No mlruns recorder under {mlruns} has both pred.pkl AND label.pkl. Last checked: {last_error}"
    )


def _build_pred_label(pred, label):
    """Concatenate qrun pred.pkl (col=score) + label.pkl (col=label) into the
    pred_label DataFrame shape that qlib's score_ic_graph and
    model_performance_graph expect: MultiIndex(datetime, instrument), columns
    [score, label].
    """
    import pandas as pd

    # Both have MultiIndex(datetime, instrument). pred has 'score', label has
    # 'label'. Outer-align on the index so we don't drop trigger days that one
    # side might be missing.
    pred_df = pred.rename(columns={pred.columns[0]: "score"}) if hasattr(pred, "columns") else pred.to_frame("score")
    label_df = (
        label.rename(columns={label.columns[0]: "label"}) if hasattr(label, "columns") else label.to_frame("label")
    )
    pred_label = pd.concat([pred_df, label_df], axis=1)
    return pred_label


def _emit_graph_html(figs_iter, out_dir: Path, prefix: str) -> list[Path]:
    """Iterate plotly.graph_objs.Figure iterable; write each to <prefix>_<i>.html."""
    paths: list[Path] = []
    for i, fig in enumerate(figs_iter):
        p = out_dir / f"{prefix}_{i}.html"
        fig.write_html(str(p))
        paths.append(p)
    return paths


def run_portfolio_report(
    recorder_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Library entry — emit qlib graphical reports from the Wave 1 qrun pickles.

    Returns a summary dict with per-graph status (OK / SKIPPED / PARTIAL), HTML
    paths, and the GRAPH_NAME_LIST coverage. Tested via
    tests/test_portfolio_report.py::test_portfolio_report_smoke (Pattern S4).
    """
    # Lazy import — qlib + plotly only available inside the qlib-research
    # container. Module-level imports would break Mac-side `pytest --collect-only`
    # (Pitfall 2 carry-over from Plan 95-03).
    from qlib.contrib.report.analysis_model.analysis_model_performance import (
        model_performance_graph,
    )
    from qlib.contrib.report.analysis_position.score_ic import score_ic_graph

    recorder_dir = Path(recorder_dir or DEFAULT_RECORDER_DIR)
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = _load_signal_record(recorder_dir)
    pred_label = _build_pred_label(sig["pred"], sig["label"])
    logger.info(
        "pred_label shape=%s, n_instruments=%d, n_dates=%d",
        pred_label.shape,
        pred_label.index.get_level_values("instrument").nunique(),
        pred_label.index.get_level_values("datetime").nunique(),
    )

    graph_results: list[dict] = []

    # Graph 1: score_ic_graph (works on pred_label only, no PortAnaRecord)
    try:
        figs = list(score_ic_graph(pred_label, show_notebook=False))
        paths = _emit_graph_html(figs, out_dir, "score_ic")
        graph_results.append(
            {
                "graph": "analysis_position.score_ic_graph",
                "status": "OK",
                "n_html": len(paths),
                "paths": [str(p) for p in paths],
            }
        )
    except Exception as e:
        graph_results.append(
            {"graph": "analysis_position.score_ic_graph", "status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        )

    # Graph 2: model_performance_graph (works on pred_label only)
    # Use the default graph_names ['group_return', 'pred_ic', 'pred_autocorr'].
    # Single-instrument NaN expected per 95-04 SUMMARY (cross-sectional metrics
    # degenerate on N=1 panels) — graphs still render with NaN traces.
    try:
        figs = list(
            model_performance_graph(
                pred_label,
                show_notebook=False,
            )
        )
        paths = _emit_graph_html(figs, out_dir, "model_performance")
        graph_results.append(
            {
                "graph": "analysis_model.model_performance_graph",
                "status": "OK",
                "n_html": len(paths),
                "paths": [str(p) for p in paths],
            }
        )
    except Exception as e:
        graph_results.append(
            {
                "graph": "analysis_model.model_performance_graph",
                "status": "PARTIAL",
                "error": f"{type(e).__name__}: {e}",
            }
        )

    # Graphs 3-6: PortAnaRecord-dependent — SKIP with structured note.
    # Wave 1 (95-03 SUMMARY Deviation #4): PortAnaRecord pickles absent by
    # design because qlib's PortfolioMetrics.init_bench requires provider data
    # bypassed via Pitfall 1 (provider_uri=""). Skipping is documented behaviour.
    skip_reason = (
        "PortAnaRecord pickles absent — Wave 1 (95-03) deliberately bypasses "
        "qlib provider data via Pitfall 1, so PortfolioMetrics.init_bench "
        "cannot run. Skip is by design per 95-03 SUMMARY Deviation #4."
    )
    for graph_name in PORTANARECORD_DEPENDENT_GRAPHS:
        graph_results.append({"graph": graph_name, "status": "SKIPPED", "reason": skip_reason})

    n_ok = sum(1 for g in graph_results if g["status"] == "OK")
    n_skipped = sum(1 for g in graph_results if g["status"] == "SKIPPED")
    n_partial = sum(1 for g in graph_results if g["status"] in ("PARTIAL", "ERROR"))
    total_html = sum(g.get("n_html", 0) for g in graph_results)

    summary = {
        "out_dir": str(out_dir),
        "recorder_artifacts_dir": str(sig["artifacts_dir"]),
        "n_dates": int(pred_label.index.get_level_values("datetime").nunique()),
        "n_instruments": int(pred_label.index.get_level_values("instrument").nunique()),
        "graphs": graph_results,
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_partial": n_partial,
        "total_html": total_html,
        "graph_name_list_coverage": {
            "ok": [g["graph"] for g in graph_results if g["status"] == "OK"],
            "skipped": [g["graph"] for g in graph_results if g["status"] == "SKIPPED"],
            "partial": [g["graph"] for g in graph_results if g["status"] in ("PARTIAL", "ERROR")],
        },
    }
    logger.info(
        "ACTIVATE-05 emitted %d HTML reports across %d OK graphs (%d skipped, %d partial) to %s",
        total_html,
        n_ok,
        n_skipped,
        n_partial,
        out_dir,
    )
    return summary


def main() -> None:
    summary = run_portfolio_report()
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
