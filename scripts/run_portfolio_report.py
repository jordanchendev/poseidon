#!/usr/bin/env python3
"""Emit qlib graphical reports from anchor signal qrun output.

Reuses the upstream qrun SignalRecord pickles emitted at::

    local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol/mlruns/

Reads the six Qlib report graph inputs from the *same fresh qrun recorder*.
PortAnaRecord requires a real provider calendar and benchmark series; a run
without its three portfolio pickles is invalid and is reported as an error.

Run on stormtrooper::

    docker compose exec -T qlib-research uv run python scripts/run_portfolio_report.py

The pytest smoke (``tests/test_portfolio_report.py``) imports
``run_portfolio_report`` and invokes it programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Bind-mounted at /app/local_dev → host poseidon/local_dev/.
# The upstream qrun writes recorder pickles under
# {DEFAULT_RECORDER_DIR}/mlruns/{exp_id}/{run_id}/artifacts/.
DEFAULT_RECORDER_DIR = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")
DEFAULT_OUT_DIR = Path("/app/local_dev/backtests/phase95_basis_vol/reports")
EXPERIMENT_NAME = "phase95_tx_basis_vol"


def _latest_artifacts_dir(recorder_dir: Path) -> Path:
    """Find the newest complete recorder for interactive CLI use only."""
    mlruns = recorder_dir / "mlruns"
    candidates = sorted(mlruns.glob("*/*/artifacts"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        if (candidate / "pred.pkl").exists() and (candidate / "label.pkl").exists():
            return candidate
    raise FileNotFoundError(f"no qrun recorder artifacts under {mlruns}")


def _load_record(artifacts_dir: Path) -> dict:
    """Load the complete SignalRecord and PortAnaRecord artifact set.

    ``artifacts_dir`` is supplied by ``run_qrun_basis_vol`` in smoke paths,
    which prevents an old successful recorder from satisfying a later run.
    """
    import pickle

    paths = {
        "pred": artifacts_dir / "pred.pkl",
        "label": artifacts_dir / "label.pkl",
        "position": artifacts_dir / "portfolio_analysis/positions_normal_1day.pkl",
        "report_normal": artifacts_dir / "portfolio_analysis/report_normal_1day.pkl",
        "analysis": artifacts_dir / "portfolio_analysis/port_analysis_1day.pkl",
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete PortAnaRecord artifacts at {artifacts_dir}: {missing}")
    with_paths = {name: pickle.loads(path.read_bytes()) for name, path in paths.items()}
    with_paths["artifacts_dir"] = artifacts_dir
    logger.info("Loaded complete qrun recorder from %s", artifacts_dir)
    return with_paths


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


def _label_for_position_graphs(label):
    """Adapt SignalRecord labels to PortAna's (instrument, datetime) slices."""
    import pandas as pd

    if not isinstance(label.index, pd.MultiIndex) or set(label.index.names) != {"datetime", "instrument"}:
        raise ValueError("PortAna label data requires MultiIndex(datetime, instrument)")
    frame = label.reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.set_index(["instrument", "datetime"]).sort_index()


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
    artifacts_dir: Path | None = None,
) -> dict:
    """Library entry — emit qlib graphical reports from the qrun pickles.

    Returns a summary dict with per-graph status (OK / SKIPPED / PARTIAL), HTML
    paths, and the GRAPH_NAME_LIST coverage. Tested via
    tests/test_portfolio_report.py::test_portfolio_report_smoke.
    """
    # Lazy import — qlib + plotly only available inside the qlib-research
    # container. Module-level imports would break Mac-side
    # `pytest --collect-only`.
    from qlib.contrib.report.analysis_model.analysis_model_performance import (
        model_performance_graph,
    )
    from qlib.contrib.report.analysis_position import (
        cumulative_return_graph,
        rank_label_graph,
        report_graph,
        risk_analysis_graph,
    )
    from qlib.contrib.report.analysis_position.score_ic import score_ic_graph

    recorder_dir = Path(recorder_dir or DEFAULT_RECORDER_DIR)
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else _latest_artifacts_dir(recorder_dir)
    sig = _load_record(artifacts_dir)
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

    # Graph 2: model_performance_graph (works on pred_label only).
    # Per 95-04 SUMMARY: single-instrument anchor (TX-only) makes cross-sectional
    # metrics structurally degenerate. _group_return raises "df is empty" on N=1
    # panels because `len(x)//N=0` produces empty quintile slices and qlib's
    # ScatterGraph rejects empty dataframes (verified on stormtrooper).
    # Workaround: drop 'group_return' from graph_names when n_instruments < 2.
    n_instruments = pred_label.index.get_level_values("instrument").nunique()
    if n_instruments < 2:
        graph_results.append(
            {
                "graph": "analysis_model.model_performance_graph",
                "status": "NOT_APPLICABLE",
                "reason": f"single-instrument panel (n={n_instruments}) has no cross-sectional IC or rank autocorrelation",
            }
        )
    else:
        mp_graph_names = ["group_return", "pred_ic", "pred_autocorr"]
        try:
            figs = list(
                model_performance_graph(
                    pred_label,
                    graph_names=mp_graph_names,
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
                    "graph_names": mp_graph_names,
                }
            )
        except Exception as e:
            graph_results.append(
                {
                    "graph": "analysis_model.model_performance_graph",
                    "status": "PARTIAL",
                    "error": f"{type(e).__name__}: {e}",
                    "graph_names": mp_graph_names,
                }
            )

    portana_graphs = (
        (
            "analysis_position.cumulative_return_graph",
            "cumulative_return",
            cumulative_return_graph,
            {
                "position": sig["position"],
                "report_normal": sig["report_normal"],
                "label_data": _label_for_position_graphs(sig["label"]),
            },
        ),
        (
            "analysis_position.risk_analysis_graph",
            "risk_analysis",
            risk_analysis_graph,
            {
                "analysis_df": sig["analysis"],
                "report_normal_df": sig["report_normal"],
            },
        ),
        ("analysis_position.report_graph", "report", report_graph, {"report_df": sig["report_normal"]}),
        (
            "analysis_position.rank_label_graph",
            "rank_label",
            rank_label_graph,
            {
                "position": sig["position"],
                "label_data": _label_for_position_graphs(sig["label"]),
            },
        ),
    )
    for graph_name, prefix, graph_fn, kwargs in portana_graphs:
        try:
            paths = _emit_graph_html(graph_fn(show_notebook=False, **kwargs), out_dir, prefix)
            graph_results.append(
                {"graph": graph_name, "status": "OK", "n_html": len(paths), "paths": [str(p) for p in paths]}
            )
        except Exception as exc:
            graph_results.append({"graph": graph_name, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})

    n_ok = sum(1 for g in graph_results if g["status"] == "OK")
    n_skipped = sum(1 for g in graph_results if g["status"] == "SKIPPED")
    n_partial = sum(1 for g in graph_results if g["status"] in ("PARTIAL", "ERROR"))
    n_not_applicable = sum(1 for g in graph_results if g["status"] == "NOT_APPLICABLE")
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
        "n_not_applicable": n_not_applicable,
        "total_html": total_html,
        "graph_name_list_coverage": {
            "ok": [g["graph"] for g in graph_results if g["status"] == "OK"],
            "skipped": [g["graph"] for g in graph_results if g["status"] == "SKIPPED"],
            "partial": [g["graph"] for g in graph_results if g["status"] in ("PARTIAL", "ERROR")],
            "not_applicable": [g["graph"] for g in graph_results if g["status"] == "NOT_APPLICABLE"],
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
