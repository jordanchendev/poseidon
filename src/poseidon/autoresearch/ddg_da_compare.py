"""Phase 92 — DDG-DA comparison orchestration library.

Exports:
    extract_per_fold_sharpe(rolling_exp_name, *, recorders_iter=None)
        Pattern B from RESEARCH 575-610; B-1 fix injection of recorder iter.
    _default_recorders_iter(experiment_name)
        Internal — deferred-qlib-import boundary for tests.
    write_comparison_summary(with_df, without_df, out_path)
        D-20 parquet. Aligns with/without by fold_start; emits delta_sharpe.
    paired_bootstrap_delta_sharpe(...)
        Canonical (W-5); tests/test_ddg_da_stats.py imports from here.
    run_comparison(thesis_name, model_class, segments, run_dir, smoke=False)
        End-to-end orchestration entry point.

All qlib imports DEFERRED to function bodies (PATTERNS.md §Deferred qlib
import). Module top must remain qlib-free so cp313 containers
(api/cpu-worker/gpu-worker) can `import` this module without crashing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _default_recorders_iter(experiment_name: str):
    """Yield (rec_id, recorder) pairs from qlib mlflow for the given experiment.

    Deferred-qlib-import boundary (PATTERNS.md Pattern §1 / RESEARCH Pitfall 2):
    all `from qlib.workflow import R` lives ONLY here, not at module top, not
    bound at module level. Tests inject a fake iterator instead of patching
    a non-existent module-level `R` symbol (B-1 fix per checker review).
    """
    from qlib.workflow import R  # deferred — lives ONLY here

    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders(rtype=exp.RT_L)
    yield from recorders.items()


def extract_per_fold_sharpe(
    rolling_exp_name: str,
    threshold: float = 1.0,
    round_trip_cost: float = 0.00032,
    bars_per_year: int = 240,
    recorders_iter=None,
) -> pd.DataFrame:
    """Walk qlib mlflow experiment, compute per-fold OOS Sharpe.

    Pattern B from 92-RESEARCH.md §Code Examples lines 575-610.

    Args:
        rolling_exp_name: qlib mlflow experiment name. Used only when
            ``recorders_iter`` is None (default).
        threshold: trigger threshold; abs(pred) > threshold engages a position.
        round_trip_cost: cost per engaged-day (32bps default per v10.0
            standing rule and Phase 95 ACTIVATE-01 baseline).
        bars_per_year: annualisation factor for Sharpe.
        recorders_iter: optional iterator yielding ``(rec_id, recorder)``
            pairs. Default = None → falls back to ``_default_recorders_iter``
            which performs the deferred qlib import. Tests pass a synthetic
            iterator directly to keep the module qlib-free on Mac (B-1 fix).

    Returns:
        DataFrame with columns
        ``[recorder_id, fold_start, fold_end, n_engaged, sharpe, cum]``,
        sorted by ``fold_start`` ascending.
    """
    if recorders_iter is None:
        recorders_iter = _default_recorders_iter(rolling_exp_name)

    rows: list[dict[str, Any]] = []
    for rec_id, rec in recorders_iter:
        pred = rec.load_object("pred.pkl")
        # Coerce DataFrame predictions (qlib SignalRecord persists a 1-col DF)
        # to a Series. If "score" column exists we use that; otherwise the
        # first column.
        if isinstance(pred, pd.DataFrame):
            pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
        label = rec.load_object("label.pkl")
        if isinstance(label, pd.DataFrame):
            label = label.iloc[:, 0]

        position = pd.Series(0.0, index=pred.index)
        position[pred > threshold] = -1.0
        position[pred < -threshold] = 1.0
        engaged = position != 0
        net = position * label - engaged.astype(float) * round_trip_cost
        std = float(net.std())
        sh = float(net.mean() / std * np.sqrt(bars_per_year)) if std > 0 and not np.isnan(std) else 0.0

        # Resolve fold start/end timestamps. qlib's pred MultiIndex has a
        # 'datetime' level; if not, fall back to the index itself.
        try:
            dt_values = pred.index.get_level_values("datetime")
        except (KeyError, AttributeError):
            dt_values = pred.index
        fold_start = pd.Timestamp(dt_values.min())
        fold_end = pd.Timestamp(dt_values.max())

        rows.append(
            {
                "recorder_id": str(rec_id),
                "fold_start": fold_start,
                "fold_end": fold_end,
                "n_engaged": int(engaged.sum()),
                "sharpe": sh,
                "cum": float((1 + net).prod() - 1),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_start").reset_index(drop=True)


def write_comparison_summary(with_df: pd.DataFrame, without_df: pd.DataFrame, out_path: Path) -> Path:
    """Align with/without by fold_start; write parquet with delta_sharpe.

    D-20: comparison_summary.parquet shape per RESEARCH §Persistence content.

    Args:
        with_df: per-fold Sharpe DataFrame from the with-DDG-DA leg.
        without_df: per-fold Sharpe DataFrame from the without-DDG-DA leg.
        out_path: parquet sink path.

    Returns:
        ``out_path`` (Path) — the file is written before return.

    Raises:
        ValueError if fold counts mismatch or the inner-join collapses rows
        (unique fold_start required).
    """
    if len(with_df) != len(without_df):
        raise ValueError(
            f"fold-count mismatch: with={len(with_df)} without={len(without_df)} "
            f"(RESEARCH OQ-2 / Pitfall 7 — single-flight worker should "
            f"produce equal counts)"
        )
    merged = pd.merge(
        with_df[["fold_start", "fold_end", "sharpe", "n_engaged"]].rename(
            columns={"sharpe": "sharpe_with", "n_engaged": "n_engaged_with"}
        ),
        without_df[["fold_start", "sharpe", "n_engaged"]].rename(
            columns={"sharpe": "sharpe_without", "n_engaged": "n_engaged_without"}
        ),
        on="fold_start",
        how="inner",
    )
    if len(merged) != len(with_df):
        raise ValueError(
            f"fold_start alignment failed: merged={len(merged)} "
            f"expected={len(with_df)} — fold_start values must match across legs"
        )
    merged["delta_sharpe"] = merged["sharpe_with"] - merged["sharpe_without"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    return out_path


# CANONICAL source of paired_bootstrap_delta_sharpe (W-5 fix per checker review).
# tests/test_ddg_da_stats.py imports this symbol — single source of truth, no
# drift. Plan 92-01 shipped a stub copy in the test file; Plan 92-03 swaps the
# test-file copy for an import from this module so behaviour stays in lockstep.
def paired_bootstrap_delta_sharpe(
    sharpe_with: np.ndarray,
    sharpe_without: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap on per-window ΔSharpe.

    RESEARCH §Code Examples Pattern C lines 614-643. Pure numpy — no qlib
    dependency, fully Mac-runnable.

    Args:
        sharpe_with: array of per-fold Sharpe ratios from the with-DDG-DA leg.
        sharpe_without: array of per-fold Sharpe ratios from the baseline leg.
            Must be the same length as ``sharpe_with`` (paired by fold).
        n_resamples: number of bootstrap resamples (default 1000).
        seed: RNG seed.

    Returns:
        dict with keys
        ``{mean_delta_sharpe, std_delta_sharpe, ci95_low, ci95_high,
            p_positive, windows_improved_pct}``.
    """
    assert len(sharpe_with) == len(sharpe_without), "fold counts must match"
    rng = np.random.default_rng(seed)
    deltas = np.asarray(sharpe_with) - np.asarray(sharpe_without)
    n = len(deltas)
    if n == 0:
        raise ValueError("paired_bootstrap_delta_sharpe: empty input arrays")
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = deltas[idx].mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_delta_sharpe": float(deltas.mean()),
        "std_delta_sharpe": float(deltas.std()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_positive": float((boot_means > 0).mean()),
        "windows_improved_pct": float((deltas > 0).mean() * 100),
    }


def run_comparison(
    thesis_name: str,
    model_class: str,
    segments: dict[str, tuple[str, str]],
    run_dir: Path,
    smoke: bool = False,
) -> dict:
    """End-to-end comparison: with-DDG-DA vs without-DDG-DA on the same dataset.

    D-20 layout:
        run_dir/
          comparison_metadata.json
          with_ddg_da/{predictions.parquet, per_window_sharpe.parquet, ic.json}
          without_ddg_da/{...}
          comparison_summary.parquet
          run.lock (file-lock per D-23)
          summary.json

    Plan 92-04 writes ``verdict.md`` AFTER this function returns.

    Args:
        thesis_name: e.g. "tx_gap_intraday" (D-09 default; only thesis
            supported in v1).
        model_class: e.g. "LGBModel" (D-11 default).
        segments: walk-forward train/valid/test segment dict
            ``{"train": (start, end), "valid": (...), "test": (...)}``.
        run_dir: artifact root (D-19).
        smoke: when True, only the last 2 walk-forward folds are kept (D-25).

    Returns:
        dict with keys ``run_dir``, ``comparison_summary_parquet``,
        ``bootstrap_result``, ``n_folds``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # File-lock (D-23 simplification of Phase 90/91 durable-row lifecycle).
    lock_path = run_dir / "run.lock"
    if lock_path.exists():
        raise RuntimeError(f"run already in progress: {lock_path}")
    lock_path.write_text(json.dumps({"started_at": pd.Timestamp.now().isoformat()}))

    try:
        # Deferred imports — keep module top qlib-free.
        from poseidon.autoresearch.ddg_da import PoseidonDDGDA

        with_dir = run_dir / "with_ddg_da"
        without_dir = run_dir / "without_ddg_da"
        with_dir.mkdir(parents=True, exist_ok=True)
        without_dir.mkdir(parents=True, exist_ok=True)

        # WITH DDG-DA leg
        # Plan 92-04.1 BUG-2 fix: handler_class="Alpha158" → qlib.contrib.data.handler.Alpha158
        # (qlib stock handler accepts the start_time/end_time/fit_start_time/fit_end_time/
        # instruments/label kwargs that qlib's Rolling driver injects). The previous
        # "Alpha158Handler" entry maps to PoseidonDataHandler whose __init__ rejects those
        # kwargs (TypeError: unexpected keyword argument 'end_time'). Plan 95 qrun work
        # continues to use "Alpha158Handler" → PoseidonDataHandler for backward compat.
        # Plan 92-04.1 BUG-1 fix: explicit instruments + provider_uri at call site for
        # audit clarity (defaults already point at TX / poseidon_tw_futures).
        wrapper = PoseidonDDGDA(
            working_dir=with_dir,
            handler_class="Alpha158",
            model_class=model_class,
            market="tw_futures",
            interval="1d",
            segments=segments,
            instruments="TX",
            provider_uri="/root/.qlib/qlib_data/poseidon_tw_futures",
        )
        with_result = wrapper.run()

        # WITHOUT DDG-DA leg — direct qlib Rolling using the same YAML.
        from qlib.contrib.rolling.base import Rolling

        baseline_yaml = wrapper._yaml_path  # reuse the same YAML emission
        baseline = Rolling(conf_path=baseline_yaml, horizon=20, step=20)
        baseline.run()
        without_result = {
            "yaml_path": str(baseline_yaml),
            "working_dir": str(without_dir),
            "rolling_exp_name": getattr(baseline, "rolling_exp", None),
        }

        # Extract per-fold Sharpe from each leg.
        with_sharpe = extract_per_fold_sharpe(with_result["rolling_exp_name"])
        without_sharpe = extract_per_fold_sharpe(without_result["rolling_exp_name"])
        if smoke:
            with_sharpe = with_sharpe.tail(2).reset_index(drop=True)
            without_sharpe = without_sharpe.tail(2).reset_index(drop=True)
        with_sharpe.to_parquet(with_dir / "per_window_sharpe.parquet")
        without_sharpe.to_parquet(without_dir / "per_window_sharpe.parquet")

        summary_path = write_comparison_summary(with_sharpe, without_sharpe, run_dir / "comparison_summary.parquet")

        bootstrap_result = paired_bootstrap_delta_sharpe(
            with_sharpe["sharpe"].to_numpy(),
            without_sharpe["sharpe"].to_numpy(),
            n_resamples=1000,
            seed=42,
        )

        metadata = {
            "thesis_name": thesis_name,
            "model_class": model_class,
            "segments": {k: list(v) for k, v in segments.items()},
            "smoke": smoke,
            "n_folds": len(with_sharpe),
            "with_ddg_da": with_result,
            "without_ddg_da": without_result,
        }
        (run_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "n_folds": len(with_sharpe),
                    "bootstrap": bootstrap_result,
                    "comparison_summary_parquet": str(summary_path),
                },
                indent=2,
            )
        )

        return {
            "run_dir": str(run_dir),
            "comparison_summary_parquet": str(summary_path),
            "bootstrap_result": bootstrap_result,
            "n_folds": len(with_sharpe),
        }
    finally:
        if lock_path.exists():
            lock_path.unlink()


__all__ = [
    "_default_recorders_iter",
    "extract_per_fold_sharpe",
    "paired_bootstrap_delta_sharpe",
    "run_comparison",
    "write_comparison_summary",
]
