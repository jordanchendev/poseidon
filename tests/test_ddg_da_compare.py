"""Phase 92 Plan 92-03 — comparison_summary.parquet writer unit tests.

D-20 mandates per-fold rows aligned with/without/Δ. Sample-based tests on
synthetic with/without arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _mk_sharpe_df(starts: list[str], sharpes: list[float], n_engaged: list[int]) -> pd.DataFrame:
    """Helper: build a per-fold Sharpe DataFrame matching the extractor schema."""
    return pd.DataFrame(
        {
            "recorder_id": [f"r{i}" for i in range(len(starts))],
            "fold_start": pd.to_datetime(starts),
            "fold_end": pd.to_datetime(starts) + pd.Timedelta(days=20),
            "n_engaged": n_engaged,
            "sharpe": sharpes,
            "cum": [s * 0.01 for s in sharpes],
        }
    )


def test_summary_alignment(tmp_path: Path):
    """DDGDA-02: comparison_summary writes one row per fold with delta_sharpe column."""
    from poseidon.autoresearch.ddg_da_compare import write_comparison_summary

    with_df = _mk_sharpe_df(["2024-01-01", "2024-02-01", "2024-03-01"], [0.5, 0.7, 0.3], [10, 12, 8])
    without_df = _mk_sharpe_df(["2024-01-01", "2024-02-01", "2024-03-01"], [0.2, 0.4, 0.1], [10, 12, 8])
    out_path = tmp_path / "comparison_summary.parquet"
    result_path = write_comparison_summary(with_df, without_df, out_path)
    assert result_path.exists()
    df = pd.read_parquet(out_path)
    assert {
        "fold_start",
        "sharpe_with",
        "sharpe_without",
        "delta_sharpe",
        "n_engaged_with",
        "n_engaged_without",
    }.issubset(df.columns)
    assert len(df) == 3
    np.testing.assert_array_almost_equal(df["delta_sharpe"].to_numpy(), [0.3, 0.3, 0.2])


def test_summary_handles_unequal_fold_counts(tmp_path: Path):
    """DDGDA-02: mismatched fold counts (RESEARCH OQ-2 edge case) raise ValueError."""
    from poseidon.autoresearch.ddg_da_compare import write_comparison_summary

    with_df = _mk_sharpe_df(["2024-01-01", "2024-02-01"], [0.5, 0.7], [10, 12])
    without_df = _mk_sharpe_df(["2024-01-01", "2024-02-01", "2024-03-01"], [0.2, 0.4, 0.1], [10, 12, 8])
    with pytest.raises(ValueError, match="fold-count mismatch"):
        write_comparison_summary(with_df, without_df, tmp_path / "x.parquet")


def test_summary_alignment_failure_when_fold_starts_disjoint(tmp_path: Path):
    """DDGDA-02: matched counts but disjoint fold_start values raise on alignment."""
    from poseidon.autoresearch.ddg_da_compare import write_comparison_summary

    with_df = _mk_sharpe_df(["2024-01-01", "2024-02-01"], [0.5, 0.7], [10, 12])
    without_df = _mk_sharpe_df(["2024-04-01", "2024-05-01"], [0.2, 0.4], [10, 12])
    with pytest.raises(ValueError, match="alignment failed"):
        write_comparison_summary(with_df, without_df, tmp_path / "x.parquet")


def test_bootstrap_imports_from_compare_module():
    """W-5 fix: library is the single source of truth for paired bootstrap.

    The numeric body lives only in ``poseidon.autoresearch.ddg_da_compare``;
    ``tests/test_ddg_da_stats.py`` is a thin contract test against this
    same imported symbol.
    """
    from poseidon.autoresearch.ddg_da_compare import paired_bootstrap_delta_sharpe

    rng = np.random.default_rng(42)
    result = paired_bootstrap_delta_sharpe(
        rng.normal(0.5, 0.3, 20),
        rng.normal(0.0, 0.3, 20),
        n_resamples=200,
        seed=42,
    )
    assert "mean_delta_sharpe" in result
    assert result["ci95_low"] <= result["mean_delta_sharpe"] <= result["ci95_high"]
    assert result["mean_delta_sharpe"] > 0  # detectable mean shift


def test_run_comparison_sets_mlflow_tracking_uri_under_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Plan 92-04.2 BUG-6 fix verification: ``run_comparison`` sets
    ``MLFLOW_TRACKING_URI`` to a per-run directory under ``run_dir``
    BEFORE invoking any qlib code, ensuring mlruns state isolation
    between iterations.

    This test runs Mac-side without qlib installed: we mock
    ``PoseidonDDGDA`` so the function fails downstream of the env-var
    write but AFTER the env var has been set. The assertion is on the
    final state of ``os.environ['MLFLOW_TRACKING_URI']``.
    """
    import contextlib
    import os
    from unittest.mock import patch

    run_dir = tmp_path / "run-uuid-92-04-2-test"
    # Ensure no leftover state from a previous test in this process.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Patch PoseidonDDGDA at the symbol used inside run_comparison's deferred
    # import (poseidon.autoresearch.ddg_da.PoseidonDDGDA). When run_comparison
    # tries to instantiate / run it, raise — by that point the env var is set.
    with patch("poseidon.autoresearch.ddg_da.PoseidonDDGDA") as mock_ddgda:
        mock_ddgda.side_effect = RuntimeError("test sentinel — bypass qlib path")
        from poseidon.autoresearch.ddg_da_compare import run_comparison

        # Expected — the env var is set BEFORE the deferred imports run, so
        # the env-var assertion below is reachable regardless of whether
        # qlib is installed (ImportError) or the mock fires (RuntimeError).
        with contextlib.suppress(RuntimeError, ImportError):
            run_comparison(
                thesis_name="tx_gap_intraday",
                model_class="LGBModel",
                segments={
                    "train": ("2021-03-22", "2023-08-31"),
                    "valid": ("2023-09-01", "2023-12-31"),
                    "test": ("2024-01-01", "2026-04-30"),
                },
                run_dir=run_dir,
                smoke=True,
            )

    # BUG-6 fix invariant: the env var must point under run_dir and use
    # the file:// scheme.
    assert "MLFLOW_TRACKING_URI" in os.environ, (
        "run_comparison must set MLFLOW_TRACKING_URI before any qlib import (BUG-6 fix)"
    )
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    assert tracking_uri.startswith("file://"), f"MLFLOW_TRACKING_URI must use file:// scheme; got {tracking_uri!r}"
    # run_dir is resolve()'d inside run_comparison, so compare against
    # the same resolved form.
    expected_mlruns = run_dir.resolve() / "mlruns"
    assert str(expected_mlruns) in tracking_uri, (
        f"MLFLOW_TRACKING_URI={tracking_uri!r} should contain run_dir/mlruns={expected_mlruns!r}"
    )
    # Per-run mlruns dir must be created on the filesystem.
    assert expected_mlruns.is_dir(), f"run_comparison must create run_dir/mlruns; missing: {expected_mlruns}"
