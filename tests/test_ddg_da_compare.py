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
