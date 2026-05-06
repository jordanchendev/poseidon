"""Phase 92 Plan 92-03 — per-fold Sharpe extractor unit tests.

Validates Pattern B from 92-RESEARCH.md §Code Examples lines 575-610 —
extractor walks qlib mlflow recorders and computes per-fold OOS Sharpe.

B-1 fix (per checker review): tests inject a synthetic ``recorders_iter``
directly instead of ``@patch``-ing a non-existent module-level ``R``.
The qlib import lives inside ``_default_recorders_iter`` (deferred-import
discipline), so there is no ``R`` at ``ddg_da_compare`` module scope to
patch. Dependency injection keeps tests qlib-free on Mac.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd


def _make_synthetic_recorder(seed: int = 0, n_days: int = 200, mean_pred: float = 0.0) -> MagicMock:
    """Build a mocked qlib Recorder with pred.pkl + label.pkl loaders.

    Returns a MagicMock whose ``load_object("pred.pkl")`` returns a
    MultiIndex-(datetime, instrument) Series of N(mean_pred, 1) values, and
    ``load_object("label.pkl")`` returns a Series of correlated labels
    (corr ~0.1 with pred plus mostly noise).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    mi = pd.MultiIndex.from_product([dates, ["TX"]], names=["datetime", "instrument"])
    pred_vals = rng.normal(mean_pred, 1.0, len(mi))
    label_vals = 0.1 * pred_vals + 0.99 * rng.normal(0, 1, len(mi))
    pred = pd.Series(pred_vals, index=mi, name="score")
    label = pd.Series(label_vals, index=mi, name="label")
    rec = MagicMock()
    rec.load_object.side_effect = lambda key: {
        "pred.pkl": pred,
        "label.pkl": label,
    }[key]
    return rec


def test_per_fold_sharpe_shape():
    """DDGDA-02: extractor returns DataFrame with required columns.

    B-1 fix: inject ``recorders_iter`` directly. The qlib import lives
    inside ``_default_recorders_iter`` (deferred-import discipline) so
    there is no ``R`` at ``ddg_da_compare`` module scope to patch.
    Dependency injection keeps tests qlib-free on Mac.
    """
    from poseidon.autoresearch.ddg_da_compare import extract_per_fold_sharpe

    rec1 = _make_synthetic_recorder(0)
    rec2 = _make_synthetic_recorder(1)
    fake_iter = iter([("r1", rec1), ("r2", rec2)])
    df = extract_per_fold_sharpe("test_exp", threshold=1.0, recorders_iter=fake_iter)
    assert set(df.columns) == {
        "recorder_id",
        "fold_start",
        "fold_end",
        "n_engaged",
        "sharpe",
        "cum",
    }
    assert len(df) == 2
    assert df["fold_start"].is_monotonic_increasing


def test_per_fold_sharpe_aligns_with_threshold_logic():
    """DDGDA-02: with threshold=1.0 on N(0,1) preds, engagement freq ≈31% (|z|>1).

    Same dependency-injection pattern as test_per_fold_sharpe_shape.
    """
    from poseidon.autoresearch.ddg_da_compare import extract_per_fold_sharpe

    rec = _make_synthetic_recorder(0, n_days=400)
    fake_iter = iter([("r1", rec)])
    df = extract_per_fold_sharpe("test_exp", threshold=1.0, recorders_iter=fake_iter)
    # |z|>1 fraction for N(0,1) is ~31.7%; expect at least 80 engaged days
    # out of 400 (well below the analytical mean to give RNG headroom).
    assert df.iloc[0]["n_engaged"] > 80
    # Recorder id must round-trip as string.
    assert df.iloc[0]["recorder_id"] == "r1"


def test_per_fold_sharpe_threshold_zero_engages_all():
    """DDGDA-02: with threshold=0, every non-zero prediction engages.

    Sanity check on the trigger logic: at threshold 0, only days with
    pred==0 are unengaged (probability 0 for continuous N(0,1)).
    """
    from poseidon.autoresearch.ddg_da_compare import extract_per_fold_sharpe

    rec = _make_synthetic_recorder(0, n_days=100)
    fake_iter = iter([("r1", rec)])
    df = extract_per_fold_sharpe("test_exp", threshold=0.0, recorders_iter=fake_iter)
    assert df.iloc[0]["n_engaged"] == 100
