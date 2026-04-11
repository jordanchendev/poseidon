"""Unit coverage for Phase 47 factor-analysis research helpers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from poseidon.research.centrality_analysis import (
    compute_centrality,
    replay_sub_signal_votes,
)
from poseidon.research.ic_analysis import compute_forward_returns, compute_rank_ic
from poseidon.research.shapley_analysis import compute_shapley_values


def test_compute_forward_returns_respects_explicit_empty_horizons():
    close = pd.Series([10.0, 11.0, 12.0], dtype=float)

    returns = compute_forward_returns(close, [])

    assert returns == {}


def test_compute_forward_returns_uses_requested_horizons():
    close = pd.Series([10.0, 12.0, 18.0], dtype=float)

    returns = compute_forward_returns(close, [1])

    pd.testing.assert_series_equal(
        returns[1],
        pd.Series([0.2, 0.5, np.nan], dtype=float),
    )


def test_compute_rank_ic_returns_none_for_too_few_shared_observations():
    feature_values = pd.Series(np.arange(10, dtype=float))
    forward_returns = pd.Series(np.linspace(0.0, 1.0, 10))
    features_df = pd.DataFrame({"feature_a": feature_values})

    result = compute_rank_ic(features_df, forward_returns, ["feature_a"])

    assert result == {"feature_a": None}


def test_compute_shapley_values_uses_positive_class_and_sampling(monkeypatch):
    class FakeExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, frame):
            zeros = np.zeros((len(frame), len(frame.columns)))
            ones = np.ones((len(frame), len(frame.columns)))
            return [zeros, ones]

    fake_shap = types.SimpleNamespace(TreeExplainer=FakeExplainer)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
        }
    )

    result = compute_shapley_values(object(), frame, max_samples=2)

    assert result["num_samples"] == 2
    assert result["features"] == {"feature_a": 1.0, "feature_b": 1.0}


def test_replay_sub_signal_votes_coerces_failures_to_nan(monkeypatch):
    def fake_evaluate_condition(signal, features, row_idx):
        if row_idx == 1:
            raise ValueError("boom")
        return bool(features.iloc[row_idx]["flag"])

    monkeypatch.setattr(
        "poseidon.strategies.dsl.executor.evaluate_condition",
        fake_evaluate_condition,
    )
    features = pd.DataFrame({"flag": [1, 0, 1]}, index=["a", "b", "c"])

    votes = replay_sub_signal_votes([{"label": "signal_a"}], features)

    assert votes.loc["a", "signal_a"] == 1.0
    assert np.isnan(votes.loc["b", "signal_a"])
    assert votes.loc["c", "signal_a"] == 1.0


def test_compute_centrality_handles_single_signal_without_linkage():
    vote_matrix = pd.DataFrame({"signal_a": [1.0, 0.0, 1.0]})

    result = compute_centrality(vote_matrix)

    assert result["signal_names"] == ["signal_a"]
    assert result["clusters"] == {"1": ["signal_a"]}
    assert result["linkage"] == []
