"""Tests for PatchTST Transformer model — Phase 9.

Tests cover: PatchTST forward pass shapes, TimeSeriesDataset sliding windows,
TransformerModel BaseModel contract (train/predict/validate/save/load),
registry integration, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from poseidon.ml.base import BaseModel
from poseidon.ml.registry import get_model, list_models
from poseidon.ml.implementations.transformer_model import (
    TransformerModel,
    LABEL_MAP,
    DEFAULT_FEATURES,
    _HAS_TORCH,
)

# Conditionally import torch-only symbols
if _HAS_TORCH:
    import torch

    from poseidon.ml.implementations.transformer_model import (
        PatchTST,
        TimeSeriesDataset,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features():
    """200-row DataFrame with all 20 DEFAULT_FEATURES + close column."""
    np.random.seed(42)
    n = 200
    data = {col: np.random.randn(n) for col in DEFAULT_FEATURES}
    data["close"] = 100.0 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(data)


@pytest.fixture
def sample_targets():
    """200-row target Series with labels 0, 1, 2."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1, 2], 200))


@pytest.fixture
def small_params():
    """Small hyperparams for fast test execution (~seconds, not minutes)."""
    return {
        "lookback_window": 20,
        "patch_length": 8,
        "stride": 4,
        "d_model": 16,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
        "epochs": 3,
        "batch_size": 16,
        "patience": 5,
    }


@pytest.fixture
def trained_model(sample_features, sample_targets, small_params):
    """Pre-trained TransformerModel for tests that need a ready model."""
    model = TransformerModel()
    model.train(sample_features, sample_targets, small_params)
    return model


# ---------------------------------------------------------------------------
# PatchTST Architecture Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestPatchTST:
    def test_forward_shape(self):
        """PatchTST forward with [2, 60, 20] input produces [2, 3] output."""
        model = PatchTST(
            n_features=20, lookback_window=60, patch_length=16, stride=8
        )
        x = torch.randn(2, 60, 20)
        out = model(x)
        assert out.shape == (2, 3)

    def test_forward_small(self):
        """PatchTST with smaller config still produces [batch, 3]."""
        model = PatchTST(
            n_features=5,
            lookback_window=30,
            patch_length=8,
            stride=4,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
        )
        x = torch.randn(4, 30, 5)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_is_logits(self):
        """Output should be raw logits (not softmaxed) -- some values < 0 or > 1."""
        model = PatchTST(
            n_features=10, lookback_window=30, patch_length=8, stride=4,
            d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        )
        torch.manual_seed(0)
        x = torch.randn(8, 30, 10)
        out = model(x)
        # Raw logits should have at least some values outside [0, 1]
        assert out.min().item() < 0 or out.max().item() > 1.0

    def test_num_parameters(self):
        """Model should have trainable parameters."""
        model = PatchTST(
            n_features=10, lookback_window=30, patch_length=8, stride=4,
            d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0


# ---------------------------------------------------------------------------
# TimeSeriesDataset Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestTimeSeriesDataset:
    def test_length(self):
        """Dataset length == n_samples - lookback_window."""
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.choice([0, 1, 2], 100).astype(np.int64)
        ds = TimeSeriesDataset(features, targets, lookback_window=20)
        assert len(ds) == 80

    def test_getitem_shapes(self):
        """__getitem__ returns (x: [lookback, n_feat], y: scalar)."""
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.choice([0, 1, 2], 100).astype(np.int64)
        ds = TimeSeriesDataset(features, targets, lookback_window=20)
        x, y = ds[0]
        assert x.shape == (20, 10)
        assert y.shape == ()  # scalar

    def test_getitem_dtype(self):
        """x should be float32, y should be int64."""
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.choice([0, 1, 2], 100).astype(np.int64)
        ds = TimeSeriesDataset(features, targets, lookback_window=20)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_normalization(self):
        """Normalized features should have approximately zero mean."""
        np.random.seed(99)
        features = np.random.randn(200, 5).astype(np.float32) * 10.0 + 50.0
        targets = np.random.choice([0, 1, 2], 200).astype(np.int64)
        ds = TimeSeriesDataset(features, targets, lookback_window=20)
        # Check that the internally stored features have mean close to 0
        mean_abs = np.abs(ds._features.mean(axis=0)).max()
        assert mean_abs < 0.5, f"Mean too far from 0: {mean_abs}"


# ---------------------------------------------------------------------------
# TransformerModel BaseModel Contract Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestTransformerModelContract:
    def test_is_base_model_subclass(self):
        assert issubclass(TransformerModel, BaseModel)

    def test_name(self):
        assert TransformerModel.name == "transformer"

    def test_train_returns_metrics(self, sample_features, sample_targets, small_params):
        model = TransformerModel()
        metrics = model.train(sample_features, sample_targets, small_params)
        assert isinstance(metrics, dict)
        for key in ("accuracy", "n_samples", "n_features", "epochs_trained", "final_loss"):
            assert key in metrics, f"Missing key: {key}"
        assert metrics["n_samples"] > 0
        assert metrics["epochs_trained"] > 0

    def test_predict_returns_dataframe(
        self, trained_model, sample_features
    ):
        result = trained_model.predict(sample_features)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert "confidence" in result.columns
        assert len(result) == len(sample_features)

    def test_predict_values_valid(self, trained_model, sample_features):
        result = trained_model.predict(sample_features)
        valid_labels = {"hold", "long", "short"}
        assert set(result["prediction"].unique()).issubset(valid_labels)
        assert (result["confidence"] >= 0.0).all()
        assert (result["confidence"] <= 1.0).all()

    def test_predict_untrained_raises(self):
        model = TransformerModel()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(pd.DataFrame({"sma_5": [1.0, 2.0]}))

    def test_validate_returns_metrics(
        self, trained_model, sample_features, sample_targets
    ):
        metrics = trained_model.validate(sample_features, sample_targets)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "n_samples" in metrics

    def test_save_creates_artifacts(
        self, tmp_path, trained_model
    ):
        trained_model.save(tmp_path / "model")
        model_dir = tmp_path / "model"
        assert (model_dir / "model.pt").exists()
        assert (model_dir / "features.json").exists()
        assert (model_dir / "metadata.json").exists()

    def test_save_load_roundtrip(
        self, tmp_path, trained_model, sample_features
    ):
        trained_model.save(tmp_path / "model")
        loaded = TransformerModel.load(tmp_path / "model")
        result = loaded.predict(sample_features)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert "confidence" in result.columns
        assert len(result) == len(sample_features)

    def test_get_default_params(self):
        model = TransformerModel()
        params = model.get_default_params()
        assert isinstance(params, dict)
        for key in ("lookback_window", "d_model", "nhead", "num_layers", "epochs", "lr"):
            assert key in params, f"Missing default param: {key}"

    def test_get_feature_list(self):
        model = TransformerModel()
        features = model.get_feature_list()
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, str) for f in features)


# ---------------------------------------------------------------------------
# Registry Integration Tests
# ---------------------------------------------------------------------------


class TestTransformerRegistry:
    def test_transformer_in_list_models(self):
        assert "transformer" in list_models()

    def test_get_model_returns_class(self):
        assert get_model("transformer") is TransformerModel
