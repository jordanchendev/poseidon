"""Tests for run_prediction GPU task and signal conversion."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest


def _make_mock_signal(action: str, confidence: float) -> MagicMock:
    sig = MagicMock()
    sig.id = uuid4()
    sig.action = action
    sig.confidence = confidence
    return sig


@patch("poseidon.core.database.SessionLocal")
@patch("poseidon.ml.manager.ModelManager")
@patch("poseidon.ml.registry.get_model")
@patch("poseidon.data.feature_engine.FeatureEngine")
@patch("poseidon.strategies.model_strategy.ModelStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline")
@patch("poseidon.core.config.settings")
def test_run_prediction_loads_model_and_generates_signals(
    mock_settings,
    mock_pipeline_cls,
    mock_strategy_cls,
    mock_engine_cls,
    mock_get_model,
    mock_manager_cls,
    mock_session_local,
):
    """run_prediction loads model, evaluates, filters by threshold, pipes through pipeline."""
    # Setup
    mock_settings.predict_confidence_threshold = 0.6

    session = MagicMock()
    mock_session_local.return_value = session

    # ModelManager.get_version returns fake ModelVersion
    mv = SimpleNamespace(
        name="xgboost", artifact_path="/tmp/model", status="active", version=1
    )
    manager = MagicMock()
    manager.get_version.return_value = mv
    mock_manager_cls.return_value = manager

    # get_model returns a model class whose .load() returns a model instance
    mock_model_instance = MagicMock()
    mock_model_cls = MagicMock()
    mock_model_cls.load.return_value = mock_model_instance
    mock_get_model.return_value = mock_model_cls

    # FeatureEngine.compute returns sample DataFrame
    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100, 101, 102]})
    mock_engine_cls.return_value = mock_engine

    # ModelStrategy.evaluate returns 2 signals: one above threshold, one below
    sig_high = _make_mock_signal("long", 0.7)
    sig_low = _make_mock_signal("short", 0.3)
    mock_strategy = MagicMock()
    mock_strategy.evaluate.return_value = [sig_high, sig_low]
    mock_strategy_cls.return_value = mock_strategy

    # SignalPipeline.process returns signal as-is
    mock_pipeline = MagicMock()
    mock_pipeline.process.side_effect = lambda s: s
    mock_pipeline_cls.return_value = mock_pipeline

    # Import and call (deferred imports are patched at module level)
    from poseidon.workers.gpu_tasks import run_prediction

    version_id = str(uuid4())
    result = run_prediction(None, version_id, "BTCUSDT", "crypto_spot", "1d")

    # Only the high-confidence signal passes threshold 0.6
    assert result["signals_generated"] == 1
    assert result["version_id"] == version_id
    assert result["symbol"] == "BTCUSDT"
    assert len(result["results"]) == 1
    assert result["results"][0]["confidence"] == 0.7

    # Pipeline.process called only once (for the passing signal)
    mock_pipeline.process.assert_called_once_with(sig_high)


@patch("poseidon.core.database.SessionLocal")
@patch("poseidon.ml.manager.ModelManager")
def test_run_prediction_model_not_found(mock_manager_cls, mock_session_local):
    """run_prediction raises ValueError when model version not found."""
    session = MagicMock()
    mock_session_local.return_value = session

    manager = MagicMock()
    manager.get_version.return_value = None
    mock_manager_cls.return_value = manager

    from poseidon.workers.gpu_tasks import run_prediction

    with pytest.raises(ValueError, match="not found"):
        run_prediction(None, str(uuid4()), "BTCUSDT", "crypto_spot")


@patch("poseidon.workers.gpu_tasks.torch", create=True)
def test_gpu_health_probe_with_torch(mock_torch):
    """gpu_health_probe reports torch/CUDA info when torch is available."""
    mock_torch.__version__ = "2.1.0"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "RTX 4070 Ti SUPER"
    mock_torch.cuda.mem_get_info.return_value = (8_000_000_000, 16_000_000_000)

    from poseidon.workers.gpu_tasks import gpu_health_probe

    result = gpu_health_probe()

    assert result["torch_version"] == mock_torch.__version__
    assert result["cuda_available"] is True
    assert "cuda_device" in result
