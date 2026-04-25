"""Tests for FeatureOrchestrator delegation to FeatureComputer.

Covers:
- FEAT-01: FeatureOrchestrator delegates compute_from_df to FeatureComputer (not reimplemented)
- FEAT-02: FeatureOrchestrator uses DataRepository for data loading
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame(
        {
            "time": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestFeatureOrchestratorDelegation:
    """Verify FeatureOrchestrator delegates compute to FeatureComputer."""

    def test_compute_from_df_delegates_to_computer(self, sample_ohlcv):
        """FeatureOrchestrator.compute_from_df must call FeatureComputer.compute_from_df."""
        from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator

        orch = FeatureOrchestrator()
        with patch.object(orch._computer, "compute_from_df", wraps=orch._computer.compute_from_df) as mock_compute:
            result = orch.compute_from_df(sample_ohlcv)
            mock_compute.assert_called_once()
            assert not result.empty

    def test_orchestrator_uses_data_repository(self):
        """FeatureOrchestrator accepts DataRepository via constructor injection."""
        from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator

        mock_repo = MagicMock()
        orch = FeatureOrchestrator(repo=mock_repo)
        assert orch._repo is mock_repo
        assert orch._computer is not None

    def test_orchestrator_has_same_api_as_feature_engine(self):
        """FeatureOrchestrator exposes compute, compute_from_df, compute_with_companions."""
        from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator

        assert hasattr(FeatureOrchestrator, "compute")
        assert hasattr(FeatureOrchestrator, "compute_from_df")
        assert hasattr(FeatureOrchestrator, "compute_with_companions")
        assert callable(FeatureOrchestrator.compute)
        assert callable(FeatureOrchestrator.compute_from_df)
        assert callable(FeatureOrchestrator.compute_with_companions)

    def test_orchestrator_default_repo_is_none(self):
        """Default construction has no repo (lazy creation in compute())."""
        from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator

        orch = FeatureOrchestrator()
        assert orch._repo is None

    def test_compute_from_df_result_matches_computer(self, sample_ohlcv):
        """FeatureOrchestrator.compute_from_df returns identical result to FeatureComputer."""
        from poseidon.data.feature_engine.computer import FeatureComputer
        from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator

        specs = [("sma", {"period": 5}), ("rsi", {"period": 14})]
        orch = FeatureOrchestrator()
        computer = FeatureComputer()
        result_orch = orch.compute_from_df(sample_ohlcv, feature_specs=specs)
        result_computer = computer.compute_from_df(sample_ohlcv, feature_specs=specs)
        pd.testing.assert_frame_equal(result_orch, result_computer)

    def test_feature_engine_alias_is_orchestrator(self):
        """FeatureEngine import resolves to FeatureOrchestrator."""
        from poseidon.data.feature_engine import FeatureEngine
        from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator

        assert FeatureEngine is FeatureOrchestrator
