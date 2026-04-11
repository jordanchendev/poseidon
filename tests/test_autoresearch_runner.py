"""Integration tests for AutoResearchRunner and autoresearch_run Celery task.

Tests D-09 (orchestration), D-11 (heartbeat), D-12 (graceful stop),
D-13 (per-market failure isolation), D-14 (10 consecutive experiments).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from poseidon.autoresearch.guard import _AUTORESEARCH_ACTIVE
from poseidon.autoresearch.runner import AutoResearchRunner, MarketResult, MarketSpec
from poseidon.backtest.param_search import SearchConfig, SearchResult


def _make_search_result(study_name: str = "test_study") -> SearchResult:
    """Create a SearchResult fixture."""
    return SearchResult(
        study_name=study_name,
        total_trials=10,
        passed_trials=5,
        rejected_trials=5,
        best_config={"rsi_period": 14, "threshold": 0.5},
        best_composite_score=0.85,
        holdout_boundary=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


def _make_ohlcv() -> pd.DataFrame:
    """Create a small OHLCV DataFrame fixture."""
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(100),
            "high": range(1, 101),
            "low": range(100),
            "close": range(100),
            "volume": [1000] * 100,
        },
        index=idx,
    )


@patch("poseidon.autoresearch.runner.read_ohlcv")
@patch("poseidon.autoresearch.runner.ParameterSearchPipeline")
@patch("poseidon.autoresearch.runner.ModelManager")
@patch("poseidon.autoresearch.runner.FeatureEngine")
@patch("poseidon.autoresearch.runner.RiskEngine")
class TestAutoResearchRunner:
    """Integration tests for AutoResearchRunner."""

    def test_full_cycle_single_market(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """AutoResearchRunner runs a full cycle for a single market."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        search_result = _make_search_result("crypto_spot_BTCUSDT_1h")
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = search_result
        mock_pipeline_cls.return_value = mock_pipeline
        mock_model_manager.return_value.list_ready_models.return_value = []

        mock_session = MagicMock()
        runner = AutoResearchRunner(
            db_session=mock_session,
            search_config=SearchConfig(n_trials=5),
        )
        markets = [MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h")]
        results = runner.run(markets)

        assert len(results) == 1
        assert results[0].search_result is not None
        assert results[0].search_result.study_name == "crypto_spot_BTCUSDT_1h"
        assert results[0].error is None
        mock_pipeline.run.assert_called_once()

    def test_graceful_stop(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """AutoResearchRunner stops after graceful stop is requested (D-12)."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_search_result()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_model_manager.return_value.list_ready_models.return_value = []

        call_count = 0

        def stop_after_first() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 1

        mock_session = MagicMock()
        runner = AutoResearchRunner(
            db_session=mock_session,
            search_config=SearchConfig(n_trials=5),
            stop_check=stop_after_first,
        )
        markets = [
            MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h"),
            MarketSpec(symbol="ETHUSDT", market="crypto_spot", interval="1h"),
            MarketSpec(symbol="SOLUSDT", market="crypto_spot", interval="1h"),
        ]
        results = runner.run(markets)

        # Should only process 1 market before stop triggers
        assert len(results) == 1

    def test_per_market_failure_isolation(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """Per-market failure does not abort entire run (D-13)."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_model_manager.return_value.list_ready_models.return_value = []

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Market 1 failed")
            return _make_search_result(f"study_{call_count}")

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = side_effect
        mock_pipeline_cls.return_value = mock_pipeline

        mock_session = MagicMock()
        runner = AutoResearchRunner(
            db_session=mock_session,
            search_config=SearchConfig(n_trials=5),
        )
        markets = [
            MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h"),
            MarketSpec(symbol="ETHUSDT", market="crypto_spot", interval="1h"),
        ]
        results = runner.run(markets)

        assert len(results) == 2
        assert results[0].error is not None  # First market failed
        assert results[1].search_result is not None  # Second market succeeded

    def test_progress_callback_called(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """Progress callback receives (current, total, symbol) (D-11)."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_search_result()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_model_manager.return_value.list_ready_models.return_value = []

        progress_calls = []

        def on_progress(current: int, total: int, symbol: str):
            progress_calls.append((current, total, symbol))

        mock_session = MagicMock()
        runner = AutoResearchRunner(
            db_session=mock_session,
            search_config=SearchConfig(n_trials=5),
            progress_callback=on_progress,
        )
        markets = [
            MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h"),
            MarketSpec(symbol="ETHUSDT", market="crypto_spot", interval="1h"),
        ]
        results = runner.run(markets)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (0, 2, "BTCUSDT")
        assert progress_calls[1] == (1, 2, "ETHUSDT")

    def test_autoresearch_guard_active_during_run(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """_AUTORESEARCH_ACTIVE is True inside pipeline.run (D-05)."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_model_manager.return_value.list_ready_models.return_value = []

        guard_values = []

        def capture_guard(*args, **kwargs):
            guard_values.append(_AUTORESEARCH_ACTIVE.get(False))
            return _make_search_result()

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = capture_guard
        mock_pipeline_cls.return_value = mock_pipeline

        mock_session = MagicMock()
        runner = AutoResearchRunner(
            db_session=mock_session,
            search_config=SearchConfig(n_trials=5),
        )
        markets = [MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h")]
        runner.run(markets)

        assert guard_values == [True]

    def test_10_consecutive(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """10 consecutive experiments run unattended without error (D-14, AUTO-05)."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_model_manager.return_value.list_ready_models.return_value = []

        counter = 0

        def make_result(*args, **kwargs):
            nonlocal counter
            counter += 1
            return _make_search_result(f"study_{counter}")

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = make_result
        mock_pipeline_cls.return_value = mock_pipeline

        mock_session = MagicMock()
        runner = AutoResearchRunner(
            db_session=mock_session,
            search_config=SearchConfig(n_trials=5),
        )
        markets = [
            MarketSpec(symbol=f"SYM{i}", market="crypto_spot", interval="1h")
            for i in range(10)
        ]
        results = runner.run(markets)

        assert len(results) == 10
        for r in results:
            assert r.search_result is not None
            assert r.error is None

    def test_discovers_models_per_market_when_not_forced(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """Runner discovers ready models and passes them into the search pipeline."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_search_result()
        mock_pipeline_cls.return_value = mock_pipeline
        models = [MagicMock(id=uuid.uuid4(), name="qlib_lgbm_crypto_spot", version=3)]
        mock_model_manager.return_value.list_ready_models.return_value = models

        runner = AutoResearchRunner(
            db_session=MagicMock(),
            search_config=SearchConfig(n_trials=5),
        )
        markets = [MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h")]
        runner.run(markets)

        mock_model_manager.return_value.list_ready_models.assert_called_once_with("crypto_spot")
        assert mock_pipeline.run.call_args.kwargs["available_models"] == models

    def test_skips_model_discovery_for_forced_override(
        self, mock_risk, mock_feature, mock_model_manager, mock_pipeline_cls, mock_read_ohlcv
    ):
        """Forced model_version_id keeps the legacy override path and skips discovery."""
        mock_read_ohlcv.return_value = _make_ohlcv()
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_search_result()
        mock_pipeline_cls.return_value = mock_pipeline

        runner = AutoResearchRunner(
            db_session=MagicMock(),
            search_config=SearchConfig(n_trials=5),
            model_version_id=uuid.uuid4(),
        )
        markets = [MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h")]
        runner.run(markets)

        mock_model_manager.return_value.list_ready_models.assert_not_called()
        assert mock_pipeline.run.call_args.kwargs["available_models"] is None


class TestCeleryTaskRegistered:
    """Verify autoresearch_run task is registered in Celery."""

    def test_celery_task_registered(self):
        """autoresearch_run is in celery_app.tasks."""
        from poseidon.workers.celery_app import celery_app

        # Force task module import to trigger registration
        import poseidon.workers.cpu_tasks  # noqa: F401

        assert "poseidon.workers.cpu_tasks.autoresearch_run" in celery_app.tasks
