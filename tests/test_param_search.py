"""Tests for ParameterSearchPipeline -- holdout enforcement, WFE gate, trial cap, study naming.

Uses mocks for heavy dependencies (FeatureEngine, RiskEngine, BacktestRunner,
WalkForwardAnalyzer) to keep tests fast and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from poseidon.backtest.holdout import HoldoutConfig
from poseidon.backtest.param_search import (
    ParameterSearchPipeline,
    SearchConfig,
)


def _make_ohlcv(n_rows: int = 100) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with datetime index."""
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n_rows)]
    return pd.DataFrame(
        {
            "open": [100.0] * n_rows,
            "high": [105.0] * n_rows,
            "low": [95.0] * n_rows,
            "close": [102.0] * n_rows,
            "volume": [1000.0] * n_rows,
        },
        index=pd.DatetimeIndex(dates),
    )


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_trial_count_cap(self):
        """n_trials is capped to max_trials via __post_init__."""
        cfg = SearchConfig(n_trials=5, max_trials=3)
        assert cfg.n_trials == 3

    def test_trial_count_no_cap(self):
        """n_trials stays unchanged if <= max_trials."""
        cfg = SearchConfig(n_trials=50, max_trials=100)
        assert cfg.n_trials == 50


class TestParameterSearchPipeline:
    """Tests for ParameterSearchPipeline orchestration."""

    def _make_pipeline(self):
        """Create pipeline with mocked dependencies."""
        fe = MagicMock()
        re = MagicMock()
        cm = MagicMock()
        tracker = MagicMock()
        tracker.save.return_value = MagicMock()  # returns uuid
        pipeline = ParameterSearchPipeline(
            feature_engine=fe,
            risk_engine=re,
            cost_model=cm,
            tracker=tracker,
        )
        return pipeline, tracker

    def _base_trial_params(self) -> dict:
        return {
            "qlib_model_enabled": 0,
            "qlib_prediction_threshold": 0.5,
        }

    def test_holdout_enforcement(self):
        """Pipeline trims OHLCV to train portion before optimization."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)

        # Default holdout is 20%, so boundary is at index 80
        holdout = HoldoutConfig(holdout_pct=0.20)
        boundary = holdout.compute_boundary(ohlcv)

        # Train portion should be rows 0..79 (80 rows)
        train = ohlcv[ohlcv.index < boundary]
        assert len(train) == 80

        # Mock optimizer to capture the ohlcv passed to it
        captured_ohlcv = {}

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            captured_ohlcv["df"] = ohlcv
            return []  # no trials

        with patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt:
            MockOpt.return_value.optimize = mock_optimize
            cfg = SearchConfig(n_trials=2, storage_url=None)
            pipeline.run(ohlcv, "BTCUSDT", "crypto_spot", "1d", config=cfg)

        # The optimizer should have received only 80 rows (train portion)
        assert len(captured_ohlcv["df"]) == 80

    def test_study_name_per_market(self):
        """Study name follows {market}_{symbol}_{interval} format."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)

        with patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt:
            MockOpt.return_value.optimize.return_value = []
            cfg = SearchConfig(n_trials=2, storage_url=None)
            result = pipeline.run(ohlcv, "BTCUSDT", "crypto_spot", "1d", config=cfg)

        assert result.study_name == "crypto_spot_BTCUSDT_1d"

    def test_storage_url_schema_append(self):
        """Optuna schema is appended to storage URL when not present."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)

        captured_storage = {}

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            captured_storage["url"] = storage
            return []

        with patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt:
            MockOpt.return_value.optimize = mock_optimize
            cfg = SearchConfig(
                n_trials=2,
                storage_url="postgresql://localhost/poseidon",
            )
            pipeline.run(ohlcv, "BTCUSDT", "crypto_spot", "1d", config=cfg)

        assert "search_path=optuna" in captured_storage["url"]

    def test_storage_url_schema_not_duplicated(self):
        """Optuna schema is NOT appended if already present in URL."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)

        captured_storage = {}

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            captured_storage["url"] = storage
            return []

        with patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt:
            MockOpt.return_value.optimize = mock_optimize
            cfg = SearchConfig(
                n_trials=2,
                storage_url="postgresql://localhost/poseidon?options=-csearch_path=optuna",
            )
            pipeline.run(ohlcv, "BTCUSDT", "crypto_spot", "1d", config=cfg)

        # Should not double-append
        assert captured_storage["url"].count("search_path") == 1

    def test_wfe_gate_rejects_low_wfe(self):
        """Trials with WFE < 0.50 are marked rejected in tracker."""
        pipeline, tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)

        # Mock optimizer to return 1 trial
        from poseidon.backtest.optimizer import OptimizationTrial

        mock_trial = OptimizationTrial(
            params={
                "rsi_period": 10,
                "ema_short": 5,
                "ema_long": 20,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bollinger_period": 20,
                "momentum_short": 5,
                "momentum_long": 10,
                "min_votes": 4,
                "atr_multiplier": 2.0,
                "position_pct": 0.08,
                "bear_min_votes": 4,
                "bear_position_pct": 0.06,
                "cooldown_bars": 12,
                "conviction_gap": 2,
                "qlib_prediction_threshold": 0.5,
                "qlib_model_enabled": 0,
            },
            metrics={"sharpe_ratio": 1.5, "trade_count": 60, "max_drawdown": 0.1, "total_return": 0.2},
            metric_value=0.8,
        )

        # Mock WFE to return low value
        mock_wf_result = MagicMock()
        mock_wf_result.wfe = 0.3
        mock_wf_result.passed = False

        with (
            patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt,
            patch("poseidon.backtest.param_search.WalkForwardAnalyzer") as MockWF,
        ):
            MockOpt.return_value.optimize.return_value = [mock_trial]
            MockWF.return_value.analyze.return_value = mock_wf_result
            cfg = SearchConfig(n_trials=1, storage_url=None)
            result = pipeline.run(ohlcv, "BTCUSDT", "crypto_spot", "1d", config=cfg)

        # The trial should be rejected
        assert result.rejected_trials == 1
        assert result.passed_trials == 0
        # tracker.save should have been called with status="rejected"
        tracker.save.assert_called_once()
        call_kwargs = tracker.save.call_args[1]
        assert call_kwargs["status"] == "rejected"

    def test_wfe_gate_passes_high_wfe(self):
        """Trials with WFE >= 0.50 are marked passed in tracker."""
        pipeline, tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)

        from poseidon.backtest.optimizer import OptimizationTrial

        mock_trial = OptimizationTrial(
            params={
                "rsi_period": 10,
                "ema_short": 5,
                "ema_long": 20,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bollinger_period": 20,
                "momentum_short": 5,
                "momentum_long": 10,
                "min_votes": 4,
                "atr_multiplier": 2.0,
                "position_pct": 0.08,
                "bear_min_votes": 4,
                "bear_position_pct": 0.06,
                "cooldown_bars": 12,
                "conviction_gap": 2,
                "qlib_prediction_threshold": 0.5,
                "qlib_model_enabled": 0,
            },
            metrics={"sharpe_ratio": 1.5, "trade_count": 60, "max_drawdown": 0.1, "total_return": 0.2},
            metric_value=0.8,
        )

        mock_wf_result = MagicMock()
        mock_wf_result.wfe = 0.6
        mock_wf_result.passed = True

        with (
            patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt,
            patch("poseidon.backtest.param_search.WalkForwardAnalyzer") as MockWF,
        ):
            MockOpt.return_value.optimize.return_value = [mock_trial]
            MockWF.return_value.analyze.return_value = mock_wf_result
            cfg = SearchConfig(n_trials=1, storage_url=None)
            result = pipeline.run(ohlcv, "BTCUSDT", "crypto_spot", "1d", config=cfg)

        assert result.passed_trials == 1
        assert result.rejected_trials == 0
        call_kwargs = tracker.save.call_args[1]
        assert call_kwargs["status"] == "passed"

    def test_dynamic_ml_params_with_models(self):
        """available_models injects ML toggle and version bounds into param_space."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)
        models = [
            SimpleNamespace(id="model-1"),
            SimpleNamespace(id="model-2"),
            SimpleNamespace(id="model-3"),
        ]
        captured_param_space = {}

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            captured_param_space.update(param_space)
            return []

        with patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt:
            MockOpt.return_value.optimize = mock_optimize
            pipeline.run(
                ohlcv,
                "BTCUSDT",
                "crypto_spot",
                "1d",
                config=SearchConfig(n_trials=2, storage_url=None),
                available_models=models,
            )

        assert captured_param_space["qlib_model_enabled"] == (0, 1, "int")
        assert captured_param_space["qlib_model_version"] == (0, 2, "int")

    def test_dynamic_ml_params_no_models(self):
        """No available models removes ML params from the search space."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)
        captured_param_space = {}

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            captured_param_space.update(param_space)
            return []

        with patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt:
            MockOpt.return_value.optimize = mock_optimize
            pipeline.run(
                ohlcv,
                "BTCUSDT",
                "crypto_spot",
                "1d",
                config=SearchConfig(n_trials=2, storage_url=None),
                available_models=[],
            )

        assert "qlib_model_enabled" not in captured_param_space
        assert "qlib_model_version" not in captured_param_space

    def test_trial_factory_resolves_model_id_when_ml_enabled(self):
        """Trial factory maps qlib_model_version index to the concrete model id."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)
        models = [
            SimpleNamespace(id="model-1"),
            SimpleNamespace(id="model-2"),
        ]

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            strategy_factory(
                {
                    **self._base_trial_params(),
                    "qlib_model_enabled": 1,
                    "qlib_model_version": 1,
                }
            )
            return []

        with (
            patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt,
            patch("poseidon.backtest.param_search._build_config_from_params") as mock_build,
            patch("poseidon.backtest.param_search.VotingStrategyFactory.from_config") as mock_from_config,
        ):
            MockOpt.return_value.optimize = mock_optimize
            mock_build.return_value = {"name": "test"}
            mock_from_config.return_value = MagicMock()
            pipeline.run(
                ohlcv,
                "BTCUSDT",
                "crypto_spot",
                "1d",
                config=SearchConfig(n_trials=2, storage_url=None),
                available_models=models,
            )

        assert mock_build.call_args.kwargs["model_version_id"] == "model-2"

    def test_trial_factory_returns_none_when_ml_disabled(self):
        """Trial factory leaves model_version_id unset for pure-TA trials."""
        pipeline, _tracker = self._make_pipeline()
        ohlcv = _make_ohlcv(100)
        models = [SimpleNamespace(id="model-1")]

        def mock_optimize(strategy_factory, ohlcv, param_space, n_trials, metric, seed, storage, study_name):
            strategy_factory(self._base_trial_params())
            return []

        with (
            patch("poseidon.backtest.param_search.BayesianOptimizer") as MockOpt,
            patch("poseidon.backtest.param_search._build_config_from_params") as mock_build,
            patch("poseidon.backtest.param_search.VotingStrategyFactory.from_config") as mock_from_config,
        ):
            MockOpt.return_value.optimize = mock_optimize
            mock_build.return_value = {"name": "test"}
            mock_from_config.return_value = MagicMock()
            pipeline.run(
                ohlcv,
                "BTCUSDT",
                "crypto_spot",
                "1d",
                config=SearchConfig(n_trials=2, storage_url=None),
                available_models=models,
            )

        assert mock_build.call_args.kwargs["model_version_id"] is None
