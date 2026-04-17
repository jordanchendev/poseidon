"""Tests for GridSearchOptimizer and BayesianOptimizer (BT-04).

Uses mocked BacktestRunner to avoid actual backtest execution.
Validates that:
- Grid search produces all parameter combinations via itertools.product
- Bayesian optimization runs exactly n_trials using Optuna TPE sampler
- Both return ranked results sorted by target metric descending
- Both accept strategy_factory callable and metric parameter
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from poseidon.backtest.optimizer import (
    BayesianOptimizer,
    GridSearchOptimizer,
    OptimizationTrial,
)
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.schemas import BacktestConfig, BacktestResult


def _make_mock_result(sharpe: float) -> BacktestResult:
    """Create a BacktestResult with controllable sharpe_ratio."""
    config = BacktestConfig(
        strategy_type="rule",
        symbol="TEST",
        market="crypto_spot",
        interval="1d",
        initial_capital=1_000_000.0,
    )
    return BacktestResult(
        config=config,
        metrics={
            "sharpe_ratio": sharpe,
            "total_return": sharpe * 0.1,
            "max_drawdown": 0.1,
        },
        trade_count=10,
        equity_curve_length=100,
        status="completed",
    )


@pytest.fixture()
def mock_components():
    """Create mock FeatureEngine, RiskEngine, CostModel."""
    feature_engine = MagicMock()
    risk_engine = MagicMock()
    cost_model = MagicMock()
    return feature_engine, risk_engine, cost_model


@pytest.fixture()
def sample_ohlcv():
    """Minimal OHLCV DataFrame for tests."""
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=10, freq="D"),
        "open": [100.0] * 10,
        "high": [105.0] * 10,
        "low": [95.0] * 10,
        "close": [100.0 + i for i in range(10)],
        "volume": [1000] * 10,
    })


class TestOptimizationTrial:
    """Tests for the OptimizationTrial dataclass."""

    def test_trial_has_required_fields(self):
        trial = OptimizationTrial(
            params={"a": 1, "b": 10},
            metrics={"sharpe_ratio": 1.5},
            metric_value=1.5,
        )
        assert trial.params == {"a": 1, "b": 10}
        assert trial.metrics == {"sharpe_ratio": 1.5}
        assert trial.metric_value == 1.5


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    def test_grid_search_produces_all_combinations(self, mock_components, sample_ohlcv):
        """Grid search with param_grid={"a": [1,2], "b": [10,20]} runs 4 trials."""
        fe, re, cm = mock_components
        optimizer = GridSearchOptimizer(fe, re, cm)

        strategy_factory = MagicMock()

        param_grid = {"a": [1, 2], "b": [10, 20]}

        # Mock BacktestRunner.run to return varied sharpe values based on params
        call_count = 0
        sharpe_values = [1.0, 2.0, 0.5, 1.5]

        def mock_run(ohlcv, feature_specs=None):
            nonlocal call_count
            result = _make_mock_result(sharpe_values[call_count])
            call_count += 1
            return result

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.side_effect = mock_run
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(strategy_factory, sample_ohlcv, param_grid)

        # Should produce all 4 combinations (2 x 2)
        assert len(results) == 4
        # strategy_factory called 4 times
        assert strategy_factory.call_count == 4

    def test_grid_search_results_sorted_descending(self, mock_components, sample_ohlcv):
        """Results are sorted by target metric descending."""
        fe, re, cm = mock_components
        optimizer = GridSearchOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_grid = {"x": [1, 2, 3]}

        call_count = 0
        sharpe_values = [0.5, 2.0, 1.0]

        def mock_run(ohlcv, feature_specs=None):
            nonlocal call_count
            result = _make_mock_result(sharpe_values[call_count])
            call_count += 1
            return result

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.side_effect = mock_run
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(strategy_factory, sample_ohlcv, param_grid)

        # Sorted descending by sharpe_ratio
        assert results[0].metric_value == 2.0
        assert results[1].metric_value == 1.0
        assert results[2].metric_value == 0.5

    def test_grid_search_results_contain_all_param_combos(self, mock_components, sample_ohlcv):
        """All parameter combinations appear in results."""
        fe, re, cm = mock_components
        optimizer = GridSearchOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_grid = {"a": [1, 2], "b": [10, 20]}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.0)
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(strategy_factory, sample_ohlcv, param_grid)

        all_params = [r.params for r in results]
        assert {"a": 1, "b": 10} in all_params
        assert {"a": 1, "b": 20} in all_params
        assert {"a": 2, "b": 10} in all_params
        assert {"a": 2, "b": 20} in all_params

    def test_grid_search_results_have_params_and_metrics(self, mock_components, sample_ohlcv):
        """Each result has params and metrics keys."""
        fe, re, cm = mock_components
        optimizer = GridSearchOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_grid = {"x": [1]}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.5)
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(strategy_factory, sample_ohlcv, param_grid)

        assert len(results) == 1
        trial = results[0]
        assert isinstance(trial, OptimizationTrial)
        assert "x" in trial.params
        assert "sharpe_ratio" in trial.metrics
        assert trial.metric_value == 1.5

    def test_grid_search_custom_metric(self, mock_components, sample_ohlcv):
        """Grid search can optimize by a custom metric."""
        fe, re, cm = mock_components
        optimizer = GridSearchOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_grid = {"x": [1, 2]}

        call_count = 0
        total_returns = [0.3, 0.1]

        def mock_run(ohlcv, feature_specs=None):
            nonlocal call_count
            result = _make_mock_result(1.0)
            result.metrics["total_return"] = total_returns[call_count]
            call_count += 1
            return result

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.side_effect = mock_run
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(
                strategy_factory, sample_ohlcv, param_grid, metric="total_return"
            )

        # Sorted by total_return descending
        assert results[0].metric_value == 0.3
        assert results[1].metric_value == 0.1


class TestBayesianOptimizer:
    """Tests for BayesianOptimizer."""

    def test_bayesian_runs_n_trials(self, mock_components, sample_ohlcv):
        """BayesianOptimizer with n_trials=5 runs exactly 5 trials."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_space = {"period": (5, 30, "int")}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.0)
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(
                strategy_factory, sample_ohlcv, param_space, n_trials=5
            )

        assert len(results) == 5
        assert strategy_factory.call_count == 5

    def test_bayesian_results_sorted_descending(self, mock_components, sample_ohlcv):
        """Results sorted by target metric descending."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_space = {"x": (1, 10, "int")}

        # Return different sharpe values each call
        call_count = 0

        def mock_run(ohlcv, feature_specs=None):
            nonlocal call_count
            result = _make_mock_result(float(call_count))
            call_count += 1
            return result

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.side_effect = mock_run
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(
                strategy_factory, sample_ohlcv, param_space, n_trials=5
            )

        # All results present
        assert len(results) == 5
        # Sorted descending
        metric_values = [r.metric_value for r in results]
        assert metric_values == sorted(metric_values, reverse=True)

    def test_bayesian_uses_tpe_sampler(self, mock_components, sample_ohlcv):
        """BayesianOptimizer uses optuna.samplers.TPESampler."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_space = {"x": (1, 10, "int")}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.0)
            MockRunner.return_value = mock_runner_instance

            with patch("poseidon.backtest.optimizer.optuna") as mock_optuna:
                # Set up the mock study
                mock_study = MagicMock()
                mock_trial = MagicMock()
                mock_trial.params = {"x": 5}
                mock_trial.value = 1.0
                mock_trial.user_attrs = {
                    "metrics": {"sharpe_ratio": 1.0, "total_return": 0.1},
                }
                mock_study.trials = [mock_trial]
                mock_optuna.create_study.return_value = mock_study

                results = optimizer.optimize(
                    strategy_factory, sample_ohlcv, param_space, n_trials=1
                )

                # Verify TPESampler was used
                mock_optuna.samplers.TPESampler.assert_called_once()
                # Verify create_study was called with maximize direction
                mock_optuna.create_study.assert_called_once()
                call_kwargs = mock_optuna.create_study.call_args
                assert call_kwargs.kwargs.get("direction") == "maximize" or \
                    (call_kwargs.args and call_kwargs.args[0] == "maximize") or \
                    call_kwargs[1].get("direction") == "maximize"

    def test_bayesian_accepts_float_params(self, mock_components, sample_ohlcv):
        """BayesianOptimizer handles float parameter types."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_space = {
            "period": (5, 30, "int"),
            "threshold": (0.1, 0.9, "float"),
        }

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.0)
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(
                strategy_factory, sample_ohlcv, param_space, n_trials=3
            )

        assert len(results) == 3
        # Each result should have both params
        for r in results:
            assert "period" in r.params
            assert "threshold" in r.params

    def test_bayesian_results_include_metrics(self, mock_components, sample_ohlcv):
        """Each result includes the full backtest metrics dict."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_space = {"x": (1, 10, "int")}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(2.0)
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(
                strategy_factory, sample_ohlcv, param_space, n_trials=2
            )

        for r in results:
            assert isinstance(r.metrics, dict)
            assert "sharpe_ratio" in r.metrics

    def test_bayesian_custom_metric(self, mock_components, sample_ohlcv):
        """BayesianOptimizer can optimize by a custom metric."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)

        strategy_factory = MagicMock()
        param_space = {"x": (1, 10, "int")}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.0)
            MockRunner.return_value = mock_runner_instance

            results = optimizer.optimize(
                strategy_factory, sample_ohlcv, param_space,
                n_trials=3, metric="total_return",
            )

        assert len(results) == 3
        # metric_value should correspond to total_return, not sharpe_ratio
        for r in results:
            assert r.metric_value == r.metrics["total_return"]


class TestBayesianOptimizerFillModel:
    """Tests for BayesianOptimizer fill_model passthrough (D-05, FIX-02)."""

    def test_fill_model_default_is_none(self, mock_components):
        """fill_model defaults to None when not provided."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm)
        assert optimizer.fill_model is None

    def test_fill_model_pessimistic_stored(self, mock_components):
        """fill_model=PESSIMISTIC is stored on optimizer."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm, fill_model=FillModel.PESSIMISTIC)
        assert optimizer.fill_model == FillModel.PESSIMISTIC

    def test_fill_model_passed_to_runner(self, mock_components, sample_ohlcv):
        """BayesianOptimizer passes fill_model to BacktestRunner."""
        fe, re, cm = mock_components
        optimizer = BayesianOptimizer(fe, re, cm, fill_model=FillModel.PESSIMISTIC)

        strategy_factory = MagicMock()
        param_space = {"x": (1, 10, "int")}

        with patch("poseidon.backtest.optimizer.BacktestRunner") as MockRunner:
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = _make_mock_result(1.0)
            MockRunner.return_value = mock_runner_instance

            optimizer.optimize(
                strategy_factory, sample_ohlcv, param_space, n_trials=1,
            )

            # Verify BacktestRunner was constructed with fill_model=PESSIMISTIC
            call_kwargs = MockRunner.call_args
            assert call_kwargs.kwargs.get("fill_model") == FillModel.PESSIMISTIC
