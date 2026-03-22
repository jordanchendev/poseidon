"""Parameter optimization for backtest strategies.

Provides two optimization approaches:
1. GridSearchOptimizer -- exhaustive search for small parameter spaces
   using itertools.product for all combinations.
2. BayesianOptimizer -- Optuna TPE sampler for larger parameter spaces
   with configurable n_trials.

Both optimizers return ranked OptimizationTrial lists sorted by the
target metric (descending). Each trial runs a full backtest via
BacktestRunner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product as itertools_product
from typing import Any, Callable

import optuna
import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.portfolio import SizingConfig
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationTrial:
    """Result of a single optimization trial.

    Attributes:
        params: Parameter values used for this trial.
        metrics: Full backtest metrics dictionary.
        metric_value: The target metric value used for ranking.
    """

    params: dict
    metrics: dict
    metric_value: float


class GridSearchOptimizer:
    """Exhaustive parameter search for small parameter spaces.

    Uses itertools.product to generate all parameter combinations.
    Each combination runs a full backtest via BacktestRunner.
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
    ) -> None:
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()

    def optimize(
        self,
        strategy_factory: Callable[[dict], BaseStrategy],
        ohlcv: pd.DataFrame,
        param_grid: dict[str, list],
        metric: str = "sharpe_ratio",
    ) -> list[OptimizationTrial]:
        """Run exhaustive grid search over all parameter combinations.

        Args:
            strategy_factory: Callable that takes a params dict and returns
                a configured BaseStrategy instance.
            ohlcv: Historical OHLCV DataFrame.
            param_grid: Dict mapping parameter names to lists of values,
                e.g. {"period": [10, 20, 30], "threshold": [0.5, 0.7]}.
            metric: Metric name from BacktestResult.metrics to optimize.
                Default: "sharpe_ratio".

        Returns:
            List of OptimizationTrial sorted by metric_value descending.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        trials: list[OptimizationTrial] = []

        total_combos = 1
        for v in values:
            total_combos *= len(v)
        logger.info("Grid search: %d combinations", total_combos)

        for combo in itertools_product(*values):
            params = dict(zip(keys, combo))
            strategy = strategy_factory(params)

            runner = BacktestRunner(
                strategy=strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
            )
            result = runner.run(ohlcv)

            metric_value = result.metrics.get(metric, 0.0)
            trials.append(
                OptimizationTrial(
                    params=params,
                    metrics=result.metrics,
                    metric_value=metric_value,
                )
            )

        # Sort by metric_value descending (best first)
        trials.sort(key=lambda t: t.metric_value, reverse=True)
        return trials


class BayesianOptimizer:
    """Bayesian optimization using Optuna TPE sampler for larger parameter spaces.

    Wraps optuna.create_study with TPESampler. Each trial suggests parameter
    values, runs a backtest, and returns the target metric value.
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
    ) -> None:
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()

    def optimize(
        self,
        strategy_factory: Callable[[dict], BaseStrategy],
        ohlcv: pd.DataFrame,
        param_space: dict[str, tuple[Any, Any, str]],
        n_trials: int = 100,
        metric: str = "sharpe_ratio",
        seed: int = 42,
    ) -> list[OptimizationTrial]:
        """Run Bayesian optimization with Optuna TPE sampler.

        Args:
            strategy_factory: Callable that takes a params dict and returns
                a configured BaseStrategy instance.
            ohlcv: Historical OHLCV DataFrame.
            param_space: Dict mapping parameter names to (low, high, type) tuples.
                Type is "int" or "float".
                E.g. {"period": (5, 60, "int"), "threshold": (0.5, 0.9, "float")}.
            n_trials: Number of optimization trials. Default: 100.
            metric: Metric name from BacktestResult.metrics to maximize.
                Default: "sharpe_ratio".
            seed: Random seed for TPE sampler reproducibility. Default: 42.

        Returns:
            List of OptimizationTrial sorted by metric_value descending.
        """
        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for name, (low, high, ptype) in param_space.items():
                if ptype == "int":
                    params[name] = trial.suggest_int(name, low, high)
                elif ptype == "float":
                    params[name] = trial.suggest_float(name, low, high)
                else:
                    msg = f"Unsupported param type: {ptype!r}. Use 'int' or 'float'."
                    raise ValueError(msg)

            strategy = strategy_factory(params)

            runner = BacktestRunner(
                strategy=strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
            )
            result = runner.run(ohlcv)

            # Store full metrics as user attributes for later retrieval
            trial.set_user_attr("metrics", result.metrics)

            return result.metrics.get(metric, 0.0)

        study.optimize(objective, n_trials=n_trials)

        # Build ranked results from completed trials
        trials: list[OptimizationTrial] = []
        for trial in study.trials:
            trials.append(
                OptimizationTrial(
                    params=dict(trial.params),
                    metrics=trial.user_attrs.get("metrics", {}),
                    metric_value=trial.value if trial.value is not None else 0.0,
                )
            )

        # Sort by metric_value descending (best first)
        trials.sort(key=lambda t: t.metric_value, reverse=True)
        return trials
