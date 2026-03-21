"""Backtest engine: portfolio simulation, cost models, metrics, runner, walk-forward, and optimization.

Public API:
    BacktestPortfolio    - Portfolio simulator with fee/slippage
    CostModel            - Market-specific cost configuration
    COST_MODELS          - Registry of all market cost models
    get_cost_model       - Lookup cost model by market name
    TradeRecord          - Completed trade data
    compute_metrics      - Performance metrics from equity + trades
    BacktestRunner       - Bar-by-bar event loop with pipeline reuse
    BacktestConfig       - Pydantic config schema for backtest runs
    BacktestResult       - Pydantic result schema for backtest output
    BacktestRepository   - DB persistence for backtest results
    WalkForwardConfig    - Walk-forward analysis configuration
    WalkForwardResult    - Walk-forward analysis result
    WalkForwardAnalyzer  - Walk-forward analysis engine
    compute_wfe          - Walk-Forward Efficiency computation
    GridSearchOptimizer  - Exhaustive parameter search
    BayesianOptimizer    - Optuna TPE Bayesian optimization
    OptimizationTrial    - Single optimization trial result
"""

from poseidon.backtest.cost_model import COST_MODELS, CostModel, get_cost_model
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.optimizer import (
    BayesianOptimizer,
    GridSearchOptimizer,
    OptimizationTrial,
)
from poseidon.backtest.portfolio import BacktestPortfolio, TradeRecord
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    compute_wfe,
)

__all__ = [
    "BacktestPortfolio",
    "CostModel",
    "COST_MODELS",
    "get_cost_model",
    "TradeRecord",
    "compute_metrics",
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "BacktestRepository",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardAnalyzer",
    "compute_wfe",
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "OptimizationTrial",
]
