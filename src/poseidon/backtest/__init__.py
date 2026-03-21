"""Backtest engine: portfolio simulation, cost models, and metrics.

Public API:
    BacktestPortfolio  - Portfolio simulator with fee/slippage
    CostModel          - Market-specific cost configuration
    COST_MODELS        - Registry of all market cost models
    get_cost_model     - Lookup cost model by market name
    TradeRecord        - Completed trade data
    compute_metrics    - Performance metrics from equity + trades
"""

from poseidon.backtest.cost_model import COST_MODELS, CostModel, get_cost_model
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import BacktestPortfolio, TradeRecord

__all__ = [
    "BacktestPortfolio",
    "CostModel",
    "COST_MODELS",
    "get_cost_model",
    "TradeRecord",
    "compute_metrics",
]
