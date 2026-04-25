"""Outperformance gate for regime routing.

Compares regime-routed VotingStrategy vs static VotingStrategy on holdout data.
Gate auto-enables RegimeRouter only when regime routing STRICTLY outperforms
the static baseline (D-07). Model and config preserved after gate failure (D-08).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from poseidon.backtest.holdout import HoldoutConfig
from poseidon.backtest.metrics import compute_composite_score
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.voting_strategy_factory import VotingStrategyFactory

if TYPE_CHECKING:
    from poseidon.backtest.cost_model import CostModel
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.risk.engine import RiskEngine
    from poseidon.strategies.regime_router import RegimeRouter

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of outperformance gate evaluation."""

    passed: bool
    static_score: float
    regime_score: float
    holdout_start: datetime
    holdout_bars: int


def evaluate_regime_gate(
    ohlcv: pd.DataFrame,
    base_config: dict,
    regime_router: RegimeRouter,
    feature_engine: FeatureEngine,
    risk_engine: RiskEngine,
    cost_model: CostModel,
    holdout_config: HoldoutConfig | None = None,
    initial_capital: float = 100_000.0,
) -> GateResult:
    """Evaluate whether regime routing outperforms static baseline on holdout.

    Runs paired backtest: static VotingStrategy vs RegimeRouter on holdout data.
    Sets regime_router.enabled based on strict outperformance comparison.
    Model and config are preserved after gate failure (D-08).

    Args:
        ohlcv: Full OHLCV DataFrame with datetime index.
        base_config: Base VotingStrategy config dict for static baseline.
        regime_router: RegimeRouter instance to evaluate and enable/disable.
        feature_engine: FeatureEngine for backtest.
        risk_engine: RiskEngine for backtest.
        cost_model: CostModel for backtest.
        holdout_config: Holdout configuration. Uses defaults if None.
        initial_capital: Initial capital for backtests.

    Returns:
        GateResult with pass/fail status and scores.
    """
    holdout_cfg = holdout_config or HoldoutConfig()

    # Step 1: Compute holdout boundary and extract holdout data
    boundary = holdout_cfg.compute_boundary(ohlcv)
    ohlcv_holdout = ohlcv[ohlcv.index >= boundary]
    holdout_bars = len(ohlcv_holdout)

    if holdout_bars < 200:
        logger.warning(
            "Holdout has only %d bars (< 200 recommended). Gate results may be unreliable.",
            holdout_bars,
        )

    # Step 2: Static baseline backtest
    static_strategy = VotingStrategyFactory.from_config(base_config)
    static_runner = BacktestRunner(
        strategy=static_strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
    static_result = static_runner.run(ohlcv_holdout)
    static_score = compute_composite_score(static_result.metrics)

    # Step 3: Regime-routed backtest
    regime_router.enabled = True
    regime_runner = BacktestRunner(
        strategy=regime_router,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
    regime_result = regime_runner.run(ohlcv_holdout)
    regime_score = compute_composite_score(regime_result.metrics)

    # Step 4: Strict outperformance comparison (D-07)
    passed = regime_score > static_score

    # Step 5: Set enabled flag (D-08: model/config preserved, only bypass)
    regime_router.enabled = passed

    logger.info(
        "Regime gate %s: regime_score=%.4f, static_score=%.4f, holdout_bars=%d",
        "PASSED" if passed else "FAILED",
        regime_score,
        static_score,
        holdout_bars,
    )

    return GateResult(
        passed=passed,
        static_score=static_score,
        regime_score=regime_score,
        holdout_start=boundary,
        holdout_bars=holdout_bars,
    )
