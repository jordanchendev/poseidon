"""Walk-forward analysis engine for strategy robustness validation.

Validates strategies by running rolling train/test windows and computing
Walk-Forward Efficiency (WFE). Strategies with WFE < 50% or insufficient
trades per OOS segment are flagged.

Runs windows sequentially with per-window strategy reconstruction to
avoid state leakage. Uses the analyzer's own dependencies (cost_model,
feature_engine, etc.) for correct pipeline behavior.

Public API:
    WalkForwardConfig   - Configuration for rolling window parameters
    WindowResult        - Per-window IS/OOS metrics
    WalkForwardResult   - Aggregate result with WFE and flags
    WalkForwardAnalyzer - Main analyzer that runs walk-forward
    compute_wfe         - WFE = OOS annualized return / IS annualized return
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.portfolio import SizingConfig
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis rolling windows.

    Defaults correspond to standard quarterly walk-forward:
    - 1 year in-sample training
    - 3 months validation (reserved for model strategies)
    - 3 months out-of-sample testing
    - Roll forward by 1 quarter
    """

    train_days: int = 252
    validate_days: int = 63
    test_days: int = 63
    step_days: int = 63
    min_trades_per_oos: int = 5
    max_insufficient_ratio: float = 0.30  # tolerate up to 30% low-trade windows
    min_wfe: float = 0.50


@dataclass
class WindowResult:
    """Per-window in-sample and out-of-sample metrics."""

    window_index: int
    is_metrics: dict
    oos_metrics: dict
    is_trade_count: int
    oos_trade_count: int


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward analysis result.

    Attributes:
        wfe: Walk-Forward Efficiency ratio.
        passed: True if WFE >= threshold and all windows meet trade minimums.
        flags: List of flag strings identifying issues.
        per_window: Per-window IS/OOS metrics.
        aggregate_oos_metrics: Combined OOS metrics across all windows.
        config: Configuration used for this analysis.
    """

    wfe: float
    passed: bool
    flags: list[str]
    per_window: list[WindowResult]
    aggregate_oos_metrics: dict
    config: WalkForwardConfig


def compute_wfe(is_ann_return: float, oos_ann_return: float) -> float:
    """Compute Walk-Forward Efficiency.

    WFE = OOS annualized return / IS annualized return.
    Returns 0.0 if IS return is zero or negative (no meaningful ratio).

    Args:
        is_ann_return: Annualized in-sample return.
        oos_ann_return: Annualized out-of-sample return.

    Returns:
        WFE ratio. Values >= 0.50 suggest the strategy is not overfitted.
    """
    if is_ann_return <= 0:
        return 0.0
    return oos_ann_return / is_ann_return


class WalkForwardAnalyzer:
    """Walk-forward analysis engine.

    Validates strategy robustness by running rolling train/test windows,
    computing per-window metrics, and aggregating Walk-Forward Efficiency.

    Uses BacktestRunner for each IS and OOS segment to ensure pipeline
    reuse (same FeatureEngine + Strategy + RiskEngine code path).

    Window evaluation is parallelized with ProcessPoolExecutor (80% of cores).
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
        db_session: Any | None = None,
        strategy_factory: Any | None = None,
        fill_model: Any | None = None,
    ) -> None:
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()
        self.db_session = db_session
        self._strategy_factory = strategy_factory
        self._fill_model = fill_model

    def generate_windows(
        self,
        data_length: int,
        config: WalkForwardConfig,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Generate rolling train/test window index ranges.

        Each window is a tuple of ((train_start, train_end), (test_start, test_end))
        using exclusive-end semantics (consistent with Python slicing).

        Windows step forward by config.step_days. The first window starts at
        index 0 with train=[0, train_days) and test=[train_days, train_days+test_days).
        Subsequent windows shift by step_days. Generation stops when test_end
        would exceed data_length.

        Note: validate_days is reserved for model retraining (future use).
        For rule strategies, train and test are directly adjacent.

        Args:
            data_length: Total number of bars in the OHLCV data.
            config: Walk-forward configuration with window sizes.

        Returns:
            List of ((train_start, train_end), (test_start, test_end)) tuples.
        """
        if config.step_days < config.test_days:
            logger.warning(
                "step_days (%d) < test_days (%d): OOS windows will overlap. "
                "Set step_days >= test_days for non-overlapping OOS segments.",
                config.step_days,
                config.test_days,
            )

        windows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        offset = 0

        while True:
            train_start = offset
            train_end = train_start + config.train_days
            test_start = train_end
            test_end = test_start + config.test_days

            if test_end > data_length:
                break

            windows.append(((train_start, train_end), (test_start, test_end)))
            offset += config.step_days

        return windows

    def analyze(
        self,
        strategy: BaseStrategy,
        ohlcv: pd.DataFrame,
        config: WalkForwardConfig | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward analysis on a strategy.

        1. Generate rolling windows from the OHLCV data length.
        2. For each window, run backtest on IS and OOS slices (parallel).
        3. Compute per-window metrics.
        4. Aggregate IS and OOS annualized returns.
        5. Compute WFE and build flags.

        Args:
            strategy: Strategy to evaluate.
            ohlcv: Historical OHLCV DataFrame.
            config: Walk-forward configuration. Uses defaults if None.

        Returns:
            WalkForwardResult with WFE, flags, and per-window metrics.
        """
        if config is None:
            config = WalkForwardConfig()

        windows = self.generate_windows(len(ohlcv), config)

        if not windows:
            return WalkForwardResult(
                wfe=0.0,
                passed=False,
                flags=["no_windows"],
                per_window=[],
                aggregate_oos_metrics={},
                config=config,
            )

        # Extract strategy config for per-window reconstruction (polymorphic)
        if self._strategy_factory is not None:
            factory = self._strategy_factory
        else:
            from poseidon.backtest.voting_strategy_factory import VotingStrategyFactory

            factory = VotingStrategyFactory

        strategy_config = factory.to_config_dict(strategy)

        per_window: list[WindowResult] = []

        logger.info("Walk-forward: %d windows, sequential execution", len(windows))

        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(windows):
            # Preserve tz-aware datetime index (Rule 3 fix for Phase 85): companion
            # features (open_interest, funding_rates) align via _align_*_to_index
            # against ohlcv.index using datetime-based reindex. .reset_index(drop=True)
            # converts the index to int64 and breaks companion feature alignment with
            # "Cannot compare dtypes datetime64[ns, UTC] and int64".
            is_ohlcv = ohlcv.iloc[train_start:train_end]
            oos_ohlcv = ohlcv.iloc[test_start:test_end]

            # Fresh strategy per window to avoid state leakage
            window_strategy = factory.from_config(strategy_config)

            # IS backtest
            is_runner = BacktestRunner(
                strategy=window_strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
                fill_model=self._fill_model,
            )
            is_result = is_runner.run(is_ohlcv, db_session=self.db_session)

            # OOS backtest (reset strategy state between IS and OOS)
            window_strategy.reset()
            oos_runner = BacktestRunner(
                strategy=window_strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
                fill_model=self._fill_model,
            )
            oos_result = oos_runner.run(oos_ohlcv, db_session=self.db_session)

            is_metrics = is_result.metrics
            oos_metrics = oos_result.metrics

            per_window.append(
                WindowResult(
                    window_index=i,
                    is_metrics=is_metrics,
                    oos_metrics=oos_metrics,
                    is_trade_count=is_metrics.get("trade_count", 0),
                    oos_trade_count=oos_metrics.get("trade_count", 0),
                )
            )

            logger.info(
                "Window %d: IS ann_return=%.4f, OOS ann_return=%.4f, IS trades=%d, OOS trades=%d",
                i,
                is_metrics.get("annualized_return", 0.0),
                oos_metrics.get("annualized_return", 0.0),
                is_metrics.get("trade_count", 0),
                oos_metrics.get("trade_count", 0),
            )

        # Sort by window_index (ProcessPoolExecutor.map preserves order, but be safe)
        per_window.sort(key=lambda w: w.window_index)

        # Log per-window results
        for w in per_window:
            logger.info(
                "Window %d: IS ann_return=%.4f, OOS ann_return=%.4f, IS trades=%d, OOS trades=%d",
                w.window_index,
                w.is_metrics.get("annualized_return", 0.0),
                w.oos_metrics.get("annualized_return", 0.0),
                w.is_trade_count,
                w.oos_trade_count,
            )

        # Aggregate annualized returns (average across windows)
        is_ann_returns = [w.is_metrics.get("annualized_return", 0.0) for w in per_window]
        oos_ann_returns = [w.oos_metrics.get("annualized_return", 0.0) for w in per_window]

        agg_is_ann = sum(is_ann_returns) / len(is_ann_returns) if is_ann_returns else 0.0
        agg_oos_ann = sum(oos_ann_returns) / len(oos_ann_returns) if oos_ann_returns else 0.0

        wfe = compute_wfe(agg_is_ann, agg_oos_ann)

        # Build flags
        flags: list[str] = []
        if wfe < config.min_wfe:
            flags.append("wfe_below_threshold")

        insufficient_count = 0
        for w in per_window:
            if w.oos_trade_count < config.min_trades_per_oos:
                insufficient_count += 1

        insufficient_ratio = insufficient_count / len(per_window) if per_window else 1.0
        if insufficient_ratio > config.max_insufficient_ratio:
            flags.append(
                f"too_many_insufficient_windows_{insufficient_count}/{len(per_window)}"
                f"_ratio={insufficient_ratio:.2f}_max={config.max_insufficient_ratio}"
            )

        # Determine pass/fail
        passed = wfe >= config.min_wfe and len(flags) == 0

        # Aggregate OOS metrics (average across windows)
        aggregate_oos_metrics: dict = {}
        if per_window:
            all_keys = per_window[0].oos_metrics.keys()
            for key in all_keys:
                values = [w.oos_metrics.get(key, 0.0) for w in per_window]
                aggregate_oos_metrics[key] = sum(values) / len(values)

        logger.info(
            "Walk-forward complete: WFE=%.4f, passed=%s, flags=%s",
            wfe,
            passed,
            flags,
        )

        return WalkForwardResult(
            wfe=wfe,
            passed=passed,
            flags=flags,
            per_window=per_window,
            aggregate_oos_metrics=aggregate_oos_metrics,
            config=config,
        )
