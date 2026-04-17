"""Parameter search pipeline with holdout, WFE gate, and experiment tracking.

Orchestrates: holdout split -> Optuna search -> WFE validation -> experiment logging.
Per D-15: each (market, interval) gets its own study. 50-100 trials per study.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.backtest.holdout import HoldoutConfig
from poseidon.backtest.metrics import compute_composite_score
from poseidon.backtest.optimizer import BayesianOptimizer
from poseidon.backtest.portfolio import SizingConfig
from poseidon.backtest.voting_strategy_factory import (
    VotingStrategyFactory,
    _build_config_from_params,
    get_param_bounds,
)
from poseidon.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for a parameter search run."""

    n_trials: int = 50
    max_trials: int = 100  # hard cap per D-05
    min_wfe: float = 0.50  # WFE gate threshold per D-11
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    seed: int = 42
    storage_url: str | None = None  # PostgreSQL connection string for Optuna RDBStorage
    strategy_mode: str = "bidirectional"  # bidirectional | long_only | regime_gated

    def __post_init__(self) -> None:
        if self.n_trials > self.max_trials:
            self.n_trials = self.max_trials


@dataclass
class SearchResult:
    """Result of a parameter search run."""

    study_name: str
    total_trials: int
    passed_trials: int
    rejected_trials: int
    best_config: dict | None
    best_composite_score: float | None
    holdout_boundary: datetime


class ParameterSearchPipeline:
    """Orchestrates parameter search with holdout, WFE gate, and experiment tracking.

    Pipeline steps:
    1. Compute holdout boundary, trim OHLCV to train portion
    2. Create per-market Optuna study with RDBStorage
    3. Run BayesianOptimizer with VotingStrategyFactory param bounds
    4. For each completed trial:
       a. Log to ExperimentTracker
       b. Run WFE validation on trial params
       c. Mark passed (WFE >= 50%) or rejected (WFE < 50%)
    5. Return SearchResult with best passing trial
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        tracker: ExperimentTracker,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
        db_session: Any | None = None,
        strategy_factory: Any | None = None,  # D-02: injectable strategy factory
    ) -> None:
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.tracker = tracker
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()
        self.db_session = db_session
        self.strategy_factory = strategy_factory  # None = VotingStrategyFactory (backward compat)

    def run(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        market: str,
        interval: str,
        config: SearchConfig | None = None,
        model_version_id: int | None = None,
        available_models: list[Any] | None = None,
    ) -> SearchResult:
        """Execute full parameter search pipeline.

        Args:
            ohlcv: Full OHLCV DataFrame with datetime index.
            symbol: Trading pair symbol (e.g. "BTCUSDT").
            market: Market identifier (e.g. "crypto_spot").
            interval: Time interval (e.g. "1d", "1h").
            config: Search configuration. Uses defaults if None.

        Returns:
            SearchResult with trial counts, best config, and holdout boundary.
        """
        cfg = config or SearchConfig()
        study_name = f"{market}_{symbol}_{interval}"

        # Step 1: Holdout boundary
        holdout_boundary = cfg.holdout.compute_boundary(ohlcv)
        train_ohlcv = ohlcv[ohlcv.index < holdout_boundary]
        cfg.holdout.validate_data_range(train_ohlcv, holdout_boundary)
        logger.info(
            "Holdout boundary: %s, train rows: %d",
            holdout_boundary,
            len(train_ohlcv),
        )

        # Step 2: Storage URL for Optuna RDBStorage
        storage = cfg.storage_url
        if storage and "search_path" not in storage:
            # Ensure optuna schema is used
            sep = "&" if "?" in storage else "?"
            storage = f"{storage}{sep}options=-csearch_path=optuna"

        # Step 3: Build optimizer and run
        optimizer = BayesianOptimizer(
            feature_engine=self.feature_engine,
            risk_engine=self.risk_engine,
            cost_model=self.cost_model,
            initial_capital=self.initial_capital,
            sizing_config=self.sizing_config,
            db_session=self.db_session,
        )

        # Strategy factory for optimizer: wraps VotingStrategyFactory.from_config
        mode = cfg.strategy_mode
        models_list = available_models or []
        forced_mv_id = model_version_id

        def resolve_model_version_id(params: dict[str, Any]) -> str | None:
            mv_id = forced_mv_id
            if mv_id is None and models_list and params.get("qlib_model_enabled") == 1:
                version_idx = int(params.get("qlib_model_version", 0))
                version_idx = min(version_idx, len(models_list) - 1)
                mv_id = models_list[version_idx].id
            # Convert UUID to string for JSON serialization in experiment records
            return str(mv_id) if mv_id is not None else None

        # Strategy factory for optimizer (D-02): polymorphic via build_trial_factory()
        if self.strategy_factory is not None:
            # Use injected factory -- call build_trial_factory() to get
            # (trial_strategy_factory_fn, param_bounds) without importing strategy internals.
            # This keeps param_search.py truly polymorphic per D-02.
            trial_strategy_factory, param_space = self.strategy_factory.build_trial_factory(
                symbol=symbol, market=market, interval=interval,
            )
        else:
            # Original VotingStrategyFactory path (unchanged for backward compat)
            def trial_strategy_factory(params: dict) -> Any:
                config_dict = _build_config_from_params(
                    params, symbol=symbol, market=market, interval=interval,
                    strategy_mode=mode,
                    model_version_id=resolve_model_version_id(params),
                )
                return VotingStrategyFactory.from_config(config_dict)

            param_space = {
                name: (low, high, ptype)
                for name, (low, high, ptype) in get_param_bounds(market).items()
            }
            if models_list and model_version_id is None:
                if len(models_list) > 1:
                    param_space["qlib_model_version"] = (0, len(models_list) - 1, "int")
            elif not models_list:
                param_space.pop("qlib_model_enabled", None)
                param_space.pop("qlib_model_version", None)

        trials = optimizer.optimize(
            strategy_factory=trial_strategy_factory,
            ohlcv=train_ohlcv,
            param_space=param_space,
            n_trials=cfg.n_trials,
            metric="composite_score",
            seed=cfg.seed,
            storage=storage,
            study_name=study_name,
        )

        # Step 4: WFE validation and experiment logging
        wf_analyzer = WalkForwardAnalyzer(
            feature_engine=self.feature_engine,
            risk_engine=self.risk_engine,
            cost_model=self.cost_model,
            initial_capital=self.initial_capital,
            sizing_config=self.sizing_config,
            db_session=self.db_session,
            strategy_factory=self.strategy_factory,
        )

        passed_count = 0
        rejected_count = 0
        best_config: dict | None = None
        best_score: float | None = None

        for i, trial in enumerate(trials):
            composite = compute_composite_score(trial.metrics)

            # Build strategy from this trial's params for WFE validation
            if self.strategy_factory is not None:
                # Use injected factory's trial_strategy_factory callable
                strategy_for_wfe = trial_strategy_factory(trial.params)
                config_dict = trial.params  # Flat params for logging
            else:
                config_dict = _build_config_from_params(
                    trial.params, symbol=symbol, market=market, interval=interval,
                    strategy_mode=mode,
                    model_version_id=resolve_model_version_id(trial.params),
                )
                strategy_for_wfe = VotingStrategyFactory.from_config(config_dict)
            try:
                strategy = strategy_for_wfe
                wf_result = wf_analyzer.analyze(
                    strategy=strategy,
                    ohlcv=train_ohlcv,
                    config=cfg.walk_forward,
                )
                wfe = wf_result.wfe
                wfe_passed = wf_result.passed and wfe >= cfg.min_wfe
            except Exception as e:
                logger.warning("WFE analysis failed for trial %d: %s", i, e)
                wfe = 0.0
                wfe_passed = False

            status = "passed" if wfe_passed else "rejected"

            self.tracker.save(
                study_name=study_name,
                config_json=config_dict,
                market=market,
                interval=interval,
                metrics_json=trial.metrics,
                composite_score=composite,
                wfe_score=wfe,
                status=status,
                optuna_study_name=study_name,
                optuna_trial_number=i,
                holdout_boundary=holdout_boundary,
            )

            if wfe_passed:
                passed_count += 1
                if best_score is None or composite > best_score:
                    best_score = composite
                    best_config = config_dict
            else:
                rejected_count += 1

        return SearchResult(
            study_name=study_name,
            total_trials=len(trials),
            passed_trials=passed_count,
            rejected_trials=rejected_count,
            best_config=best_config,
            best_composite_score=best_score,
            holdout_boundary=holdout_boundary,
        )
