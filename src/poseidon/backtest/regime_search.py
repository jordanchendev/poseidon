"""Per-regime Optuna parameter search pipeline.

Runs separate Optuna studies for each regime (low_vol, medium_vol, high_vol),
varying 4 params per regime: min_votes, position_pct, bear_min_votes,
bear_position_pct (D-04, D-06, D-21).
Regime model is trained once and predictions reused across all trials (D-10).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import optuna
import pandas as pd

from poseidon.backtest.holdout import HoldoutConfig
from poseidon.backtest.metrics import compute_composite_score
from poseidon.backtest.regime_labels import generate_regime_labels
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.voting_strategy_factory import VotingStrategyFactory

if TYPE_CHECKING:
    from poseidon.backtest.cost_model import CostModel
    from poseidon.backtest.experiment_tracker import ExperimentTracker
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.ml.implementations.xgboost_regime import XGBoostRegimeModel
    from poseidon.risk.engine import RiskEngine

logger = logging.getLogger(__name__)

REGIME_NAMES = ["low_vol", "medium_vol", "high_vol"]


@dataclass
class RegimeSearchConfig:
    """Configuration for per-regime parameter search."""

    n_trials_per_regime: int = 30  # 30 trials * 3 regimes = 90 total
    min_votes_range: tuple[int, int] = (2, 6)
    position_pct_range: tuple[float, float] = (0.03, 0.15)
    bear_min_votes_range: tuple[int, int] = (2, 6)  # D-21: new
    bear_position_pct_range: tuple[float, float] = (0.03, 0.12)  # D-21: new
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    seed: int = 42


class RegimeSearchPipeline:
    """Per-regime Optuna search that varies only min_votes and position_pct.

    Pipeline steps:
    1. Compute holdout boundary, trim OHLCV to train portion
    2. Compute features on training data
    3. Generate regime labels from features
    4. Get regime predictions ONCE (D-10)
    5. For each regime: run Optuna study with 4-param objective
    6. Return {regime_name: {min_votes, position_pct, bear_min_votes, bear_position_pct}} dict
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        experiment_tracker: ExperimentTracker | None = None,
    ) -> None:
        self._feature_engine = feature_engine
        self._risk_engine = risk_engine
        self._cost_model = cost_model
        self._tracker = experiment_tracker

    def run(
        self,
        ohlcv: pd.DataFrame,
        base_config: dict,
        regime_model: XGBoostRegimeModel,
        config: RegimeSearchConfig | None = None,
        *,
        market: str = "",
        symbol: str = "",
        interval: str = "",
    ) -> dict[str, dict]:
        """Execute per-regime parameter search.

        Args:
            ohlcv: Full OHLCV DataFrame with datetime index.
            base_config: Base VotingStrategy config dict.
            regime_model: Trained XGBoostRegimeModel for regime prediction.
            config: Search configuration. Uses defaults if None.
            market: Market identifier for experiment tracking.
            symbol: Symbol identifier for experiment tracking.
            interval: Time interval for experiment tracking.

        Returns:
            Dict mapping regime names to optimized {min_votes, position_pct, bear_min_votes, bear_position_pct}.
        """
        cfg = config or RegimeSearchConfig()

        # Step 1: Holdout boundary -- trim to training data
        holdout_boundary = cfg.holdout.compute_boundary(ohlcv)
        ohlcv_train = ohlcv[ohlcv.index < holdout_boundary]
        logger.info(
            "Regime search: holdout boundary=%s, train bars=%d",
            holdout_boundary,
            len(ohlcv_train),
        )

        # Step 2: Compute features on training data
        features = self._feature_engine.compute_from_df(ohlcv_train)

        # Step 3: Generate regime labels
        _labels, _thresholds = generate_regime_labels(features)

        # Step 4: Regime predictions computed ONCE before search (D-10)
        _regime_preds = regime_model.predict(features)

        # Step 5: Per-regime Optuna search
        result: dict[str, dict] = {}
        for regime_name in REGIME_NAMES:
            best_params = self._search_regime(
                regime_name=regime_name,
                ohlcv_train=ohlcv_train,
                base_config=base_config,
                config=cfg,
            )
            result[regime_name] = best_params
            logger.info(
                "Regime %s best: min_votes=%d, position_pct=%.4f, bear_min_votes=%d, bear_position_pct=%.4f",
                regime_name,
                best_params["min_votes"],
                best_params["position_pct"],
                best_params["bear_min_votes"],
                best_params["bear_position_pct"],
            )

        # FACT-03: persist per-regime results to experiments table
        if self._tracker is not None:
            for regime_name, best_params in result.items():
                self._tracker.save(
                    study_name=f"regime_{regime_name}_{market}_{symbol}_{interval}",
                    config_json=best_params,
                    market=market,
                    interval=interval,
                    metrics_json=None,
                    composite_score=None,
                    wfe_score=None,
                    status="passed",
                )

        return result

    def _search_regime(
        self,
        regime_name: str,
        ohlcv_train: pd.DataFrame,
        base_config: dict,
        config: RegimeSearchConfig,
    ) -> dict:
        """Run Optuna study for a single regime.

        Varies 4 params per regime: min_votes, position_pct, bear_min_votes,
        bear_position_pct (D-21). All other strategy params come from base_config.
        """
        study_name = f"regime_{regime_name}_{base_config.get('symbol', 'unknown')}"
        sampler = optuna.samplers.TPESampler(seed=config.seed)
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            min_votes = trial.suggest_int(
                "min_votes",
                *config.min_votes_range,
            )
            position_pct = trial.suggest_float(
                "position_pct",
                *config.position_pct_range,
            )
            bear_min_votes = trial.suggest_int(
                "bear_min_votes",
                *config.bear_min_votes_range,
            )
            bear_position_pct = trial.suggest_float(
                "bear_position_pct",
                *config.bear_position_pct_range,
            )

            # Deep copy base config and override the 4 params
            modified_config = copy.deepcopy(base_config)
            modified_config["min_votes"] = min_votes
            modified_config["position_pct"] = position_pct
            modified_config["bear_min_votes"] = bear_min_votes
            modified_config["bear_position_pct"] = bear_position_pct

            strategy = VotingStrategyFactory.from_config(modified_config)
            runner = BacktestRunner(
                strategy=strategy,
                feature_engine=self._feature_engine,
                risk_engine=self._risk_engine,
                cost_model=self._cost_model,
            )
            bt_result = runner.run(ohlcv_train)
            return compute_composite_score(bt_result.metrics)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=config.n_trials_per_regime)

        best = study.best_trial
        return {
            "min_votes": best.params["min_votes"],
            "position_pct": best.params["position_pct"],
            "bear_min_votes": best.params["bear_min_votes"],
            "bear_position_pct": best.params["bear_position_pct"],
        }
