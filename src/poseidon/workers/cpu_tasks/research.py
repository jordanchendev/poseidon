"""Autoresearch and factor-analysis tasks."""

import contextlib
import logging
import uuid
from datetime import UTC, datetime

from poseidon.core.database import db_session
from poseidon.core.redis import get_redis
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="poseidon.workers.cpu_tasks.autoresearch_run", bind=True)
def autoresearch_run(self, search_config: dict, markets: list[dict]) -> dict:
    """Run autonomous parameter search across markets.

    Single long-running task. Internally loops per-market calling
    ParameterSearchPipeline.run() via AutoResearchRunner.

    Args:
        search_config: SearchConfig fields as plain dict (Celery JSON serializable).
            Keys: n_trials, max_trials, min_wfe, seed, storage_url,
            holdout (dict with holdout_pct), walk_forward (dict with n_splits, min_wfe).
        markets: List of {"symbol": str, "market": str, "interval": str} dicts.

    Returns:
        {"status": "completed"|"stopped", "markets_processed": int, "report": dict}
    """
    from poseidon.autoresearch.report import generate_report
    from poseidon.autoresearch.runner import AutoResearchRunner, MarketSpec
    from poseidon.backtest.experiment_tracker import ExperimentTracker
    from poseidon.backtest.holdout import HoldoutConfig
    from poseidon.backtest.param_search import SearchConfig as SearchConfigClass
    from poseidon.backtest.walk_forward import WalkForwardConfig

    started_at = datetime.now(UTC)
    redis_client = get_redis("celery")
    task_id = self.request.id or "local"

    with db_session() as db:
        try:
            # Reconstruct SearchConfig from plain dict
            holdout_dict = search_config.get("holdout", {})
            wf_dict = search_config.get("walk_forward", {})
            cfg = SearchConfigClass(
                n_trials=search_config.get("n_trials", 50),
                max_trials=search_config.get("max_trials", 100),
                min_wfe=search_config.get("min_wfe", 0.50),
                seed=search_config.get("seed", 42),
                storage_url=search_config.get("storage_url"),
                holdout=HoldoutConfig(**holdout_dict) if holdout_dict else HoldoutConfig(),
                walk_forward=WalkForwardConfig(**wf_dict) if wf_dict else WalkForwardConfig(),
                strategy_mode=search_config.get("strategy_mode", "bidirectional"),
            )

            market_specs = [MarketSpec(**m) for m in markets]

            # graceful stop check via Redis flag
            def check_stop() -> bool:
                return bool(redis_client.get(f"autoresearch:stop:{task_id}"))

            # heartbeat via Celery task state update
            def update_progress(current: int, total: int, symbol: str) -> None:
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current_market": current + 1,
                        "total_markets": total,
                        "symbol": symbol,
                    },
                )

            # optional ML model for sub-signal search
            mv_id = search_config.get("model_version_id")

            # extract dispatch fields from search_config
            strategy_type = search_config.get("strategy_type", "voting")
            feature_names = search_config.get("feature_names")
            base_config_json = search_config.get("base_config_json")

            # regime_router special path -- does NOT use AutoResearchRunner
            if strategy_type == "regime_router":
                from poseidon.backtest.cost_model import COST_MODELS
                from poseidon.backtest.experiment_tracker import ExperimentTracker as ET
                from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline
                from poseidon.data.feature_engine import FeatureOrchestrator
                from poseidon.data.remote_repository import RemoteDataRepository
                from poseidon.risk.engine import RiskEngine

                regime_tracker = ET(db)
                feature_engine = FeatureOrchestrator()
                risk_engine = RiskEngine()

                base_config = base_config_json or {}
                regime_results_all = []

                for i, spec_dict in enumerate(markets):
                    spec = MarketSpec(**spec_dict)
                    if check_stop():
                        break
                    update_progress(i, len(markets), spec.symbol)

                    cost_model = COST_MODELS.get(spec.market)
                    if cost_model is None:
                        from poseidon.backtest.cost_model import CostModel

                        cost_model = CostModel(
                            market=spec.market,
                            buy_commission_rate=0.0,
                            sell_commission_rate=0.0,
                            tax_rate=0.0,
                            slippage_pct=0.0,
                            slippage_ticks=0.0,
                            description=f"Default for {spec.market}",
                        )

                    repo = RemoteDataRepository.from_settings()
                    ohlcv = repo.read_ohlcv(spec.symbol, spec.market, spec.interval)
                    if ohlcv.empty:
                        continue

                    # Load regime model
                    from poseidon.ml.manager import ModelManager

                    manager = ModelManager(db)
                    regime_models = manager.list_ready_models(spec.market)
                    regime_model = None
                    for m in regime_models:
                        if "regime" in m.name.lower():
                            regime_model = m
                            break
                    if regime_model is None:
                        logger.warning("No regime model found for %s, skipping regime_router", spec.market)
                        continue

                    pipeline = RegimeSearchPipeline(
                        feature_engine=feature_engine,
                        risk_engine=risk_engine,
                        cost_model=cost_model,
                        experiment_tracker=regime_tracker,
                    )
                    regime_cfg = RegimeSearchConfig(
                        n_trials_per_regime=max(cfg.n_trials // 3, 10),
                        seed=cfg.seed,
                    )
                    result = pipeline.run(
                        ohlcv=ohlcv,
                        base_config=base_config,
                        regime_model=regime_model,
                        config=regime_cfg,
                        market=spec.market,
                        symbol=spec.symbol,
                        interval=spec.interval,
                    )
                    regime_results_all.append({"spec": spec_dict, "result": result})

                completed_at = datetime.now(UTC)
                return {
                    "status": "stopped" if check_stop() else "completed",
                    "markets_processed": len(regime_results_all),
                    "report": {"regime_results": regime_results_all},
                }

            # FACTORY_REGISTRY dict for standard AutoResearchRunner path.
            # voting maps to None (VotingStrategyFactory is the default when strategy_factory=None).
            # rule uses a lambda to capture feature_names at dispatch time (constructor arg).
            # model is initialised with available_models=None; AutoResearchRunner's per-market loop
            #   (runner.py:164-173) calls list_ready_models(market) and passes
            #   available_models into the pipeline, which reaches build_trial_factory() dynamically.
            from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
            from poseidon.backtest.model_strategy_factory import ModelStrategyFactory
            from poseidon.backtest.rule_strategy_factory import RuleStrategyFactory

            FACTORY_REGISTRY = {
                "voting": lambda: None,
                "rule": lambda: RuleStrategyFactory(
                    feature_names=feature_names or [],
                    direction_mode=search_config.get("strategy_mode", "bidirectional"),
                ),
                # DSL v2: the search space is the spec's own $param grid.
                "rule_v2": lambda: DslV2StrategyFactory(spec=base_config_json or {}),
                "model": lambda: ModelStrategyFactory(available_models=None),
            }

            factory_builder = FACTORY_REGISTRY.get(strategy_type)
            if factory_builder is None:
                raise ValueError(f"Unknown strategy_type {strategy_type!r}; should have been caught by API validation")
            strategy_factory = factory_builder()

            runner = AutoResearchRunner(
                db_session=db,
                search_config=cfg,
                stop_check=check_stop,
                progress_callback=update_progress,
                feature_specs="r2",  # Signal runner to use get_r2_specs() per-market
                model_version_id=mv_id,
                strategy_factory=strategy_factory,
            )
            results = runner.run(market_specs)

            # Generate report
            completed_at = datetime.now(UTC)
            tracker = ExperimentTracker(db)
            study_names = [r.search_result.study_name for r in results if r.search_result is not None]
            report = generate_report(
                tracker,
                study_names,
                run_id=task_id,
                started_at=started_at,
                completed_at=completed_at,
            )
            report["markets_failed"] = sum(1 for r in results if r.error is not None)

            was_stopped = check_stop()
            return {
                "status": "stopped" if was_stopped else "completed",
                "markets_processed": len(results),
                "report": report,
            }
        finally:
            # clean up stop flag
            with contextlib.suppress(Exception):
                redis_client.delete(f"autoresearch:stop:{task_id}")


@celery_app.task(
    name="poseidon.workers.cpu_tasks.factor_ic_analysis",
    queue="poseidon_cpu",
    bind=True,
    max_retries=0,
)
def factor_ic_analysis(self, run_id: str):
    """Compute Rank IC for features against forward returns."""
    from poseidon.models.factor_analysis_run import FactorAnalysisRun
    from poseidon.research.ic_analysis import run_ic_analysis

    with db_session() as session:
        run = None
        try:
            run = session.query(FactorAnalysisRun).filter_by(id=uuid.UUID(run_id)).one()
            run.status = "running"
            run.error = None
            session.commit()

            config = run.config_json
            run.results_json = run_ic_analysis(
                market=config["market"],
                symbols=config.get("symbols"),
                start_date=config["start_date"],
                end_date=config["end_date"],
                horizons=config.get("horizons", [1, 5, 20]),
                features=config.get("features"),
                interval=config.get("interval", "1d"),
            )
            run.status = "succeeded"
            session.commit()
            return {"run_id": run_id, "status": "succeeded"}
        except Exception as exc:
            logger.exception("factor_ic_analysis failed for run_id=%s", run_id)
            if run is not None:
                session.rollback()
                run = session.query(FactorAnalysisRun).filter_by(id=uuid.UUID(run_id)).first()
                if run is not None:
                    run.status = "failed"
                    run.error = str(exc)[:2000]
                    session.commit()
            return {"run_id": run_id, "status": "failed", "error": str(exc)}


@celery_app.task(
    name="poseidon.workers.cpu_tasks.factor_centrality_analysis",
    queue="poseidon_cpu",
    bind=True,
    max_retries=0,
)
def factor_centrality_analysis(self, run_id: str):
    """Compute sub-signal centrality (overlap detection)."""
    from poseidon.models.factor_analysis_run import FactorAnalysisRun
    from poseidon.research.centrality_analysis import run_centrality_analysis

    with db_session() as session:
        run = None
        try:
            run = session.query(FactorAnalysisRun).filter_by(id=uuid.UUID(run_id)).one()
            run.status = "running"
            run.error = None
            session.commit()

            config = run.config_json
            run.results_json = run_centrality_analysis(
                market=config["market"],
                sub_signals=config["sub_signals"],
                symbols=config.get("symbols"),
                start_date=config["start_date"],
                end_date=config["end_date"],
                interval=config.get("interval", "1d"),
                distance_threshold=config.get("distance_threshold", 0.7),
            )
            run.status = "succeeded"
            session.commit()
            return {"run_id": run_id, "status": "succeeded"}
        except Exception as exc:
            logger.exception("factor_centrality_analysis failed for run_id=%s", run_id)
            if run is not None:
                session.rollback()
                run = session.query(FactorAnalysisRun).filter_by(id=uuid.UUID(run_id)).first()
                if run is not None:
                    run.status = "failed"
                    run.error = str(exc)[:2000]
                    session.commit()
            return {"run_id": run_id, "status": "failed", "error": str(exc)}
