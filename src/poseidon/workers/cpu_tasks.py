"""CPU worker Celery tasks for backtest, optimization, trading, and research.

Phase 61: All data-fetching / ingest / backfill tasks removed.
Poseidon reads data exclusively via Thalassa RemoteDataRepository.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.optimizer import BayesianOptimizer, GridSearchOptimizer
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig
from poseidon.core.config import settings
from poseidon.core.database import db_session, SessionLocal
from poseidon.core.redis import get_redis
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.symbols import load_symbols
from poseidon.models.backtest import BacktestRecord
from poseidon.models.strategy import StrategyRecord
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.voting_strategy import VotingStrategy
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _build_portfolio_strategy(record, repo):  # noqa: ANN001
    """Build a PortfolioStrategy instance from StrategyRecord config."""
    from poseidon.strategies.portfolio.crypto_trend import CryptoTrendConfig
    from poseidon.strategies.portfolio.fundamental_selection import (
        FundamentalSelectionConfig,
    )
    from poseidon.strategies.portfolio.prediction_ranking import (
        PredictionRankingConfig,
    )
    from poseidon.strategies.portfolio.registry import get_portfolio_strategy
    from poseidon.strategies.portfolio.schemas import RevenueBreakoutConfig

    raw_cfg = dict(record.config or {})
    strategy_name = raw_cfg.get("strategy")
    if not strategy_name:
        raise ValueError("portfolio_strategy config must include 'strategy'")

    raw_cfg.setdefault("market", record.market)
    raw_cfg.setdefault("symbols", [record.symbol] if record.symbol else [])

    config_models = {
        "fundamental_selection": FundamentalSelectionConfig,
        "revenue_breakout": RevenueBreakoutConfig,
        "crypto_trend": CryptoTrendConfig,
        "prediction_ranking": PredictionRankingConfig,
    }
    if strategy_name not in config_models:
        raise ValueError(f"Unknown portfolio strategy: {strategy_name!r}")

    strategy_cls = get_portfolio_strategy(strategy_name)
    strategy_cfg = config_models[strategy_name](**raw_cfg)

    try:
        return strategy_cls(strategy_cfg, repo=repo)
    except TypeError:
        return strategy_cls(strategy_cfg)


def _serialize_portfolio_trades(trades: list[dict]):  # noqa: ANN001
    """Convert portfolio backtester trade dicts to repository TradeRecord objects."""
    from poseidon.backtest.portfolio import TradeRecord

    result: list[TradeRecord] = []
    for trade in trades:
        trade_dt = datetime.fromisoformat(trade["date"]).replace(tzinfo=timezone.utc)
        metadata = {
            key: value
            for key, value in trade.items()
            if key not in {"symbol", "action", "date", "price", "shares"}
        }
        result.append(
            TradeRecord(
                symbol=trade["symbol"],
                action=trade["action"],
                entry_time=trade_dt,
                entry_price=float(trade["price"]),
                quantity=float(trade.get("shares", 0.0)),
                fees=0.0,
                metadata=metadata,
            )
        )
    return result


def _serialize_portfolio_equity_curve(equity_curve: list[tuple]):  # noqa: ANN001
    """Convert portfolio equity points to repository-compatible tuples."""
    result: list[tuple[datetime, float, float]] = []
    peak = 0.0

    for point_in_time, equity in equity_curve:
        if isinstance(point_in_time, datetime):
            ts = point_in_time if point_in_time.tzinfo else point_in_time.replace(tzinfo=timezone.utc)
        else:
            ts = datetime(
                point_in_time.year,
                point_in_time.month,
                point_in_time.day,
                tzinfo=timezone.utc,
            )

        equity_value = float(equity)
        peak = max(peak, equity_value)
        drawdown = 0.0 if peak <= 0 else (peak - equity_value) / peak
        result.append((ts, equity_value, drawdown))

    return result


def _build_tw_stock_broker(broker_cfg):
    """Build the existing TW stock execution adapter from broker config."""
    from poseidon.broker.paper_adapter import PaperBrokerAdapter

    if broker_cfg.mode != "live" or broker_cfg.execution_backend == "paper":
        broker = PaperBrokerAdapter(SessionLocal)
        broker.login()
        return broker

    if broker_cfg.execution_backend != "shioaji":
        raise ValueError(f"Unsupported TW stock execution backend: {broker_cfg.execution_backend}")

    from poseidon.broker.shioaji_adapter import ShioajiBrokerAdapter

    broker = ShioajiBrokerAdapter(
        api_key=broker_cfg.shioaji_api_key,
        secret_key=broker_cfg.shioaji_secret_key,
        simulation=broker_cfg.shioaji_simulation,
        ca_cert_path=broker_cfg.shioaji_ca_cert_path,
        ca_password=broker_cfg.shioaji_ca_password,
        person_id=broker_cfg.shioaji_person_id,
    )
    broker.login()
    return broker


@celery_app.task(name="poseidon.workers.cpu_tasks.fetch_market_data")
def fetch_market_data(market: str, interval: str, symbol: str | None = None) -> dict:
    """Removed in Phase 61 -- data fetching moved to Thalassa.

    Task name kept so any stale Beat schedule entry does not crash the worker.
    """
    logger.warning("fetch_market_data: removed in Phase 61 — data fetching is now handled by Thalassa")
    return {"skipped": "removed_phase_61", "market": market, "interval": interval}


@celery_app.task(
    name="poseidon.workers.cpu_tasks.run_backtest_task",
    bind=True,
    max_retries=0,
)
def run_backtest_task(
    self,
    strategy_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 1_000_000.0,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
    fill_model: str | None = None,
    include_funding: bool = False,
) -> dict:
    """Run a backtest for an existing strategy.

    Loads the strategy from the DB, fetches OHLCV data, runs BacktestRunner,
    and persists results to BacktestRecord.

    Args:
        strategy_id: UUID string of the strategy to backtest.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).
        initial_capital: Starting capital for the backtest.
        sizing_mode: Position sizing mode (fixed_pct, fixed_notional, vol_target, fixed_risk).
        sizing_params: Additional sizing parameters dict.
        fill_model: Fill model for limit orders ('optimistic' or 'pessimistic').
        include_funding: Whether to include funding rate settlement costs.

    Returns:
        Dict with backtest_id, status, and trade_count.
    """
    backtest_id = uuid.uuid4()
    with db_session() as session:
        from poseidon.data.remote_repository import RemoteDataRepository
        repo = RemoteDataRepository.from_settings()
        try:
            # Load strategy record from DB
            sid = uuid.UUID(strategy_id)
            record = session.get(StrategyRecord, sid)
            if not record:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Reconstruct strategy object
            if record.strategy_type == "liquidity_sweep":
                raise ValueError("liquidity_sweep strategy archived in v15.0 -- see _archived/")
            elif record.strategy_type == "rule":
                strategy = RuleStrategy(config=record.config, strategy_id=record.id)
            elif record.strategy_type == "voting":
                strategy = VotingStrategy(config=record.config, strategy_id=record.id)
            elif record.strategy_type == "model":
                # Route to GPU worker which has model dependencies
                from poseidon.workers.gpu_tasks import run_model_backtest
                result = run_model_backtest.delay(
                    strategy_id, start_date, end_date, initial_capital,
                    sizing_mode, sizing_params,
                )
                logger.info("Routed model backtest to GPU worker: %s", result.id)
                return {"backtest_id": result.id, "status": "routed_to_gpu", "trade_count": 0}
            elif record.strategy_type == "portfolio_strategy":
                from poseidon.backtest.portfolio_backtester import PortfolioBacktester
                from poseidon.backtest.schemas import BacktestResult

                parsed_start = datetime.fromisoformat(start_date) if start_date else None
                parsed_end = datetime.fromisoformat(end_date) if end_date else None

                strategy = _build_portfolio_strategy(record, repo)
                symbols = list(
                    dict.fromkeys(
                        (record.config or {}).get("symbols")
                        or ([record.symbol] if record.symbol else [])
                    )
                )
                if not symbols:
                    raise ValueError("portfolio_strategy config must include at least one symbol")

                ohlcv_dict: dict[str, pd.DataFrame] = {}
                for symbol in symbols:
                    symbol_ohlcv = repo.read_ohlcv(
                        symbol,
                        record.market,
                        record.interval,
                        start=parsed_start,
                        end=parsed_end,
                    )
                    if not symbol_ohlcv.empty:
                        ohlcv_dict[symbol] = symbol_ohlcv

                if not ohlcv_dict:
                    raise ValueError(f"No OHLCV data for portfolio strategy {record.id}")

                if parsed_start is None:
                    parsed_start = min(
                        pd.Timestamp(df.index.min()).to_pydatetime()
                        for df in ohlcv_dict.values()
                    )
                if parsed_end is None:
                    parsed_end = max(
                        pd.Timestamp(df.index.max()).to_pydatetime()
                        for df in ohlcv_dict.values()
                    )

                backtester = PortfolioBacktester(
                    cost_model=COST_MODELS[record.market],
                    initial_capital=initial_capital,
                )
                portfolio_result = backtester.run(
                    strategy=strategy,
                    ohlcv_dict=ohlcv_dict,
                    start_date=parsed_start.date(),
                    end_date=parsed_end.date(),
                )

                trade_records = _serialize_portfolio_trades(portfolio_result.trades)
                equity_curve = _serialize_portfolio_equity_curve(
                    portfolio_result.equity_curve
                )
                bt_config = BacktestConfig(
                    strategy_type=record.strategy_type,
                    symbol=record.symbol,
                    market=record.market,
                    interval=record.interval,
                    initial_capital=initial_capital,
                    start_date=parsed_start,
                    end_date=parsed_end,
                    strategy_params=record.config,
                    sizing_mode=sizing_mode,
                    sizing_params=sizing_params or {},
                )
                result_model = BacktestResult(
                    config=bt_config,
                    metrics=portfolio_result.metrics,
                    trade_count=len(trade_records),
                    equity_curve_length=len(equity_curve),
                    status=portfolio_result.status,
                    error_message=portfolio_result.error_message,
                    trades=portfolio_result.trades,
                )

                repo = BacktestRepository(session)
                repo.save_result(
                    config=bt_config,
                    result=result_model,
                    trades=trade_records,
                    equity_curve=equity_curve,
                    strategy_id=sid,
                    completed_at=datetime.now(timezone.utc),
                )
                session.commit()

                return {
                    "backtest_id": str(result_model.backtest_id),
                    "status": portfolio_result.status,
                    "trade_count": len(trade_records),
                }
            else:
                raise ValueError(f"Unknown strategy_type: {record.strategy_type!r}")

            # Parse date filters
            parsed_start = datetime.fromisoformat(start_date) if start_date else None
            parsed_end = datetime.fromisoformat(end_date) if end_date else None

            # Load OHLCV data
            ohlcv_df = repo.read_ohlcv(
                record.symbol, record.market, record.interval,
                start=parsed_start, end=parsed_end,
            )
            if ohlcv_df.empty:
                raise ValueError(
                    f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}"
                )

            # Build pipeline components
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            cost_model = COST_MODELS[record.market]
            sizing_cfg = SizingConfig(
                mode=SizingMode(sizing_mode),
                **(sizing_params or {}),
            )

            # Resolve fill model enum (Phase 53 API-01)
            from poseidon.backtest.pending_orders import FillModel

            fill_model_enum = FillModel(fill_model) if fill_model else None

            # Load funding rates if requested and market is crypto_perp (Phase 53 API-01)
            funding_df = None
            if include_funding and record.market == "crypto_perp":
                from poseidon.models.funding_rate import FundingRateRecord

                rows = (
                    session.query(FundingRateRecord)
                    .filter(FundingRateRecord.symbol == record.symbol)
                    .order_by(FundingRateRecord.time)
                    .all()
                )
                if rows:
                    funding_df = pd.DataFrame(
                        [{"datetime": r.time, "funding_rate": r.funding_rate} for r in rows]
                    )
                    funding_df.index = pd.to_datetime(funding_df["datetime"], utc=True)
                    funding_df = funding_df.drop(columns=["datetime"])

            # Run backtest
            runner = BacktestRunner(
                strategy=strategy,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
                initial_capital=initial_capital,
                sizing_config=sizing_cfg,
                fill_model=fill_model_enum,
                include_funding=include_funding,
                funding_rates=funding_df,
            )
            result = runner.run(ohlcv_df)
            backtest_id = result.backtest_id

            # Build BacktestConfig for repository persistence
            bt_config = BacktestConfig(
                strategy_type=record.strategy_type,
                symbol=record.symbol,
                market=record.market,
                interval=record.interval,
                initial_capital=initial_capital,
                start_date=parsed_start,
                end_date=parsed_end,
                strategy_params=record.config,
            )

            # Persist via BacktestRepository (trades + equity curve included)
            repo = BacktestRepository(session)
            repo.save_result(
                config=bt_config,
                result=result,
                trades=runner.portfolio.trades if runner.portfolio else [],
                equity_curve=runner.portfolio.equity_curve if runner.portfolio else [],
                strategy_id=sid,
                completed_at=datetime.now(timezone.utc),
            )
            session.commit()

            logger.info(
                "Backtest completed: %s (strategy=%s, trades=%d)",
                backtest_id, strategy_id, result.trade_count,
            )
            return {
                "backtest_id": str(backtest_id),
                "status": result.status,
                "trade_count": result.trade_count,
            }

        except Exception as exc:
            # Persist error state
            try:
                error_record = BacktestRecord(
                    id=backtest_id,
                    strategy_id=uuid.UUID(strategy_id) if strategy_id else None,
                    strategy_type="unknown",
                    symbol="",
                    market="",
                    interval="1d",
                    config={},
                    status="failed",
                    error_message=str(exc),
                )
                session.rollback()
                session.add(error_record)
                session.commit()
            except Exception:
                logger.exception("Failed to persist error backtest record")
            logger.exception("Backtest task failed for strategy %s", strategy_id)
            raise


@celery_app.task(
    name="poseidon.workers.cpu_tasks.run_dual_mode_task",
    bind=True,
    max_retries=0,
)
def run_dual_mode_task(
    self,
    strategy_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 1_000_000.0,
    include_funding: bool = False,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
) -> dict:
    """Run dual-mode fill comparison for a strategy (Phase 53 API-03).

    Runs the same strategy with both OPTIMISTIC and PESSIMISTIC fill models
    via run_dual_mode_comparison(), returns serialized DualModeResult.

    Args:
        strategy_id: UUID string of the strategy to compare.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).
        initial_capital: Starting capital for the backtest.
        include_funding: Whether to include funding rate settlement costs.
        sizing_mode: Position sizing mode.
        sizing_params: Additional sizing parameters dict.

    Returns:
        Dict with optimistic_metrics, pessimistic_metrics, delta_metrics, is_viable.
    """
    from poseidon.backtest.comparison import run_dual_mode_comparison

    with db_session() as session:
        from poseidon.data.remote_repository import RemoteDataRepository
        repo = RemoteDataRepository.from_settings()
        try:
            sid = uuid.UUID(strategy_id)
            record = session.get(StrategyRecord, sid)
            if not record:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Build strategy factory based on strategy_type (Phase 53 API-03)
            if record.strategy_type == "voting":
                strategy_factory = lambda: VotingStrategy(config=record.config, strategy_id=record.id)  # noqa: E731
            elif record.strategy_type == "liquidity_sweep":
                raise ValueError("liquidity_sweep strategy archived in v15.0 -- see _archived/")
            elif record.strategy_type == "rule":
                strategy_factory = lambda: RuleStrategy(config=record.config, strategy_id=record.id)  # noqa: E731
            elif record.strategy_type == "model":
                raise ValueError("Dual-mode comparison not supported for model strategies")
            else:
                raise ValueError(f"Unknown strategy_type: {record.strategy_type!r}")

            # Parse date filters
            parsed_start = datetime.fromisoformat(start_date) if start_date else None
            parsed_end = datetime.fromisoformat(end_date) if end_date else None

            # Load OHLCV data
            ohlcv_df = repo.read_ohlcv(
                record.symbol, record.market, record.interval,
                start=parsed_start, end=parsed_end,
            )
            if ohlcv_df.empty:
                raise ValueError(
                    f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}"
                )

            # Build pipeline components
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            cost_model = COST_MODELS[record.market]
            sizing_cfg = SizingConfig(
                mode=SizingMode(sizing_mode),
                **(sizing_params or {}),
            )

            # Load funding rates if requested (Phase 53 API-03)
            funding_df = None
            if include_funding and record.market == "crypto_perp":
                from poseidon.models.funding_rate import FundingRateRecord

                rows = (
                    session.query(FundingRateRecord)
                    .filter(FundingRateRecord.symbol == record.symbol)
                    .order_by(FundingRateRecord.time)
                    .all()
                )
                if rows:
                    funding_df = pd.DataFrame(
                        [{"datetime": r.time, "funding_rate": r.funding_rate} for r in rows]
                    )
                    funding_df.index = pd.to_datetime(funding_df["datetime"], utc=True)
                    funding_df = funding_df.drop(columns=["datetime"])

            # Run dual-mode comparison
            dual_result = run_dual_mode_comparison(
                strategy_factory=strategy_factory,
                ohlcv=ohlcv_df,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
                initial_capital=initial_capital,
                sizing_config=sizing_cfg,
                include_funding=include_funding,
                db_session=session,
                funding_rates=funding_df,
            )

            logger.info(
                "Dual-mode comparison completed: strategy=%s, is_viable=%s",
                strategy_id, dual_result.is_viable,
            )
            return {
                "optimistic_metrics": dual_result.optimistic_result.metrics,
                "pessimistic_metrics": dual_result.pessimistic_result.metrics,
                "delta_metrics": dual_result.delta_metrics,
                "is_viable": dual_result.is_viable,
            }

        except Exception:
            logger.exception("Dual-mode task failed for strategy %s", strategy_id)
            raise


@celery_app.task(
    name="poseidon.workers.cpu_tasks.run_optimization_task",
    bind=True,
    max_retries=0,
)
def run_optimization_task(
    self,
    strategy_id: str,
    param_grid: dict,
    method: str = "grid",
    n_trials: int = 50,
    target_metric: str = "sharpe_ratio",
    start_date: str | None = None,
    end_date: str | None = None,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
) -> dict:
    """Run parameter optimization for an existing strategy.

    Loads the strategy template from the DB, runs grid or Bayesian optimization
    over the param_grid, and persists the best result to BacktestRecord.

    Args:
        strategy_id: UUID string of the strategy to optimize.
        param_grid: Parameter grid/space to search.
        method: Optimization method ("grid" or "bayesian").
        n_trials: Number of trials for Bayesian optimization.
        target_metric: Metric to maximize (default: sharpe_ratio).
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).

    Returns:
        Dict with trial count, best params, and best metric value.
    """
    with db_session() as session:
        from poseidon.data.remote_repository import RemoteDataRepository
        repo = RemoteDataRepository.from_settings()
        try:
            # Load strategy record from DB
            sid = uuid.UUID(strategy_id)
            record = session.get(StrategyRecord, sid)
            if not record:
                raise ValueError(f"Strategy {strategy_id} not found")

            if record.strategy_type != "rule":
                raise NotImplementedError(
                    f"Optimization for strategy_type={record.strategy_type!r} not yet supported"
                )

            # Parse date filters
            parsed_start = datetime.fromisoformat(start_date) if start_date else None
            parsed_end = datetime.fromisoformat(end_date) if end_date else None

            # Load OHLCV data
            ohlcv_df = repo.read_ohlcv(
                record.symbol, record.market, record.interval,
                start=parsed_start, end=parsed_end,
            )
            if ohlcv_df.empty:
                raise ValueError(
                    f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}"
                )

            # Build pipeline components
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            cost_model = COST_MODELS[record.market]
            sizing_cfg = SizingConfig(
                mode=SizingMode(sizing_mode),
                **(sizing_params or {}),
            )

            # Strategy factory: merges param_grid values into the base config
            base_config = dict(record.config) if record.config else {}

            def strategy_factory(params: dict) -> RuleStrategy:
                merged = {**base_config, **params}
                return RuleStrategy(config=merged, strategy_id=record.id)

            # Run optimization
            if method == "grid":
                optimizer = GridSearchOptimizer(
                    feature_engine=feature_engine,
                    risk_engine=risk_engine,
                    cost_model=cost_model,
                    initial_capital=1_000_000.0,
                    sizing_config=sizing_cfg,
                )
                trials = optimizer.optimize(
                    strategy_factory=strategy_factory,
                    ohlcv=ohlcv_df,
                    param_grid=param_grid,
                    metric=target_metric,
                )
            elif method == "bayesian":
                optimizer = BayesianOptimizer(
                    feature_engine=feature_engine,
                    risk_engine=risk_engine,
                    cost_model=cost_model,
                    initial_capital=1_000_000.0,
                    sizing_config=sizing_cfg,
                )
                # Convert param_grid to Bayesian param_space format
                # Expected input: {"param": [low, high, "int"|"float"]}
                param_space = {}
                for key, spec in param_grid.items():
                    if isinstance(spec, (list, tuple)) and len(spec) == 3:
                        param_space[key] = (spec[0], spec[1], spec[2])
                    else:
                        raise ValueError(
                            f"Bayesian param_grid[{key!r}] must be [low, high, type], "
                            f"got {spec!r}"
                        )
                trials = optimizer.optimize(
                    strategy_factory=strategy_factory,
                    ohlcv=ohlcv_df,
                    param_space=param_space,
                    n_trials=n_trials,
                    metric=target_metric,
                )
            else:
                raise ValueError(f"Unknown optimization method: {method!r}. Use 'grid' or 'bayesian'.")

            # Persist best result as a BacktestRecord with walk_forward metadata
            best = trials[0] if trials else None
            if best:
                bt_record = BacktestRecord(
                    id=uuid.uuid4(),
                    strategy_id=sid,
                    strategy_type=record.strategy_type,
                    symbol=record.symbol,
                    market=record.market,
                    interval=record.interval,
                    config={
                        "optimization_method": method,
                        "param_grid": param_grid,
                        "target_metric": target_metric,
                        "strategy_params": record.config,
                    },
                    metrics=best.metrics,
                    walk_forward={
                        "best_params": best.params,
                        "best_metric_value": best.metric_value,
                        "total_trials": len(trials),
                    },
                    status="completed",
                    completed_at=datetime.now(timezone.utc),
                )
                session.add(bt_record)
                session.commit()

            logger.info(
                "Optimization completed: strategy=%s method=%s trials=%d best_metric=%.4f",
                strategy_id, method, len(trials),
                best.metric_value if best else 0.0,
            )
            return {
                "trials": len(trials),
                "best_params": best.params if best else {},
                "best_metric": best.metric_value if best else 0.0,
            }

        except Exception as exc:
            logger.exception("Optimization task failed for strategy %s", strategy_id)
            raise


@celery_app.task(name="poseidon.workers.cpu_tasks.compute_var_snapshot")
def compute_var_snapshot(method: str = "all") -> dict:
    """Compute VaR snapshot(s) and cache to Redis + TimescaleDB.

    Args:
        method: "parametric", "historical", "cornish_fisher", or "all"

    Returns:
        Dict with status and computed methods.
    """
    import msgpack
    import numpy as np

    from poseidon.models.var_snapshot import VaRSnapshot
    from poseidon.risk.portfolio import VirtualPortfolio
    from poseidon.risk.var.calculators import VaRCalculator
    from poseidon.risk.var.covariance import load_cached_covariance
    from poseidon.risk.var.returns import align_returns, compute_returns
    from poseidon.risk.var.types import VaRMethod

    redis_client = get_redis("cache")
    with db_session() as db:
        from poseidon.data.remote_repository import RemoteDataRepository
        repo = RemoteDataRepository.from_settings()
        # 1. Rebuild portfolio from DB
        portfolio = VirtualPortfolio()
        portfolio.rebuild_from_db(db)
        if portfolio.open_position_count == 0:
            logger.info("No open positions, skipping VaR computation")
            return {"status": "skipped", "reason": "no open positions"}

        weights = portfolio.weights()
        symbols = portfolio.position_symbols()
        as_of = datetime.now(timezone.utc)
        portfolio_value = portfolio.total_exposure()

        calculator = VaRCalculator()
        results = []

        # 2. Load cached covariance
        cached = load_cached_covariance(redis_client)
        if cached is None:
            logger.warning("No cached covariance matrix, skipping VaR computation")
            return {"status": "skipped", "reason": "no cached covariance matrix"}
        cov_symbols, cov_matrix, cov_meta = cached

        # 3. Load portfolio returns for historical/cornish-fisher methods
        # TODO: Load real aligned returns from OHLCV storage once portfolio
        # return loading is fully wired. For now, compute from DB if possible.
        portfolio_returns = np.array([])
        try:
            return_series = {}
            for sym in symbols:
                ohlcv_df = repo.read_ohlcv(sym, market="", interval="1d",
                                           start=as_of - timedelta(days=settings.var_lookback_days),
                                           end=as_of)
                if not ohlcv_df.empty and "close" in ohlcv_df.columns:
                    close = ohlcv_df["close"]
                    if hasattr(close, "index"):
                        ret = compute_returns(close, as_of=as_of)
                        if not ret.empty:
                            return_series[sym] = ret
            if return_series:
                aligned = align_returns(return_series, as_of=as_of)
                if not aligned.empty:
                    portfolio_returns = calculator.compute_portfolio_returns(aligned, weights)
        except Exception:
            logger.warning("Could not load portfolio returns, historical/CF may use empty array")

        methods_to_compute = (
            [VaRMethod.PARAMETRIC, VaRMethod.HISTORICAL, VaRMethod.CORNISH_FISHER]
            if method == "all"
            else [VaRMethod(method)]
        )

        for m in methods_to_compute:
            result = None
            if m == VaRMethod.PARAMETRIC:
                result = calculator.parametric(weights, cov_matrix, portfolio_value, as_of)
            elif m == VaRMethod.HISTORICAL:
                if len(portfolio_returns) >= settings.var_min_observations:
                    result = calculator.historical_simulation(
                        portfolio_returns, portfolio_value, as_of
                    )
                else:
                    logger.warning(
                        "Insufficient portfolio returns (%d) for historical VaR, skipping",
                        len(portfolio_returns),
                    )
                    continue
            elif m == VaRMethod.CORNISH_FISHER:
                if len(portfolio_returns) >= settings.var_min_observations:
                    result = calculator.cornish_fisher(
                        weights, cov_matrix, portfolio_returns,
                        portfolio_value, as_of,
                    )
                else:
                    logger.warning(
                        "Insufficient portfolio returns (%d) for Cornish-Fisher VaR, skipping",
                        len(portfolio_returns),
                    )
                    continue

            if result is None:
                continue

            results.append(result)

            # 4. Cache latest to Redis (poseidon:var:latest:{method})
            snapshot_data = {
                "method": result.method,
                "var_95": result.var_95,
                "var_99": result.var_99,
                "cvar_95": result.cvar_95,
                "cvar_99": result.cvar_99,
                "portfolio_value": result.portfolio_value,
                "as_of": result.as_of.isoformat(),
                "computed_at": result.computed_at.isoformat(),
            }
            redis_key = f"poseidon:var:latest:{result.method}"
            redis_client.set(
                redis_key,
                msgpack.packb(snapshot_data, use_bin_type=True),
                ex=settings.var_cache_ttl,
            )
            logger.info("Cached VaR snapshot at %s (var_95=%.6f)", redis_key, result.var_95)

            # 5. Store in TimescaleDB
            db.add(VaRSnapshot(
                time=result.as_of,
                method=result.method,
                var_95=result.var_95,
                var_99=result.var_99,
                cvar_95=result.cvar_95,
                cvar_99=result.cvar_99,
                portfolio_value=result.portfolio_value,
                holding_period=result.holding_period,
                details=result.details,
            ))

        db.commit()
        computed_methods = [r.method for r in results]
        logger.info("VaR computation completed: methods=%s", computed_methods)
        return {"status": "ok", "methods": computed_methods}


@celery_app.task(name="poseidon.workers.cpu_tasks.compute_mc_var")
def compute_mc_var() -> dict:
    """Compute Monte Carlo VaR using Cholesky decomposition (per D-05, D-06, D-07).

    Uses cached covariance matrix (never recomputes), rebuilds portfolio for
    weights, and stores result in VaR snapshot table + Redis cache.

    Returns:
        Dict with status and result summary.
    """
    import msgpack
    import numpy as np

    from poseidon.models.var_snapshot import VaRSnapshot
    from poseidon.risk.portfolio import VirtualPortfolio
    from poseidon.risk.var.calculators import VaRCalculator
    from poseidon.risk.var.covariance import load_cached_covariance

    redis_client = get_redis("cache")
    with db_session() as db:
        # 1. Load cached covariance (per D-06: never recompute inline)
        cached = load_cached_covariance(redis_client)
        if cached is None:
            logger.warning("No cached covariance matrix, skipping MC VaR")
            return {"status": "skipped", "reason": "no cached covariance matrix"}
        cov_symbols, cov_matrix, cov_meta = cached

        # 2. Rebuild portfolio from DB
        portfolio = VirtualPortfolio()
        portfolio.rebuild_from_db(db)
        if portfolio.open_position_count == 0:
            logger.info("No open positions, skipping MC VaR computation")
            return {"status": "skipped", "reason": "no open positions"}

        weights = portfolio.weights()
        portfolio_value = portfolio.total_exposure()
        as_of = datetime.now(timezone.utc)

        # 3. Compute Monte Carlo VaR
        calculator = VaRCalculator()
        result = calculator.monte_carlo(
            weights=weights,
            cov_matrix=cov_matrix,
            portfolio_value=portfolio_value,
            as_of=as_of,
            n_simulations=settings.mc_simulations,
        )

        # 4. Cache latest to Redis
        snapshot_data = {
            "method": result.method,
            "var_95": result.var_95,
            "var_99": result.var_99,
            "cvar_95": result.cvar_95,
            "cvar_99": result.cvar_99,
            "portfolio_value": result.portfolio_value,
            "as_of": result.as_of.isoformat(),
            "computed_at": result.computed_at.isoformat(),
        }
        redis_key = "poseidon:var:latest:monte_carlo"
        redis_client.set(
            redis_key,
            msgpack.packb(snapshot_data, use_bin_type=True),
            ex=settings.var_cache_ttl,
        )
        logger.info("Cached MC VaR snapshot at %s (var_95=%.6f)", redis_key, result.var_95)

        # 5. Store in TimescaleDB
        db.add(VaRSnapshot(
            time=result.as_of,
            method=result.method,
            var_95=result.var_95,
            var_99=result.var_99,
            cvar_95=result.cvar_95,
            cvar_99=result.cvar_99,
            portfolio_value=result.portfolio_value,
            holding_period=result.holding_period,
            details=result.details,
        ))
        db.commit()

        logger.info("MC VaR computation completed: var_95=%.6f var_99=%.6f", result.var_95, result.var_99)
        return {
            "status": "ok",
            "method": "monte_carlo",
            "var_95": result.var_95,
            "var_99": result.var_99,
        }


@celery_app.task(name="poseidon.workers.cpu_tasks.update_covariance_matrix")
def update_covariance_matrix() -> dict:
    """Recompute and cache the covariance matrix from aligned returns.

    Loads OHLCV close prices for all portfolio symbols, computes log returns,
    aligns cross-market, computes covariance, and caches to Redis.

    Returns:
        Dict with status and symbol count.
    """
    import numpy as np

    from poseidon.risk.portfolio import VirtualPortfolio
    from poseidon.risk.var.covariance import cache_covariance, compute_covariance
    from poseidon.risk.var.returns import align_returns, compute_returns

    redis_client = get_redis("cache")
    with db_session() as db:
        from poseidon.data.remote_repository import RemoteDataRepository
        repo = RemoteDataRepository.from_settings()
        # 1. Get active portfolio symbols
        portfolio = VirtualPortfolio()
        portfolio.rebuild_from_db(db)
        if portfolio.open_position_count == 0:
            logger.info("No open positions, skipping covariance update")
            return {"status": "skipped", "reason": "no open positions"}

        symbols = portfolio.position_symbols()
        as_of = datetime.now(timezone.utc)

        # 2. Load OHLCV close prices and compute returns for each symbol
        return_series = {}
        for sym in symbols:
            start = as_of - timedelta(days=settings.var_lookback_days)
            ohlcv_df = repo.read_ohlcv(sym, market="", interval="1d",
                                       start=start, end=as_of)
            if ohlcv_df.empty:
                logger.warning("No OHLCV data for %s, skipping in covariance", sym)
                continue
            if "close" not in ohlcv_df.columns:
                logger.warning("No close column for %s, skipping in covariance", sym)
                continue
            close = ohlcv_df["close"]
            ret = compute_returns(close, as_of=as_of)
            if not ret.empty:
                return_series[sym] = ret

        if len(return_series) < 2:
            logger.warning("Need at least 2 symbols for covariance, got %d", len(return_series))
            return {"status": "skipped", "reason": f"insufficient symbols ({len(return_series)})"}

        # 3. Align returns cross-market
        aligned = align_returns(return_series, as_of=as_of)
        if aligned.empty or len(aligned) < settings.var_min_observations:
            logger.warning(
                "Insufficient aligned observations (%d) for covariance",
                len(aligned),
            )
            return {"status": "skipped", "reason": "insufficient aligned observations"}

        # 4. Compute covariance
        cov_matrix = compute_covariance(aligned, min_observations=settings.var_min_observations)

        # 5. Cache to Redis
        cov_symbols = list(aligned.columns)
        cache_covariance(redis_client, cov_symbols, cov_matrix, as_of)
        logger.info("Covariance matrix updated: %d symbols, %d observations", len(cov_symbols), len(aligned))
        return {"status": "ok", "symbols": len(cov_symbols), "observations": len(aligned)}


@celery_app.task(name="poseidon.workers.cpu_tasks.autoresearch_run", bind=True)
def autoresearch_run(self, search_config: dict, markets: list[dict]) -> dict:
    """Run autonomous parameter search across markets (D-09).

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

    started_at = datetime.now(timezone.utc)
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

            # D-12: graceful stop check via Redis flag
            def check_stop() -> bool:
                return bool(redis_client.get(f"autoresearch:stop:{task_id}"))

            # D-11: heartbeat via Celery task state update
            def update_progress(current: int, total: int, symbol: str) -> None:
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current_market": current + 1,
                        "total_markets": total,
                        "symbol": symbol,
                    },
                )

            # Phase 45: optional ML model for sub-signal search (D-20)
            mv_id = search_config.get("model_version_id")

            # Phase 69: extract new dispatch fields from search_config
            strategy_type = search_config.get("strategy_type", "voting")
            feature_names = search_config.get("feature_names")
            base_config_json = search_config.get("base_config_json")

            # D-11: regime_router special path -- does NOT use AutoResearchRunner
            if strategy_type == "regime_router":
                from poseidon.backtest.regime_search import RegimeSearchPipeline, RegimeSearchConfig
                from poseidon.backtest.experiment_tracker import ExperimentTracker as ET
                from poseidon.data.feature_engine import FeatureOrchestrator
                from poseidon.risk.engine import RiskEngine
                from poseidon.backtest.cost_model import COST_MODELS
                from poseidon.data.remote_repository import RemoteDataRepository

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
                        cost_model = CostModel(market=spec.market, buy_commission_rate=0.0,
                                               sell_commission_rate=0.0, tax_rate=0.0,
                                               slippage_pct=0.0, slippage_ticks=0.0,
                                               description=f"Default for {spec.market}")

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
                        ohlcv=ohlcv, base_config=base_config, regime_model=regime_model,
                        config=regime_cfg,
                        market=spec.market, symbol=spec.symbol, interval=spec.interval,
                    )
                    regime_results_all.append({"spec": spec_dict, "result": result})

                completed_at = datetime.now(timezone.utc)
                return {
                    "status": "stopped" if check_stop() else "completed",
                    "markets_processed": len(regime_results_all),
                    "report": {"regime_results": regime_results_all},
                }

            # D-15: FACTORY_REGISTRY dict for standard AutoResearchRunner path.
            # voting maps to None (VotingStrategyFactory is the default when strategy_factory=None).
            # rule uses a lambda to capture feature_names at dispatch time (constructor arg).
            # model is initialised with available_models=None; AutoResearchRunner's per-market loop
            #   (runner.py:164-173, Phase 46 pattern) calls list_ready_models(market) and passes
            #   available_models into the pipeline, which reaches build_trial_factory() dynamically.
            from poseidon.backtest.rule_strategy_factory import RuleStrategyFactory
            from poseidon.backtest.model_strategy_factory import ModelStrategyFactory

            FACTORY_REGISTRY = {
                "voting": lambda: None,
                "rule": lambda: RuleStrategyFactory(
                    feature_names=feature_names or [],
                    direction_mode=search_config.get("strategy_mode", "bidirectional"),
                ),
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

            # Generate report (D-15, D-16)
            completed_at = datetime.now(timezone.utc)
            tracker = ExperimentTracker(db)
            study_names = [
                r.search_result.study_name
                for r in results
                if r.search_result is not None
            ]
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
            # D-12: clean up stop flag
            try:
                redis_client.delete(f"autoresearch:stop:{task_id}")
            except Exception:
                pass


@celery_app.task(name="poseidon.workers.cpu_tasks.run_stress_test")
def run_stress_test(
    scenario_name: str, custom_shocks: dict | None = None
) -> dict:
    """Run stress test scenario asynchronously (per D-12).

    Supports both named scenarios (from JSON config) and ad-hoc hypothetical
    scenarios (custom_shocks dict). Results are returned as JSON-serializable
    dicts for the Celery result backend.

    Args:
        scenario_name: Name of the scenario JSON file (without .json).
        custom_shocks: Optional dict of market -> shock factor for ad-hoc
            hypothetical scenarios. When provided, scenario_name is used
            as the label but shocks come from this dict.

    Returns:
        Dict with StressTestResult fields (JSON-serializable).
    """
    from datetime import timedelta

    import numpy as np

    from poseidon.risk.portfolio import VirtualPortfolio
    from poseidon.risk.stress.engine import StressTestEngine
    from poseidon.risk.stress.types import ScenarioConfig
    from poseidon.risk.var.calculators import VaRCalculator
    from poseidon.risk.var.covariance import load_cached_covariance
    from poseidon.risk.var.returns import align_returns, compute_returns

    redis_client = get_redis("cache")
    with db_session() as db:
        from poseidon.data.remote_repository import RemoteDataRepository
        repo = RemoteDataRepository.from_settings()
        # 1. Rebuild portfolio from DB
        portfolio = VirtualPortfolio()
        portfolio.rebuild_from_db(db)
        if portfolio.open_position_count == 0:
            return {"status": "skipped", "reason": "no open positions"}

        weights = portfolio.weights()
        symbols = portfolio.position_symbols()
        as_of = datetime.now(timezone.utc)
        portfolio_value = portfolio.total_exposure()

        # 2. Load cached covariance
        cached = load_cached_covariance(redis_client)
        if cached is None:
            return {"status": "skipped", "reason": "no cached covariance matrix"}
        cov_symbols, cov_matrix, _cov_meta = cached
        cov_matrix = np.array(cov_matrix)

        # 3. Load aligned returns for historical scenarios
        aligned_returns = None
        try:
            return_series = {}
            for sym in symbols:
                ohlcv_df = repo.read_ohlcv(
                    sym, market="", interval="1d",
                    start=as_of - timedelta(days=settings.var_lookback_days),
                    end=as_of,
                )
                if not ohlcv_df.empty and "close" in ohlcv_df.columns:
                    ret = compute_returns(ohlcv_df["close"], as_of=as_of)
                    if not ret.empty:
                        return_series[sym] = ret
            if return_series:
                aligned_returns = align_returns(return_series, as_of=as_of)
        except Exception:
            logger.warning("Could not load aligned returns for stress test")

        # 4. Create engine and run scenario
        calculator = VaRCalculator()
        scenarios_dir = str(
            __import__("pathlib").Path(__file__).resolve().parents[3]
            / "config"
            / "stress_scenarios"
        )
        engine = StressTestEngine(calculator, scenarios_dir=scenarios_dir)

        if custom_shocks is not None:
            # Ad-hoc hypothetical scenario
            config = ScenarioConfig(
                name=scenario_name,
                type="hypothetical",
                description=f"Custom hypothetical: {scenario_name}",
                shocks=custom_shocks,
            )
            result = engine.run_config(
                config, weights, symbols, cov_matrix,
                aligned_returns=aligned_returns,
                portfolio_value=portfolio_value,
            )
        else:
            # Named scenario from JSON config
            result = engine.run_scenario(
                scenario_name, weights, symbols, cov_matrix,
                aligned_returns=aligned_returns,
                portfolio_value=portfolio_value,
            )

        # 5. Convert to JSON-serializable dict
        return {
            "status": "completed",
            "scenario_name": result.scenario_name,
            "scenario_type": result.scenario_type,
            "portfolio_pnl": result.portfolio_pnl,
            "worst_case_loss": result.worst_case_loss,
            "var_result": (
                {
                    "method": result.var_result.method,
                    "var_95": result.var_result.var_95,
                    "var_99": result.var_result.var_99,
                    "cvar_95": result.var_result.cvar_95,
                    "cvar_99": result.var_result.cvar_99,
                    "portfolio_value": result.var_result.portfolio_value,
                }
                if result.var_result
                else None
            ),
            "details": result.details,
            "computed_at": (
                result.computed_at.isoformat() if result.computed_at else None
            ),
        }


@celery_app.task(name="poseidon.workers.cpu_tasks.evaluate_active_strategies")
def evaluate_active_strategies(
    fetch_result: dict | None = None,
    market: str | None = None,
    interval: str | None = None,
) -> dict:
    """Evaluate all active strategies for a given market/interval.

    Can be called directly via Beat schedule (with market/interval kwargs) or
    as a chain callback from fetch_market_data (with fetch_result positional arg).

    Args:
        fetch_result: Return value from fetch_market_data when used as chain callback.
        market: Market name (e.g., "crypto_spot"). Overrides fetch_result if provided.
        interval: Candle interval (e.g., "1h"). Overrides fetch_result if provided.

    Returns:
        Dict with evaluation summary.
    """
    from poseidon.risk.pipeline import SignalPipeline
    from poseidon.strategies.voting_strategy import VotingStrategy

    # Extract market/interval from fetch_result if not provided directly
    market = market or (fetch_result or {}).get("market")
    interval = interval or (fetch_result or {}).get("interval")
    if not market or not interval:
        return {"error": "market and interval required"}

    with db_session() as session:
        strategies = (
            session.query(StrategyRecord)
            .filter(
                StrategyRecord.active == True,  # noqa: E712
                StrategyRecord.market == market,
                StrategyRecord.interval == interval,
            )
            .all()
        )

        engine = FeatureEngine()
        pipeline = SignalPipeline(session)
        total_signals = 0
        errors = []

        for record in strategies:
            try:
                if record.strategy_type == "rule":
                    strategy = RuleStrategy(config=record.config, strategy_id=record.id)
                elif record.strategy_type == "voting":
                    config = {
                        **record.config,
                        "symbol": record.symbol,
                        "market": record.market,
                        "interval": record.interval,
                    }
                    strategy = VotingStrategy(config=config, strategy_id=record.id)
                else:
                    # model strategies use GPU predict path
                    continue

                features = engine.compute(record.symbol, record.market, record.interval)
                raw_signals = strategy.evaluate(features)
                for sig in raw_signals:
                    pipeline.process(sig)
                total_signals += len(raw_signals)
            except Exception:
                logger.exception("Failed to evaluate strategy %s", record.name)
                errors.append(record.name)

        return {
            "market": market,
            "interval": interval,
            "strategies_evaluated": len(strategies),
            "signals_generated": total_signals,
            "errors": errors,
        }


@celery_app.task(name="poseidon.workers.cpu_tasks.trigger_risk_update")
def trigger_risk_update(eval_result: dict | None = None) -> dict:
    """Trigger risk pipeline update after signal generation.

    Calls compute_var_snapshot("historical") directly (same CPU worker process)
    to recalculate VaR after new signals have been generated.

    Args:
        eval_result: Return value from evaluate_active_strategies (unused, for chain compat).

    Returns:
        Dict with risk update status.
    """
    compute_var_snapshot("historical")
    return {
        "risk_update": "completed",
        "var_method": "historical",
        "trigger": "signal_generation",
    }


# ---------------------------------------------------------------------------
# Phase 24: Portfolio scheduling tasks
# ---------------------------------------------------------------------------


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_monthly_rebalance")
def portfolio_monthly_rebalance(signal_id: str | None = None) -> dict:
    """Monthly rebalance -- currently disabled.

    RevenueBreakoutStrategy removed in Phase 61 (FinLab data access
    not yet available via Thalassa). Re-implement when Thalassa
    serves FinLab-equivalent data.
    """
    logger.warning(
        "portfolio_monthly_rebalance: skipped — RevenueBreakoutStrategy "
        "removed (FinLab data not available via Thalassa)"
    )
    return {"skipped": "strategy_unavailable", "reason": "RevenueBreakoutStrategy removed — Phase 61"}


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_stop_loss_monitor")
def portfolio_stop_loss_monitor() -> dict:
    """Intraday stop-loss monitor: check holdings against stop-loss thresholds.

    Runs every 5 min via beat. Self-gates to TW trading hours (01:00-05:30 UTC).
    Skips weekends. Dispatches sell orders for breached positions.
    """
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.broker.paper_adapter import PaperBrokerAdapter
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.orders.manager import OrderManager
    from poseidon.strategies.portfolio.schemas import RebalanceOrder

    now = datetime.now(timezone.utc)

    # Skip weekends
    if now.weekday() >= 5:
        return {"skipped": "weekend"}

    # Gate to TW trading hours: 01:00-05:30 UTC (09:00-13:30 UTC+8)
    current_minutes = now.hour * 60 + now.minute
    trading_start = 1 * 60  # 01:00 UTC
    trading_end = 5 * 60 + 30  # 05:30 UTC
    if current_minutes < trading_start or current_minutes > trading_end:
        return {"skipped": "outside_trading_hours"}

    position_tracker = _build_position_tracker()
    holdings = position_tracker.current_holdings()

    if not holdings:
        return {"checked": 0, "stopped_out": []}

    # Get latest prices
    prices = _get_latest_prices(list(holdings.keys()))

    stopped_symbols: list[str] = []
    sell_orders: list[RebalanceOrder] = []

    for sym, holding in holdings.items():
        if holding.stop_loss_pct is None or holding.entry_price is None:
            continue

        current_price = prices.get(sym)
        if current_price is None:
            logger.warning("No price data for %s, skipping stop-loss check", sym)
            continue

        stop_price = holding.entry_price * (1 - holding.stop_loss_pct)
        if current_price <= stop_price:
            logger.warning(
                "Stop-loss breached: %s price=%.2f <= stop=%.2f (entry=%.2f, pct=%.2f%%)",
                sym, current_price, stop_price, holding.entry_price, holding.stop_loss_pct * 100,
            )
            sell_orders.append(
                RebalanceOrder(
                    symbol=sym,
                    action="sell",
                    target_weight=0.0,
                    current_weight=holding.weight,
                    delta_weight=-holding.weight,
                )
            )
            stopped_symbols.append(sym)

    if sell_orders:
        with open("config/broker.yaml") as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
        broker = _build_tw_stock_broker(broker_cfg)
        risk_checker = OrderRiskChecker(
            position_limit_pct=0.15,
            max_exposure=1.0,
            stop_loss_pct=broker_cfg.slippage_pct,
        )
        order_manager = OrderManager(broker, risk_checker, position_tracker, SessionLocal, broker_cfg)

        results = order_manager.execute_rebalance(
            sell_orders, strategy_name="stop_loss", prices=prices, market="tw_stock",
        )

        # Create TradeLogRecords for stopped-out positions
        with db_session() as session:
            for result in results:
                if result.success:
                    sym = result.order.symbol
                    holding = holdings.get(sym)
                    if holding and holding.entry_price and holding.shares:
                        exit_price = prices.get(sym, 0.0)
                        trade_log = TradeLogRecord(
                            strategy_name="stop_loss",
                            symbol=sym,
                            market=holding.market,
                            entry_price=holding.entry_price,
                            exit_price=exit_price,
                            entry_date=holding.entry_date or now,
                            exit_date=now,
                            shares=holding.shares,
                            realized_pnl=(exit_price - holding.entry_price) * holding.shares,
                            holding_days=(now - holding.entry_date).days if holding.entry_date else 0,
                            signal_id=None,  # Stop-loss exits are not signal-triggered
                        )
                        session.add(trade_log)
            session.commit()

    logger.info(
        "portfolio_stop_loss_monitor: checked=%d stopped_out=%s",
        len(holdings),
        stopped_symbols,
    )
    return {"checked": len(holdings), "stopped_out": stopped_symbols}


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_nav_snapshot")
def portfolio_nav_snapshot() -> dict:
    """Daily NAV snapshot: record portfolio value post-close.

    Triggered daily at 06:00 UTC (14:00 UTC+8, after TW close).
    Skips weekends. Creates NavSnapshotRecord.
    """
    from datetime import date as date_type

    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.nav_snapshot import NavSnapshotRecord

    now = datetime.now(timezone.utc)

    # Skip weekends
    if now.weekday() >= 5:
        return {"skipped": "weekend"}

    position_tracker = _build_position_tracker()
    holdings = position_tracker.current_holdings()
    with open("config/broker.yaml") as bf:
        broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    initial_nav = broker_cfg.paper_initial_nav

    # Compute holdings value from latest prices
    holdings_value = 0.0
    deployed_cost = 0.0
    if holdings:
        prices = _get_latest_prices(list(holdings.keys()))
        for sym, holding in holdings.items():
            price = prices.get(sym, 0.0)
            shares = holding.shares or 0
            entry_price = holding.entry_price or 0.0
            holdings_value += shares * price
            deployed_cost += shares * entry_price

    # Cash = initial NAV minus total cost of open positions
    cash = initial_nav - deployed_cost
    total_nav = holdings_value + cash

    today = date_type.today()
    with db_session() as session:
        # Upsert: avoid duplicate if re-run on same day
        existing = (
            session.query(NavSnapshotRecord)
            .filter(NavSnapshotRecord.snapshot_date == today)
            .first()
        )
        if existing:
            existing.total_nav = total_nav
            existing.holdings_value = holdings_value
            existing.cash = cash
            existing.holdings_count = len(holdings)
        else:
            record = NavSnapshotRecord(
                snapshot_date=today,
                total_nav=total_nav,
                holdings_value=holdings_value,
                cash=cash,
                holdings_count=len(holdings),
            )
            session.add(record)
        session.commit()

    logger.info(
        "portfolio_nav_snapshot: date=%s total_nav=%.2f holdings=%d",
        today, total_nav, len(holdings),
    )
    return {"date": str(today), "total_nav": total_nav, "holdings_count": len(holdings)}


# ---------------------------------------------------------------------------
# Phase 24: Helper functions for portfolio tasks
# ---------------------------------------------------------------------------


def _build_position_tracker():
    """Build and initialize a PositionTracker."""
    from poseidon.strategies.portfolio.position_tracker import PositionTracker

    tracker = PositionTracker(SessionLocal)
    tracker.rebuild_from_db()
    return tracker


def _get_latest_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest close prices from Thalassa for given symbols.

    Returns:
        Dict mapping symbol -> latest close price.
    """
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    prices: dict[str, float] = {}
    for sym in symbols:
        latest = repo.read_ohlcv(sym, "tw_stock", "1d")
        if not latest.empty:
            prices[sym] = float(latest["close"].iloc[-1])
    return prices


# ---------------------------------------------------------------------------
# Phase 27: Perpetual contract task helpers
# ---------------------------------------------------------------------------


def _build_perp_adapter_from_db():
    """Build PerpPaperAdapter and populate positions from DB holdings.

    Each Celery task needs its own adapter instance because in-memory
    _positions dict is NOT shared across tasks (pitfall #1 from research).
    """
    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter, PerpPosition, calc_liquidation_price
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord

    adapter = PerpPaperAdapter(SessionLocal)
    with db_session() as session:
        perp_holdings = (
            session.query(PortfolioHoldingRecord)
            .filter(
                PortfolioHoldingRecord.closed == False,  # noqa: E712
                PortfolioHoldingRecord.market == "crypto_perp",
            )
            .all()
        )
        for h in perp_holdings:
            leverage = adapter._leverage_per_symbol.get(
                h.symbol, adapter._default_leverage
            )
            entry_price = h.entry_price or 0.0
            quantity = h.shares or 0.0
            side = h.side or "long"
            liq_price = calc_liquidation_price(entry_price, leverage, side)
            adapter._positions[h.symbol] = PerpPosition(
                symbol=h.symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                leverage=leverage,
                margin=entry_price * quantity / leverage,
                liquidation_price=liq_price,
            )

    return adapter


def _get_perp_mark_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest perpetual mark prices from Thalassa."""
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    prices = {}
    for sym in symbols:
        latest = repo.read_latest_price(sym)
        if latest is not None:
            prices[sym] = float(latest)
    return prices


# ---------------------------------------------------------------------------
# Phase 27: Perpetual contract Celery tasks
# ---------------------------------------------------------------------------


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_liquidation_monitor")
def perp_liquidation_monitor() -> dict:
    """Every 1 min: check perp margin ratios, close all on breach (D-01, D-02, D-03).

    24/7 -- NO weekend/holiday skip. Checks margin ratio for all open perp
    positions. If any position has marginRatio < 0.15, closes ALL perp positions
    (full close, not partial -- per D-03, same as stop-loss monitor pattern).
    """
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.manager import OrderManager
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.strategies.portfolio.schemas import RebalanceOrder

    MARGIN_THRESHOLD = 0.15  # D-02: 15% threshold

    # 1. Rebuild perp adapter from DB
    adapter = _build_perp_adapter_from_db()
    if not adapter._positions:
        return {"checked": 0, "closed": []}

    # 2. Get mark prices and update adapter
    symbols = list(adapter._positions.keys())
    mark_prices = _get_perp_mark_prices(symbols)
    adapter.update_mark_prices(mark_prices)

    # 3. Check margin ratios
    positions = adapter.query_positions()
    breach_detected = False
    for pos in positions:
        if pos["marginRatio"] < MARGIN_THRESHOLD:
            logger.warning(
                "Liquidation risk: %s marginRatio=%.4f < %.4f threshold",
                pos["symbol"], pos["marginRatio"], MARGIN_THRESHOLD,
            )
            breach_detected = True

    if not breach_detected:
        return {"checked": len(positions), "closed": []}

    # 4. Full close ALL perp positions (D-03: close all, not partial)
    position_tracker = _build_position_tracker()
    perp_holdings = {
        sym: h for sym, h in position_tracker.current_holdings().items()
        if h.market == "crypto_perp"
    }

    close_orders = []
    for sym, holding in perp_holdings.items():
        close_orders.append(
            RebalanceOrder(
                symbol=sym,
                action="sell",
                target_weight=0.0,
                current_weight=holding.weight,
                delta_weight=-holding.weight,
                side=holding.side,
            )
        )

    if not close_orders:
        return {"checked": len(positions), "closed": []}

    # 5. Dispatch close orders via OrderManager (same as stop-loss pattern)
    broker_yaml_path = "config/broker_perp.yaml"
    try:
        with open(broker_yaml_path) as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    except FileNotFoundError:
        # Fallback: create minimal perp config
        broker_cfg = BrokerConfig(
            mode="paper",
            paper_initial_nav=100_000.0,  # USDT
            lot_size=1,
            fractional_qty=True,
        )

    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter
    close_adapter = PerpPaperAdapter(SessionLocal)
    # Rebuild positions for the close adapter too
    close_adapter._positions = adapter._positions.copy()

    risk_checker = OrderRiskChecker(
        position_limit_pct=1.0,  # No position limit for liquidation close
        max_exposure=10.0,  # No exposure limit for close
        stop_loss_pct=None,  # Not applicable for close
    )
    order_manager = OrderManager(
        close_adapter, risk_checker, position_tracker, SessionLocal, broker_cfg,
    )

    results = order_manager.execute_rebalance(
        close_orders, strategy_name="liquidation_protection",
        prices=mark_prices, market="crypto_perp",
    )

    # 6. Create TradeLogRecords for liquidation-closed positions
    closed_symbols = []
    now = datetime.now(timezone.utc)
    with db_session() as session:
        for result in results:
            if result.success:
                sym = result.order.symbol
                holding = perp_holdings.get(sym)
                if holding and holding.entry_price and holding.shares:
                    exit_price = mark_prices.get(sym, 0.0)
                    direction = 1 if holding.side == "long" else -1
                    trade_log = TradeLogRecord(
                        strategy_name="liquidation_protection",
                        symbol=sym,
                        market="crypto_perp",
                        entry_price=holding.entry_price,
                        exit_price=exit_price,
                        entry_date=holding.entry_date or now,
                        exit_date=now,
                        shares=holding.shares,
                        realized_pnl=(exit_price - holding.entry_price) * holding.shares * direction,
                        holding_days=(now - holding.entry_date).days if holding.entry_date else 0,
                        signal_id=None,  # Liquidation protection is not signal-triggered
                    )
                    session.add(trade_log)
                closed_symbols.append(sym)
        session.commit()

    logger.warning(
        "LIQUIDATION PROTECTION: closed %d perp positions: %s",
        len(closed_symbols), closed_symbols,
    )
    return {"checked": len(positions), "closed": closed_symbols}


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_rebalance")
def perp_rebalance(signal_id: str | None = None) -> dict:
    """Every 4h: run CryptoTrendStrategy -> Rebalancer -> OrderManager (D-04, D-05).

    24/7 -- NO weekend/holiday skip. Triggered 5 min after 4h OHLCV fetch.
    Uses leverage_limits from crypto_trend.yaml risk section (PRSK-03).

    Args:
        signal_id: Optional signal UUID that triggered this rebalance.
    """
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.manager import OrderManager
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.strategies.portfolio.crypto_trend import CryptoTrendConfig, CryptoTrendStrategy
    from poseidon.strategies.portfolio.rebalancer import PortfolioRebalancer

    now = datetime.now(timezone.utc)
    # NO weekend skip -- crypto 24/7

    # Load strategy config
    config_path = "config/strategies/crypto_trend.yaml"
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    strategy_cfg = CryptoTrendConfig(**raw_cfg)

    # Extract leverage limits from risk section (Plan 01)
    leverage_limits = {}
    if raw_cfg.get("risk", {}).get("max_leverage"):
        leverage_limits = raw_cfg["risk"]["max_leverage"]

    # Load broker config (perp-specific)
    broker_yaml_path = "config/broker_perp.yaml"
    try:
        with open(broker_yaml_path) as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    except FileNotFoundError:
        broker_cfg = BrokerConfig(
            mode="paper",
            paper_initial_nav=100_000.0,
            lot_size=1,
            fractional_qty=True,
        )

    # Build components
    position_tracker = _build_position_tracker()
    # Phase 56: CryptoTrendStrategy now uses DataRepository instead of PerpDataLoader
    strategy = CryptoTrendStrategy(strategy_cfg)
    adapter = PerpPaperAdapter(SessionLocal, leverage=strategy_cfg.allocation.leverage)

    # Set per-symbol leverage on adapter
    for sym in strategy_cfg.symbols:
        adapter.set_leverage(sym, strategy_cfg.allocation.leverage)

    # Rebuild adapter positions from DB
    _rebuild_adapter = _build_perp_adapter_from_db()
    adapter._positions = _rebuild_adapter._positions

    risk_checker = OrderRiskChecker(
        position_limit_pct=strategy_cfg.allocation.position_limit_pct,
        max_exposure=1.0,
        stop_loss_pct=0.15,  # Nominal — perps use liquidation monitor for actual protection
        market="crypto_perp",
    )
    order_manager = OrderManager(
        adapter, risk_checker, position_tracker, SessionLocal, broker_cfg,
        leverage_limits=leverage_limits,
    )
    rebalancer = PortfolioRebalancer()

    # --- Protection checks (D-11, D-12, D-13) ---
    from poseidon.protections.manager import ProtectionManager

    protection_mgr = ProtectionManager.from_defaults()

    # Check portfolio-level protections first (daily_loss halts ALL trading)
    with db_session() as prot_db:
        daily_results = protection_mgr.check_all("__portfolio__", strategy_cfg.market, prot_db)
        portfolio_locked = any(r.locked and r.protection_type == "daily_loss" for r in daily_results)
        if portfolio_locked:
            reason = next(r.reason for r in daily_results if r.locked and r.protection_type == "daily_loss")
            logger.warning("perp_rebalance: portfolio locked by daily_loss protection — %s", reason)
            return {"skipped": "protection_locked", "reason": reason}

    # Run strategy (inject DataRepository with session for perp data access)
    with db_session() as strategy_db:
        from poseidon.data.remote_repository import RemoteDataRepository
        strategy._repo = RemoteDataRepository.from_settings()
        targets = strategy.select_stocks(pd.DataFrame(), as_of=now.date())

    # Compute differential orders
    current_holdings = {
        sym: h for sym, h in position_tracker.current_holdings().items()
        if h.market == "crypto_perp"
    }
    rebalance_orders = rebalancer.rebalance(targets, current_holdings)

    if not rebalance_orders:
        logger.info("perp_rebalance: no rebalance orders")
        return {"rebalanced": True, "orders": 0, "sells": 0}

    # Filter out locked symbols (per D-12)
    with db_session() as prot_db:
        unlocked_orders = []
        for ro in rebalance_orders:
            if protection_mgr.is_locked(ro.symbol, strategy_cfg.market, prot_db):
                locks = protection_mgr.check_all(ro.symbol, strategy_cfg.market, prot_db)
                active = [r for r in locks if r.locked]
                logger.info("perp_rebalance: skipping %s — locked by %s", ro.symbol, [r.protection_type for r in active])
            else:
                unlocked_orders.append(ro)
        rebalance_orders = unlocked_orders

    if not rebalance_orders:
        logger.info("perp_rebalance: all symbols locked by protection")
        return {"rebalanced": True, "orders": 0, "sells": 0, "protection_filtered": True}

    # Get latest prices for weight-to-shares
    order_symbols = [ro.symbol for ro in rebalance_orders]
    prices = _get_perp_mark_prices(order_symbols)

    # Capture sell holding info BEFORE execution
    sell_holdings_info: dict[str, dict] = {}
    perp_holdings = current_holdings
    for ro in rebalance_orders:
        if ro.action == "sell":
            h = perp_holdings.get(ro.symbol)
            if h and h.entry_price and h.shares:
                sell_holdings_info[ro.symbol] = {
                    "entry_price": h.entry_price,
                    "entry_date": h.entry_date,
                    "shares": h.shares,
                    "side": h.side,
                }

    # Execute rebalance (with leverage enforcement from Plan 01)
    results = order_manager.execute_rebalance(
        rebalance_orders, strategy_name=strategy_cfg.name,
        prices=prices, market=strategy_cfg.market,
    )

    # Create TradeLogRecords for filled sell orders
    trade_log_count = 0
    with db_session() as session:
        for result in results:
            if result.success and result.order.action == "sell":
                sym = result.order.symbol
                info = sell_holdings_info.get(sym)
                if info and info.get("entry_price") and info.get("shares"):
                    exit_price = prices.get(sym, 0.0)
                    entry_price = info["entry_price"]
                    shares = info["shares"]
                    entry_date = info["entry_date"]
                    direction = 1 if info.get("side", "long") == "long" else -1
                    trade_log = TradeLogRecord(
                        strategy_name=strategy_cfg.name,
                        symbol=sym,
                        market=strategy_cfg.market,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_date=entry_date,
                        exit_date=now,
                        shares=shares,
                        realized_pnl=(exit_price - entry_price) * shares * direction,
                        holding_days=(now - entry_date).days if entry_date else 0,
                        signal_id=uuid.UUID(signal_id) if signal_id else None,
                    )
                    session.add(trade_log)
                    trade_log_count += 1
        session.commit()

    logger.info(
        "perp_rebalance: %d orders, %d trade logs",
        len(results), trade_log_count,
    )
    return {"rebalanced": True, "orders": len(results), "sells": trade_log_count}


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_funding_settlement")
def perp_funding_settlement() -> dict:
    """Every 8h: settle funding rates for open perp positions (D-06).

    24/7 -- NO weekend/holiday skip. Calls the existing
    record_funding_settlement function (Phase 26) for each open perp position.
    Funding rate retrieved from latest FundingRateRecord in DB.
    """
    import yaml

    from poseidon.funding.settlement import record_funding_settlement
    from poseidon.models.funding_rate import FundingRateRecord

    now = datetime.now(timezone.utc)
    # NO weekend skip -- crypto 24/7

    # Load strategy config for strategy name
    config_path = "config/strategies/crypto_trend.yaml"
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    strategy_name = raw_cfg.get("name", "Crypto Trend 4H")

    # Get open perp positions
    adapter = _build_perp_adapter_from_db()
    if not adapter._positions:
        return {"settled": 0, "symbols": []}

    # Get mark prices
    symbols = list(adapter._positions.keys())
    mark_prices = _get_perp_mark_prices(symbols)
    adapter.update_mark_prices(mark_prices)

    # Get latest funding rates from DB
    settled_symbols = []
    with db_session() as session:
        for sym, pos in adapter._positions.items():
            # Query latest funding rate for this symbol
            funding_record = (
                session.query(FundingRateRecord)
                .filter(FundingRateRecord.symbol == sym)
                .order_by(FundingRateRecord.time.desc())
                .first()
            )
            if funding_record is None:
                logger.warning("No funding rate found for %s, skipping settlement", sym)
                continue

            funding_rate = float(funding_record.funding_rate)

            # Calculate funding amount
            # Longs pay when rate > 0, shorts receive
            payment = pos.quantity * pos.entry_price * funding_rate
            direction = -1 if pos.side == "long" else 1
            funding_amount = payment * direction

            # Record settlement (idempotent)
            mark_price = mark_prices.get(sym, pos.entry_price)
            result = record_funding_settlement(
                session_factory=SessionLocal,
                symbol=sym,
                strategy_name=strategy_name,
                funding_amount=funding_amount,
                position_quantity=pos.quantity,
                mark_price=mark_price,
                settlement_time=now,
                market="crypto_perp",
            )
            if result is not None:
                settled_symbols.append(sym)
                logger.info(
                    "Funding settled %s: rate=%.6f amount=%.4f",
                    sym, funding_rate, funding_amount,
                )

    return {"settled": len(settled_symbols), "symbols": settled_symbols}


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_nav_snapshot")
def perp_nav_snapshot() -> dict:
    """Every 4h: record perp portfolio NAV snapshot (D-05).

    24/7 -- NO weekend/holiday skip. Creates NavSnapshotRecord with
    market='crypto_perp' to distinguish from TW stock NAV snapshots.
    """
    from datetime import date as date_type

    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.nav_snapshot import NavSnapshotRecord

    now = datetime.now(timezone.utc)
    # NO weekend skip -- crypto 24/7

    # Get perp holdings only
    position_tracker = _build_position_tracker()
    all_holdings = position_tracker.current_holdings()
    perp_holdings = {
        sym: h for sym, h in all_holdings.items()
        if h.market == "crypto_perp"
    }

    # Load perp broker config for initial NAV
    broker_yaml_path = "config/broker_perp.yaml"
    try:
        with open(broker_yaml_path) as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
        initial_nav = broker_cfg.paper_initial_nav
    except FileNotFoundError:
        initial_nav = 100_000.0  # default USDT

    # Compute holdings value from mark prices
    holdings_value = 0.0
    deployed_cost = 0.0
    if perp_holdings:
        prices = _get_perp_mark_prices(list(perp_holdings.keys()))
        for sym, holding in perp_holdings.items():
            price = prices.get(sym, 0.0)
            shares = holding.shares or 0
            entry_price = holding.entry_price or 0.0
            direction = 1 if holding.side == "long" else -1
            # For perps: value = margin + unrealized PnL
            holdings_value += shares * price  # mark-to-market value
            deployed_cost += shares * entry_price

    cash = initial_nav - deployed_cost
    total_nav = holdings_value + cash

    today = date_type.today()
    with db_session() as session:
        # Upsert: check for existing snapshot for this date+market
        existing = (
            session.query(NavSnapshotRecord)
            .filter(
                NavSnapshotRecord.snapshot_date == today,
                NavSnapshotRecord.market == "crypto_perp",
            )
            .first()
        )
        if existing:
            existing.total_nav = total_nav
            existing.holdings_value = holdings_value
            existing.cash = cash
            existing.holdings_count = len(perp_holdings)
        else:
            record = NavSnapshotRecord(
                snapshot_date=today,
                total_nav=total_nav,
                holdings_value=holdings_value,
                cash=cash,
                holdings_count=len(perp_holdings),
                market="crypto_perp",
            )
            session.add(record)
        session.commit()

    logger.info(
        "perp_nav_snapshot: date=%s total_nav=%.2f holdings=%d",
        today, total_nav, len(perp_holdings),
    )
    return {"date": str(today), "total_nav": total_nav, "holdings_count": len(perp_holdings)}




# --- Factor Analysis Tasks (Phase 47, D-13, D-15) ---


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
