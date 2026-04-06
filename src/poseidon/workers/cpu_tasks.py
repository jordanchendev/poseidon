"""CPU worker Celery tasks for data fetching, backfill, backtest, and optimization."""

import logging
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import redis as redis_lib

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.optimizer import BayesianOptimizer, GridSearchOptimizer
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig
from poseidon.core.config import settings
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.fetchers import get_fetcher
from poseidon.data.cache import CacheManager
from poseidon.data.rate_limiter import CircuitBreaker, DistributedRateLimiter, PROVIDER_LIMITS
from poseidon.data.validation import validate_ohlcv
from poseidon.data.storage import (
    get_or_create_backfill_progress,
    read_ohlcv,
    update_backfill_progress,
    upsert_ohlcv,
)
from poseidon.data.symbols import get_market_config, get_symbols_for_market, load_symbols
from poseidon.models.backtest import BacktestRecord
from poseidon.models.base import SessionLocal
from poseidon.models.strategy import StrategyRecord
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.voting_strategy import VotingStrategy
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Market -> provider mapping for rate limiting and circuit breaker
MARKET_TO_PROVIDER = {
    "tw_stock": "finmind",
    "tw_futures": "finmind",
    "us_stock": "yfinance",
    "crypto_spot": "ccxt",
    "crypto_perp": "ccxt",
}


def _get_redis_client() -> redis_lib.Redis:
    """Create a Redis client for rate limiter and circuit breaker."""
    return redis_lib.from_url(settings.redis_url, decode_responses=False)


# Backfill target: 10 years of historical data
BACKFILL_YEARS = 10

# Batch sizes for backfill pagination (per provider)
BATCH_DAYS = {
    "tw_stock": 365,       # FinMind: 1 year per request
    "tw_futures": 365,     # FinMind: 1 year per request
    "us_stock": 1825,      # yfinance: all 5 years in one call
    "crypto_spot": 41,     # CCXT 1h: ~41 days per 1000 candles
}
BATCH_DAYS_DAILY = {
    "crypto_spot": 900,    # CCXT 1d: ~2.7 years per 1000 candles
}
BATCH_DAYS_5M = {
    "crypto_spot": 3,      # CCXT 5m: ~3.47 days per 1000 candles
}


@celery_app.task(name="poseidon.workers.cpu_tasks.fetch_market_data")
def fetch_market_data(market: str, interval: str, symbol: str | None = None) -> dict:
    """Fetch latest data for all symbols (or a specific symbol) in a market.

    This is the task called by Celery Beat on schedule.

    Args:
        market: Market name (e.g., "crypto_spot", "tw_stock").
        interval: Candle interval ("1d" or "1h").
        symbol: Optional symbol ID to fetch. If None, fetches all symbols in the market.
    """
    config = load_symbols()
    symbols = get_symbols_for_market(market, config)
    market_cfg = get_market_config(market, config)

    # Filter to specific symbol if requested
    if symbol and symbols:
        symbols = [s for s in symbols if s.id == symbol or s.ccxt_symbol == symbol]

    # If a specific symbol was requested but not found in config, create an ad-hoc entry
    if not symbols and symbol:
        from poseidon.data.symbols import SymbolInfo
        symbols = [SymbolInfo(id=symbol, name=symbol)]
        logger.info("Symbol %s not in config, fetching ad-hoc for market %s", symbol, market)
    elif not symbols:
        logger.warning("No symbols configured for market: %s", market)
        return {"market": market, "interval": interval, "fetched": 0}

    fetcher = get_fetcher(market)
    instrument = market_cfg.instrument if market_cfg else "spot"

    # Determine date range: fetch last 7 days to catch any missed data
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    fetched_count = 0

    # Initialize rate limiter and circuit breaker once per task call
    provider = MARKET_TO_PROVIDER.get(market, "unknown")
    redis_client = _get_redis_client()
    circuit = CircuitBreaker(
        redis_client,
        provider,
        failure_threshold=settings.circuit_failure_threshold,
        open_timeout=settings.circuit_open_timeout,
        failure_window=settings.circuit_failure_window,
    )
    rate_limiter = DistributedRateLimiter(redis_client)
    cache = CacheManager(redis_client) if settings.cache_enabled else None
    provider_cfg = PROVIDER_LIMITS.get(provider, {})
    window = provider_cfg.get("window_seconds", 3600)
    limit = getattr(settings, provider_cfg.get("limit_key", "ratelimit_finmind_hourly"), 500)

    session = SessionLocal()
    try:
        for sym_info in symbols:
            # 0. Check cache first (three-layer fallback: cache -> DB -> API)
            if cache is not None:
                cached_df = cache.get(sym_info.id, interval, start_date, end_date)
                if cached_df is not None and not cached_df.empty:
                    logger.debug("Cache hit for %s/%s/%s", market, sym_info.id, interval)
                    count = upsert_ohlcv(session, cached_df, sym_info.id, market, instrument, interval)
                    fetched_count += count
                    continue

            # 1. Check circuit breaker
            if not circuit.allow_request():
                logger.warning("Circuit open for %s, skipping %s", provider, sym_info.id)
                continue

            # 2. Acquire rate limit (wait up to 30s)
            if not rate_limiter.wait_and_acquire(provider, window, limit, timeout=30):
                logger.warning("Rate limit timeout for %s, skipping %s", provider, sym_info.id)
                continue

            # 3. Fetch
            try:
                fetch_symbol = sym_info.ccxt_symbol or sym_info.id
                df = fetcher.fetch_ohlcv(fetch_symbol, interval, start_date, end_date)
                circuit.record_success()
            except Exception as exc:
                circuit.record_failure()
                session.rollback()
                logger.error("Failed to fetch %s/%s/%s: %s", market, sym_info.id, interval, exc)
                continue

            # 4. Validate (per D-01: between fetch and upsert)
            if not df.empty:
                vresult = validate_ohlcv(df, market)
                if vresult.has_critical:
                    logger.error(
                        "CRITICAL validation failure for %s/%s: %s",
                        market,
                        sym_info.id,
                        [c for c in vresult.checks if not c.passed and c.severity.value == "critical"],
                    )
                    continue  # skip upsert per D-03
                if vresult.warning_count > 0:
                    logger.warning(
                        "Validation warnings for %s/%s: %s",
                        market,
                        sym_info.id,
                        [c for c in vresult.checks if not c.passed and c.severity.value == "warning"],
                    )
                # 5. Upsert (only if no CRITICAL)
                count = upsert_ohlcv(session, df, sym_info.id, market, instrument, interval)
                fetched_count += count
                # 6. Cache the validated data
                if cache is not None:
                    cache.set(sym_info.id, interval, start_date, end_date, df)
                logger.info("Fetched %d rows for %s/%s/%s", count, market, sym_info.id, interval)
            else:
                logger.info("No new data for %s/%s/%s", market, sym_info.id, interval)
    finally:
        session.close()

    logger.info("Completed fetch for %s/%s: %d total rows", market, interval, fetched_count)
    return {"market": market, "interval": interval, "fetched": fetched_count}


@celery_app.task(
    name="poseidon.workers.cpu_tasks.backfill_symbol",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def backfill_symbol(self, symbol: str, market: str, interval: str) -> dict:
    """Backfill historical data for a single symbol. Resumable via backfill_progress table.

    Args:
        symbol: Symbol ID (e.g., "2330", "BTCUSDT")
        market: Market name (e.g., "tw_stock", "crypto_spot")
        interval: Candle interval ("1d" or "1h")

    Returns:
        Dict with backfill status.
    """
    session = SessionLocal()
    try:
        target_start = datetime.now(timezone.utc) - timedelta(days=BACKFILL_YEARS * 365)
        progress = get_or_create_backfill_progress(session, symbol, market, interval, target_start)

        if progress.status == "completed":
            logger.info("Backfill already completed for %s/%s/%s", market, symbol, interval)
            return {"symbol": symbol, "market": market, "interval": interval, "status": "already_completed"}

        update_backfill_progress(session, progress, status="in_progress")

        # Determine the fetcher and instrument
        config = load_symbols()
        market_cfg = get_market_config(market, config)
        fetcher = get_fetcher(market)
        instrument = market_cfg.instrument if market_cfg else "spot"

        # Determine the ccxt_symbol for crypto
        symbols_list = get_symbols_for_market(market, config)
        fetch_symbol = symbol
        for s in symbols_list:
            if s.id == symbol and s.ccxt_symbol:
                fetch_symbol = s.ccxt_symbol
                break

        # Resume from last checkpoint or start from target
        start_dt = progress.last_fetched_date or progress.target_start_date
        end_dt = datetime.now(timezone.utc)

        # Choose batch size based on market and interval
        if interval == "1d" and market in BATCH_DAYS_DAILY:
            batch_days = BATCH_DAYS_DAILY[market]
        elif interval == "5m" and market in BATCH_DAYS_5M:
            batch_days = BATCH_DAYS_5M[market]
        else:
            batch_days = BATCH_DAYS.get(market, 365)

        # Initialize rate limiter and circuit breaker for backfill
        provider = MARKET_TO_PROVIDER.get(market, "unknown")
        redis_client = _get_redis_client()
        circuit = CircuitBreaker(
            redis_client,
            provider,
            failure_threshold=settings.circuit_failure_threshold,
            open_timeout=settings.circuit_open_timeout,
            failure_window=settings.circuit_failure_window,
        )
        rate_limiter = DistributedRateLimiter(redis_client)
        provider_cfg = PROVIDER_LIMITS.get(provider, {})
        window = provider_cfg.get("window_seconds", 3600)
        limit = getattr(settings, provider_cfg.get("limit_key", "ratelimit_finmind_hourly"), 500)

        try:
            # Fetch in batches
            current_start = start_dt
            total_rows = 0
            while current_start < end_dt:
                # Check circuit breaker before each batch
                if not circuit.allow_request():
                    logger.warning("Circuit open for %s, pausing backfill %s", provider, symbol)
                    break

                # Acquire rate limit before each batch (wait up to 60s for backfill)
                if not rate_limiter.wait_and_acquire(provider, window, limit, timeout=60):
                    logger.warning("Rate limit timeout for %s, pausing backfill %s", provider, symbol)
                    break

                batch_end = min(current_start + timedelta(days=batch_days), end_dt)
                start_str = current_start.strftime("%Y-%m-%d")
                end_str = batch_end.strftime("%Y-%m-%d")

                logger.info("Backfill %s/%s/%s: %s to %s", market, symbol, interval, start_str, end_str)
                try:
                    df = fetcher.fetch_ohlcv(fetch_symbol, interval, start_str, end_str)
                    circuit.record_success()
                except Exception as fetch_exc:
                    circuit.record_failure()
                    raise fetch_exc

                if not df.empty:
                    # Validate before upsert
                    vresult = validate_ohlcv(df, market)
                    if vresult.has_critical:
                        logger.error(
                            "CRITICAL validation failure in backfill %s/%s batch %s-%s: %s",
                            market,
                            symbol,
                            start_str,
                            end_str,
                            [c for c in vresult.checks if not c.passed and c.severity.value == "critical"],
                        )
                        # Skip this batch but continue with next
                        current_start = batch_end + timedelta(days=1)
                        continue
                    if vresult.warning_count > 0:
                        logger.warning(
                            "Validation warnings in backfill %s/%s batch %s-%s: %s",
                            market,
                            symbol,
                            start_str,
                            end_str,
                            [c for c in vresult.checks if not c.passed and c.severity.value == "warning"],
                        )
                    count = upsert_ohlcv(session, df, symbol, market, instrument, interval)
                    total_rows += count
                    # Update checkpoint after each batch
                    update_backfill_progress(session, progress, status="in_progress", last_fetched_date=batch_end)

                current_start = batch_end + timedelta(days=1)

            update_backfill_progress(session, progress, status="completed", last_fetched_date=end_dt)
            logger.info("Backfill completed for %s/%s/%s: %d total rows", market, symbol, interval, total_rows)
            return {"symbol": symbol, "market": market, "interval": interval, "status": "completed", "rows": total_rows}

        except Exception as exc:
            update_backfill_progress(session, progress, status="failed", error_message=str(exc))
            logger.error("Backfill failed for %s/%s/%s: %s", market, symbol, interval, exc)
            raise self.retry(exc=exc)

    finally:
        session.close()


@celery_app.task(name="poseidon.workers.cpu_tasks.trigger_backfill")
def trigger_backfill(market: str | None = None, symbol: str | None = None) -> dict:
    """Trigger backfill for all symbols (or a specific market/symbol).

    Dispatches individual backfill_symbol tasks for each symbol/interval combo.
    If a specific symbol is requested but not in config, it is still backfilled
    with the default interval ("1d").

    Args:
        market: Optional market to limit backfill to.
        symbol: Optional symbol ID to limit backfill to.
    """
    config = load_symbols()
    dispatched = 0

    markets_to_process = [market] if market else list(config.markets.keys())

    for m in markets_to_process:
        market_cfg = config.markets.get(m)
        if not market_cfg:
            # Market not in config but symbol explicitly requested — backfill with default interval
            if symbol and market:
                backfill_symbol.delay(symbol, m, "1d")
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/1d (not in config)", m, symbol)
            continue
        configured_ids = {sym.id for sym in market_cfg.symbols}
        for sym in market_cfg.symbols:
            if symbol and sym.id != symbol:
                continue
            for interval in market_cfg.intervals:
                backfill_symbol.delay(sym.id, m, interval)
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/%s", m, sym.id, interval)

        # Symbol requested but not in this market's config — backfill anyway
        if symbol and symbol not in configured_ids:
            for interval in market_cfg.intervals:
                backfill_symbol.delay(symbol, m, interval)
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/%s (not in config)", m, symbol, interval)

    logger.info("Dispatched %d backfill tasks", dispatched)
    return {"dispatched": dispatched}


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
) -> dict:
    """Run a backtest for an existing strategy.

    Loads the strategy from the DB, fetches OHLCV data, runs BacktestRunner,
    and persists results to BacktestRecord.

    Args:
        strategy_id: UUID string of the strategy to backtest.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).
        initial_capital: Starting capital for the backtest.

    Returns:
        Dict with backtest_id, status, and trade_count.
    """
    session = SessionLocal()
    backtest_id = uuid.uuid4()
    try:
        # Load strategy record from DB
        sid = uuid.UUID(strategy_id)
        record = session.get(StrategyRecord, sid)
        if not record:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Reconstruct strategy object
        if record.strategy_type == "rule":
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
        else:
            raise ValueError(f"Unknown strategy_type: {record.strategy_type!r}")

        # Parse date filters
        parsed_start = datetime.fromisoformat(start_date) if start_date else None
        parsed_end = datetime.fromisoformat(end_date) if end_date else None

        # Load OHLCV data
        ohlcv_df = read_ohlcv(
            session, record.symbol, record.market, record.interval,
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

        # Run backtest
        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=initial_capital,
            sizing_config=sizing_cfg,
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

    finally:
        session.close()


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
    session = SessionLocal()
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
        ohlcv_df = read_ohlcv(
            session, record.symbol, record.market, record.interval,
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

    finally:
        session.close()


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

    db = SessionLocal()
    redis_client = _get_redis_client()
    try:
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
                ohlcv_df = read_ohlcv(db, sym, market="", interval="1d",
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
    except Exception:
        db.rollback()
        logger.exception("VaR computation failed")
        raise
    finally:
        db.close()


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

    db = SessionLocal()
    redis_client = _get_redis_client()
    try:
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
    except Exception:
        db.rollback()
        logger.exception("MC VaR computation failed")
        raise
    finally:
        db.close()


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

    db = SessionLocal()
    redis_client = _get_redis_client()
    try:
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
            ohlcv_df = read_ohlcv(db, sym, market="", interval="1d",
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

    except Exception:
        logger.exception("Covariance matrix update failed")
        raise
    finally:
        db.close()


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
    import redis as redis_lib

    from poseidon.autoresearch.report import generate_report
    from poseidon.autoresearch.runner import AutoResearchRunner, MarketSpec
    from poseidon.backtest.experiment_tracker import ExperimentTracker
    from poseidon.backtest.holdout import HoldoutConfig
    from poseidon.backtest.param_search import SearchConfig as SearchConfigClass
    from poseidon.backtest.walk_forward import WalkForwardConfig

    started_at = datetime.now(timezone.utc)
    db = SessionLocal()
    redis_client = redis_lib.from_url(celery_app.conf.broker_url)

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
        task_id = self.request.id or "local"

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

        runner = AutoResearchRunner(
            db_session=db,
            search_config=cfg,
            stop_check=check_stop,
            progress_callback=update_progress,
            feature_specs="r2",  # Signal runner to use get_r2_specs() per-market
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
        db.close()


@celery_app.task(name="poseidon.workers.cpu_tasks.compute_quality_scores")
def compute_quality_scores() -> dict:
    """Compute data quality scores for all symbols (per D-09, DVAL-05).

    Iterates over symbols in symbols.yaml, computes quality score per
    symbol+interval, stores results in quality_scores table.

    Runs daily at 02:00 UTC via Celery Beat.
    """
    from poseidon.data.quality_scorer import DataQualityScorer
    from poseidon.data.storage import read_ohlcv
    from poseidon.data.symbols import load_symbols
    from poseidon.models.base import SessionLocal
    from poseidon.models.quality_score import QualityScore
    from poseidon.risk.var.returns import MARKET_CALENDARS

    scorer = DataQualityScorer()
    config = load_symbols()
    now = datetime.now(timezone.utc)
    lookback_days = settings.var_lookback_days  # default 252
    scored = 0
    skipped = 0

    db = SessionLocal()
    try:
        for market_name, market_cfg in config.markets.items():
            calendar = MARKET_CALENDARS.get(market_name, {"freq": "B", "tz": "UTC"})
            symbols = market_cfg.symbols
            intervals = market_cfg.intervals

            for sym in symbols:
                for interval in intervals:
                    # Compute expected rows using market calendar
                    start_date = now - timedelta(days=lookback_days)
                    if calendar["freq"] == "D":
                        # Calendar days (crypto)
                        expected_rows = lookback_days
                    else:
                        # Business days (TW/US): approximate with pandas
                        bdays = pd.bdate_range(start=start_date, end=now)
                        expected_rows = len(bdays)

                    # For intraday intervals, multiply by bars per day
                    if interval == "1h":
                        expected_rows *= 24
                    elif interval == "5m":
                        expected_rows *= 288  # 24*60/5

                    # Read recent OHLCV data
                    df = read_ohlcv(
                        db,
                        symbol=sym.id,
                        market=market_name,
                        interval=interval,
                        start=start_date,
                        end=now,
                    )

                    # Skip symbols with no data in last 7 days
                    if df.empty:
                        skipped += 1
                        continue
                    # read_ohlcv returns time as index, not column
                    latest_time = pd.Timestamp(df.index.max())
                    if latest_time.tzinfo is None:
                        latest_time = latest_time.tz_localize("UTC")
                    if (now - latest_time).days > 7:
                        skipped += 1
                        continue

                    # Compute quality score
                    # reset_index so validation_rules can access df["time"]
                    df = df.reset_index()
                    dims = scorer.compute(
                        df,
                        market=market_name,
                        interval=interval,
                        expected_rows=expected_rows,
                        latest_expected=now,
                    )

                    # Insert QualityScore row
                    row = QualityScore(
                        time=now,
                        symbol=sym.id,
                        interval=interval,
                        score=dims.composite,
                        completeness=dims.completeness,
                        consistency=dims.consistency,
                        anomaly_free=dims.anomaly_free,
                        timeliness=dims.timeliness,
                    )
                    db.merge(row)
                    scored += 1

        db.commit()
        logger.info("Quality scoring complete: scored=%d, skipped=%d", scored, skipped)
    except Exception:
        db.rollback()
        logger.exception("Quality scoring failed")
        raise
    finally:
        db.close()

    return {"scored": scored, "skipped": skipped}


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

    db = SessionLocal()
    redis_client = _get_redis_client()
    try:
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
                ohlcv_df = read_ohlcv(
                    db, sym, market="", interval="1d",
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
    except Exception:
        logger.exception("Stress test failed for scenario=%s", scenario_name)
        raise
    finally:
        db.close()


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

    session = SessionLocal()
    try:
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
    finally:
        session.close()


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
    """Monthly rebalance: run RevenueBreakoutStrategy -> Rebalancer -> OrderManager.

    Triggered on the 15th of each month at 01:30 UTC (09:30 UTC+8, after TW open).
    Skips weekends. Creates TradeLogRecord for each sell fill.

    Args:
        signal_id: Optional signal UUID that triggered this rebalance.
    """
    from datetime import date as date_type

    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.broker.paper_adapter import PaperBrokerAdapter
    from poseidon.models.ohlcv import OHLCV
    from poseidon.models.portfolio_holding import PortfolioHoldingRecord
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.orders.manager import OrderManager
    from poseidon.strategies.portfolio.rebalancer import PortfolioRebalancer
    from poseidon.strategies.portfolio.revenue_breakout import RevenueBreakoutStrategy
    from poseidon.strategies.portfolio.schemas import RevenueBreakoutConfig

    now = datetime.now(timezone.utc)

    # Skip weekends
    if now.weekday() >= 5:
        logger.info("portfolio_monthly_rebalance: skipping weekend (weekday=%d)", now.weekday())
        return {"skipped": "weekend"}

    # Load strategy config from YAML
    config_path = "config/strategies/revenue_breakout.yaml"
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    strategy_cfg = RevenueBreakoutConfig(**raw_cfg)

    # Load broker config from YAML
    broker_yaml_path = "config/broker.yaml"
    with open(broker_yaml_path) as bf:
        broker_raw = yaml.safe_load(bf)
    broker_cfg = BrokerConfig(**broker_raw)
    position_tracker = _build_position_tracker()
    broker = PaperBrokerAdapter(SessionLocal)
    risk_checker = OrderRiskChecker(
        position_limit_pct=strategy_cfg.allocation.position_limit_pct,
        max_exposure=1.0,
        stop_loss_pct=strategy_cfg.allocation.stop_loss_pct,
    )
    order_manager = OrderManager(broker, risk_checker, position_tracker, SessionLocal, broker_cfg)
    rebalancer = PortfolioRebalancer()

    # Run strategy
    strategy = RevenueBreakoutStrategy(strategy_cfg)
    targets = strategy.select_stocks(pd.DataFrame(), as_of=date_type.today())

    # Compute differential orders
    current_holdings = position_tracker.current_holdings()
    rebalance_orders = rebalancer.rebalance(targets, current_holdings)

    if not rebalance_orders:
        logger.info("portfolio_monthly_rebalance: no rebalance orders")
        return {"rebalanced": True, "orders": 0, "sells": 0}

    # Ensure OHLCV data exists for all selected symbols (fetch missing ones)
    order_symbols = [ro.symbol for ro in rebalance_orders]
    _ensure_ohlcv_data(order_symbols, market=strategy_cfg.market)

    # Get latest prices for weight-to-shares conversion
    prices = _get_latest_prices(order_symbols)

    # Capture sell holding info BEFORE execution (positions get closed)
    sell_holdings_info: dict[str, dict] = {}
    session = SessionLocal()
    try:
        for ro in rebalance_orders:
            if ro.action == "sell":
                record = (
                    session.query(PortfolioHoldingRecord)
                    .filter(
                        PortfolioHoldingRecord.symbol == ro.symbol,
                        PortfolioHoldingRecord.closed == False,  # noqa: E712
                    )
                    .first()
                )
                if record:
                    sell_holdings_info[ro.symbol] = {
                        "entry_price": record.entry_price,
                        "entry_date": record.entry_date,
                        "shares": record.shares,
                    }
    finally:
        session.close()

    # Execute rebalance
    results = order_manager.execute_rebalance(
        rebalance_orders, strategy_name=strategy_cfg.name, prices=prices, market=strategy_cfg.market,
    )

    # Create TradeLogRecords for filled sell orders
    trade_log_count = 0
    session = SessionLocal()
    try:
        for result in results:
            if result.success and result.order.action == "sell":
                sym = result.order.symbol
                info = sell_holdings_info.get(sym)
                if info and info.get("entry_price") and info.get("shares"):
                    exit_price = prices.get(sym, 0.0)
                    entry_price = info["entry_price"]
                    shares = info["shares"]
                    entry_date = info["entry_date"]
                    trade_log = TradeLogRecord(
                        strategy_name=strategy_cfg.name,
                        symbol=sym,
                        market=strategy_cfg.market,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_date=entry_date,
                        exit_date=now,
                        shares=shares,
                        realized_pnl=(exit_price - entry_price) * shares,
                        holding_days=(now - entry_date).days if entry_date else 0,
                        signal_id=uuid.UUID(signal_id) if signal_id else None,
                    )
                    session.add(trade_log)
                    trade_log_count += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    logger.info(
        "portfolio_monthly_rebalance: %d orders, %d trade logs",
        len(results),
        trade_log_count,
    )
    return {"rebalanced": True, "orders": len(results), "sells": trade_log_count}


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_stop_loss_monitor")
def portfolio_stop_loss_monitor() -> dict:
    """Intraday stop-loss monitor: check holdings against stop-loss thresholds.

    Runs every 5 min via beat. Self-gates to TW trading hours (01:00-05:30 UTC).
    Skips weekends. Dispatches sell orders for breached positions.
    """
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.broker.paper_adapter import PaperBrokerAdapter
    from poseidon.models.ohlcv import OHLCV
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
        broker = PaperBrokerAdapter(SessionLocal)
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
        session = SessionLocal()
        try:
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
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
    session = SessionLocal()
    try:
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

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


def _ensure_ohlcv_data(symbols: list[str], market: str = "tw_stock") -> None:
    """Ensure OHLCV data exists in DB for given symbols.

    Checks which symbols are missing from the DB and fetches recent data
    (last 30 days) using the appropriate market fetcher. This maintains the
    three-layer architecture: strategy selects from full universe (FinLab),
    then we ensure Poseidon DB has the data for order execution.
    """
    from poseidon.models.ohlcv import OHLCV
    from poseidon.data.fetchers import get_fetcher
    from poseidon.data.storage import upsert_ohlcv

    session = SessionLocal()
    try:
        existing = set(
            row[0]
            for row in session.query(OHLCV.symbol)
            .filter(OHLCV.market == market, OHLCV.interval == "1d")
            .distinct()
            .all()
        )
    finally:
        session.close()

    missing = [s for s in symbols if s not in existing]
    if not missing:
        return

    logger.info("_ensure_ohlcv_data: fetching %d missing symbols: %s", len(missing), missing)

    fetcher = get_fetcher(market)
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    session = SessionLocal()
    try:
        for sym in missing:
            try:
                df = fetcher.fetch_ohlcv(sym, "1d", start_date, end_date)
                if df is not None and not df.empty:
                    upsert_ohlcv(session, df, sym, market, "spot", "1d")
                    session.commit()
                    logger.info("_ensure_ohlcv_data: fetched %d rows for %s", len(df), sym)
                else:
                    logger.warning("_ensure_ohlcv_data: no data returned for %s", sym)
            except Exception as e:
                session.rollback()
                logger.warning("_ensure_ohlcv_data: failed to fetch %s: %s", sym, e)
    finally:
        session.close()


def _get_latest_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest close prices from OHLCV for given symbols.

    Returns:
        Dict mapping symbol -> latest close price.
    """
    from poseidon.models.ohlcv import OHLCV

    prices: dict[str, float] = {}
    session = SessionLocal()
    try:
        for sym in symbols:
            record = (
                session.query(OHLCV)
                .filter(OHLCV.symbol == sym, OHLCV.market == "tw_stock", OHLCV.interval == "1d")
                .order_by(OHLCV.time.desc())
                .first()
            )
            if record:
                prices[sym] = float(record.close)
    finally:
        session.close()
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
    session = SessionLocal()
    try:
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
    finally:
        session.close()

    return adapter


def _get_perp_mark_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest perpetual OHLCV close prices as mark prices."""
    from poseidon.models.ohlcv import OHLCV

    prices = {}
    session = SessionLocal()
    try:
        for sym in symbols:
            record = (
                session.query(OHLCV.close)
                .filter(
                    OHLCV.symbol == sym,
                    OHLCV.market == "crypto_perp",
                )
                .order_by(OHLCV.time.desc())
                .first()
            )
            if record:
                prices[sym] = float(record.close)
    finally:
        session.close()
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
    session = SessionLocal()
    try:
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

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
    from poseidon.data.loaders.perp_data_loader import PerpDataLoader
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
    data_loader = PerpDataLoader(SessionLocal)
    strategy = CryptoTrendStrategy(strategy_cfg, data_loader)
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

    # Run strategy
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
    session = SessionLocal()
    try:
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

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
    session = SessionLocal()
    try:
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
    finally:
        session.close()

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
    session = SessionLocal()
    try:
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    logger.info(
        "perp_nav_snapshot: date=%s total_nav=%.2f holdings=%d",
        today, total_nav, len(perp_holdings),
    )
    return {"date": str(today), "total_nav": total_nav, "holdings_count": len(perp_holdings)}


# ---------------------------------------------------------------------------
# Universe refresh (Phase 35)
# ---------------------------------------------------------------------------


@celery_app.task(
    name="poseidon.workers.cpu_tasks.refresh_universe",
    bind=True,
    max_retries=2,
)
def refresh_universe(self, market: str):
    """Refresh trading universe for a market: resolve source -> filter -> persist snapshot.

    If refresh fails, previous snapshot remains active (D-15).
    """
    from poseidon.universe.pipeline import UniversePipeline
    from poseidon.universe.snapshot import save_snapshot
    from poseidon.universe.yaml_source import YamlSource  # ensure registered

    session = SessionLocal()
    try:
        # TODO: make source/filter configurable per market (D-03)
        # For now, default to YamlSource with no filters
        source = YamlSource()
        pipeline = UniversePipeline(source=source, filters=[])
        symbols = pipeline.run(market, db_session=session)

        if not symbols:
            logger.warning(
                "refresh_universe: empty result for market=%s, skipping snapshot",
                market,
            )
            return {"market": market, "symbols": 0, "status": "skipped_empty"}

        snapshot = save_snapshot(
            db=session,
            market=market,
            snapshot_time=datetime.now(timezone.utc),
            symbols=symbols,
            source_type=source.name,
            filter_config=None,
        )
        logger.info("refresh_universe: market=%s, symbols=%d", market, len(symbols))
        return {
            "market": market,
            "symbols": len(symbols),
            "snapshot_id": str(snapshot.id),
        }
    except Exception as exc:
        logger.error("refresh_universe failed for market=%s: %s", market, exc)
        session.rollback()
        raise self.retry(exc=exc, countdown=60)
    finally:
        session.close()
