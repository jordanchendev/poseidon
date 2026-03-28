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
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Market -> provider mapping for rate limiting and circuit breaker
MARKET_TO_PROVIDER = {
    "tw_stock": "finmind",
    "tw_futures": "finmind",
    "us_stock": "yfinance",
    "crypto_spot": "ccxt",
}


def _get_redis_client() -> redis_lib.Redis:
    """Create a Redis client for rate limiter and circuit breaker."""
    return redis_lib.from_url(settings.redis_url, decode_responses=False)


# Backfill target: 5 years of historical data
BACKFILL_YEARS = 5

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

        # Build config dict for persistence
        config_dict = {
            "strategy_type": record.strategy_type,
            "symbol": record.symbol,
            "market": record.market,
            "interval": record.interval,
            "initial_capital": initial_capital,
            "start_date": start_date,
            "end_date": end_date,
            "strategy_params": record.config,
        }

        # Persist to BacktestRecord directly (the repository expects TradeRecord
        # objects, but we only have trade dicts from BacktestResult; persist
        # the main record with metrics for the API to query)
        bt_record = BacktestRecord(
            id=backtest_id,
            strategy_id=sid,
            strategy_type=record.strategy_type,
            symbol=record.symbol,
            market=record.market,
            interval=record.interval,
            config=config_dict,
            metrics=result.metrics,
            status=result.status,
            error_message=result.error_message,
            completed_at=datetime.now(timezone.utc),
        )
        session.add(bt_record)
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
