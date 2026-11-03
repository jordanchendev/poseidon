"""Portfolio risk tasks: VaR snapshots, Monte Carlo VaR, covariance, and stress tests."""

import logging
from datetime import UTC, datetime, timedelta

from poseidon.core.config import settings
from poseidon.core.database import db_session
from poseidon.core.redis import get_redis
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


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
        as_of = datetime.now(UTC)
        portfolio_value = portfolio.total_exposure()

        calculator = VaRCalculator()
        results = []

        # 2. Load cached covariance
        cached = load_cached_covariance(redis_client)
        if cached is None:
            logger.warning("No cached covariance matrix, skipping VaR computation")
            return {"status": "skipped", "reason": "no cached covariance matrix"}
        _cov_symbols, cov_matrix, _cov_meta = cached

        # 3. Load portfolio returns for historical/cornish-fisher methods
        # TODO: Load real aligned returns from OHLCV storage once portfolio
        # return loading is fully wired. For now, compute from DB if possible.
        portfolio_returns = np.array([])
        try:
            return_series = {}
            for sym in symbols:
                ohlcv_df = repo.read_ohlcv(
                    sym, market="", interval="1d", start=as_of - timedelta(days=settings.var_lookback_days), end=as_of
                )
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
                    result = calculator.historical_simulation(portfolio_returns, portfolio_value, as_of)
                else:
                    logger.warning(
                        "Insufficient portfolio returns (%d) for historical VaR, skipping",
                        len(portfolio_returns),
                    )
                    continue
            elif m == VaRMethod.CORNISH_FISHER:
                if len(portfolio_returns) >= settings.var_min_observations:
                    result = calculator.cornish_fisher(
                        weights,
                        cov_matrix,
                        portfolio_returns,
                        portfolio_value,
                        as_of,
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
            db.add(
                VaRSnapshot(
                    time=result.as_of,
                    method=result.method,
                    var_95=result.var_95,
                    var_99=result.var_99,
                    cvar_95=result.cvar_95,
                    cvar_99=result.cvar_99,
                    portfolio_value=result.portfolio_value,
                    holding_period=result.holding_period,
                    details=result.details,
                )
            )

        db.commit()
        computed_methods = [r.method for r in results]
        logger.info("VaR computation completed: methods=%s", computed_methods)
        return {"status": "ok", "methods": computed_methods}


@celery_app.task(name="poseidon.workers.cpu_tasks.compute_mc_var")
def compute_mc_var() -> dict:
    """Compute Monte Carlo VaR using Cholesky decomposition.

    Uses cached covariance matrix (never recomputes), rebuilds portfolio for
    weights, and stores result in VaR snapshot table + Redis cache.

    Returns:
        Dict with status and result summary.
    """
    import msgpack

    from poseidon.models.var_snapshot import VaRSnapshot
    from poseidon.risk.portfolio import VirtualPortfolio
    from poseidon.risk.var.calculators import VaRCalculator
    from poseidon.risk.var.covariance import load_cached_covariance

    redis_client = get_redis("cache")
    with db_session() as db:
        # 1. Load cached covariance (never recompute inline)
        cached = load_cached_covariance(redis_client)
        if cached is None:
            logger.warning("No cached covariance matrix, skipping MC VaR")
            return {"status": "skipped", "reason": "no cached covariance matrix"}
        _cov_symbols, cov_matrix, _cov_meta = cached

        # 2. Rebuild portfolio from DB
        portfolio = VirtualPortfolio()
        portfolio.rebuild_from_db(db)
        if portfolio.open_position_count == 0:
            logger.info("No open positions, skipping MC VaR computation")
            return {"status": "skipped", "reason": "no open positions"}

        weights = portfolio.weights()
        portfolio_value = portfolio.total_exposure()
        as_of = datetime.now(UTC)

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
        db.add(
            VaRSnapshot(
                time=result.as_of,
                method=result.method,
                var_95=result.var_95,
                var_99=result.var_99,
                cvar_95=result.cvar_95,
                cvar_99=result.cvar_99,
                portfolio_value=result.portfolio_value,
                holding_period=result.holding_period,
                details=result.details,
            )
        )
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
        as_of = datetime.now(UTC)

        # 2. Load OHLCV close prices and compute returns for each symbol
        return_series = {}
        for sym in symbols:
            start = as_of - timedelta(days=settings.var_lookback_days)
            ohlcv_df = repo.read_ohlcv(sym, market="", interval="1d", start=start, end=as_of)
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


@celery_app.task(name="poseidon.workers.cpu_tasks.run_stress_test")
def run_stress_test(scenario_name: str, custom_shocks: dict | None = None) -> dict:
    """Run stress test scenario asynchronously.

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
        as_of = datetime.now(UTC)
        portfolio_value = portfolio.total_exposure()

        # 2. Load cached covariance
        cached = load_cached_covariance(redis_client)
        if cached is None:
            return {"status": "skipped", "reason": "no cached covariance matrix"}
        _cov_symbols, cov_matrix, _cov_meta = cached
        cov_matrix = np.array(cov_matrix)

        # 3. Load aligned returns for historical scenarios
        aligned_returns = None
        try:
            return_series = {}
            for sym in symbols:
                ohlcv_df = repo.read_ohlcv(
                    sym,
                    market="",
                    interval="1d",
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
        scenarios_dir = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "config" / "stress_scenarios")
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
                config,
                weights,
                symbols,
                cov_matrix,
                aligned_returns=aligned_returns,
                portfolio_value=portfolio_value,
            )
        else:
            # Named scenario from JSON config
            result = engine.run_scenario(
                scenario_name,
                weights,
                symbols,
                cov_matrix,
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
            "computed_at": (result.computed_at.isoformat() if result.computed_at else None),
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
