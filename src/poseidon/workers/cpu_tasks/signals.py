"""Legacy signal path: evaluate active strategies and publish their signals."""

import logging

from poseidon.core.database import db_session
from poseidon.data.feature_engine import FeatureEngine
from poseidon.models.strategy import StrategyRecord
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


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
    from poseidon.decision_loop.releases import ExecutableStrategyConfig
    from poseidon.models.strategy_version import StrategyVersion
    from poseidon.risk.pipeline import SignalPipeline
    from poseidon.signals.schemas import InstrumentType
    from poseidon.strategies.voting_strategy import VotingStrategy

    # Extract market/interval from fetch_result if not provided directly
    market = market or (fetch_result or {}).get("market")
    interval = interval or (fetch_result or {}).get("interval")
    if not market or not interval:
        return {"error": "market and interval required"}

    with db_session() as session:
        strategies = (
            session.query(StrategyRecord, StrategyVersion)
            .join(StrategyVersion, StrategyVersion.id == StrategyRecord.active_version_id)
            .filter(
                StrategyRecord.active_version_id.is_not(None),
                StrategyVersion.config_json["market"].as_string() == market,
                StrategyVersion.config_json["interval"].as_string() == interval,
            )
            .all()
        )

        engine = FeatureEngine()
        pipeline = SignalPipeline(session)
        total_signals = 0
        errors = []

        for record, version in strategies:
            try:
                executable = ExecutableStrategyConfig.model_validate(version.config_json)
                if executable.strategy_type == "rule":
                    strategy = RuleStrategy(
                        config=executable.runtime_config(),
                        instrument=InstrumentType(executable.instrument),
                        strategy_id=record.id,
                    )
                elif executable.strategy_type == "voting":
                    strategy = VotingStrategy(
                        config=executable.runtime_config(),
                        instrument=InstrumentType(executable.instrument),
                        strategy_id=record.id,
                    )
                elif executable.strategy_type == "liquidity_sweep":
                    # supports_live = False -> deferred to backtest/research path only.
                    logger.debug(
                        "Skipping live signal eval for liquidity_sweep strategy %s "
                        "(supports_live=False; use run_backtest path instead)",
                        record.name,
                    )
                    continue
                else:
                    # model strategies use GPU predict path
                    continue

                features = engine.compute(
                    executable.symbol,
                    executable.market,
                    executable.interval,
                    feature_specs=executable.live_feature_specs(),
                )
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
