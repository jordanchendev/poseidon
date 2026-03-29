"""Phase 21 SIG2-05 validation: AutoResearch with R2 sub_signals.

Run Optuna search with R2 features enabled for tw_stock and crypto_spot.
Target: at least one strategy with sharpe > 0.5 and profit factor > 1.5.
"""

from __future__ import annotations

import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from poseidon.autoresearch.runner import AutoResearchRunner, MarketSpec
from poseidon.backtest.param_search import SearchConfig
from poseidon.core.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = Settings()
    engine = create_engine(str(settings.database_url))
    Session = sessionmaker(bind=engine)
    session = Session()

    markets = [
        MarketSpec(symbol="2330", market="tw_stock", interval="1d"),
        MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1d"),
        MarketSpec(symbol="ETHUSDT", market="crypto_spot", interval="1d"),
    ]

    search_config = SearchConfig(
        n_trials=30,
        max_trials=50,
        strategy_mode="bidirectional",
    )

    logger.info("Starting R2 AutoResearch: %d markets, %d trials each", len(markets), search_config.n_trials)

    runner = AutoResearchRunner(
        db_session=session,
        search_config=search_config,
        feature_specs="r2",
    )

    results = runner.run(markets)

    # Query experiment tracker for best metrics per market
    from poseidon.backtest.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(session)

    best_sharpe = -999.0
    best_pf = -999.0
    best_market = None

    for r in results:
        if r.error:
            logger.warning("Market %s/%s FAILED: %s", r.spec.symbol, r.spec.market, r.error)
            continue

        sr = r.search_result
        if not sr:
            logger.info("Market %s/%s: no search result", r.spec.symbol, r.spec.market)
            continue

        logger.info(
            "Market %s/%s: %d trials, %d passed WFE, best_composite=%.4f",
            r.spec.symbol, r.spec.market,
            sr.total_trials, sr.passed_trials,
            sr.best_composite_score or 0,
        )

        # Query tracker for passed experiments with metrics
        try:
            experiments = tracker.query_passed_by_study(sr.study_name)
            for exp in experiments:
                metrics = exp.metrics_json or {}
                sharpe = metrics.get("sharpe_ratio", 0)
                pf = metrics.get("profit_factor", 0)
                logger.info(
                    "  Trial %s: sharpe=%.3f, pf=%.3f, composite=%.4f",
                    exp.optuna_trial_number, sharpe, pf, exp.composite_score or 0,
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_pf = pf
                    best_market = f"{r.spec.symbol}/{r.spec.market}"
        except Exception as e:
            logger.warning("Failed to query tracker for %s: %s", sr.study_name, e)

    print("\n" + "=" * 60)
    print("SIG2-05 VALIDATION RESULT")
    print("=" * 60)
    if best_market and best_sharpe > 0.5 and best_pf > 1.5:
        print(f"PASSED: {best_market} sharpe={best_sharpe:.3f} pf={best_pf:.3f}")
        session.close()
        sys.exit(0)
    elif best_market:
        print(f"BEST: {best_market} sharpe={best_sharpe:.3f} pf={best_pf:.3f}")
        print("Target: sharpe > 0.5 AND profit_factor > 1.5")
        print("STATUS: NOT YET MET (may need more trials or data)")
        session.close()
        sys.exit(1)
    else:
        print("NO RESULTS: All markets failed or had no passed trials")
        session.close()
        sys.exit(2)


if __name__ == "__main__":
    main()
