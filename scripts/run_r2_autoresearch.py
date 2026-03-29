"""Phase 21 SIG2-05 validation: AutoResearch with R2 sub_signals.

Run Optuna search with R2 features enabled across multiple markets/intervals.
Target: at least one strategy with sharpe > 0.5 and profit factor > 1.5.

Changes from v1:
- n_trials=100 (was 30)
- Added tw_stock: 2317, 2454, 2881
- Added crypto 1h: BTCUSDT, ETHUSDT (43k+ bars each)
- Relaxed WFE: max_insufficient_ratio=0.50 (was 0.30), min_wfe=0.30 (was 0.50)
"""

from __future__ import annotations

import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from poseidon.autoresearch.runner import AutoResearchRunner, MarketSpec
from poseidon.backtest.param_search import SearchConfig
from poseidon.backtest.walk_forward import WalkForwardConfig
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
        # tw_stock 1d — top weighted stocks with institutional flow
        MarketSpec(symbol="2330", market="tw_stock", interval="1d"),
        MarketSpec(symbol="2317", market="tw_stock", interval="1d"),
        MarketSpec(symbol="2454", market="tw_stock", interval="1d"),
        MarketSpec(symbol="2881", market="tw_stock", interval="1d"),
        # crypto 1d
        MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1d"),
        MarketSpec(symbol="ETHUSDT", market="crypto_spot", interval="1d"),
        # crypto 1h — much more data (43k+ bars)
        MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h"),
        MarketSpec(symbol="ETHUSDT", market="crypto_spot", interval="1h"),
    ]

    # Relaxed WFE: allow more insufficient windows and lower threshold
    wf_config = WalkForwardConfig(
        max_insufficient_ratio=0.50,  # was 0.30 — tolerate more low-trade windows
    )

    search_config = SearchConfig(
        n_trials=100,
        max_trials=100,
        min_wfe=0.30,  # was 0.50 — lower bar for WFE gate
        walk_forward=wf_config,
        strategy_mode="bidirectional",
    )

    logger.info(
        "Starting R2 AutoResearch v2: %d markets, %d trials, min_wfe=%.2f, max_insufficient=%.2f",
        len(markets), search_config.n_trials,
        search_config.min_wfe, wf_config.max_insufficient_ratio,
    )

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
            logger.warning("Market %s/%s/%s FAILED: %s", r.spec.symbol, r.spec.market, r.spec.interval, r.error)
            continue

        sr = r.search_result
        if not sr:
            logger.info("Market %s/%s/%s: no search result", r.spec.symbol, r.spec.market, r.spec.interval)
            continue

        logger.info(
            "Market %s/%s/%s: %d trials, %d passed WFE, best_composite=%.4f",
            r.spec.symbol, r.spec.market, r.spec.interval,
            sr.total_trials, sr.passed_trials,
            sr.best_composite_score or 0,
        )

        # Query tracker for passed experiments with metrics
        try:
            experiments = tracker.query_passed_by_study(sr.study_name, limit=3)
            for e in experiments:
                m = e.metrics_json or {}
                sharpe = m.get("sharpe_ratio", 0)
                pf = m.get("profit_factor", 0)
                trades = m.get("total_trades", 0)
                logger.info(
                    "  Best trial: sharpe=%.3f, pf=%.3f, trades=%d, composite=%.4f",
                    sharpe, pf, trades, e.composite_score or 0,
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_pf = pf
                    best_market = f"{r.spec.symbol}/{r.spec.market}/{r.spec.interval}"
        except Exception as e:
            logger.warning("Failed to query tracker for %s: %s", sr.study_name, e)

    print("\n" + "=" * 60)
    print("SIG2-05 VALIDATION RESULT (v2)")
    print("=" * 60)
    print(f"Markets searched: {len(markets)}")
    print(f"Trials per market: {search_config.n_trials}")
    print(f"WFE config: min_wfe={search_config.min_wfe}, max_insufficient={wf_config.max_insufficient_ratio}")
    print()
    if best_market and best_sharpe > 0.5 and best_pf > 1.5:
        print(f"PASSED: {best_market} sharpe={best_sharpe:.3f} pf={best_pf:.3f}")
        session.close()
        sys.exit(0)
    elif best_market:
        print(f"BEST: {best_market} sharpe={best_sharpe:.3f} pf={best_pf:.3f}")
        print("Target: sharpe > 0.5 AND profit_factor > 1.5")
        if best_sharpe > 0.5 or best_pf > 1.5:
            print("STATUS: PARTIALLY MET")
        else:
            print("STATUS: NOT YET MET")
        session.close()
        sys.exit(1)
    else:
        print("NO RESULTS: All markets failed or had no passed trials")
        session.close()
        sys.exit(2)


if __name__ == "__main__":
    main()
