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
        # tw_stock skipped for now -- FinLab concurrent session limit
        # MarketSpec(symbol="2330", market="tw_stock", interval="1d"),
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

    # Report
    best_sharpe = -999.0
    best_pf = -999.0
    best_market = None

    for r in results:
        if r.error:
            logger.warning("Market %s/%s failed: %s", r.spec.symbol, r.spec.market, r.error)
            continue
        if r.search_result and r.search_result.best_config:
            cfg = r.search_result.best_config
            sharpe = cfg.get("sharpe_ratio", cfg.get("sharpe", 0))
            pf = cfg.get("profit_factor", 0)
            logger.info(
                "Market %s/%s: %d trials, %d passed, best_sharpe=%.3f, profit_factor=%.3f",
                r.spec.symbol, r.spec.market,
                r.search_result.total_trials, r.search_result.passed_trials,
                sharpe, pf,
            )
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_pf = pf
                best_market = f"{r.spec.symbol}/{r.spec.market}"
        else:
            logger.info(
                "Market %s/%s: %d trials, %d passed, no best config",
                r.spec.symbol, r.spec.market,
                r.search_result.total_trials if r.search_result else 0,
                r.search_result.passed_trials if r.search_result else 0,
            )

    print("\n" + "=" * 60)
    print("SIG2-05 VALIDATION RESULT")
    print("=" * 60)
    if best_market and best_sharpe > 0.5 and best_pf > 1.5:
        print(f"PASSED: {best_market} sharpe={best_sharpe:.3f} pf={best_pf:.3f}")
        sys.exit(0)
    elif best_market:
        print(f"BEST: {best_market} sharpe={best_sharpe:.3f} pf={best_pf:.3f}")
        print("Target: sharpe > 0.5 AND profit_factor > 1.5")
        print("STATUS: NOT YET MET (may need more trials or data)")
        sys.exit(1)
    else:
        print("NO RESULTS: All markets failed or had no data")
        sys.exit(2)

    session.close()


if __name__ == "__main__":
    main()
