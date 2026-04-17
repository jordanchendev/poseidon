#!/usr/bin/env python3
"""Phase 65 FIX-02: LiquiditySweepStrategy Optuna validation.

Runs 100-trial Optuna search per symbol (BTCUSDT, ETHUSDT) under
FillModel.PESSIMISTIC with WFE >= 50% as the pass gate.

Per D-05: Uses LiquiditySweepStrategyFactory + ParameterSearchPipeline.
Per D-06: WFE < 50% marks strategy as experimental, does not block v13.0.

Run inside stormtrooper docker:
  docker compose exec -T cpu-worker python scripts/verify_liquidity_sweep.py
"""

import sys
import time

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory
from poseidon.backtest.param_search import ParameterSearchPipeline, SearchConfig
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.core.database import db_session
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MARKET = "crypto_perp"
INTERVAL = "1h"
N_TRIALS = 100
WFE_THRESHOLD = 0.50


def main() -> int:
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine()
    cost_model = COST_MODELS["crypto_perp"]
    sizing = SizingConfig(mode=SizingMode.FIXED_RISK, risk_pct=0.01)
    repo = RemoteDataRepository.from_settings()

    results = {}
    for symbol in SYMBOLS:
        print(f"\n{'=' * 60}")
        print(f"FIX-02: {symbol} {MARKET} {INTERVAL} -- {N_TRIALS} trials PESSIMISTIC")
        print(f"{'=' * 60}")

        ohlcv = repo.read_ohlcv(symbol, MARKET, INTERVAL)
        print(f"OHLCV rows: {len(ohlcv)}, range: {ohlcv.index[0]} to {ohlcv.index[-1]}")

        start_time = time.time()

        with db_session() as session:
            tracker = ExperimentTracker(db_session=session)

            pipeline = ParameterSearchPipeline(
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
                tracker=tracker,
                sizing_config=sizing,
                strategy_factory=LiquiditySweepStrategyFactory(),
                fill_model=FillModel.PESSIMISTIC,
            )

            search_config = SearchConfig(n_trials=N_TRIALS)
            result = pipeline.run(
                ohlcv=ohlcv,
                symbol=symbol,
                market=MARKET,
                interval=INTERVAL,
                config=search_config,
            )
            session.commit()

        elapsed = time.time() - start_time
        results[symbol] = result

        print(f"\nResults for {symbol} (elapsed: {elapsed:.0f}s):")
        print(f"  Total trials:    {result.total_trials}")
        print(f"  Passed trials:   {result.passed_trials}")
        print(f"  Rejected:        {result.rejected_trials}")
        print(f"  Best score:      {result.best_composite_score}")
        if result.best_config:
            print(f"  Best config:     {result.best_config}")

    print(f"\n{'=' * 60}")
    print("FIX-02 SUMMARY")
    print(f"{'=' * 60}")

    all_pass = True
    for symbol, result in results.items():
        has_passed = result.passed_trials > 0
        if has_passed:
            status = f"PASS (WFE >= {WFE_THRESHOLD*100:.0f}%) -- {result.passed_trials}/{result.total_trials} trials passed"
        else:
            status = f"EXPERIMENTAL (no trials passed WFE >= {WFE_THRESHOLD*100:.0f}%)"
            all_pass = False
        print(f"  {symbol}: {status}")

    overall = "PASS" if all_pass else "EXPERIMENTAL"
    print(f"\n>>> FIX-02 Overall: {overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
