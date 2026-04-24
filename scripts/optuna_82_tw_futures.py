#!/usr/bin/env python3
"""Phase 82: Optuna TrendFollowing parameter optimization for TX daily.

Searches EMA fast/slow periods, ATR period, and ATR multiplier using
BayesianOptimizer with IS Sharpe ratio objective. Compares best result
against baseline (default TrendFollowingConfig params: EMA 20/60,
ATR 14, multiplier 2.0, baseline Sharpe 0.21).

Run on stormtrooper:
  docker compose exec cpu-worker python scripts/optuna_82_tw_futures.py

Expected runtime: 10-30 min (100 trials on TX daily ~1000 bars)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.optimizer import BayesianOptimizer, OptimizationTrial
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.tw_futures.configs import TrendFollowingConfig
from poseidon.strategies.tw_futures.trend_following import TrendFollowingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (per D-02, D-09, D-11)
# ---------------------------------------------------------------------------
START = datetime(2021, 3, 22)  # Earliest TX daily data (D-02)
END = datetime.now()
SYMBOL = "TX"
MARKET = "tw_futures"
INITIAL_CAPITAL = 1_000_000.0  # 1M TWD
INTERVAL = "1d"
BARS_PER_YEAR = 252
COST_MODEL = COST_MODELS["tw_futures"]
N_TRIALS = 100  # D-09
SEED = 42
SIZING = SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.03)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "optuna_82_results.json"

# Parameter search space (D-08): BayesianOptimizer (low, high, type) format
PARAM_SPACE: dict[str, tuple] = {
    "ema_fast": (5, 50, "int"),
    "ema_slow": (20, 120, "int"),
    "atr_period": (7, 28, "int"),
    "atr_multiplier": (1.0, 4.0, "float"),
}


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def fetch_ohlcv(repo: RemoteDataRepository) -> pd.DataFrame:
    """Fetch TX 1d OHLCV from Thalassa via RemoteDataRepository.

    Strips timezone from index for BacktestRunner compatibility (Pitfall 8).
    Raises RuntimeError if no data returned.
    """
    logger.info("Fetching TX %s OHLCV from %s to %s ...", INTERVAL, START.date(), END.date())
    df = repo.read_ohlcv(
        symbol=SYMBOL,
        market=MARKET,
        interval=INTERVAL,
        start=START,
        end=END,
    )

    if df.empty:
        raise RuntimeError(f"No OHLCV data for {SYMBOL} {MARKET} {INTERVAL}")

    # Strip timezone from index (Pitfall 8)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.tz_localize(None)

    logger.info(
        "TX %s: %d bars (%s to %s)",
        INTERVAL, len(df), df.index[0], df.index[-1],
    )
    return df


# ---------------------------------------------------------------------------
# Strategy factory for BayesianOptimizer
# ---------------------------------------------------------------------------
def make_strategy_factory() -> Callable[[dict], TrendFollowingStrategy]:
    """Create strategy factory for BayesianOptimizer.

    Handles cross-parameter constraint (Pitfall 2): if ema_fast >= ema_slow,
    validate_config() raises ValueError which Optuna records as a failed trial.
    BayesianOptimizer's internal objective catches the exception via
    BacktestRunner and returns metric_value=0.0 for these trials.
    """

    def factory(params: dict) -> TrendFollowingStrategy:
        config = TrendFollowingConfig(
            ema_fast=params["ema_fast"],
            ema_slow=params["ema_slow"],
            atr_period=params["atr_period"],
            atr_multiplier=params["atr_multiplier"],
        )
        strategy = TrendFollowingStrategy(config=config)
        # validate_config() raises ValueError if ema_fast >= ema_slow (Pitfall 2)
        strategy.validate_config()
        return strategy

    return factory


# ---------------------------------------------------------------------------
# Numpy type serialization helper
# ---------------------------------------------------------------------------
def _to_native(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def _sanitize_dict(d: dict) -> dict:
    """Recursively convert numpy types in a dict to Python native."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [_to_native(x) if not isinstance(x, dict) else _sanitize_dict(x) for x in v]
        else:
            result[k] = _to_native(v)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run Optuna TrendFollowing parameter optimization."""
    repo = RemoteDataRepository.from_settings()

    # Step 1: Fetch OHLCV data
    print(f"Phase 82: TrendFollowing Optuna Parameter Search")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} TWD")
    print(f"Cost Model: {COST_MODEL.description}")
    print(f"N_TRIALS: {N_TRIALS}, SEED: {SEED}")
    print()

    ohlcv = fetch_ohlcv(repo)
    print(f"OHLCV loaded: {len(ohlcv)} bars ({ohlcv.index[0]} to {ohlcv.index[-1]})")
    print()

    # Step 2: Run baseline backtest with default params
    print(f"{'='*60}")
    print("Running baseline (default TrendFollowingConfig)...")
    print(f"{'='*60}")

    baseline_config = TrendFollowingConfig()
    baseline_strategy = TrendFollowingStrategy(config=baseline_config)
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine(rules=[])

    baseline_runner = BacktestRunner(
        strategy=baseline_strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
        sizing_config=SIZING,
    )
    baseline_result = baseline_runner.run(ohlcv, feature_specs=baseline_strategy.get_feature_specs())
    baseline_metrics = baseline_result.metrics
    baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0.0)

    print(f"  Baseline params: ema_fast={baseline_config.ema_fast}, "
          f"ema_slow={baseline_config.ema_slow}, "
          f"atr_period={baseline_config.atr_period}, "
          f"atr_multiplier={baseline_config.atr_multiplier}")
    print(f"  Baseline Sharpe: {baseline_sharpe:.4f}")
    print(f"  Baseline MaxDD:  {baseline_metrics.get('max_drawdown', 0.0):.4f}")
    print(f"  Baseline Return: {baseline_metrics.get('total_return', 0.0):.4f}")
    print(f"  Baseline Trades: {baseline_metrics.get('trade_count', 0)}")
    print()

    # Step 3: Create BayesianOptimizer and run search (D-10, D-11)
    print(f"{'='*60}")
    print(f"Running BayesianOptimizer: {N_TRIALS} trials")
    print(f"Search space: {PARAM_SPACE}")
    print(f"{'='*60}")
    print()

    optimizer = BayesianOptimizer(
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
        sizing_config=SIZING,
    )

    trials = optimizer.optimize(
        strategy_factory=make_strategy_factory(),
        ohlcv=ohlcv,
        param_space=PARAM_SPACE,
        n_trials=N_TRIALS,
        metric="sharpe_ratio",  # D-11: IS Sharpe as objective
        seed=SEED,
    )

    # Step 4: Filter and analyze results
    # BayesianOptimizer returns 0.0 for failed trials (ValueError from validate_config)
    completed = [t for t in trials if t.metric_value > 0.0]
    completed.sort(key=lambda t: t.metric_value, reverse=True)

    print(f"\nCompleted trials: {len(completed)}/{len(trials)}")

    if not completed:
        print("ERROR: No valid trials completed. All returned 0.0.")
        return 1

    # Print top 5
    print(f"\n{'='*60}")
    print("Top 5 Parameter Sets")
    print(f"{'='*60}")
    print(
        f"{'Rank':<5} {'EMA_F':<7} {'EMA_S':<7} {'ATR_P':<7} {'ATR_M':<7} "
        f"{'Sharpe':<8} {'MaxDD':<8} {'Trades':<8}"
    )
    print("-" * 65)

    for i, trial in enumerate(completed[:5], 1):
        p = trial.params
        m = trial.metrics
        print(
            f"{i:<5} "
            f"{p.get('ema_fast', '-'):<7} "
            f"{p.get('ema_slow', '-'):<7} "
            f"{p.get('atr_period', '-'):<7} "
            f"{p.get('atr_multiplier', 0):<7.2f} "
            f"{trial.metric_value:<8.4f} "
            f"{m.get('max_drawdown', 0):<8.4f} "
            f"{m.get('trade_count', 0):<8}"
        )

    best = completed[0]
    print(f"\nBest params: {best.params}")
    print(f"Best Sharpe: {best.metric_value:.4f}")
    print(f"Baseline Sharpe: {baseline_sharpe:.4f}")
    improvement = best.metric_value - baseline_sharpe
    print(f"Improvement: {improvement:+.4f}")

    # Step 5: Build output JSON
    top_10 = []
    for trial in completed[:10]:
        top_10.append({
            "params": _sanitize_dict(trial.params) if isinstance(trial.params, dict) else trial.params,
            "sharpe": _to_native(trial.metric_value),
            "metrics": _sanitize_dict(trial.metrics),
        })

    output = {
        "phase": "82",
        "script": "optuna_82_tw_futures.py",
        "n_trials": N_TRIALS,
        "n_completed": len(completed),
        "bars_per_year": BARS_PER_YEAR,
        "cost_model": "tw_futures",
        "period": {
            "start": START.date().isoformat(),
            "end": END.date().isoformat(),
        },
        "baseline_sharpe": _to_native(baseline_sharpe),
        "baseline_params": {
            "ema_fast": baseline_config.ema_fast,
            "ema_slow": baseline_config.ema_slow,
            "atr_period": baseline_config.atr_period,
            "atr_multiplier": baseline_config.atr_multiplier,
        },
        "best_params": _sanitize_dict(best.params),
        "best_sharpe": _to_native(best.metric_value),
        "best_metrics": _sanitize_dict(best.metrics),
        "top_10_trials": top_10,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
