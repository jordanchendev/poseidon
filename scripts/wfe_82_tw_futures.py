#!/usr/bin/env python3
"""Phase 82: Walk-Forward Efficiency validation for TrendFollowing TX daily.

Reads best parameters from optuna_82_results.json (Plan 01 output).
Runs quarterly rolling WFE with train_days=252, test_days=63, step_days=63.
Computes per-window IS/OOS metrics, aggregate WFE, and OOS-matched B&H Sharpe.

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/wfe_82_tw_futures.py

Expected runtime: ~5 minutes (16 windows x 2 backtests each)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    compute_wfe,
)
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
# Configuration (same data range as optuna_82)
# ---------------------------------------------------------------------------
START = datetime(2021, 3, 22)  # Earliest TX daily data (D-02)
END = datetime.now()
SYMBOL = "TX"
MARKET = "tw_futures"
INITIAL_CAPITAL = 1_000_000.0  # 1M TWD
INTERVAL = "1d"
BARS_PER_YEAR = 252
COST_MODEL = COST_MODELS["tw_futures"]
SIZING = SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.03)

# Walk-forward configuration (D-04, D-05, D-06)
WF_CONFIG = WalkForwardConfig(
    train_days=252,          # D-04: 1 year IS
    test_days=63,            # D-04: 1 quarter OOS
    step_days=63,            # D-04: roll by 1 quarter
    min_trades_per_oos=5,    # D-05
    min_wfe=0.50,            # D-06
)

OPTUNA_JSON = Path(__file__).parent / "output" / "optuna_82_results.json"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "wfe_82_results.json"


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
# Single backtest helper
# ---------------------------------------------------------------------------
def _run_single_backtest(
    config: TrendFollowingConfig,
    ohlcv: pd.DataFrame,
) -> dict:
    """Run a single backtest with a FRESH strategy instance (Pitfall 3).

    Creates a new TrendFollowingStrategy each call to prevent state leakage
    between IS and OOS runs.

    Returns metrics dict on success, or safe defaults on failure.
    """
    strategy = TrendFollowingStrategy(config=config)
    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
        sizing_config=SIZING,
    )

    try:
        result = runner.run(ohlcv, feature_specs=strategy.get_feature_specs())
        if result.status == "failed":
            logger.warning("Backtest failed: %s", result.error_message)
            return {
                "sharpe_ratio": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0,
                "total_return": 0.0,
            }
        return result.metrics
    except Exception as exc:
        logger.warning("Backtest exception: %s", exc)
        return {
            "sharpe_ratio": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "total_return": 0.0,
        }


# ---------------------------------------------------------------------------
# OOS-matched B&H Sharpe (Pitfall 4 fix)
# ---------------------------------------------------------------------------
def compute_bh_oos_sharpe(
    ohlcv: pd.DataFrame,
    windows: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[float, list[float]]:
    """Compute B&H Sharpe over the SAME OOS windows (not full period).

    For each OOS window, builds a buy-and-hold equity curve:
        equity = INITIAL_CAPITAL + (close - first_close) * point_value

    Then compute_metrics() gives per-window B&H Sharpe.
    Returns (average_bh_sharpe, list_of_per_window_sharpes).
    """
    point_value = COST_MODEL.point_value  # 200.0 for TX
    per_window_sharpes: list[float] = []

    for _i, ((_train_s, _train_e), (test_s, test_e)) in enumerate(windows):
        oos_ohlcv = ohlcv.iloc[test_s:test_e]
        if oos_ohlcv.empty or len(oos_ohlcv) < 2:
            per_window_sharpes.append(0.0)
            continue

        first_close = float(oos_ohlcv["close"].iloc[0])
        bh_pnl = (oos_ohlcv["close"] - first_close) * point_value
        bh_equity = INITIAL_CAPITAL + bh_pnl

        bh_metrics = compute_metrics(bh_equity, trades=[], bars_per_year=BARS_PER_YEAR)
        per_window_sharpes.append(bh_metrics.get("sharpe_ratio", 0.0))

    avg_bh_sharpe = (
        sum(per_window_sharpes) / len(per_window_sharpes)
        if per_window_sharpes
        else 0.0
    )
    return avg_bh_sharpe, per_window_sharpes


# ---------------------------------------------------------------------------
# Main WFE loop
# ---------------------------------------------------------------------------
def run_wfe(
    ohlcv: pd.DataFrame,
    best_config: TrendFollowingConfig,
) -> dict:
    """Run walk-forward efficiency validation with quarterly rolling windows.

    Uses generate_windows() + manual loop pattern (NOT analyzer.analyze()).
    Pitfall 1: analyzer.analyze() requires a factory, manual loop is simpler.
    Pitfall 3: FRESH strategy instance per IS and OOS run.
    """
    analyzer = WalkForwardAnalyzer(
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
    )

    windows = analyzer.generate_windows(len(ohlcv), WF_CONFIG)

    if not windows:
        logger.warning("No WFE windows generated (data_length=%d)", len(ohlcv))
        return {
            "windows": [],
            "avg_wfe": 0.0,
            "oos_aggregate_sharpe": 0.0,
            "oos_aggregate_max_drawdown": 0.0,
            "total_oos_trades": 0,
            "n_windows": 0,
            "n_insufficient": 0,
            "insufficient_flagged": False,
            "passed": False,
        }

    print(f"\n  WFE: {len(windows)} windows, data_length={len(ohlcv)}")
    print(
        f"  Config: train={WF_CONFIG.train_days} bars, "
        f"test={WF_CONFIG.test_days} bars, step={WF_CONFIG.step_days} bars"
    )

    window_results: list[dict] = []
    for i, ((train_s, train_e), (test_s, test_e)) in enumerate(windows):
        # Slice OHLCV for IS and OOS
        is_ohlcv = ohlcv.iloc[train_s:train_e].copy()
        oos_ohlcv = ohlcv.iloc[test_s:test_e].copy()

        # Run IS and OOS backtests with FRESH strategy instances
        is_metrics = _run_single_backtest(best_config, is_ohlcv)
        oos_metrics = _run_single_backtest(best_config, oos_ohlcv)

        # Compute WFE using annualized return (NOT Sharpe)
        wfe = compute_wfe(
            is_metrics.get("annualized_return", 0.0),
            oos_metrics.get("annualized_return", 0.0),
        )

        window_result = {
            "window": i,
            "train_range": [train_s, train_e],
            "test_range": [test_s, test_e],
            "is_sharpe": is_metrics.get("sharpe_ratio", 0.0),
            "oos_sharpe": oos_metrics.get("sharpe_ratio", 0.0),
            "is_ann_return": is_metrics.get("annualized_return", 0.0),
            "oos_ann_return": oos_metrics.get("annualized_return", 0.0),
            "is_max_drawdown": is_metrics.get("max_drawdown", 0.0),
            "oos_max_drawdown": oos_metrics.get("max_drawdown", 0.0),
            "is_trade_count": is_metrics.get("trade_count", 0),
            "oos_trade_count": oos_metrics.get("trade_count", 0),
            "wfe": wfe,
        }
        window_results.append(window_result)

        print(
            f"  Window {i}: IS_ann={is_metrics.get('annualized_return', 0):.4f}, "
            f"OOS_ann={oos_metrics.get('annualized_return', 0):.4f}, "
            f"IS_sharpe={is_metrics.get('sharpe_ratio', 0):.4f}, "
            f"OOS_sharpe={oos_metrics.get('sharpe_ratio', 0):.4f}, "
            f"WFE={wfe:.4f}"
        )

    # Compute aggregate metrics
    wfe_values = [w["wfe"] for w in window_results]
    oos_sharpes = [w["oos_sharpe"] for w in window_results]
    oos_max_dds = [w["oos_max_drawdown"] for w in window_results]
    oos_trades = [w["oos_trade_count"] for w in window_results]

    avg_wfe = sum(wfe_values) / len(wfe_values) if wfe_values else 0.0
    oos_aggregate_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0.0
    oos_aggregate_max_drawdown = max(oos_max_dds) if oos_max_dds else 0.0
    total_oos_trades = sum(oos_trades)

    # Check insufficient windows (D-05)
    n_insufficient = sum(1 for t in oos_trades if t < WF_CONFIG.min_trades_per_oos)
    insufficient_ratio = n_insufficient / len(windows) if windows else 0.0
    insufficient_flagged = insufficient_ratio > WF_CONFIG.max_insufficient_ratio

    passed = avg_wfe >= WF_CONFIG.min_wfe

    print(f"\n  Aggregate WFE:       {avg_wfe:.4f} ({'PASSED' if passed else 'FAILED'})")
    print(f"  OOS Aggregate Sharpe: {oos_aggregate_sharpe:.4f}")
    print(f"  OOS Max Drawdown:     {oos_aggregate_max_drawdown:.4f}")
    print(f"  Total OOS Trades:     {total_oos_trades}")
    print(f"  Insufficient windows: {n_insufficient}/{len(windows)} (flagged={insufficient_flagged})")

    return {
        "windows": window_results,
        "avg_wfe": avg_wfe,
        "oos_aggregate_sharpe": oos_aggregate_sharpe,
        "oos_aggregate_max_drawdown": oos_aggregate_max_drawdown,
        "total_oos_trades": total_oos_trades,
        "n_windows": len(windows),
        "n_insufficient": n_insufficient,
        "insufficient_flagged": insufficient_flagged,
        "passed": passed,
    }


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


def _sanitize(obj):
    """Recursively sanitize an object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return _to_native(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run Phase 82 WFE validation for TrendFollowing TX daily."""
    # Step 1: Load best params from Optuna results
    print(f"Phase 82: WFE Validation for TrendFollowing TX Daily")
    print(f"{'='*60}")

    if not OPTUNA_JSON.exists():
        print(f"ERROR: {OPTUNA_JSON} not found. Run optuna_82_tw_futures.py first.")
        return 1

    with open(OPTUNA_JSON) as f:
        optuna_data = json.load(f)

    best_params = optuna_data["best_params"]
    print(f"Loaded best params from {OPTUNA_JSON.name}:")
    print(f"  ema_fast={best_params['ema_fast']}, ema_slow={best_params['ema_slow']}, "
          f"atr_period={best_params['atr_period']}, atr_multiplier={best_params['atr_multiplier']}")
    print(f"  Best IS Sharpe: {optuna_data.get('best_sharpe', 'N/A')}")
    print()

    # Construct config from best params
    best_config = TrendFollowingConfig(
        ema_fast=best_params["ema_fast"],
        ema_slow=best_params["ema_slow"],
        atr_period=best_params["atr_period"],
        atr_multiplier=best_params["atr_multiplier"],
    )

    # Step 2: Fetch OHLCV
    repo = RemoteDataRepository.from_settings()
    ohlcv = fetch_ohlcv(repo)
    print(f"OHLCV loaded: {len(ohlcv)} bars ({ohlcv.index[0]} to {ohlcv.index[-1]})")
    print()

    # Step 3: Run WFE validation
    print(f"{'='*60}")
    print("Running Walk-Forward Efficiency Validation")
    print(f"{'='*60}")
    wfe_results = run_wfe(ohlcv, best_config)

    # Step 4: Compute B&H OOS Sharpe (Pitfall 4: over same OOS windows)
    print(f"\n{'='*60}")
    print("Computing B&H OOS Sharpe (OOS-matched windows)")
    print(f"{'='*60}")

    # Regenerate windows for B&H computation
    analyzer = WalkForwardAnalyzer(
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODEL,
    )
    windows = analyzer.generate_windows(len(ohlcv), WF_CONFIG)
    bh_oos_sharpe, bh_per_window_sharpes = compute_bh_oos_sharpe(ohlcv, windows)

    print(f"  B&H OOS Sharpe (avg): {bh_oos_sharpe:.4f}")
    for i, s in enumerate(bh_per_window_sharpes):
        print(f"    Window {i}: B&H Sharpe={s:.4f}")

    # Step 5: Build output JSON
    output = {
        "phase": "82",
        "script": "wfe_82_tw_futures.py",
        "best_params": _sanitize(best_params),
        "wfe_config": {
            "train_days": WF_CONFIG.train_days,
            "test_days": WF_CONFIG.test_days,
            "step_days": WF_CONFIG.step_days,
            "min_trades_per_oos": WF_CONFIG.min_trades_per_oos,
            "min_wfe": WF_CONFIG.min_wfe,
        },
        "wfe_results": _sanitize(wfe_results),
        "bh_oos_sharpe": _sanitize(bh_oos_sharpe),
        "bh_per_window_sharpes": _sanitize(bh_per_window_sharpes),
        "period": {
            "start": START.date().isoformat(),
            "end": END.date().isoformat(),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Step 6: Print summary
    print(f"\n{'='*60}")
    print("Phase 82 WFE Validation Summary")
    print(f"{'='*60}")
    print(f"  Aggregate WFE:        {wfe_results['avg_wfe']:.4f}")
    print(f"  OOS Aggregate Sharpe: {wfe_results['oos_aggregate_sharpe']:.4f}")
    print(f"  OOS Max Drawdown:     {wfe_results['oos_aggregate_max_drawdown']:.4f}")
    print(f"  B&H OOS Sharpe:       {bh_oos_sharpe:.4f}")
    print(f"  Total OOS Trades:     {wfe_results['total_oos_trades']}")
    print(f"  Windows:              {wfe_results['n_windows']}")
    print(f"  WFE Passed:           {'YES' if wfe_results['passed'] else 'NO'}")
    print(f"\n  Results saved to {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
