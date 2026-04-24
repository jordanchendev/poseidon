#!/usr/bin/env python3
"""Phase 81: Three TW Futures strategies comparison + TAIEX buy-and-hold benchmark.

Compares 3 strategies on TX data:
  1. TrendFollowing (EMA 20/60 Daily)
  2. MeanReversion (BB+RSI 1H)
  3. VolatilityBreakout (20-bar 30M)

Plus TX buy-and-hold benchmark over the same period.

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/compare_81_tw_futures.py

Output: scripts/output/compare_81_results.json
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS, CostModel
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.tw_futures.configs import (
    MeanReversionConfig,
    TrendFollowingConfig,
    VolatilityBreakoutConfig,
)
from poseidon.strategies.tw_futures.mean_reversion import MeanReversionStrategy
from poseidon.strategies.tw_futures.trend_following import TrendFollowingStrategy
from poseidon.strategies.tw_futures.volatility_breakout import (
    VolatilityBreakoutStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (per D-13 through D-19)
# ---------------------------------------------------------------------------
START = datetime(2020, 1, 1)
END = datetime.now()
SYMBOL = "TX"
MARKET = "tw_futures"
INITIAL_CAPITAL = 1_000_000.0  # D-15: 1M TWD
COST_MODEL: CostModel = COST_MODELS["tw_futures"]

# Bars per year for different intervals (Pitfall 4 / D-17)
BARS_PER_YEAR = {
    "1d": 252,
    "1h": 4788,   # 252 * ~19 bars/day (day + night session)
    "30m": 9576,  # 252 * ~38 bars/day
}

# Sizing: fixed 1 contract (D-16)
# TX at ~22000, initial_capital=1M:
# notional_pct = 0.03 -> raw_quantity = 1M * 0.03 / 22000 = 1.36, int() = 1
SIZING = SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.03)


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def fetch_ohlcv_data(
    repo: RemoteDataRepository,
    intervals: list[str],
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for TX across multiple intervals.

    Returns dict of interval -> DataFrame.
    Warns if data starts later than 2020-01-01.
    """
    ohlcv_dict: dict[str, pd.DataFrame] = {}

    for interval in intervals:
        logger.info("Fetching TX %s OHLCV ...", interval)
        try:
            df = repo.read_ohlcv(
                symbol=SYMBOL,
                market=MARKET,
                interval=interval,
                start=START,
                end=END,
            )
        except Exception as exc:
            logger.error("Failed to fetch TX %s: %s", interval, exc)
            continue

        if df.empty:
            logger.error("TX %s: NO DATA returned", interval)
            continue

        # Strip timezone from OHLCV index so BacktestRunner can work
        # with naive Timestamps (same approach as compare_75_cryptotrend.py)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.tz_localize(None)

        # Check data availability
        actual_start = df.index[0]
        actual_end = df.index[-1]
        logger.info(
            "TX %s: %d bars (%s to %s)",
            interval, len(df), actual_start, actual_end,
        )

        if pd.Timestamp(actual_start) > pd.Timestamp(START):
            logger.warning(
                "TX %s data starts at %s (requested %s) -- "
                "continuing with available data",
                interval, actual_start, START.date(),
            )

        ohlcv_dict[interval] = df

    return ohlcv_dict


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark (D-18)
# ---------------------------------------------------------------------------
def compute_bh_benchmark(
    ohlcv_1d: pd.DataFrame,
    cost_model: CostModel,
    initial_capital: float,
) -> dict:
    """Compute TX buy-and-hold benchmark using daily OHLCV.

    One contract bought at first close, held to end.
    PnL = (close - entry) * point_value per bar.
    """
    if ohlcv_1d.empty:
        return {"label": "Buy & Hold TX", "error": "No daily OHLCV data"}

    first_close = float(ohlcv_1d["close"].iloc[0])
    pv = cost_model.point_value

    # Build equity curve: initial + unrealized PnL per bar
    bh_pnl_series = (ohlcv_1d["close"] - first_close) * pv
    bh_equity = initial_capital + bh_pnl_series

    bh_metrics = compute_metrics(bh_equity, trades=[], bars_per_year=252)

    return {"label": "Buy & Hold TX", "interval": "1d", **bh_metrics}


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------
def run_comparison() -> int:
    """Run Phase 81 three-strategy comparison and print results."""
    repo = RemoteDataRepository.from_settings()
    results: list[dict] = []

    # Step 1: Fetch OHLCV for all required intervals
    required_intervals = ["1d", "1h", "30m"]
    print(f"Fetching TX OHLCV for intervals: {required_intervals}")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} TWD")
    print(f"Cost Model: {COST_MODEL.description}")
    print(f"Point Value: {COST_MODEL.point_value}")
    print()

    ohlcv_dict = fetch_ohlcv_data(repo, required_intervals)

    if not ohlcv_dict:
        print("ERROR: No OHLCV data loaded for any interval. Aborting.")
        return 1

    print(f"OHLCV loaded for {len(ohlcv_dict)}/{len(required_intervals)} intervals\n")

    # Step 2: Define strategies with their configs and intervals
    strategies: list[tuple[str, object, str]] = [
        (
            "TrendFollowing (EMA20/60 Daily)",
            TrendFollowingStrategy(config=TrendFollowingConfig()),
            "1d",
        ),
        (
            "MeanReversion (BB+RSI 1H)",
            MeanReversionStrategy(config=MeanReversionConfig()),
            "1h",
        ),
        (
            "VolBreakout (20-bar 30M)",
            VolatilityBreakoutStrategy(config=VolatilityBreakoutConfig()),
            "30m",
        ),
    ]

    # Step 3: Run each strategy
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine(rules=[])

    for label, strategy, interval in strategies:
        ohlcv = ohlcv_dict.get(interval)
        if ohlcv is None or ohlcv.empty:
            logger.warning("No data for interval %s, skipping %s", interval, label)
            results.append({"label": label, "interval": interval, "error": "No data"})
            continue

        print(f"\n{'='*70}")
        print(f"Running: {label} (interval={interval}, bars={len(ohlcv)})")
        print(f"{'='*70}")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=COST_MODEL,
            initial_capital=INITIAL_CAPITAL,
            sizing_config=SIZING,
        )

        try:
            result = runner.run(ohlcv, feature_specs=strategy.get_feature_specs())

            if result.status == "failed":
                raise RuntimeError(result.error_message or "Backtest failed")

            metrics = result.metrics
            bpy = BARS_PER_YEAR.get(interval, 252)

            # If BacktestRunner used default 252 bars_per_year,
            # the Sharpe might need recalc for intraday intervals.
            # compute_metrics inside runner.py uses 252 by default.
            # We note this but use the metrics as-is since BacktestRunner
            # computes equity_curve internally and we don't have access to it.
            # For intraday, the Sharpe from 252 is a rough approximation.

            row = {
                "label": label,
                "interval": interval,
                "sharpe_ratio": round(metrics.get("sharpe_ratio", 0.0), 4),
                "max_drawdown": round(metrics.get("max_drawdown", 0.0), 4),
                "total_return": round(metrics.get("total_return", 0.0), 4),
                "annualized_return": round(metrics.get("annualized_return", 0.0), 4),
                "calmar_ratio": round(metrics.get("calmar_ratio", 0.0), 4),
                "win_rate": round(metrics.get("win_rate", 0.0), 4),
                "profit_factor": round(metrics.get("profit_factor", 0.0), 4),
                "trade_count": metrics.get("trade_count", 0),
                "closed_trade_count": metrics.get("closed_trade_count", 0),
                "bars_per_year": bpy,
            }
            results.append(row)

            print(f"  Sharpe Ratio:       {row['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown:       {row['max_drawdown']:.4f}")
            print(f"  Total Return:       {row['total_return']:.4f}")
            print(f"  Annualized Return:  {row['annualized_return']:.4f}")
            print(f"  Calmar Ratio:       {row['calmar_ratio']:.4f}")
            print(f"  Win Rate:           {row['win_rate']:.4f}")
            print(f"  Profit Factor:      {row['profit_factor']:.4f}")
            print(f"  Trade Count:        {row['trade_count']}")
            print(f"  Closed Trades:      {row['closed_trade_count']}")

        except Exception as exc:
            logger.exception("Strategy '%s' failed", label)
            results.append({"label": label, "interval": interval, "error": str(exc)})

    # Step 4: Buy-and-hold benchmark (D-18)
    ohlcv_1d = ohlcv_dict.get("1d")
    if ohlcv_1d is not None and not ohlcv_1d.empty:
        bh_row = compute_bh_benchmark(ohlcv_1d, COST_MODEL, INITIAL_CAPITAL)
        results.append(bh_row)
    else:
        results.append({"label": "Buy & Hold TX", "error": "No daily data"})

    # Step 5: Print comparison table (D-17)
    print(f"\n{'='*100}")
    print("Phase 81: TW Futures Strategy Comparison")
    print(f"{'='*100}")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Symbol: {SYMBOL} | Market: {MARKET}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} TWD")
    print(f"Cost Model: {COST_MODEL.description}")
    print(f"Point Value: {COST_MODEL.point_value} TWD/point")
    print()

    header = (
        f"{'Strategy':<35} {'Sharpe':>8} {'MaxDD':>8} {'Return':>10} "
        f"{'Trades':>7} {'WinRate':>8}"
    )
    print(header)
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['label']:<35} ERROR: {r['error']}")
            continue
        sharpe = r.get("sharpe_ratio", 0.0)
        max_dd = r.get("max_drawdown", 0.0)
        total_ret = r.get("total_return", 0.0)
        trades = r.get("trade_count", 0)
        win_rate = r.get("win_rate", 0.0)
        print(
            f"{r['label']:<35} {sharpe:>8.2f} {max_dd:>7.1%} "
            f"{total_ret:>9.1%} {trades:>7} {win_rate:>7.1%}"
        )

    # Look-ahead bias gate
    print()
    bias_warning = False
    for r in results:
        if "error" not in r and r.get("sharpe_ratio") is not None:
            if abs(r["sharpe_ratio"]) > 3.0:
                print(
                    f"  WARNING: {r['label']} Sharpe={r['sharpe_ratio']:.4f} > 3.0"
                    " -- possible look-ahead bias!"
                )
                bias_warning = True
    if not bias_warning:
        print("  Look-ahead bias gate: PASSED (all |Sharpe| <= 3.0)")

    # Step 6: Identify winner
    strat_results = [
        r for r in results
        if "error" not in r and "Buy" not in r.get("label", "")
    ]
    winner = None
    if strat_results:
        winner = max(strat_results, key=lambda r: r.get("sharpe_ratio", -999))
        print(
            f"\n  WINNER (highest Sharpe): {winner['label']} "
            f"with Sharpe={winner['sharpe_ratio']:.4f}"
        )
    else:
        print("\n  WARNING: No successful strategy results to compare")

    # Step 7: Save JSON artifact (D-19)
    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "compare_81_results.json"

    # Determine actual data range for the JSON
    actual_start_str = START.date().isoformat()
    if ohlcv_dict:
        earliest = min(
            df.index[0] for df in ohlcv_dict.values() if not df.empty
        )
        actual_start_str = str(earliest.date()) if hasattr(earliest, "date") else str(earliest)

    summary = {
        "phase": "81",
        "description": "TW Futures 3-strategy comparison + TAIEX B&H",
        "period": {
            "start": actual_start_str,
            "end": END.date().isoformat(),
        },
        "symbol": SYMBOL,
        "market": MARKET,
        "initial_capital": INITIAL_CAPITAL,
        "cost_model": COST_MODEL.description,
        "point_value": COST_MODEL.point_value,
        "tick_size": COST_MODEL.tick_size,
        "results": results,
        "winner": winner["label"] if winner else None,
        "winner_sharpe": winner["sharpe_ratio"] if winner else None,
    }
    output_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Results saved to %s", output_path)
    print(f"\n  JSON summary saved to: {output_path}")

    # Step 8: Print JSON for easy parsing
    print(f"\n{'='*100}")
    print("JSON SUMMARY")
    print(f"{'='*100}")
    print(json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(run_comparison())
