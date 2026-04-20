#!/usr/bin/env python3
"""Phase 71 D-14: 4-way PortfolioBacktester comparison.

Compares:
  1. 3-dimension + equal_weight (Phase 67 baseline)
  2. 4-dimension + equal_weight (momentum added)
  3. 3-dimension + market_cap_weight (allocation changed)
  4. 4-dimension + market_cap_weight (full Phase 71)

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/compare_71_strategies.py
"""

import json
import sys
from datetime import date, datetime

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio_backtester import PortfolioBacktester
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.strategies.portfolio.fundamental_selection import (
    FundamentalSelectionConfig,
    FundamentalSelectionStrategy,
)


# TW stock universe (same as poseidon/config/symbols.yaml tw_stock)
TW_STOCK_SYMBOLS = [
    "2330", "2317", "2454", "2308", "2881", "2882", "2891", "2303",
    "1301", "1303", "2002", "2412", "3711", "2886", "6505", "2892",
    "3008", "2382", "2357", "3045", "2603", "2880",
]

START = date(2023, 1, 1)
END = date(2026, 4, 15)


# 4 config variants per D-14
CONFIGS = [
    {
        "label": "3D + equal_weight (Phase 67 baseline)",
        "config": {
            "strategy": "fundamental_selection",
            "name": "Fundamental Selection",
            "market": "tw_stock",
            "symbols": TW_STOCK_SYMBOLS,
            "scoring": {
                "quality_weight": 0.333,
                "growth_weight": 0.333,
                "flow_weight": 0.334,
                "momentum_weight": 0.0,
            },
            "max_stocks": 10,
            "min_score": 0.0,
            "allocation_method": "equal_weight",
            "position_limit_pct": 0.10,
            "stop_loss_pct": 0.10,
            "rebalance_frequency": "monthly",
            "rebalance_day_of_month": 15,
            "publication_lag_days": 10,
        },
    },
    {
        "label": "4D + equal_weight (momentum added)",
        "config": {
            "strategy": "fundamental_selection",
            "name": "Fundamental Selection",
            "market": "tw_stock",
            "symbols": TW_STOCK_SYMBOLS,
            "scoring": {
                "quality_weight": 0.25,
                "growth_weight": 0.25,
                "flow_weight": 0.25,
                "momentum_weight": 0.25,
            },
            "max_stocks": 10,
            "min_score": 0.0,
            "allocation_method": "equal_weight",
            "position_limit_pct": 0.10,
            "stop_loss_pct": 0.10,
            "rebalance_frequency": "monthly",
            "rebalance_day_of_month": 15,
            "publication_lag_days": 10,
        },
    },
    {
        "label": "3D + market_cap_weight (allocation changed)",
        "config": {
            "strategy": "fundamental_selection",
            "name": "Fundamental Selection",
            "market": "tw_stock",
            "symbols": TW_STOCK_SYMBOLS,
            "scoring": {
                "quality_weight": 0.333,
                "growth_weight": 0.333,
                "flow_weight": 0.334,
                "momentum_weight": 0.0,
            },
            "max_stocks": 10,
            "min_score": 0.0,
            "allocation_method": "market_cap_weight",
            "position_limit_pct": 0.20,
            "stop_loss_pct": 0.10,
            "rebalance_frequency": "monthly",
            "rebalance_day_of_month": 15,
            "publication_lag_days": 10,
        },
    },
    {
        "label": "4D + market_cap_weight (full Phase 71)",
        "config": {
            "strategy": "fundamental_selection",
            "name": "Fundamental Selection",
            "market": "tw_stock",
            "symbols": TW_STOCK_SYMBOLS,
            "scoring": {
                "quality_weight": 0.25,
                "growth_weight": 0.25,
                "flow_weight": 0.25,
                "momentum_weight": 0.25,
            },
            "max_stocks": 10,
            "min_score": 0.0,
            "allocation_method": "market_cap_weight",
            "position_limit_pct": 0.20,
            "stop_loss_pct": 0.10,
            "rebalance_frequency": "monthly",
            "rebalance_day_of_month": 15,
            "publication_lag_days": 10,
        },
    },
]


def run_comparison():
    """Run 4-way portfolio backtest comparison and print results."""
    repo = RemoteDataRepository.from_settings()
    cost_model = COST_MODELS["tw_stock"]
    results = []

    # Pre-fetch OHLCV data for all symbols (shared across all 4 configs)
    print(f"Fetching OHLCV data for {len(TW_STOCK_SYMBOLS)} symbols...")
    ohlcv_dict: dict[str, "pd.DataFrame"] = {}
    import pandas as pd

    for symbol in TW_STOCK_SYMBOLS:
        df = repo.read_ohlcv(
            symbol=symbol,
            market="tw_stock",
            interval="1d",
            start=datetime(START.year, START.month, START.day),
            end=datetime(END.year, END.month, END.day),
        )
        if not df.empty:
            ohlcv_dict[symbol] = df
            print(f"  {symbol}: {len(df)} bars")
        else:
            print(f"  {symbol}: NO DATA")

    print(f"\nOHLCV loaded for {len(ohlcv_dict)}/{len(TW_STOCK_SYMBOLS)} symbols\n")

    # Run 4 backtest variants
    for entry in CONFIGS:
        label = entry["label"]
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        try:
            # Build strategy from config
            cfg = FundamentalSelectionConfig(**entry["config"])
            strategy = FundamentalSelectionStrategy(config=cfg, repo=repo)

            # Run backtest
            backtester = PortfolioBacktester(cost_model=cost_model)
            result = backtester.run(
                strategy=strategy,
                ohlcv_dict=ohlcv_dict,
                start_date=START,
                end_date=END,
            )

            if result.status == "failed":
                raise RuntimeError(result.error_message or "Backtest failed")

            metrics = result.metrics
            row = {
                "label": label,
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "annualized_return": metrics.get("annualized_return"),
                "total_return": metrics.get("total_return"),
                "calmar_ratio": metrics.get("calmar_ratio"),
                "trade_count": metrics.get("trade_count"),
                "num_rebalances": len(result.rebalance_log),
            }
            results.append(row)

            print(f"  Sharpe Ratio:      {row['sharpe_ratio']:.4f}" if row['sharpe_ratio'] else "  Sharpe Ratio: N/A")
            print(f"  Max Drawdown:      {row['max_drawdown']:.4f}" if row['max_drawdown'] else "  Max Drawdown: N/A")
            print(f"  Annualized Return: {row['annualized_return']:.4f}" if row['annualized_return'] else "  Ann Return: N/A")
            print(f"  Total Return:      {row['total_return']:.4f}" if row['total_return'] else "  Total Return: N/A")
            print(f"  Calmar Ratio:      {row['calmar_ratio']:.4f}" if row['calmar_ratio'] else "  Calmar Ratio: N/A")
            print(f"  Rebalances:        {row['num_rebalances']}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"label": label, "error": str(exc)})

    # Summary table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY (D-14)")
    print(f"{'='*60}")
    print(f"Period: {START} to {END}")
    print(f"Universe: {len(TW_STOCK_SYMBOLS)} TW stock symbols")
    print()

    # Look-ahead bias gate (T-71-10): flag Sharpe > 3.0
    for r in results:
        if "error" not in r and r.get("sharpe_ratio") is not None:
            if r["sharpe_ratio"] > 3.0:
                print(f"  WARNING: {r['label']} Sharpe={r['sharpe_ratio']:.4f} > 3.0 -- possible look-ahead bias!")

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run_comparison()
