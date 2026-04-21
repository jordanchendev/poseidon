#!/usr/bin/env python3
"""Phase 72 D-17: Phase 71 baseline vs Phase 72 (hold_until + revenue trigger) comparison.

Compares:
  1. Phase 71 baseline (4D + market_cap_weight, no hold_until)
  2. Phase 72 (4D + market_cap_weight + hold_until revenue trigger + max_holding_days=180)

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/compare_72_strategies.py
"""

import json
import sys
from datetime import date, datetime

import pandas as pd

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


# Shared config for both variants (Phase 71 winner: 4D + market_cap_weight)
SHARED_CONFIG = {
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
}

CONFIGS = [
    {
        "label": "Phase 71 baseline (4D + market_cap_weight, no hold_until)",
        "config": {**SHARED_CONFIG},
    },
    {
        "label": "Phase 72 (4D + market_cap_weight + hold_until revenue trigger)",
        "config": {
            **SHARED_CONFIG,
            "hold_until": {
                "conditions": [
                    {"type": "revenue_yoy_positive"},
                    {"type": "max_holding_days", "value": 180},
                ],
            },
        },
    },
]


def run_comparison():
    """Run Phase 71 vs Phase 72 portfolio backtest comparison and print results."""
    repo = RemoteDataRepository.from_settings()
    cost_model = COST_MODELS["tw_stock"]
    results = []

    # Pre-fetch OHLCV data for all symbols (shared across both configs)
    print(f"Fetching OHLCV data for {len(TW_STOCK_SYMBOLS)} symbols...")
    ohlcv_dict: dict[str, pd.DataFrame] = {}

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

    # Run backtest for each config variant
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
            backtester = PortfolioBacktester(
                cost_model=cost_model, initial_capital=1_000_000.0
            )
            result = backtester.run(
                strategy=strategy,
                ohlcv_dict=ohlcv_dict,
                start_date=START,
                end_date=END,
            )

            if result.status == "failed":
                raise RuntimeError(result.error_message or "Backtest failed")

            metrics = result.metrics
            trades = result.trades

            # Compute trade statistics
            total_trades = len(trades)
            buy_trades = len([t for t in trades if t.get("action") == "buy"])
            sell_trades = len([t for t in trades if t.get("action") == "sell"])
            hold_until_exits = len(
                [t for t in trades if t.get("reason") == "hold_until_exit"]
            )
            stop_loss_exits = len(
                [t for t in trades if t.get("reason") == "stop_loss"]
            )

            row = {
                "label": label,
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "annualized_return": metrics.get("annualized_return"),
                "total_return": metrics.get("total_return"),
                "total_trades": total_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "hold_until_exits": hold_until_exits,
                "stop_loss_exits": stop_loss_exits,
                "num_rebalances": len(result.rebalance_log),
            }
            results.append(row)

            print(f"  Sharpe Ratio:      {row['sharpe_ratio']:.4f}" if row["sharpe_ratio"] else "  Sharpe Ratio: N/A")
            print(f"  Max Drawdown:      {row['max_drawdown']:.4f}" if row["max_drawdown"] else "  Max Drawdown: N/A")
            print(f"  Annualized Return: {row['annualized_return']:.4f}" if row["annualized_return"] else "  Ann Return: N/A")
            print(f"  Total Return:      {row['total_return']:.4f}" if row["total_return"] else "  Total Return: N/A")
            print(f"  Total Trades:      {row['total_trades']}")
            print(f"  Hold Until Exits:  {row['hold_until_exits']}")
            print(f"  Stop Loss Exits:   {row['stop_loss_exits']}")
            print(f"  Rebalances:        {row['num_rebalances']}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"label": label, "error": str(exc)})

    # Summary table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY (D-17: Phase 71 vs Phase 72)")
    print(f"{'='*60}")
    print(f"Period: {START} to {END}")
    print(f"Universe: {len(TW_STOCK_SYMBOLS)} TW stock symbols")
    print(f"Initial Capital: 1,000,000")
    print()

    # Formatted comparison table
    header = f"{'Metric':<25} | {'Phase 71 (baseline)':<25} | {'Phase 72 (hold_until)':<25}"
    print(header)
    print("-" * len(header))

    if len(results) == 2 and "error" not in results[0] and "error" not in results[1]:
        r71, r72 = results[0], results[1]
        metrics_to_show = [
            ("Sharpe Ratio", "sharpe_ratio", ".4f"),
            ("Max Drawdown", "max_drawdown", ".4f"),
            ("Annualized Return", "annualized_return", ".4f"),
            ("Total Return", "total_return", ".4f"),
            ("Total Trades", "total_trades", "d"),
            ("Buy Trades", "buy_trades", "d"),
            ("Sell Trades", "sell_trades", "d"),
            ("Hold Until Exits", "hold_until_exits", "d"),
            ("Stop Loss Exits", "stop_loss_exits", "d"),
            ("Rebalances", "num_rebalances", "d"),
        ]
        for metric_name, key, fmt in metrics_to_show:
            v71 = r71.get(key)
            v72 = r72.get(key)
            s71 = f"{v71:{fmt}}" if v71 is not None else "N/A"
            s72 = f"{v72:{fmt}}" if v72 is not None else "N/A"
            print(f"{metric_name:<25} | {s71:<25} | {s72:<25}")

    print()

    # Look-ahead bias gate (D-18): flag Sharpe > 3.0
    bias_warning = False
    for r in results:
        if "error" not in r and r.get("sharpe_ratio") is not None:
            if r["sharpe_ratio"] > 3.0:
                print(
                    f"  WARNING (D-18): {r['label']} Sharpe={r['sharpe_ratio']:.4f} > 3.0"
                    " -- possible look-ahead bias!"
                )
                bias_warning = True

    if not bias_warning:
        print("  Look-ahead bias gate (D-18): PASSED (all Sharpe <= 3.0)")

    # JSON summary for easy parsing
    print(f"\n{'='*60}")
    print("JSON SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    run_comparison()
