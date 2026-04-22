#!/usr/bin/env python3
"""Phase 73 D-19: Expanded universe + weekly rebalance + 0050 benchmark comparison.

Compares 4 variants:
  1. Phase 72 baseline (monthly rebalance, 22 symbols, hold_until)
  2. Phase 73a (weekly rebalance, ~50 symbols, hold_until)
  3. Phase 73b (monthly rebalance, ~50 symbols, hold_until)
  4. 0050 ETF buy-and-hold benchmark

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/compare_73_benchmark.py
"""

import json
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio_backtester import PortfolioBacktester
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.strategies.portfolio.fundamental_selection import (
    FundamentalSelectionConfig,
    FundamentalSelectionStrategy,
)


# Phase 72 baseline universe (22 symbols)
TW_STOCK_22 = [
    "2330", "2317", "2454", "2308", "2881", "2882", "2891", "2303",
    "1301", "1303", "2002", "2412", "3711", "2886", "6505", "2892",
    "3008", "2382", "2357", "3045", "2603", "2880",
]

# Expanded 0050 universe (~50 symbols, from poseidon/config/symbols.yaml)
TW_STOCK_50 = sorted(TW_STOCK_22 + [
    "1216", "2059", "2301", "2327", "2344", "2345", "2360", "2368",
    "2379", "2383", "2395", "2408", "2449", "2615", "2883", "2884",
    "2885", "2887", "2890", "3017", "3034", "3037", "3231", "3653",
    "3661", "4904", "4938", "5871", "5876", "5880", "6669",
])

BENCHMARK_SYMBOL = "0050"
START = date(2023, 1, 1)
END = date(2026, 4, 15)


# Shared config for all 3 strategy variants (Phase 72 winner: 4D + market_cap_weight + hold_until)
SHARED_CONFIG = {
    "strategy": "fundamental_selection",
    "name": "Fundamental Selection",
    "market": "tw_stock",
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
    "rebalance_day_of_month": 15,
    "publication_lag_days": 10,
    "hold_until": {
        "conditions": [
            {"type": "revenue_yoy_positive"},
            {"type": "max_holding_days", "value": 180},
        ],
    },
}

CONFIGS = [
    {
        "label": "Phase 72 baseline (monthly/22sym/hold_until)",
        "config": {
            **SHARED_CONFIG,
            "symbols": TW_STOCK_22,
            "rebalance_frequency": "monthly",
        },
    },
    {
        "label": "Phase 73a (weekly/50sym/hold_until)",
        "config": {
            **SHARED_CONFIG,
            "symbols": TW_STOCK_50,
            "rebalance_frequency": "weekly",
            "rebalance_day_of_week": 4,
        },
    },
    {
        "label": "Phase 73b (monthly/50sym/hold_until)",
        "config": {
            **SHARED_CONFIG,
            "symbols": TW_STOCK_50,
            "rebalance_frequency": "monthly",
        },
    },
]


def compute_benchmark_metrics(ohlcv_0050: pd.DataFrame, start: date, end: date) -> dict:
    """Compute buy-and-hold metrics for 0050 ETF (D-10, D-11: adj_close)."""
    if ohlcv_0050.empty:
        return {"label": "0050 buy-and-hold benchmark", "error": "No 0050 OHLCV data"}
    # Phase 74 D-11: use adj_close for benchmark (split/dividend adjusted)
    price_col = "adj_close" if "adj_close" in ohlcv_0050.columns else "close"
    start_price = float(ohlcv_0050.iloc[0][price_col])
    end_price = float(ohlcv_0050.iloc[-1][price_col])
    total_return = end_price / start_price - 1

    equity = ohlcv_0050[price_col] / start_price * 1_000_000
    returns = equity.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0

    cummax = equity.cummax()
    max_dd = float(((cummax - equity) / cummax).max())

    n_days = len(ohlcv_0050)
    ann_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0

    return {
        "label": "0050 buy-and-hold benchmark",
        "sharpe_ratio": round(sharpe, 4),
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "max_drawdown": round(max_dd, 4),
        "total_trades": 0,
        "buy_trades": 0,
        "sell_trades": 0,
        "hold_until_exits": 0,
        "stop_loss_exits": 0,
        "num_rebalances": 0,
    }


def compute_excess_metrics(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> dict:
    """Compute strategy vs benchmark comparison metrics (D-11)."""
    strat_returns = strategy_equity.pct_change().dropna()
    bench_returns = benchmark_equity.pct_change().dropna()

    common_idx = strat_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 10:
        return {"excess_return_annual": None, "tracking_error": None, "jensens_alpha": None, "beta": None}

    sr = strat_returns.loc[common_idx]
    br = bench_returns.loc[common_idx]

    excess_returns = sr - br
    tracking_error = float(excess_returns.std() * np.sqrt(252))

    var_br = float(np.var(br))
    beta = float(np.cov(sr, br)[0][1] / var_br) if var_br > 0 else 1.0
    alpha_daily = float(sr.mean() - beta * br.mean())
    alpha_annual = alpha_daily * 252

    return {
        "excess_return_annual": round(float(excess_returns.mean() * 252), 4),
        "tracking_error": round(tracking_error, 4),
        "jensens_alpha": round(alpha_annual, 4),
        "beta": round(beta, 4),
    }


def run_comparison() -> int:
    """Run Phase 73 4-variant portfolio backtest comparison and print results."""
    repo = RemoteDataRepository.from_settings()
    cost_model = COST_MODELS["tw_stock"]
    results = []

    # Pre-fetch OHLCV data for ALL symbols (TW_STOCK_50 superset + 0050 benchmark)
    all_symbols = sorted(set(TW_STOCK_50) | {BENCHMARK_SYMBOL})
    print(f"Fetching OHLCV data for {len(all_symbols)} symbols...")
    ohlcv_dict: dict[str, pd.DataFrame] = {}

    for symbol in all_symbols:
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

    print(f"\nOHLCV loaded for {len(ohlcv_dict)}/{len(all_symbols)} symbols\n")

    # Pop benchmark OHLCV from dict (strategies should not trade 0050)
    benchmark_ohlcv = ohlcv_dict.pop(BENCHMARK_SYMBOL, pd.DataFrame())
    if benchmark_ohlcv.empty:
        print("WARNING: No 0050 OHLCV data -- benchmark metrics will be empty")

    # Build benchmark equity series for excess metrics (Phase 74 D-11: adj_close)
    if not benchmark_ohlcv.empty:
        price_col = "adj_close" if "adj_close" in benchmark_ohlcv.columns else "close"
        bench_start_price = float(benchmark_ohlcv.iloc[0][price_col])
        benchmark_equity = benchmark_ohlcv[price_col] / bench_start_price * 1_000_000
        benchmark_equity.name = "benchmark_equity"
    else:
        benchmark_equity = pd.Series(dtype=float, name="benchmark_equity")

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

            # Compute excess metrics vs 0050 (D-11)
            if result.equity_curve and not benchmark_equity.empty:
                strat_equity = pd.Series(
                    {d: v for d, v in result.equity_curve},
                    name="strategy_equity",
                )
                strat_equity.index = pd.to_datetime(strat_equity.index)
                bench_eq = benchmark_equity.copy()
                if not isinstance(bench_eq.index, pd.DatetimeIndex):
                    bench_eq.index = pd.to_datetime(bench_eq.index)
                excess = compute_excess_metrics(strat_equity, bench_eq)
                row.update(excess)
            else:
                row.update({
                    "excess_return_annual": None,
                    "tracking_error": None,
                    "jensens_alpha": None,
                    "beta": None,
                })

            results.append(row)

            print(f"  Sharpe Ratio:      {row['sharpe_ratio']:.4f}" if row["sharpe_ratio"] else "  Sharpe Ratio: N/A")
            print(f"  Max Drawdown:      {row['max_drawdown']:.4f}" if row["max_drawdown"] else "  Max Drawdown: N/A")
            print(f"  Annualized Return: {row['annualized_return']:.4f}" if row["annualized_return"] else "  Ann Return: N/A")
            print(f"  Total Return:      {row['total_return']:.4f}" if row["total_return"] else "  Total Return: N/A")
            print(f"  Total Trades:      {row['total_trades']}")
            print(f"  Hold Until Exits:  {row['hold_until_exits']}")
            print(f"  Stop Loss Exits:   {row['stop_loss_exits']}")
            print(f"  Rebalances:        {row['num_rebalances']}")
            if row.get("jensens_alpha") is not None:
                print(f"  Jensen's Alpha:    {row['jensens_alpha']:.4f}")
                print(f"  Tracking Error:    {row['tracking_error']:.4f}")
                print(f"  Beta vs 0050:      {row['beta']:.4f}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"label": label, "error": str(exc)})

    # Compute 0050 benchmark metrics (D-10)
    benchmark_row = compute_benchmark_metrics(benchmark_ohlcv, START, END)
    results.append(benchmark_row)

    # Summary table
    print(f"\n{'='*80}")
    print("Phase 73: Expanded Universe + Weekly Rebalance + 0050 Benchmark")
    print(f"{'='*80}")
    print(f"Period: {START} to {END}")
    print(f"Universe: 22 (baseline) / {len(TW_STOCK_50)} (expanded) symbols")
    print(f"Benchmark: 0050 ETF buy-and-hold")
    print(f"Initial Capital: 1,000,000")
    print()

    # Formatted comparison table
    header = (
        f"{'Label':<45} | {'Sharpe':>8} | {'AnnRtn':>8} | {'MaxDD':>8} | "
        f"{'TotRtn':>8} | {'Trades':>6} | {'Rebals':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['label']:<45} | {'ERROR':>8} |")
            continue
        sharpe = f"{r['sharpe_ratio']:.4f}" if r.get("sharpe_ratio") is not None else "N/A"
        ann_ret = f"{r['annualized_return']:.4f}" if r.get("annualized_return") is not None else "N/A"
        max_dd = f"{r['max_drawdown']:.4f}" if r.get("max_drawdown") is not None else "N/A"
        tot_ret = f"{r['total_return']:.4f}" if r.get("total_return") is not None else "N/A"
        trades = f"{r.get('total_trades', 0):>6d}"
        rebals = f"{r.get('num_rebalances', 0):>6d}"
        print(f"{r['label']:<45} | {sharpe:>8} | {ann_ret:>8} | {max_dd:>8} | {tot_ret:>8} | {trades} | {rebals}")

    # Excess metrics section (D-11)
    print(f"\n{'='*80}")
    print("Excess Metrics vs 0050 Benchmark (D-11)")
    print(f"{'='*80}")
    ex_header = f"{'Label':<45} | {'ExRtn/yr':>8} | {'TrkErr':>8} | {'Alpha':>8} | {'Beta':>8}"
    print(ex_header)
    print("-" * len(ex_header))

    for r in results:
        if "error" in r or r.get("label") == "0050 buy-and-hold benchmark":
            continue
        ex_ret = f"{r['excess_return_annual']:.4f}" if r.get("excess_return_annual") is not None else "N/A"
        trk_err = f"{r['tracking_error']:.4f}" if r.get("tracking_error") is not None else "N/A"
        alpha = f"{r['jensens_alpha']:.4f}" if r.get("jensens_alpha") is not None else "N/A"
        beta = f"{r['beta']:.4f}" if r.get("beta") is not None else "N/A"
        print(f"{r['label']:<45} | {ex_ret:>8} | {trk_err:>8} | {alpha:>8} | {beta:>8}")

    # Look-ahead bias gate (D-18): flag Sharpe > 3.0
    print()
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
    print(f"\n{'='*80}")
    print("JSON SUMMARY")
    print(f"{'='*80}")
    print(json.dumps(results, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(run_comparison())
