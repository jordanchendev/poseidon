#!/usr/bin/env python3
"""Phase 77-02: Single-symbol StructuralReversal backtest with pessimistic fill.

Validates StructuralReversalStrategy under:
  - FillModel.PESSIMISTIC (strict penetration required)
  - bars_per_year=2190 (4H crypto, not default 252)
  - Funding cost deduction for net_sharpe
  - Sharpe haircut sensitivity (30-50% range)

Output: scripts/output/bt_77_results.json
  - Per-symbol (BTCUSDT, ETHUSDT) + combined metrics
  - Both gross_sharpe and net_sharpe
  - Haircut sensitivity values

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/backtest_77_structural.py
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.portfolio import SizingConfig
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.structural_reversal import (
    StructuralReversalConfig,
    StructuralReversalStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (per CONTEXT.md decisions)
# ---------------------------------------------------------------------------
BARS_PER_YEAR = 2190  # 365.25 * 24 / 4 for 4H crypto (Pitfall 1 fix)
START = datetime(2024, 1, 1)  # Post-ETF only (D-01)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "4h"
INITIAL_CAPITAL = 100_000.0  # USDT
FUNDING_RATE_ANNUAL = 0.1095  # D-22: 0.01%/8h * 3 * 365.25
HAIRCUT_RANGE = [0.30, 0.40, 0.50]  # D-21: 30-50% Sharpe haircut sensitivity
BAR_DURATION_HOURS = 4  # 4H interval


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _load_ohlcv(repo: RemoteDataRepository, symbol: str) -> pd.DataFrame:
    """Fetch 4H OHLCV for symbol from Thalassa, filtered to post-ETF.

    Returns DataFrame with datetime index, columns: open, high, low, close, volume.
    """
    end = datetime.now()
    df = repo.read_ohlcv(
        symbol=symbol,
        market="crypto_perp",
        interval=INTERVAL,
        start=START,
        end=end,
    )
    if df.empty:
        logger.warning("No OHLCV data for %s", symbol)
        return df

    # Strip timezone if present (BacktestRunner expects naive timestamps)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.tz_localize(None)

    # Filter to post-ETF only (>= 2024-01-01)
    post_etf = pd.Timestamp("2024-01-01")
    df = df[df.index >= post_etf]

    logger.info(
        "%s: %d bars (%s to %s)",
        symbol,
        len(df),
        df.index[0] if not df.empty else "N/A",
        df.index[-1] if not df.empty else "N/A",
    )
    return df


def _make_runner(
    strategy: StructuralReversalStrategy,
) -> BacktestRunner:
    """Create BacktestRunner with pessimistic fill model for StructuralReversal.

    D-20: FillModel.PESSIMISTIC (strict penetration)
    D-14: max_pending_bars=6 (order expiry 24h at 4H)
    D-23: crypto_perp cost model (maker 0.02%, taker 0.05%, slippage 0.05%)
    """
    return BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODELS["crypto_perp"],
        initial_capital=INITIAL_CAPITAL,
        sizing_config=SizingConfig(),
        fill_model=FillModel.PESSIMISTIC,
        max_pending_bars=6,
    )


def _recompute_metrics(runner: BacktestRunner) -> dict:
    """Recompute metrics with bars_per_year=2190 (overriding runner's default 252).

    CRITICAL: BacktestRunner._run_loop uses compute_metrics with default
    bars_per_year=252 which is WRONG for 4H crypto. We must recompute with 2190.
    """
    portfolio = runner._ar_last_portfolio

    if not portfolio.equity_curve:
        return {
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0,
            "closed_trade_count": 0,
        }

    equity_series = pd.Series(
        [eq for _, eq, _ in portfolio.equity_curve],
        index=pd.DatetimeIndex([t for t, _, _ in portfolio.equity_curve]),
    )

    metrics = compute_metrics(
        equity_series,
        portfolio.trades,
        bars_per_year=BARS_PER_YEAR,
    )
    return metrics


def _compute_funding_deduction(
    metrics: dict, trades: list, total_bars: int
) -> dict:
    """Estimate funding cost and compute net_sharpe.

    Funding cost estimation:
    - For each closed trade, compute holding duration in bars
    - time_in_market_fraction = total bars in position / total bars in dataset
    - funding_cost_annual = FUNDING_RATE_ANNUAL * time_in_market_fraction
    - net_sharpe = gross_sharpe - funding_cost_sharpe_equivalent

    The funding cost reduces the annualized return, which maps to a Sharpe
    reduction of approximately funding_cost_annual / volatility. For simplicity,
    we subtract the funding cost in Sharpe units by dividing by the annualized
    volatility (or use the direct deduction from gross Sharpe).
    """
    gross_sharpe = metrics.get("sharpe_ratio", 0.0)

    # Calculate time in market from trades
    total_holding_bars = 0
    for trade in trades:
        if trade.exit_time is not None and trade.entry_time is not None:
            # Compute holding bars from timestamps
            holding_duration = trade.exit_time - trade.entry_time
            holding_hours = holding_duration.total_seconds() / 3600
            holding_bars = holding_hours / BAR_DURATION_HOURS
            total_holding_bars += holding_bars

    time_in_market_fraction = (
        total_holding_bars / total_bars if total_bars > 0 else 0.0
    )

    # Funding cost as annualized drag
    funding_cost_annual = FUNDING_RATE_ANNUAL * time_in_market_fraction

    # Direct subtraction of annualised funding cost from gross Sharpe (D-24).
    # This matches the canonical formula validated in test_net_sharpe_deduction:
    #   net_sharpe = gross_sharpe - (funding_rate_annual * time_in_market_fraction)
    funding_sharpe_deduction = funding_cost_annual
    net_sharpe = gross_sharpe - funding_sharpe_deduction

    return {
        "gross_sharpe": round(gross_sharpe, 4),
        "net_sharpe": round(net_sharpe, 4),
        "funding_cost_annual": round(funding_cost_annual, 6),
        "funding_sharpe_deduction": round(funding_sharpe_deduction, 4),
        "time_in_market_fraction": round(time_in_market_fraction, 4),
        "total_holding_bars": round(total_holding_bars, 1),
    }


def _compute_sharpe_haircuts(net_sharpe: float) -> dict:
    """Compute Sharpe haircut sensitivity for 30-50% range (D-21).

    Each haircut represents the net Sharpe after applying that percentage
    reduction, simulating real-world degradation from IS to OOS.
    """
    haircuts = {}
    for haircut in HAIRCUT_RANGE:
        pct_label = int(haircut * 100)
        haircuts[f"haircut_{pct_label}pct"] = round(
            net_sharpe * (1.0 - haircut), 4
        )
    return haircuts


def _combine_equity_curves(
    runners: list[BacktestRunner],
) -> pd.Series | None:
    """Combine equity curves from multiple symbols by summing equity values.

    Each runner contributes its equity curve; we align on timestamps and
    sum the equity values (treating each as an equal-capital sub-portfolio).
    """
    equity_series_list = []
    for runner in runners:
        portfolio = runner._ar_last_portfolio
        if portfolio.equity_curve:
            es = pd.Series(
                [eq for _, eq, _ in portfolio.equity_curve],
                index=pd.DatetimeIndex(
                    [t for t, _, _ in portfolio.equity_curve]
                ),
            )
            equity_series_list.append(es)

    if not equity_series_list:
        return None

    # Align all equity series on the union of timestamps
    combined = pd.concat(equity_series_list, axis=1)
    combined = combined.ffill().bfill()
    # Sum across symbols (combined portfolio equity)
    combined_equity = combined.sum(axis=1)
    return combined_equity


def _combine_trades(runners: list[BacktestRunner]) -> list:
    """Pool all trades from multiple runners."""
    all_trades = []
    for runner in runners:
        portfolio = runner._ar_last_portfolio
        all_trades.extend(portfolio.trades)
    return all_trades


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_backtest() -> int:
    """Run StructuralReversal pessimistic fill backtest for BTC + ETH."""
    print("=" * 80)
    print("Phase 77-02: StructuralReversal Pessimistic Fill Backtest")
    print("=" * 80)
    print(f"Period: {START.date()} to {datetime.now().date()}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Interval: {INTERVAL} | Bars/Year: {BARS_PER_YEAR}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} USDT (per symbol)")
    print(f"Fill Model: PESSIMISTIC")
    print(f"Funding Rate (annualized): {FUNDING_RATE_ANNUAL:.4f}")
    print(f"Haircut Range: {[f'{h:.0%}' for h in HAIRCUT_RANGE]}")
    print()

    # Step 1: Load data
    repo = RemoteDataRepository.from_settings()
    ohlcv_dict: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        df = _load_ohlcv(repo, symbol)
        if not df.empty:
            ohlcv_dict[symbol] = df
        else:
            print(f"  WARNING: No data for {symbol}, skipping.")

    if not ohlcv_dict:
        print("ERROR: No OHLCV data loaded. Aborting.")
        return 1

    # Step 2: Run backtest per symbol
    per_symbol_results: dict[str, dict] = {}
    runners: list[BacktestRunner] = []

    for symbol in SYMBOLS:
        if symbol not in ohlcv_dict:
            continue

        ohlcv = ohlcv_dict[symbol]
        print(f"\n{'='*60}")
        print(f"Running: {symbol} ({len(ohlcv)} bars)")
        print(f"{'='*60}")

        config = StructuralReversalConfig()  # default params
        strategy = StructuralReversalStrategy(config=config, symbol=symbol)
        runner = _make_runner(strategy)

        result = runner.run(ohlcv)

        if result.status != "completed":
            print(f"  FAILED: {result.error_message}")
            per_symbol_results[symbol] = {"error": result.error_message}
            continue

        # Recompute metrics with correct bars_per_year=2190
        metrics = _recompute_metrics(runner)

        # Compute funding deduction
        portfolio = runner._ar_last_portfolio
        total_bars = len(ohlcv)
        net_metrics = _compute_funding_deduction(
            metrics, portfolio.trades, total_bars
        )

        # Compute Sharpe haircuts
        haircuts = _compute_sharpe_haircuts(net_metrics["net_sharpe"])

        # Build per-symbol result
        sym_result = {
            **net_metrics,
            **haircuts,
            "trade_count": metrics.get("trade_count", 0),
            "closed_trade_count": metrics.get("closed_trade_count", 0),
            "max_drawdown": round(metrics.get("max_drawdown", 0.0), 4),
            "total_return": round(metrics.get("total_return", 0.0), 4),
            "annualized_return": round(
                metrics.get("annualized_return", 0.0), 4
            ),
            "win_rate": round(metrics.get("win_rate", 0.0), 4),
            "profit_factor": round(metrics.get("profit_factor", 0.0), 4),
            "calmar_ratio": round(metrics.get("calmar_ratio", 0.0), 4),
            "bars": len(ohlcv),
        }
        per_symbol_results[symbol] = sym_result
        runners.append(runner)

        # Print per-symbol summary
        print(f"  Gross Sharpe:      {sym_result['gross_sharpe']:.4f}")
        print(f"  Net Sharpe:        {sym_result['net_sharpe']:.4f}")
        print(f"  Funding Cost (yr): {sym_result['funding_cost_annual']:.6f}")
        print(f"  Time in Market:    {sym_result['time_in_market_fraction']:.4f}")
        print(f"  Max Drawdown:      {sym_result['max_drawdown']:.4f}")
        print(f"  Total Return:      {sym_result['total_return']:.4f}")
        print(f"  Trade Count:       {sym_result['trade_count']}")
        print(f"  Win Rate:          {sym_result['win_rate']:.4f}")
        for h in HAIRCUT_RANGE:
            pct = int(h * 100)
            key = f"haircut_{pct}pct"
            print(f"  Haircut {pct}%:       {sym_result[key]:.4f}")

    # Step 3: Combined metrics (pool trades + merge equity curves)
    combined_result: dict = {}
    if runners:
        combined_equity = _combine_equity_curves(runners)
        combined_trades = _combine_trades(runners)

        if combined_equity is not None and len(combined_equity) > 0:
            combined_metrics = compute_metrics(
                combined_equity,
                combined_trades,
                bars_per_year=BARS_PER_YEAR,
            )

            # Total bars = average across symbols (they should be similar)
            total_bars_avg = int(
                np.mean([len(ohlcv_dict[s]) for s in ohlcv_dict])
            )
            combined_net = _compute_funding_deduction(
                combined_metrics, combined_trades, total_bars_avg
            )
            combined_haircuts = _compute_sharpe_haircuts(
                combined_net["net_sharpe"]
            )

            combined_result = {
                **combined_net,
                **combined_haircuts,
                "trade_count": combined_metrics.get("trade_count", 0),
                "closed_trade_count": combined_metrics.get(
                    "closed_trade_count", 0
                ),
                "max_drawdown": round(
                    combined_metrics.get("max_drawdown", 0.0), 4
                ),
                "total_return": round(
                    combined_metrics.get("total_return", 0.0), 4
                ),
                "annualized_return": round(
                    combined_metrics.get("annualized_return", 0.0), 4
                ),
                "win_rate": round(
                    combined_metrics.get("win_rate", 0.0), 4
                ),
                "profit_factor": round(
                    combined_metrics.get("profit_factor", 0.0), 4
                ),
                "calmar_ratio": round(
                    combined_metrics.get("calmar_ratio", 0.0), 4
                ),
            }

    # Step 4: Print combined summary
    print(f"\n{'='*80}")
    print("COMBINED (BTC + ETH)")
    print(f"{'='*80}")
    if combined_result:
        print(f"  Gross Sharpe:      {combined_result['gross_sharpe']:.4f}")
        print(f"  Net Sharpe:        {combined_result['net_sharpe']:.4f}")
        print(f"  Funding Cost (yr): {combined_result['funding_cost_annual']:.6f}")
        print(f"  Max Drawdown:      {combined_result['max_drawdown']:.4f}")
        print(f"  Total Return:      {combined_result['total_return']:.4f}")
        print(f"  Trade Count:       {combined_result['trade_count']}")
        print(f"  Win Rate:          {combined_result['win_rate']:.4f}")
        for h in HAIRCUT_RANGE:
            pct = int(h * 100)
            key = f"haircut_{pct}pct"
            print(f"  Haircut {pct}%:       {combined_result[key]:.4f}")
    else:
        print("  No combined results (no successful per-symbol runs)")

    # Step 5: Summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    header = (
        f"{'Symbol':<12} | {'Gross SR':>9} | {'Net SR':>9} | "
        f"{'HC 30%':>8} | {'HC 40%':>8} | {'HC 50%':>8} | "
        f"{'Trades':>6} | {'MaxDD':>8} | {'WinRate':>7}"
    )
    print(header)
    print("-" * len(header))

    for symbol, r in per_symbol_results.items():
        if "error" in r:
            print(f"{symbol:<12} | ERROR: {r['error']}")
            continue
        print(
            f"{symbol:<12} | "
            f"{r['gross_sharpe']:>9.4f} | "
            f"{r['net_sharpe']:>9.4f} | "
            f"{r.get('haircut_30pct', 0.0):>8.4f} | "
            f"{r.get('haircut_40pct', 0.0):>8.4f} | "
            f"{r.get('haircut_50pct', 0.0):>8.4f} | "
            f"{r['trade_count']:>6d} | "
            f"{r['max_drawdown']:>8.4f} | "
            f"{r['win_rate']:>7.4f}"
        )

    if combined_result:
        print(
            f"{'COMBINED':<12} | "
            f"{combined_result['gross_sharpe']:>9.4f} | "
            f"{combined_result['net_sharpe']:>9.4f} | "
            f"{combined_result.get('haircut_30pct', 0.0):>8.4f} | "
            f"{combined_result.get('haircut_40pct', 0.0):>8.4f} | "
            f"{combined_result.get('haircut_50pct', 0.0):>8.4f} | "
            f"{combined_result['trade_count']:>6d} | "
            f"{combined_result['max_drawdown']:>8.4f} | "
            f"{combined_result['win_rate']:>7.4f}"
        )

    # Step 6: Write JSON output
    end_date = datetime.now().date().isoformat()
    output = {
        "phase": "77",
        "plan": "02",
        "description": "StructuralReversal pessimistic fill backtest",
        "period": {"start": "2024-01-01", "end": end_date},
        "fill_model": "pessimistic",
        "bars_per_year": BARS_PER_YEAR,
        "funding_rate_annual": FUNDING_RATE_ANNUAL,
        "initial_capital": INITIAL_CAPITAL,
        "cost_model": COST_MODELS["crypto_perp"].description,
        "per_symbol": per_symbol_results,
        "combined": combined_result,
        "haircuts": {
            "range": HAIRCUT_RANGE,
            "applied_to": "net_sharpe",
        },
    }

    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bt_77_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nJSON output saved to: {output_path}")

    # Step 7: Print JSON for easy parsing
    print(f"\n{'='*100}")
    print("JSON OUTPUT")
    print(f"{'='*100}")
    print(json.dumps(output, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(run_backtest())
