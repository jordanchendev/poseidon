#!/usr/bin/env python3
"""Funding rate contrarian backtest — thesis test on BTC/ETH perp.

Pulls daily OHLCV + daily funding rate from Thalassa, runs FundingRateStrategy
through BacktestRunner, reports two PnL views:

  1. Price-only PnL  (BacktestRunner.metrics.total_pnl)
  2. Price + funding PnL  (post-process: +Σ funding_rate_daily × notional × side)

For each hold-day:
    side=+1 if long, -1 if short
    daily_funding_pnl = -side × notional × funding_rate_daily
    (long pays positive funding; short receives positive funding)

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/backtest_funding_rate.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.funding_rate_strategy import FundingRateStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START = datetime(2024, 1, 1)
END = datetime.now()
INITIAL_CAPITAL = 100_000.0
INTERVAL = "1d"
COST_MODEL = COST_MODELS["crypto_perp"]

# Symbol mapping: OHLCV uses "BTCUSDT", funding API uses "BTC/USDT:USDT"
SYMBOLS = [
    {"ohlcv": "BTCUSDT", "funding": "BTC/USDT:USDT"},
    {"ohlcv": "ETHUSDT", "funding": "ETH/USDT:USDT"},
]

# Strategy params (data-driven from p95/p5 of BTC funding distribution 2024-01-01..2026-04-30)
STRAT_PARAMS = {
    "entry_high": 0.0006,  # ~p95: ~0.06% daily ≈ 22% annualised
    "entry_low": -0.0001,  # ~p5
    "exit_band": 0.0002,  # close when |funding| < 0.02% daily
    "position_pct": 0.5,  # 50% notional per signal
    "hold_max_bars": 10,  # max 10-day hold
}


def fetch_data(repo: RemoteDataRepository, ohlcv_symbol: str, funding_symbol: str) -> pd.DataFrame:
    """Pull OHLCV + funding, merge on date, return DataFrame indexed by time."""
    ohlcv = repo.read_ohlcv(
        symbol=ohlcv_symbol,
        market="crypto_perp",
        interval=INTERVAL,
        start=START,
        end=END,
    )
    if ohlcv.empty:
        raise RuntimeError(f"No OHLCV for {ohlcv_symbol}")

    funding = repo.read_funding_rates(symbol=funding_symbol, start=START.strftime("%Y-%m-%d"))
    if funding.empty:
        raise RuntimeError(f"No funding for {funding_symbol}")

    # Normalise both to date keys for merge.
    # `read_funding_rates` returns DataFrame indexed by `date`; OHLCV is indexed by `time`.
    if ohlcv.index.name == "time":
        ohlcv = ohlcv.reset_index()
    ohlcv["date"] = pd.to_datetime(ohlcv["time"]).dt.tz_localize(None).dt.normalize()

    funding_df = funding.reset_index()
    date_col = "date" if "date" in funding_df.columns else funding_df.columns[0]
    funding_df["date"] = pd.to_datetime(funding_df[date_col]).dt.tz_localize(None).dt.normalize()

    merged = ohlcv.merge(funding_df[["date", "funding_rate_daily"]], on="date", how="left")
    merged = merged.set_index("time").drop(columns=["date"])
    merged.index = pd.to_datetime(merged.index).tz_localize(None)

    # Forward-fill missing funding (Thalassa may have gaps); drop rows with no funding at all
    merged["funding_rate_daily"] = merged["funding_rate_daily"].ffill()
    merged = merged.dropna(subset=["funding_rate_daily"])

    logger.info(
        "%s: %d bars, %s..%s, funding_rate_daily mean=%.6f stdev=%.6f",
        ohlcv_symbol,
        len(merged),
        merged.index.min().date(),
        merged.index.max().date(),
        merged["funding_rate_daily"].mean(),
        merged["funding_rate_daily"].std(),
    )
    return merged


def post_process_funding_pnl(trades: list[dict], data: pd.DataFrame) -> dict:
    """Add daily funding payments for each trade's holding period.

    For each trade:
        side = +1 if "long" / -1 if "short" (action field)
        notional ≈ entry_price × quantity (constant for closed trade)
        For each day in [entry_time, exit_time]:
            daily_pnl = -side × notional × funding_rate_daily(day)
        Sum across days, add to trade pnl.
    """
    funding_pnl_total = 0.0
    funding_pnl_by_trade = []

    for t in trades:
        action = t.get("action", "").lower()
        if action == "long":
            side = 1
        elif action == "short":
            side = -1
        else:
            funding_pnl_by_trade.append(0.0)
            continue

        entry_time = pd.to_datetime(t["entry_time"]).tz_localize(None) if t.get("entry_time") else None
        exit_time = pd.to_datetime(t["exit_time"]).tz_localize(None) if t.get("exit_time") else None
        if entry_time is None or exit_time is None:
            funding_pnl_by_trade.append(0.0)
            continue

        notional = float(t["entry_price"]) * float(t["quantity"])
        # Slice funding rates over the hold period (inclusive)
        mask = (data.index >= entry_time) & (data.index < exit_time)
        funding_slice = data.loc[mask, "funding_rate_daily"]
        # Long pays positive funding; short receives positive funding
        trade_funding_pnl = float(-side * notional * funding_slice.sum())

        funding_pnl_by_trade.append(trade_funding_pnl)
        funding_pnl_total += trade_funding_pnl

    return {
        "funding_pnl_total": funding_pnl_total,
        "funding_pnl_by_trade": funding_pnl_by_trade,
    }


def run_one(repo: RemoteDataRepository, sym_pair: dict) -> dict:
    ohlcv_symbol = sym_pair["ohlcv"]
    funding_symbol = sym_pair["funding"]

    data = fetch_data(repo, ohlcv_symbol, funding_symbol)

    strategy = FundingRateStrategy(
        symbol=ohlcv_symbol,
        market="crypto_perp",
        interval=INTERVAL,
        **STRAT_PARAMS,
    )
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine(rules=[])

    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
    )

    result = runner.run(data)
    funding = post_process_funding_pnl(result.trades, data)

    metrics = dict(result.metrics)
    price_only_pnl = metrics.get("total_pnl", 0.0)
    funding_pnl = funding["funding_pnl_total"]
    combined_pnl = price_only_pnl + funding_pnl

    return {
        "symbol": ohlcv_symbol,
        "bars": len(data),
        "trade_count": result.trade_count,
        "price_only_metrics": {
            "total_return": metrics.get("total_return"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
            "total_pnl": price_only_pnl,
        },
        "funding_pnl_total": funding_pnl,
        "combined_pnl": combined_pnl,
        "combined_return_pct": combined_pnl / INITIAL_CAPITAL * 100.0,
        "trades_sample": result.trades[:5],
    }


def main():
    repo = RemoteDataRepository.from_settings()
    out = {"params": STRAT_PARAMS, "results": []}
    for sym_pair in SYMBOLS:
        logger.info("=== %s ===", sym_pair["ohlcv"])
        try:
            res = run_one(repo, sym_pair)
            out["results"].append(res)
            print(f"\n=== {res['symbol']} ===")
            print(f"  bars: {res['bars']}, trades: {res['trade_count']}")
            print(
                f"  price-only: total_return={res['price_only_metrics']['total_return']:.4f}, "
                f"sharpe={res['price_only_metrics']['sharpe_ratio']:.3f}, "
                f"maxdd={res['price_only_metrics']['max_drawdown']:.4f}, "
                f"win_rate={res['price_only_metrics']['win_rate']:.3f}"
            )
            print(f"  funding_pnl_total: ${res['funding_pnl_total']:.2f}")
            print(f"  combined_pnl: ${res['combined_pnl']:.2f}  ({res['combined_return_pct']:.2f}%)")
        except Exception as e:
            logger.exception("Failed %s", sym_pair["ohlcv"])
            out["results"].append({"symbol": sym_pair["ohlcv"], "error": str(e)})

    out_path = Path("scripts/output/funding_rate_backtest_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
