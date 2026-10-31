#!/usr/bin/env python3
"""Run a DSL v2 spec (default: the TX ORB30 example) on real Thalassa bars.

Prints every round trip (entry/exit time, prices, exit kind, PnL, fees) and
totals so the result can be reconciled by hand. Read-only: fetches OHLCV from
Thalassa, writes nothing.

Run in an isolated container on stormtrooper, e.g.:
  python scripts/backtest_dsl_v2_orb.py --start 2026-08-01 --end 2026-10-01
  python scripts/backtest_dsl_v2_orb.py --spec my_spec.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", help="path to a DSL v2 JSON spec (default: tx_orb30_day example)")
    p.add_argument("--start", default="2026-08-01", help="UTC start date (inclusive)")
    p.add_argument("--end", default="2026-10-01", help="UTC end date (exclusive)")
    p.add_argument("--cost-model", default=None, help="COST_MODELS key (default: the spec's market)")
    p.add_argument("--sizing", default="fixed_notional", choices=[m.value for m in SizingMode])
    p.add_argument("--contracts", type=int, default=1, help="contracts per entry for --sizing fixed_contracts")
    p.add_argument("--gap-aware-stops", action="store_true", help="fill gapped stops at the worse open")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--json", action="store_true", help="print a JSON summary instead of a table")
    return p.parse_args(argv)


def _sizing(args: argparse.Namespace) -> SizingConfig:
    mode = SizingMode(args.sizing)
    if mode.value == "fixed_contracts":
        return SizingConfig(mode=mode, contracts=args.contracts)
    return SizingConfig(mode=mode)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doc = json.loads(Path(args.spec).read_text()) if args.spec else tx_orb30_day()
    strategy = RuleStrategyV2(config=doc)
    cost_model = COST_MODELS[args.cost_model or strategy.market]

    repo = RemoteDataRepository.from_settings()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    ohlcv = repo.read_ohlcv(strategy.symbol, strategy.market, strategy.interval, start=start, end=end)
    ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]

    runner_kwargs = {"gap_aware_stops": True} if args.gap_aware_stops else {}
    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(),
        cost_model=cost_model,
        initial_capital=args.capital,
        sizing_config=_sizing(args),
        **runner_kwargs,
    )
    t0 = time.perf_counter()
    result = runner.run(ohlcv)
    elapsed = time.perf_counter() - t0
    if result.status != "completed":
        print(f"backtest failed: {result.error_message}", file=sys.stderr)
        return 1

    sides = {t["entry_time"]: t["action"] for t in result.trades if t["exit_time"] is None}
    trips = [t for t in result.trades if t["exit_time"] is not None]
    total_pnl = sum(t["pnl"] for t in trips)
    total_fees = sum(t["fees"] for t in trips)
    summary = {
        "spec": strategy.name,
        "bars": len(ohlcv),
        "first_bar": str(ohlcv.index.min()),
        "last_bar": str(ohlcv.index.max()),
        "cost_model": cost_model.market,
        "sizing": args.sizing,
        "gap_aware_stops": bool(args.gap_aware_stops),
        "round_trips": len(trips),
        "wins": sum(1 for t in trips if t["pnl"] > 0),
        "total_pnl_twd": round(total_pnl, 2),
        "total_fees_twd": round(total_fees, 2),
        "exit_kinds": pd.Series([t["action"] for t in trips]).value_counts().to_dict() if trips else {},
        "elapsed_s": round(elapsed, 2),
    }
    if args.json:
        print(json.dumps({"summary": summary, "trades": trips}, indent=1, default=str))
        return 0
    print(
        f"{'entry (Asia/Taipei)':<20} {'side':<5} {'entry':>9} {'exit (Asia/Taipei)':<20} {'exit':>9} "
        f"{'kind':<9} {'qty':>4} {'fees':>9} {'pnl_twd':>11}"
    )
    for t in trips:
        e = pd.Timestamp(t["entry_time"]).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")
        x = pd.Timestamp(t["exit_time"]).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M")
        print(
            f"{e:<20} {sides.get(t['entry_time'], '?'):<5} {t['entry_price']:>9.1f} {x:<20} "
            f"{t['exit_price']:>9.1f} {t['action']:<9} {t['quantity']:>4g} {t['fees']:>9.1f} {t['pnl']:>11.1f}"
        )
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
