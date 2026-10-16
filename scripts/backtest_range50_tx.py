#!/usr/bin/env python3
"""Backtest the Range 50% Reversion strategy on TX 5m (optionally with TE SMT).

Loads TX (and the companion) 5m OHLCV from Thalassa, computes features once
with companion injection, runs BacktestRunner with a fixed contract count and
recomputes intraday-aware metrics. Writes JSON + trade CSV to --out-dir.

Run on stormtrooper (no DB needed, only Thalassa env):
  docker run --rm --env-file ~/Projects/poseidon/.env \\
    -v $PWD/src/poseidon:/app/.venv/lib/python3.13/site-packages/poseidon \\
    -v $PWD/scripts:/app/scripts -v $PWD/scripts/output:/app/scripts/output \\
    poseidon-cpu-worker python scripts/backtest_range50_tx.py --start 2020-03-02 --smt-mode off
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.range_reversion import RangeReversionStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("backtest_range50_tx")

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "src" / "poseidon" / "strategies" / "configs" / "range50_tx_5m.json"
)
BARS_PER_YEAR_5M = 187 * 250  # ~187 TX 5m bars per trading day (day + night session)


def load_config(path: Path, overrides: dict) -> dict:
    cfg = json.loads(path.read_text())
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
    return cfg


def parse_set(values: list[str]) -> dict:
    out: dict = {}
    for item in values:
        key, _, raw = item.partition("=")
        if raw.lower() in ("true", "false"):
            val: object = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--start", default="2020-03-02")
    ap.add_argument("--end", default=None)
    ap.add_argument("--smt-mode", choices=["off", "optional", "required"], default=None)
    ap.add_argument("--direction", choices=["long_only", "short_only", "bidirectional"], default=None)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--set", action="append", default=[], help="dotted config override, e.g. risk.min_rr=2.0")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("scripts/output"))
    args = ap.parse_args()

    overrides = parse_set(args.set)
    if args.smt_mode:
        overrides["confirm.smt_mode"] = args.smt_mode
    if args.direction:
        overrides["direction_mode"] = args.direction
    cfg = load_config(args.config, overrides)
    strategy = RangeReversionStrategy(config=cfg)
    strategy.validate_config()

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else datetime.now(UTC)

    repo = RemoteDataRepository.from_settings()
    logger.info(
        "Fetching %s/%s %s %s..%s", strategy.market, strategy.symbol, strategy.interval, start.date(), end.date()
    )
    ohlcv = repo.read_ohlcv(strategy.symbol, strategy.market, strategy.interval, start=start, end=end)
    if ohlcv.empty:
        logger.error("no OHLCV returned")
        return 1
    ohlcv = ohlcv.sort_index()
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
    logger.info("primary bars=%d %s..%s", len(ohlcv), ohlcv.index[0], ohlcv.index[-1])

    companion_cache: dict[tuple[str, str], pd.DataFrame] = {}
    if strategy._smt_mode != "off":
        key = (strategy._companion_market, strategy._companion_symbol)
        comp = repo.read_ohlcv(key[1], key[0], strategy.interval, start=start, end=end)
        comp = comp.sort_index()
        comp = comp[~comp.index.duplicated(keep="last")]
        logger.info("companion %s bars=%d", key, len(comp))
        companion_cache[key] = comp

    specs = strategy.get_feature_specs()
    engine = FeatureEngine()
    features = engine._computer.compute_with_data(ohlcv, specs, companion_cache=companion_cache)
    feature_cols = [c for c in features.columns if c not in ohlcv.columns]
    logger.info(
        "features computed: %d cols, warmup NaN rows=%d",
        len(feature_cols),
        int(features[feature_cols].isna().any(axis=1).sum()),
    )

    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=engine,
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODELS[strategy.market],
        initial_capital=args.capital,
        sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=args.contracts),
    )
    t0 = datetime.now(UTC)
    result = runner.run(features, feature_specs=[])
    elapsed = (datetime.now(UTC) - t0).total_seconds()
    if result.status != "completed":
        logger.error("backtest failed: %s", result.error_message)
        return 1

    equity = runner._equity_series_cached
    trade_records = runner._trade_records_cached
    metrics = compute_metrics(equity, trade_records, bars_per_year=BARS_PER_YEAR_5M)

    closes = [t for t in trade_records if t.pnl is not None]
    by_exit: dict[str, dict] = {}
    for t in closes:
        d = by_exit.setdefault(t.action, {"n": 0, "pnl": 0.0, "wins": 0})
        d["n"] += 1
        d["pnl"] += float(t.pnl)
        d["wins"] += int(t.pnl > 0)
    point_value = COST_MODELS[strategy.market].point_value
    pnl_points = [float(t.pnl) / (point_value * t.quantity) for t in closes if t.quantity]
    summary = {
        "label": args.label
        or f"range50_{strategy.symbol}_{strategy.interval}_smt-{strategy._smt_mode}_{cfg['direction_mode']}",
        "config": cfg,
        "period": {"start": str(ohlcv.index[0]), "end": str(ohlcv.index[-1]), "bars": len(ohlcv)},
        "contracts": args.contracts,
        "elapsed_sec": round(elapsed, 1),
        "metrics": metrics,
        "closed_trades": len(closes),
        "total_pnl_twd": round(sum(float(t.pnl) for t in closes), 0),
        "avg_pnl_points": round(sum(pnl_points) / len(pnl_points), 2) if pnl_points else None,
        "by_exit": by_exit,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = summary["label"]
    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(
        [
            {
                "action": t.action,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "fees": t.fees,
                "pnl": t.pnl,
            }
            for t in trade_records
        ]
    ).to_csv(args.out_dir / f"{stem}_trades.csv", index=False)

    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
