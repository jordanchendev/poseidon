#!/usr/bin/env python3
"""Backtest the Trendline Ruler strategy ("one line, one ruler") on TX 5m.

Loads TX 5m OHLCV from Thalassa, computes features once, runs BacktestRunner
with a fixed contract count and recomputes intraday-aware metrics. Writes a
JSON summary + trade CSV to --out-dir. ``run_variant`` is reused by
``sweep_trendline_ruler_tx.py`` on a single pre-fetched OHLCV frame.

Run on stormtrooper in a throwaway container (no DB needed, only Thalassa env):
  docker run --rm --env-file ~/.poseidon-range50.env \\
    -v $PWD/src/poseidon:/app/.venv/lib/python3.13/site-packages/poseidon \\
    -v $PWD/scripts:/app/scripts -v $PWD/scripts/output:/app/scripts/output -w /app \\
    poseidon-cpu-worker python scripts/backtest_trendline_ruler_tx.py --start 2020-03-02 --label base

Overrides are dotted config paths with JSON values, e.g.
  --set htf.mode=off --set risk.min_rr=1.5
  --set 'session.allowed_windows=[["15:00","04:50"],["08:45","13:35"]]'
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
from poseidon.strategies.trendline_ruler import TrendlineRulerStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("backtest_trendline_ruler_tx")

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "src" / "poseidon" / "strategies" / "configs" / "trendline_ruler_tx_5m.json"
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
    """``key=value`` pairs; values are parsed as JSON when possible, else kept as strings."""
    out: dict = {}
    for item in values:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def fetch_ohlcv(cfg: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """TX 5m (or the config's symbol) OHLCV from Thalassa, sorted and de-duplicated."""
    symbol, market, interval = cfg.get("symbol", "TX"), cfg.get("market", "tw_futures"), cfg.get("interval", "5m")
    repo = RemoteDataRepository.from_settings()
    logger.info("Fetching %s/%s %s %s..%s", market, symbol, interval, start.date(), end.date())
    ohlcv = repo.read_ohlcv(symbol, market, interval, start=start, end=end)
    if ohlcv.empty:
        return ohlcv
    ohlcv = ohlcv.sort_index()
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
    logger.info("bars=%d %s..%s", len(ohlcv), ohlcv.index[0], ohlcv.index[-1])
    return ohlcv


def _trade_dd(pnls: list[float]) -> float:
    """Peak-to-trough of cumulative closed-trade PnL (starting from 0)."""
    peak = cum = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def run_variant(
    cfg: dict,
    ohlcv: pd.DataFrame,
    *,
    label: str | None = None,
    overrides: dict | None = None,
    contracts: int = 1,
    capital: float = 1_000_000.0,
) -> tuple[dict, list]:
    """Run one config on a pre-fetched OHLCV frame -> (summary, trade_records)."""
    strategy = TrendlineRulerStrategy(config=cfg)
    strategy.validate_config()

    engine = FeatureEngine()
    t_feat = datetime.now(UTC)
    features = engine._computer.compute_with_data(ohlcv, strategy.get_feature_specs())
    feature_sec = (datetime.now(UTC) - t_feat).total_seconds()
    feature_cols = [c for c in features.columns if c not in ohlcv.columns]
    logger.info(
        "features computed in %.1fs: %d cols, warmup NaN rows=%d",
        feature_sec,
        len(feature_cols),
        int(features[feature_cols].isna().any(axis=1).sum()),
    )

    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=engine,
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODELS[strategy.market],
        initial_capital=capital,
        sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=contracts),
    )
    t0 = datetime.now(UTC)
    result = runner.run(features, feature_specs=[])
    elapsed = (datetime.now(UTC) - t0).total_seconds()
    if result.status != "completed":
        raise RuntimeError(f"backtest failed: {result.error_message}")

    equity = runner._equity_series_cached
    trade_records = runner._trade_records_cached
    metrics = compute_metrics(equity, trade_records, bars_per_year=BARS_PER_YEAR_5M)

    closes = [t for t in trade_records if t.pnl is not None]
    by_exit: dict[str, dict] = {}
    holding: dict[str, list[float]] = {}
    for t in closes:
        d = by_exit.setdefault(t.action, {"n": 0, "pnl": 0.0, "wins": 0})
        d["n"] += 1
        d["pnl"] += float(t.pnl)
        d["wins"] += int(t.pnl > 0)
        if t.exit_time is not None:
            bars = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 300.0
            holding.setdefault(t.action, []).append(bars)
    point_value = COST_MODELS[strategy.market].point_value
    pnl_points = [float(t.pnl) / (point_value * t.quantity) for t in closes if t.quantity]
    wins = [p for p in pnl_points if p > 0]
    losses = [p for p in pnl_points if p <= 0]
    years = (ohlcv.index[-1] - ohlcv.index[0]).total_seconds() / (365.25 * 86_400)
    avg_win, avg_loss = _mean(wins), _mean(losses)

    by_year: dict[str, float] = {}
    by_hour: dict[str, float] = {}
    for t in closes:
        if not t.quantity:
            continue
        pts = float(t.pnl) / (point_value * t.quantity)
        local = pd.Timestamp(t.entry_time).tz_convert(strategy._tz)
        by_year[str(local.year)] = by_year.get(str(local.year), 0.0) + pts
        by_hour[f"{local.hour:02d}"] = by_hour.get(f"{local.hour:02d}", 0.0) + pts

    full_label = (
        f"trendline_ruler_{strategy.symbol}_{strategy.interval}_{cfg['direction_mode']}_htf-{strategy._htf_mode}"
    )
    if label:
        full_label = f"{full_label}_{label}"
    summary = {
        "label": full_label,
        "overrides": overrides or {},
        "config": cfg,
        "period": {"start": str(ohlcv.index[0]), "end": str(ohlcv.index[-1]), "bars": len(ohlcv)},
        "contracts": contracts,
        "feature_sec": round(feature_sec, 1),
        "elapsed_sec": round(elapsed, 1),
        "metrics": metrics,
        "closed_trades": len(closes),
        "trades_per_year": round(len(closes) / years, 1) if years > 0 else None,
        "win_rate": round(len(wins) / len(pnl_points), 4) if pnl_points else None,
        "total_pnl_twd": round(sum(float(t.pnl) for t in closes), 0),
        "avg_pnl_points": _mean(pnl_points),
        "avg_win_points": avg_win,
        "avg_loss_points": avg_loss,
        "realised_avg_rr": round(avg_win / abs(avg_loss), 2) if avg_win and avg_loss else None,
        "planned_avg_rr": _mean(strategy.entry_rr),
        "avg_stop_points": _mean(strategy.entry_stop_points),
        "avg_target_points": _mean(strategy.entry_target_points),
        "trade_dd_twd": round(_trade_dd([float(t.pnl) for t in closes]), 0),
        "pnl_by_year": {k: round(v, 1) for k, v in sorted(by_year.items())},
        "pnl_by_entry_hour_tpe": {k: round(v, 1) for k, v in sorted(by_hour.items())},
        "median_holding_bars_by_exit": {k: float(pd.Series(v).median()) for k, v in sorted(holding.items())},
        "by_exit": by_exit,
        "strategy_exit_counts": dict(strategy.exit_counts),
        "reject_counts": dict(strategy.reject_counts),
        "bars_short_line_active": int(features["tls_active"].sum()),
        "bars_long_line_active": int(features["tll_active"].sum()),
        "short_lines_formed": int(features["tls_id"].max() + 1),
        "long_lines_formed": int(features["tll_id"].max() + 1),
    }
    return summary, trade_records


def write_trades_csv(trade_records: list, path: Path) -> None:
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
    ).to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--start", default="2020-03-02")
    ap.add_argument("--end", default=None)
    ap.add_argument("--direction", choices=["long_only", "short_only", "bidirectional"], default=None)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--set", action="append", default=[], help="dotted config override (JSON value)")
    ap.add_argument("--label", default=None, help="suffix appended to the default output label")
    ap.add_argument("--out-dir", type=Path, default=Path("scripts/output"))
    args = ap.parse_args()

    overrides = parse_set(args.set)
    if args.direction:
        overrides["direction_mode"] = args.direction
    cfg = load_config(args.config, overrides)
    TrendlineRulerStrategy(config=cfg).validate_config()

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else datetime.now(UTC)
    ohlcv = fetch_ohlcv(cfg, start, end)
    if ohlcv.empty:
        logger.error("no OHLCV returned")
        return 1

    summary, trade_records = run_variant(
        cfg, ohlcv, label=args.label, overrides=overrides, contracts=args.contracts, capital=args.capital
    )
    label = summary["label"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{label}.json").write_text(json.dumps(summary, indent=2, default=str))
    write_trades_csv(trade_records, args.out_dir / f"{label}_trades.csv")

    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
