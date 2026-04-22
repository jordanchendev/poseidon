#!/usr/bin/env python3
"""Phase 77-03: WFE + grid search + random baseline for StructuralReversal.

BTC+ETH combined walk-forward optimization with 3 grid search variations
(GATE-04 max), random baseline comparison (GATE-05), and net_sharpe as
optimization target (D-37).

Output: scripts/output/wfo_77_results.json
  - 3 variations grid-searched (v1_2cond, v2_3cond, v3_threshold)
  - BTC+ETH combined WFE with generate_windows + manual loop (Pitfall 7 fix)
  - 100-seed random baseline Sharpe distribution
  - Gate check indicators for Phase 78 decision gate

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/wfo_77_structural.py
Expected runtime: 10-30 min (grid search + 100 random seeds)
"""

import json
import logging
import random
import sys
from datetime import datetime
from itertools import product as itertools_product
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.portfolio import SizingConfig
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig, compute_wfe
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
BARS_PER_DAY = 6  # 24h / 4h
START = datetime(2024, 1, 1)  # Post-ETF only (D-01)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "4h"
INITIAL_CAPITAL = 100_000.0  # USDT per symbol
FUNDING_RATE_ANNUAL = 0.1095  # D-22: 0.01%/8h * 3 * 365.25
BAR_DURATION_HOURS = 4  # 4H interval
RANDOM_SEEDS = 100  # D-39: 100 random seeds for baseline

# CRITICAL: WalkForwardConfig uses BAR COUNTS not calendar days (Pitfall 2 fix)
WF_CONFIG = WalkForwardConfig(
    train_days=180 * BARS_PER_DAY,  # 1080 bars (D-28)
    test_days=90 * BARS_PER_DAY,  # 540 bars (D-28)
    step_days=90 * BARS_PER_DAY,  # 540 bars (D-28)
    min_trades_per_oos=30,  # D-30
    min_wfe=0.60,  # = 1.0 - 0.40 degradation max (Pitfall 3 fix)
)

# D-33: Variation 1 param grid (2-condition: oiwap + cascade)
PARAM_GRID_V1 = {
    "atr_multiplier": [0.5, 1.0, 1.5, 2.0],
    "stop_atr_multiplier": [2.0, 3.0, 4.0, 5.0],
}
# D-34: Variation 2 param grid (3-condition: + cvd_change)
PARAM_GRID_V2 = {
    "atr_multiplier": [0.5, 1.0, 1.5, 2.0],
    "stop_atr_multiplier": [2.0, 3.0, 4.0, 5.0],
}
# D-35: Variation 3 param grid (2-condition with variable oiwap_threshold)
PARAM_GRID_V3 = {
    "atr_multiplier": [0.5, 1.0, 1.5, 2.0],
    "stop_atr_multiplier": [2.0, 3.0, 4.0, 5.0],
    "oiwap_threshold": [2.0, 3.0, 5.0, 7.0],
}


# ---------------------------------------------------------------------------
# Random entry strategy for baseline comparison (D-38)
# ---------------------------------------------------------------------------
class RandomEntryStructuralStrategy(StructuralReversalStrategy):
    """Random entry variant: replaces structural entry with random coin flip.

    Keeps the same exit logic (ATR trailing stop, time expiry, MaxDD)
    so the comparison is purely about entry quality.
    """

    def __init__(
        self,
        config: StructuralReversalConfig,
        symbol: str = "BTCUSDT",
        seed: int = 0,
        entry_probability: float = 0.05,
    ):
        super().__init__(config=config, symbol=symbol)
        self._rng = random.Random(seed)
        self._entry_probability = entry_probability

    def _check_entry(self, row: pd.Series) -> tuple[str | None, float | None]:
        """Random entry: flip a coin with calibrated probability.

        Uses same limit price formula (OIWAP +/- ATR) but direction is random.
        """
        atr = row.get("atr_14")
        oiwap = row.get("oiwap_168")

        if pd.isna(atr) or pd.isna(oiwap):
            return None, None

        # Random entry with calibrated probability
        if self._rng.random() >= self._entry_probability:
            return None, None

        # Random direction
        direction = "long" if self._rng.random() < 0.5 else "short"

        # Same limit price derivation as real strategy (D-12)
        oiwap_val = float(oiwap)
        atr_val = float(atr)
        if direction == "long":
            limit_price = oiwap_val - (self.config.atr_multiplier * atr_val)
        else:
            limit_price = oiwap_val + (self.config.atr_multiplier * atr_val)

        return direction, limit_price


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _load_ohlcv(repo: RemoteDataRepository, symbol: str) -> pd.DataFrame:
    """Fetch 4H OHLCV for symbol from Thalassa, filtered to post-ETF."""
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
    # Convert to UTC first to avoid silent timestamp corruption for non-UTC sources
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.tz_convert("UTC").tz_localize(None)

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


def _make_runner(strategy: StructuralReversalStrategy) -> BacktestRunner:
    """Create BacktestRunner with pessimistic fill model.

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


def _compute_net_sharpe(metrics: dict, trades: list, total_bars: int) -> float:
    """Compute net_sharpe after funding cost deduction (D-37 optimization target).

    Returns just the net_sharpe value for use as optimization metric.
    """
    gross_sharpe = metrics.get("sharpe_ratio", 0.0)

    # Calculate time in market from trades
    total_holding_bars = 0
    for trade in trades:
        if trade.exit_time is not None and trade.entry_time is not None:
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
    return gross_sharpe - funding_cost_annual


def _run_single_backtest(
    params: dict,
    symbol: str,
    ohlcv: pd.DataFrame,
    use_cvd: bool = False,
) -> dict:
    """Run single-symbol backtest with given params.

    Returns metrics dict including trade_count, sharpe_ratio, annualized_return,
    max_drawdown, and net_sharpe.
    """
    cfg_params = {**params, "use_cvd_filter": use_cvd}
    config = StructuralReversalConfig(**cfg_params)
    strategy = StructuralReversalStrategy(config=config, symbol=symbol)
    runner = _make_runner(strategy)

    result = runner.run(ohlcv)
    if result.status != "completed":
        return {
            "sharpe_ratio": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "net_sharpe": 0.0,
            "error": result.error_message,
        }

    metrics = _recompute_metrics(runner)
    portfolio = runner._ar_last_portfolio
    net_sharpe = _compute_net_sharpe(metrics, portfolio.trades, len(ohlcv))

    return {
        "sharpe_ratio": round(metrics.get("sharpe_ratio", 0.0), 4),
        "annualized_return": round(metrics.get("annualized_return", 0.0), 4),
        "max_drawdown": round(metrics.get("max_drawdown", 0.0), 4),
        "total_return": round(metrics.get("total_return", 0.0), 4),
        "trade_count": metrics.get("trade_count", 0),
        "win_rate": round(metrics.get("win_rate", 0.0), 4),
        "net_sharpe": round(net_sharpe, 4),
        "trades": portfolio.trades,
        "equity_curve": portfolio.equity_curve,
    }


def _run_combined_window(
    btc_ohlcv_slice: pd.DataFrame,
    eth_ohlcv_slice: pd.DataFrame,
    params: dict,
    use_cvd: bool = False,
) -> dict:
    """Run BTC+ETH combined backtest on a data slice (D-29 combined optimization).

    Runs _run_single_backtest for BTC and ETH separately with same params.
    Pools all trades. Merges equity curves (sum). Recomputes combined metrics.
    """
    btc_result = _run_single_backtest(params, "BTCUSDT", btc_ohlcv_slice, use_cvd)
    eth_result = _run_single_backtest(params, "ETHUSDT", eth_ohlcv_slice, use_cvd)

    # Pool trades from both symbols
    btc_trades = btc_result.get("trades", [])
    eth_trades = eth_result.get("trades", [])
    all_trades = list(btc_trades) + list(eth_trades)

    # Merge equity curves by summing per-symbol equity
    btc_eq = btc_result.get("equity_curve", [])
    eth_eq = eth_result.get("equity_curve", [])

    if btc_eq and eth_eq:
        btc_series = pd.Series(
            [eq for _, eq, _ in btc_eq],
            index=pd.DatetimeIndex([t for t, _, _ in btc_eq]),
        )
        eth_series = pd.Series(
            [eq for _, eq, _ in eth_eq],
            index=pd.DatetimeIndex([t for t, _, _ in eth_eq]),
        )
        combined = pd.concat([btc_series, eth_series], axis=1)
        combined = combined.ffill().bfill()
        combined_equity = combined.sum(axis=1)

        combined_metrics = compute_metrics(
            combined_equity,
            all_trades,
            bars_per_year=BARS_PER_YEAR,
        )
    elif btc_eq:
        combined_equity = pd.Series(
            [eq for _, eq, _ in btc_eq],
            index=pd.DatetimeIndex([t for t, _, _ in btc_eq]),
        )
        combined_metrics = compute_metrics(
            combined_equity, all_trades, bars_per_year=BARS_PER_YEAR
        )
    elif eth_eq:
        combined_equity = pd.Series(
            [eq for _, eq, _ in eth_eq],
            index=pd.DatetimeIndex([t for t, _, _ in eth_eq]),
        )
        combined_metrics = compute_metrics(
            combined_equity, all_trades, bars_per_year=BARS_PER_YEAR
        )
    else:
        combined_metrics = {"sharpe_ratio": 0.0, "annualized_return": 0.0, "trade_count": 0}

    # Compute net_sharpe for combined
    total_bars = max(len(btc_ohlcv_slice), len(eth_ohlcv_slice))
    net_sharpe = _compute_net_sharpe(combined_metrics, all_trades, total_bars)

    return {
        "sharpe_ratio": round(combined_metrics.get("sharpe_ratio", 0.0), 4),
        "annualized_return": round(combined_metrics.get("annualized_return", 0.0), 4),
        "max_drawdown": round(combined_metrics.get("max_drawdown", 0.0), 4),
        "total_return": round(combined_metrics.get("total_return", 0.0), 4),
        "trade_count": combined_metrics.get("trade_count", 0),
        "win_rate": round(combined_metrics.get("win_rate", 0.0), 4),
        "net_sharpe": round(net_sharpe, 4),
        "btc_trades": len(btc_trades),
        "eth_trades": len(eth_trades),
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def run_grid_search_variation(
    variation_name: str,
    param_grid: dict,
    base_config: dict,
    btc_ohlcv: pd.DataFrame,
    eth_ohlcv: pd.DataFrame,
) -> dict:
    """Run grid search for a variation over BTC+ETH combined.

    For each param combination in the grid:
    - Merge with base_config
    - Run combined window (full data) with those params
    - Record net_sharpe as the metric

    Returns best params by net_sharpe (D-37: optimization target is net_sharpe).
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    trials = []

    total_combos = 1
    for v in values:
        total_combos *= len(v)
    print(f"\n  {variation_name}: {total_combos} combinations")

    for combo in itertools_product(*values):
        params = {**base_config, **dict(zip(keys, combo))}
        use_cvd = params.pop("use_cvd_filter", False)

        combined = _run_combined_window(btc_ohlcv, eth_ohlcv, params, use_cvd)

        trial = {
            "params": {k: v for k, v in params.items()
                       if k in list(param_grid.keys()) + ["use_cvd_filter"]},
            "net_sharpe": combined["net_sharpe"],
            "sharpe_ratio": combined["sharpe_ratio"],
            "trade_count": combined["trade_count"],
            "max_drawdown": combined["max_drawdown"],
            "annualized_return": combined["annualized_return"],
        }
        trials.append(trial)
        print(
            f"    {dict(zip(keys, combo))} -> net_sharpe={combined['net_sharpe']:.4f}, "
            f"trades={combined['trade_count']}, maxdd={combined['max_drawdown']:.4f}"
        )

    # Sort by net_sharpe descending
    trials.sort(key=lambda t: t["net_sharpe"], reverse=True)
    best = trials[0] if trials else {"params": {}, "net_sharpe": 0.0}

    print(f"  Best {variation_name}: net_sharpe={best['net_sharpe']:.4f}, params={best['params']}")

    return {
        "best_params": best["params"],
        "net_sharpe": best["net_sharpe"],
        "sharpe_ratio": best.get("sharpe_ratio", 0.0),
        "trade_count": best.get("trade_count", 0),
        "max_drawdown": best.get("max_drawdown", 0.0),
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Combined WFE (Pitfall 7 fix: manual loop for multi-symbol)
# ---------------------------------------------------------------------------
def run_combined_wfe(
    btc_ohlcv: pd.DataFrame,
    eth_ohlcv: pd.DataFrame,
    best_params: dict,
    use_cvd: bool = False,
) -> list[dict]:
    """BTC+ETH combined WFE using generate_windows + manual loop.

    Cannot use WalkForwardAnalyzer.analyze() for multi-symbol (Pitfall 7).
    Instead: generate windows from data length, run combined backtest per window.
    """
    analyzer = WalkForwardAnalyzer(
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODELS["crypto_perp"],
        initial_capital=INITIAL_CAPITAL,
        fill_model=FillModel.PESSIMISTIC,
    )

    data_length = min(len(btc_ohlcv), len(eth_ohlcv))
    windows = analyzer.generate_windows(data_length, WF_CONFIG)

    if not windows:
        logger.warning("No WFE windows generated (data_length=%d)", data_length)
        return []

    print(f"\n  WFE: {len(windows)} windows, data_length={data_length}")
    print(f"  Config: train={WF_CONFIG.train_days} bars, test={WF_CONFIG.test_days} bars, "
          f"step={WF_CONFIG.step_days} bars")

    window_results = []
    for i, ((train_s, train_e), (test_s, test_e)) in enumerate(windows):
        # IS: combined backtest on training slice
        is_combined = _run_combined_window(
            btc_ohlcv.iloc[train_s:train_e],
            eth_ohlcv.iloc[train_s:train_e],
            best_params,
            use_cvd,
        )

        # OOS: combined backtest on test slice
        oos_combined = _run_combined_window(
            btc_ohlcv.iloc[test_s:test_e],
            eth_ohlcv.iloc[test_s:test_e],
            best_params,
            use_cvd,
        )

        # Compute WFE
        wfe = compute_wfe(
            is_combined["annualized_return"],
            oos_combined["annualized_return"],
        )

        window_result = {
            "window": i,
            "train_range": [train_s, train_e],
            "test_range": [test_s, test_e],
            "is_ann_return": is_combined["annualized_return"],
            "is_net_sharpe": is_combined["net_sharpe"],
            "is_trade_count": is_combined["trade_count"],
            "oos_ann_return": oos_combined["annualized_return"],
            "oos_net_sharpe": oos_combined["net_sharpe"],
            "oos_trade_count": oos_combined["trade_count"],
            "oos_max_drawdown": oos_combined["max_drawdown"],
            "wfe": round(wfe, 4),
        }
        window_results.append(window_result)

        # D-30: warn if OOS trade count < 30
        if oos_combined["trade_count"] < WF_CONFIG.min_trades_per_oos:
            logger.warning(
                "Window %d: OOS trade count %d < %d (D-30)",
                i, oos_combined["trade_count"], WF_CONFIG.min_trades_per_oos,
            )

        print(
            f"    Window {i}: IS ann_ret={is_combined['annualized_return']:.4f}, "
            f"OOS ann_ret={oos_combined['annualized_return']:.4f}, "
            f"WFE={wfe:.4f}, OOS trades={oos_combined['trade_count']}"
        )

    return window_results


# ---------------------------------------------------------------------------
# Random baseline (D-38, D-39)
# ---------------------------------------------------------------------------
def run_random_baseline(
    btc_ohlcv: pd.DataFrame,
    eth_ohlcv: pd.DataFrame,
    best_params: dict,
    real_entry_count: int,
    n_seeds: int = RANDOM_SEEDS,
) -> dict:
    """Random entry baseline: same exit logic, random entries (D-38).

    Calibrates entry probability from real strategy's signal count.
    Returns median/p25/p75 net_sharpe distribution over n_seeds (D-39).
    """
    # Calibrate entry probability from real strategy's signal frequency
    total_bars = max(len(btc_ohlcv), len(eth_ohlcv))
    entry_probability = (
        real_entry_count / (total_bars * 2) if total_bars > 0 else 0.05
    )
    entry_probability = max(0.01, min(entry_probability, 0.20))

    print(f"\n  Random baseline: {n_seeds} seeds, entry_prob={entry_probability:.4f}")

    sharpes = []
    for seed in range(n_seeds):
        # Create random entry strategy for BTC
        btc_cfg = StructuralReversalConfig(**best_params)
        btc_strategy = RandomEntryStructuralStrategy(
            config=btc_cfg, symbol="BTCUSDT",
            seed=seed, entry_probability=entry_probability,
        )
        btc_runner = _make_runner(btc_strategy)
        btc_result = btc_runner.run(btc_ohlcv)

        # Create random entry strategy for ETH
        eth_cfg = StructuralReversalConfig(**best_params)
        eth_strategy = RandomEntryStructuralStrategy(
            config=eth_cfg, symbol="ETHUSDT",
            seed=seed + 10000, entry_probability=entry_probability,
        )
        eth_runner = _make_runner(eth_strategy)
        eth_result = eth_runner.run(eth_ohlcv)

        # Combined metrics
        if btc_result.status == "completed" and eth_result.status == "completed":
            btc_metrics = _recompute_metrics(btc_runner)
            eth_metrics = _recompute_metrics(eth_runner)

            btc_portfolio = btc_runner._ar_last_portfolio
            eth_portfolio = eth_runner._ar_last_portfolio

            # Combine equity curves
            all_trades = list(btc_portfolio.trades) + list(eth_portfolio.trades)

            if btc_portfolio.equity_curve and eth_portfolio.equity_curve:
                btc_eq = pd.Series(
                    [eq for _, eq, _ in btc_portfolio.equity_curve],
                    index=pd.DatetimeIndex([t for t, _, _ in btc_portfolio.equity_curve]),
                )
                eth_eq = pd.Series(
                    [eq for _, eq, _ in eth_portfolio.equity_curve],
                    index=pd.DatetimeIndex([t for t, _, _ in eth_portfolio.equity_curve]),
                )
                combined = pd.concat([btc_eq, eth_eq], axis=1).ffill().bfill()
                combined_equity = combined.sum(axis=1)
                combined_metrics = compute_metrics(
                    combined_equity, all_trades, bars_per_year=BARS_PER_YEAR
                )
                net_sharpe = _compute_net_sharpe(
                    combined_metrics, all_trades, total_bars
                )
                sharpes.append(round(net_sharpe, 4))
            else:
                sharpes.append(0.0)
        else:
            sharpes.append(0.0)

        if (seed + 1) % 20 == 0:
            print(f"    Completed {seed + 1}/{n_seeds} seeds")

    sharpes.sort()
    n = len(sharpes)
    result = {
        "median": sharpes[n // 2] if n > 0 else 0.0,
        "p25": sharpes[n // 4] if n > 0 else 0.0,
        "p75": sharpes[3 * n // 4] if n > 0 else 0.0,
        "count": n,
        "entry_probability": round(entry_probability, 4),
    }

    print(f"  Baseline: median={result['median']:.4f}, "
          f"p25={result['p25']:.4f}, p75={result['p75']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_wfo() -> int:
    """Run WFE + grid search + random baseline for StructuralReversal."""
    print("=" * 80)
    print("Phase 77-03: StructuralReversal WFE + Grid Search + Random Baseline")
    print("=" * 80)
    print(f"Period: {START.date()} to {datetime.now().date()}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Interval: {INTERVAL} | Bars/Year: {BARS_PER_YEAR}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} USDT (per symbol)")
    print(f"Fill Model: PESSIMISTIC")
    print(f"WF Config: train={WF_CONFIG.train_days} bars, test={WF_CONFIG.test_days} bars, "
          f"step={WF_CONFIG.step_days} bars")
    print(f"Min WFE: {WF_CONFIG.min_wfe} | Min trades OOS: {WF_CONFIG.min_trades_per_oos}")
    print(f"Random seeds: {RANDOM_SEEDS}")
    print()

    # Step 1: Load data
    repo = RemoteDataRepository.from_settings()
    btc_ohlcv = _load_ohlcv(repo, "BTCUSDT")
    eth_ohlcv = _load_ohlcv(repo, "ETHUSDT")

    if btc_ohlcv.empty or eth_ohlcv.empty:
        print("ERROR: Missing OHLCV data. Aborting.")
        return 1

    print(f"\nBTC: {len(btc_ohlcv)} bars, ETH: {len(eth_ohlcv)} bars")

    # Step 2: Grid search - 3 variations (D-33, D-34, D-35)
    print(f"\n{'=' * 80}")
    print("GRID SEARCH (3 variations, GATE-04 max)")
    print(f"{'=' * 80}")

    # Variation 1: 2-condition (oiwap + cascade), no CVD
    v1_result = run_grid_search_variation(
        "v1_2cond",
        PARAM_GRID_V1,
        {"use_cvd_filter": False},
        btc_ohlcv,
        eth_ohlcv,
    )

    # Variation 2: 3-condition (oiwap + cascade + cvd)
    v2_result = run_grid_search_variation(
        "v2_3cond",
        PARAM_GRID_V2,
        {"use_cvd_filter": True},
        btc_ohlcv,
        eth_ohlcv,
    )

    # Variation 3: 2-condition with variable oiwap_threshold
    v3_result = run_grid_search_variation(
        "v3_threshold",
        PARAM_GRID_V3,
        {"use_cvd_filter": False},
        btc_ohlcv,
        eth_ohlcv,
    )

    # Step 3: Select overall best variation by net_sharpe
    variations = {
        "v1_2cond": v1_result,
        "v2_3cond": v2_result,
        "v3_threshold": v3_result,
    }
    best_name = max(variations, key=lambda k: variations[k]["net_sharpe"])
    best_variation = variations[best_name]
    best_params = best_variation["best_params"]

    # Determine use_cvd from best variation
    use_cvd_best = best_name == "v2_3cond"

    print(f"\n{'=' * 80}")
    print(f"BEST VARIATION: {best_name}")
    print(f"  Net Sharpe: {best_variation['net_sharpe']:.4f}")
    print(f"  Params: {best_params}")
    print(f"{'=' * 80}")

    # Step 4: Run combined WFE with best params
    print(f"\n{'=' * 80}")
    print("COMBINED WFE (BTC + ETH)")
    print(f"{'=' * 80}")

    window_results = run_combined_wfe(
        btc_ohlcv, eth_ohlcv, best_params, use_cvd=use_cvd_best,
    )

    # Compute WFE aggregates
    if window_results:
        wfe_values = [w["wfe"] for w in window_results]
        avg_wfe = sum(wfe_values) / len(wfe_values)
        degradation = 1.0 - avg_wfe if avg_wfe > 0 else 1.0
        wfe_passes = avg_wfe >= WF_CONFIG.min_wfe

        # Check OOS Sharpe
        oos_sharpes = [w["oos_net_sharpe"] for w in window_results]
        avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes)

        # Total signals across all windows
        total_oos_signals = sum(w["oos_trade_count"] for w in window_results)
    else:
        avg_wfe = 0.0
        degradation = 1.0
        wfe_passes = False
        avg_oos_sharpe = 0.0
        total_oos_signals = 0

    print(f"\n  Average WFE: {avg_wfe:.4f}")
    print(f"  Degradation: {degradation:.4f}")
    print(f"  Passes gate (WFE >= {WF_CONFIG.min_wfe}): {wfe_passes}")
    print(f"  Average OOS net_sharpe: {avg_oos_sharpe:.4f}")
    print(f"  Total OOS signals: {total_oos_signals}")

    # Step 5: Random baseline (100 seeds)
    print(f"\n{'=' * 80}")
    print("RANDOM BASELINE (D-38, D-39)")
    print(f"{'=' * 80}")

    # Calibrate from best variation's trade count
    real_entry_count = best_variation.get("trade_count", 50)
    baseline_result = run_random_baseline(
        btc_ohlcv, eth_ohlcv, best_params,
        real_entry_count=real_entry_count,
    )

    # Step 6: Gate check
    beats_random = best_variation["net_sharpe"] > baseline_result["median"]
    oos_sharpe_positive = avg_oos_sharpe > 0.0
    wfe_below_40pct = degradation < 0.40

    print(f"\n{'=' * 80}")
    print("GATE CHECK")
    print(f"{'=' * 80}")
    print(f"  Best net_sharpe:     {best_variation['net_sharpe']:.4f}")
    print(f"  Random median:       {baseline_result['median']:.4f}")
    print(f"  Beats random:        {beats_random}")
    print(f"  OOS Sharpe > 0:      {oos_sharpe_positive} (avg={avg_oos_sharpe:.4f})")
    print(f"  WFE degradation:     {degradation:.4f} (< 0.40 = {wfe_below_40pct})")
    print(f"  Total OOS signals:   {total_oos_signals}")

    # Step 7: Write JSON output
    output = {
        "phase": "77",
        "plan": "03",
        "description": "WFE + grid search + random baseline",
        "period": {
            "start": START.date().isoformat(),
            "end": datetime.now().date().isoformat(),
        },
        "wf_config": {
            "train_bars": WF_CONFIG.train_days,
            "test_bars": WF_CONFIG.test_days,
            "step_bars": WF_CONFIG.step_days,
            "min_trades_oos": WF_CONFIG.min_trades_per_oos,
            "min_wfe": WF_CONFIG.min_wfe,
        },
        "variations": {
            "v1_2cond": {
                "best_params": v1_result["best_params"],
                "net_sharpe": v1_result["net_sharpe"],
                "sharpe_ratio": v1_result.get("sharpe_ratio", 0.0),
                "trade_count": v1_result.get("trade_count", 0),
                "trials": v1_result["trials"],
            },
            "v2_3cond": {
                "best_params": v2_result["best_params"],
                "net_sharpe": v2_result["net_sharpe"],
                "sharpe_ratio": v2_result.get("sharpe_ratio", 0.0),
                "trade_count": v2_result.get("trade_count", 0),
                "trials": v2_result["trials"],
            },
            "v3_threshold": {
                "best_params": v3_result["best_params"],
                "net_sharpe": v3_result["net_sharpe"],
                "sharpe_ratio": v3_result.get("sharpe_ratio", 0.0),
                "trade_count": v3_result.get("trade_count", 0),
                "trials": v3_result["trials"],
            },
        },
        "best_variation": best_name,
        "best_params": best_params,
        "wfe": {
            "windows": window_results,
            "avg_wfe": round(avg_wfe, 4),
            "degradation": round(degradation, 4),
            "passes_gate": wfe_passes,
            "avg_oos_net_sharpe": round(avg_oos_sharpe, 4),
            "total_oos_signals": total_oos_signals,
        },
        "random_baseline": baseline_result,
        "gate_check": {
            "beats_random": beats_random,
            "oos_sharpe_positive": oos_sharpe_positive,
            "total_signals": total_oos_signals,
            "wfe_degradation_below_40pct": wfe_below_40pct,
        },
    }

    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wfo_77_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nJSON output saved to: {output_path}")

    # Print JSON for easy parsing
    print(f"\n{'=' * 100}")
    print("JSON OUTPUT")
    print(f"{'=' * 100}")
    print(json.dumps(output, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(run_wfo())
