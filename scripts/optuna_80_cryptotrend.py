#!/usr/bin/env python3
"""Phase 80: Optuna CryptoTrend parameter optimization + WFE validation.

Searches EMA fast/slow periods and funding filter thresholds using
PortfolioBacktester with market-order cost model (crypto_perp).
Extends BacktestCryptoTrend wrapper to apply funding filter point-in-time
using historical funding data. Validates top-3 parameter sets via
walk-forward efficiency (WFE). Produces a decision recommendation per D-19.

4 optimized parameters:
  - ema_fast_period (6-20)
  - ema_slow_period (20-50)
  - max_funding_rate_long (0.0005-0.003)
  - max_funding_rate_short (-0.003 to -0.0005)

Fixed parameters (D-09):
  - leverage=3, allocation=equal_weight, position_limit_pct=0.5
  - symbols=[BTCUSDT, ETHUSDT]

Run on stormtrooper:
  docker compose exec cpu-worker python scripts/optuna_80_cryptotrend.py

Expected runtime: 30-90 min (100 trials + WFE for top-3)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio_backtester import PortfolioBacktester
from poseidon.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig, compute_wfe
from poseidon.data.features.engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.portfolio.crypto_trend import (
    CryptoTrendConfig,
    FundingFilterConfig,
    MomentumConfig,
    PerpAllocationConfig,
)
from poseidon.strategies.portfolio.schemas import TargetPosition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1: Configuration (per D-08, D-09, D-15)
# ---------------------------------------------------------------------------
START = datetime(2024, 1, 1)  # Post-ETF only
END = datetime.now()
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_CAPITAL = 100_000.0  # 100K USDT
INTERVAL = "4h"
BARS_PER_YEAR = 2190  # 365.25 * 24 / 4 -- NOT 252 (Pitfall 1)
BARS_PER_DAY = 6  # 24 / 4
COST_MODEL = COST_MODELS["crypto_perp"]  # taker 0.05%, slippage 0.05% (D-15)
N_TRIALS = 100
BASELINE_SHARPE = 1.30  # Phase 75 baseline (D-14)

# D-09: Fixed params (NOT optimized)
FIXED_LEVERAGE = 3
FIXED_ALLOCATION = "equal_weight"
FIXED_POSITION_LIMIT = 0.5

# WFE config: IN BARS not calendar days (Pitfall 8)
WF_CONFIG = WalkForwardConfig(
    train_days=180 * BARS_PER_DAY,  # 1080 bars
    test_days=90 * BARS_PER_DAY,  # 540 bars
    step_days=90 * BARS_PER_DAY,  # 540 bars
    min_trades_per_oos=10,
    min_wfe=0.60,  # D-19
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "optuna_80_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_of_to_ts(as_of: date | None) -> pd.Timestamp | None:
    """Convert date to end-of-day naive Timestamp for OHLCV index comparison.

    OHLCV indices are tz-stripped after fetch, so we use naive timestamps.
    """
    if as_of is None:
        return None
    ts = pd.Timestamp(as_of)
    return ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def compute_ema_signal(close: pd.Series, fast: int, slow: int) -> str:
    """Compute EMA crossover signal (same logic as CryptoTrendStrategy)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    latest_fast = ema_fast.iloc[-1]
    latest_slow = ema_slow.iloc[-1]

    if latest_fast > latest_slow:
        return "long"
    elif latest_fast < latest_slow:
        return "short"
    return "neutral"


# ---------------------------------------------------------------------------
# Section 2: Extended BacktestCryptoTrend with funding filter
# ---------------------------------------------------------------------------
class BacktestCryptoTrend:
    """CryptoTrend strategy with point-in-time funding filter for backtest.

    Extends the Phase 75 BacktestCryptoTrend wrapper (compare_75_cryptotrend.py
    lines 107-181) with historical funding data support. Uses pre-fetched
    funding rate DataFrames to apply funding filter point-in-time, matching
    what the live CryptoTrendStrategy would have seen at each as_of date.
    """

    name = "crypto_trend_backtest"

    def __init__(
        self,
        config: CryptoTrendConfig,
        ohlcv_dict: dict[str, pd.DataFrame],
        funding_df: dict[str, pd.DataFrame] | None = None,
    ):
        self.config = config
        self._ohlcv_dict = ohlcv_dict
        self._funding_df = funding_df or {}

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        cfg = self.config
        targets: list[TargetPosition] = []

        for symbol in cfg.symbols:
            ohlcv = self._ohlcv_dict.get(symbol)
            if ohlcv is None or ohlcv.empty:
                continue

            # Slice data up to as_of for point-in-time correctness
            if as_of is not None:
                as_of_end = _as_of_to_ts(as_of)
                ohlcv_slice = ohlcv[ohlcv.index <= as_of_end]
            else:
                ohlcv_slice = ohlcv

            if len(ohlcv_slice) < cfg.momentum.ema_slow_period:
                continue

            # EMA crossover signal (same logic as CryptoTrendStrategy)
            signal = compute_ema_signal(
                ohlcv_slice["close"],
                cfg.momentum.ema_fast_period,
                cfg.momentum.ema_slow_period,
            )

            if signal == "neutral":
                continue

            # --- Funding filter (point-in-time using historical data) ---
            if cfg.funding_filter and self._funding_df.get(symbol) is not None:
                funding = self._funding_df[symbol]
                if as_of is not None:
                    as_of_end = _as_of_to_ts(as_of)
                    funding_slice = funding[funding.index <= as_of_end]
                else:
                    funding_slice = funding

                if not funding_slice.empty:
                    # Use funding_rate_daily column (from Thalassa funding rate data)
                    latest_funding = float(
                        funding_slice.iloc[-1].get("funding_rate_daily", 0)
                    )
                    if (
                        signal == "long"
                        and latest_funding > cfg.funding_filter.max_funding_rate_long
                    ):
                        continue  # skip: funding too expensive for long
                    if (
                        signal == "short"
                        and latest_funding < cfg.funding_filter.max_funding_rate_short
                    ):
                        continue  # skip: funding too expensive for short

            # Equal weight allocation (same as CryptoTrendStrategy)
            n_symbols = len(cfg.symbols)
            raw_weight = 1.0 / n_symbols
            weight = min(raw_weight, cfg.allocation.position_limit_pct)

            targets.append(
                TargetPosition(
                    symbol=symbol,
                    weight=weight,
                    reason=f"ema_{cfg.momentum.ema_fast_period}_{cfg.momentum.ema_slow_period}={signal}",
                    side=signal,
                    leverage=float(cfg.allocation.leverage),
                )
            )

        return targets

    def validate_config(self) -> bool:
        return (
            self.config.momentum.ema_fast_period > 0
            and self.config.momentum.ema_slow_period > self.config.momentum.ema_fast_period
            and len(self.config.symbols) > 0
        )


# ---------------------------------------------------------------------------
# Section 3: Optuna objective function (per D-08, D-11, D-12)
# ---------------------------------------------------------------------------
def create_objective(
    ohlcv_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
):
    """Factory returning an Optuna objective function.

    Captures ohlcv_dict and funding_dict in closure.
    Returns Sharpe ratio from recomputed metrics (bars_per_year=2190).
    """

    def objective(trial: optuna.Trial) -> float:
        # D-08: 4 optimized parameters
        ema_fast = trial.suggest_int("ema_fast_period", 6, 20)
        ema_slow = trial.suggest_int("ema_slow_period", 20, 50)
        max_f_long = trial.suggest_float("max_funding_rate_long", 0.0005, 0.003)
        max_f_short = trial.suggest_float("max_funding_rate_short", -0.003, -0.0005)

        # Pitfall 3: fast must be strictly less than slow
        if ema_fast >= ema_slow:
            return -999.0

        # Construct CryptoTrendConfig with suggested params + fixed params (D-09)
        config = CryptoTrendConfig(
            symbols=SYMBOLS,
            momentum=MomentumConfig(
                ema_fast_period=ema_fast,
                ema_slow_period=ema_slow,
                interval=INTERVAL,
                lookback_days=30,
            ),
            funding_filter=FundingFilterConfig(
                max_funding_rate_long=max_f_long,
                max_funding_rate_short=max_f_short,
            ),
            allocation=PerpAllocationConfig(
                method=FIXED_ALLOCATION,
                position_limit_pct=FIXED_POSITION_LIMIT,
                leverage=FIXED_LEVERAGE,
            ),
        )

        # Create strategy with funding filter support
        strategy = BacktestCryptoTrend(
            config=config,
            ohlcv_dict=ohlcv_dict,
            funding_df=funding_dict,
        )

        # Run backtest with PortfolioBacktester (D-15: crypto_perp cost model)
        backtester = PortfolioBacktester(
            cost_model=COST_MODEL,
            initial_capital=INITIAL_CAPITAL,
        )
        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=start_date,
            end_date=end_date,
        )

        if result.status == "failed" or not result.equity_curve:
            return -999.0

        # D-12 constraint: minimum trade count
        if len(result.trades) < 50:
            return -999.0

        # Build equity series for recomputed metrics
        equity_series = pd.Series(
            [nav for _, nav in result.equity_curve],
            index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in result.equity_curve]),
        )

        # CRITICAL (Pitfall 1): Recompute metrics with bars_per_year=2190
        # PortfolioBacktester hardcodes bars_per_year=252 internally
        metrics = compute_metrics(
            equity_series, trades=[], bars_per_year=BARS_PER_YEAR
        )

        # D-12 constraint: positive total return
        if metrics.get("total_return", 0) <= 0:
            return -999.0

        # Store trial attributes for later retrieval
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("trade_count", len(result.trades))

        sharpe = metrics.get("sharpe_ratio", 0.0)
        return sharpe

    return objective


# ---------------------------------------------------------------------------
# Section 4: WFE validation (per D-13)
# ---------------------------------------------------------------------------
def run_wfe_validation(
    ohlcv_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
    best_params: dict,
) -> dict:
    """Run walk-forward efficiency validation for a parameter set.

    Uses generate_windows + manual loop (cannot use analyzer.analyze()
    because it expects BaseStrategy, not PortfolioStrategy).

    Returns dict with windows, avg_wfe, passed status.
    """
    analyzer = WalkForwardAnalyzer(
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(rules=[]),
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
    )

    data_length = min(len(df) for df in ohlcv_dict.values())
    windows = analyzer.generate_windows(data_length, WF_CONFIG)

    if not windows:
        logger.warning("No WFE windows generated (data_length=%d)", data_length)
        return {"windows": [], "avg_wfe": 0.0, "passed": False}

    print(f"\n  WFE: {len(windows)} windows, data_length={data_length}")
    print(
        f"  Config: train={WF_CONFIG.train_days} bars, "
        f"test={WF_CONFIG.test_days} bars, step={WF_CONFIG.step_days} bars"
    )

    # Build config from best_params
    config = CryptoTrendConfig(
        symbols=SYMBOLS,
        momentum=MomentumConfig(
            ema_fast_period=best_params["ema_fast_period"],
            ema_slow_period=best_params["ema_slow_period"],
            interval=INTERVAL,
            lookback_days=30,
        ),
        funding_filter=FundingFilterConfig(
            max_funding_rate_long=best_params["max_funding_rate_long"],
            max_funding_rate_short=best_params["max_funding_rate_short"],
        ),
        allocation=PerpAllocationConfig(
            method=FIXED_ALLOCATION,
            position_limit_pct=FIXED_POSITION_LIMIT,
            leverage=FIXED_LEVERAGE,
        ),
    )

    window_results = []
    for i, ((train_s, train_e), (test_s, test_e)) in enumerate(windows):
        # Slice OHLCV and funding by bar indices
        is_ohlcv = {sym: df.iloc[train_s:train_e] for sym, df in ohlcv_dict.items()}
        oos_ohlcv = {sym: df.iloc[test_s:test_e] for sym, df in ohlcv_dict.items()}

        # Slice funding data to match OHLCV time range
        is_funding = _slice_funding(funding_dict, is_ohlcv)
        oos_funding = _slice_funding(funding_dict, oos_ohlcv)

        # IS backtest
        is_metrics = _run_portfolio_backtest(config, is_ohlcv, is_funding)

        # OOS backtest
        oos_metrics = _run_portfolio_backtest(config, oos_ohlcv, oos_funding)

        # CRITICAL: compute_wfe uses annualized_return, NOT Sharpe ratio
        # (confirmed by wfo_77_structural.py lines 499-501)
        wfe = compute_wfe(
            is_metrics.get("annualized_return", 0.0),
            oos_metrics.get("annualized_return", 0.0),
        )

        window_result = {
            "window": i,
            "train_range": [train_s, train_e],
            "test_range": [test_s, test_e],
            "is_ann_return": is_metrics.get("annualized_return", 0.0),
            "oos_ann_return": oos_metrics.get("annualized_return", 0.0),
            "is_sharpe": is_metrics.get("sharpe_ratio", 0.0),
            "oos_sharpe": oos_metrics.get("sharpe_ratio", 0.0),
            "wfe": wfe,
        }
        window_results.append(window_result)

        print(
            f"  Window {i}: IS_ann={is_metrics.get('annualized_return', 0):.4f}, "
            f"OOS_ann={oos_metrics.get('annualized_return', 0):.4f}, WFE={wfe:.4f}"
        )

    # Compute average WFE
    wfe_values = [w["wfe"] for w in window_results]
    avg_wfe = sum(wfe_values) / len(wfe_values) if wfe_values else 0.0
    passed = avg_wfe >= 0.60  # D-19 threshold

    print(f"  Avg WFE: {avg_wfe:.4f} ({'PASSED' if passed else 'FAILED'})")

    return {
        "windows": window_results,
        "avg_wfe": avg_wfe,
        "passed": passed,
    }


def _slice_funding(
    funding_dict: dict[str, pd.DataFrame],
    ohlcv_slice: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Slice funding DataFrames to match the time range of OHLCV slice."""
    result = {}
    for symbol, funding in funding_dict.items():
        if symbol not in ohlcv_slice or ohlcv_slice[symbol].empty:
            continue
        ohlcv = ohlcv_slice[symbol]
        start_ts = ohlcv.index[0]
        end_ts = ohlcv.index[-1]
        mask = (funding.index >= start_ts) & (funding.index <= end_ts)
        result[symbol] = funding[mask]
    return result


def _run_portfolio_backtest(
    config: CryptoTrendConfig,
    ohlcv_dict: dict[str, pd.DataFrame],
    funding_dict: dict[str, pd.DataFrame],
) -> dict:
    """Run a single PortfolioBacktester pass and return recomputed metrics."""
    strategy = BacktestCryptoTrend(
        config=config,
        ohlcv_dict=ohlcv_dict,
        funding_df=funding_dict,
    )

    # Determine date range from OHLCV data
    all_dates = set()
    for df in ohlcv_dict.values():
        for idx in df.index:
            if isinstance(idx, pd.Timestamp):
                all_dates.add(idx.date())
            elif isinstance(idx, date):
                all_dates.add(idx)

    if not all_dates:
        return {"sharpe_ratio": 0.0, "annualized_return": 0.0, "total_return": 0.0}

    sorted_dates = sorted(all_dates)
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]

    backtester = PortfolioBacktester(
        cost_model=COST_MODEL,
        initial_capital=INITIAL_CAPITAL,
    )
    result = backtester.run(
        strategy=strategy,
        ohlcv_dict=ohlcv_dict,
        start_date=start_date,
        end_date=end_date,
    )

    if result.status == "failed" or not result.equity_curve:
        return {"sharpe_ratio": 0.0, "annualized_return": 0.0, "total_return": 0.0}

    # Recompute metrics with bars_per_year=2190 (Pitfall 1 fix)
    equity_series = pd.Series(
        [nav for _, nav in result.equity_curve],
        index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in result.equity_curve]),
    )
    return compute_metrics(equity_series, trades=[], bars_per_year=BARS_PER_YEAR)


# ---------------------------------------------------------------------------
# Section 5: Main function and output (per D-18, D-19)
# ---------------------------------------------------------------------------
def main() -> int:
    """Run Optuna CryptoTrend optimization + WFE validation."""
    repo = RemoteDataRepository.from_settings()

    # Step 1: Pre-fetch OHLCV for all symbols
    print(f"Fetching OHLCV data for {len(SYMBOLS)} symbols ({INTERVAL})...")
    print(f"Period: {START.date()} to {END.date()}")
    ohlcv_dict: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        df = repo.read_ohlcv(
            symbol=symbol,
            market="crypto_perp",
            interval=INTERVAL,
            start=START,
            end=END,
        )
        if not df.empty:
            ohlcv_dict[symbol] = df
            print(f"  {symbol}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        else:
            print(f"  {symbol}: NO DATA")

    if not ohlcv_dict:
        print("ERROR: No OHLCV data loaded. Aborting.")
        return 1

    # Strip timezone from OHLCV indices (Pitfall 7 fix)
    # PortfolioBacktester uses pd.Timestamp(date) for comparison, needs naive TZ
    for symbol in list(ohlcv_dict):
        df = ohlcv_dict[symbol]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            ohlcv_dict[symbol] = df.tz_localize(None)

    # Step 2: Pre-fetch funding rates for all symbols (post-ETF period)
    print(f"\nFetching funding rates for {len(SYMBOLS)} symbols...")
    funding_dict: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        funding = repo.read_funding_rates(symbol=symbol, start="2024-01-01")
        if not funding.empty:
            # Ensure DatetimeIndex and strip timezone for consistency
            if not isinstance(funding.index, pd.DatetimeIndex):
                if "timestamp" in funding.columns:
                    funding = funding.set_index("timestamp")
                elif "time" in funding.columns:
                    funding = funding.set_index("time")
                funding.index = pd.to_datetime(funding.index)
            if hasattr(funding.index, "tz") and funding.index.tz is not None:
                funding = funding.tz_localize(None)
            funding = funding.sort_index()
            funding_dict[symbol] = funding
            print(
                f"  {symbol}: {len(funding)} funding records "
                f"({funding.index[0]} to {funding.index[-1]})"
            )
        else:
            print(f"  {symbol}: NO FUNDING DATA")

    if not funding_dict:
        print("WARNING: No funding data loaded. Optimization will run without funding filter.")

    # Step 3: Run Optuna search
    print(f"\n{'='*60}")
    print(f"Running Optuna TPE search: {N_TRIALS} trials")
    print(f"Baseline Sharpe (Phase 75): {BASELINE_SHARPE}")
    print(f"{'='*60}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    start_d = START.date() if isinstance(START, datetime) else START
    end_d = END.date() if isinstance(END, datetime) else END

    objective = create_objective(ohlcv_dict, funding_dict, start_d, end_d)
    study.optimize(objective, n_trials=N_TRIALS)

    # Extract top-3 completed trials (filter out -999.0 penalty trials)
    completed_trials = [
        t for t in study.trials
        if t.value is not None and t.value > -999.0
    ]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    top3_trials = completed_trials[:3]

    if not top3_trials:
        print("\nERROR: No valid trials completed. All returned penalty value.")
        return 1

    # Print top-3 results header
    print(f"\n{'='*60}")
    print("Top-3 Parameter Sets")
    print(f"{'='*60}")
    print(
        f"{'Rank':<5} {'EMA_F':<6} {'EMA_S':<6} {'MaxF_L':<10} {'MaxF_S':<10} "
        f"{'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'WinRate':<8} {'WFE':<8}"
    )
    print("-" * 88)

    # Step 4: Run WFE validation for each of top-3 parameter sets
    top3_results = []
    for rank, trial in enumerate(top3_trials, 1):
        params = dict(trial.params)
        metrics = trial.user_attrs.get("metrics", {})
        trade_count = trial.user_attrs.get("trade_count", 0)

        print(f"\nValidating rank #{rank} via WFE...")
        wfe_result = run_wfe_validation(ohlcv_dict, funding_dict, params)

        entry = {
            "rank": rank,
            "trial_number": trial.number,
            "params": params,
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "annualized_return": metrics.get("annualized_return", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "trade_count": trade_count,
            "wfe": wfe_result,
        }
        top3_results.append(entry)

    # Step 5: Print comparison table (per D-18)
    print(f"\n{'='*60}")
    print("Final Comparison Table (D-18)")
    print(f"{'='*60}")
    print(
        f"{'Rank':<5} {'EMA_F':<6} {'EMA_S':<6} {'MaxF_L':<10} {'MaxF_S':<10} "
        f"{'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'WinRate':<8} {'WFE':<8}"
    )
    print("-" * 88)

    for entry in top3_results:
        p = entry["params"]
        print(
            f"{entry['rank']:<5} "
            f"{p['ema_fast_period']:<6} "
            f"{p['ema_slow_period']:<6} "
            f"{p['max_funding_rate_long']:<10.4f} "
            f"{p['max_funding_rate_short']:<10.4f} "
            f"{entry['sharpe_ratio']:<8.3f} "
            f"{entry['max_drawdown']:<8.3f} "
            f"{entry['trade_count']:<8} "
            f"{entry['win_rate']:<8.3f} "
            f"{entry['wfe']['avg_wfe']:<8.3f}"
        )

    print(f"\nBaseline (Phase 75): Sharpe {BASELINE_SHARPE}")

    # Step 6: Save JSON artifact (per D-19)
    best = top3_results[0]
    best_sharpe = best["sharpe_ratio"]
    best_wfe = best["wfe"]["avg_wfe"]
    best_wfe_passed = best["wfe"]["passed"]

    # D-19: recommendation logic
    if best_sharpe > BASELINE_SHARPE and best_wfe_passed:
        recommendation = "UPDATE_CONFIG"
        rec_reason = (
            f"Best Sharpe {best_sharpe:.3f} > baseline {BASELINE_SHARPE} "
            f"AND WFE {best_wfe:.3f} >= 0.60"
        )
    else:
        recommendation = "NO_CHANGE"
        reasons = []
        if best_sharpe <= BASELINE_SHARPE:
            reasons.append(
                f"Best Sharpe {best_sharpe:.3f} <= baseline {BASELINE_SHARPE}"
            )
        if not best_wfe_passed:
            reasons.append(f"WFE {best_wfe:.3f} < 0.60")
        rec_reason = " AND ".join(reasons)

    print(f"\nRecommendation: {recommendation}")
    print(f"Reason: {rec_reason}")

    # Build output JSON
    output = {
        "phase": "80",
        "script": "optuna_80_cryptotrend.py",
        "baseline_sharpe_phase75": BASELINE_SHARPE,
        "n_trials": N_TRIALS,
        "n_completed": len(completed_trials),
        "bars_per_year": BARS_PER_YEAR,
        "cost_model": "crypto_perp",
        "period": {
            "start": START.date().isoformat(),
            "end": END.date().isoformat(),
        },
        "fixed_params": {
            "leverage": FIXED_LEVERAGE,
            "allocation": FIXED_ALLOCATION,
            "position_limit_pct": FIXED_POSITION_LIMIT,
            "symbols": SYMBOLS,
        },
        "top3": [],
        "recommendation": recommendation,
        "recommendation_reason": rec_reason,
    }

    for entry in top3_results:
        # Sanitize WFE window results for JSON serialization
        wfe_windows = []
        for w in entry["wfe"]["windows"]:
            wfe_windows.append({
                "window": w["window"],
                "train_range": w["train_range"],
                "test_range": w["test_range"],
                "is_ann_return": float(w["is_ann_return"]),
                "oos_ann_return": float(w["oos_ann_return"]),
                "wfe": float(w["wfe"]),
            })

        output["top3"].append({
            "rank": entry["rank"],
            "trial_number": entry["trial_number"],
            "params": {
                k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in entry["params"].items()
            },
            "sharpe_ratio": float(entry["sharpe_ratio"]),
            "max_drawdown": float(entry["max_drawdown"]),
            "total_return": float(entry["total_return"]),
            "annualized_return": float(entry["annualized_return"]),
            "win_rate": float(entry["win_rate"]),
            "trade_count": int(entry["trade_count"]),
            "wfe": {
                "avg_wfe": float(entry["wfe"]["avg_wfe"]),
                "passed": bool(entry["wfe"]["passed"]),
                "windows": wfe_windows,
            },
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
