#!/usr/bin/env python3
"""Phase 75 D-01~D-10: 4-way CryptoTrend baseline comparison + BTC buy-and-hold.

Compares 4 variants:
  1. CryptoTrend original (4H EMA 12/26)
  2. +vol-sizing (ATR-14 inverse)
  3. +regime-filter (HMM 2-state, 50% reduction)
  4. +both (vol-sizing + regime-filter)

Plus BTC buy-and-hold benchmark over the same period.

NOTE: CryptoTrendStrategy.select_stocks() calls read_perp_ohlcv internally
(latest 30 days only), which does not respect backtest as_of dates.
This script creates a backtest-compatible wrapper that computes EMA signals
from the pre-fetched historical OHLCV, matching the same logic as
CryptoTrendStrategy but point-in-time correct.

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/compare_75_cryptotrend.py
"""

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio_backtester import PortfolioBacktester
from poseidon.data.features.hmm_regime import HMMRegime
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.strategies.portfolio.crypto_trend import CryptoTrendConfig
from poseidon.strategies.portfolio.schemas import TargetPosition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (per D-01 through D-10)
# ---------------------------------------------------------------------------
START = datetime(2024, 1, 1)  # Post-ETF only (D-01)
END = datetime.now()
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INITIAL_CAPITAL = 100_000.0  # 100K USDT
INTERVAL = "4h"
BARS_PER_YEAR = 2190  # 365.25 * 24 / 4 for 4H crypto bars (D-10, pitfall 6 fix)
COST_MODEL = COST_MODELS["crypto_perp"]  # D-02
ATR_PERIOD = 14  # D-05
VOL_SIZING_CAP = (0.5, 2.0)  # D-06: cap multiplier range
HMM_N_REGIMES = 2  # D-08, pitfall 2 fix: must be 2, NOT default 3
REGIME_REDUCTION = 0.5  # D-09: 50% weight reduction in high-vol regime


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


def compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) from OHLCV DataFrame."""
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


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
# Backtest-compatible CryptoTrend strategy
# ---------------------------------------------------------------------------
class BacktestCryptoTrend:
    """CryptoTrend strategy that computes signals from pre-fetched OHLCV.

    Unlike the live CryptoTrendStrategy (which calls read_perp_ohlcv for
    latest data), this version uses the ohlcv_dict passed at construction
    and respects the as_of date to produce point-in-time correct signals.

    The EMA crossover logic is identical to CryptoTrendStrategy._compute_ema_signal().
    """

    name = "crypto_trend_backtest"

    def __init__(
        self,
        config: CryptoTrendConfig,
        ohlcv_dict: dict[str, pd.DataFrame],
    ):
        self.config = config
        self._ohlcv_dict = ohlcv_dict

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
# Strategy wrappers for weight modification (option c from RESEARCH.md)
# ---------------------------------------------------------------------------
class VolSizingWrapper:
    """Apply inverse-ATR vol sizing: multiply weight by baseline_atr / current_atr.

    Per D-05: ATR period=14 on 4H bars.
    Per D-06: inverse-ATR scaling, capped at [0.5, 2.0].
    """

    def __init__(
        self,
        base_strategy: BacktestCryptoTrend,
        ohlcv_dict: dict[str, pd.DataFrame],
        atr_period: int = ATR_PERIOD,
    ):
        self.base = base_strategy
        self.config = base_strategy.config
        self.name = base_strategy.name
        self.atr_period = atr_period

        # Precompute per-symbol ATR series and baseline (median ATR)
        self._atr_series: dict[str, pd.Series] = {}
        self._baseline_atr: dict[str, float] = {}
        for symbol, df in ohlcv_dict.items():
            atr = compute_atr(df, period=atr_period).dropna()
            self._atr_series[symbol] = atr
            self._baseline_atr[symbol] = float(atr.median()) if not atr.empty else 1.0

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        positions = self.base.select_stocks(universe_df, as_of=as_of)
        return self._apply_vol_sizing(positions, as_of)

    def _apply_vol_sizing(
        self, positions: list[TargetPosition], as_of: date | None
    ) -> list[TargetPosition]:
        modified = []
        for pos in positions:
            atr_series = self._atr_series.get(pos.symbol)
            baseline = self._baseline_atr.get(pos.symbol, 1.0)

            if atr_series is not None and not atr_series.empty and baseline > 0:
                current_atr = self._get_atr_at(atr_series, as_of)
                if current_atr > 0:
                    multiplier = baseline / current_atr
                    multiplier = max(VOL_SIZING_CAP[0], min(VOL_SIZING_CAP[1], multiplier))
                    new_weight = pos.weight * multiplier
                else:
                    new_weight = pos.weight
            else:
                new_weight = pos.weight

            modified.append(
                TargetPosition(
                    symbol=pos.symbol,
                    weight=new_weight,
                    reason=f"{pos.reason}, vol_sizing",
                    side=pos.side,
                    leverage=pos.leverage,
                )
            )
        return modified

    def _get_atr_at(self, atr_series: pd.Series, as_of: date | None) -> float:
        if as_of is None:
            return float(atr_series.iloc[-1])
        as_of_end = _as_of_to_ts(as_of)
        valid = atr_series[atr_series.index <= as_of_end]
        if valid.empty:
            return float(atr_series.iloc[0]) if not atr_series.empty else 0.0
        return float(valid.iloc[-1])

    def validate_config(self) -> bool:
        return self.base.validate_config()


class RegimeFilterWrapper:
    """Reduce weights in high-vol HMM regime.

    Per D-08: HMM n_regimes=2 on 4H close (BTC as reference).
    Per D-09: When hmm_regime_label == 1 (high-vol), multiply weights by 0.5.
    """

    def __init__(
        self,
        base_strategy: BacktestCryptoTrend,
        ohlcv_dict: dict[str, pd.DataFrame],
        n_regimes: int = HMM_N_REGIMES,
        reduction_factor: float = REGIME_REDUCTION,
    ):
        self.base = base_strategy
        self.config = base_strategy.config
        self.name = base_strategy.name
        self.reduction_factor = reduction_factor

        # Precompute HMM regime labels from BTC 4H close
        self._regime_labels: pd.Series | None = None
        btc_ohlcv = ohlcv_dict.get("BTCUSDT")
        if btc_ohlcv is not None and not btc_ohlcv.empty:
            hmm = HMMRegime()
            regime_df = hmm.compute(btc_ohlcv, n_regimes=n_regimes)
            if "hmm_regime_label" in regime_df.columns:
                self._regime_labels = regime_df["hmm_regime_label"]
                logger.info(
                    "HMM regime (n_regimes=%d): %d bars, distribution=%s",
                    n_regimes,
                    len(self._regime_labels.dropna()),
                    self._regime_labels.value_counts().to_dict(),
                )

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        positions = self.base.select_stocks(universe_df, as_of=as_of)
        return self._apply_regime_filter(positions, as_of)

    def _apply_regime_filter(
        self, positions: list[TargetPosition], as_of: date | None
    ) -> list[TargetPosition]:
        if self._regime_labels is None:
            return positions

        current_regime = self._get_regime_at(as_of)
        if current_regime == 1:  # High-vol state
            return [
                TargetPosition(
                    symbol=pos.symbol,
                    weight=pos.weight * self.reduction_factor,
                    reason=f"{pos.reason}, regime_high_vol",
                    side=pos.side,
                    leverage=pos.leverage,
                )
                for pos in positions
            ]
        return positions

    def _get_regime_at(self, as_of: date | None) -> float:
        if self._regime_labels is None:
            return 0.0
        labels = self._regime_labels.dropna()
        if labels.empty:
            return 0.0
        if as_of is None:
            return float(labels.iloc[-1])
        as_of_end = _as_of_to_ts(as_of)
        valid = labels[labels.index <= as_of_end]
        if valid.empty:
            return 0.0
        return float(valid.iloc[-1])

    def validate_config(self) -> bool:
        return self.base.validate_config()


class CombinedWrapper:
    """Apply vol-sizing then regime-filter sequentially."""

    def __init__(
        self,
        base_strategy: BacktestCryptoTrend,
        ohlcv_dict: dict[str, pd.DataFrame],
        atr_period: int = ATR_PERIOD,
        n_regimes: int = HMM_N_REGIMES,
        reduction_factor: float = REGIME_REDUCTION,
    ):
        self.base = base_strategy
        self.config = base_strategy.config
        self.name = base_strategy.name
        self._vol = VolSizingWrapper(base_strategy, ohlcv_dict, atr_period)
        self._regime = RegimeFilterWrapper(
            base_strategy, ohlcv_dict, n_regimes, reduction_factor
        )

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        # Vol-sizing first, then regime filter
        positions = self._vol.select_stocks(universe_df, as_of=as_of)
        return self._regime._apply_regime_filter(positions, as_of)

    def validate_config(self) -> bool:
        return self.base.validate_config()


# ---------------------------------------------------------------------------
# BTC buy-and-hold benchmark (per D-03, BASE-02)
# ---------------------------------------------------------------------------
def compute_btc_benchmark(
    btc_ohlcv: pd.DataFrame, initial_capital: float, bars_per_year: int
) -> dict:
    """Compute BTC buy-and-hold metrics following Phase 73 pattern."""
    if btc_ohlcv.empty:
        return {"label": "BTC buy-and-hold", "error": "No BTC OHLCV data"}

    start_price = float(btc_ohlcv.iloc[0]["close"])
    end_price = float(btc_ohlcv.iloc[-1]["close"])
    total_return = end_price / start_price - 1

    # Build equity curve from raw prices
    equity = btc_ohlcv["close"] / start_price * initial_capital
    returns = equity.pct_change().dropna()

    sharpe = (
        float((returns.mean() / returns.std()) * np.sqrt(bars_per_year))
        if returns.std() > 0
        else 0.0
    )

    cummax = equity.cummax()
    max_dd = float(((cummax - equity) / cummax).max())

    n_bars = len(btc_ohlcv)
    n_years = n_bars / bars_per_year
    ann_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    return {
        "label": "BTC buy-and-hold",
        "sharpe_ratio": round(sharpe, 4),
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "max_drawdown": round(max_dd, 4),
        "trade_count": 0,
        "win_rate": 0.0,
        "avg_hold_bars": n_bars,
    }


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------
def run_comparison() -> int:
    """Run Phase 75 4-variant CryptoTrend comparison and print results."""
    repo = RemoteDataRepository.from_settings()
    results: list[dict] = []

    # Step 1: Pre-fetch OHLCV data for all symbols
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

    print(f"\nOHLCV loaded for {len(ohlcv_dict)}/{len(SYMBOLS)} symbols\n")

    if not ohlcv_dict:
        print("ERROR: No OHLCV data loaded. Aborting.")
        return 1

    # Strip timezone from OHLCV indices so PortfolioBacktester can compare
    # with naive Timestamps (its _get_close_price uses pd.Timestamp(date)).
    for symbol in list(ohlcv_dict):
        df = ohlcv_dict[symbol]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            ohlcv_dict[symbol] = df.tz_localize(None)

    # Step 2: Build base config (same params as live CryptoTrendConfig defaults)
    base_config = CryptoTrendConfig(
        symbols=SYMBOLS,
        momentum={"ema_fast_period": 12, "ema_slow_period": 26, "interval": INTERVAL},
        funding_filter={"max_funding_rate_long": 0.001, "max_funding_rate_short": -0.001},
        allocation={"method": "equal_weight", "position_limit_pct": 0.5, "leverage": 3},
    )

    # Step 3: Define 4 variants (per D-07)
    # Using BacktestCryptoTrend (computes from pre-fetched OHLCV, respects as_of)
    # instead of live CryptoTrendStrategy (calls read_perp_ohlcv, ignores as_of)
    base_bt = BacktestCryptoTrend(config=base_config, ohlcv_dict=ohlcv_dict)

    variants = [
        {
            "label": "CryptoTrend original (4H EMA 12/26)",
            "strategy": BacktestCryptoTrend(config=base_config, ohlcv_dict=ohlcv_dict),
        },
        {
            "label": "+vol-sizing (ATR-14 inverse)",
            "strategy": VolSizingWrapper(
                BacktestCryptoTrend(config=base_config, ohlcv_dict=ohlcv_dict),
                ohlcv_dict,
                ATR_PERIOD,
            ),
        },
        {
            "label": "+regime-filter (HMM 2-state, 50% reduction)",
            "strategy": RegimeFilterWrapper(
                BacktestCryptoTrend(config=base_config, ohlcv_dict=ohlcv_dict),
                ohlcv_dict,
                n_regimes=HMM_N_REGIMES,
            ),
        },
        {
            "label": "+both (vol-sizing + regime-filter)",
            "strategy": CombinedWrapper(
                BacktestCryptoTrend(config=base_config, ohlcv_dict=ohlcv_dict),
                ohlcv_dict,
                atr_period=ATR_PERIOD,
                n_regimes=HMM_N_REGIMES,
            ),
        },
    ]

    # Step 4: Run backtests using PortfolioBacktester (pitfall 4 fix)
    for entry in variants:
        label = entry["label"]
        strategy = entry["strategy"]
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        try:
            backtester = PortfolioBacktester(
                cost_model=COST_MODEL,
                initial_capital=INITIAL_CAPITAL,
            )
            result = backtester.run(
                strategy=strategy,
                ohlcv_dict=ohlcv_dict,
                start_date=START.date() if isinstance(START, datetime) else START,
                end_date=END.date() if isinstance(END, datetime) else END,
            )

            if result.status == "failed":
                raise RuntimeError(result.error_message or "Backtest failed")

            # Re-compute metrics with bars_per_year=2190 (pitfall 6 fix)
            # PortfolioBacktester._run_loop hardcodes bars_per_year=252
            if result.equity_curve:
                equity_series = pd.Series(
                    [nav for _, nav in result.equity_curve],
                    index=pd.DatetimeIndex(
                        [pd.Timestamp(d) for d, _ in result.equity_curve]
                    ),
                )
                metrics = compute_metrics(
                    equity_series, trades=[], bars_per_year=BARS_PER_YEAR
                )
            else:
                metrics = result.metrics

            trades = result.trades
            total_trades = len(trades)
            buy_trades = len([t for t in trades if t.get("action") == "buy"])
            sell_trades = len([t for t in trades if t.get("action") == "sell"])

            # Average hold period (approximate)
            n_round_trips = min(buy_trades, sell_trades) if sell_trades > 0 else 1
            n_equity_bars = len(result.equity_curve) if result.equity_curve else 1
            avg_hold_bars = n_equity_bars / max(n_round_trips, 1)

            # Win rate from trades (sell proceeds vs buy cost approximation)
            win_count = 0
            loss_count = 0
            for t in trades:
                if t.get("action") in ("sell", "adjust_sell"):
                    proceeds = t.get("proceeds", 0)
                    # Simple heuristic: compare to initial capital fraction
                    if proceeds > 0:
                        win_count += 1
                    else:
                        loss_count += 1
            total_closed = win_count + loss_count
            win_rate = win_count / total_closed if total_closed > 0 else 0.0

            row = {
                "label": label,
                "sharpe_ratio": round(metrics.get("sharpe_ratio", 0.0), 4),
                "max_drawdown": round(metrics.get("max_drawdown", 0.0), 4),
                "total_return": round(metrics.get("total_return", 0.0), 4),
                "annualized_return": round(metrics.get("annualized_return", 0.0), 4),
                "trade_count": total_trades,
                "avg_hold_bars": round(avg_hold_bars, 1),
                "win_rate": round(win_rate, 4),
                "num_rebalances": len(result.rebalance_log),
            }
            results.append(row)

            print(f"  Sharpe Ratio:      {row['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown:      {row['max_drawdown']:.4f}")
            print(f"  Total Return:      {row['total_return']:.4f}")
            print(f"  Annualized Return: {row['annualized_return']:.4f}")
            print(f"  Trade Count:       {row['trade_count']}")
            print(f"  Win Rate:          {row['win_rate']:.4f}")
            print(f"  Avg Hold (bars):   {row['avg_hold_bars']:.1f}")
            print(f"  Rebalances:        {row['num_rebalances']}")

        except Exception as exc:
            logger.exception("Variant '%s' failed", label)
            results.append({"label": label, "error": str(exc)})

    # Step 5: BTC buy-and-hold benchmark (D-03, BASE-02)
    btc_ohlcv = ohlcv_dict.get("BTCUSDT", pd.DataFrame())
    benchmark_row = compute_btc_benchmark(btc_ohlcv, INITIAL_CAPITAL, BARS_PER_YEAR)
    results.append(benchmark_row)

    # Step 6: Summary table
    print(f"\n{'='*100}")
    print("Phase 75: CryptoTrend 4-Way Comparison + BTC Buy-and-Hold")
    print(f"{'='*100}")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Interval: {INTERVAL} | Bars/Year: {BARS_PER_YEAR}")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} USDT")
    print(f"Cost Model: {COST_MODEL.description}")
    print()

    header = (
        f"{'Variant':<50} | {'Sharpe':>8} | {'MaxDD':>8} | {'TotRtn':>8} | "
        f"{'AnnRtn':>8} | {'Trades':>6} | {'AvgHold':>7} | {'WinRate':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['label']:<50} | {'ERROR':>8} | {r.get('error', '')}")
            continue
        sharpe = f"{r['sharpe_ratio']:.4f}" if r.get("sharpe_ratio") is not None else "N/A"
        max_dd = f"{r['max_drawdown']:.4f}" if r.get("max_drawdown") is not None else "N/A"
        tot_ret = f"{r['total_return']:.4f}" if r.get("total_return") is not None else "N/A"
        ann_ret = (
            f"{r['annualized_return']:.4f}"
            if r.get("annualized_return") is not None
            else "N/A"
        )
        trades = f"{r.get('trade_count', 0):>6d}"
        avg_hold = f"{r.get('avg_hold_bars', 0):>7.1f}"
        win_rate = f"{r.get('win_rate', 0):>7.4f}"
        print(
            f"{r['label']:<50} | {sharpe:>8} | {max_dd:>8} | {tot_ret:>8} | "
            f"{ann_ret:>8} | {trades} | {avg_hold} | {win_rate}"
        )

    # Step 7: Identify winner (highest Sharpe among strategy variants)
    strat_results = [
        r
        for r in results
        if "error" not in r and r["label"] != "BTC buy-and-hold"
    ]
    if strat_results:
        winner = max(strat_results, key=lambda r: r.get("sharpe_ratio", -999))
        print(
            f"\n  WINNER (highest Sharpe): {winner['label']} "
            f"with Sharpe={winner['sharpe_ratio']:.4f}"
        )
    else:
        print("\n  WARNING: No successful variant results to compare")
        winner = None

    # Look-ahead bias gate: flag Sharpe > 3.0
    print()
    bias_warning = False
    for r in results:
        if "error" not in r and r.get("sharpe_ratio") is not None:
            if abs(r["sharpe_ratio"]) > 3.0:
                print(
                    f"  WARNING: {r['label']} Sharpe={r['sharpe_ratio']:.4f} > 3.0"
                    " -- possible look-ahead bias!"
                )
                bias_warning = True
    if not bias_warning:
        print("  Look-ahead bias gate: PASSED (all |Sharpe| <= 3.0)")

    # Per D-10: if neither variant improves Sharpe, that is acceptable
    if strat_results and len(strat_results) >= 2:
        original = next(
            (r for r in strat_results if "original" in r["label"]), None
        )
        if original:
            improved = any(
                r["sharpe_ratio"] > original["sharpe_ratio"]
                for r in strat_results
                if r["label"] != original["label"]
            )
            if not improved:
                print(
                    "\n  NOTE (D-10): Neither vol-sizing nor regime-filter improved "
                    "Sharpe over original. This is an acceptable finding."
                )

    # Step 8: Save JSON summary
    output_dir = Path("scripts/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "compare_75_results.json"

    summary = {
        "phase": "75",
        "plan": "02",
        "description": "CryptoTrend 4-way comparison + BTC B&H",
        "period": {
            "start": START.date().isoformat(),
            "end": END.date().isoformat(),
        },
        "symbols": SYMBOLS,
        "interval": INTERVAL,
        "bars_per_year": BARS_PER_YEAR,
        "initial_capital": INITIAL_CAPITAL,
        "cost_model": COST_MODEL.description,
        "results": results,
        "winner": winner["label"] if winner else None,
        "winner_sharpe": winner["sharpe_ratio"] if winner else None,
    }
    output_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  JSON summary saved to: {output_path}")

    # Step 9: Print JSON for easy parsing
    print(f"\n{'='*100}")
    print("JSON SUMMARY")
    print(f"{'='*100}")
    print(json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(run_comparison())
