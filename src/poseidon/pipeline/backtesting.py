"""Run one DSL v2 strategy on pipeline bars and summarise it in TWD.

Every pipeline backtest uses the frozen ``config_json["backtest"]`` block:
``tw_futures_pipeline`` costs, ``FIXED_CONTRACTS`` = 1, gap-aware stops and
1,000,000 TWD capital. Metrics are TWD-based (portfolio equity metrics were
historically wrong for futures, see Step 0):

* ``total_pnl_twd`` -- sum of closed round trips, net of commission and tax
  (slippage is inside the fill prices); ``gross_pnl_twd`` adds the fees back.
* Sharpe -- daily mark-to-market PnL per **trading day** (TAIFEX convention:
  the night session belongs to the next trading day; computed as Asia/Taipei
  time + 9 h, Saturday/Sunday rolled to Monday). Every trading day with bars
  counts, flat days included. ``sharpe_daily`` = mean / std (ddof=1) and is
  the per-period figure the trial ledger and DSR use;
  ``sharpe_annualised`` = ``sharpe_daily * sqrt(252)``.
* ``skew`` and ``kurtosis`` are population moments of the same daily series;
  ``kurtosis`` is **non-excess** (Pearson, 3.0 for normal returns).
  Daily PnL / capital gives the same three numbers (scale invariant).
* ``max_drawdown_twd`` -- trade-level: peak-to-trough of cumulative closed-trade
  PnL in exit order, starting from 0.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import SignalAction
from poseidon.strategies.dsl.v2.evaluator import DEFAULT_BAR_LABEL

MARKET_TZ = "Asia/Taipei"
TRADING_DAY_SHIFT = pd.Timedelta(hours=9)
EXIT_REASONS = {"close_sl": "stop_loss", "close_tp": "take_profit"}
SHARPE_CONVENTION = (
    "daily mark-to-market PnL per trading day (night session -> next trading day; Asia/Taipei + 9h, "
    "weekend rolled to Monday), all trading days with bars incl. flat days; sharpe_daily = mean/std(ddof=1), "
    "sharpe_annualised = sharpe_daily*sqrt(252); skew/kurtosis are population moments, kurtosis non-excess"
)


class BacktestFailed(RuntimeError):
    """The runner reported ``status != completed`` for this strategy."""


class _Interrupted(Exception):
    """Raised inside the runner loop once the stage was told to stop."""


def finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def build_runner(strategy, backtest_config: dict) -> BacktestRunner:
    sizing = backtest_config["sizing"]
    if sizing["mode"] != SizingMode.FIXED_CONTRACTS.value:
        raise ValueError(f"pipeline backtests size by fixed_contracts, not {sizing['mode']!r}")
    return BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(),
        cost_model=get_cost_model(backtest_config["cost_model"]),
        initial_capital=float(backtest_config["initial_capital_twd"]),
        sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=int(sizing["contracts"])),
        gap_aware_stops=bool(backtest_config["gap_aware_stops"]),
    )


def trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Trading-day label (midnight, naive) of each UTC bar timestamp."""

    local = index.tz_convert(MARKET_TZ).tz_localize(None) + TRADING_DAY_SHIFT
    days = local.normalize()
    weekday = days.weekday
    roll = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
    return days + pd.to_timedelta(roll, unit="D")


def daily_pnl(equity_curve, initial_capital: float) -> pd.DataFrame:
    """``trading_day, pnl_twd, equity_twd`` from the runner's per-bar equity curve."""

    if not equity_curve:
        return pd.DataFrame(columns=["pnl_twd", "equity_twd"])
    times = pd.DatetimeIndex([t for t, _, _ in equity_curve])
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    equity = pd.Series([float(e) for _, e, _ in equity_curve], index=times)
    end_of_day = equity.groupby(trading_days(times)).last()
    pnl = end_of_day.diff()
    pnl.iloc[0] = end_of_day.iloc[0] - initial_capital
    frame = pd.DataFrame({"pnl_twd": pnl, "equity_twd": end_of_day})
    frame.index.name = "trading_day"
    return frame


def daily_stats(pnl: pd.Series, annualisation: float) -> dict:
    values = pnl.to_numpy(dtype=float)
    n_obs = len(values)
    stats = {
        "n_obs": n_obs,
        "mean_daily_pnl_twd": finite(values.mean()) if n_obs else None,
        "std_daily_pnl_twd": None,
        "sharpe_daily": None,
        "sharpe_annualised": None,
        "skew": None,
        "kurtosis": None,
    }
    if n_obs < 2:
        return stats
    std = float(values.std(ddof=1))
    stats["std_daily_pnl_twd"] = finite(std)
    if not std > 0:
        return stats
    from scipy.stats import kurtosis, skew

    sharpe = float(values.mean()) / std
    stats["sharpe_daily"] = finite(sharpe)
    stats["sharpe_annualised"] = finite(sharpe * math.sqrt(annualisation))
    stats["skew"] = finite(skew(values, bias=True))
    stats["kurtosis"] = finite(kurtosis(values, fisher=False, bias=True))
    return stats


@dataclass
class BacktestOutcome:
    metrics: dict
    trades: list[dict]
    daily: pd.DataFrame
    bars: int
    extra: dict = field(default_factory=dict)

    def files(self) -> dict[str, bytes]:
        return {"trades.csv": trades_csv(self.trades), "daily_pnl.csv": daily_csv(self.daily)}


TRADE_COLUMNS = (
    "trade_no",
    "side",
    "rule_id",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl_twd",
    "fees_twd",
    "pnl_twd",
    "exit_reason",
)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def trades_csv(trades: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(TRADE_COLUMNS)
    for trade in trades:
        writer.writerow([_cell(trade[column]) for column in TRADE_COLUMNS])
    return buffer.getvalue().encode("utf-8")


def daily_csv(daily: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("trading_day", "pnl_twd", "equity_twd"))
    for day, row in daily.iterrows():
        writer.writerow(
            (pd.Timestamp(day).date().isoformat(), repr(float(row["pnl_twd"])), repr(float(row["equity_twd"])))
        )
    return buffer.getvalue().encode("utf-8")


def _iso(value) -> str | None:
    return pd.Timestamp(value).isoformat() if value is not None else None


def _instrument(strategy, should_stop: Callable[[], bool]) -> tuple[dict, dict]:
    """Record exit reasons / entry rule ids from emitted signals; stop on request."""

    exits: dict[pd.Timestamp, str] = {}
    entries: dict[pd.Timestamp, str] = {}
    evaluate = strategy.evaluate

    def instrumented(features):
        if should_stop():
            raise _Interrupted("stage stop requested")
        signals = evaluate(features)
        for signal in signals:
            metadata = signal.metadata or {}
            stamp = pd.Timestamp(signal.signal_time)
            if signal.action == SignalAction.CLOSE:
                exits[stamp] = metadata.get("exit") or f"rule:{metadata.get('rule_id', '?')}"
            elif signal.action in (SignalAction.LONG, SignalAction.SHORT) and metadata.get("rule_id"):
                entries[stamp] = metadata["rule_id"]
        return signals

    strategy.evaluate = instrumented
    return exits, entries


def run_backtest(
    strategy,
    bars: pd.DataFrame,
    backtest_config: dict,
    *,
    should_stop: Callable[[], bool] = lambda: False,
) -> BacktestOutcome:
    """Run ``strategy`` on ``bars``; raises ``StageStopped`` when asked to stop mid-run."""

    exits, entries = _instrument(strategy, should_stop)
    runner = build_runner(strategy, backtest_config)
    result = runner.run(bars)
    if should_stop():
        from poseidon.pipeline.stages import StageStopped

        raise StageStopped("stage stopped during a backtest")
    if result.status != "completed":
        raise BacktestFailed(result.error_message or "backtest failed")
    portfolio = runner.portfolio
    cost_model = runner.cost_model
    return summarise(
        portfolio.trades,
        portfolio.equity_curve,
        backtest_config,
        cost_model=cost_model,
        exits=exits,
        entries=entries,
        bars=len(bars),
    )


def summarise(trade_records, equity_curve, backtest_config, *, cost_model, exits, entries, bars) -> BacktestOutcome:
    capital = float(backtest_config["initial_capital_twd"])
    annualisation = float(backtest_config["sharpe"]["annualisation_factor"])
    sides: dict = {}
    trips: list[dict] = []
    for record in trade_records:
        if record.exit_time is None:
            sides[pd.Timestamp(record.entry_time)] = record.action
            continue
        entry = pd.Timestamp(record.entry_time)
        reason = EXIT_REASONS.get(record.action) or exits.get(pd.Timestamp(record.exit_time), "signal")
        pnl = float(record.pnl)
        fees = float(record.fees)
        trips.append(
            {
                "trade_no": len(trips) + 1,
                "side": sides.get(entry, "?"),
                "rule_id": entries.get(entry),
                "entry_time": _iso(record.entry_time),
                "exit_time": _iso(record.exit_time),
                "entry_price": float(record.entry_price),
                "exit_price": float(record.exit_price),
                "quantity": float(record.quantity),
                "gross_pnl_twd": pnl + fees,
                "fees_twd": fees,
                "pnl_twd": pnl,
                "exit_reason": reason,
                "_slipped_legs": 1 + (record.action == "close"),
            }
        )
    open_at_end = len(sides) - len({pd.Timestamp(t["entry_time"]) for t in trips})

    pnls = np.array([t["pnl_twd"] for t in trips], dtype=float)
    fees_total = float(sum(t["fees_twd"] for t in trips))
    quantity_legs = float(sum(2 * t["quantity"] for t in trips))
    commission = quantity_legs * float(getattr(cost_model, "commission_per_contract", 0.0))
    tick = float(cost_model.tick_size or 0.0) * float(cost_model.slippage_ticks) * float(cost_model.point_value)
    slippage = float(sum(t["_slipped_legs"] * t["quantity"] for t in trips)) * tick
    for trip in trips:
        trip.pop("_slipped_legs")

    if len(pnls):
        cumulative = np.cumsum(pnls)
        peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
        max_dd = float(np.max(peaks - cumulative))
        max_dd = max(max_dd, 0.0)
    else:
        max_dd = 0.0

    daily = daily_pnl(equity_curve, capital)
    stats = daily_stats(daily["pnl_twd"], annualisation) if len(daily) else daily_stats(pd.Series(dtype=float), 1)
    n_trades = len(trips)
    wins = int((pnls > 0).sum()) if n_trades else 0
    metrics = {
        "total_pnl_twd": finite(pnls.sum()) if n_trades else 0.0,
        "gross_pnl_twd": finite(pnls.sum() + fees_total) if n_trades else 0.0,
        "fees_twd": finite(fees_total),
        "costs": {
            "commission_twd": finite(commission),
            "tax_twd": finite(fees_total - commission),
            "slippage_twd_estimate": finite(slippage),
            "note": "fees_twd = commission + tax; slippage is inside fill prices (1 tick per market-order leg, "
            "none on stop/target fills)",
        },
        "n_trades": n_trades,
        "n_long": sum(1 for t in trips if t["side"] == "long"),
        "n_short": sum(1 for t in trips if t["side"] == "short"),
        "wins": wins,
        "win_rate": finite(wins / n_trades) if n_trades else None,
        "avg_trade_twd": finite(pnls.mean()) if n_trades else None,
        "max_drawdown_twd": finite(max_dd),
        "max_drawdown_convention": "trade-level, cumulative closed-trade PnL in exit order from 0",
        "exit_reasons": dict(sorted(Counter(t["exit_reason"] for t in trips).items())),
        "open_position_at_end": open_at_end > 0,
        **stats,
        "sharpe_convention": SHARPE_CONVENTION,
        "kurtosis_convention": "non_excess",
    }
    return BacktestOutcome(metrics=metrics, trades=trips, daily=daily, bars=bars)


def backtest_settings(backtest_config: dict) -> dict:
    """The frozen settings a report restates (for humans reading one artifact)."""

    return {
        "cost_model": backtest_config["cost_model"],
        "sizing": backtest_config["sizing"],
        "gap_aware_stops": backtest_config["gap_aware_stops"],
        "initial_capital_twd": backtest_config["initial_capital_twd"],
        "bar_label": backtest_config.get("bar_label", DEFAULT_BAR_LABEL),
    }
