"""Run one DSL v2 strategy on pipeline bars and summarise it in TWD.

Every pipeline backtest uses the frozen ``config_json["backtest"]`` block:
``tw_futures_pipeline`` costs, ``FIXED_CONTRACTS`` = 1, gap-aware stops and
1,000,000 TWD capital. Metrics are TWD-based (portfolio equity metrics were
historically wrong for futures, see Step 0):

* ``total_pnl_twd`` -- sum of closed round trips, net of commission and tax
  (slippage is inside the fill prices); ``gross_pnl_twd`` adds the fees back.
* Sharpe -- daily mark-to-market PnL per **trading day** (TAIFEX convention:
  the night session belongs to the next trading day, :func:`trading_days`).
  Every trading day with bars counts, flat days included. ``sharpe_daily`` = mean / std (ddof=1) and is
  the per-period figure the trial ledger and DSR use;
  ``sharpe_annualised`` = ``sharpe_daily * sqrt(252)``.
* ``skew`` and ``kurtosis`` are population moments of the same daily series;
  ``kurtosis`` is **non-excess** (Pearson, 3.0 for normal returns).
  Daily PnL / capital gives the same three numbers (scale invariant).
* ``max_drawdown_twd`` -- trade-level: peak-to-trough of cumulative closed-trade
  PnL in exit order, starting from 0.

Engine (``backtest.engine``, absent = ``v1``): ``v1`` is ``BacktestRunner`` +
``BacktestPortfolio``, the engine of every run frozen so far; ``target_v1``
runs the same ``RuleStrategyV2`` as the spec v3 fast path
(``v3.dsl_bridge.v2_to_v3``) on the target-position engine's intraday clock
(``engine.intraday``: signal-close fills, protective stops, the synthetic R1
roll on settlement days). Both feed :func:`summarise`, so reports keep their
shape; trades match v1 one for one except that a position held across a
settlement-day 13:25 -> 13:30 switch pays the roll and is not charged the gap
between the months (``scripts/pv2_dsl_parity.py``).
"""

from __future__ import annotations

import csv
import dataclasses
import io
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import CostModel, get_cost_model
from poseidon.backtest.portfolio import SizingConfig, SizingMode, TradeRecord
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import SignalAction
from poseidon.strategies.dsl.v2.evaluator import DEFAULT_BAR_LABEL

MARKET_TZ = "Asia/Taipei"
TRADING_DAY_SHIFT = pd.Timedelta(hours=9)
DAY_SESSION_HOURS = (pd.Timedelta(hours=8), pd.Timedelta(hours=15))  # local open labels of day-session bars
NEXT_WEEKDAY = "next_weekday"
NEXT_DAY_SESSION = "next_day_session"
NIGHT_SESSION_DAY = {
    NEXT_WEEKDAY: "night session -> next trading day; Asia/Taipei + 9h, weekend rolled to Monday",
    NEXT_DAY_SESSION: "night session -> the next day with a day session in the bars, the day TAIFEX books it to",
}
EXIT_REASONS = {"close_sl": "stop_loss", "close_tp": "take_profit"}
SHARPE_CONVENTION = (
    "daily mark-to-market PnL per trading day ({night}), all trading days with bars incl. flat days; "
    "sharpe_daily = mean/std(ddof=1), sharpe_annualised = sharpe_daily*sqrt(252); skew/kurtosis are population "
    "moments, kurtosis non-excess"
)


V1_ENGINE = "v1"
TARGET_ENGINE = "target_v1"
CLOSE_ACTIONS = {"stop": "close_sl", "take_profit": "close_tp"}  # other closing fills: "close"


class BacktestFailed(RuntimeError):
    """The runner reported ``status != completed`` for this strategy."""


class _Interrupted(Exception):
    """Raised inside the runner loop once the stage was told to stop."""


def finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def stressed_cost_model(name: str, multiplier: float) -> CostModel:
    """``name``'s cost model with commission, tax and slippage all multiplied by ``multiplier``."""

    base = get_cost_model(name)
    return dataclasses.replace(
        base,
        market=f"{base.market}_x{multiplier:g}",
        buy_commission_rate=base.buy_commission_rate * multiplier,
        sell_commission_rate=base.sell_commission_rate * multiplier,
        commission_per_contract=base.commission_per_contract * multiplier,
        tax_rate=base.tax_rate * multiplier,
        slippage_pct=base.slippage_pct * multiplier,
        slippage_ticks=base.slippage_ticks * multiplier,
        description=f"{base.description} -- commission, tax and slippage x{multiplier:g}",
    )


def build_runner(
    strategy, backtest_config: dict, *, cost_model: CostModel | None = None, stop_slippage_ticks: float = 0.0
) -> BacktestRunner:
    sizing = backtest_config["sizing"]
    if sizing["mode"] != SizingMode.FIXED_CONTRACTS.value:
        raise ValueError(f"pipeline backtests size by fixed_contracts, not {sizing['mode']!r}")
    return BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(),
        cost_model=cost_model if cost_model is not None else get_cost_model(backtest_config["cost_model"]),
        initial_capital=float(backtest_config["initial_capital_twd"]),
        sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=int(sizing["contracts"])),
        gap_aware_stops=bool(backtest_config["gap_aware_stops"]),
        stop_slippage_ticks=float(stop_slippage_ticks),
    )


def night_session_rule(backtest_config: Mapping) -> str:
    """The run's frozen ``backtest.night_session``; :data:`NEXT_WEEKDAY` for a run frozen without it."""

    rule = backtest_config.get("night_session", NEXT_WEEKDAY)
    if rule not in NIGHT_SESSION_DAY:
        raise ValueError(f"unknown backtest.night_session {rule!r}")
    return rule


def trading_days(index: pd.DatetimeIndex, night_session: str = NEXT_WEEKDAY) -> pd.DatetimeIndex:
    """Trading-day label (midnight, naive) of each UTC bar timestamp.

    A day-session bar belongs to its own date. A night-session bar (15:00-05:00 Asia/Taipei) belongs to the
    next trading day: ``next_weekday`` takes the next weekday, so the night before a weekday closure is a
    night-only day of its own; ``next_day_session`` takes the next day with a day session in ``index``, as
    TAIFEX does (Thu 2026-10-22's night -> Tue 10-27 after the 10-23 typhoon closure and the 10-26 holiday).
    A night after the last day session in ``index`` keeps the next weekday.
    """

    local = index.tz_convert(MARKET_TZ).tz_localize(None)
    days = (local + TRADING_DAY_SHIFT).normalize()
    weekday = days.weekday
    roll = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
    labels = days + pd.to_timedelta(roll, unit="D")
    if night_session == NEXT_WEEKDAY:
        return labels
    sessions = day_sessions(index)
    if not len(sessions):
        return labels
    position = sessions.searchsorted(days)
    inside = position < len(sessions)
    return pd.DatetimeIndex(np.where(inside, sessions.values[np.minimum(position, len(sessions) - 1)], labels.values))


def day_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates (midnight, naive) with a day-session bar in ``index`` (UTC bar timestamps)."""

    local = index.tz_convert(MARKET_TZ).tz_localize(None)
    clock = local - local.normalize()
    return pd.DatetimeIndex(
        np.unique(local.normalize()[(clock >= DAY_SESSION_HOURS[0]) & (clock < DAY_SESSION_HOURS[1])])
    )


def daily_pnl(equity_curve, initial_capital: float, night_session: str = NEXT_WEEKDAY) -> pd.DataFrame:
    """``trading_day, pnl_twd, equity_twd`` from the runner's per-bar equity curve."""

    if not equity_curve:
        return pd.DataFrame(columns=["pnl_twd", "equity_twd"])
    times = pd.DatetimeIndex([t for t, _, _ in equity_curve])
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    equity = pd.Series([float(e) for _, e, _ in equity_curve], index=times)
    end_of_day = equity.groupby(trading_days(times, night_session)).last()
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


def _instrument(strategy, should_stop: Callable[[], bool], trade_from=None) -> tuple[dict, dict]:
    """Record exit reasons / entry rule ids from emitted signals; stop on request.

    With ``trade_from`` every entry signal stamped before it is dropped, so the
    bars before it only warm features up (no position, no PnL there).
    """

    exits: dict[pd.Timestamp, str] = {}
    entries: dict[pd.Timestamp, str] = {}
    evaluate = strategy.evaluate
    cutoff = pd.Timestamp(trade_from) if trade_from is not None else None
    entry_actions = (SignalAction.LONG, SignalAction.SHORT)

    def instrumented(features):
        if should_stop():
            raise _Interrupted("stage stop requested")
        signals = evaluate(features)
        if cutoff is not None:
            signals = [s for s in signals if s.action not in entry_actions or pd.Timestamp(s.signal_time) >= cutoff]
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


def dsl_engine(backtest_config: dict) -> str:
    engine = backtest_config.get("engine", V1_ENGINE)
    if engine not in (V1_ENGINE, TARGET_ENGINE):
        raise ValueError(f"unknown backtest engine {engine!r}")
    return engine


def run_backtest(
    strategy,
    bars: pd.DataFrame,
    backtest_config: dict,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    cost_model: CostModel | None = None,
    stop_slippage_ticks: float = 0.0,
    trade_from=None,
    expiries: Mapping[str, Any] | None = None,
) -> BacktestOutcome:
    """Run ``strategy`` on ``bars``; raises ``StageStopped`` when asked to stop mid-run.

    ``cost_model`` / ``stop_slippage_ticks`` override the frozen costs (the
    cost_stress gate only). ``trade_from``: bars before it are feature warm-up
    only -- entries are refused there and the daily series starts at its
    trading day. ``expiries`` (contract month -> last trading day) places the
    synthetic roll of the ``target_v1`` engine; by default it is read from the
    refdata settlement calendar.
    """

    exits, entries = _instrument(strategy, should_stop, trade_from)
    if dsl_engine(backtest_config) == TARGET_ENGINE:
        cost_model = cost_model if cost_model is not None else get_cost_model(backtest_config["cost_model"])
        records, equity_curve = run_target_engine(
            strategy,
            bars,
            backtest_config,
            cost_model=cost_model,
            stop_slippage_ticks=stop_slippage_ticks,
            should_stop=should_stop,
            expiries=expiries,
        )
        return summarise(
            records,
            equity_curve,
            backtest_config,
            cost_model=cost_model,
            exits=exits,
            entries=entries,
            bars=len(bars),
            stop_slippage_ticks=stop_slippage_ticks,
            trade_from=trade_from,
        )
    runner = build_runner(strategy, backtest_config, cost_model=cost_model, stop_slippage_ticks=stop_slippage_ticks)
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
        stop_slippage_ticks=stop_slippage_ticks,
        trade_from=trade_from,
    )


def settlement_expiries(product: str) -> dict[str, pd.Timestamp]:
    """Contract month -> last trading day of ``product`` from the refdata settlement calendar."""

    from poseidon.pipeline.refdata.datasets import CALENDAR
    from poseidon.pipeline.refdata.store import RefdataStore

    rows = RefdataStore().read(CALENDAR)
    rows = rows[rows["product"] == product]
    if rows.empty:
        raise BacktestFailed(f"no {product} rows in the refdata settlement calendar")
    return dict(zip(rows["contract_month"].astype(str), pd.to_datetime(rows["last_trading_day"]), strict=True))


def target_contract_spec(symbol: str, cost_model: CostModel):
    """The engine's ``ContractSpec`` for a frozen futures cost model (fees as ``BacktestPortfolio``)."""

    from poseidon.pipeline.engine import ContractSpec

    unsupported = {
        "buy_commission_rate": cost_model.buy_commission_rate,
        "sell_commission_rate": cost_model.sell_commission_rate,
        "slippage_pct": cost_model.slippage_pct,
    }
    if any(unsupported.values()) or not cost_model.tax_on_entry or not cost_model.tick_size:
        raise BacktestFailed(
            f"cost model {cost_model.market} is not a per-contract futures model (tax on both legs, tick slippage)"
        )
    return ContractSpec(
        symbol=symbol,
        point_value=float(cost_model.point_value),
        tick_size=float(cost_model.tick_size),
        commission_per_unit=float(cost_model.commission_per_contract),
        tax_rate=float(cost_model.tax_rate),
        slippage_ticks=float(cost_model.slippage_ticks),
    )


def run_target_engine(
    strategy,
    bars: pd.DataFrame,
    backtest_config: dict,
    *,
    cost_model: CostModel,
    stop_slippage_ticks: float = 0.0,
    should_stop: Callable[[], bool] = lambda: False,
    expiries: Mapping[str, Any] | None = None,
) -> tuple[list[TradeRecord], list[tuple]]:
    """``RuleStrategyV2`` as its spec v3 fast path on ``engine.intraday``.

    Returns what :func:`summarise` reads from ``BacktestPortfolio``: one entry ``TradeRecord`` per
    round trip plus a closing one when it closed (``close_sl`` / ``close_tp`` / ``close``), and the
    per-bar ``(time, equity, 0.0)`` curve.
    """

    from poseidon.pipeline.engine.intraday import IntradayConfig, IntradayInput, r1_contracts, run_intraday
    from poseidon.pipeline.v3.compiler import plan
    from poseidon.pipeline.v3.dsl_bridge import v2_to_v3
    from poseidon.pipeline.v3.runtime import Conventions, SpecStrategy

    sizing = backtest_config["sizing"]
    if sizing["mode"] != SizingMode.FIXED_CONTRACTS.value:
        raise ValueError(f"pipeline backtests size by fixed_contracts, not {sizing['mode']!r}")
    spec = v2_to_v3(strategy.config.model_dump(mode="json", exclude_none=True))
    spec["sleeves"][0]["sizing"]["params"]["n"] = int(sizing["contracts"])
    spec_plan = plan(spec)
    (sleeve,) = spec_plan.sleeves
    target = SpecStrategy(spec_plan.sleeves, spec_plan.allocation, Conventions(), record_trace=False)
    target.state(sleeve.id, "direction").bind(strategy, bars)
    data = IntradayInput(
        spec=target_contract_spec(strategy.symbol, cost_model),
        bars=bars,
        contracts=r1_contracts(
            bars.index,
            expiries if expiries is not None else settlement_expiries(strategy.symbol),
            night_session_rule(backtest_config),
        ),
    )
    config = IntradayConfig(
        initial_capital=float(backtest_config["initial_capital_twd"]),
        gap_aware_stops=bool(backtest_config["gap_aware_stops"]),
        stop_slippage_ticks=float(stop_slippage_ticks),
    )
    try:
        result = run_intraday(target, sleeve.alias, data, config)
    except Exception as error:
        if should_stop():
            from poseidon.pipeline.stages import StageStopped

            raise StageStopped("stage stopped during a backtest") from error
        raise BacktestFailed(f"{type(error).__name__}: {error}") from error
    records = []
    for trip in result.round_trips().itertuples(index=False):
        entry_time = trip.entry_time.to_pydatetime()
        records.append(
            TradeRecord(strategy.symbol, trip.side, entry_time, trip.entry_price, float(trip.units), trip.entry_fees)
        )
        if pd.notna(trip.exit_time):
            records.append(
                TradeRecord(
                    symbol=strategy.symbol,
                    action=CLOSE_ACTIONS.get(trip.exit_reason, "close"),
                    entry_time=entry_time,
                    exit_time=trip.exit_time.to_pydatetime(),
                    entry_price=trip.entry_price,
                    exit_price=trip.exit_price,
                    quantity=float(trip.units),
                    fees=trip.fees,
                    pnl=trip.pnl,
                )
            )
    curve = [(time, float(value), 0.0) for time, value in result.equity.items()]
    return records, curve


def summarise(
    trade_records,
    equity_curve,
    backtest_config,
    *,
    cost_model,
    exits,
    entries,
    bars,
    stop_slippage_ticks: float = 0.0,
    trade_from=None,
) -> BacktestOutcome:
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
                # Ticks of slippage inside the fills: every market-order leg,
                # plus the stop leg when a stressed run slips stops too.
                "_slipped_ticks": float(cost_model.slippage_ticks) * (1 + (record.action == "close"))
                + float(stop_slippage_ticks) * (record.action == "close_sl"),
            }
        )
    open_at_end = len(sides) - len({pd.Timestamp(t["entry_time"]) for t in trips})

    pnls = np.array([t["pnl_twd"] for t in trips], dtype=float)
    fees_total = float(sum(t["fees_twd"] for t in trips))
    quantity_legs = float(sum(2 * t["quantity"] for t in trips))
    commission = quantity_legs * float(getattr(cost_model, "commission_per_contract", 0.0))
    tick = float(cost_model.tick_size or 0.0) * float(cost_model.point_value)
    slippage = float(sum(t["_slipped_ticks"] * t["quantity"] for t in trips)) * tick
    for trip in trips:
        trip.pop("_slipped_ticks")

    if len(pnls):
        cumulative = np.cumsum(pnls)
        peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
        max_dd = float(np.max(peaks - cumulative))
        max_dd = max(max_dd, 0.0)
    else:
        max_dd = 0.0

    night_session = night_session_rule(backtest_config)
    daily = daily_pnl(equity_curve, capital, night_session)
    if trade_from is not None and len(daily):
        stamp = pd.Timestamp(trade_from)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        daily = daily[daily.index >= trading_days(pd.DatetimeIndex([stamp]))[0]]
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
            "note": "fees_twd = commission + tax; slippage is inside fill prices "
            f"({float(cost_model.slippage_ticks):g} tick(s) per market-order leg, "
            + (
                f"{float(stop_slippage_ticks):g} on stop fills, none on target fills)"
                if stop_slippage_ticks
                else "none on stop/target fills)"
            ),
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
        "sharpe_convention": SHARPE_CONVENTION.format(night=NIGHT_SESSION_DAY[night_session]),
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
