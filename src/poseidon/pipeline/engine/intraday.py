"""Target-position simulator on intraday bars (design 2026-10-24 §3.4, §5.1, §5.5; step 7).

Input is one continuous front-month series (Thalassa TX 5m R1, open-labelled UTC bars) plus the
contract each bar belongs to (:func:`r1_contracts`, from the settlement calendar). The strategy
decides at every bar close and its orders fill at that close (``execution.signal_close``, the v1
runner's convention). Each bar, in this order:

1. A synthetic roll left over from the previous bar reopens at this bar's open (new month).
2. Protective orders (:class:`~poseidon.pipeline.engine.view.Protect`) on the position: the v1
   runner's Phase A. Long: stop when low <= stop, take-profit when high >= target; short mirrored;
   the stop wins a bar that touches both. A gap-aware stop fills at the open when the bar opens
   through it, else at the stop; a take-profit fills at its price. No slippage on either, except
   ``stop_slippage_ticks`` against stops (the cost-stress gate).
3. Risk check at the bar's worst price for the position: equity / initial margin below
   ``liquidation_risk_ratio`` closes it there (``forced_liquidation``).
4. Mark to the close; equity <= 0 is ruin and ends the run.
5. Decision: the strategy sees bars up to this one and the book, and returns integer targets and
   protective orders. Orders fill now at the close plus slippage. There is no leverage or margin
   clipping: intraday specs are fixed-contract (the v1 behaviour, PnL judged in TWD).
6. Synthetic roll (§5.5): when the next bar belongs to another contract and a position is open,
   it closes at this close (old month) and reopens at the next bar's open (step 1). Both legs
   pay fees and slippage; the gap between the months is never PnL. Protective levels are R1
   prices and are not shifted by the gap (the DSL computes its levels on R1 too).

Every fill settles against the current mark (cash += units x (mark - fill) x point value), so the
position always carries the mark price, as in the daily simulator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from poseidon.pipeline.engine.records import EVENT_COLUMNS, FILL_COLUMNS, _frame
from poseidon.pipeline.engine.simulator import Decision, TargetStrategy, _check_targets
from poseidon.pipeline.engine.specs import ContractSpec
from poseidon.pipeline.engine.view import BookView, Fill, MarketView, Protect, SeriesBuffer, SeriesView

MARKET_TZ = "Asia/Taipei"
# On a TX/MTX last trading day the expiring month trades until 13:30 Asia/Taipei: R1 bars that open
# before then (the preceding night session and the day session up to the 13:25 bar) are the old
# month, bars from 13:30 the new one (checked on all 80 settlements 2020-03..2026-10 against the
# TAIFEX per-contract day and night rows).
SETTLEMENT_SWITCH = time(13, 30)
ROUND_TRIP_COLUMNS = (
    "side",
    "units",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "entry_fees",
    "fees",
    "pnl",
    "exit_reason",
    "legs",
)


def r1_contracts(index: pd.DatetimeIndex, expiries: Mapping[str, Any]) -> np.ndarray:
    """Contract month of every R1 bar: the nearest expiry not yet passed at the bar's open.

    ``expiries`` maps contract month -> last trading day (``taifex_settlement_calendar``). A bar
    of trading day D (TAIFEX: the night session belongs to the next trading day) belongs to the
    first contract whose last trading day is after D, or is D itself when the bar opens before
    :data:`SETTLEMENT_SWITCH`.
    """

    from poseidon.pipeline.backtesting import trading_days

    if not len(index):
        return np.array([], dtype=object)
    names = sorted(expiries, key=lambda c: (pd.Timestamp(expiries[c]), c))
    last_days = pd.DatetimeIndex([pd.Timestamp(expiries[c]).normalize() for c in names])
    days = trading_days(index)
    local = index.tz_convert(MARKET_TZ).tz_localize(None)
    switch = pd.Timedelta(hours=SETTLEMENT_SWITCH.hour, minutes=SETTLEMENT_SWITCH.minute)
    # Night bars of trading day D open before D's midnight, so "opens before the switch on D" is
    # local time < D + 13:30 for every bar of D.
    after_switch = local >= days + switch
    position = last_days.searchsorted(days, side="left")
    in_bounds = position < len(last_days)
    on_expiry = np.zeros(len(index), dtype=bool)
    on_expiry[in_bounds] = last_days[position[in_bounds]] == days[in_bounds]
    position = position + (on_expiry & after_switch)
    if (position >= len(last_days)).any():
        first = index[position >= len(last_days)][0]
        raise ValueError(f"settlement calendar ends before the bar at {first.isoformat()}")
    return np.asarray(names, dtype=object)[position]


@dataclass(frozen=True)
class IntradayInput:
    """``bars``: sorted unique UTC DatetimeIndex, columns open/high/low/close (others ignored);
    ``contracts``: the contract month of each bar (:func:`r1_contracts`)."""

    spec: ContractSpec
    bars: pd.DataFrame
    contracts: Any  # array-like of str, one per bar


@dataclass(frozen=True)
class IntradayConfig:
    initial_capital: float = 1_000_000.0
    gap_aware_stops: bool = True
    stop_slippage_ticks: float = 0.0
    liquidation_risk_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.stop_slippage_ticks < 0:
            raise ValueError("stop_slippage_ticks must be >= 0")


@dataclass(frozen=True)
class IntradayResult:
    """``equity`` is the marked equity after every bar's close (fills and rolls included)."""

    config: IntradayConfig
    point_value: float
    fills: pd.DataFrame
    equity: pd.Series
    events: pd.DataFrame
    ruined: pd.Timestamp | None

    def round_trips(self) -> pd.DataFrame:
        """Flat-to-flat segments of the fills (§5.8), one row each; roll legs stay inside (the book is
        flat between a roll's two legs, but the position goes on).

        ``gross_pnl`` = -sum(units x price) x point value over the segment's legs, so a segment
        that rolled excludes the gap between the months. The last row has no exit when a position
        is still open at the end.
        """

        rows, legs, units = [], [], 0
        for fill in self.fills.itertuples(index=False):
            legs.append(fill)
            units += fill.units
            if units or fill.reason == "roll":
                continue
            rows.append(self._segment(legs, closed=True))
            legs = []
        if legs:
            rows.append(self._segment(legs, closed=False))
        return pd.DataFrame(rows, columns=list(ROUND_TRIP_COLUMNS))

    def _segment(self, legs: list, *, closed: bool) -> dict:
        first, last = legs[0], legs[-1]
        fees = float(sum(leg.tax + leg.commission for leg in legs))
        if closed and len(legs) == 2:
            gross = (last.price - first.price) * first.units * self.point_value
        else:
            gross = -sum(leg.units * leg.price for leg in legs) * self.point_value
        return {
            "side": "long" if first.units > 0 else "short",
            "units": abs(first.units),
            "entry_time": first.date,
            "exit_time": last.date if closed else None,
            "entry_price": first.price,
            "exit_price": last.price if closed else None,
            "gross_pnl": gross if closed else None,
            "entry_fees": first.tax + first.commission,
            "fees": fees,
            "pnl": gross - fees if closed else None,
            "exit_reason": last.reason if closed else None,
            "legs": len(legs),
        }


class _Run:
    def __init__(self, strategy: TargetStrategy, name: str, data: IntradayInput, config: IntradayConfig) -> None:
        bars = data.bars
        index = pd.DatetimeIndex(bars.index)
        if index.tz is None or not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("intraday bars need a sorted, unique, tz-aware index")
        contracts = np.asarray(data.contracts, dtype=object)
        if len(contracts) != len(bars):
            raise ValueError("one contract per bar is required")
        self.strategy, self.name, self.spec, self.config = strategy, name, data.spec, config
        self.index = index
        self.contracts = contracts
        self.o, self.h, self.l, self.c = (bars[col].to_numpy(float) for col in ("open", "high", "low", "close"))
        if not np.isfinite(np.stack([self.o, self.h, self.l, self.c])).all():
            raise ValueError("intraday bars need finite open/high/low/close")
        self.series = SeriesBuffer(len(bars))
        self.adjusted = math.nan
        self.cash = float(config.initial_capital)
        self.units = 0
        self.mark = math.nan
        self.protect: Protect | None = None
        self.reopen = 0
        self.fills: list[Fill] = []
        self.since: list[Fill] = []
        self.events: list[dict] = []

    def trade(self, i: int, units: int, ref: float, session: str, reason: str, *, slip: bool = True) -> None:
        if not units:
            return
        spec = self.spec
        fill = spec.fill_price(units, ref) if slip else ref
        tax, commission = spec.leg_fees(units, fill)
        self.cash += units * (self.mark - fill) * spec.point_value - tax - commission
        self.units += units
        record = Fill(
            date=self.index[i],
            session=session,
            instrument=self.name,
            contract=str(self.contracts[i]),
            units=units,
            price=fill,
            ref_price=ref,
            tax=tax,
            commission=commission,
            slippage_twd=abs(units) * abs(fill - ref) * spec.point_value,
            reason=reason,
        )
        self.fills.append(record)
        self.since.append(record)

    def event(self, i: int, kind: str, value: float = math.nan, threshold: float = math.nan, note: str = "") -> None:
        self.events.append(
            {
                "date": self.index[i],
                "kind": kind,
                "instrument": self.name,
                "value": value,
                "threshold": threshold,
                "note": note,
            }
        )

    def feed(self, i: int) -> None:
        if i == 0:
            self.adjusted = self.c[0]
        elif self.contracts[i] == self.contracts[i - 1]:
            self.adjusted *= self.c[i] / self.c[i - 1]
        else:
            self.adjusted *= self.c[i] / self.o[i]  # the gap between the months is not a return
        self.series.append_bar(
            self.index[i],
            str(self.contracts[i]),
            {
                "open": self.o[i],
                "high": self.h[i],
                "low": self.l[i],
                "close": self.c[i],
                "adjusted_close": self.adjusted,
            },
        )

    def protective(self, i: int) -> None:
        if not self.units or self.protect is None:
            return
        stop, target = self.protect.stop, self.protect.take_profit
        long = self.units > 0
        hit_stop = stop is not None and (self.l[i] <= stop if long else self.h[i] >= stop)
        hit_target = target is not None and (self.h[i] >= target if long else self.l[i] <= target)
        if hit_stop:
            price = stop
            if self.config.gap_aware_stops:
                price = min(self.o[i], stop) if long else max(self.o[i], stop)
            price -= math.copysign(self.config.stop_slippage_ticks * self.spec.tick_size, self.units)
            self.trade(i, -self.units, price, "intraday", "stop", slip=False)
        elif hit_target:
            self.trade(i, -self.units, target, "intraday", "take_profit", slip=False)

    def risk_check(self, i: int) -> None:
        if not self.units:
            return
        worst = self.l[i] if self.units > 0 else self.h[i]
        equity = self.cash + self.units * (worst - self.mark) * self.spec.point_value
        initial_margin = self.spec.notional(self.units, worst) * self.spec.initial_margin_ratio
        ratio = equity / initial_margin
        if ratio >= self.config.liquidation_risk_ratio:
            return
        self.event(
            i,
            "forced_liquidation",
            ratio,
            self.config.liquidation_risk_ratio,
            f"equity {equity:,.0f} vs initial margin {initial_margin:,.0f}",
        )
        self.trade(i, -self.units, worst, "intraday", "forced_liquidation")

    def decide(self, i: int) -> Decision:
        book = BookView(
            date=self.index[i],
            initial_capital=self.config.initial_capital,
            cash=self.cash,
            equity=self.cash,
            positions={self.name: self.units},
            contracts={self.name: str(self.contracts[i])},
            margin_required=self.spec.notional(self.units, self.c[i]) * self.spec.initial_margin_ratio,
            maintenance_required=self.spec.notional(self.units, self.c[i]) * self.spec.maintenance_margin_ratio,
            fills_today=tuple(self.since),
            protective={self.name: self.protect} if self.units and self.protect is not None else {},
        )
        self.since = []
        view = MarketView(self.index[i], {self.name: SeriesView(self.series, self.series.n)}, {})
        decision = self.strategy.on_bar(view, book)
        _check_targets(decision, (self.name,))
        unknown = set(decision.protective) - {self.name}
        if unknown:
            raise ValueError(f"protective orders for unknown instruments {sorted(unknown)}")
        return decision

    def execute(self, i: int, decision: Decision) -> None:
        before = self.units
        target = int(decision.targets.get(self.name, before))
        self.trade(i, target - before, self.c[i], "close", "rebalance")
        if not self.units:
            self.protect = None
        elif self.name in decision.protective:
            self.protect = decision.protective[self.name]
        elif before == 0 or (before > 0) != (self.units > 0):
            self.protect = None  # a new position never inherits the previous one's orders

    def roll(self, i: int) -> None:
        if not self.units or i + 1 >= len(self.index) or self.contracts[i + 1] == self.contracts[i]:
            return
        units = self.units
        self.trade(i, -units, self.c[i], "close", "roll")
        self.reopen = units
        self.event(i, "roll", note=f"{self.contracts[i]} -> {self.contracts[i + 1]} ({units} units)")

    def run(self) -> IntradayResult:
        n = len(self.index)
        equity = np.full(n, np.nan)
        ruined = None
        last = n - 1
        for i in range(n):
            self.feed(i)
            if self.reopen:
                self.mark = self.o[i]  # the new month's position starts marked at its first price
                units, self.reopen = self.reopen, 0
                self.trade(i, units, self.o[i], "open", "roll")
            self.protective(i)
            self.risk_check(i)
            if self.units:
                self.cash += self.units * (self.c[i] - self.mark) * self.spec.point_value
            self.mark = self.c[i]
            if self.cash <= 0:
                ruined, last = self.index[i], i
                self.event(i, "ruin", self.cash, 0.0)
                equity[i] = self.cash
                break
            self.execute(i, self.decide(i))
            self.roll(i)
            equity[i] = self.cash
        return IntradayResult(
            config=self.config,
            point_value=self.spec.point_value,
            fills=_frame([vars(f) for f in self.fills], FILL_COLUMNS),
            equity=pd.Series(equity[: last + 1], index=self.index[: last + 1], name="equity"),
            events=_frame(self.events, EVENT_COLUMNS),
            ruined=ruined,
        )


def run_intraday(
    strategy: TargetStrategy, name: str, data: IntradayInput, config: IntradayConfig | None = None
) -> IntradayResult:
    """Simulate ``strategy`` trading instrument ``name`` on ``data``'s bars (signal-close fills)."""

    return _Run(strategy, name, data, config or IntradayConfig()).run()
