"""Target-position simulator, daily futures core (design §5.1, §5.3-5.6, §5.8).

Each trading day t, in this order:

1. ``next_open`` only: fill yesterday's orders at today's open of the held contract, after a gap
   check of the overnight position at the open.
2. Intraday risk check on the position held through the day, at the bar's worst price for it
   (low for longs, high for shorts): equity / initial-margin requirement < ``liquidation_risk_ratio``
   -> close everything there (``forced_liquidation``).
3. Close: daily settlement (every contract marked to its own close, PnL into cash). Equity <= 0
   -> ``ruin`` and the run stops. Equity < maintenance margin -> ``margin_call``; the next
   execution may not exceed the initial-margin level, whatever the strategy asks.
4. ``next_open`` only: a roll that is due is executed now at both contracts' closes.
5. From ``config.start`` on: the strategy gets the market view (data up to t) and the book and
   returns signed integer targets. The engine applies the margin-call cap, the gross leverage cap
   and the margin-utilisation cap, then the ``min_change_units`` band.
6. Execution: ``signal_close`` fills now at the close (a due roll is netted into the same
   execution: close the old contract, open the target on the new one); ``next_open`` queues the
   orders for tomorrow's open.

Every fill settles against the current mark immediately (cash += units x (mark - fill) x point
value), so the position always carries the mark price and the close settles the whole position.

What each step does to an instrument follows its ``spec.policy`` (``specs.KIND_POLICIES``), never
its symbol. A kind without daily settlement (stocks) keeps mark = 0, so the fill pays the whole
notional from cash and equity carries units x price; it has no margin, so the margin steps skip
it and its buys are trimmed so cash never drops below the margined positions' initial margin
(``clip_cash`` at the decision, ``trim_cash`` at the fill, sells fill first). Corporate actions
on the bars (``data.DAY_COLUMNS``) are booked at the start of the day, before any fill. A
``traded = False`` bar fills nothing: its ``next_open`` orders stay queued, and the day's decision
replaces the queue as on any other day (a re-sizing strategy asks again; ``signal_close`` fills nothing).

Hedges (``engine.options``, bought options only): after the futures settle, a held series that
reaches its final settlement day is cash-settled, the hedge orders waiting for today fill at
today's settlement prices, and every held series is marked at its settlement. Premiums move cash;
equity = cash + the marked option value. Each decision sizes and selects the hedge from its
underlying's final target units; ``next_open`` fills it at the next day's settlement,
``signal_close`` at today's.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

from poseidon.pipeline.engine.data import (
    ContractTable,
    FuturesInput,
    build_contract_table,
    normalise_calendar,
    validate_aux,
)
from poseidon.pipeline.engine.options import (
    HEDGE_UNAVAILABLE,
    HedgeBook,
    HedgePolicy,
    OptionChain,
    OptionInput,
    OptionSeries,
)
from poseidon.pipeline.engine.records import EngineResult
from poseidon.pipeline.engine.specs import ROUNDING_EPS, ContractSpec
from poseidon.pipeline.engine.view import (
    AuxView,
    BookView,
    Fill,
    MarketView,
    Protect,
    SeriesBuffer,
    SeriesView,
    _Buffer,
)

Execution = Literal["next_open", "signal_close"]


@dataclass(frozen=True)
class Decision:
    """Signed integer units per instrument. An instrument left out keeps its current position.

    ``target_leverage`` is optional and only reported (positions_daily: target vs actual leverage
    and the rounding error). ``protective`` replaces the resting stop / take-profit of an
    instrument's position after this decision (left out = keep); only the intraday engine
    (:mod:`poseidon.pipeline.engine.intraday`) executes them, this daily engine refuses them.
    """

    targets: Mapping[str, int] = field(default_factory=dict)
    target_leverage: Mapping[str, float] = field(default_factory=dict)
    protective: Mapping[str, Protect] = field(default_factory=dict)


class TargetStrategy(Protocol):
    def on_bar(self, view: MarketView, book: BookView) -> Decision: ...


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float = 1_000_000.0
    execution: Execution = "next_open"
    leverage_cap: float = 5.0  # gross notional / equity at the decision
    max_margin_utilisation: float = 1.0  # initial margin / equity at the decision
    liquidation_risk_ratio: float = 0.25  # equity / initial margin below this -> forced liquidation
    min_change_units: int = 1
    start: Any = None  # first decision day; earlier bars only warm up the view
    end: Any = None  # last simulated day (inclusive)

    def __post_init__(self) -> None:
        if self.execution not in ("next_open", "signal_close"):
            raise ValueError(f"unknown execution {self.execution!r}")
        if self.initial_capital <= 0 or self.leverage_cap <= 0 or self.max_margin_utilisation <= 0:
            raise ValueError("initial_capital, leverage_cap and max_margin_utilisation must be positive")
        if self.min_change_units < 1:
            raise ValueError("min_change_units must be >= 1")


@dataclass
class _Position:
    spec: ContractSpec
    table: ContractTable
    units: int = 0
    contract: int | None = None  # column in table
    mark: float = math.nan  # last settlement / mark price of the held contract

    def contract_name(self, col: int | None = None) -> str | None:
        col = self.contract if col is None else col
        return None if col is None else self.table.contracts[col]


def _check_targets(decision: Decision, instruments: Sequence[str]) -> dict[str, int]:
    if not isinstance(decision, Decision):
        raise TypeError(f"strategy must return Decision, got {type(decision).__name__}")
    out = {}
    for inst, units in decision.targets.items():
        if inst not in instruments:
            raise ValueError(f"target for unknown instrument {inst!r}")
        if isinstance(units, bool | float) or not hasattr(units, "__index__"):
            raise TypeError(f"target for {inst} must be an integer number of units, got {units!r}")
        out[inst] = int(units)
    return out


def _check_short(targets: Mapping[str, int], positions: Mapping[str, _Position]) -> None:
    for inst, units in targets.items():
        if units < 0 and not positions[inst].spec.policy.short:
            raise ValueError(f"target for {inst}: a {positions[inst].spec.kind} cannot be sold short ({units})")


def _mark_after_roll(spec: ContractSpec, close: float) -> float:
    """Mark of a freshly held contract: its close for kinds that settle daily, else 0 (paid in full)."""

    return close if spec.policy.daily_settlement else 0.0


def _scale_toward_zero(units: dict[str, int], scale: float) -> dict[str, int]:
    """Scale each position by ``scale`` and floor the magnitude (same float guard as sizing)."""

    return {
        k: math.trunc(v * scale + math.copysign(ROUNDING_EPS * max(1.0, abs(v * scale)), v)) for k, v in units.items()
    }


class _Run:
    def __init__(
        self,
        strategy: TargetStrategy,
        futures: Mapping[str, FuturesInput],
        config: EngineConfig,
        aux: Mapping[str, pd.DataFrame],
        calendar: Sequence[Any] | None,
        options: Mapping[str, OptionInput] | None = None,
        hedges: Sequence[HedgePolicy] = (),
    ) -> None:
        if not futures:
            raise ValueError("need at least one instrument")
        self.strategy = strategy
        self.config = config
        if calendar is None:
            calendar = sorted({d for f in futures.values() for d in pd.to_datetime(f.bars["date"]).dt.normalize()})
        cal = normalise_calendar(calendar)
        self.pos = {name: _Position(f.spec, build_contract_table(name, f, cal)) for name, f in futures.items()}
        dates = {tuple(p.table.dates) for p in self.pos.values()}
        if len(dates) != 1:
            raise ValueError("Phase 1 engine: all instruments must have bars on the same dates")
        all_dates = next(iter(self.pos.values())).table.dates
        end = pd.Timestamp(config.end) if config.end is not None else all_dates[-1]
        self.n_days = int(all_dates.searchsorted(end, side="right"))
        self.dates = all_dates[: self.n_days]
        if not self.n_days:
            raise ValueError("no bars on or before config.end")
        self.start = pd.Timestamp(config.start) if config.start is not None else self.dates[0]
        self.instruments = tuple(self.pos)
        self.chains = {name: OptionChain(name, option, cal) for name, option in (options or {}).items()}
        self.hedges = {policy.name: HedgeBook(policy) for policy in hedges}
        for policy in hedges:
            if policy.underlying not in self.pos or policy.chain not in self.chains:
                raise ValueError(f"hedge {policy.name}: unknown underlying {policy.underlying} or chain {policy.chain}")
        if len(self.hedges) != len(hedges) or set(self.hedges) & set(self.pos):
            raise ValueError("hedge names must be unique and differ from instrument names")

        self.series = {name: SeriesBuffer(self.n_days) for name in self.pos}
        self.aux_frames = validate_aux(aux)
        self.aux_rows = {
            name: (frame.index[: frame.index.searchsorted(self.dates[-1], side="right")], frame.to_numpy(float))
            for name, frame in self.aux_frames.items()
        }
        self.aux_buffers = {
            name: _Buffer(len(self.aux_rows[name][0]), tuple(frame.columns)) for name, frame in self.aux_frames.items()
        }
        self.adjusted = dict.fromkeys(self.pos, math.nan)

        self.cash = float(config.initial_capital)
        self.pending: dict[str, int] | None = None
        self.margin_cap: dict[str, int] | None = None
        self.fills: list[Fill] = []
        self.fills_today: list[Fill] = []
        self.events: list[dict] = []
        self.positions_daily: list[dict] = []
        self.daily: list[dict] = []
        self.decisions: list[dict] = []
        self.ruined: pd.Timestamp | None = None

    # ------------------------------------------------------------------ accounting primitives

    def option_value(self) -> float:
        return sum(
            b.units * b.mark * self.chains[b.policy.chain].spec.point_value for b in self.hedges.values() if b.units
        )

    def equity(self) -> float:
        """After the close: every future is settled, so cash plus the marked options."""

        return self.cash + self.option_value()

    def equity_at(self, prices: Mapping[str, float]) -> float:
        return self.equity() + sum(
            p.units * (prices[k] - p.mark) * p.spec.point_value for k, p in self.pos.items() if p.units
        )

    def margin(self, prices: Mapping[str, float], units: Mapping[str, int] | None = None) -> tuple[float, float]:
        """(initial, maintenance) margin requirement for ``units`` (default: current) at ``prices``."""

        im = mm = 0.0
        for k, p in self.pos.items():
            u = p.units if units is None else units[k]
            if not u:
                continue
            notional = p.spec.notional(u, prices[k])
            im += notional * p.spec.initial_margin_ratio
            mm += notional * p.spec.maintenance_margin_ratio
        return im, mm

    def trade(self, t: pd.Timestamp, name: str, units: int, ref: float, session: str, reason: str) -> None:
        """Change the position by ``units`` at ``ref`` (before slippage); settles against the mark."""

        if not units:
            return
        p = self.pos[name]
        fill = p.spec.fill_price(units, ref)
        tax, commission = p.spec.leg_fees(units, fill)
        self.cash += units * (p.mark - fill) * p.spec.point_value - tax - commission
        p.units += units
        record = Fill(
            date=t,
            session=session,
            instrument=name,
            contract=p.contract_name(),
            units=units,
            price=fill,
            ref_price=ref,
            tax=tax,
            commission=commission,
            slippage_twd=abs(units) * abs(fill - ref) * p.spec.point_value,
            reason=reason,
        )
        self.fills.append(record)
        self.fills_today.append(record)

    def event(
        self,
        t: pd.Timestamp,
        kind: str,
        instrument: str = "",
        value: float = math.nan,
        threshold: float = math.nan,
        note: str = "",
    ) -> None:
        self.events.append(
            {"date": t, "kind": kind, "instrument": instrument, "value": value, "threshold": threshold, "note": note}
        )

    def held_price(self, field_name: str, i: int) -> dict[str, float]:
        return {k: p.table.price(field_name, i, p.contract) for k, p in self.pos.items() if p.contract is not None}

    def execute(self, t: pd.Timestamp, targets: Mapping[str, int], prices: Mapping[str, float], session: str) -> None:
        """Fill the orders toward ``targets`` at ``prices``. Buys of instruments paid in full go last
        and are trimmed so cash stays at or above the initial margin of the margined targets."""

        floor, _ = self.margin(prices, {k: targets.get(k, p.units) for k, p in self.pos.items()})
        paid_in_full_buy = {
            k: u > self.pos[k].units and not self.pos[k].spec.policy.margined for k, u in targets.items()
        }
        for k in sorted(targets, key=paid_in_full_buy.__getitem__):
            p = self.pos[k]
            units = targets[k] - p.units
            if paid_in_full_buy[k]:
                fill = p.spec.fill_price(units, prices[k])
                per_unit = fill * p.spec.point_value + sum(p.spec.leg_fees(1, fill))
                affordable = max(0, math.floor((self.cash - floor) / per_unit + ROUNDING_EPS))
                if affordable < units:
                    self.event(
                        t,
                        "trim_cash",
                        k,
                        value=self.cash - floor,
                        threshold=units * per_unit,
                        note=f"{session}: {units} -> {affordable} units at {fill:g}",
                    )
                    units = affordable
            reason = "margin_call" if self.margin_cap is not None and k in self.margin_cap else "rebalance"
            self.trade(t, k, units, prices[k], session, reason)

    def liquidate(self, t: pd.Timestamp, prices: Mapping[str, float], session: str) -> None:
        for k, p in self.pos.items():
            if p.units:
                self.trade(t, k, -p.units, prices[k], session, "forced_liquidation")

    def risk_breached(self, t: pd.Timestamp, prices: Mapping[str, float], session: str) -> bool:
        if not any(p.units for p in self.pos.values()):
            return False
        equity = self.equity_at(prices)
        im, _ = self.margin(prices)
        if im <= 0:
            return False
        ratio = equity / im
        if ratio >= self.config.liquidation_risk_ratio:
            return False
        self.event(
            t,
            "forced_liquidation",
            value=ratio,
            threshold=self.config.liquidation_risk_ratio,
            note=f"{session}: equity {equity:,.0f} vs initial margin {im:,.0f}",
        )
        return True

    # ------------------------------------------------------------------ the day

    def feed(self, i: int, t: pd.Timestamp) -> None:
        for k, p in self.pos.items():
            tab = p.table
            col = int(tab.held[i])
            close = tab.price("close", i, col)
            if i == 0:
                self.adjusted[k] = close
            else:
                prev = int(tab.held[i - 1])
                self.adjusted[k] *= tab.price("close", i, prev) / tab.price("close", i - 1, prev) / tab.factor[i]
            self.series[k].append_bar(
                t,
                tab.contracts[col],
                {
                    "open": tab.price("open", i, col),
                    "high": tab.price("high", i, col),
                    "low": tab.price("low", i, col),
                    "close": close,
                    "adjusted_close": self.adjusted[k],
                },
            )
        for name, (index, values) in self.aux_rows.items():
            buf = self.aux_buffers[name]
            while buf.n < len(index) and index[buf.n] <= t:
                buf.append(index[buf.n], dict(zip(buf.columns, values[buf.n], strict=True)))

    def corporate_actions(self, i: int, t: pd.Timestamp) -> None:
        """Book the actions effective today on the held units, before any fill: cash per share
        into cash, units x unit_ratio (the fraction left over is paid out at the reference
        price), and pending orders and margin caps rescaled the same way."""

        for k, p in self.pos.items():
            tab = p.table
            ratio, cash = float(tab.unit_ratio[i]), float(tab.cash_per_share[i])
            if ratio == 1.0 and cash == 0.0:
                continue
            col = int(tab.held[i])
            reference = (tab.price("close", max(i - 1, 0), col) - cash) / ratio
            before = p.units
            self.cash += before * cash * p.spec.point_value
            exact = before * ratio
            p.units = math.trunc(exact + math.copysign(ROUNDING_EPS * max(1.0, abs(exact)), exact))
            self.cash += (exact - p.units) * reference * p.spec.point_value
            for orders in (self.pending, self.margin_cap):
                if orders is not None and k in orders:
                    orders[k] = math.trunc(orders[k] * ratio)
            self.event(
                t,
                "corporate_action",
                k,
                value=ratio,
                threshold=cash,
                note=f"{before} -> {p.units} units, {cash:g} TWD/share, reference {reference:g}",
            )

    def fill_pending(self, i: int, t: pd.Timestamp) -> None:
        orders, self.pending = self.pending, None
        if not orders:
            return
        opens = self.held_price("open", i)
        if self.risk_breached(t, opens, "open gap"):
            self.liquidate(t, opens, "open")
            return
        self.pending = {k: u for k, u in orders.items() if not self.pos[k].table.traded[i]} or None
        self.execute(t, {k: u for k, u in orders.items() if self.pos[k].table.traded[i]}, opens, "open")

    def intraday_check(self, i: int, t: pd.Timestamp) -> None:
        worst = {}
        for k, p in self.pos.items():
            if not p.units:
                continue
            bar = [p.table.price(f, i, p.contract) for f in ("open", "high", "low", "close")]
            worst[k] = min(bar) if p.units > 0 else max(bar)
        if worst and self.risk_breached(t, {**self.held_price("close", i), **worst}, "intraday"):
            self.liquidate(t, worst, "intraday")

    def settle(self, i: int, t: pd.Timestamp) -> dict[str, float]:
        closes = self.held_price("close", i)
        for k, p in self.pos.items():
            if p.contract is not None and p.spec.policy.daily_settlement:
                self.cash += p.units * (closes[k] - p.mark) * p.spec.point_value
                p.mark = closes[k]
        return closes

    def margin_call_check(self, t: pd.Timestamp, closes: Mapping[str, float]) -> None:
        self.margin_cap = None
        if not any(p.units for p in self.pos.values()):
            return
        equity = self.equity_at(closes)
        im, mm = self.margin(closes)
        if equity >= mm:
            return
        cap = _scale_toward_zero({k: abs(p.units) for k, p in self.pos.items()}, equity / im)
        self.margin_cap = {k: cap[k] for k, p in self.pos.items() if p.units and p.spec.policy.margined}
        self.event(
            t,
            "margin_call",
            value=equity,
            threshold=mm,
            note="cap " + ", ".join(f"{k}<={v}" for k, v in self.margin_cap.items()),
        )

    def roll(self, i: int, t: pd.Timestamp, targets: Mapping[str, int] | None) -> None:
        """Move every instrument whose held contract changes after today's close to the new one.

        ``targets`` given (``signal_close``): the new leg opens at the target directly.
        """

        for k, p in self.pos.items():
            new = int(p.table.held[i])
            if p.contract == new:
                continue
            new_close = p.table.price("close", i, new)
            if not p.units:
                p.contract, p.mark = new, _mark_after_roll(p.spec, new_close)
                continue
            old_name = p.contract_name()
            units = p.units
            self.trade(t, k, -units, p.mark, "close", "roll")
            p.contract, p.mark = new, _mark_after_roll(p.spec, new_close)
            target = units if targets is None else targets[k]
            self.trade(t, k, target, new_close, "close", "roll")
            self.event(t, "roll", k, note=f"{old_name} -> {p.contract_name()} ({units} -> {target} units)")

    # ------------------------------------------------------------------ hedges (engine.options)

    def option_trade(
        self, t: pd.Timestamp, book: HedgeBook, series: OptionSeries, units: int, settle: float, reason: str
    ):
        spec = self.chains[book.policy.chain].spec
        fill = spec.fill_price(units, settle)
        tax, commission = spec.leg_fees(units, fill)
        self.cash -= units * fill * spec.point_value + tax + commission
        book.units += units
        book.series, book.mark = (series, settle) if book.units else (None, math.nan)
        record = Fill(
            date=t,
            session="close",
            instrument=book.policy.name,
            contract=series.key,
            units=units,
            price=fill,
            ref_price=settle,
            tax=tax,
            commission=commission,
            slippage_twd=abs(units) * abs(fill - settle) * spec.point_value,
            reason=reason,
        )
        self.fills.append(record)
        self.fills_today.append(record)

    def option_expiry(self, t: pd.Timestamp, book: HedgeBook) -> None:
        chain, series = self.chains[book.policy.chain], book.series
        final = chain.final.get(series.contract_month)
        if final is None or final[0] > t:
            return
        day, price = final
        units, intrinsic = book.units, series.intrinsic(price)
        value = units * intrinsic * chain.spec.point_value
        tax, commission = chain.spec.leg_fees(units, intrinsic) if intrinsic > 0 else (0.0, 0.0)
        self.cash += value - tax - commission
        book.series, book.units, book.mark = None, 0, math.nan
        record = Fill(
            t, "close", book.policy.name, series.key, -units, intrinsic, intrinsic, tax, commission, 0.0, "expiry"
        )
        self.fills.append(record)
        self.fills_today.append(record)
        self.event(
            t,
            "option_expiry",
            book.policy.name,
            value=value,
            note=f"{series.key}: final settlement {price:g} on {day.date()}, intrinsic {intrinsic:g} x {units} units",
        )

    def fill_hedge(self, t: pd.Timestamp, book: HedgeBook) -> None:
        chain = self.chains[book.policy.chain]
        series, target = book.pending
        book.pending = None
        reason = "rebalance" if series is None or book.series in (None, series) else "roll"
        if book.series is not None and series != book.series:
            settle = chain.settlement(t, book.series)
            if not math.isfinite(settle):
                self.event(t, "hedge_fill_missing", book.policy.name, note=f"{book.series.key}: no settlement, kept")
                return
            self.option_trade(t, book, book.series, -book.units, settle, reason)
        if series is None or target == book.units:
            return
        settle = chain.settlement(t, series)
        if not math.isfinite(settle):
            self.event(t, "hedge_fill_missing", book.policy.name, note=f"{series.key}: no settlement, not bought")
            return
        self.option_trade(t, book, series, target - book.units, settle, reason)

    def options_close(self, t: pd.Timestamp) -> None:
        """Expiry, then the orders due today, then the marks (all at today's settlement prices)."""

        for book in self.hedges.values():
            if book.series is not None:
                self.option_expiry(t, book)
            if book.pending is not None:
                self.fill_hedge(t, book)
            if book.series is not None:
                settle = self.chains[book.policy.chain].settlement(t, book.series)
                if math.isfinite(settle):
                    book.mark = settle

    def decide_hedge(self, t: pd.Timestamp, book: HedgeBook, final: Mapping[str, int], prices: Mapping[str, float]):
        policy = book.policy
        chain = self.chains[policy.chain]
        target = policy.units(
            final[policy.underlying], self.pos[policy.underlying].spec.point_value, chain.spec.point_value
        )
        book.target_units = target
        held = book.series
        if not target:
            book.pending = (None, 0) if held is not None else None
        elif held is not None and chain.days_left(t, held.expiry) > policy.roll_days:
            book.pending = (held, target) if target != book.units else None
        else:
            series, why = chain.select(t, policy, prices[policy.underlying])
            if series is None:
                self.event(t, HEDGE_UNAVAILABLE, policy.name, threshold=policy.max_strike_gap, note=why)
            book.pending = None if series is None else (series, target)
        order = book.units if book.pending is None else book.pending[1]
        self.decisions.append({"date": t, "instrument": policy.name, "requested": target, "final": order})

    def decide(self, i: int, t: pd.Timestamp) -> dict[str, int]:
        cfg = self.config
        view = MarketView(
            t,
            {k: SeriesView(self.series[k], self.series[k].n) for k in self.pos},
            {k: AuxView(b, b.n) for k, b in self.aux_buffers.items()},
        )
        prices = {k: p.table.price("close", i, int(p.table.held[i])) for k, p in self.pos.items()}
        # settled kinds are marked at the close of the contract still held (a signal_close roll is
        # executed after this decision); kinds paid in full carry units x close.
        equity = self.equity_at(self.held_price("close", i))
        im, mm = self.margin(prices)
        book = BookView(
            date=t,
            initial_capital=cfg.initial_capital,
            cash=self.cash,
            equity=equity,
            positions={k: p.units for k, p in self.pos.items()},
            contracts={k: p.contract_name(int(p.table.held[i])) for k, p in self.pos.items()},
            margin_required=im,
            maintenance_required=mm,
            fills_today=tuple(self.fills_today),
        )
        decision = self.strategy.on_bar(view, book)
        requested = _check_targets(decision, self.instruments)
        _check_short(requested, self.pos)
        if decision.protective:
            raise NotImplementedError("protective orders run on the intraday engine only (engine.intraday)")
        final = {k: requested.get(k, p.units) for k, p in self.pos.items()}

        if self.margin_cap is not None:
            for k, cap in self.margin_cap.items():
                final[k] = int(math.copysign(min(abs(final[k]), cap), final[k]))
        gross = sum(self.pos[k].spec.notional(u, prices[k]) for k, u in final.items())
        if gross > cfg.leverage_cap * equity:
            clipped = _scale_toward_zero(final, cfg.leverage_cap * equity / gross)
            self.event(
                t, "clip_leverage", value=gross / equity, threshold=cfg.leverage_cap, note=f"{final} -> {clipped}"
            )
            final = clipped
        im_target, _ = self.margin(prices, final)
        if im_target > cfg.max_margin_utilisation * equity:
            clipped = _scale_toward_zero(final, cfg.max_margin_utilisation * equity / im_target)
            self.event(
                t,
                "clip_margin",
                value=im_target / equity,
                threshold=cfg.max_margin_utilisation,
                note=f"{final} -> {clipped}",
            )
            final = clipped
        final = self.clip_cash(t, final, prices)
        if self.margin_cap is None:
            final = {
                k: (u if abs(u - self.pos[k].units) >= cfg.min_change_units else self.pos[k].units)
                for k, u in final.items()
            }
        for book in self.hedges.values():
            self.decide_hedge(t, book, final, prices)

        lev = decision.target_leverage
        for k, p in self.pos.items():
            req = requested.get(k)
            self.decisions.append({"date": t, "instrument": k, "requested": req, "final": final[k]})
            self.positions_daily.append(
                {
                    "date": t,
                    "instrument": k,
                    "target_units": req,
                    "order_units": final[k],
                    "target_leverage": lev.get(k, math.nan),
                    "implied_target_leverage": (
                        math.nan if req is None else req * prices[k] * p.spec.point_value / equity
                    ),
                }
            )
        return final

    def clip_cash(self, t: pd.Timestamp, final: dict[str, int], prices: Mapping[str, float]) -> dict[str, int]:
        """Scale down the buys of instruments paid in full so that, at today's closes, cash after
        every order stays at or above the initial margin of the margined targets."""

        cash_after, buy_cost, buys = self.cash, 0.0, {}
        for k, u in final.items():
            p = self.pos[k]
            delta = u - p.units
            if not delta:
                continue
            fill = p.spec.fill_price(delta, prices[k])
            mark = _mark_after_roll(p.spec, prices[k])  # what the order settles against once it fills today
            flow = delta * (mark - fill) * p.spec.point_value - sum(p.spec.leg_fees(delta, fill))
            if delta > 0 and not p.spec.policy.margined:
                buy_cost -= flow
                buys[k] = delta
            else:
                cash_after += flow
        floor, _ = self.margin(prices, final)
        if not buy_cost or buy_cost <= cash_after - floor:
            return final
        scaled = _scale_toward_zero(buys, max(cash_after - floor, 0.0) / buy_cost)
        clipped = {**final, **{k: self.pos[k].units + delta for k, delta in scaled.items()}}
        self.event(t, "clip_cash", value=buy_cost, threshold=cash_after - floor, note=f"{final} -> {clipped}")
        return clipped

    def execute_now(self, i: int, t: pd.Timestamp, final: Mapping[str, int]) -> None:
        self.roll(i, t, final)
        closes = self.held_price("close", i)
        self.execute(t, {k: u for k, u in final.items() if self.pos[k].table.traded[i]}, closes, "close")

    def record(self, i: int, t: pd.Timestamp, deciding: bool) -> None:
        closes = self.held_price("close", i)
        equity = self.equity_at(closes)
        im, mm = self.margin(closes)
        gross = sum(p.spec.notional(p.units, closes[k]) for k, p in self.pos.items())
        prev_equity = self.daily[-1]["equity"] if self.daily else self.config.initial_capital
        fees = sum(f.tax + f.commission for f in self.fills_today)
        self.daily.append(
            {
                "date": t,
                "equity": equity,
                "pnl_twd": equity - prev_equity,
                "daily_return": (equity - prev_equity) / prev_equity,
                "fees_twd": fees,
                "slippage_twd": sum(f.slippage_twd for f in self.fills_today),
                "gross_exposure_twd": gross,
                "leverage": gross / equity if equity > 0 else math.inf,
                "margin_required_twd": im,
                "maintenance_required_twd": mm,
                "margin_utilisation": im / equity if equity > 0 else math.inf,
            }
        )
        held = {k: (p.units, p.contract_name(), closes[k]) for k, p in self.pos.items()}
        rows = self.positions_daily[-len(self.pos) :] if deciding else []
        if not deciding:
            for k in self.pos:
                rows.append(
                    {
                        "date": t,
                        "instrument": k,
                        "target_units": None,
                        "order_units": None,
                        "target_leverage": math.nan,
                        "implied_target_leverage": math.nan,
                    }
                )
            self.positions_daily.extend(rows)
        for row in rows:
            k = row["instrument"]
            units, contract, close = held[k]
            spec = self.pos[k].spec
            row.update(
                contract=contract,
                held_units=units,
                close=close,
                actual_leverage=units * close * spec.point_value / equity if equity > 0 else math.inf,
                rounding_error=row["target_leverage"] - row["implied_target_leverage"],
                margin_utilisation=spec.notional(units, close) * spec.initial_margin_ratio / equity
                if equity > 0
                else math.inf,
                proxy=spec.price_proxy is not None
                or (spec.listed_since is not None and t < pd.Timestamp(spec.listed_since)),
            )
        for name, book in self.hedges.items():
            order = book.units if book.pending is None else book.pending[1]
            self.positions_daily.append(
                {
                    "date": t,
                    "instrument": name,
                    "contract": None if book.series is None else book.series.key,
                    "target_units": book.target_units if deciding else None,
                    "order_units": order if deciding else None,
                    "held_units": book.units,
                    "close": book.mark,
                    "target_leverage": math.nan,
                    "implied_target_leverage": math.nan,
                    "rounding_error": math.nan,
                    "actual_leverage": math.nan,
                    "margin_utilisation": 0.0,
                    "proxy": False,
                }
            )

    def run(self) -> EngineResult:
        cfg = self.config
        for i, t in enumerate(self.dates):
            self.fills_today = []
            self.feed(i, t)
            self.corporate_actions(i, t)
            if self.pending is not None:
                self.fill_pending(i, t)
            self.intraday_check(i, t)
            closes = self.settle(i, t)
            self.options_close(t)
            equity = self.equity_at(closes)
            if equity <= 0:
                self.ruined = t
                self.event(t, "ruin", value=equity, threshold=0.0)
                if t >= self.start:
                    self.record(i, t, deciding=False)
                break
            self.margin_call_check(t, closes)
            deciding = t >= self.start
            if cfg.execution == "next_open":
                self.roll(i, t, None)
                if deciding:
                    final = self.decide(i, t)
                    self.pending = {k: u for k, u in final.items() if u != self.pos[k].units}
            elif deciding:
                self.execute_now(i, t, self.decide(i, t))
                for book in self.hedges.values():
                    if book.pending is not None:
                        self.fill_hedge(t, book)
            else:
                self.roll(i, t, None)
            if deciding:
                self.record(i, t, deciding=True)
        unfilled = dict(self.pending or {}) if self.ruined is None else {}
        if self.ruined is None:
            unfilled.update({k: b.pending[1] for k, b in self.hedges.items() if b.pending is not None})
        return EngineResult.build(
            config=cfg,
            fills=self.fills,
            positions_daily=self.positions_daily,
            daily=self.daily,
            events=self.events,
            decisions=self.decisions,
            ruined=self.ruined,
            unfilled=unfilled,
        )


def run(
    strategy: TargetStrategy,
    futures: Mapping[str, FuturesInput],
    config: EngineConfig | None = None,
    *,
    aux: Mapping[str, pd.DataFrame] | None = None,
    calendar: Sequence[Any] | None = None,
    options: Mapping[str, OptionInput] | None = None,
    hedges: Sequence[HedgePolicy] = (),
) -> EngineResult:
    """Simulate ``strategy`` over ``futures`` (name -> input). ``calendar`` defaults to the bar dates.

    ``options`` (chain name -> input) and ``hedges`` add bought-option overlays (``engine.options``)."""

    return _Run(strategy, futures, config or EngineConfig(), aux or {}, calendar, options, hedges).run()


StrategyFactory = Callable[[], TargetStrategy]
