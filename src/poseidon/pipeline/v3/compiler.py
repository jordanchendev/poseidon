"""Compile a resolved spec v3 into an engine run (design 2026-10-24 §5, §10 step 3).

``compile_spec(resolved, start=, end=)`` returns a :class:`CompiledSpec`: a fresh-strategy
factory (:class:`~poseidon.pipeline.v3.runtime.SpecStrategy`), the ``EngineConfig`` and the
engine inputs read from the refdata store for ``[start - warm-up, end]``.

Spec -> engine
    * each instrument alias -> one engine instrument of that name: ``ContractSpec`` from its
      ``INSTRUMENTS`` row, bars of its price source (``price_proxy`` or itself),
      ``roll.before_settlement.days_before`` -> ``FuturesInput.roll_days_before``;
    * ``execution.<x>.fill`` -> ``EngineConfig.execution`` (``Conventions.execution`` overrides);
    * ``capital.initial_twd`` -> initial capital; policy caps from the validator's constants:
      gross leverage 5x (§12 #7), initial margin <= 80 % of equity at the decision (§3.7.6);
    * rebalancing is the spec's rebalance component inside the strategy, so the engine's own
      ``min_change_units`` band stays at 1 (a no-op for integer units).

Data (refdata, read-only)
    ``taifex_futures_daily``: monthly contracts of the price source. A bar of trading day D is
    the day-session row D plus the after-hours row TAIFEX dates D (the session that opened at
    15:00 on the previous trading day): ``open`` = after-hours open when that session traded
    (from 2017-05-16), else the 08:45 day open; ``high``/``low`` over both sessions; ``close``
    = day-session close (settlement when it did not trade). So ``execution.next_open`` fills at
    15:00 on the decision day from 2017-05-16 on, and at the next morning's 08:45 before that.
    ``taifex_settlement_calendar``: ``last_trading_day`` per contract = engine expiry (contracts
    still trading carry the third-Wednesday projection; noted). Trading calendar: the price
    source's day-session dates plus projected weekdays after the data (so a roll due on the last
    day is still seen; noted). Aux ``taiex`` (``twse_taiex_daily``) and ``taiex_tr``
    (``twse_taiex_total_return``) are always attached (``AUX_DATASETS``, from
    ``instruments.SIGNAL_INDICES``); a signal ref ``index:<name>`` reads aux ``<name>``, attached
    (and recorded in ``inputs``) only when a component reads it (``index:vix``). Aux rows are
    re-dated by the index's ``lag_days`` (``instruments.align_signal_frame``): the VIX close of US
    date u is a row dated u + 1, so a decision on TW date t reads US dates <= t - 1.
    ``inputs`` records each dataset's ``content_sha256`` for run input hashes.
    Hedged specs only: ``taifex_txo_daily`` day-session monthly puts (settlement prices, expiry
    dates) and ``taifex_txo_final_settlement`` become the ``TXO`` option chain, and each
    ``hedges`` entry an ``engine.HedgePolicy`` named ``<alias>.<component suffix>``. A run whose
    hedge was unavailable on more than 5 % of the days it wanted one raises
    :class:`HedgeCoverageError` (§5.7).

The warm-up is the largest ``history()`` of the bound components plus ``signal_lag``; the
compiler checks that many trading days exist before ``start``, and that many rows of every
index a signal reads. More history never changes a result: decisions start at ``start``, the
MA-band state starts there, and every indicator is scale-free in the causal adjusted series.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from poseidon.pipeline.engine import (
    EngineConfig,
    EngineResult,
    FuturesInput,
    HedgePolicy,
    OptionInput,
    TruncationAuditReport,
    hedge_coverage,
    run,
    truncation_audit,
)
from poseidon.pipeline.engine.data import stock_input
from poseidon.pipeline.engine.options import MAX_UNAVAILABLE_SHARE
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, MARKET_CALENDAR, TAIEX
from poseidon.pipeline.refdata.engine_input import StockDataGap, TaifexCalendar, align_stock_bars, read_taifex_calendar
from poseidon.pipeline.refdata.store import DatasetMissingError, RefdataStore
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET as OPTION_QUOTES
from poseidon.pipeline.refdata.taifex_options import FSP_DATASET as OPTION_FINAL
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_corporate_actions import require_actions
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.v3.instruments import (
    INSTRUMENTS,
    MAX_MARGIN_UTILISATION,
    SIGNAL_INDICES,
    Instrument,
    align_signal_frame,
    contract_spec,
    declared,
    option_spec,
    signal_index,
)
from poseidon.pipeline.v3.registry import REGISTRY
from poseidon.pipeline.v3.runtime import (
    Bound,
    Conventions,
    SleevePlan,
    SpecStrategy,
    StrategyFactory,
    parse_bound,
)
from poseidon.pipeline.v3.spec import DEFAULT_INITIAL_CAPITAL_TWD, is_param_ref, substitute, walk
from poseidon.pipeline.v3.validation import validate_spec_v3, warmup_calendar_days
from poseidon.pipeline.v3.validation_types import LEVERAGE_CAP, FeasibilityContext, SessionWarmup

AUX_DATASETS = {name: index.dataset for name, index in SIGNAL_INDICES.items()}
ALWAYS_ATTACHED = tuple(name for name, index in SIGNAL_INDICES.items() if index.always_attached)
HOOKS = {
    "direction": "step",
    "leverage": "step",
    "sizing": "units",
    "rebalance": "should_trade",
    "allocation": "weights",
    "roll": "roll_due",
    "hedge": "policy",
}
PROJECTED_CALENDAR_DAYS = 10
OPTION_DATASETS = (OPTION_QUOTES, OPTION_FINAL)
OPTION_QUOTE_FIELDS = ["trade_date", "contract_month", "expiry_date", "strike", "cp", "settlement"]


class HedgeCoverageError(ValueError):
    """A hedge was unavailable on more than ``MAX_UNAVAILABLE_SHARE`` of the days it wanted a position (§5.7)."""


def hedged(spec: dict) -> bool:
    """Whether any sleeve of a (template or resolved) spec carries a hedge."""

    return any(sleeve.get("hedges") for sleeve in spec.get("sleeves") or [])


def option_quote_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows and columns a protective put reads: day session, monthly expiries, puts."""

    keep = (frame["session"] == "day") & (frame["expiry_kind"] == "monthly") & (frame["cp"] == "P")
    return frame.loc[keep, OPTION_QUOTE_FIELDS].reset_index(drop=True)


# Datasets read through a row filter (``RefdataStore.read(rows=)``): what a hedged run reads of them.
ROW_FILTERS = {OPTION_QUOTES: option_quote_rows}


class NotCompilable(ValueError):
    """The spec is valid but this engine cannot run it yet, or the data cannot support the window."""


# --------------------------------------------------------------------------- spec -> plan


def resolve(template: dict, values: Mapping[str, int | float] | None = None, **feasibility: Any) -> dict:
    """Validate a spec template and bind its ``$param``s (defaults, overridden by ``values``)."""

    result = validate_spec_v3(template, feasibility=FeasibilityContext(**feasibility))
    if not result.ok:
        problems = "; ".join(issue.text() for issue in (*result.errors, *result.unsupported))
        raise ValueError(f"spec does not validate: {problems}")
    values = dict(values or {})
    unknown = set(values) - set(result.params)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}")
    off_grid = {name: value for name, value in values.items() if not result.params[name].contains(value)}
    if off_grid:
        raise ValueError(f"values off their declared grid: {off_grid}")
    bound = {name: spec.cast(spec.default) for name, spec in result.params.items()}
    bound.update({name: result.params[name].cast(value) for name, value in values.items()})
    return substitute(result.template, bound)


@dataclass(frozen=True)
class Plan:
    sleeves: tuple[SleevePlan, ...]
    allocation: Bound
    execution: str
    rolls: dict[str, int]  # alias -> days_before
    instruments: dict[str, Instrument]  # alias -> row
    initial_capital: float
    hedges: tuple[tuple[str, Bound], ...] = ()  # (sleeve instrument alias, hedge component)

    @property
    def bounds(self) -> list[Bound]:
        return [*(b for s in self.sleeves for b in (s.direction, *s.leverage, s.sizing, s.rebalance)), self.allocation]

    @property
    def warmup_bars(self) -> int:
        return max((b.component.history(b.params) for b in self.bounds), default=0)

    @property
    def signal_indices(self) -> tuple[str, ...]:
        """Aux names of the indices the components' ``signal`` refs read (``index:<name>``)."""

        refs = (getattr(b.params, "signal", None) for b in self.bounds)
        return tuple(sorted({index.name for ref in refs if isinstance(ref, str) and (index := signal_index(ref))}))


def plan(resolved: dict) -> Plan:
    """The engine plan of a resolved spec. An intraday spec (the DSL v2 fast path, validated to one
    ``direction.dsl_rules`` sleeve) runs on ``engine.intraday`` with the synthetic R1 roll, so its
    ``rolls`` is empty; :func:`compile_spec` loads daily refdata and takes daily specs only."""

    if any(is_param_ref(node) for _path, node in walk(resolved)):
        raise ValueError("compile a resolved spec: bind every $param first (see resolve())")
    intraday = resolved["frequency"] != "1d"
    expected_roll = "roll.at_settlement" if intraday else "roll.before_settlement"
    instruments, rolls = {}, {}
    for alias, decl in resolved["instruments"].items():
        instrument = declared(decl)
        roll = decl.get("roll")
        if instrument.kind == "future":
            if roll is None or roll["component"] != expected_roll:
                kind = "intraday" if intraday else "daily"
                raise NotCompilable(f"instruments.{alias}: {kind} specs trade futures with {expected_roll}")
            if not intraday:
                rolls[alias] = parse_bound(REGISTRY[roll["component"]], roll.get("params")).params.days_before
        elif intraday or instrument.kind != "stock":
            engine = "intraday" if intraday else "daily"
            raise NotCompilable(f"instruments.{alias}: the {engine} engine has no ledger for {instrument.kind}")
        instruments[alias] = instrument

    def bound(ref: dict) -> Bound:
        component = REGISTRY[ref["component"]]
        hook = HOOKS[component.slot]
        base = next(cls for cls in type(component).__mro__ if cls.__dict__.get("slot") == component.slot)
        if getattr(type(component), hook) is getattr(base, hook):
            raise NotCompilable(f"{component.name} has no daily behaviour ({hook}) yet")
        return parse_bound(component, ref.get("params"))

    sleeves, traded, hedges = [], set(), []
    for sleeve in resolved["sleeves"]:
        if (sleeve["direction"]["component"] == "direction.dsl_rules") != intraday:
            raise NotCompilable("direction.dsl_rules is the only direction on intraday bars, and only there")
        if intraday and (sleeve.get("exits") or sleeve.get("hedges")):
            raise NotCompilable(f"sleeve {sleeve['id']}: exits and hedges are not in the intraday engine")
        if sleeve.get("exits"):
            raise NotCompilable(f"sleeve {sleeve['id']}: exits are not in the daily engine yet")
        hedges.extend((sleeve["instrument"], bound(ref)) for ref in sleeve.get("hedges") or [])
        if sleeve["instrument"] in traded:
            raise NotCompilable(f"instrument alias {sleeve['instrument']!r} is traded by two sleeves")
        traded.add(sleeve["instrument"])
        sleeves.append(
            SleevePlan(
                id=sleeve["id"],
                alias=sleeve["instrument"],
                instrument=instruments[sleeve["instrument"]],
                direction=bound(sleeve["direction"]),
                leverage=tuple(bound(link) for link in sleeve.get("leverage") or []),
                sizing=bound(sleeve["sizing"]),
                rebalance=bound(sleeve["rebalance"]),
            )
        )
    execution = REGISTRY[resolved["execution"]["component"]]
    capital = float((resolved.get("capital") or {}).get("initial_twd", DEFAULT_INITIAL_CAPITAL_TWD))
    return Plan(
        sleeves=tuple(sleeves),
        allocation=bound(resolved["allocation"]),
        execution=execution.fill,
        rolls=rolls,
        instruments=instruments,
        initial_capital=capital,
        hedges=tuple(hedges),
    )


def strategy_factory(spec_plan: Plan, conventions: Conventions = Conventions()) -> StrategyFactory:
    return lambda: SpecStrategy(spec_plan.sleeves, spec_plan.allocation, conventions)


def engine_config(spec_plan: Plan, *, start: Any, end: Any = None, conventions: Conventions = Conventions()):
    return EngineConfig(
        initial_capital=spec_plan.initial_capital,
        execution=conventions.execution or spec_plan.execution,
        leverage_cap=LEVERAGE_CAP,
        max_margin_utilisation=MAX_MARGIN_UTILISATION,
        min_change_units=1,
        start=pd.Timestamp(start),
        end=None if end is None else pd.Timestamp(end),
    )


# --------------------------------------------------------------------------- refdata -> engine inputs


def futures_bars(quotes: pd.DataFrame, product: str) -> pd.DataFrame:
    """Engine bars (see the module docstring) from ``taifex_futures_daily`` rows of one product."""

    monthly = quotes[(quotes["product"] == product) & quotes["contract_month"].str.fullmatch(r"\d{6}")]
    keys = ["trade_date", "contract_month"]
    day = monthly[monthly["session"] == "day"][[*keys, "open", "high", "low", "close", "settlement"]]
    night = monthly[monthly["session"] == "night"][[*keys, "open", "high", "low"]]
    frame = day.merge(night, on=keys, how="left", suffixes=("", "_night"))
    frame["open"] = frame["open_night"].fillna(frame["open"])
    frame["high"] = frame[["high", "high_night"]].max(axis=1)
    frame["low"] = frame[["low", "low_night"]].min(axis=1)
    frame = frame.rename(columns={"trade_date": "date", "contract_month": "contract", "settlement": "settle"})
    frame["contract"] = frame["contract"].astype(str)
    return frame[["date", "contract", "open", "high", "low", "close", "settle"]].reset_index(drop=True)


def trading_calendar(
    bars: pd.DataFrame, projected_days: int = PROJECTED_CALENDAR_DAYS, market: TaifexCalendar | None = None
) -> pd.DatetimeIndex:
    """The bars' dates, then ``projected_days`` more: ``market``'s trading days, or weekdays without it."""

    days = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))
    if market is None:
        projected = pd.bdate_range(days[-1] + pd.offsets.BDay(1), periods=projected_days)
    else:
        projected = market.after(days[-1], projected_days)
    return days.append(projected)


def projection_calendar(rule: Mapping | None, store, since: pd.Timestamp) -> TaifexCalendar | None:
    """The trading days a run frozen with ``data.trading_calendar`` (``v3.config.trading_calendar_record``)
    projects past its data end, read from ``store`` (a stage's ``WindowStore``: sessions up to the window's
    upper bound, the official schedule after it). None for a run frozen without the rule: weekdays."""

    if rule is None:
        return None
    if (rule.get("sessions"), rule.get("schedule")) != (TAIEX, MARKET_CALENDAR):
        raise ValueError(f"unknown data.trading_calendar {dict(rule)!r}")
    return read_taifex_calendar(store, since)


@dataclass(frozen=True)
class CompiledSpec:
    name: str
    plan: Plan
    conventions: Conventions
    config: EngineConfig
    futures: dict[str, FuturesInput]
    aux: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    inputs: dict[str, str]  # refdata dataset -> content_sha256
    notes: tuple[str, ...] = field(default_factory=tuple)
    options: dict[str, OptionInput] = field(default_factory=dict)
    hedges: tuple[HedgePolicy, ...] = ()

    def strategy(self) -> SpecStrategy:
        return strategy_factory(self.plan, self.conventions)()

    def run(self, strategy: SpecStrategy | None = None) -> EngineResult:
        """The engine run; raises :class:`HedgeCoverageError` past the §5.7 hedge-unavailable share."""

        result = self.engine_run(strategy)
        for name, cover in hedge_coverage(result.positions_daily, result.events, [h.name for h in self.hedges]).items():
            if cover["unavailable_share"] > MAX_UNAVAILABLE_SHARE:
                raise HedgeCoverageError(
                    f"{name}: hedge unavailable on {cover['unavailable_days']} of {cover['hedged_days']} hedged days "
                    f"({cover['unavailable_share']:.1%} > {MAX_UNAVAILABLE_SHARE:.0%}), first "
                    f"{', '.join(cover['unavailable_dates'][:5])}"
                )
        return result

    def engine_run(self, strategy: SpecStrategy | None = None) -> EngineResult:
        """The engine run without the hedge-coverage rule (reports that list unavailable days)."""

        return run(
            strategy or self.strategy(),
            self.futures,
            self.config,
            aux=self.aux,
            calendar=self.calendar,
            options=self.options,
            hedges=self.hedges,
        )

    def audit(self, n_cuts: int = 20, seed: int = 0) -> TruncationAuditReport:
        return truncation_audit(
            strategy_factory(self.plan, self.conventions),
            self.futures,
            self.config,
            aux=self.aux,
            calendar=self.calendar,
            n_cuts=n_cuts,
            seed=seed,
            options=self.options,
            hedges=self.hedges,
        )


def compile_spec(
    resolved: dict,
    *,
    start: Any,
    end: Any = None,
    store: RefdataStore | None = None,
    conventions: Conventions = Conventions(),
    sessions: SessionWarmup | None = None,
    calendar_rule: Mapping | None = None,
) -> CompiledSpec:
    """Strategy, config and refdata inputs for a resolved daily spec over ``[start, end]``.

    ``sessions``: the run's frozen session warm-up rule; data loads from that many sessions before ``start``.
    None (a run frozen without ``data.warmup``): from ``warmup_calendar_days`` calendar days before it.
    ``calendar_rule``: the run's frozen ``data.trading_calendar``; the days projected past the data end are then
    TAIFEX trading days (:func:`projection_calendar`). None: weekdays, holidays unknown.
    """

    if resolved["frequency"] != "1d":
        raise NotCompilable("compile_spec reads daily refdata; intraday specs run on engine.intraday")
    spec_plan = plan(resolved)
    store = store or RefdataStore()
    start = pd.Timestamp(start).normalize()
    warmup = spec_plan.warmup_bars + conventions.signal_lag
    if sessions is None:
        load_from = start - timedelta(days=warmup_calendar_days(warmup, "1d"))
    else:
        load_from = pd.Timestamp(sessions.load_from(start.date(), warmup))
    upper = None if end is None else pd.Timestamp(end).normalize()

    by_kind = {kind: [a for a, i in spec_plan.instruments.items() if i.kind == kind] for kind in ("future", "stock")}
    if by_kind["future"]:
        quotes = store.read(FUTURES, start=load_from, end=upper)
        settlement = store.read(CALENDAR)
    sources = {
        alias: instrument.price_proxy or instrument.symbol for alias, instrument in spec_plan.instruments.items()
    }
    market = projection_calendar(calendar_rule, store, load_from)
    notes: list[str] = []
    futures, calendars = {}, []
    for alias in by_kind["future"]:
        instrument = spec_plan.instruments[alias]
        product = sources[alias]
        bars = futures_bars(quotes, product)
        if bars.empty:
            raise NotCompilable(f"no {product} bars in refdata between {load_from.date()} and {upper}")
        calendar_rows = settlement[settlement["product"] == product].set_index("contract_month")
        contracts = sorted(set(bars["contract"]))
        missing = [c for c in contracts if c not in calendar_rows.index]
        if missing:
            raise NotCompilable(f"{product}: no settlement-calendar row for contracts {missing[:5]}")
        projected = [c for c in contracts if calendar_rows.at[c, "status"] != "settled"]
        if projected:
            notes.append(
                f"{alias}: expiry of {', '.join(projected)} projected by the third-Wednesday rule "
                "(taifex_settlement_calendar status active)"
            )
        futures[alias] = FuturesInput(
            spec=contract_spec(instrument),
            bars=bars,
            expiries={c: calendar_rows.at[c, "last_trading_day"] for c in contracts},
            roll_days_before=spec_plan.rolls[alias],
        )
        calendars.append(trading_calendar(bars, market=market))
        note = instrument.proxy_note(start.date())
        if note:
            notes.append(note)
        night = quotes.loc[(quotes["product"] == product) & (quotes["session"] == "night"), "trade_date"]
        if night.empty or start < night.min():
            notes.append(
                f"{alias}: before the first {product} after-hours session in the data, next_open fills at the "
                "next 08:45 day-session open"
            )
    stock_rows = {}
    for alias in by_kind["stock"]:
        symbol = spec_plan.instruments[alias].symbol
        rows = read_bars(store, symbol, start=load_from, end=upper) if store.exists(STOCKS) else pd.DataFrame()
        if rows.empty:
            raise NotCompilable(f"no {symbol} bars in {STOCKS} between {load_from.date()} and {upper}")
        stock_rows[alias] = rows
    if calendars:
        calendar = calendars[0]
        if any(not other.equals(calendar) for other in calendars[1:]):
            raise NotCompilable("all futures instruments must share one trading calendar")
    else:
        union = pd.concat([rows[["date"]] for rows in stock_rows.values()], ignore_index=True)
        calendar = trading_calendar(union, market=market)
    data_days = calendar[: len(calendar) - PROJECTED_CALENDAR_DAYS]
    for alias, rows in stock_rows.items():
        instrument = spec_plan.instruments[alias]
        try:
            aligned = align_stock_bars(rows, data_days, alias)
        except StockDataGap as error:
            raise NotCompilable(str(error)) from error
        try:
            actions = require_actions(store, instrument.symbol)
            futures[alias] = stock_input(contract_spec(instrument), aligned, actions)
        except (DatasetMissingError, ValueError) as error:
            raise NotCompilable(f"{alias}: {error}") from error
        carried = int((~aligned["traded"]).sum())
        if carried:
            notes.append(f"{alias}: {carried} session(s) without a {instrument.symbol} bar carried forward (no fills)")
        booked = futures[alias].bars
        applied = int(((booked["unit_ratio"] != 1.0) | (booked["cash_per_share"] != 0.0)).sum())
        if applied:
            notes.append(f"{alias}: {applied} corporate action day(s) of {instrument.symbol} booked from {ACTIONS}")
    before = int((data_days < start).sum())
    if before < warmup:
        raise NotCompilable(
            f"{before} trading days of data before {start.date()}, the components need {warmup} for warm-up"
        )
    if start > data_days[-1]:
        raise NotCompilable(f"start {start.date()} is after the last bar {data_days[-1].date()}")
    if market is None:
        notes.append(
            f"trading calendar: {PROJECTED_CALENDAR_DAYS} weekdays after {data_days[-1].date()} are projected "
            "(holidays unknown) so a roll due on the last day is seen"
        )
    else:
        projected = calendar[len(data_days) :]
        notes.append(
            f"trading calendar: the {PROJECTED_CALENDAR_DAYS} trading days after {data_days[-1].date()} "
            f"({projected[0].date()} .. {projected[-1].date()}) are {TAIEX} sessions, then the {MARKET_CALENDAR} "
            "schedule, so a roll due on the last day is seen"
        )
    aux = {}
    attached = sorted({*ALWAYS_ATTACHED, *spec_plan.signal_indices})
    for name in attached:
        try:
            frame = store.read(AUX_DATASETS[name], start=load_from, end=upper)
        except DatasetMissingError as error:
            raise NotCompilable(f"signal index:{name}: {error}") from error
        if len(frame):
            aux[name] = align_signal_frame(name, frame)
    for name in spec_plan.signal_indices:
        rows = aux.get(name)
        if rows is None:
            raise NotCompilable(
                f"signal index:{name}: no {AUX_DATASETS[name]} rows between {load_from.date()} and {upper}"
            )
        rows_before = int((rows.index < start).sum())
        if rows_before < warmup:
            raise NotCompilable(
                f"signal index:{name}: {rows_before} rows of {AUX_DATASETS[name]} before {start.date()}, the components "
                f"need {warmup} for warm-up (data from {SIGNAL_INDICES[name].data_since.isoformat()})"
            )
    datasets = [*((FUTURES, CALENDAR) if by_kind["future"] else ()), *(AUX_DATASETS[n] for n in attached)]
    if by_kind["stock"]:
        datasets += [STOCKS, *((ACTIONS,) if store.exists(ACTIONS) else ())]
    inputs = {dataset: store.content_sha256(dataset) for dataset in datasets}
    if market is not None:
        inputs.update(market.sources)
    options, hedges = {}, []
    if spec_plan.hedges:
        options, hedges = option_inputs(spec_plan, store, load_from, upper)
        inputs.update({dataset: store.content_sha256(dataset) for dataset in OPTION_DATASETS})
        notes.append(
            "TXO: monthly puts from taifex_txo_daily, valued and filled at daily settlement prices plus "
            "max(1 tick, 5% of premium) slippage; a put held to expiry is cash-settled at the final settlement price"
        )
    return CompiledSpec(
        name=resolved["name"],
        plan=spec_plan,
        conventions=conventions,
        config=engine_config(spec_plan, start=start, end=upper, conventions=conventions),
        futures=futures,
        aux=aux,
        calendar=calendar,
        inputs=inputs,
        notes=tuple(notes),
        options=options,
        hedges=tuple(hedges),
    )


def option_inputs(spec_plan: Plan, store, load_from: pd.Timestamp, upper) -> tuple[dict, list[HedgePolicy]]:
    """The option chains and hedge policies of a hedged plan, from refdata over ``[load_from, upper]``."""

    hedges = hedge_policies(spec_plan)
    options: dict[str, OptionInput] = {}
    for symbol in sorted({hedge.chain for hedge in hedges}):
        quotes = store.read(OPTION_QUOTES, start=load_from, end=upper, rows=ROW_FILTERS[OPTION_QUOTES])
        if quotes.empty:
            raise NotCompilable(f"no {symbol} quotes in refdata between {load_from.date()} and {upper}")
        options[symbol] = option_chain(symbol, quotes, store.read(OPTION_FINAL))
    return options, hedges


def hedge_policies(spec_plan: Plan) -> list[HedgePolicy]:
    """One ``HedgePolicy`` per ``hedges`` entry, named ``<alias>.<component suffix>``."""

    hedges = []
    for alias, bound_hedge in spec_plan.hedges:
        component = bound_hedge.component
        name = f"{alias}.{component.name.split('.', 1)[1]}"
        hedges.append(component.policy(bound_hedge.params, name=name, underlying=alias, chain=component.option_symbol))
    return hedges


def option_chain(symbol: str, quotes: pd.DataFrame, final: pd.DataFrame) -> OptionInput:
    """The chain of ``symbol`` from its ``option_quote_rows`` and the ``taifex_txo_final_settlement`` rows."""

    final = final[final["expiry_kind"] == "monthly"]
    return OptionInput(
        spec=option_spec(INSTRUMENTS[symbol]),
        quotes=quotes.rename(columns={"trade_date": "date"}),
        expiries=dict(quotes.groupby("contract_month")["expiry_date"].first()),
        final_settlement={
            row.contract_month: (row.final_settlement_date, row.final_settlement_price) for row in final.itertuples()
        },
    )
