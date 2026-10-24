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
    ``instruments.SIGNAL_INDICES``); a signal ref ``index:<name>`` reads aux ``<name>``.
    ``inputs`` records each dataset's ``content_sha256`` for run input hashes.

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
    ContractSpec,
    EngineConfig,
    EngineResult,
    FuturesInput,
    TruncationAuditReport,
    run,
    truncation_audit,
)
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.instruments import (
    INSTRUMENTS,
    MAX_MARGIN_UTILISATION,
    SIGNAL_INDICES,
    Instrument,
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
from poseidon.pipeline.v3.validation import LEVERAGE_CAP, FeasibilityContext, validate_spec_v3, warmup_calendar_days

AUX_DATASETS = {name: index.dataset for name, index in SIGNAL_INDICES.items()}
HOOKS = {
    "direction": "step",
    "leverage": "step",
    "sizing": "units",
    "rebalance": "should_trade",
    "allocation": "weights",
    "roll": "roll_due",
}
PROJECTED_CALENDAR_DAYS = 10


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

    @property
    def bounds(self) -> list[Bound]:
        return [b for s in self.sleeves for b in (s.direction, *s.leverage, s.sizing, s.rebalance)]

    @property
    def warmup_bars(self) -> int:
        return max((b.component.history(b.params) for b in self.bounds), default=0)

    @property
    def signal_indices(self) -> tuple[str, ...]:
        """Aux names of the indices the components' ``signal`` refs read (``index:<name>``)."""

        refs = (getattr(b.params, "signal", None) for b in self.bounds)
        return tuple(sorted({index.name for ref in refs if isinstance(ref, str) and (index := signal_index(ref))}))


def plan(resolved: dict) -> Plan:
    if any(is_param_ref(node) for _path, node in walk(resolved)):
        raise ValueError("compile a resolved spec: bind every $param first (see resolve())")
    if resolved["frequency"] != "1d":
        raise NotCompilable("intraday specs stay on the v1 runner until DSL parity (step 7)")
    instruments, rolls = {}, {}
    for alias, decl in resolved["instruments"].items():
        instrument = INSTRUMENTS[decl["symbol"]]
        roll = decl.get("roll")
        if instrument.kind != "future" or roll is None or roll["component"] != "roll.before_settlement":
            raise NotCompilable(f"instruments.{alias}: the daily engine runs futures with roll.before_settlement")
        instruments[alias] = instrument
        rolls[alias] = parse_bound(REGISTRY[roll["component"]], roll.get("params")).params.days_before

    def bound(ref: dict) -> Bound:
        component = REGISTRY[ref["component"]]
        hook = HOOKS[component.slot]
        base = next(cls for cls in type(component).__mro__ if cls.__dict__.get("slot") == component.slot)
        if getattr(type(component), hook) is getattr(base, hook):
            raise NotCompilable(f"{component.name} has no daily behaviour ({hook}) yet")
        return parse_bound(component, ref.get("params"))

    sleeves, traded = [], set()
    for sleeve in resolved["sleeves"]:
        if sleeve["direction"]["component"] == "direction.dsl_rules":
            raise NotCompilable("direction.dsl_rules runs on the new engine from step 7")
        if sleeve.get("exits") or sleeve.get("hedges"):
            raise NotCompilable(f"sleeve {sleeve['id']}: exits and hedges are not in the daily engine yet")
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


def trading_calendar(bars: pd.DataFrame, projected_days: int = PROJECTED_CALENDAR_DAYS) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))
    projected = pd.bdate_range(days[-1] + pd.offsets.BDay(1), periods=projected_days)
    return days.append(projected)


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

    def strategy(self) -> SpecStrategy:
        return strategy_factory(self.plan, self.conventions)()

    def run(self, strategy: SpecStrategy | None = None) -> EngineResult:
        return run(strategy or self.strategy(), self.futures, self.config, aux=self.aux, calendar=self.calendar)

    def audit(self, n_cuts: int = 20, seed: int = 0) -> TruncationAuditReport:
        return truncation_audit(
            strategy_factory(self.plan, self.conventions),
            self.futures,
            self.config,
            aux=self.aux,
            calendar=self.calendar,
            n_cuts=n_cuts,
            seed=seed,
        )


def compile_spec(
    resolved: dict,
    *,
    start: Any,
    end: Any = None,
    store: RefdataStore | None = None,
    conventions: Conventions = Conventions(),
) -> CompiledSpec:
    """Strategy, config and refdata inputs for a resolved daily spec over ``[start, end]``."""

    spec_plan = plan(resolved)
    store = store or RefdataStore()
    start = pd.Timestamp(start).normalize()
    warmup = spec_plan.warmup_bars + conventions.signal_lag
    load_from = start - timedelta(days=warmup_calendar_days(warmup, "1d"))
    upper = None if end is None else pd.Timestamp(end).normalize()

    quotes = store.read(FUTURES, start=load_from, end=upper)
    settlement = store.read(CALENDAR)
    sources = {
        alias: instrument.price_proxy or instrument.symbol for alias, instrument in spec_plan.instruments.items()
    }
    notes: list[str] = []
    futures, calendars = {}, []
    for alias, instrument in spec_plan.instruments.items():
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
            spec=ContractSpec.from_instrument(instrument),
            bars=bars,
            expiries={c: calendar_rows.at[c, "last_trading_day"] for c in contracts},
            roll_days_before=spec_plan.rolls[alias],
        )
        calendars.append(trading_calendar(bars))
        note = instrument.proxy_note(start.date())
        if note:
            notes.append(note)
        night = quotes.loc[(quotes["product"] == product) & (quotes["session"] == "night"), "trade_date"]
        if night.empty or start < night.min():
            notes.append(
                f"{alias}: before the first {product} after-hours session in the data, next_open fills at the "
                "next 08:45 day-session open"
            )
    calendar = calendars[0]
    if any(not other.equals(calendar) for other in calendars[1:]):
        raise NotCompilable("all instruments must share one trading calendar")
    data_days = calendar[: len(calendar) - PROJECTED_CALENDAR_DAYS]
    before = int((data_days < start).sum())
    if before < warmup:
        raise NotCompilable(
            f"{before} trading days of data before {start.date()}, the components need {warmup} for warm-up"
        )
    if start > data_days[-1]:
        raise NotCompilable(f"start {start.date()} is after the last bar {data_days[-1].date()}")
    notes.append(
        f"trading calendar: {PROJECTED_CALENDAR_DAYS} weekdays after {data_days[-1].date()} are projected "
        "(holidays unknown) so a roll due on the last day is seen"
    )
    aux = {}
    for name, dataset in AUX_DATASETS.items():
        frame = store.read(dataset, start=load_from, end=upper)
        if len(frame):
            aux[name] = frame.set_index("trade_date").sort_index()
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
    inputs = {dataset: store.content_sha256(dataset) for dataset in (FUTURES, CALENDAR, *AUX_DATASETS.values())}
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
    )
