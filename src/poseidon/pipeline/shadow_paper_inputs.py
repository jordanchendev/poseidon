"""Daily shadow paper inputs: the compiled-strategy seam types and the refdata loader that cuts the store at the last complete trading day."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from poseidon.pipeline.engine import ContractSpec, EngineConfig, FuturesInput, HedgePolicy, OptionInput, TargetStrategy
from poseidon.pipeline.engine.data import stock_input
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, MARKET_CALENDAR
from poseidon.pipeline.refdata.engine_input import (
    RefdataReader,
    StockDataGap,
    align_stock_bars,
    day_session_dates,
    futures_input,
    last_day_session,
    product_quotes,
    trading_calendar,
    unscheduled_closures,
)
from poseidon.pipeline.refdata.store import RefdataError
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET as OPTION_QUOTES
from poseidon.pipeline.refdata.taifex_options import FSP_DATASET as OPTION_FINAL
from poseidon.pipeline.refdata.tw_corporate_actions import COMPLETE_THROUGH, actions_complete_through, require_actions
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.v3.instruments import align_signal_frame, complete_through


@dataclass(frozen=True)
class PaperInstrument:
    product: str  # TAIFEX product whose quotes price it (TMF priced from TX: "TX"); a stock's own symbol
    spec: ContractSpec  # ``spec.kind`` says which dataset prices it (future: FUTURES, stock: STOCKS)
    roll_days_before: int = 1  # futures only


@dataclass(frozen=True)
class CompiledStrategy:
    factory: Callable[[], TargetStrategy]  # a NEW strategy instance per call (strategies keep state)
    instruments: Mapping[str, PaperInstrument]
    engine: EngineConfig  # frozen run settings; ``start`` = train start (first decision day)
    data_start: date | None  # first bar loaded (warm-up before ``engine.start``); None: ``warmup`` counts it
    train_end: date  # exclusive end of the train window (protection basis)
    fingerprint: str  # sha256 over spec + params + registry/components; checked on resume
    aux: Mapping[str, str] = field(default_factory=dict)  # aux name -> refdata dataset (rows by trade_date)
    # A run frozen with ``data.warmup``: (that rule, warm-up bars). The loader counts them in the rule's session
    # calendar back from ``engine.start``, as ``v3.compiler.compile_spec`` does for the backtest.
    warmup: tuple[dict, int] | None = None
    # A run frozen with ``data.trading_calendar``: the schedule appended after the cutoff skips the unscheduled
    # closures already detected, as ``v3.compiler.compile_spec`` projects for the backtest.
    calendar_rule: Mapping | None = None
    # Bought-option overlays (a run frozen with ``data.option_chain``): each chain is read from refdata
    # (``taifex_txo_daily`` day-session monthly puts) and the cutoff waits for it (``load_inputs``).
    hedges: tuple[HedgePolicy, ...] = ()


class StrategyNotCompiled(Exception):
    """The run cannot be turned into a target strategy (yet)."""


def compile_strategy(ctx) -> CompiledStrategy:
    """Step 4 fills this in: the run's frozen v3 spec (``ctx.input_spec_json`` / spec stages) with
    optimize's best params -> registry components -> ``CompiledStrategy``."""

    raise StrategyNotCompiled("daily shadow paper needs compile_strategy (spec v3 -> engine), wired in step 4")


@dataclass(frozen=True)
class PaperInputs:
    futures: dict[str, FuturesInput]
    aux: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    cutoff: pd.Timestamp
    schedule: MarketSchedule
    day_hashes: dict[str, str]
    history_sha256: str
    sources: dict[str, str]
    last_dates: dict[str, str | None]
    closures: pd.DatetimeIndex
    options: dict[str, OptionInput] = field(default_factory=dict)


def _iso(value) -> str | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).date().isoformat()


# Futures bars carry the first five; a stock's bars (``engine.data.stock_input``) the rest as well, so a
# corporate action booked late or revised shows up as a revision of that day.
HASHED_BAR_COLUMNS = ("open", "high", "low", "close", "settle", "traded", "factor", "unit_ratio", "cash_per_share")


def input_hashes(
    futures: Mapping[str, FuturesInput],
    aux: Mapping[str, pd.DataFrame],
    paper_start: date,
    options: Mapping[str, OptionInput] | None = None,
):
    """(sha256 per trading day >= paper start, one sha256 of everything before it).

    An option chain adds its quote rows by date and each final settlement on its day; a run without
    hedges hashes exactly what it did before.
    """

    digests: dict[str, hashlib._Hash] = defaultdict(hashlib.sha256)
    history = hashlib.sha256()
    start = pd.Timestamp(paper_start)

    def feed(day: pd.Timestamp, line: str) -> None:
        (history if day < start else digests[day.date().isoformat()]).update(line.encode())

    for name in sorted(futures):
        bars = futures[name].bars.sort_values(["date", "contract"], kind="mergesort")
        columns = [c for c in HASHED_BAR_COLUMNS if c in bars.columns]
        for row in bars[["date", "contract", *columns]].itertuples(index=False, name=None):
            feed(pd.Timestamp(row[0]), f"{name}|{row[1]}|{'|'.join(repr(float(v)) for v in row[2:])}\n")
    for name in sorted(aux):
        frame = aux[name]
        for stamp, *values in frame.itertuples(name=None):
            feed(pd.Timestamp(stamp), f"aux:{name}|{'|'.join(repr(float(v)) for v in values)}\n")
    for name in sorted(options or {}):
        chain = options[name]
        quotes = chain.quotes.sort_values(["date", "contract_month", "cp", "strike"], kind="mergesort")
        columns = ["date", "contract_month", "cp", "strike", "settlement"]
        for day, month, cp, strike, settlement in quotes[columns].itertuples(index=False, name=None):
            expiry = pd.Timestamp(chain.expiries[month]).date().isoformat()
            feed(pd.Timestamp(day), f"opt:{name}|{month}|{expiry}|{cp}|{float(strike)!r}|{float(settlement)!r}\n")
        for month, (day, price) in sorted(chain.final_settlement.items()):
            feed(pd.Timestamp(day), f"fsp:{name}|{month}|{float(price)!r}\n")
    return {day: digest.hexdigest() for day, digest in sorted(digests.items())}, history.hexdigest()


def load_inputs(reader: RefdataReader, compiled: CompiledStrategy, settings: dict, paper_start: date) -> PaperInputs:
    """Read refdata and cut it at the last complete trading day. Raises ``RefdataError``.

    The cutoff is the last day every input has: the last day session of each futures product, the
    last bar of each stock, the last row of each aux series, and for stocks the day the corporate
    actions dataset is complete through (stale actions make the paper wait, as a lagged price series
    does). The trading calendar is the futures
    products' common day-session dates, or, with no futures, the stocks' union of bar dates; a stock
    without a bar on a calendar day is carried forward (``align_stock_bars``), and its corporate
    actions are booked from ``tw_corporate_actions``, exactly as ``v3.compiler.compile_spec`` does for
    the backtest; a store without that dataset raises (the paper waits, it never books raw prices).

    A run with hedges also waits for its option chain (``_option_bound``): the last TXO day session,
    a paper day without TXO quotes, a monthly expiry on a paper day without its final settlement
    price. A put leg is never booked on a day the chain cannot price.
    """

    schedule = MarketSchedule.from_frame(reader.read(MARKET_CALENDAR))
    futures_insts = {k: i for k, i in compiled.instruments.items() if i.spec.kind == "future"}
    stock_insts = {k: i for k, i in compiled.instruments.items() if i.spec.kind == "stock"}
    other = {k: i.spec.kind for k, i in compiled.instruments.items() if k not in futures_insts and k not in stock_insts}
    if other:
        raise ValueError(f"the daily paper has no refdata loader for instrument kind(s) {other}")
    products = sorted({inst.product for inst in futures_insts.values()})
    data_start = compiled.data_start
    if compiled.warmup is not None:
        from poseidon.pipeline.v3.config import session_warmup

        record, bars = compiled.warmup
        data_start = session_warmup(record, reader).load_from(pd.Timestamp(compiled.engine.start).date(), bars)
    quotes = reader.read(FUTURES, start=data_start) if products else pd.DataFrame()
    last = last_day_session(quotes, products) if products else {}
    stock_rows = {k: read_bars(reader, i.spec.symbol, start=data_start) for k, i in stock_insts.items()}
    aux_full = {
        name: align_signal_frame(name, reader.read(dataset, start=data_start)) for name, dataset in compiled.aux.items()
    }
    last_dates = {f"{FUTURES}:{p}": _iso(day) for p, day in last.items()}
    stock_last = {k: pd.Timestamp(rows["date"].max()) if len(rows) else None for k, rows in stock_rows.items()}
    last_dates.update({f"{STOCKS}:{stock_insts[k].spec.symbol}": _iso(day) for k, day in stock_last.items()})
    last_dates.update(
        {f"aux:{name}": _iso(frame.index.max()) if len(frame) else None for name, frame in aux_full.items()}
    )
    ends = [*last.values(), *stock_last.values()]
    ends += [complete_through(name, frame) for name, frame in aux_full.items()]
    if stock_insts:  # raw bars need the actions; one dated after the snapshot is not booked, so wait for it
        actions = require_actions(reader)
        actions_end = pd.Timestamp(actions_complete_through(reader))
        ends.append(actions_end)
        last_dates[f"{ACTIONS}:{COMPLETE_THROUGH}"] = _iso(actions_end)
    chains = sorted({hedge.chain for hedge in compiled.hedges})
    if chains:
        from poseidon.pipeline.v3.compiler import option_quote_rows

        option_quotes = reader.read(OPTION_QUOTES, start=data_start, rows=option_quote_rows)
        option_final = reader.read(OPTION_FINAL)
        option_last = pd.Timestamp(option_quotes["trade_date"].max()) if len(option_quotes) else None
        last_dates[OPTION_QUOTES] = _iso(option_last)
        ends.append(option_last)
    if any(day is None for day in ends):
        raise RefdataError(f"no data for {sorted(k for k, v in last_dates.items() if v is None)}")
    cutoff = pd.Timestamp(min(ends)).normalize()
    common = None
    for product in products:
        dates = day_session_dates(product_quotes(quotes, product))
        common = dates if common is None else common.intersection(dates)
    if common is None:
        common = pd.DatetimeIndex(sorted({pd.Timestamp(d) for rows in stock_rows.values() for d in rows["date"]}))
    if chains:
        cutoff = _option_bound(option_quotes, option_final, common, cutoff, paper_start)
        last_dates[f"{OPTION_QUOTES}:{COMPLETE_THROUGH}"] = _iso(cutoff)
    closures = unscheduled_closures(product_quotes(quotes, products[0]), schedule) if products else pd.DatetimeIndex([])
    calendar = trading_calendar(common, schedule, cutoff, int(settings["calendar_horizon_days"]))
    if compiled.calendar_rule is not None:
        calendar = calendar[~calendar.isin(closures)]
    futures = {
        name: futures_input(
            quotes,
            reader.read(CALENDAR),
            schedule,
            product=inst.product,
            spec=inst.spec,
            through=cutoff,
            roll_days_before=inst.roll_days_before,
        )
        for name, inst in futures_insts.items()
    }
    if stock_insts:
        data_days = calendar[calendar <= cutoff]
        for name, inst in stock_insts.items():
            symbol = inst.spec.symbol
            rows = stock_rows[name]
            try:
                aligned = align_stock_bars(rows[rows["date"] <= cutoff], data_days, name)
            except StockDataGap as error:
                raise RefdataError(str(error)) from error
            mine = actions[actions["symbol"] == symbol]
            try:
                futures[name] = stock_input(inst.spec, aligned, mine)
            except ValueError as error:
                raise RefdataError(f"{name}: {error}") from error
    aux = {name: frame[frame.index <= cutoff] for name, frame in aux_full.items()}
    options = {}
    if chains:
        from poseidon.pipeline.v3.compiler import option_chain

        rows = option_quotes[option_quotes["trade_date"] <= cutoff]
        final = option_final[option_final["final_settlement_date"] <= cutoff]
        options = {symbol: option_chain(symbol, rows, final) for symbol in chains}
    hashes, history = input_hashes(futures, aux, paper_start, options)
    datasets = [*((FUTURES, CALENDAR) if products else ()), MARKET_CALENDAR, *sorted(set(compiled.aux.values()))]
    if stock_insts:
        datasets += [STOCKS, *((ACTIONS,) if reader.exists(ACTIONS) else ())]
    if chains:
        datasets += [OPTION_QUOTES, OPTION_FINAL]
    sources = {name: reader.content_sha256(name) for name in datasets}
    return PaperInputs(
        futures, aux, calendar, cutoff, schedule, hashes, history, sources, last_dates, closures, options
    )


def _option_bound(quotes: pd.DataFrame, final: pd.DataFrame, days, cutoff: pd.Timestamp, paper_start: date):
    """``cutoff``, or the trading day before the first paper day the chain cannot price: a day of ``days`` with
    no quote row, or a monthly expiry without its final settlement price (a held put would stay marked at its
    last settlement instead of being cash-settled)."""

    span = days[(days >= pd.Timestamp(paper_start)) & (days <= cutoff)]
    gaps = span[~span.isin(pd.DatetimeIndex(quotes["trade_date"].unique()))]
    settled = set(final.loc[final["expiry_kind"] == "monthly", "contract_month"])
    expiries = quotes.drop_duplicates("contract_month").set_index("contract_month")["expiry_date"]
    unsettled = expiries[~expiries.index.isin(settled) & expiries.isin(span)]
    blocked = [*gaps[:1], *unsettled.sort_values()[:1]]
    if not blocked:
        return cutoff
    first = min(blocked)
    before = days[days < first]
    return before[-1] if len(before) else first - pd.Timedelta(days=1)
