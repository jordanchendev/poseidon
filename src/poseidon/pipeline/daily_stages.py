"""Spec v3 daily runs (CONFIG_VERSION 6, engine ``target_v1``) as pipeline stages (design 2026-10-24 §2, §7, §10 step 4).

Plan (``plan.V3_DAILY_STAGE_PLAN``)::

    spec_validate -> backtest -> optimize -> param_stability -> dsr -> cost_stress -> regime_split
    -> benchmark -> execution_timing -> wfe -> holdout -> shadow_paper (daily_target) -> awaiting_release

Every stage reads only the run's frozen config and its upstream artifacts; thresholds come from
``config["thresholds"]`` (the class table entry frozen at creation), gate settings from
``config["gates"]``. The registry routes a stage name to these bodies only for a v3 run
(``stages.EngineRouted``); a CONFIG_VERSION <= 5 run never reaches this module.

Data: refdata only (``taifex_futures_daily`` + ``taifex_settlement_calendar`` through
``v3.compiler.compile_spec``; ``twse_taiex_*`` aux). Train stages read through a :class:`WindowStore`
whose price datasets end the day before ``data.holdout_start``: no train stage can load a holdout
bar. ``holdout`` reads up to the day before ``data.end``, once, after every train gate passed.

Metric basis per class (``gates.daily.basis_for``): ``daily_equity`` judges daily returns,
``daily_fixed`` daily TWD PnL. Trials go into the family ledger with ``gates.daily.moments`` of that
basis, so DSR compares like with like (and a family never mixes classes: ``service.create``).

Stage reports follow ``gate_stages``: ``{"summary", "gate", "verdict", ..., "gate_report", "inputs"}``
for gates, and a ``summary`` line on every other stage; Kairos renders them generically.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline import optimization as opt
from poseidon.pipeline.engine import ContractSpec, EngineResult, OptionSpec
from poseidon.pipeline.engine.data import stock_bars
from poseidon.pipeline.gate_stages import DsrStage, ParamStabilityStage, to_stage_result
from poseidon.pipeline.gates import benchmark, cost_stress, execution_timing, holdout, liquidation, regime_split
from poseidon.pipeline.gates import wfe as wfe_gate
from poseidon.pipeline.gates.base import failed, finite_or_none, invalid
from poseidon.pipeline.gates.daily import (
    LIQUIDATION_EVENTS,
    Basis,
    DailyResult,
    RerunRequest,
    basis_for,
    choose_walk_forward,
    from_engine,
    moments,
    underlying_returns,
)
from poseidon.pipeline.gates.param_stability import EDGE_AT_BOUND_KEY
from poseidon.pipeline.plan import V3_GATE_STAGES
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, TAIEX, TAIEX_TR, VIX
from poseidon.pipeline.refdata.engine_input import StockDataGap, align_stock_bars
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_corporate_actions import load_actions
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.stages import BaseStage, StageResult
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.trials import TRIAL_COMPLETE, TRIAL_FAIL, TRIAL_PRUNED, MemoryTrialLedger
from poseidon.pipeline.v3.compiler import OPTION_DATASETS, ROW_FILTERS, NotCompilable, compile_spec, hedged
from poseidon.pipeline.v3.config import ENGINE, SPEC_VALIDATOR, session_warmup, used_index_datasets
from poseidon.pipeline.v3.runtime import Conventions
from poseidon.pipeline.v3.spec import substitute
from poseidon.pipeline.v3.validation import FeasibilityContext, SessionWarmup, validate_spec_v3

V3_WIRING_VERSION = 1  # part of every v3 stage's input hash
# clipped at the window's last day; calendars stay whole
PRICE_DATASETS = frozenset({FUTURES, TAIEX, TAIEX_TR, VIX, STOCKS, ACTIONS, *OPTION_DATASETS})
STOCK_UNDERLYING = "0050"  # DAILY_GATES.benchmark.underlying.tw_stock: 0050 total return
UNDERLYING_ROLL_DAYS_BEFORE = 1  # DAILY_GATES.benchmark.underlying: TX back-adjusted, roll before settlement
TRADING_DAYS_PER_YEAR = 252.0
EXECUTION_TIMING = "execution_timing"


class ComponentDrift(RuntimeError):
    """A registry component's source no longer matches the fingerprint frozen into the run (§3.6)."""


# ----------------------------------------------------------------------------- frozen window


@dataclass(frozen=True)
class DailyWindow:
    """``config["data"]`` of a daily run: ``[start, end)`` trading days, holdout from ``holdout_start``."""

    start: date
    end: date  # exclusive
    holdout_start: date

    @classmethod
    def from_config(cls, config: dict) -> DailyWindow:
        data = config["data"]
        window = cls(
            date.fromisoformat(data["start"]),
            date.fromisoformat(data["end"]),
            date.fromisoformat(data["holdout_start"]),
        )
        if not window.start < window.holdout_start < window.end:
            raise ValueError("holdout_start must fall strictly inside the data window")
        return window

    @property
    def train_last(self) -> date:
        return self.holdout_start - timedelta(days=1)

    @property
    def holdout_last(self) -> date:
        return self.end - timedelta(days=1)

    def describe(self, part: str) -> dict:
        first, last = (self.start, self.train_last) if part == "train" else (self.holdout_start, self.holdout_last)
        return {"part": part, "first_day": first.isoformat(), "last_day": last.isoformat(), "inclusive": True}


class WindowStore:
    """Read-through cache over a refdata store whose price datasets end at ``upper`` (inclusive).

    Each dataset is read once (``store.read(..., end=upper)`` for prices, whole for calendars) and
    filtered in memory, so many engine runs over one window cost one read (a dataset in
    ``compiler.ROW_FILTERS`` through its filter, whatever ``rows`` a caller passes). ``data_sha256``
    hashes exactly what was read: the trial ledger's window identity. ``kinds`` (the spec's
    instrument kinds, :func:`instrument_kinds`) chooses the datasets that identity covers: the
    futures datasets for a futures spec (as before step 8), the stock bars and corporate actions
    for a spec that trades a stock / ETF; the index series whenever the store has them. ``extra``
    names the optional datasets the spec reads beyond those (``index:vix``, a hedge's
    ``compiler.OPTION_DATASETS``).
    """

    def __init__(
        self,
        store,
        upper: date | None,
        kinds: frozenset[str] = frozenset({"future"}),
        extra: tuple[str, ...] = (),
    ):
        self._store = store
        self.upper = upper
        self.kinds = kinds
        self._extra = extra  # hashed into data_sha256 as well
        self._frames: dict[str, pd.DataFrame] = {}
        self._columns: dict[str, str] = {}

    def _full(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._frames:
            clip = self.upper if dataset in PRICE_DATASETS else None
            rows = ROW_FILTERS.get(dataset)
            self._frames[dataset] = (
                self._store.read(dataset, end=clip) if rows is None else self._store.read(dataset, end=clip, rows=rows)
            )
            self._columns[dataset] = self._store.manifest(dataset)["date_column"]
        return self._frames[dataset]

    def read(self, dataset: str, start=None, end=None, rows=None) -> pd.DataFrame:
        frame = self._full(dataset)
        if not len(frame):
            return frame
        column = frame[self._columns[dataset]]
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= column >= pd.Timestamp(start)
        if end is not None:
            mask &= column <= pd.Timestamp(end)
        return frame[mask].reset_index(drop=True)

    def manifest(self, dataset: str) -> dict:
        return self._store.manifest(dataset)

    def content_sha256(self, dataset: str) -> str:
        return self._store.content_sha256(dataset)

    def exists(self, dataset: str) -> bool:
        return self._store.exists(dataset)

    def data_sha256(self) -> str:
        from poseidon.pipeline.refdata.store import frame_sha256

        unknown = sorted(self.kinds - {"future", "stock"})
        if unknown:
            raise NotCompilable(f"no refdata window for instrument kind(s) {unknown}")
        names = [FUTURES, CALENDAR] if "future" in self.kinds else []
        names += sorted(name for name in (TAIEX, TAIEX_TR) if self._store.exists(name))
        if "stock" in self.kinds:
            names += [STOCKS, *((ACTIONS,) if self._store.exists(ACTIONS) else ())]
        names += list(self._extra)
        return content_sha256(
            {
                "upper": self.upper.isoformat() if self.upper else None,
                "frames": {name: frame_sha256(self._full(name)) for name in names},
            }
        )


# ----------------------------------------------------------------------------- engine runs


def daily_metrics(result: DailyResult, basis: Basis) -> dict:
    """``moments`` of the class basis (the trial-ledger keys) plus the headline numbers of a run."""

    equity = np.asarray(result.equity, dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[result.initial_equity], equity])) if len(equity) else np.array([1.0])
    drawdown = float(np.max(1.0 - np.concatenate([[result.initial_equity], equity]) / peak)) if len(equity) else 0.0
    years = result.n_days / TRADING_DAYS_PER_YEAR
    cagr = None
    if years > 0 and result.final_equity > 0 and result.initial_equity > 0:
        cagr = (result.final_equity / result.initial_equity) ** (1.0 / years) - 1.0
    return {
        **moments(result.metric(basis)),
        "basis": basis,
        "total_pnl_twd": finite_or_none(result.total_pnl_twd),
        "total_return": finite_or_none(result.total_return),
        "final_equity_twd": finite_or_none(result.final_equity),
        "initial_equity_twd": finite_or_none(result.initial_equity),
        "cagr": finite_or_none(cagr),
        "max_drawdown_pct": finite_or_none(drawdown),
        "n_days": result.n_days,
        "first_day": result.dates[0].isoformat() if result.dates else None,
        "last_day": result.dates[-1].isoformat() if result.dates else None,
        "exposure_changes": result.exposure_changes,
        "costs": result.costs.as_report(),
        "forced_liquidations": len(result.event_days(["forced_liquidation"])),
        "margin_calls": len(result.event_days(["margin_call"])),
        "ruined": result.ruined.isoformat() if result.ruined is not None else None,
    }


def instrument_kinds(spec) -> frozenset[str]:
    """Kinds of the registered instruments a spec document declares (unknown symbols are left out)."""

    from poseidon.pipeline.v3.instruments import INSTRUMENTS

    declared = spec.get("instruments") if isinstance(spec, dict) else None
    if not isinstance(declared, dict):
        return frozenset()
    rows = (INSTRUMENTS.get(decl.get("symbol")) for decl in declared.values() if isinstance(decl, dict))
    return frozenset(row.kind for row in rows if row is not None)


def scaled_costs(spec: ContractSpec, multiplier: float) -> ContractSpec:
    if multiplier == 1.0:
        return spec
    return dataclasses.replace(
        spec,
        commission_per_unit=spec.commission_per_unit * multiplier,
        commission_rate=spec.commission_rate * multiplier,
        tax_rate=spec.tax_rate * multiplier,
        slippage_ticks=spec.slippage_ticks * multiplier,
    )


def scaled_option_costs(spec: OptionSpec, multiplier: float) -> OptionSpec:
    return dataclasses.replace(
        spec,
        commission_per_unit=spec.commission_per_unit * multiplier,
        tax_rate=spec.tax_rate * multiplier,
        slippage_ticks=spec.slippage_ticks * multiplier,
        slippage_pct=spec.slippage_pct * multiplier,
    )


@dataclass
class SpecRunner:
    """Engine runs of one spec template over one window ``[start, end]`` (both inclusive).

    ``bind`` substitutes a grid point (spec_validate already validated the whole declared grid);
    an off-grid point (grid-edge extension) must pass :meth:`admissible` first, which validates the
    template widened to contain it (``optimization.effective_template``) with the run's frozen
    feasibility context. Each run compiles afresh (the warm-up depends on the params) from the
    cached :class:`WindowStore`.
    """

    template: dict
    grid: opt.ParamGrid
    store: WindowStore
    start: date
    end: date
    basis: Basis
    feasibility: FeasibilityContext = field(default_factory=FeasibilityContext)
    calendar_rule: Mapping | None = None  # the run's frozen data.trading_calendar (compile_spec)

    def bind(self, values: Mapping) -> dict:
        snapped = {name: self.grid.specs[name].cast(values[name]) for name in self.grid.names}
        return substitute(self.template, snapped)

    def admissible(self, values: Mapping) -> str | None:
        effective = opt.effective_template(self.template, self.grid, dict(values))
        if effective is self.template:
            return None
        result = validate_spec_v3(effective, feasibility=self.feasibility)
        if result.ok:
            return None
        first = (result.errors or result.unsupported)[0]
        text = first.text()[:300]
        return opt.BoundReason(text) if first in result.errors and first.type in opt.BOUND_ERROR_TYPES else text

    def compile(self, values: Mapping, *, conventions: Conventions = Conventions(), cost_multiplier: float = 1.0):
        compiled = compile_spec(
            self.bind(values),
            start=self.start,
            end=self.end,
            store=self.store,
            conventions=conventions,
            sessions=self.feasibility.warmup,
            calendar_rule=self.calendar_rule,
        )
        if cost_multiplier != 1.0:
            futures = {
                name: dataclasses.replace(f, spec=scaled_costs(f.spec, cost_multiplier))
                for name, f in compiled.futures.items()
            }
            options = {
                name: dataclasses.replace(o, spec=scaled_option_costs(o.spec, cost_multiplier))
                for name, o in compiled.options.items()
            }
            compiled = dataclasses.replace(compiled, futures=futures, options=options)
        return compiled

    def run(self, values: Mapping, **kwargs) -> tuple[EngineResult, DailyResult]:
        result = self.compile(values, **kwargs).run()
        return result, from_engine(result)

    def rerun(self, request: RerunRequest) -> DailyResult:
        """``gates.daily.Rerun``: costs x ``cost_multiplier``; ``execution`` / ``signal_lag_extra`` as Conventions."""

        conventions = Conventions(execution=request.execution, signal_lag=request.signal_lag_extra)
        return self.run(request.params, conventions=conventions, cost_multiplier=float(request.cost_multiplier))[1]


def engine_files(result: EngineResult, prefix: str = "") -> dict[str, bytes]:
    """§5.8 frames as CSV files."""

    out = {}
    for name in ("fills", "positions_daily", "daily_pnl", "events"):
        frame = getattr(result, name)
        out[f"{prefix}{name}.csv"] = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return out


class EngineTrialRunner(opt.TrialRunner):
    """``optimization.TrialRunner`` whose trials are target-engine runs of a :class:`SpecRunner`."""

    def __init__(self, spec_runner: SpecRunner, *, ledger, window_sha256, should_stop, check_stop, first_trial_no=1):
        super().__init__(
            build=None,
            bars=None,
            backtest_config={},
            ledger=ledger,
            window_sha256=window_sha256,
            should_stop=should_stop,
            check_stop=check_stop,
            first_trial_no=first_trial_no,
        )
        self.spec_runner = spec_runner
        self.daily: dict[str, DailyResult] = {}  # grid key -> DailyResult (small: dates + equity + costs)

    def evaluate(self, params: dict, source: str):
        self.check_stop()
        try:
            engine_result, daily = self.spec_runner.run(params)
        except (NotCompilable, ValueError) as error:
            return self.record(params, source, TRIAL_FAIL, None, f"{type(error).__name__}: {error}"[:500])
        self.daily[self.spec_runner.grid.key(params)] = daily
        metrics = daily_metrics(daily, self.spec_runner.basis)
        state = TRIAL_COMPLETE if metrics.get("sharpe_daily") is not None else TRIAL_PRUNED
        return self.record(params, source, state, metrics, outcome=engine_result)


# ----------------------------------------------------------------------------- stage plumbing


def check_components(config: dict) -> dict:
    """Live component sources must match the fingerprints frozen into the run (``name@version`` immutable)."""

    from poseidon.pipeline.v3.registry import component_fingerprints

    frozen = config.get("components") or {}
    try:
        live = component_fingerprints(sorted(frozen))
    except KeyError as error:
        raise ComponentDrift(f"component {error} frozen into this run is no longer registered") from error
    if live != frozen:
        changed = sorted(set(frozen.items()) ^ set(live.items()))
        raise ComponentDrift(
            f"registry components changed since the run was frozen ({', '.join(sorted({k for k, _ in changed}))}); "
            "a behaviour change needs a new component version and a new run"
        )
    return frozen


def frozen_series_since(config: dict) -> dict[str, date] | None:
    """The series first days frozen with a window start that moved (``data.start_rule``), else None."""

    rule = (config.get("data") or {}).get("start_rule")
    if not rule:
        return None
    return {key: date.fromisoformat(day) for key, day in rule["series_since"].items()}


def moved_start_text(config: dict) -> str | None:
    """``", window from <start> (<series> data from <day> + warm-up)"`` for a start that moved, else None."""

    data = config.get("data") or {}
    rule = data.get("start_rule")
    if not rule:
        return None
    binding = rule["binding_series"]
    return f", window from {data['start']} ({binding} data from {rule['series_since'][binding]} + warm-up)"


def feasibility_from(report: dict, warmup: SessionWarmup | None = None) -> FeasibilityContext:
    block = report.get("feasibility") or {}
    prices = {name: tuple(values) for name, values in (block.get("prices") or {}).items()}
    series = {key: date.fromisoformat(day) for key, day in (block.get("series_since") or {}).items()}
    return FeasibilityContext(
        start=date.fromisoformat(block["start"]) if block.get("start") else None,
        end=date.fromisoformat(block["end"]) if block.get("end") else None,
        prices=prices or None,
        series_since=series or None,
        warmup=warmup,
    )


def summary_text(name: str, ok: bool, text: str) -> str:
    return f"{name} {'✅' if ok else '❌'} {text}".rstrip()


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


class DailyStage(BaseStage):
    """A v3 daily stage: frozen-component check, refdata access, the v3 wiring version in the input hash."""

    lease_seconds: ClassVar[int] = 1800

    def __init__(self, *, refdata: Callable[[], Any] | None = None):
        self.refdata = refdata  # () -> RefdataStore-like (tests pass a synthetic store)

    def store(self):
        if self.refdata is not None:
            return self.refdata()
        from poseidon.pipeline.refdata.store import RefdataStore

        return RefdataStore()

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update({"engine": ENGINE, "v3_wiring_version": V3_WIRING_VERSION})
        return payload

    def run(self, ctx):
        check_components(ctx.config)
        try:
            return self.evaluate(ctx)
        except NotCompilable as error:
            text = str(error)
            return StageResult.unsupported(
                text, {"summary": f"{self.name} ⛔ not runnable on the daily engine: {text}"}
            )

    def evaluate(self, ctx) -> StageResult:  # pragma: no cover - interface
        raise NotImplementedError

    # ---- shared helpers

    @staticmethod
    def basis(ctx) -> Basis:
        return basis_for(ctx.config["strategy_class"])

    def validated(self, ctx) -> dict:
        return ctx.upstream["spec_validate"].report or {}

    def spec_runner(self, ctx, part: str, *, start: date | None = None, end: date | None = None) -> SpecRunner:
        window = DailyWindow.from_config(ctx.config)
        validated = self.validated(ctx)
        upper = window.train_last if part == "train" else window.holdout_last
        first = window.start if part == "train" else window.holdout_start
        start, end = start or first, end or upper
        if end > upper:
            raise ValueError(f"{self.name} may not read {part} data after {upper}")
        template = validated["template"]
        store = WindowStore(
            self.store(),
            upper,
            instrument_kinds(template),
            used_index_datasets(template) + (OPTION_DATASETS if hedged(template) else ()),
        )
        feasibility = feasibility_from(validated, session_warmup(ctx.config["data"].get("warmup"), store))
        params = validate_spec_v3(template, feasibility=feasibility).params
        return SpecRunner(
            template=template,
            grid=opt.ParamGrid(params),
            store=store,
            start=start,
            end=end,
            basis=self.basis(ctx),
            feasibility=feasibility,
            calendar_rule=ctx.config["data"].get("trading_calendar"),
        )

    @staticmethod
    def best(ctx) -> dict:
        return (ctx.upstream["optimize"].report or {}).get("best") or {}

    @staticmethod
    def best_source(ctx) -> dict:
        report = ctx.upstream["optimize"].report or {}
        best = report.get("best") or {}
        return {
            "best_trial_no": best.get("trial_no"),
            "best_params": best.get("params"),
            "best_spec_sha256": report.get("best_spec_sha256"),
        }

    def train_best(self, ctx, runner: SpecRunner | None = None) -> tuple[SpecRunner, DailyResult]:
        runner = runner or self.spec_runner(ctx, "train")
        return runner, runner.run(self.best(ctx)["params"])[1]


def window_sha(store_sha: str, part: str, start: date, end: date, **extra) -> str:
    return content_sha256(
        {"engine": ENGINE, "data_sha256": store_sha, "part": part, "first": start.isoformat(), "last": end.isoformat()}
        | extra
    )


# ----------------------------------------------------------------------------- spec_validate


def price_points(store, row, first: date, last: date) -> tuple[float, float] | None:
    """(close on the first day, max close) of instrument ``row``'s price source in the window: the
    nearest monthly contract of a future, the stock's own daily close."""

    if row.kind == "stock":
        closes = read_bars(store, row.symbol, start=first, end=last)["close"]
        return (float(closes.iloc[0]), float(closes.max())) if len(closes) else None
    symbol = row.price_proxy or row.symbol
    quotes = store.read(FUTURES, start=first, end=last)
    rows = quotes[
        (quotes["product"] == symbol)
        & (quotes["session"] == "day")
        & quotes["contract_month"].astype(str).str.fullmatch(r"\d{6}")
    ]
    if not len(rows):
        return None
    close = rows["close"].fillna(rows["settlement"])
    front = rows.assign(close=close).sort_values(["trade_date", "contract_month"]).groupby("trade_date").first()
    return float(front["close"].iloc[0]), float(front["close"].max())


class DailySpecValidateStage(DailyStage):
    """§3.7 in order on the run's spec, with the run's data window for feasibility (train part only).

    The spec is ``input_spec_json`` (hand-written / Sisyphus-confirmed), or, for an idea-only run routed
    here (``spec_source`` = spec_generate, CONFIG_VERSION 7), the spec spec_generate v3 wrote.
    """

    name = "spec_validate"
    requires: ClassVar[tuple[str, ...]] = ()
    lease_seconds = 300

    @staticmethod
    def generated(config) -> bool:
        from poseidon.pipeline.v3.config import SPEC_SOURCE_GENERATED

        return (config or {}).get("spec_source") == SPEC_SOURCE_GENERATED

    def upstream_names(self, config) -> tuple[str, ...]:
        return ("spec_generate",) if self.generated(config) else super().upstream_names(config)

    def spec(self, ctx):
        if self.generated(ctx.config):
            return (ctx.upstream["spec_generate"].report or {}).get("spec")
        return ctx.input_spec_json

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update({"validator": SPEC_VALIDATOR, "input_spec_sha256": content_sha256(self.spec(ctx))})
        return payload

    def evaluate(self, ctx):
        from poseidon.pipeline.v3.instruments import INSTRUMENTS

        spec = self.spec(ctx)
        window = DailyWindow.from_config(ctx.config)
        kinds = instrument_kinds(spec)
        store = WindowStore(self.store(), window.train_last, kinds)
        prices: dict[str, list[float]] = {}
        if isinstance(spec, dict) and isinstance(spec.get("instruments"), dict):
            for decl in spec["instruments"].values():
                row = INSTRUMENTS.get(decl.get("symbol")) if isinstance(decl, dict) else None
                if row is None or row.kind not in ("future", "stock"):
                    continue
                if row.kind == "stock" and not store.exists(STOCKS):
                    continue
                source = row.price_proxy or row.symbol
                points = price_points(store, row, window.start, window.train_last)
                if points is not None:
                    prices[source] = list(points)
        series_since = frozen_series_since(ctx.config)
        context = FeasibilityContext(
            start=window.start,
            end=window.end,
            prices={k: tuple(v) for k, v in prices.items()} or None,
            series_since=series_since,
            warmup=session_warmup(ctx.config["data"].get("warmup"), store),
        )
        result = validate_spec_v3(spec, feasibility=context)
        sources = {
            "future": "refdata taifex_futures_daily, nearest monthly contract's day close",
            "stock": f"refdata {STOCKS}, the stock's own daily close (unadjusted)",
        }
        feasibility = {
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "prices": prices,
            "prices_source": "; ".join(sources[kind] for kind in sorted(kinds) if kind in sources)
            + ": (first train day, max over the train part); the holdout part is not read",
        }
        if series_since is not None:
            feasibility["series_since"] = {key: day.isoformat() for key, day in sorted(series_since.items())}
        report = {
            "validator": SPEC_VALIDATOR,
            "spec_sha256": content_sha256(spec),
            "errors": [issue.text() for issue in result.errors],
            "unsupported": [issue.text() for issue in result.unsupported],
            "notes": list(result.notes),
            "feasibility": feasibility,
        }
        if EDGE_AT_BOUND_KEY in ctx.config["thresholds"] and result.warnings:
            report["warnings"] = list(result.warnings)  # runs frozen without the key keep today's report
        if result.errors:
            report["summary"] = f"spec_validate ❌ {len(result.errors)} error(s): {result.errors[0].text()[:160]}"
            return StageResult.failed(f"spec invalid: {len(result.errors)} error(s)", report)
        if result.unsupported:
            text = "; ".join(report["unsupported"])
            report["summary"] = f"spec_validate ⛔ unsupported: {text[:160]}"
            return StageResult.unsupported(text, report)
        if result.strategy_class != ctx.config["strategy_class"]:
            report["summary"] = "spec_validate ❌ strategy class differs from the frozen one"
            return StageResult.failed(
                f"spec is {result.strategy_class}, the run was frozen as {ctx.config['strategy_class']}", report
            )
        grid = opt.ParamGrid(result.params)
        report.update(
            {
                "strategy_class": result.strategy_class,
                "spec": result.resolved,
                "resolved_spec_sha256": content_sha256(result.resolved),
                "template": result.template,
                "params": grid.describe(),
                "grid_size": grid.size() if grid.names else 1,
                "summary": f"spec_validate ✅ {result.strategy_class}, {len(grid.names)} $params "
                f"({grid.size() if grid.names else 1} grid points)"
                + (f", {len(result.notes)} note(s)" if result.notes else "")
                + (f", {len(report['warnings'])} warning(s)" if report.get("warnings") else "")
                + (moved_start_text(ctx.config) or ""),
            }
        )
        return StageResult.passed(report, reason=report["summary"])


# ----------------------------------------------------------------------------- backtest


class DailyBacktestStage(DailyStage):
    """The spec at its defaults on the train part; liquidation gate; truncation audit (no look-ahead)."""

    name = "backtest"
    requires: ClassVar[tuple[str, ...]] = ("spec_validate",)

    def evaluate(self, ctx):
        thresholds = ctx.config["thresholds"]
        runner = self.spec_runner(ctx, "train")
        defaults = {name: spec.cast(spec.default) for name, spec in runner.grid.specs.items()}
        compiled = runner.compile(defaults)
        result = compiled.run()
        daily = from_engine(result)
        metrics = daily_metrics(daily, runner.basis)
        verdict = liquidation.evaluate(daily, thresholds)
        cuts = int(ctx.config["backtest"]["truncation_audit_cuts"])
        ctx.check_stop()
        audit = compiled.audit(n_cuts=cuts, seed=int(ctx.config.get("seed") or 0))
        audit_report = {
            "n_cuts": cuts,
            "cuts": [cut.date().isoformat() for cut in audit.cuts],
            "mismatches": list(audit.mismatches)[:20],
            "ok": audit.ok,
            "proof": "decisions, equity and held units on or before each cut identical to the full run",
        }
        report = {
            "spec_sha256": content_sha256(runner.bind(defaults)),
            "params": defaults,
            "window": DailyWindow.from_config(ctx.config).describe("train"),
            "data_sha256": runner.store.data_sha256(),
            "refdata": compiled.inputs,
            "engine": {
                "execution": compiled.config.execution,
                "initial_capital": compiled.config.initial_capital,
                "leverage_cap": compiled.config.leverage_cap,
                "max_margin_utilisation": compiled.config.max_margin_utilisation,
                "liquidation_risk_ratio": compiled.config.liquidation_risk_ratio,
            },
            "metrics": metrics,
            "liquidation": {"verdict": verdict.verdict.value, "reason_code": verdict.reason_code, **verdict.report},
            "truncation_audit": audit_report,
            "notes": list(compiled.notes),
            "files": {name: f"{name}.csv" for name in ("fills", "positions_daily", "daily_pnl", "events")},
        }
        headline = (
            f"{metrics['n_days']} days, return {pct(metrics['total_return'])}, CAGR {pct(metrics['cagr'])}, "
            f"MDD {pct(metrics['max_drawdown_pct'])}, Sharpe/day {metrics['sharpe_daily'] or 0:.3f}; "
            f"audit {len(audit.cuts)} cuts {'ok' if audit.ok else 'MISMATCH'}"
        )
        files = engine_files(result)
        if not audit.ok:
            report["summary"] = summary_text("backtest", False, f"look-ahead: truncation audit {headline}")
            return StageResult(Verdict.FAIL, f"truncation audit mismatch: {audit.mismatches[0]}", report, files)
        if not verdict.passed:
            report["summary"] = summary_text("backtest", False, f"liquidation: {verdict.headline}")
            return StageResult(Verdict.FAIL, f"liquidation: {verdict.reason}", report, files)
        report["summary"] = summary_text("backtest", True, f"{headline}; {verdict.headline}")
        return StageResult.passed(report, reason=report["summary"], files=files)


# ----------------------------------------------------------------------------- optimize


class DailyOptimizeStage(DailyStage):
    """TPE over the spec's ``$param`` grid with the engine on the train part; every trial in the ledger.

    Same search as the DSL optimize stage (``optimization.tpe_search``, grid-edge extension with the
    v3 validator as the hard bound, the full one-step neighbourhood), daily moments of the class basis
    as the objective. The selected point is then judged by the liquidation gate (§7).
    """

    name = "optimize"
    requires: ClassVar[tuple[str, ...]] = ("spec_validate", "backtest")

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update({"optimize": ctx.config.get("optimize"), "seed": ctx.config.get("seed")})
        return payload

    def evaluate(self, ctx):
        settings = ctx.config["optimize"]
        runner = self.spec_runner(ctx, "train")
        grid = runner.grid
        ledger = ctx.trials if ctx.trials is not None else MemoryTrialLedger()
        started = time.monotonic()
        window = DailyWindow.from_config(ctx.config)
        trials = EngineTrialRunner(
            runner,
            ledger=ledger,
            window_sha256=window_sha(runner.store.data_sha256(), "train", runner.start, runner.end),
            should_stop=ctx.should_stop,
            check_stop=ctx.check_stop,
        )
        steps = int(settings["neighbour_steps"])
        report = {
            "objective": settings["objective"],
            "objective_convention": f"per-trading-day Sharpe of the daily {runner.basis} series (not annualised)",
            "sampler": {"name": settings["sampler"], "seed": ctx.config["seed"]},
            "max_trials": settings["max_trials"],
            "param_grid": grid.describe(),
            "grid_size": grid.size() if grid.names else 1,
            "window": window.describe("train"),
            "data_sha256": runner.store.data_sha256(),
        }
        defaults = {name: spec.cast(spec.default) for name, spec in grid.specs.items()}
        if not grid.names:  # pragma: no cover - spec v3 requires 1-4 $params
            trials.evaluate({}, "base")
            neighbours, settled, extension = [], True, None
            search = opt.SearchSummary("no_params", 0, 0, False)
        else:
            search = opt.tpe_search(
                trials,
                grid,
                defaults,
                seed=int(ctx.config["seed"]),
                max_trials=int(settings["max_trials"]),
                max_consecutive_duplicates=int(settings["max_consecutive_duplicates"]),
                budget_seconds=settings.get("budget_seconds"),
                reserve_trials=2 * len(grid.names) * steps,
            )
            extension = opt.EdgeExtension(
                admissible=runner.admissible,
                budget_seconds=settings.get("budget_seconds"),
                started=started,
                reserve_trials=2 * len(grid.names) * steps,
                tag_kind=EDGE_AT_BOUND_KEY in ctx.config["thresholds"],
            )
            neighbours, settled = opt.complete_neighbourhood(trials, grid, steps, extension)
        results = trials.results
        by_state = Counter(result.state for result in results)
        report.update(
            {
                "search": {
                    "stopped_because": search.stopped_because,
                    "duplicates_skipped": search.duplicates_skipped,
                    "asks": search.asks,
                    "budget_exhausted": search.budget_exhausted,
                },
                "n_trials": len(results),
                "n_trials_by_source": {
                    source: sum(1 for r in results if r.source == source)
                    for source in ("base", "search", "edge_extension", "neighbour")
                },
                "n_trials_by_state": dict(sorted(by_state.items())),
                "neighbour_steps": steps,
                "neighbours": neighbours,
                "neighbourhood_settled": settled,
                "trials_file": "trials.json",
            }
        )
        if extension is not None:
            report["edge_extension"] = extension.report()
            report["effective_range"] = grid.effective_range(results, extension.max_steps)
        files = {"trials.json": canonical_json([result.row() for result in results]).encode("utf-8")}
        best = trials.best
        elapsed = time.monotonic() - started
        if best is None:
            report.update({"best": None, "summary": f"optimize ❌ no trial produced a daily Sharpe ({len(results)})"})
            return StageResult(Verdict.FAIL, f"no trial produced a daily Sharpe ({len(results)} trials)", report, files)
        best_spec = runner.bind(best.params)
        report["best"] = {"trial_no": best.trial_no, "params": best.params, "metrics": best.metrics}
        report["best_spec"] = best_spec
        report["best_spec_sha256"] = content_sha256(best_spec)
        if best.outcome is not None:
            files.update(engine_files(best.outcome, "best_"))
        daily = trials.daily[grid.key(best.params)]
        verdict = liquidation.evaluate(daily, ctx.config["thresholds"])
        report["liquidation"] = {"verdict": verdict.verdict.value, "reason_code": verdict.reason_code, **verdict.report}
        text = (
            f"{len(results)} trials in {elapsed:.0f}s; best #{best.trial_no} {canonical_json(best.params)} "
            f"Sharpe/day {best.sharpe:.3f}, return {pct(best.metrics.get('total_return'))}"
        )
        if not verdict.passed:
            report["summary"] = summary_text("optimize", False, f"best point: {verdict.headline}")
            return StageResult(Verdict.FAIL, f"liquidation at the selected point: {verdict.reason}", report, files)
        report["summary"] = summary_text("optimize", True, f"{text}; {verdict.headline}")
        return StageResult.passed(report, reason=report["summary"], files=files)


# ----------------------------------------------------------------------------- ledger-only gates


class _OptimizeGuard:
    """Shared ``run`` for the v3 versions of the ledger-only DSL gate stages."""

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update({"engine": ENGINE, "v3_wiring_version": V3_WIRING_VERSION})
        return payload

    def run(self, ctx):
        check_components(ctx.config)
        optimize = ctx.upstream["optimize"].report or {}
        if not optimize.get("best"):
            return StageResult.unsupported("optimize selected no point")
        return self.evaluate(ctx, optimize)


class DailyParamStabilityStage(_OptimizeGuard, ParamStabilityStage):
    """Gate 5 on optimize's neighbourhood; profitable = final equity > initial (total PnL > 0)."""


class DailyDsrStage(_OptimizeGuard, DsrStage):
    """Gate 4: the selected daily Sharpe (class basis) against every trial of the (single-class) family."""


# ----------------------------------------------------------------------------- re-running gates


class DailyGateStage(DailyStage):
    requires: ClassVar[tuple[str, ...]] = ("spec_validate", "optimize")

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload["gate"] = self.name
        return payload


def underlying_daily(store, symbol: str, *, through: date) -> pd.DataFrame:
    """The 1x underlying's daily OHLC (DAILY_GATES.benchmark.underlying), indexed by trading day.

    TX family: ratio back-adjusted continuous futures, rolled the day before the last trading day,
    OHLC of the carried contract on the same scale. Stock / ETF family: 0050's own bars, total-return
    adjusted by ``tw_corporate_actions`` (the latest bar keeps its price, earlier bars are scaled
    down by the factors of the actions after them).
    """

    from poseidon.pipeline.refdata.continuous import continuous_series
    from poseidon.pipeline.v3.instruments import INSTRUMENTS

    if INSTRUMENTS[symbol].kind == "stock":
        return stock_underlying_daily(store, symbol, through=through)
    product = symbol
    quotes = store.read(FUTURES, end=through)
    calendar = store.read(CALENDAR)
    series = continuous_series(quotes, calendar, product, days_before=UNDERLYING_ROLL_DAYS_BEFORE)
    factor = series["close_adj_ratio"] / series["prev_close"]
    frame = pd.DataFrame(
        {
            "open": series["open"] * factor,
            "high": series["high"] * factor,
            "low": series["low"] * factor,
            "close": series["close_adj_ratio"],
        }
    )
    frame.index = [pd.Timestamp(day).date() for day in series["trade_date"]]
    return frame


def stock_underlying_daily(store, symbol: str, *, through: date) -> pd.DataFrame:
    """``symbol`` on every session the stock dataset has (from its first bar): a session it lacks a bar
    for is carried forward (zero return), as ``align_stock_bars`` does for a traded stock, so every day a
    stock run books has an underlying return."""

    bars = read_bars(store, symbol, end=through)
    if bars.empty:
        raise NotCompilable(f"benchmark underlying: no {symbol} bars in {STOCKS} through {through}")
    sessions = pd.DatetimeIndex(
        sorted(store.read(STOCKS, start=bars["date"].min(), end=through)["trade_date"].unique())
    )
    actions = load_actions(store, symbol) if store.exists(ACTIONS) else None
    try:
        booked = stock_bars(align_stock_bars(bars, sessions, symbol), actions)
    except (StockDataGap, ValueError) as error:
        raise NotCompilable(f"benchmark underlying: {error}") from error
    # back-adjusted like the futures series: a bar is scaled by the factors of the actions booked after it
    after = booked["factor"].iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0).to_numpy()
    frame = pd.DataFrame(
        {column: booked[column].to_numpy(float) * after for column in ("open", "high", "low", "close")}
    )
    frame.index = [pd.Timestamp(day).date() for day in booked["date"]]
    return frame


def underlying_product(ctx) -> str:
    """Symbol of the run's 1x underlying: TX for the TX futures family, 0050 for a stock / ETF spec."""

    from poseidon.pipeline.v3.instruments import INSTRUMENTS

    spec = (ctx.upstream["spec_validate"].report or {}).get("template") or {}
    symbols = [decl["symbol"] for decl in spec.get("instruments", {}).values()]
    rows = [INSTRUMENTS[symbol] for symbol in symbols]
    kinds = {row.kind for row in rows}
    if kinds == {"stock"}:
        return STOCK_UNDERLYING
    sources = {row.price_proxy or row.symbol for row in rows if row.kind == "future"}
    if kinds != {"future"} or sources != {"TX"}:
        raise NotCompilable(
            "daily benchmark/regime underlying is defined for the TX futures family and for stock / ETF specs, "
            f"not for a mix (spec: {symbols})"
        )
    return "TX"


def underlying_label(ctx, symbol: str) -> str:
    """The frozen ``gates.benchmark.underlying`` text of the run's family (report only)."""

    labels = ctx.config["gates"]["benchmark"]["underlying"]
    return labels["tw_stock" if symbol == STOCK_UNDERLYING else "tw_futures"]


class DailyCostStressStage(DailyGateStage):
    name = "cost_stress"

    def evaluate(self, ctx):
        thresholds = ctx.config["thresholds"]
        runner, base = self.train_best(ctx)
        params = self.best(ctx)["params"]
        result = cost_stress.run_daily(runner.rerun, params, base, runner.basis, thresholds)
        inputs = {
            **self.best_source(ctx),
            "window": DailyWindow.from_config(ctx.config).describe("train"),
            "scope": ctx.config["gates"]["cost_stress"]["scope"],
            "base_metrics": daily_metrics(base, runner.basis),
        }
        return to_stage_result(result, inputs)


class DailyRegimeSplitStage(DailyGateStage):
    name = "regime_split"

    def evaluate(self, ctx):
        thresholds = ctx.config["thresholds"]
        runner, base = self.train_best(ctx)
        daily = underlying_daily(runner.store, underlying_product(ctx), through=runner.end)
        states = regime_split.regime_states(daily, **regime_split.threshold_windows(thresholds))
        result = regime_split.evaluate_daily(
            regime_split.RegimeDailyInputs(base, runner.basis, states.to_dict()), thresholds
        )
        labelled = states.dropna()
        inputs = {
            **self.best_source(ctx),
            "underlying": {
                "source": ctx.config["gates"]["regime_split"]["daily_bars"],
                "days": len(daily),
                "labelled_days": len(labelled),
                "first_labelled": labelled.index[0].isoformat() if len(labelled) else None,
                "state_days": dict(sorted(Counter(labelled).items())),
            },
            "labelling": ctx.config["gates"]["regime_split"]["day_label"],
        }
        return to_stage_result(result, inputs)


def beta_matched_spec(template: dict, symbol: str, leverage: float) -> dict | benchmark.NotApplicable:
    """The no-timing version of ``template`` for the risk route: its 1x underlying ``symbol`` at a constant
    ``leverage``, rebalanced daily, with the template's capital and execution.

    TX family: the futures contract the first sleeve trades, with its roll (TX, MTX or TMF on TX prices). Stock /
    ETF family: 0050 as the template declares it (or by board lot). Not applicable when the leverage cannot be
    held: above 1x on a stock (paid in full), above the policy cap or the initial-margin limit on a future.
    """

    from poseidon.pipeline.v3.instruments import INSTRUMENTS, MAX_MARGIN_UTILISATION
    from poseidon.pipeline.v3.validation import LEVERAGE_CAP

    declared = template["instruments"]
    if symbol == STOCK_UNDERLYING:
        alias = next((name for name, decl in declared.items() if decl["symbol"] == symbol), "underlying")
        decl = declared.get(alias, {"symbol": symbol})
        if leverage > 1.0:
            return benchmark.NotApplicable(f"{symbol} is paid in full: {leverage:g}x cannot be held")
    else:
        alias = template["sleeves"][0]["instrument"]
        decl = declared[alias]
        instrument = INSTRUMENTS[decl["symbol"]]
        margin = leverage * instrument.initial_margin_ratio
        if leverage > LEVERAGE_CAP or margin > MAX_MARGIN_UTILISATION:
            return benchmark.NotApplicable(
                f"{leverage:g}x {instrument.symbol} cannot be held (cap {LEVERAGE_CAP:g}x, initial margin "
                f"{margin:.0%} of equity > {MAX_MARGIN_UTILISATION:.0%})"
            )
    return {
        "spec_version": template["spec_version"],
        "name": f"{template['name']}_beta_matched",
        "frequency": "1d",
        "capital": template.get("capital") or {},
        "instruments": {alias: decl},
        "sleeves": [
            {
                "id": "matched",
                "instrument": alias,
                "direction": {"component": "direction.constant", "params": {"side": "long"}},
                "leverage": [{"component": "leverage.base", "params": {"value": leverage}}],
                "sizing": {"component": "sizing.target_leverage", "params": {"rounding": "nearest"}},
                "rebalance": {"component": "rebalance.daily", "params": {"min_change_units": 1}},
                "exits": [],
                "hedges": [],
            }
        ],
        "allocation": {"component": "allocation.single"},
        "execution": template["execution"],
    }


def beta_matched_run(
    runner: SpecRunner, symbol: str, leverage: float, min_contracts: int | None = None
) -> DailyResult | benchmark.NotApplicable:
    """``benchmark.MatchedRun``: the beta-matched spec over the runner's window and refdata.

    Not applicable when a futures matched run holds fewer than ``min_contracts`` (frozen
    ``benchmark_risk_min_contracts``; None: no such rule) on its initial capital at the window's first or highest
    close (``price_points``, as spec_validate sizes): one contract is then half or more of its exposure, so
    whole-contract rounding, not timing, decides the comparison.
    """

    from poseidon.pipeline.v3.components import round_units
    from poseidon.pipeline.v3.instruments import INSTRUMENTS
    from poseidon.pipeline.v3.spec import DEFAULT_INITIAL_CAPITAL_TWD

    spec = beta_matched_spec(runner.template, symbol, leverage)
    if isinstance(spec, benchmark.NotApplicable):
        return spec
    instrument = INSTRUMENTS[spec["instruments"][spec["sleeves"][0]["instrument"]]["symbol"]]
    if min_contracts is not None and instrument.kind == "future":
        capital = float(spec["capital"].get("initial_twd", DEFAULT_INITIAL_CAPITAL_TWD))
        points = price_points(runner.store, instrument, runner.start, runner.end)  # the window has quotes: it ran
        units, price = min((round_units(capital * leverage / instrument.notional_twd(p), "nearest"), p) for p in points)
        if units < min_contracts:
            return benchmark.NotApplicable(
                f"the {leverage:g}x matched run holds {units} {instrument.symbol} at {price:,.0f}, fewer than "
                f"{min_contracts}: contract rounding, not timing, would decide the comparison"
            )
    compiled = compile_spec(
        spec,
        start=runner.start,
        end=runner.end,
        store=runner.store,
        sessions=runner.feasibility.warmup,
        calendar_rule=runner.calendar_rule,
    )
    result = from_engine(compiled.run())
    lost = result.event_days(LIQUIDATION_EVENTS)
    if lost:
        return benchmark.NotApplicable(f"the {leverage:g}x matched run hit {lost[0]['kind']} on {lost[0]['date']}")
    return result


class DailyBenchmarkStage(DailyGateStage):
    name = "benchmark"
    lease_seconds = 600

    def evaluate(self, ctx):
        settings = ctx.config["gates"]["benchmark"]
        min_contracts = ctx.config["thresholds"].get(benchmark.MIN_CONTRACTS_KEY)
        runner, base = self.train_best(ctx)
        symbol = underlying_product(ctx)
        daily = underlying_daily(runner.store, symbol, through=runner.end)
        underlying = underlying_returns(daily["close"])
        result = benchmark.evaluate_daily(
            benchmark.BenchmarkDailyInputs(
                base,
                underlying,
                runner.basis,
                int(settings["hac_lags"]),
                matched=lambda leverage: beta_matched_run(runner, symbol, leverage, min_contracts),
            ),
            ctx.config["thresholds"],
        )
        inputs = {
            **self.best_source(ctx),
            "underlying": underlying_label(ctx, symbol),
            "hac_lags": int(settings["hac_lags"]),
            "train_metrics": daily_metrics(base, runner.basis),
        }
        return to_stage_result(result, inputs)


class ExecutionTimingStage(DailyGateStage):
    """Execution-timing gate, blocking for daily classes (owner decision 2026-10-24 afternoon #2).

    The stage supplies the train run of the selected params, the spec's own fill
    (``compile_spec(...).plan.execution``) and the re-run hook (``SpecRunner.rerun``: execution and
    ``signal_lag_extra`` as run Conventions); ``gates.execution_timing.run_daily`` re-runs every
    registered scenario and decides with the frozen ``execution_timing_*`` thresholds.
    """

    name = EXECUTION_TIMING

    def evaluate(self, ctx):
        runner, base = self.train_best(ctx)
        params = self.best(ctx)["params"]
        execution = runner.compile(params).plan.execution
        result = execution_timing.run_daily(
            runner.rerun,
            params,
            base,
            base_execution=execution,
            basis=runner.basis,
            thresholds=ctx.config["thresholds"],
        )
        inputs = {
            **self.best_source(ctx),
            "base_execution": execution,
            "scenarios": ctx.config["gates"].get("execution_timing"),
            "window": DailyWindow.from_config(ctx.config).describe("train"),
        }
        return to_stage_result(result, inputs)


class DailyWfeStage(DailyGateStage):
    """Gate 1: rolling walk-forward, per-window TPE on the IS days, the chosen point run on the OOS days."""

    name = "wfe"

    def evaluate(self, ctx):
        settings = ctx.config["gates"]["wfe"]
        window = DailyWindow.from_config(ctx.config)
        chosen = choose_walk_forward(window.start, window.holdout_start, settings["layouts"])
        if chosen is None:
            result = failed(
                wfe_gate.GATE,
                "insufficient_history",
                f"train part {window.start} .. {window.train_last} fits none of the frozen layouts",
                "train part too short",
                {"layouts": settings["layouts"]},
            )
            return to_stage_result(result, {"layouts": settings["layouts"]})
        base = self.spec_runner(ctx, "train")
        ledger = ctx.trials if ctx.trials is not None else MemoryTrialLedger()
        data_sha = base.store.data_sha256()
        seed = int(ctx.config["seed"])
        max_duplicates = int(ctx.config["optimize"]["max_consecutive_duplicates"])
        budget = float(settings["budget_seconds"]) / len(chosen["windows"])
        defaults = {name: spec.cast(spec.default) for name, spec in base.grid.specs.items()}
        metrics, rows, trial_rows, next_trial = [], [], [], 1
        started = time.monotonic()
        for spec_window in chosen["windows"]:
            ctx.check_stop()
            is_last = spec_window["oos_start"] - timedelta(days=1)
            in_sample = dataclasses.replace(base, start=spec_window["is_start"], end=is_last)
            trials = EngineTrialRunner(
                in_sample,
                ledger=ledger,
                window_sha256=window_sha(data_sha, "train", in_sample.start, is_last, wfe_window=spec_window["index"]),
                should_stop=ctx.should_stop,
                check_stop=ctx.check_stop,
                first_trial_no=next_trial,
            )
            search = opt.tpe_search(
                trials,
                base.grid,
                defaults,
                seed=seed + spec_window["index"],
                max_trials=int(settings["max_trials_per_window"]),
                max_consecutive_duplicates=max_duplicates,
                budget_seconds=budget,
            )
            next_trial += len(trials.results)
            trial_rows += [{"window": spec_window["index"], **r.row()} for r in trials.results]
            pick = trials.best
            if pick is None:
                result = invalid(wfe_gate.GATE, f"window {spec_window['index']}: no IS trial produced a daily Sharpe")
                return to_stage_result(result, {"layout": chosen["layout"], "windows": rows})
            oos_last = spec_window["oos_end"] - timedelta(days=1)
            out_of_sample = dataclasses.replace(base, start=spec_window["oos_start"], end=oos_last)
            oos = out_of_sample.run(pick.params)[1]
            is_result = trials.daily[base.grid.key(pick.params)]
            metrics.append(wfe_gate.window_metrics_daily(spec_window["index"], is_result, oos, base.basis))
            rows.append(
                {
                    "index": spec_window["index"],
                    "is_first": spec_window["is_start"].isoformat(),
                    "oos_first": spec_window["oos_start"].isoformat(),
                    "oos_last": oos_last.isoformat(),
                    "chosen_trial_no": pick.trial_no,
                    "chosen_params": pick.params,
                    "n_trials": len(trials.results),
                    "search_stopped_because": search.stopped_because,
                    "is_metrics": daily_metrics(is_result, base.basis),
                    "oos_metrics": daily_metrics(oos, base.basis),
                }
            )
        result = wfe_gate.evaluate(wfe_gate.WfeInputs(metrics), ctx.config["thresholds"])
        inputs = {
            "layout": chosen["layout"],
            "windows": rows,
            "days": settings["days"],
            "n_trials_recorded": next_trial - 1,
            "ledger_note": "every per-window IS trial is in the family ledger (counts toward DSR N for later runs)",
            "oos_runs": "fresh capital at each OOS start, warm-up bars before it",
            "seconds": round(time.monotonic() - started, 1),
            "trials_file": "wfe_trials.json",
        }
        files = {"wfe_trials.json": canonical_json(trial_rows).encode("utf-8")}
        return to_stage_result(result, inputs, files)


class DailyHoldoutStage(DailyGateStage):
    """Gate 2: the selected params on the locked holdout (fresh capital at its first day). The only
    reader of holdout data; the look is recorded before any of it is read."""

    name = "holdout"
    requires: ClassVar[tuple[str, ...]] = (
        "spec_validate",
        "optimize",
        *V3_GATE_STAGES[: V3_GATE_STAGES.index("holdout")],
    )

    def evaluate(self, ctx):
        from poseidon.pipeline.family import MemoryFamily

        optimize = ctx.upstream["optimize"].report or {}
        family = ctx.family if ctx.family is not None else MemoryFamily(ctx.trials, run_id=ctx.run_id)
        prior = family.holdout_evaluations()
        family.record_holdout_evaluated(
            {
                "stage": self.name,
                "attempt": ctx.attempt,
                "stage_run_id": str(ctx.stage_run_id) if ctx.stage_run_id else None,
                "best_spec_sha256": optimize.get("best_spec_sha256"),
                "family_runs_already_evaluated": prior["other_runs"],
            }
        )
        params = self.best(ctx)["params"]
        _train_runner, train = self.train_best(ctx)
        runner = self.spec_runner(ctx, "holdout")
        engine_result, held = runner.run(params)
        lags = int(ctx.config["gates"]["benchmark"]["hac_lags"])
        symbol = underlying_product(ctx)
        daily = underlying_daily(runner.store, symbol, through=runner.end)
        stats = benchmark.regress_daily(held, underlying_returns(daily["close"]), basis=runner.basis, lags=lags)
        result = holdout.evaluate_daily(
            holdout.HoldoutDailyInputs(
                train,
                held,
                runner.basis,
                stats,
                matched=lambda leverage: beta_matched_run(
                    runner, symbol, leverage, ctx.config["thresholds"].get(benchmark.MIN_CONTRACTS_KEY)
                ),
            ),
            ctx.config["thresholds"],
        )
        inputs = {
            **self.best_source(ctx),
            "window": DailyWindow.from_config(ctx.config).describe("holdout"),
            "train_metrics": daily_metrics(train, runner.basis),
            "holdout_metrics": daily_metrics(held, runner.basis),
            "fresh_capital": "the holdout run starts with the spec's initial capital on its first day",
            "family_holdout_evaluations": {
                **prior,
                "policy": ctx.config["gates"]["holdout"]["family_reuse"],
                "note": "runs of this family that evaluated the holdout before this attempt (report only, not gated)",
            },
        }
        return to_stage_result(result, inputs, engine_files(engine_result, "holdout_"))


# ----------------------------------------------------------------------------- shadow paper seam


def compile_strategy(ctx):
    """``shadow_paper_daily.compile_strategy`` for a v3 run: optimize's selected spec -> CompiledStrategy.

    The engine starts on the train start (the paper re-simulates the whole history every wake-up and
    books only days from its anchor on); the protection basis is the train window ``[start,
    holdout_start)``. The fingerprint covers the selected spec, the registry version and the frozen
    component fingerprints, all verified against the live registry first. Aux series: only the indices
    the spec's signals read (``Plan.signal_indices``, e.g. ``index:taiex``), so the paper's cutoff
    waits for that day's index row and for nothing it does not use. The warm-up loads as the backtest's
    does (``compile_spec``): in sessions for a run frozen with ``data.warmup``, else in calendar days.
    Hedges run only for a run frozen with ``data.option_chain`` (``v3.config.option_chain_record``); an
    older hedged run stays unsupported, as it was frozen.
    """

    from poseidon.pipeline.shadow_paper_daily import CompiledStrategy, PaperInstrument, StrategyNotCompiled
    from poseidon.pipeline.v3.compiler import AUX_DATASETS, engine_config, hedge_policies, plan, strategy_factory
    from poseidon.pipeline.v3.validation import warmup_calendar_days

    try:
        components = check_components(ctx.config)
    except ComponentDrift as error:
        raise StrategyNotCompiled(str(error)) from error
    optimize = ctx.upstream["optimize"].report or {}
    best_spec = optimize.get("best_spec")
    if not isinstance(best_spec, dict) or best_spec.get("spec_version") != 3:
        raise StrategyNotCompiled("optimize selected no spec v3 point")
    try:
        spec_plan = plan(best_spec)
    except NotCompilable as error:
        raise StrategyNotCompiled(str(error)) from error
    if spec_plan.hedges and "option_chain" not in ctx.config.get("data", {}):
        raise StrategyNotCompiled(
            "daily shadow paper runs hedges only for runs frozen with data.option_chain; this run predates it "
            "(create a new run)"
        )
    window = DailyWindow.from_config(ctx.config)
    conventions = Conventions()
    warmup = spec_plan.warmup_bars + conventions.signal_lag
    record = ctx.config["data"].get("warmup")
    data_start = None if record else window.start - timedelta(days=warmup_calendar_days(warmup, "1d"))
    instruments = {
        alias: PaperInstrument(
            product=row.price_proxy or row.symbol,
            spec=ContractSpec.from_instrument(row),
            roll_days_before=spec_plan.rolls.get(alias, 0),
        )
        for alias, row in spec_plan.instruments.items()
    }
    fingerprint = content_sha256(
        {
            "best_spec_sha256": content_sha256(best_spec),
            "registry_version": ctx.config["registry_version"],
            "components": components,
            "engine": ENGINE,
        }
    )
    return CompiledStrategy(
        factory=strategy_factory(spec_plan, conventions),
        instruments=instruments,
        engine=engine_config(spec_plan, start=window.start, end=None, conventions=conventions),
        data_start=data_start,
        train_end=window.holdout_start,
        fingerprint=fingerprint,
        aux={name: AUX_DATASETS[name] for name in spec_plan.signal_indices},
        warmup=(record, warmup) if record else None,
        calendar_rule=ctx.config["data"].get("trading_calendar"),
        hedges=tuple(hedge_policies(spec_plan)),
    )


def build_daily_stages(*, refdata=None) -> dict:
    """The v3 bodies, by stage name (``stages.build_default_registry`` routes v3 runs to them)."""

    from poseidon.pipeline.shadow_paper import ShadowPaperStage

    class DailyShadowPaperStage(ShadowPaperStage):
        requires: ClassVar[tuple[str, ...]] = ("backtest", "optimize", *V3_GATE_STAGES)

    stages = [
        DailySpecValidateStage(refdata=refdata),
        DailyBacktestStage(refdata=refdata),
        DailyOptimizeStage(refdata=refdata),
        DailyParamStabilityStage(),
        DailyDsrStage(),
        DailyCostStressStage(refdata=refdata),
        DailyRegimeSplitStage(refdata=refdata),
        DailyBenchmarkStage(refdata=refdata),
        ExecutionTimingStage(refdata=refdata),
        DailyWfeStage(refdata=refdata),
        DailyHoldoutStage(refdata=refdata),
        DailyShadowPaperStage(refdata=refdata, compile_strategy=compile_strategy),
    ]
    return {stage.name: stage for stage in stages}
