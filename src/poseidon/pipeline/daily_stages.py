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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import optimization as opt
from poseidon.pipeline.engine import ContractSpec, EngineResult, OptionSpec
from poseidon.pipeline.gates.base import finite_or_none
from poseidon.pipeline.gates.daily import (
    Basis,
    DailyResult,
    RerunRequest,
    basis_for,
    from_engine,
    moments,
)
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, TAIEX, TAIEX_TR, VIX
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.stage_api import BaseStage, StageResult
from poseidon.pipeline.trials import TRIAL_COMPLETE, TRIAL_FAIL, TRIAL_PRUNED
from poseidon.pipeline.v3.compiler import OPTION_DATASETS, ROW_FILTERS, NotCompilable, compile_spec, hedged
from poseidon.pipeline.v3.config import ENGINE, session_warmup, used_index_datasets
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
