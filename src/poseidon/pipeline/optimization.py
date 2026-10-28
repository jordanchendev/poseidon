"""The optimize stage's search: Optuna TPE over a spec's ``$param`` grid.

* Seeded ``TPESampler`` (``config_json["seed"]``) with the spec defaults
  enqueued as trial 1, driven with ask/tell so every finished trial is written
  to the ledger before the next one starts, and the stage can stop between
  (and, via ``should_stop``, inside) backtests.
* Values are snapped to the grid; a sampler suggestion that repeats an
  evaluated point is answered from memory and is **not** a new trial (no
  backtest, no ledger row): N counts distinct evaluated configurations.
* The search ends at ``max_trials`` distinct points, when the grid is
  exhausted, after ``max_consecutive_duplicates`` repeats (a grid that fits
  the cap is then finished in grid order), or (safety only) when the next
  trial plus one neighbour round would overrun ``budget_seconds``.
* Grid-edge extension (owner decision 2026-10-23): when the best point sits
  on the edge of a param's declared grid, one more step outward (same step)
  is backtested as a normal trial (source ``edge_extension``, in the ledger,
  so it counts toward DSR's N), provided the resolved spec still passes the
  DSL v2/v2.1 validator (every field the param feeds stays in its hard
  bounds). A better outward point becomes the best and the check repeats, at
  most ``MAX_EDGE_EXTENSION_STEPS`` per param per side. That cap is absolute
  (owner decision): the best never moves beyond declared edge +/- 3 steps,
  not even through the neighbour round; a point beyond it is backtested only
  as a neighbour (a trial, counted in N) and, if it is better, reported as
  ``better_beyond_cap`` (the declared grid was too narrow). Extensions count
  against ``budget_seconds`` and stop (with the reason recorded) once the
  next one plus the neighbour round would overrun it. The declared grid is
  never changed: off-grid values are built from a template whose ``params``
  entry is widened just enough to contain them.
* Afterwards every one-at-a-time +/-1 step neighbour of the best point that
  was not evaluated yet is backtested and logged as a trial too, so
  ``param_stability`` always has the full neighbourhood. A neighbour outside
  the declared grid is evaluated too unless it violates a validator hard
  bound; only then is it excluded (``in_grid`` false, ``excluded_because``).
  With ``tag_kind`` (runs frozen with ``param_edge_at_bound_pass``) an excluded row also
  carries ``excluded_kind``: ``"component_bound"`` when the admissibility check named a
  component field bound (a :class:`BoundReason`), ``"declared_grid"`` or ``"other"`` otherwise.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from poseidon.decision_loop.manifest import canonical_json
from poseidon.pipeline.backtesting import BacktestFailed, BacktestOutcome, run_backtest
from poseidon.pipeline.trials import TRIAL_COMPLETE, TRIAL_FAIL, TRIAL_PRUNED, make_record

logger = logging.getLogger(__name__)

SEARCH_SUMMARY_KEYS = (
    "total_pnl_twd",
    "sharpe_daily",
    "sharpe_annualised",
    "n_trades",
    "win_rate",
    "max_drawdown_twd",
    "n_obs",
    "skew",
    "kurtosis",
)


@dataclass
class TrialResult:
    trial_no: int
    params: dict
    source: str  # base | search | edge_extension | neighbour
    state: str
    metrics: dict | None = None
    error: str | None = None
    outcome: BacktestOutcome | None = field(default=None, repr=False)

    @property
    def sharpe(self) -> float | None:
        return (self.metrics or {}).get("sharpe_daily")

    def row(self) -> dict:
        return {
            "trial_no": self.trial_no,
            "params": self.params,
            "source": self.source,
            "state": self.state,
            "error": self.error,
            "metrics": self.metrics,
        }

    def summary(self) -> dict:
        metrics = self.metrics or {}
        return {key: metrics.get(key) for key in SEARCH_SUMMARY_KEYS}


class BoundReason(str):
    """An ``admissible`` refusal caused by a component field's own hard bound (not the spec's range)."""


# pydantic error types of a value outside a component field's ge/le/gt/lt bound
BOUND_ERROR_TYPES = frozenset({"greater_than_equal", "less_than_equal", "greater_than", "less_than"})
EXCLUDED_COMPONENT_BOUND = "component_bound"
EXCLUDED_DECLARED_GRID = "declared_grid"
EXCLUDED_OTHER = "other"


class ParamGrid:
    """The ``$param`` grid of a validated template (``ValidatedV2.params``)."""

    def __init__(self, params: dict):
        self.specs = dict(sorted(params.items()))
        self.names = tuple(self.specs)

    def snap(self, values: dict) -> dict:
        return {name: self.specs[name].cast(values[name]) for name in self.names}

    def key(self, values: dict) -> str:
        return canonical_json(self.snap(values))

    def points(self):
        """Every grid point in lexicographic order of ``names``."""

        import itertools

        for combo in itertools.product(*(self.specs[name].grid() for name in self.names)):
            yield dict(zip(self.names, combo, strict=True))

    def size(self) -> int:
        return math.prod(spec.grid_size() for spec in self.specs.values())

    def describe(self) -> dict:
        return {
            name: {
                "type": spec.type,
                "low": spec.cast(spec.low),
                "high": spec.cast(spec.high),
                "step": spec.step if spec.type == "float" else int(spec.step),
                "default": spec.cast(spec.default),
                "grid": spec.grid(),
            }
            for name, spec in self.specs.items()
        }

    def distributions(self) -> dict:
        from optuna.distributions import FloatDistribution, IntDistribution

        out = {}
        for name, spec in self.specs.items():
            if spec.type == "int":
                out[name] = IntDistribution(int(spec.low), int(spec.high), step=int(spec.step))
            else:
                out[name] = FloatDistribution(float(spec.low), float(spec.high), step=float(spec.step))
        return out

    def on_declared_grid(self, name: str, value) -> bool:
        spec = self.specs[name]
        return spec.low - 1e-9 <= value <= spec.high + 1e-9

    def neighbours(self, best: dict, steps: int = 1) -> list[dict]:
        """One-at-a-time ``+/- steps`` grid neighbours of ``best``.

        ``in_grid`` is whether the neighbour lies on the declared grid;
        ``candidate`` is the neighbour's params either way (``params`` stays
        None off the declared grid, as before).
        """

        out = []
        for name in self.names:
            spec = self.specs[name]
            for direction in (-1, 1):
                value = best[name] + direction * steps * spec.step
                in_grid = self.on_declared_grid(name, value)
                candidate = {**best, name: spec.cast(value)}
                params = candidate if in_grid else None
                out.append(
                    {
                        "param": name,
                        "direction": direction,
                        "in_grid": in_grid,
                        "params": params,
                        "candidate": candidate,
                    }
                )
        return out

    def within_cap(self, values: dict, max_steps: int) -> bool:
        """Whether every value lies within its declared grid widened by ``max_steps`` steps per side."""

        for name in self.names:
            spec = self.specs[name]
            reach = max_steps * spec.step
            if not spec.low - reach - 1e-9 <= values[name] <= spec.high + reach + 1e-9:
                return False
        return True

    def effective_range(self, results: list[TrialResult], max_steps: int | None = None) -> dict:
        """Declared grid vs the range actually evaluated (edge extensions widen it)."""

        out = {}
        for name, spec in self.specs.items():
            values = [r.params[name] for r in results if name in r.params]
            low = min([spec.cast(spec.low), *values])
            high = max([spec.cast(spec.high), *values])
            out[name] = {
                "declared_low": spec.cast(spec.low),
                "declared_high": spec.cast(spec.high),
                "step": spec.step if spec.type == "float" else int(spec.step),
                "low": low,
                "high": high,
                "extended_below": low < spec.low - 1e-9,
                "extended_above": high > spec.high + 1e-9,
            }
            if max_steps is not None:
                # The range the best may move in (declared edge +/- max_steps steps).
                out[name]["best_cap_low"] = spec.cast(spec.low - max_steps * spec.step)
                out[name]["best_cap_high"] = spec.cast(spec.high + max_steps * spec.step)
        return out


# ------------------------------------------------------------- off-grid values


def effective_template(template: dict, grid: ParamGrid, values: dict) -> dict:
    """``template`` itself when every value is on its declared grid, else a copy whose
    ``params`` entry for each off-grid value spans just [declared edge, value].

    The narrow span keeps ``MAX_GRID_POINTS`` and the validator's whole-grid
    range check limited to the edge and the extension values; the declared
    ``params`` of the spec are never changed.
    """

    off = {name: values[name] for name in grid.names if not grid.on_declared_grid(name, values[name])}
    if not off:
        return template
    out = copy.deepcopy(template)
    for name, value in off.items():
        spec = grid.specs[name]
        edge = spec.cast(spec.high if value > spec.high else spec.low)
        out["params"][name] = {
            "type": spec.type,
            "low": min(edge, value),
            "high": max(edge, value),
            "step": spec.step if spec.type == "float" else int(spec.step),
            "default": value,
        }
    return out


def hard_bound_check(template: dict, grid: ParamGrid) -> Callable[[dict], str | None]:
    """``values -> None`` when the resolved spec passes the DSL v2/v2.1 validator, else the first error.

    This is the validator's own check of every field each ``$param`` feeds
    (pydantic field bounds plus the cross-field rules), applied to one point.
    """

    from poseidon.strategies.dsl.v2.validate import DslV2Error, validate_v2

    def check(values: dict) -> str | None:
        try:
            validate_v2(effective_template(template, grid, values), param_values=dict(values))
        except DslV2Error as error:
            first = error.errors[0]
            text = f"{first['path'] or '<root>'}: {first['msg']}"[:300]
            return BoundReason(text) if first.get("type") in BOUND_ERROR_TYPES else text
        return None

    return check


def off_grid_builder(
    factory, template: dict, grid: ParamGrid, make_factory: Callable[[dict], object]
) -> Callable[[dict], object]:
    """``factory.build`` for declared-grid points; a factory over the widened template otherwise."""

    def build(values: dict):
        effective = effective_template(template, grid, values)
        if effective is template:
            return factory.build(values)
        return make_factory(effective).build(values)

    return build


class TrialRunner:
    """Evaluates parameter points and appends one ledger row per evaluation."""

    def __init__(
        self,
        *,
        build: Callable[[dict], object],
        bars,
        backtest_config: dict,
        ledger,
        window_sha256: str,
        should_stop: Callable[[], bool],
        check_stop: Callable[[], None],
        first_trial_no: int = 1,
    ):
        self.first_trial_no = first_trial_no  # >1 when one stage attempt runs several searches (wfe)
        self.build = build
        self.bars = bars
        self.backtest_config = backtest_config
        self.ledger = ledger
        self.window_sha256 = window_sha256
        self.should_stop = should_stop
        self.check_stop = check_stop
        self.results: list[TrialResult] = []
        self.best: TrialResult | None = None
        # Which points may become the best (the optimize stage caps grid-edge extensions);
        # a better point that may not is kept in ``better_ineligible`` instead.
        self.eligible: Callable[[dict], bool] = lambda _params: True
        self.better_ineligible: list[TrialResult] = []

    def record(
        self, params: dict, source: str, state: str, metrics: dict | None, error: str | None = None, outcome=None
    ):
        result = TrialResult(self.first_trial_no + len(self.results), params, source, state, metrics, error, outcome)
        self.ledger.append(make_record(result.trial_no, params, state, metrics, self.window_sha256))
        self.results.append(result)
        better = result.state == TRIAL_COMPLETE and (self.best is None or result.sharpe > self.best.sharpe)
        if better and not self.eligible(params):
            self.better_ineligible.append(result)
            better = False
        if better:
            if self.best is not None:
                self.best.outcome = None  # keep only the current best's trades in memory
            self.best = result
        else:
            result.outcome = None
        return result

    def evaluate(self, params: dict, source: str) -> TrialResult:
        self.check_stop()
        started = time.perf_counter()
        try:
            strategy = self.build(params)
            outcome = run_backtest(strategy, self.bars, self.backtest_config, should_stop=self.should_stop)
        except (BacktestFailed, ValueError) as error:  # DslValidationError is a ValueError
            logger.warning("optimize trial %s failed: %s", params, error)
            return self.record(params, source, TRIAL_FAIL, None, f"{type(error).__name__}: {error}"[:500])
        state = TRIAL_COMPLETE if outcome.metrics.get("sharpe_daily") is not None else TRIAL_PRUNED
        logger.info(
            "optimize trial %d %s: %s sharpe_daily=%s in %.1fs",
            self.first_trial_no + len(self.results),
            params,
            state,
            outcome.metrics.get("sharpe_daily"),
            time.perf_counter() - started,
        )
        return self.record(params, source, state, outcome.metrics, outcome=outcome)


@dataclass
class SearchSummary:
    stopped_because: str
    duplicates_skipped: int
    asks: int
    budget_exhausted: bool


def tpe_search(
    runner: TrialRunner,
    grid: ParamGrid,
    defaults: dict,
    *,
    seed: int,
    max_trials: int,
    max_consecutive_duplicates: int,
    budget_seconds: float | None,
    reserve_trials: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> SearchSummary:
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(grid.snap(defaults))
    distributions = grid.distributions()
    seen: dict[str, TrialResult] = {}
    grid_size = grid.size()
    duplicates = consecutive = asks = 0
    started = clock()
    stopped = "max_trials"
    while len(seen) < max_trials:
        if len(seen) >= grid_size:
            stopped = "grid_exhausted"
            break
        if consecutive >= max_consecutive_duplicates:
            stopped = "duplicates"
            break
        if budget_seconds is not None and seen:
            # Safety only (max_trials is sized to stay well inside it): stop
            # once the next trial plus one neighbour round would not fit.
            elapsed = clock() - started
            per_trial = elapsed / len(seen)
            if elapsed + per_trial * (1 + reserve_trials) > budget_seconds:
                stopped = "budget"
                break
        runner.check_stop()
        trial = study.ask(distributions)
        asks += 1
        params = grid.snap(trial.params)
        key = grid.key(params)
        result = seen.get(key)
        if result is None:
            consecutive = 0
            result = runner.evaluate(params, "search")
            seen[key] = result
        else:
            duplicates += 1
            consecutive += 1
        if result.state == TRIAL_COMPLETE:
            study.tell(trial, result.sharpe)
        elif result.state == TRIAL_PRUNED:
            study.tell(trial, state=TrialState.PRUNED)
        else:
            study.tell(trial, state=TrialState.FAIL)
    if stopped == "duplicates" and grid_size - len(seen) <= max_trials - len(seen):
        # TPE keeps re-suggesting known points on a small grid; the whole grid
        # fits the trial cap, so finish it in grid order (still deterministic).
        for values in grid.points():
            if grid.key(values) not in seen:
                seen[grid.key(values)] = runner.evaluate(values, "search")
        stopped = "grid_filled"
    return SearchSummary(stopped, duplicates, asks, stopped == "budget")


MAX_NEIGHBOUR_ROUNDS = 5
MAX_EDGE_EXTENSION_STEPS = 3  # per param per side, over the whole stage attempt


@dataclass
class EdgeExtension:
    """Settings and log of the grid-edge extension of one optimize attempt."""

    admissible: Callable[[dict], str | None]  # hard_bound_check(...)
    budget_seconds: float | None = None
    started: float = 0.0
    reserve_trials: int = 0  # the neighbour round still to come
    clock: Callable[[], float] = time.monotonic
    max_steps: int = MAX_EDGE_EXTENSION_STEPS
    steps: list = field(default_factory=list)
    stops: list = field(default_factory=list)
    tag_kind: bool = False  # excluded neighbour rows carry ``excluded_kind`` (param_edge_at_bound_pass runs)
    better_beyond_cap: list = field(default_factory=list)

    def over_budget(self, n_done: int) -> str | None:
        if self.budget_seconds is None or n_done == 0:
            return None
        elapsed = self.clock() - self.started
        per_trial = elapsed / n_done
        needed = elapsed + per_trial * (1 + self.reserve_trials)
        if needed > self.budget_seconds:
            return (
                f"{elapsed:.0f}s elapsed + {per_trial:.0f}s x (1 extension + {self.reserve_trials} neighbours) "
                f"> budget {self.budget_seconds:.0f}s"
            )
        return None

    def report(self) -> dict:
        return {
            "max_steps_per_param_side": self.max_steps,
            "n_trials": len(self.steps),
            "steps": self.steps,
            "stops": self.stops,
            # Points beyond the cap that beat the best at the time (never made the best):
            # the declared grid was too narrow for this spec.
            "better_beyond_cap": bool(self.better_beyond_cap),
            "better_beyond_cap_points": self.better_beyond_cap,
        }

    def note_beyond_cap(self, runner: TrialRunner) -> None:
        """Move the runner's better-but-ineligible trials into the report."""

        for result in runner.better_ineligible:
            self.better_beyond_cap.append(
                {
                    "trial_no": result.trial_no,
                    "params": result.params,
                    "source": result.source,
                    "sharpe_daily": result.sharpe,
                    "total_pnl_twd": (result.metrics or {}).get("total_pnl_twd"),
                    "best_trial_no_then": runner.best.trial_no if runner.best is not None else None,
                    "best_sharpe_daily_then": runner.best.sharpe if runner.best is not None else None,
                }
            )
        runner.better_ineligible.clear()


def extend_edges(runner: TrialRunner, grid: ParamGrid, extension: EdgeExtension, round_no: int = 1) -> None:
    """Step outward from every declared-grid edge the best sits on (see the module docstring)."""

    for name in grid.names:
        spec = grid.specs[name]
        for direction in (-1, 1):
            edge = spec.high if direction > 0 else spec.low
            while runner.best is not None:
                best = runner.best
                value = best.params[name]
                if direction * (value - edge) < -1e-9:
                    break  # not on (or beyond) this edge: nothing to extend
                target = spec.cast(value + direction * spec.step)
                params = {**best.params, name: target}
                stop = {"round": round_no, "param": name, "direction": direction, "best_value": value, "value": target}
                if not grid.within_cap(params, extension.max_steps):
                    extension.stops.append({**stop, "stopped_because": "max_steps"})
                    break
                existing = next((r for r in runner.results if grid.key(r.params) == grid.key(params)), None)
                if existing is not None:
                    extension.stops.append({**stop, "stopped_because": "not_better", "trial_no": existing.trial_no})
                    break
                reason = extension.admissible(params)
                if reason is not None:
                    extension.stops.append({**stop, "stopped_because": "hard_bound", "detail": reason})
                    break
                budget = extension.over_budget(len(runner.results))
                if budget is not None:
                    extension.stops.append({**stop, "stopped_because": "budget", "detail": budget})
                    break
                result = runner.evaluate(params, "edge_extension")
                better = runner.best is result
                extension.steps.append(
                    {
                        **stop,
                        "step": round(abs(target - edge) / spec.step),
                        "trial_no": result.trial_no,
                        "state": result.state,
                        "sharpe_daily": result.sharpe,
                        "total_pnl_twd": (result.metrics or {}).get("total_pnl_twd"),
                        "became_best": better,
                    }
                )
                if not better:
                    extension.stops.append({**stop, "stopped_because": "not_better", "trial_no": result.trial_no})
                    break


def complete_neighbourhood(
    runner: TrialRunner, grid: ParamGrid, steps: int, extension: EdgeExtension | None = None
) -> tuple[list[dict], bool]:
    """Extend grid edges, then evaluate the best point's neighbours, until the best stops moving.

    A neighbour can beat the search's best; the best then moves and its own
    edges and neighbours are handled in the next round. Returns the final
    best's neighbourhood and whether it settled within ``MAX_NEIGHBOUR_ROUNDS``.
    """

    rows: list[dict] = []
    admissible = extension.admissible if extension is not None else None
    if extension is not None:
        runner.eligible = lambda params: grid.within_cap(params, extension.max_steps)
    for round_no in range(1, MAX_NEIGHBOUR_ROUNDS + 1):
        if extension is not None:
            extend_edges(runner, grid, extension, round_no)
        before = runner.best.trial_no if runner.best is not None else None
        rows = evaluate_neighbours(
            runner, grid, steps, admissible, tag_kind=extension is not None and extension.tag_kind
        )
        if extension is not None:
            extension.note_beyond_cap(runner)
            for row in rows:
                row["beyond_cap"] = row["in_grid"] and not grid.within_cap(row["params"], extension.max_steps)
        after = runner.best.trial_no if runner.best is not None else None
        if after == before:
            return rows, True
    return rows, False


def evaluate_neighbours(
    runner: TrialRunner,
    grid: ParamGrid,
    steps: int,
    admissible: Callable[[dict], str | None] | None = None,
    *,
    tag_kind: bool = False,
) -> list[dict]:
    """Backtest every missing one-step neighbour of the best point; return the neighbourhood.

    Without ``admissible`` a neighbour off the declared grid is excluded (the
    pre-extension behaviour). With it, such a neighbour is looked up among the
    evaluated trials (edge extensions) or backtested, and excluded only when
    it violates a validator hard bound.
    """

    if runner.best is None:
        return []
    by_key = {grid.key(result.params): result for result in runner.results}
    best_params = dict(runner.best.params)
    rows = []
    for neighbour in grid.neighbours(best_params, steps):
        result = None
        params = neighbour["candidate"]
        key = grid.key(params)
        excluded_because = None
        if neighbour["in_grid"] or admissible is not None:
            result = by_key.get(key)
            if result is None and not neighbour["in_grid"]:
                excluded_because = admissible(params)
            if result is None and excluded_because is None:
                result = runner.evaluate(params, "neighbour")
                by_key[key] = result
        else:
            excluded_because = "outside the declared grid"
        evaluable = result is not None
        metrics = (result.metrics if result is not None else None) or {}
        row_kind = {}
        if tag_kind and not evaluable:
            if not neighbour["in_grid"] and admissible is None:
                kind = EXCLUDED_DECLARED_GRID
            else:
                kind = EXCLUDED_COMPONENT_BOUND if isinstance(excluded_because, BoundReason) else EXCLUDED_OTHER
            row_kind = {"excluded_kind": kind}
        rows.append(
            {
                **row_kind,
                "param": neighbour["param"],
                "direction": neighbour["direction"],
                # in_grid: inside the effective (declared + extended) range, i.e. counted by param_stability
                "in_grid": evaluable,
                "on_declared_grid": neighbour["in_grid"],
                "params": params if evaluable else None,
                "trial_no": result.trial_no if result is not None else None,
                "state": result.state if result is not None else None,
                "total_pnl_twd": metrics.get("total_pnl_twd") if result is not None else None,
                "sharpe_daily": metrics.get("sharpe_daily"),
                "excluded_because": None if evaluable else excluded_because,
            }
        )
    return rows
