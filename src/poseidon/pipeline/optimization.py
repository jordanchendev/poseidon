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
* Afterwards every one-at-a-time +/-1 step neighbour of the best point that
  was not evaluated yet is backtested and logged as a trial too, so
  ``param_stability`` always has the full neighbourhood.
"""

from __future__ import annotations

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
    source: str  # base | search | neighbour
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

    def neighbours(self, best: dict, steps: int = 1) -> list[dict]:
        """One-at-a-time ``+/- steps`` grid neighbours of ``best`` (``params`` None when off-grid)."""

        out = []
        for name in self.names:
            spec = self.specs[name]
            for direction in (-1, 1):
                value = best[name] + direction * steps * spec.step
                in_grid = spec.low - 1e-9 <= value <= spec.high + 1e-9
                params = {**best, name: spec.cast(value)} if in_grid else None
                out.append({"param": name, "direction": direction, "in_grid": in_grid, "params": params})
        return out


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
    ):
        self.build = build
        self.bars = bars
        self.backtest_config = backtest_config
        self.ledger = ledger
        self.window_sha256 = window_sha256
        self.should_stop = should_stop
        self.check_stop = check_stop
        self.results: list[TrialResult] = []
        self.best: TrialResult | None = None

    def record(
        self, params: dict, source: str, state: str, metrics: dict | None, error: str | None = None, outcome=None
    ):
        result = TrialResult(len(self.results) + 1, params, source, state, metrics, error, outcome)
        self.ledger.append(make_record(result.trial_no, params, state, metrics, self.window_sha256))
        self.results.append(result)
        if result.state == TRIAL_COMPLETE and (self.best is None or result.sharpe > self.best.sharpe):
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
            len(self.results) + 1,
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


def complete_neighbourhood(runner: TrialRunner, grid: ParamGrid, steps: int) -> tuple[list[dict], bool]:
    """Evaluate the best point's neighbours until the best stops moving.

    A neighbour can beat the search's best; the best then moves and its own
    neighbours are evaluated in the next round. Returns the final best's
    neighbourhood and whether it settled within ``MAX_NEIGHBOUR_ROUNDS``.
    """

    rows: list[dict] = []
    for _ in range(MAX_NEIGHBOUR_ROUNDS):
        before = runner.best.trial_no if runner.best is not None else None
        rows = evaluate_neighbours(runner, grid, steps)
        after = runner.best.trial_no if runner.best is not None else None
        if after == before:
            return rows, True
    return rows, False


def evaluate_neighbours(runner: TrialRunner, grid: ParamGrid, steps: int) -> list[dict]:
    """Backtest every missing one-step neighbour of the best point; return the neighbourhood."""

    if runner.best is None:
        return []
    by_key = {grid.key(result.params): result for result in runner.results}
    best_params = dict(runner.best.params)
    rows = []
    for neighbour in grid.neighbours(best_params, steps):
        result = None
        if neighbour["in_grid"]:
            key = grid.key(neighbour["params"])
            result = by_key.get(key)
            if result is None:
                result = runner.evaluate(neighbour["params"], "neighbour")
                by_key[key] = result
        metrics = (result.metrics if result is not None else None) or {}
        rows.append(
            {
                "param": neighbour["param"],
                "direction": neighbour["direction"],
                "in_grid": neighbour["in_grid"],
                "params": neighbour["params"],
                "trial_no": result.trial_no if result is not None else None,
                "state": result.state if result is not None else None,
                "total_pnl_twd": metrics.get("total_pnl_twd") if result is not None else None,
                "sharpe_daily": metrics.get("sharpe_daily"),
            }
        )
    return rows
