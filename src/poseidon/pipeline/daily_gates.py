"""Daily (spec v3) gates that re-run the engine: cost stress, regime split, benchmark, execution timing, wfe, holdout."""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from datetime import date, timedelta
from typing import ClassVar

import pandas as pd

from poseidon.core.canonical import canonical_json
from poseidon.pipeline import optimization as opt
from poseidon.pipeline.daily_stages import (
    EXECUTION_TIMING,
    STOCK_UNDERLYING,
    UNDERLYING_ROLL_DAYS_BEFORE,
    DailyStage,
    DailyWindow,
    EngineTrialRunner,
    SpecRunner,
    daily_metrics,
    engine_files,
    window_sha,
)
from poseidon.pipeline.daily_validate import price_points
from poseidon.pipeline.engine.data import stock_bars
from poseidon.pipeline.gate_stages import to_stage_result
from poseidon.pipeline.gates import benchmark, cost_stress, execution_timing, holdout, regime_split
from poseidon.pipeline.gates import wfe as wfe_gate
from poseidon.pipeline.gates.base import failed, invalid
from poseidon.pipeline.gates.daily import (
    LIQUIDATION_EVENTS,
    DailyResult,
    choose_walk_forward,
    from_engine,
    underlying_returns,
)
from poseidon.pipeline.plan import V3_GATE_STAGES
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES
from poseidon.pipeline.refdata.engine_input import StockDataGap, align_stock_bars
from poseidon.pipeline.refdata.store import DatasetMissingError
from poseidon.pipeline.refdata.tw_corporate_actions import require_actions
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.trials import MemoryTrialLedger
from poseidon.pipeline.v3.compiler import NotCompilable, compile_spec


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
    try:
        booked = stock_bars(align_stock_bars(bars, sessions, symbol), require_actions(store, symbol))
    except (StockDataGap, ValueError, DatasetMissingError) as error:
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

    from poseidon.pipeline.v3.components_sizing import round_units
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
