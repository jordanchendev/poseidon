"""Daily (spec v3) stages that validate, backtest and optimize, and the ledger-only gates that read optimize's trials."""

from __future__ import annotations

import time
from collections import Counter
from datetime import date
from typing import ClassVar

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.pipeline import optimization as opt
from poseidon.pipeline.daily_stages import (
    V3_WIRING_VERSION,
    DailyStage,
    DailyWindow,
    EngineTrialRunner,
    WindowStore,
    check_components,
    daily_metrics,
    engine_files,
    frozen_series_since,
    instrument_kinds,
    moved_start_text,
    pct,
    summary_text,
    window_sha,
)
from poseidon.pipeline.gate_stages import DsrStage, ParamStabilityStage
from poseidon.pipeline.gates import liquidation
from poseidon.pipeline.gates.daily import from_engine
from poseidon.pipeline.gates.param_stability import EDGE_AT_BOUND_KEY
from poseidon.pipeline.refdata.datasets import FUTURES
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.refdata.tw_stocks import read_bars
from poseidon.pipeline.stage_api import StageResult
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.trials import MemoryTrialLedger
from poseidon.pipeline.v3.config import ENGINE, SPEC_VALIDATOR, session_warmup
from poseidon.pipeline.v3.validation import validate_spec_v3
from poseidon.pipeline.v3.validation_types import FeasibilityContext


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
