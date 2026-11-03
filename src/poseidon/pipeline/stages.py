"""The concrete stages of the v1 plan: spec_generate, spec_validate, backtest, optimize."""

from __future__ import annotations

import time

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.pipeline.plan import (
    V3_RUN_CLASSES,
    is_routed,
)
from poseidon.pipeline.spec_validation import V2_REQUIRED, get_spec_validator
from poseidon.pipeline.stage_api import (
    BaseStage,
    StageResult,
    _DataStage,
    _step2_config_missing,
)
from poseidon.pipeline.state import Verdict


class SpecGenerateStage(BaseStage):
    """Pass a hand-written spec through, or generate one from the idea with the LLM (Step 4).

    An idea-only run frozen at CONFIG_VERSION 7 (routed) generates with spec_generate v3 first
    (:func:`poseidon.pipeline.spec_llm_v3.generate_routed`); its report's ``route`` picks the class whose
    frozen config the rest of the run uses. ``review(idea_text, clauses, spec) -> [{"path", "msg"}]`` is the
    reviewer of v3 specs; by default the JEV ``FaithfulnessReviewer`` (report-only, one repair round).
    ``refdata`` (() -> store) is read, before any LLM call, for the first day of every daily series
    (``v3.config.series_first_days``): a daily spec's window start is chosen from it.
    """

    name = "spec_generate"

    def __init__(self, *, client_factory=None, pin_provider=None, review=None, jev_client_factory=None, refdata=None):
        self._client_factory = client_factory
        self._pin_provider = pin_provider
        self._review = review  # None: the JEV reviewer on the run's frozen spec_faithfulness section
        self._jev_client_factory = jev_client_factory
        self._refdata = refdata

    def daily_refdata(self, config):
        """``(series_since, warmup)`` for choosing a daily window start; ``(None, None)`` when the run froze no
        daily class. ``series_since``: first day per daily series from the refdata manifests; ``warmup``: the
        session calendar of the daily classes' frozen ``data.warmup`` rule (None when they froze none)."""

        by_class = config.get("by_class") or {}
        daily = [variant for name, variant in by_class.items() if name in V3_RUN_CLASSES]
        if not daily:
            return None, None
        from poseidon.pipeline.v3.config import series_first_days, session_warmup

        if self._refdata is not None:
            store = self._refdata()
        else:
            from poseidon.pipeline.refdata.store import RefdataStore

            store = RefdataStore()
        return series_first_days(store), session_warmup(daily[0]["data"].get("warmup"), store)

    def input_payload(self, ctx):
        payload = {
            "stage": self.name,
            "config_sha256": ctx.config_sha256,
            "idea_text": ctx.idea_text,
            "input_spec_json": ctx.input_spec_json,
        }
        if ctx.input_spec_json is None:
            from poseidon.pipeline.spec_llm import PROMPT_SHA256

            payload["prompt_sha256"] = PROMPT_SHA256
            if is_routed(ctx.config):
                from poseidon.pipeline.spec_llm_v3 import PROMPT_SHA256_V3

                payload["prompt_v3_sha256"] = PROMPT_SHA256_V3
        return payload

    def run(self, ctx):
        if ctx.input_spec_json is not None:
            spec = ctx.input_spec_json
            return StageResult.passed(
                {"origin": "input_spec_json", "spec": spec, "spec_sha256": content_sha256(spec)},
            )
        from poseidon.core.config import settings
        from poseidon.pipeline.spec_llm import generate_spec

        ctx.check_stop()
        pin = self._pin_provider() if self._pin_provider else settings.pipeline_openrouter_key_label
        if is_routed(ctx.config):
            from poseidon.pipeline.refdata.store import RefdataError
            from poseidon.pipeline.spec_llm_v3 import generate_routed

            try:
                series_since, warmup = self.daily_refdata(ctx.config)
            except (RefdataError, OSError) as error:
                reason = f"refdata is not readable for the daily window start: {error}"[:300]
                return StageResult(Verdict.ERROR, reason, {"summary": f"spec_generate ⚠️ {reason}"[:300]})
            outcome = generate_routed(
                ctx.idea_text,
                ctx.config,
                pin=pin,
                review=self._review,
                jev_client_factory=self._jev_client_factory,
                series_since=series_since,
                warmup=warmup,
                client_factory=self._client_factory,
            )
        else:
            outcome = generate_spec(
                ctx.idea_text, ctx.config.get("spec_generate"), pin=pin, client_factory=self._client_factory
            )
        return StageResult(outcome.verdict, outcome.reason, outcome.report, outcome.files)


class SpecValidateStage(BaseStage):
    name = "spec_validate"
    requires = ("spec_generate",)

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload["validator"] = ctx.config["spec_validator"]
        return payload

    def run(self, ctx):
        spec = (ctx.upstream["spec_generate"].report or {}).get("spec")
        validator = get_spec_validator(ctx.config["spec_validator"])
        ok, errors, unsupported = validator.validate_spec(spec)
        report = {
            "validator": validator.name,
            "spec_sha256": content_sha256(spec),
            "errors": list(errors),
            "unsupported": list(unsupported),
        }
        if errors:
            return StageResult.failed(f"spec invalid: {len(errors)} error(s)", report)
        if unsupported:
            return StageResult.unsupported("; ".join(unsupported), report)
        if not ok:  # a validator that reports not-ok without saying why
            return StageResult.failed("spec rejected by validator", report)
        resolve = getattr(validator, "resolve", None)
        if resolve is None:
            report["spec"] = spec
            return StageResult.passed(report)
        # The concrete spec ($param defaults substituted) is what backtest runs;
        # the template and its grid are what optimize searches.
        resolved = resolve(spec)
        report.update(
            {
                "spec": resolved["spec"],
                "resolved_spec_sha256": content_sha256(resolved["spec"]),
                "template": resolved["template"],
                "params": resolved["params"],
            }
        )
        interval = (ctx.config.get("data") or {}).get("interval")
        if interval is not None and resolved["spec"]["interval"] != interval:
            reason = f"pipeline data window is {interval} bars, spec asks for {resolved['spec']['interval']}"
            return StageResult.unsupported(reason, report)
        return StageResult.passed(report)


class BacktestStage(_DataStage):
    """The concrete spec on the train part of the frozen window (never the holdout)."""

    name = "backtest"
    requires = ("spec_validate",)
    lease_seconds = 1800

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update(
            {
                "spec_sha256": (ctx.upstream["spec_validate"].report or {}).get("spec_sha256"),
                "data": ctx.config.get("data"),
                "backtest": ctx.config.get("backtest"),
                "part": "train",
            }
        )
        return payload

    def run(self, ctx):
        from poseidon.pipeline.backtesting import backtest_settings, run_backtest

        blocked = _step2_config_missing(ctx)
        if blocked is not None:
            return blocked
        upstream = ctx.upstream["spec_validate"].report or {}
        spec = upstream.get("spec")
        if not isinstance(spec, dict) or spec.get("dsl_version") != 2:
            return StageResult.unsupported(V2_REQUIRED)
        bars = self.bars(ctx, "train")
        strategy = self.strategy_builder(ctx, spec).build()
        outcome = run_backtest(strategy, bars.frame, ctx.config["backtest"], should_stop=ctx.should_stop)
        report = {
            "spec_sha256": content_sha256(spec),
            "data": bars.provenance(),
            "settings": backtest_settings(ctx.config["backtest"]),
            "metrics": outcome.metrics,
            "files": {"trades": "trades.csv", "daily_pnl": "daily_pnl.csv"},
        }
        return StageResult.passed(report, files=outcome.files())


class OptimizeStage(_DataStage):
    """TPE over the spec's ``$param`` grid on the train part; every trial goes to the ledger."""

    name = "optimize"
    requires = ("spec_validate", "backtest")
    lease_seconds = 1800

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update(
            {
                "spec_sha256": (ctx.upstream["spec_validate"].report or {}).get("spec_sha256"),
                "data": ctx.config.get("data"),
                "backtest": ctx.config.get("backtest"),
                "optimize": ctx.config.get("optimize"),
                "seed": ctx.config.get("seed"),
                "part": "train",
            }
        )
        return payload

    def run(self, ctx):
        from poseidon.pipeline import optimization as opt
        from poseidon.pipeline.backtesting import backtest_settings
        from poseidon.pipeline.gates.param_stability import EDGE_AT_BOUND_KEY
        from poseidon.pipeline.trials import TRIAL_COMPLETE, MemoryTrialLedger
        from poseidon.strategies.dsl.v2 import validate_v2

        blocked = _step2_config_missing(ctx)
        if blocked is not None:
            return blocked
        validated = ctx.upstream["spec_validate"].report or {}
        backtest = ctx.upstream["backtest"].report or {}
        template = validated.get("template") or validated.get("spec")
        if not isinstance(template, dict) or template.get("dsl_version") != 2:
            return StageResult.unsupported(V2_REQUIRED)
        settings = ctx.config["optimize"]
        ledger = ctx.trials if ctx.trials is not None else MemoryTrialLedger()
        grid = opt.ParamGrid(validate_v2(template).params)
        started = time.monotonic()

        report = {
            "objective": settings["objective"],
            "objective_convention": "per-trading-day Sharpe of daily PnL (not annualised); see backtest report",
            "sampler": {"name": settings["sampler"], "seed": ctx.config["seed"]},
            "max_trials": settings["max_trials"],
            "param_grid": grid.describe(),
            "grid_size": grid.size() if grid.names else 1,
            "settings": backtest_settings(ctx.config["backtest"]),
        }
        files: dict[str, bytes] = {}
        if not grid.names:
            # Nothing to tune: the base backtest is the only trial, so N counts it.
            data = backtest.get("data") or {}
            metrics = backtest.get("metrics") or {}
            runner = opt.TrialRunner(
                build=None,
                bars=None,
                backtest_config=ctx.config["backtest"],
                ledger=ledger,
                window_sha256=data["window_sha256"],
                should_stop=ctx.should_stop,
                check_stop=ctx.check_stop,
            )
            state = TRIAL_COMPLETE if metrics.get("sharpe_daily") is not None else opt.TRIAL_PRUNED
            runner.record({}, "base", state, metrics)
            report.update({"data": data, "search": {"stopped_because": "no_params", "duplicates_skipped": 0}})
            neighbours, settled, factory, build, extension = [], True, None, None, None
        else:
            bars = self.bars(ctx, "train")
            factory = self.strategy_builder(ctx, template)
            # Off-grid values (edge extensions) are built from a widened copy of the template.
            build = opt.off_grid_builder(
                factory, template, grid, lambda effective: self.strategy_builder(ctx, effective)
            )
            runner = opt.TrialRunner(
                build=build,
                bars=bars.frame,
                backtest_config=ctx.config["backtest"],
                ledger=ledger,
                window_sha256=bars.window_sha256,
                should_stop=ctx.should_stop,
                check_stop=ctx.check_stop,
            )
            search = opt.tpe_search(
                runner,
                grid,
                factory.defaults,
                seed=int(ctx.config["seed"]),
                max_trials=int(settings["max_trials"]),
                max_consecutive_duplicates=int(settings["max_consecutive_duplicates"]),
                budget_seconds=settings.get("budget_seconds"),
                reserve_trials=2 * len(grid.names) * int(settings["neighbour_steps"]),
            )
            extension = opt.EdgeExtension(
                admissible=opt.hard_bound_check(template, grid),
                budget_seconds=settings.get("budget_seconds"),
                started=started,
                reserve_trials=2 * len(grid.names) * int(settings["neighbour_steps"]),
                tag_kind=EDGE_AT_BOUND_KEY in ctx.config["thresholds"],
            )
            neighbours, settled = opt.complete_neighbourhood(runner, grid, int(settings["neighbour_steps"]), extension)
            report.update(
                {
                    "data": bars.provenance(),
                    "search": {
                        "stopped_because": search.stopped_because,
                        "duplicates_skipped": search.duplicates_skipped,
                        "asks": search.asks,
                        "budget_exhausted": search.budget_exhausted,
                    },
                }
            )

        results = runner.results
        by_state: dict[str, int] = {}
        for result in results:
            by_state[result.state] = by_state.get(result.state, 0) + 1
        report.update(
            {
                "n_trials": len(results),
                "n_trials_by_source": {
                    source: sum(1 for r in results if r.source == source)
                    for source in ("base", "search", "edge_extension", "neighbour")
                },
                "n_trials_by_state": dict(sorted(by_state.items())),
                "neighbour_steps": int(settings["neighbour_steps"]),
                "neighbours": neighbours,
                "neighbourhood_settled": settled,
                "trials_file": "trials.json",
            }
        )
        if grid.names:
            # The spec's declared grid stays as declared (``param_grid``); this is what was evaluated.
            report["edge_extension"] = extension.report()
            report["effective_range"] = grid.effective_range(results, extension.max_steps)
        files["trials.json"] = canonical_json([result.row() for result in results]).encode("utf-8")
        best = runner.best
        elapsed = time.monotonic() - started
        if best is None:
            report["best"] = None
            return StageResult(
                Verdict.FAIL,
                f"no trial produced a daily Sharpe ({len(results)} trials, {elapsed:.0f}s)",
                report,
                files,
            )
        best_spec = factory.to_config_dict(build(best.params)) if grid.names else validated["spec"]
        report["best"] = {
            "trial_no": best.trial_no,
            "params": best.params,
            "metrics": best.metrics,
        }
        report["best_spec"] = best_spec
        report["best_spec_sha256"] = content_sha256(best_spec)
        if best.outcome is not None:
            for name, data in best.outcome.files().items():
                files[f"best_{name}"] = data
        reason = (
            f"{len(results)} trials in {elapsed:.0f}s; best trial {best.trial_no} "
            f"sharpe_daily={best.sharpe:.4f} params={canonical_json(best.params)}"
        )
        return StageResult.passed(report, reason=reason, files=files)
