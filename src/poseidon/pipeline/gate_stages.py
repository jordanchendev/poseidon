"""The validation gates wired as pipeline stages (Step 3b).

Order (``plan.V1_STAGE_PLAN``): param_stability -> dsr -> cost_stress ->
regime_split -> benchmark -> wfe -> holdout. Cheap ledger-only gates run
first, then the gates that re-run the strategy on the train part, and the
locked holdout is looked at exactly once, last, only for a candidate that
passed everything on train (``holdout`` requires every train gate, and a gate
``fail`` closes the run). ``benchmark`` exists from CONFIG_VERSION 5; runs
frozen at 4 keep their plan without it (``plan.STAGE_INTRODUCED_AT``).

Every gate consumes the optimize artifact's concrete ``best_spec`` (and its
best trial's metrics). Each stage report is::

    {"summary": "<gates.summary_line>", "gate", "verdict", "reason_code",
     "reason", "headline", "gate_report": <the gate's full report>,
     "inputs": <what the stage fed the gate, and from where>}

The Kairos viewer reads ``artifact_json.report.summary``. Thresholds come only
from the run's frozen ``config_json["thresholds"]``; gate-stage settings from
``config_json["gates"]`` (CONFIG_VERSION 3). Runs frozen before that end
``unsupported`` at the first gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline import gates
from poseidon.pipeline.gates import benchmark, cost_stress, dsr, holdout, param_stability, regime_split, wfe
from poseidon.pipeline.gates.base import CostBreakdown, GateResult, ResultSummary, Trade, failed, invalid
from poseidon.pipeline.plan import GATE_STAGES, GATES_CONFIG_VERSION
from poseidon.pipeline.spec_validation import V2_REQUIRED
from poseidon.pipeline.stages import StageResult, _DataStage, _step2_config_missing
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.window import MARKET_TZ, SPLIT_LOCAL_TIME, DataWindow

GATES_CONFIG_MISSING = "run config predates Step 3 (no frozen gate settings); create a new run"
BENCHMARK_CONFIG_MISSING = "run config predates the benchmark gate (config version < 5); create a new run"
WIRING_VERSION = 1  # part of every gate stage's input hash


class GateArtifactError(ValueError):
    """An upstream file does not match its recorded sha256."""


# ----------------------------------------------------------------- helpers


def _gates_config_missing(ctx) -> StageResult | None:
    blocked = _step2_config_missing(ctx)
    if blocked is not None:
        return blocked
    if int(ctx.config.get("version") or 0) < GATES_CONFIG_VERSION or not isinstance(ctx.config.get("gates"), dict):
        return StageResult.unsupported(GATES_CONFIG_MISSING)
    return None


def gate_report(result: GateResult, inputs: dict) -> dict:
    return {
        "summary": gates.summary_line(result),
        "gate": result.gate,
        "verdict": result.verdict.value,
        "reason_code": result.reason_code,
        "reason": result.reason,
        "headline": result.headline,
        "gate_report": result.report,
        "inputs": inputs,
    }


def to_stage_result(result: GateResult, inputs: dict, files: dict | None = None) -> StageResult:
    report = gate_report(result, inputs)
    if result.passed:
        return StageResult.passed(report, reason=report["summary"], files=files)
    return StageResult(Verdict.FAIL, result.reason, report, files or {})


def summary_from_metrics(metrics: dict) -> ResultSummary:
    """A gate ``ResultSummary`` from a pipeline backtest's TWD metrics (Sharpe per trading day)."""

    costs = metrics.get("costs") or {}
    return ResultSummary(
        total_pnl_twd=metrics.get("total_pnl_twd"),
        sharpe=metrics.get("sharpe_daily"),
        trade_count=int(metrics.get("n_trades") or 0),
        costs=CostBreakdown(
            commission_twd=costs.get("commission_twd"),
            tax_twd=costs.get("tax_twd"),
            slippage_twd=costs.get("slippage_twd_estimate"),
        ),
    )


def headline_metrics(metrics: dict) -> dict:
    keys = ("total_pnl_twd", "sharpe_daily", "sharpe_annualised", "n_trades", "n_obs", "win_rate", "max_drawdown_twd")
    return {key: metrics.get(key) for key in keys}


def family_trials(ctx) -> list:
    """Every ledger row of the run's family (the dispatcher's view), else this ledger's rows."""

    if ctx.family is not None:
        return list(ctx.family.trials())
    return list(getattr(ctx.trials, "records", []) or [])


def read_upstream_file(upstream, name: str) -> bytes | None:
    """An upstream attempt's file, verified against its manifest; ``None`` if it never wrote one."""

    entry = next((item for item in upstream.files if item.get("name") == name), None)
    if entry is None or not upstream.artifact_uri:
        return None
    directory = Path(urllib.request.url2pathname(urllib.parse.urlparse(upstream.artifact_uri).path))
    data = (directory / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise GateArtifactError(f"{upstream.stage} file {name} does not match its recorded sha256")
    return data


def parse_trades(data: bytes) -> list[Trade]:
    frame = pd.read_csv(io.BytesIO(data))
    return [
        Trade(
            entry_time=pd.Timestamp(row.entry_time).to_pydatetime(),
            exit_time=pd.Timestamp(row.exit_time).to_pydatetime(),
            side=row.side,
            pnl_twd=float(row.pnl_twd),
            fees_twd=float(row.fees_twd),
        )
        for row in frame.itertuples(index=False)
    ]


def daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """TX daily OHLCV per trading day (night session -> the next day session), indexed by that date.

    Resampled from the same label-normalised cached 5m bars the backtests use,
    so both come from one source; a day's bar completes at its day-session close.
    """

    from poseidon.pipeline.backtesting import trading_days

    grouped = frame.groupby(trading_days(frame.index))
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    )
    daily.index = [stamp.date() for stamp in daily.index]
    return daily


DEFAULT_HAC_LAGS = 5
BUY_AND_HOLD = (
    "TX buy-and-hold: contracts x point value x (day-session close - previous trading day's close) per trading "
    "day (night session -> next day session), no costs; the first day with no earlier close uses close - open. "
    "Thalassa TX is the Shioaji R1 near-month series, unadjusted: a contract roll is a price jump inside the "
    "series, seen alike by the strategy (when it holds across it) and by buy-and-hold"
)


def buy_and_hold_daily(frame: pd.DataFrame, *, point_value: float, contracts: float) -> pd.Series:
    """TX buy-and-hold PnL (TWD) per trading day of ``frame``, indexed by that date (see ``BUY_AND_HOLD``)."""

    daily = daily_bars(frame)
    close = daily["close"].astype(float)
    pnl = close.diff()
    if len(pnl):
        pnl.iloc[0] = close.iloc[0] - float(daily["open"].iloc[0])
    return pnl * float(point_value) * float(contracts)


def daily_series(daily: pd.DataFrame) -> pd.Series:
    """``pnl_twd`` of a ``daily_pnl`` frame (or its csv), indexed by trading-day date."""

    series = daily["pnl_twd"].astype(float)
    series.index = [pd.Timestamp(day).date() for day in daily.index]
    return series


def parse_daily(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), index_col="trading_day")


def align_daily(strategy: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Both series on their common trading days, plus what was dropped."""

    common = sorted(set(strategy.index) & set(benchmark.index))
    info = {
        "days": len(common),
        "first": common[0].isoformat() if common else None,
        "last": common[-1].isoformat() if common else None,
        "strategy_only_days": len(set(strategy.index) - set(common)),
        "benchmark_only_days": len(set(benchmark.index) - set(common)),
    }
    return strategy.loc[common], benchmark.loc[common], info


def time_in_market(trades: list[Trade], index: pd.DatetimeIndex) -> float | None:
    """Share of the bars in ``index`` (sorted) at which a position was open (entry <= bar < exit)."""

    if not len(index):
        return None
    stamps = index.asi8
    held = 0
    for trade in trades:
        entry = pd.Timestamp(trade.entry_time)
        exit_ = pd.Timestamp(trade.exit_time)
        entry = entry.tz_localize("UTC") if entry.tzinfo is None else entry.tz_convert("UTC")
        exit_ = exit_.tz_localize("UTC") if exit_.tzinfo is None else exit_.tz_convert("UTC")
        held += int(np.searchsorted(stamps, exit_.value, "left") - np.searchsorted(stamps, entry.value, "left"))
    return held / len(index)


def benchmark_series(ctx, frame: pd.DataFrame, daily: pd.DataFrame, trades: list[Trade], *, from_day=None) -> dict:
    """The strategy's daily PnL and TX buy-and-hold over ``frame``'s trading days, aligned.

    ``from_day`` drops earlier (warm-up) days after the buy-and-hold diff, so the
    first evaluated day still uses the previous close. Returns ``strategy`` and
    ``benchmark`` (aligned Series), ``time_in_market`` and ``provenance``.
    """

    from poseidon.backtest.cost_model import get_cost_model

    backtest = ctx.config["backtest"]
    point_value = float(get_cost_model(backtest["cost_model"]).point_value)
    contracts = float(backtest["sizing"]["contracts"])
    lags = int(((ctx.config.get("gates") or {}).get("benchmark") or {}).get("hac_lags", DEFAULT_HAC_LAGS))
    annualisation = float(backtest["sharpe"]["annualisation_factor"])
    bh = buy_and_hold_daily(frame, point_value=point_value, contracts=contracts)
    strategy = daily_series(daily)
    evaluated = frame.index
    if from_day is not None:
        bh = bh[[day >= from_day for day in bh.index]]
        strategy = strategy[[day >= from_day for day in strategy.index]]
        evaluated = frame.index[trading_day_dates(frame.index) >= from_day]
    y, x, alignment = align_daily(strategy, bh)
    return {
        "strategy": y,
        "benchmark": x,
        "time_in_market": time_in_market(trades, evaluated),
        "provenance": {
            "buy_and_hold": BUY_AND_HOLD,
            "point_value_twd": point_value,
            "contracts": contracts,
            "hac_lags": lags,
            "annualisation_factor": annualisation,
            "alignment": alignment,
            "time_in_market": "share of the evaluated 5m bars with a position open (entry <= bar < exit)",
        },
    }


def benchmark_stats(ctx, frame, daily, trades, *, from_day=None) -> tuple[dict, dict]:
    """``benchmark.regress`` (plus ``time_in_market``) of ``benchmark_series``; returns (stats, provenance)."""

    series = benchmark_series(ctx, frame, daily, trades, from_day=from_day)
    provenance = series["provenance"]
    stats = benchmark.regress(
        series["strategy"].to_numpy(),
        series["benchmark"].to_numpy(),
        lags=provenance["hac_lags"],
        annualisation=provenance["annualisation_factor"],
    )
    stats["time_in_market"] = series["time_in_market"]
    return stats, provenance


def trading_day_dates(index: pd.DatetimeIndex) -> np.ndarray:
    from poseidon.pipeline.backtesting import trading_days

    return np.array([stamp.date() for stamp in trading_days(index)])


def annualised_return(metrics: dict, capital: float) -> float | None:
    """``total_pnl_twd / capital`` scaled to 252 trading days by the run's trading-day count."""

    n_obs = metrics.get("n_obs") or 0
    pnl = metrics.get("total_pnl_twd")
    if not n_obs or pnl is None:
        return None
    return float(pnl) / capital * 252.0 / float(n_obs)


def snap_to_split_time(moment: datetime) -> datetime:
    """The latest 08:00 Asia/Taipei at or before ``moment`` (between the night close and the day open)."""

    local = moment.astimezone(MARKET_TZ)
    snapped = datetime.combine(local.date(), SPLIT_LOCAL_TIME, tzinfo=MARKET_TZ)
    if snapped > local:
        snapped -= timedelta(days=1)
    return snapped.astimezone(UTC)


def walk_forward_windows(train_until: datetime, *, windows: int, is_days: int, oos_days: int) -> list[dict]:
    """Rolling IS/OOS windows anchored at the end of the train part (most recent data last)."""

    out = []
    for index in range(windows):
        oos_end = snap_to_split_time(train_until - timedelta(days=oos_days * (windows - 1 - index)))
        oos_start = snap_to_split_time(oos_end - timedelta(days=oos_days))
        is_start = snap_to_split_time(oos_start - timedelta(days=is_days))
        out.append({"index": index, "is_start": is_start, "oos_start": oos_start, "oos_end": oos_end})
    return out


def _slice(frame: pd.DataFrame, lower: datetime, upper: datetime) -> pd.DataFrame:
    return frame[(frame.index >= lower) & (frame.index < upper)]


# ----------------------------------------------------------------- base stage


class GateStage(_DataStage):
    """Shared plumbing: config/version guard, the optimize best spec, the report shape."""

    requires: ClassVar[tuple[str, ...]] = ("optimize",)
    lease_seconds: ClassVar[int] = 1800

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update({"gate": self.name, "gate_wiring_version": WIRING_VERSION})
        return payload

    def run(self, ctx):
        blocked = _gates_config_missing(ctx)
        if blocked is not None:
            return blocked
        optimize = ctx.upstream["optimize"].report or {}
        best_spec = optimize.get("best_spec")
        if not isinstance(best_spec, dict) or best_spec.get("dsl_version") != 2 or not optimize.get("best"):
            return StageResult.unsupported(V2_REQUIRED)
        return self.evaluate(ctx, optimize)

    def evaluate(self, ctx, optimize: dict) -> StageResult:  # pragma: no cover - interface
        raise NotImplementedError

    def backtest_spec(self, ctx, spec: dict, frame: pd.DataFrame, **kwargs):
        from poseidon.pipeline.backtesting import run_backtest

        strategy = self.strategy_builder(ctx, spec).build()
        return run_backtest(strategy, frame, ctx.config["backtest"], should_stop=ctx.should_stop, **kwargs)

    @staticmethod
    def best_source(optimize: dict) -> dict:
        best = optimize["best"]
        return {
            "best_trial_no": best.get("trial_no"),
            "best_params": best.get("params"),
            "best_spec_sha256": optimize.get("best_spec_sha256"),
        }


# ----------------------------------------------------------------- gates


class ParamStabilityStage(GateStage):
    """Gate 5 on the optimize neighbourhood (one-at-a-time +/-1 grid steps of the best point)."""

    name = "param_stability"
    lease_seconds = 300

    def evaluate(self, ctx, optimize):
        from poseidon.pipeline.trials import TRIAL_FAIL

        best = optimize["best"]
        neighbours, failed_counted = [], []
        for row in optimize.get("neighbours") or []:
            if not row.get("in_grid"):
                neighbours.append(param_stability.NeighbourResult(row["param"], int(row["direction"]), {}, None))
                continue
            pnl = row.get("total_pnl_twd")
            if row.get("state") == TRIAL_FAIL or pnl is None:
                # A grid neighbour whose backtest failed is counted as not
                # profitable (conservative), never silently excluded.
                failed_counted.append(
                    {"param": row["param"], "direction": row["direction"], "trial_no": row.get("trial_no")}
                )
                pnl = 0.0
            neighbours.append(param_stability.NeighbourResult(row["param"], int(row["direction"]), row["params"], pnl))
        result = param_stability.evaluate(
            param_stability.ParamStabilityInputs(
                best_params=best["params"],
                best_total_pnl_twd=(best.get("metrics") or {}).get("total_pnl_twd"),
                neighbours=neighbours,
            ),
            ctx.config["thresholds"],
        )
        inputs = {
            **self.best_source(optimize),
            "source": "optimize neighbours: every one-at-a-time +/-1 grid step of the best point, all backtested",
            "neighbour_steps": optimize.get("neighbour_steps"),
            "neighbourhood_settled": optimize.get("neighbourhood_settled"),
            "failed_neighbours_counted_unprofitable": failed_counted,
            # Declared grid vs evaluated range (grid-edge extension); a neighbour is excluded only
            # when it violates a DSL validator hard bound (see the neighbour rows' excluded_because).
            "effective_range": optimize.get("effective_range"),
            "better_beyond_cap": (optimize.get("edge_extension") or {}).get("better_beyond_cap"),
            "excluded_neighbours": [
                {"param": row["param"], "direction": row["direction"], "because": row.get("excluded_because")}
                for row in optimize.get("neighbours") or []
                if not row.get("in_grid")
            ],
        }
        return to_stage_result(result, inputs)


class DsrStage(GateStage):
    """Gate 4: the best trial's per-trading-day Sharpe against every trial of the family."""

    name = "dsr"
    lease_seconds = 300

    def evaluate(self, ctx, optimize):
        rows = family_trials(ctx)
        metrics = optimize["best"].get("metrics") or {}
        result = dsr.evaluate(
            dsr.DsrInputs(
                sharpe=metrics.get("sharpe_daily"),
                n_obs=metrics.get("n_obs"),
                skew=metrics.get("skew"),
                kurtosis=metrics.get("kurtosis"),
                trial_sharpes=dsr.trial_sharpes_from_ledger(rows),
            ),
            ctx.config["thresholds"],
        )
        has_run = rows and all(hasattr(row, "run_id") for row in rows)
        this_run = sum(1 for row in rows if row.run_id == ctx.run_id) if has_run else len(rows)
        inputs = {
            **self.best_source(optimize),
            "n_trials_used": len(rows),
            "n_trials_this_run": this_run,
            "n_trials_other_runs": len(rows) - this_run,
            "trial_states": dict(sorted(Counter(str(getattr(row, "state", None)) for row in rows).items())),
            "counting": "every pipeline_trials row of the family, incl. errored/aborted attempts and pruned trials",
            "order_note": "dsr runs before wfe: this run's per-window wfe trials are recorded later, so they are "
            "not in this N; they count toward N for later runs of the family",
            "sharpe_convention": "per trading day (sharpe_daily), not annualised; kurtosis non-excess",
        }
        return to_stage_result(result, inputs)


class CostStressStage(GateStage):
    """Gate 3: the best spec re-run on the train part with every cost x multiplier."""

    name = "cost_stress"

    def evaluate(self, ctx, optimize):
        from poseidon.pipeline.backtesting import daily_csv, stressed_cost_model, trades_csv

        thresholds = ctx.config["thresholds"]
        multiplier = float(thresholds["cost_stress_multiplier"])
        stop_ticks = float(ctx.config["gates"]["cost_stress"]["stop_slippage_ticks"])
        model = stressed_cost_model(ctx.config["backtest"]["cost_model"], multiplier)
        bars = self.bars(ctx, "train")
        started = time.monotonic()
        outcome = self.backtest_spec(
            ctx, optimize["best_spec"], bars.frame, cost_model=model, stop_slippage_ticks=stop_ticks
        )
        base_metrics = optimize["best"].get("metrics") or {}
        result = cost_stress.evaluate(
            cost_stress.CostStressInputs(
                base=summary_from_metrics(base_metrics),
                stressed=summary_from_metrics(outcome.metrics),
                cost_multiplier=multiplier,
            ),
            thresholds,
        )
        inputs = {
            **self.best_source(optimize),
            "data": bars.provenance(),
            "base_source": "optimize best trial (same spec, same train bars, frozen costs)",
            "stressed_cost_model": dataclasses.asdict(model),
            "stop_slippage_ticks": stop_ticks,
            "note": f"stressed run: commission, tax and slippage x{multiplier:g}, plus {stop_ticks:g} tick(s) of "
            "slippage on every stop-loss fill; base stop fills have none, so the stress is stricter than a pure "
            f"x{multiplier:g}. Take-profit fills never slip.",
            "base_metrics": headline_metrics(base_metrics),
            "stressed_metrics": {**headline_metrics(outcome.metrics), "costs": outcome.metrics.get("costs")},
            "seconds": round(time.monotonic() - started, 1),
        }
        files = {"stressed_trades.csv": trades_csv(outcome.trades), "stressed_daily_pnl.csv": daily_csv(outcome.daily)}
        return to_stage_result(result, inputs, files)


class RegimeSplitStage(GateStage):
    """Gate 6: the best spec's train trades bucketed by bull/bear x high/low vol."""

    name = "regime_split"

    def evaluate(self, ctx, optimize):
        from poseidon.pipeline.backtesting import trades_csv

        thresholds = ctx.config["thresholds"]
        bars = self.bars(ctx, "train")
        data = read_upstream_file(ctx.upstream["optimize"], "best_trades.csv")
        source = "optimize best_trades.csv (sha256-verified)"
        if data is None:
            data = trades_csv(self.backtest_spec(ctx, optimize["best_spec"], bars.frame).trades)
            source = "re-run of best_spec on the train part"
        trades = parse_trades(data)
        daily = daily_bars(bars.frame)
        states = regime_split.regime_states(daily, **regime_split.threshold_windows(thresholds))
        result = regime_split.evaluate(regime_split.RegimeInputs(trades, states.to_dict()), thresholds)
        labelled = states.dropna()
        inputs = {
            **self.best_source(optimize),
            "trades_source": source,
            "daily_bars": {
                "count": len(daily),
                "first": daily.index[0].isoformat() if len(daily) else None,
                "last": daily.index[-1].isoformat() if len(daily) else None,
                "labelled_days": len(labelled),
                "first_labelled": labelled.index[0].isoformat() if len(labelled) else None,
                "state_days": dict(sorted(Counter(labelled).items())),
                "source": ctx.config["gates"]["regime_split"]["daily_bars"],
            },
            "data": bars.provenance(),
            "labelling": "a trade takes the state of the last daily bar dated strictly before its entry date "
            "(Asia/Taipei calendar date): causal; a night-session entry before midnight uses the bar one trading "
            "day older than strictly necessary",
        }
        return to_stage_result(result, inputs)


class BenchmarkStage(GateStage):
    """Gate 7: alpha of the best spec's train daily PnL over TX buy-and-hold (OLS, HAC t)."""

    name = "benchmark"
    lease_seconds = 600

    def evaluate(self, ctx, optimize):
        from poseidon.pipeline.backtesting import trades_csv
        from poseidon.pipeline.plan import BENCHMARK_CONFIG_VERSION, config_version

        settings = ctx.config["gates"].get("benchmark")
        if config_version(ctx.config) < BENCHMARK_CONFIG_VERSION or not isinstance(settings, dict):
            return StageResult.unsupported(BENCHMARK_CONFIG_MISSING)
        bars = self.bars(ctx, "train")
        upstream = ctx.upstream["optimize"]
        daily_data = read_upstream_file(upstream, "best_daily_pnl.csv")
        trades_data = read_upstream_file(upstream, "best_trades.csv")
        source = "optimize best_daily_pnl.csv + best_trades.csv (sha256-verified)"
        if daily_data is None or trades_data is None:
            outcome = self.backtest_spec(ctx, optimize["best_spec"], bars.frame)
            daily_data, trades_data = daily_csv_bytes(outcome.daily), trades_csv(outcome.trades)
            source = "re-run of best_spec on the train part"
        series = benchmark_series(ctx, bars.frame, parse_daily(daily_data), parse_trades(trades_data))
        provenance = series["provenance"]
        result = benchmark.evaluate(
            benchmark.BenchmarkInputs(
                strategy_daily=series["strategy"].to_numpy(),
                benchmark_daily=series["benchmark"].to_numpy(),
                hac_lags=provenance["hac_lags"],
                annualisation_factor=provenance["annualisation_factor"],
                time_in_market=series["time_in_market"],
            ),
            ctx.config["thresholds"],
        )
        inputs = {
            **self.best_source(optimize),
            "daily_source": source,
            "train_metrics": headline_metrics(optimize["best"].get("metrics") or {}),
            "data": bars.provenance(),
            **provenance,
        }
        return to_stage_result(result, inputs)


def daily_csv_bytes(daily: pd.DataFrame) -> bytes:
    from poseidon.pipeline.backtesting import daily_csv

    return daily_csv(daily)


class WfeStage(GateStage):
    """Gate 1: rolling walk-forward with per-window re-optimisation on the train part.

    ``backtest/walk_forward.py`` is not reused: it re-optimises through
    ``ParameterSearchPipeline`` (portfolio-equity objective, no trial ledger,
    no stop/lease checks) and measures returns from portfolio equity. Here each
    window runs the optimize stage's own TPE search (``tpe_search`` on the
    spec's ``$param`` grid, capped at ``max_trials_per_window``, every trial in
    the family ledger) on its IS bars, then backtests the chosen point on the
    OOS bars (``warmup_days`` of feature warm-up before them, no entries
    there). Annualised returns are TWD PnL / capital scaled by trading days,
    and go through ``gates.wfe`` (``compute_wfe`` of walk_forward.py).
    """

    name = "wfe"
    requires = ("spec_validate", "optimize")

    def evaluate(self, ctx, optimize):
        from poseidon.pipeline import optimization as opt
        from poseidon.pipeline.backtesting import run_backtest
        from poseidon.pipeline.trials import MemoryTrialLedger
        from poseidon.strategies.dsl.v2 import validate_v2

        settings = ctx.config["gates"]["wfe"]
        warmup = timedelta(days=int(ctx.config["gates"]["warmup_days"]))
        capital = float(ctx.config["backtest"]["initial_capital_twd"])
        template = (ctx.upstream["spec_validate"].report or {}).get("template")
        if not isinstance(template, dict) or template.get("dsl_version") != 2:
            return StageResult.unsupported(V2_REQUIRED)
        train_from, train_until = DataWindow.from_config(ctx.config).bounds("train")
        windows = walk_forward_windows(
            train_until,
            windows=int(settings["windows"]),
            is_days=int(settings["is_days"]),
            oos_days=int(settings["oos_days"]),
        )
        layout = {
            "windows": int(settings["windows"]),
            "is_days": int(settings["is_days"]),
            "oos_days": int(settings["oos_days"]),
            "anchored_at": train_until.isoformat(),
            "max_trials_per_window": int(settings["max_trials_per_window"]),
            "warmup_days": int(ctx.config["gates"]["warmup_days"]),
        }
        if windows[0]["is_start"] < train_from:
            result = failed(
                wfe.GATE,
                "insufficient_history",
                f"train part starts {train_from.isoformat()}, the first IS window needs {windows[0]['is_start'].isoformat()}",
                "train part too short",
                {"layout": layout},
            )
            return to_stage_result(result, {"layout": layout})

        bars = self.bars(ctx, "train")
        frame = bars.frame
        factory = self.strategy_builder(ctx, template)
        grid = opt.ParamGrid(validate_v2(template).params)
        ledger = ctx.trials if ctx.trials is not None else MemoryTrialLedger()
        seed = int(ctx.config["seed"])
        per_window_budget = float(settings["budget_seconds"]) / len(windows)
        max_duplicates = int(ctx.config["optimize"]["max_consecutive_duplicates"])
        next_trial = 1
        metrics, rows, trial_rows = [], [], []
        started = time.monotonic()
        for window in windows:
            ctx.check_stop()
            window_started = time.monotonic()
            is_frame = _slice(frame, window["is_start"], window["oos_start"])
            oos_bars = len(_slice(frame, window["oos_start"], window["oos_end"]))
            if not len(is_frame) or not oos_bars:
                result = failed(
                    wfe.GATE,
                    "insufficient_history",
                    f"window {window['index']} has {len(is_frame)} IS and {oos_bars} OOS bars",
                    "train part too short",
                    {"layout": layout},
                )
                return to_stage_result(result, {"layout": layout, "windows": rows})
            is_sha = content_sha256(
                {
                    "data_sha256": bars.data_sha256,
                    "part": "train",
                    "wfe_window": window["index"],
                    "kind": "is",
                    "from_utc": window["is_start"].isoformat(),
                    "until_utc": window["oos_start"].isoformat(),
                }
            )
            runner = opt.TrialRunner(
                build=factory.build,
                bars=is_frame,
                backtest_config=ctx.config["backtest"],
                ledger=ledger,
                window_sha256=is_sha,
                should_stop=ctx.should_stop,
                check_stop=ctx.check_stop,
                first_trial_no=next_trial,
            )
            search = opt.tpe_search(
                runner,
                grid,
                factory.defaults,
                seed=seed + window["index"],
                max_trials=int(settings["max_trials_per_window"]),
                max_consecutive_duplicates=max_duplicates,
                budget_seconds=per_window_budget,
            )
            next_trial += len(runner.results)
            trial_rows += [{"window": window["index"], **result.row()} for result in runner.results]
            chosen = runner.best or (runner.results[0] if runner.results else None)
            if chosen is None or chosen.metrics is None:
                result = invalid(wfe.GATE, f"window {window['index']}: no IS trial produced metrics")
                return to_stage_result(result, {"layout": layout, "windows": rows})
            oos_frame = _slice(frame, window["oos_start"] - warmup, window["oos_end"])
            oos = run_backtest(
                factory.build(chosen.params),
                oos_frame,
                ctx.config["backtest"],
                should_stop=ctx.should_stop,
                trade_from=window["oos_start"],
            )
            is_ann = annualised_return(chosen.metrics, capital)
            oos_ann = annualised_return(oos.metrics, capital)
            metrics.append(
                wfe.WindowMetrics(
                    index=window["index"],
                    is_annualized_return=is_ann,
                    oos_annualized_return=oos_ann,
                    is_total_pnl_twd=chosen.metrics.get("total_pnl_twd"),
                    oos_total_pnl_twd=oos.metrics.get("total_pnl_twd"),
                    is_trade_count=int(chosen.metrics.get("n_trades") or 0),
                    oos_trade_count=int(oos.metrics.get("n_trades") or 0),
                )
            )
            rows.append(
                {
                    "index": window["index"],
                    "is_from_utc": window["is_start"].isoformat(),
                    "oos_from_utc": window["oos_start"].isoformat(),
                    "oos_until_utc": window["oos_end"].isoformat(),
                    "is_bars": len(is_frame),
                    "oos_bars_incl_warmup": len(oos_frame),
                    "chosen_trial_no": chosen.trial_no,
                    "chosen_params": chosen.params,
                    "chosen_by": "best IS sharpe_daily" if runner.best is chosen else "defaults (no IS Sharpe)",
                    "n_trials": len(runner.results),
                    "search_stopped_because": search.stopped_because,
                    "is_metrics": headline_metrics(chosen.metrics),
                    "oos_metrics": headline_metrics(oos.metrics),
                    "seconds": round(time.monotonic() - window_started, 1),
                }
            )
        result = wfe.evaluate(wfe.WfeInputs(metrics), ctx.config["thresholds"])
        inputs = {
            "layout": layout,
            "windows": rows,
            "n_trials_recorded": next_trial - 1,
            "ledger_note": "every per-window IS trial is in the family trial ledger (it counts toward DSR N for "
            "later runs of the family; this run's dsr ran before wfe)",
            "annualisation": "total_pnl_twd / initial_capital_twd * 252 / trading days with bars",
            "data": bars.provenance(),
            "seconds": round(time.monotonic() - started, 1),
            "trials_file": "wfe_trials.json",
        }
        files = {"wfe_trials.json": canonical_json(trial_rows).encode("utf-8")}
        return to_stage_result(result, inputs, files)


class HoldoutStage(GateStage):
    """Gate 2: train vs locked holdout of the best spec. The only reader of holdout bars."""

    name = "holdout"
    # ``benchmark`` only for runs frozen at CONFIG_VERSION >= 5 (BaseStage.upstream_names).
    requires = ("optimize", "param_stability", "dsr", "cost_stress", "regime_split", "benchmark", "wfe")

    def evaluate(self, ctx, optimize):
        from poseidon.pipeline.backtesting import daily_csv, trades_csv
        from poseidon.pipeline.family import MemoryFamily

        family = ctx.family if ctx.family is not None else MemoryFamily(ctx.trials, run_id=ctx.run_id)
        prior = family.holdout_evaluations()
        # Recorded before any holdout bar is read, so an attempt that later
        # errors or is aborted still counts as a look.
        family.record_holdout_evaluated(
            {
                "stage": self.name,
                "attempt": ctx.attempt,
                "stage_run_id": str(ctx.stage_run_id) if ctx.stage_run_id else None,
                "best_spec_sha256": optimize.get("best_spec_sha256"),
                "family_runs_already_evaluated": prior["other_runs"],
            }
        )
        warmup = timedelta(days=int(ctx.config["gates"]["warmup_days"]))
        held = self.bars(ctx, "holdout")
        train = self.bars(ctx, "train")
        holdout_from, _ = DataWindow.from_config(ctx.config).bounds("holdout")
        frame = pd.concat([train.frame[train.frame.index >= holdout_from - warmup], held.frame])
        outcome = self.backtest_spec(ctx, optimize["best_spec"], frame, trade_from=holdout_from)
        base_metrics = optimize["best"].get("metrics") or {}
        from poseidon.pipeline.backtesting import trading_days

        first_day = trading_days(pd.DatetimeIndex([pd.Timestamp(holdout_from)]))[0].date()
        stats, provenance = benchmark_stats(
            ctx, frame, outcome.daily, parse_trades(trades_csv(outcome.trades)), from_day=first_day
        )
        result = holdout.evaluate(
            holdout.HoldoutInputs(
                train=summary_from_metrics(base_metrics),
                holdout=summary_from_metrics(outcome.metrics),
                holdout_n_obs=outcome.metrics.get("n_obs"),
                benchmark=stats,
            ),
            ctx.config["thresholds"],
        )
        inputs = {
            **self.best_source(optimize),
            "train_source": "optimize best trial (train part)",
            "train_metrics": headline_metrics(base_metrics),
            "holdout_metrics": headline_metrics(outcome.metrics),
            "data": held.provenance(),
            "warmup": {
                "days": int(ctx.config["gates"]["warmup_days"]),
                "note": "train bars before the split warm features up; no entries before the split",
            },
            "benchmark": {**provenance, "first_day": first_day.isoformat()},
            "family_holdout_evaluations": {
                **prior,
                "policy": ctx.config["gates"]["holdout"]["family_reuse"],
                "note": "runs of this family that evaluated the holdout before this attempt (report only, not gated)",
            },
        }
        files = {"holdout_trades.csv": trades_csv(outcome.trades), "holdout_daily_pnl.csv": daily_csv(outcome.daily)}
        return to_stage_result(result, inputs, files)


GATE_STAGE_CLASSES = (
    ParamStabilityStage,
    DsrStage,
    CostStressStage,
    RegimeSplitStage,
    BenchmarkStage,
    WfeStage,
    HoldoutStage,
)


def build_gate_stages(*, data_source=None, strategy_factory=None) -> dict:
    stages = {cls.name: cls(data_source, strategy_factory) for cls in GATE_STAGE_CLASSES}
    if tuple(stages) != GATE_STAGES:
        raise RuntimeError("gate stages drifted from plan.GATE_STAGES")
    return stages
