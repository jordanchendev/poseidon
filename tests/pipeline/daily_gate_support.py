"""Synthetic engine runs for the daily-class gate controls (design 2026-10-24 §7).

* :func:`synthetic_market` -- one TMF-like contract (no roll) over ``n_days`` business days:
  a random walk with slowly cycling volatility and a small upward drift, plus a public daily
  signal ``s_t`` (known at the close of t; AR(1) with coefficient ``persistence``, 0 = i.i.d.).
  With ``edge > 0`` the next day's open-to-close move carries ``edge * sign(s_t)`` on top of the
  noise (the planted edge); with ``gap_edge > 0`` the overnight gap close(t) -> open(t+1) carries
  ``gap_edge * sign(s_t)`` (an edge only a fill at the signal close can take).
* :class:`SignalLeverage` -- target leverage ``lev`` in the direction of a per-day sign table
  (the signal for the planted edge, a seeded coin for random exposure, always +1 for static
  leverage), flat when ``|s_t| <= threshold``; ``signal_lag`` reads the table that many trading
  days late (the execution-timing gate's ``signal_lag_extra``).
* :func:`judge` -- runs every daily gate the way step 4 will: all grid points on train (the
  trial ledger), the best one through liquidation / param_stability / dsr / cost_stress /
  regime_split / benchmark / wfe / execution_timing, then the locked holdout.
  WFE windows slice the full-train runs of each grid point (IS choice and OOS of the chosen
  params) instead of re-running per window; the engine's state at a window start is the only
  difference, which the controls do not depend on.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import ContractSpec, Decision, EngineConfig, FuturesInput, run
from poseidon.pipeline.gates import (
    DAILY_GATES,
    benchmark,
    cost_stress,
    dsr,
    execution_timing,
    holdout,
    liquidation,
    param_stability,
    regime_split,
    wfe,
)
from poseidon.pipeline.gates.daily import DailyResult, RerunRequest, choose_walk_forward, from_engine, moments, sharpe
from poseidon.pipeline.optimization import ParamGrid
from poseidon.pipeline.plan import class_settings
from poseidon.pipeline.v3.components import round_units
from poseidon.strategies.dsl.v2.schema import ParamSpec

TMF = ContractSpec(symbol="TMF", point_value=10.0, tick_size=1.0, commission_per_unit=20.0, tax_rate=2e-5)
FAR = "2100-01-01"
CLASS = "daily_equity"
BASIS = "return"


@dataclasses.dataclass(frozen=True)
class Market:
    bars: pd.DataFrame  # date, contract, open, high, low, close
    signal: dict[date, float]  # s_t, public at the close of t
    daily: pd.DataFrame  # the 1x underlying's daily OHLC indexed by date (regime states)
    underlying: dict[date, float]  # its simple daily returns

    @property
    def dates(self) -> list[date]:
        return list(self.daily.index)


def synthetic_market(
    n_days: int,
    *,
    seed: int = 3,
    edge: float = 0.0,
    gap_edge: float = 0.0,
    persistence: float = 0.0,
    drift: float = 0.0003,
    crash_day: int | None = None,
    crash_low: float = 0.243,
    crash_close: float = 0.08,
) -> Market:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n_days)
    signal = rng.normal(0.0, 1.0, n_days)
    if persistence:  # same draws, AR(1) with unit variance: the sign of s_t lasts ~1 / P(flip) days
        innovations, scale = signal.copy(), float(np.sqrt(1.0 - persistence**2))
        for i in range(1, n_days):
            signal[i] = persistence * signal[i - 1] + scale * innovations[i]
    vol = 0.010 * (1.0 + 0.5 * np.sin(np.arange(n_days) * 2 * np.pi / 500.0))
    gap = rng.normal(0.0, 0.004, n_days)
    move = rng.normal(0.0, 1.0, n_days) * vol
    if crash_day is not None:
        signal[crash_day - 1] = 3.0  # every threshold goes long into the crash
    rows, close = [], 18000.0
    for i, t in enumerate(dates):
        side = np.sign(signal[i - 1]) if i else 0.0
        o = close * (1.0 + gap[i] + gap_edge * side)
        planted = edge * side
        c = o * (1.0 + drift + planted + move[i])
        wick = abs(rng.normal(0.0, 0.3)) * vol[i]
        h, lo = max(o, c) * (1.0 + wick), min(o, c) * (1.0 - wick)
        if i == crash_day:
            lo, c = close * (1.0 - crash_low), close * (1.0 - crash_close)
            h = max(o, c) * 1.001
        rows.append((t, "A", round(o), round(h), round(lo), round(c)))
        close = c
    bars = pd.DataFrame(rows, columns=["date", "contract", "open", "high", "low", "close"])
    daily = bars.set_index(bars["date"].dt.date)[["open", "high", "low", "close"]].astype(float)
    underlying = {day: float(value) for day, value in (daily["close"] / daily["close"].shift(1) - 1.0).iloc[1:].items()}
    return Market(bars, {t.date(): float(s) for t, s in zip(dates, signal, strict=True)}, daily, underlying)


def coin(seed: int, dates) -> dict[date, float]:
    """A seeded fair coin per day (+-1): random exposure."""

    out = {}
    for day in dates:
        digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).digest()
        out[day] = 1.0 if digest[0] & 1 else -1.0
    return out


class SignalLeverage:
    def __init__(self, signs: Mapping[date, float], *, lev: float, threshold: float = 0.0, signal_lag: int = 0):
        self.signs, self.lev, self.threshold, self.signal_lag = signs, float(lev), float(threshold), signal_lag

    def on_bar(self, view, book) -> Decision:
        series = view["TMF"]
        row = len(series.dates) - 1 - self.signal_lag
        s = self.signs.get(pd.Timestamp(series.dates[row]).date(), 0.0) if row >= 0 else 0.0
        if abs(s) <= self.threshold:
            return Decision({"TMF": 0})
        direction = 1 if s > 0 else -1
        raw = (
            book.equity * direction * self.lev / (float(series.close[-1]) * TMF.point_value) if book.equity > 0 else 0.0
        )
        units = round_units(raw, "floor")
        return Decision({"TMF": units}, {"TMF": self.lev})


def grid(lev: tuple[float, float, float] = (1.0, 3.0, 1.0)) -> ParamGrid:
    low, high, step = lev
    return ParamGrid(
        {
            "lev": ParamSpec(type="float", low=low, high=high, step=step, default=low),
            "threshold": ParamSpec(type="float", low=0.0, high=0.5, step=0.25, default=0.0),
        }
    )


class Runner:
    """The re-run hook (gates.daily.Rerun) over one window: costs scaled per RerunRequest."""

    def __init__(
        self,
        market: Market,
        signs: Mapping[date, float],
        *,
        start: date,
        end: date | None,
        execution: str = "next_open",
    ):
        self.market, self.signs, self.start, self.end, self.execution = market, signs, start, end, execution

    def __call__(self, request: RerunRequest) -> DailyResult:
        m = float(request.cost_multiplier)
        spec = dataclasses.replace(
            TMF,
            commission_per_unit=TMF.commission_per_unit * m,
            tax_rate=TMF.tax_rate * m,
            slippage_ticks=TMF.slippage_ticks * m,
        )
        config = EngineConfig(execution=request.execution or self.execution, start=self.start, end=self.end)
        strategy = SignalLeverage(self.signs, **request.params, signal_lag=request.signal_lag_extra)
        futures = {"TMF": FuturesInput(spec=spec, bars=self.market.bars, expiries={"A": FAR})}
        return from_engine(run(strategy, futures, config))


def judge(
    market: Market,
    signs: Mapping[date, float],
    params: ParamGrid,
    *,
    holdout_fraction: float = 0.2,
    execution: str = "next_open",
) -> dict:
    settings = class_settings(CLASS)
    t, g = settings["thresholds"], settings["gates"]
    dates = market.dates
    warmup = 5
    first = dates[warmup]
    holdout_start = dates[int(len(dates) * (1.0 - holdout_fraction))]
    train_end = dates[dates.index(holdout_start) - 1]
    train = Runner(market, signs, start=first, end=train_end, execution=execution)

    trials = {params.key(point): (point, train(RerunRequest(point))) for point in params.points()}
    trial_sharpes = [moments(result.metric(BASIS))["sharpe_daily"] for _point, result in trials.values()]
    best_key = max(trials, key=lambda key: sharpe(trials[key][1], BASIS) or -np.inf)
    best_params, best = trials[best_key]

    results = {liquidation.GATE: liquidation.evaluate(best, t)}
    rows = params.neighbours(best_params, 1)
    neighbours, _failed = param_stability.neighbours_daily(rows, lambda p: trials[params.key(p)][1])
    results[param_stability.GATE] = param_stability.evaluate(
        param_stability.ParamStabilityInputs(best_params, best.total_pnl_twd, neighbours), t
    )
    results[dsr.GATE] = dsr.evaluate(dsr.inputs_daily(best, BASIS, trial_sharpes), t)
    results[cost_stress.GATE] = cost_stress.run_daily(train, best_params, best, BASIS, t)
    states = regime_split.regime_states(
        market.daily.loc[market.daily.index <= train_end], **regime_split.threshold_windows(t)
    ).to_dict()
    results[regime_split.GATE] = regime_split.evaluate_daily(regime_split.RegimeDailyInputs(best, BASIS, states), t)
    hac = int(g["benchmark"]["hac_lags"])
    always_long = dict.fromkeys(dates, 1.0)

    def matched(start: date, end: date | None):  # the risk route's beta-matched run: long at a constant leverage
        return lambda leverage: Runner(market, always_long, start=start, end=end, execution=execution)(
            RerunRequest({"lev": leverage, "threshold": 0.0})
        )

    results[benchmark.GATE] = benchmark.evaluate_daily(
        benchmark.BenchmarkDailyInputs(best, market.underlying, BASIS, hac, matched=matched(first, train_end)), t
    )
    chosen = choose_walk_forward(first, train_end + timedelta(days=1), g["wfe"]["layouts"])
    layout = None
    if chosen is None:
        results[wfe.GATE] = None
    else:
        layout = chosen["layout"]
        windows = []
        for window in chosen["windows"]:
            in_sample = {
                key: result.window(window["is_start"], window["oos_start"]) for key, (_p, result) in trials.items()
            }
            pick = max(in_sample, key=lambda key: sharpe(in_sample[key], BASIS) or -np.inf)
            oos = trials[pick][1].window(window["oos_start"], window["oos_end"])
            windows.append(wfe.window_metrics_daily(window["index"], in_sample[pick], oos, BASIS))
        results[wfe.GATE] = wfe.evaluate(wfe.WfeInputs(windows), t)
    results[execution_timing.GATE] = execution_timing.run_daily(
        train, best_params, best, base_execution=execution, basis=BASIS, thresholds=t
    )
    held = Runner(market, signs, start=holdout_start, end=None, execution=execution)(RerunRequest(best_params))
    stats = benchmark.regress_daily(held, market.underlying, basis=BASIS, lags=hac)
    results[holdout.GATE] = holdout.evaluate_daily(
        holdout.HoldoutDailyInputs(best, held, BASIS, stats, matched=matched(holdout_start, None)), t
    )
    assert set(results) == set(DAILY_GATES)
    return {
        "results": results,
        "best_params": best_params,
        "train": best,
        "holdout": held,
        "wfe_layout": layout,
        "n_trials": len(trial_sharpes),
    }


def failed_gates(judged: dict) -> list[str]:
    return [name for name, result in judged["results"].items() if result is None or not result.passed]
