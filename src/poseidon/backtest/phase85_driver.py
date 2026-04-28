"""Phase 85 — orchestrator. Runs Optuna search + WFE + per-window OOS re-backtest for ONE symbol.

Composition (D-10 serial, one symbol per process):

  1. Build factory (phase85_strategy_factory.build_phase85_factory) — 9-dim D-01 PARAM_SPACE.
  2. ``Phase85BayesianOptimizer(BayesianOptimizer)`` overrides ``optimize`` to inject
     ``bars_per_year=525_600`` into the IS Sharpe objective (A1 branch B; Pitfall 6).
  3. Optuna study with persistent Postgres RDBStorage (D-05) +
     per-trial budget callback (every 10 trials, D-12).
  4. Walk-forward on best_params via ``WalkForwardAnalyzer`` (D-07, D-09).
  5. Per-window OOS-only re-backtest to capture the trade ledger
     (``WindowResult`` only stores ``oos_trade_count``; D-19 max_consecutive_losses
     needs the actual pnl-bearing trade list).
  6. Budget log: psutil RSS + perf_counter wall (D-12).

Returns ``Phase85SymbolResult`` — RAW evidence only. NO PASS/FAIL logic. Verdict
math lives in ``phase85_metrics`` and Phase 86.

A1 resolution — branch B (subclass), confirmed by grep:

  $ grep -n "bars_per_year\\|compute_metrics\\|annualiz" \\
      poseidon/src/poseidon/backtest/optimizer.py
  (zero matches)

The kwarg branch (A) — passing ``bars_per_year`` through ``optimize()`` — is
dead code in the parent class. We commit to branch (B): the override below
re-derives metrics by calling ``compute_metrics(equity_series, trades,
bars_per_year=BARS_PER_YEAR_1M)`` on the runner's cached equity series before
returning the trial value to Optuna.

D-19 / B-1 fix: ``WalkForwardAnalyzer`` discards ``BacktestResult.trades`` and
only retains ``oos_metrics`` + ``oos_trade_count`` per window. We re-run the
OOS slice with the same factory + best_params and attach ``oos_trades:
list[dict]`` to each window dict so Phase 86 can compute D-19.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import optuna
import pandas as pd
import psutil

from poseidon.backtest.cost_model import COST_MODELS, CostModel
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.optimizer import BayesianOptimizer, OptimizationTrial
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.phase85_metrics import BARS_PER_YEAR_1M
from poseidon.backtest.phase85_strategy_factory import (
    build_phase85_factory,
    make_phase85_strategy_factory,
    resolve_factory_params,
)
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
)
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class BudgetSample:
    """One budget reading. Stored as ``study.user_attrs['budget_samples'][i]`` and
    serialized to the artifact. ``trial_index`` is None for non-trial samples
    (optuna_start, optuna_end, wfe_start, wfe_end)."""

    label: str
    wall_seconds: float
    rss_mb: float
    trial_index: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase85SymbolResult:
    """Raw evidence for one symbol — Phase 86 reads this to compute verdict.

    Driver writes ``best_params`` (9-dim D-01) AND ``factory_params_resolved``
    (full 17-key dict with factory defaults applied) so Phase 86 can replay
    the exact strategy config that produced ``wfe_per_window`` (Pitfall 5).
    """

    symbol: str
    study_name: str
    storage_url_redacted: str
    best_params: dict
    factory_params_resolved: dict
    best_value: float  # IS Sharpe with bars_per_year=525_600
    n_trials_completed: int
    n_trials_failed: int  # state==FAIL count (D-18 ValueError catch)
    wfe_per_window: list[dict]  # each dict has 'oos_trades' key (B-1 fix)
    wfe_aggregate: dict
    wfe_flags: list[str]
    budget_samples: list[BudgetSample]
    budget_summary: dict


# --------------------------------------------------------------------------- #
# A1 branch B — Phase85BayesianOptimizer subclass with bars_per_year=525_600
# --------------------------------------------------------------------------- #


class Phase85BayesianOptimizer(BayesianOptimizer):
    """A1 resolution branch B (Pitfall 6): override ``optimize`` so the trial
    objective uses ``bars_per_year=525_600`` in ``compute_metrics``.

    Why a subclass: the parent ``BayesianOptimizer.optimize`` calls
    ``BacktestRunner.run`` whose internal ``compute_metrics`` defaults to
    ``bars_per_year=252``. For 1m bars this under-annualizes Sharpe by
    ``sqrt(525_600 / 252) ≈ 45.66``. The override re-derives metrics on the
    runner's cached equity series + raw ``TradeRecord`` list with
    ``bars_per_year=BARS_PER_YEAR_1M``.

    Branch (A) — passing ``bars_per_year`` as a kwarg to ``optimize`` — is dead
    code. ``optimizer.py`` has zero references to ``bars_per_year``; verified
    by grep.
    """

    def optimize(
        self,
        strategy_factory: Callable[[dict], Any],
        ohlcv: pd.DataFrame,
        param_space: dict[str, tuple[Any, Any, str]],
        n_trials: int = 100,
        metric: str = "sharpe_ratio",  # GATE.yaml objective per D-04
        seed: int = 42,
        storage: str | None = None,
        study_name: str | None = None,
        callbacks: list[Callable] | None = None,
    ) -> list[OptimizationTrial]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=20)
        study_kwargs: dict[str, Any] = {"direction": "maximize", "sampler": sampler}
        if storage is not None:
            study_kwargs["storage"] = storage
            study_kwargs["study_name"] = study_name or f"study_{seed}"
            study_kwargs["load_if_exists"] = True
        study = optuna.create_study(**study_kwargs)

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for name, (low, high, ptype) in param_space.items():
                if ptype == "int":
                    params[name] = trial.suggest_int(name, int(low), int(high))
                elif ptype == "float":
                    params[name] = trial.suggest_float(name, float(low), float(high))
                else:
                    raise ValueError(f"Unsupported param type: {ptype!r}. Use 'int' or 'float'.")

            # Build strategy once after the params dict is fully populated.
            strategy = strategy_factory(params)
            runner = BacktestRunner(
                strategy=strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
                fill_model=self.fill_model,
            )
            # Run the full backtest — we drop the result object because the
            # default-annualized metrics are not what we want; the cached
            # equity_series + trade_records on the runner are re-fed below
            # into compute_metrics(bars_per_year=525_600).
            runner.run(ohlcv, db_session=self.db_session)

            # --- A1 RESOLUTION (branch B): re-compute metrics with 1m annualization ---
            # The parent's ``result.metrics`` was computed with bars_per_year=252.
            # Re-derive on the runner's cached equity_series + raw TradeRecord list
            # (cached by 85-03 patch to runner.py:705 — ``self._equity_series_cached``,
            # ``self._trade_records_cached``).
            equity_series = getattr(runner, "_equity_series_cached", None)
            trade_records = getattr(runner, "_trade_records_cached", None)
            if equity_series is None or trade_records is None:
                raise RuntimeError(
                    "BacktestRunner is missing _equity_series_cached / "
                    "_trade_records_cached attributes — Phase 85 patch (85-03) "
                    "did not apply. Re-deploy poseidon."
                )
            metrics_1m = compute_metrics(equity_series, trade_records, bars_per_year=BARS_PER_YEAR_1M)

            trial.set_user_attr("metrics", metrics_1m)
            trial.set_user_attr("bars_per_year", BARS_PER_YEAR_1M)
            return float(metrics_1m.get(metric, 0.0))

        study.optimize(
            objective,
            n_trials=n_trials,
            catch=(ValueError,),  # D-18: per-trial constraint failures don't kill study
            callbacks=callbacks or [],
        )

        trials: list[OptimizationTrial] = []
        for t in study.trials:
            trials.append(
                OptimizationTrial(
                    params=dict(t.params),
                    metrics=t.user_attrs.get("metrics", {}),
                    metric_value=t.value if t.value is not None else 0.0,
                )
            )
        trials.sort(key=lambda x: x.metric_value, reverse=True)
        return trials


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _redact_password(url: str) -> str:
    """Pattern S-1: mask password in a Postgres URL before persisting/logging.

    Returns a URL with the password replaced by ``***``. Username, hostname,
    port, dbname, and query params are preserved so downstream introspection
    of which database was hit still works.
    """
    from urllib.parse import urlparse, urlunparse

    p = urlparse(url)
    if p.password:
        netloc = f"{p.username}:***@{p.hostname}:{p.port}" if p.port else f"{p.username}:***@{p.hostname}"
        p = p._replace(netloc=netloc)
    return urlunparse(p)


def _sample_budget(
    label: str,
    run_start: float,
    trial_index: int | None = None,
) -> BudgetSample:
    """Capture wall-time delta + RSS in MB at the given label."""
    rss_bytes = psutil.Process().memory_info().rss
    return BudgetSample(
        label=label,
        wall_seconds=time.perf_counter() - run_start,
        rss_mb=rss_bytes / (1024 * 1024),
        trial_index=trial_index,
    )


def _make_budget_callback(
    run_start: float,
    budget_samples: list[BudgetSample],
    sample_every_n_trials: int = 10,
):
    """Optuna trial callback (D-12): sample RSS every N trials.

    Optuna fires callbacks AFTER each trial completes. We sample when
    ``trial.number % sample_every_n_trials == 0`` so we capture trial 0, 10,
    20, ... (in addition to the explicit optuna_start / optuna_end pair the
    driver records). This gives Phase 86 an RSS growth curve usable for
    OOM postmortem (D-12).
    """

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number % sample_every_n_trials == 0:
            budget_samples.append(_sample_budget(f"trial_{trial.number}", run_start, trial_index=trial.number))

    return _callback


def _slice_window_oos_trades(
    ohlcv: pd.DataFrame,
    test_start: int,
    test_end: int,
    best_params: dict,
    *,
    symbol: str,
    feature_engine: FeatureEngine,
    risk_engine: RiskEngine,
    cost_model: CostModel,
    fill_model: FillModel | None,
    initial_capital: float,
) -> list[dict]:
    """Re-run OOS-only backtest on one window slice and return the trade ledger.

    ``WalkForwardAnalyzer.analyze`` discards each window's
    ``BacktestResult.trades``; only ``oos_metrics`` + ``oos_trade_count``
    survive on ``WindowResult``. To compute D-19 ``max_consecutive_losses``
    we MUST re-run the OOS slice with the same factory + best_params and
    pull ``result.trades`` ourselves.
    """
    # Preserve tz-aware datetime index (Rule 1 fix): companion features like
    # open_interest call ``_align_oi_to_index(ohlcv.index)`` which requires a
    # datetime index to reindex against. Calling .reset_index(drop=True) here
    # converts the index to int64 and breaks the OI/funding companion alignment.
    oos_ohlcv = ohlcv.iloc[test_start:test_end]
    factory = make_phase85_strategy_factory(symbol)
    strategy = factory(best_params)
    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
        fill_model=fill_model,
    )
    result = runner.run(oos_ohlcv)
    # result.trades is list[dict] (runner.py:714-727) with keys:
    #   symbol, action, entry_time(ISO|None), exit_time(ISO|None),
    #   entry_price, exit_price, quantity, fees, pnl(float|None)
    return [dict(t) for t in (result.trades or [])]


# --------------------------------------------------------------------------- #
# Driver entry point
# --------------------------------------------------------------------------- #


def run_phase85_for_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    *,
    feature_engine: FeatureEngine,
    risk_engine: RiskEngine,
    n_trials: int = 100,
    seed: int = 42,
    storage_url: str,
    study_name: str | None = None,
    wf_config: WalkForwardConfig | None = None,
    budget_sample_every_n_trials: int = 10,
    initial_capital: float = 100_000.0,  # GATE.yaml frozen value (Phase 84)
) -> Phase85SymbolResult:
    """Run Optuna search + WFE + per-window OOS re-backtest for ONE symbol.

    D-10: serial per-symbol execution. The 85-05 stormtrooper E2E run calls
    this twice (BTC then ETH); cross-symbol parallelism is forbidden because
    distinct symbols share study storage paths only via the symbol-stamped
    ``study_name``.
    """
    run_start = time.perf_counter()
    budget_samples: list[BudgetSample] = []

    study_name = study_name or f"phase85_{symbol.lower()}_seed{seed}"
    if wf_config is None:
        # D-07 + D-09: 90/0/90 quarterly with 30d step → 4 OOS windows on 270d.
        # Calendar days × 1440 minutes (Pitfall 4 — DO NOT pass literal 90).
        wf_config = WalkForwardConfig(
            train_days=90 * 1440,
            validate_days=0,
            test_days=90 * 1440,
            step_days=30 * 1440,
            min_trades_per_oos=25,
            max_insufficient_ratio=0.30,
            min_wfe=0.50,
        )

    cost_model = COST_MODELS["crypto_perp"]
    fill_model: FillModel = FillModel.PESSIMISTIC  # GATE.yaml frozen fill model

    budget_samples.append(_sample_budget("optuna_start", run_start))

    factory_callable, param_space = build_phase85_factory(symbol)

    # A1 BRANCH B: Phase85BayesianOptimizer subclass injects bars_per_year=525_600.
    optimizer = Phase85BayesianOptimizer(
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
        fill_model=fill_model,
    )

    # D-12: per-trial budget callback (every 10 trials).
    budget_callback = _make_budget_callback(
        run_start,
        budget_samples,
        sample_every_n_trials=budget_sample_every_n_trials,
    )

    # D-18: catch=(ValueError,) is applied inside Phase85BayesianOptimizer.optimize.
    trials = optimizer.optimize(
        strategy_factory=factory_callable,
        ohlcv=ohlcv,
        param_space=param_space,
        n_trials=n_trials,
        metric="sharpe_ratio",
        seed=seed,
        storage=storage_url,
        study_name=study_name,
        callbacks=[budget_callback],
    )

    # Reload study from storage to get true completed/failed counts.
    study_view = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = sum(1 for t in study_view.trials if t.state == optuna.trial.TrialState.COMPLETE)
    failed = sum(1 for t in study_view.trials if t.state == optuna.trial.TrialState.FAIL)

    budget_samples.append(_sample_budget("optuna_end", run_start))

    if not trials:
        raise RuntimeError(f"phase85_driver: no trials returned for {symbol} (study_name={study_name})")
    best_trial = trials[0]
    best_params = dict(best_trial.params)
    best_value = float(best_trial.metric_value)
    factory_params_resolved = resolve_factory_params(best_params)

    # ---- Walk-forward on best params (D-07, D-09) ----
    budget_samples.append(_sample_budget("wfe_start", run_start))
    analyzer = WalkForwardAnalyzer(
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
        strategy_factory=None,  # let analyzer fallback to VotingStrategyFactory not used here;
        fill_model=fill_model,
    )
    # We must give WalkForwardAnalyzer a strategy_factory module that knows
    # how to reconstruct the strategy. The analyzer's polymorphic to_config_dict /
    # from_config interface — see walk_forward.py:227-245 — accepts any module
    # with these classmethods. LiquiditySweepStrategyFactory matches.
    from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory

    analyzer._strategy_factory = LiquiditySweepStrategyFactory
    best_strategy_for_wfe = make_phase85_strategy_factory(symbol)(best_params)
    wfe_result = analyzer.analyze(
        strategy=best_strategy_for_wfe,
        ohlcv=ohlcv,
        config=wf_config,
    )
    budget_samples.append(_sample_budget("wfe_end", run_start))

    # ---- Per-window OOS re-backtest to capture trade ledger (B-1 fix) ----
    # WindowResult only carries oos_trade_count; D-19 needs the actual trades.
    windows_indices = analyzer.generate_windows(len(ohlcv), wf_config)
    per_window_dicts: list[dict] = []
    for w, ((train_start, train_end), (test_start, test_end)) in zip(
        wfe_result.per_window, windows_indices, strict=False
    ):
        d = asdict(w) if hasattr(w, "__dataclass_fields__") else dict(w.__dict__)
        oos_trades = _slice_window_oos_trades(
            ohlcv,
            test_start,
            test_end,
            best_params,
            symbol=symbol,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            fill_model=fill_model,
            initial_capital=initial_capital,
        )
        d["oos_trades"] = oos_trades
        d["test_start_idx"] = int(test_start)
        d["test_end_idx"] = int(test_end)
        d["train_start_idx"] = int(train_start)
        d["train_end_idx"] = int(train_end)
        per_window_dicts.append(d)

    # Sanity check: every window has 'oos_trades' (possibly empty list).
    for d in per_window_dicts:
        assert "oos_trades" in d, "phase85_driver: per-window dict missing oos_trades — invariant broken"

    total_seconds = time.perf_counter() - run_start
    peak_rss = max((s.rss_mb for s in budget_samples), default=0.0)
    optuna_secs = next((s.wall_seconds for s in budget_samples if s.label == "optuna_end"), 0.0)
    wfe_start_secs = next((s.wall_seconds for s in budget_samples if s.label == "wfe_start"), 0.0)
    wfe_end_secs = next((s.wall_seconds for s in budget_samples if s.label == "wfe_end"), 0.0)

    return Phase85SymbolResult(
        symbol=symbol,
        study_name=study_name,
        storage_url_redacted=_redact_password(storage_url),
        best_params=best_params,
        factory_params_resolved=factory_params_resolved,
        best_value=best_value,
        n_trials_completed=completed,
        n_trials_failed=failed,
        wfe_per_window=per_window_dicts,
        wfe_aggregate=dict(getattr(wfe_result, "aggregate_oos_metrics", {})),
        wfe_flags=list(getattr(wfe_result, "flags", [])),
        budget_samples=budget_samples,
        budget_summary={
            "total_seconds": total_seconds,
            "optuna_seconds": optuna_secs,
            "wfe_seconds": wfe_end_secs - wfe_start_secs,
            "peak_rss_mb": peak_rss,
        },
    )


__all__ = [
    "BudgetSample",
    "Phase85BayesianOptimizer",
    "Phase85SymbolResult",
    "_make_budget_callback",
    "_redact_password",
    "_sample_budget",
    "_slice_window_oos_trades",
    "run_phase85_for_symbol",
]
