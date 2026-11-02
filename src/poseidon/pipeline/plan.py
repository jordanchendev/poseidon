"""The v1 stage plan and the frozen per-run configuration."""

from __future__ import annotations

import copy
from datetime import date

# Step 3b gate order: cheap ledger-only gates first (param_stability, dsr), then
# the gates that re-run the strategy on the train part (cost_stress,
# regime_split, benchmark, wfe), and the locked holdout exactly once, last, only
# for a candidate that passed everything on train. ``benchmark`` (alpha over TX
# buy-and-hold) exists from CONFIG_VERSION 5; see STAGE_INTRODUCED_AT.
V1_STAGE_PLAN: tuple[str, ...] = (
    "spec_generate",
    "spec_validate",
    "backtest",
    "optimize",
    "param_stability",
    "dsr",
    "cost_stress",
    "regime_split",
    "benchmark",
    "wfe",
    "holdout",
    "shadow_paper",
    "awaiting_release",
)
GATE_STAGES: tuple[str, ...] = ("param_stability", "dsr", "cost_stress", "regime_split", "benchmark", "wfe", "holdout")
# Stages added after runs were already frozen, with the CONFIG_VERSION that
# introduced them. A run frozen earlier has no such stage in its stage_plan, so
# for that run every stage that ``requires`` it behaves as it did at the run's
# version: the stage is left out of its upstream map, its input hash and its
# plan validation (``requires_for``). A live run therefore keeps its input
# hashes across the upgrade (shadow_paper: config_sha256 + upstream shas +
# PAPER_VERSION).
STAGE_INTRODUCED_AT: dict[str, int] = {"benchmark": 5, "execution_timing": 6, "spec_faithfulness": 6}
# The gates every run with a holdout has had (completion_reason).
CORE_GATE_STAGES: tuple[str, ...] = tuple(name for name in GATE_STAGES if name not in STAGE_INTRODUCED_AT)
# The one stage order every plan is a subsequence of (design 2026-10-24 §8). ``execution_timing`` is the
# blocking execution-timing gate of spec v3 daily runs (CONFIG_VERSION 6, owner decision 2026-10-24 #2);
# ``spec_faithfulness`` is the report-only JEV review of the spec against the idea (§4, step 6). Nothing
# requires it, so adding it changes no other stage's requires or input hash. V1_STAGE_PLAN (the DSL v2
# plan, CONFIG_VERSION <= 5) is unchanged and never contains either.
CANONICAL_ORDER: tuple[str, ...] = (
    *V1_STAGE_PLAN[: V1_STAGE_PLAN.index("spec_validate") + 1],
    "spec_faithfulness",
    *V1_STAGE_PLAN[V1_STAGE_PLAN.index("spec_validate") + 1 : V1_STAGE_PLAN.index("benchmark") + 1],
    "execution_timing",
    *V1_STAGE_PLAN[V1_STAGE_PLAN.index("benchmark") + 1 :],
)
# Spec v3 daily runs (CONFIG_VERSION 6, engine target_v1): the spec is hand-written or Sisyphus-confirmed
# (``input_spec_json``), so there is no ``spec_generate`` until spec_generate v3 (M2). Liquidation is judged
# inside ``backtest`` and on optimize's best point (design §7).
V3_GATE_STAGES: tuple[str, ...] = (
    "param_stability",
    "dsr",
    "cost_stress",
    "regime_split",
    "benchmark",
    "execution_timing",
    "wfe",
    "holdout",
)
# ``spec_faithfulness`` joined the plan of runs created after step 6b (no CONFIG_VERSION bump: a v3 run
# frozen earlier keeps its stored plan without it, and no stage requires it).
V3_DAILY_STAGE_PLAN: tuple[str, ...] = (
    "spec_validate",
    "spec_faithfulness",
    "backtest",
    "optimize",
    *V3_GATE_STAGES,
    "shadow_paper",
    "awaiting_release",
)
# An idea-only spec v3 daily run (CONFIG_VERSION 7, routed): spec_generate v3 writes the spec, then the
# daily plan follows; spec_validate reads the generated spec (``spec_source``).
V3_ROUTED_DAILY_STAGE_PLAN: tuple[str, ...] = ("spec_generate", *V3_DAILY_STAGE_PLAN)
# Step 7: a run that passes the holdout continues into shadow_paper and parks in
# awaiting_release (CONFIG_VERSION 4). Runs frozen earlier keep their own
# stage_plan, which ends at holdout. CONFIG_VERSION 5 adds ``benchmark``; runs
# frozen at 4 keep their plan without it.
DEFAULT_STAGE_PLAN: tuple[str, ...] = V1_STAGE_PLAN
ALL_GATES_PASSED = "passed all validation gates"
ALL_STAGES_PASSED = "all stages passed"


def completion_reason(stage_plan) -> str:
    """Reason recorded on a run whose whole plan passed."""

    return ALL_GATES_PASSED if set(CORE_GATE_STAGES) <= set(stage_plan) else ALL_STAGES_PASSED


def config_version(config) -> int:
    return int((config or {}).get("version") or 0)


def requires_for(requires, version: int) -> tuple[str, ...]:
    """``requires`` as it was for a run frozen at config ``version`` (see STAGE_INTRODUCED_AT)."""

    return tuple(name for name in requires if STAGE_INTRODUCED_AT.get(name, 0) <= version)


# --- Spec v3 daily runs (CONFIG_VERSION 6): settings outside the class table ---
#
# The daily data window (§12 #3): 2001-01-02 (after TX's first years) to the run's creation day, the
# last HOLDOUT_FRACTION locked. Train stages read refdata only up to the day before the holdout
# starts; only ``holdout`` reads the rest, once, last.
DAILY_DATA_START = "2001-01-02"
DAILY_BACKTEST: dict = {
    # Truncation audit recorded by ``backtest`` (engine.truncation_audit; the step-3 proof used 20 cuts).
    "truncation_audit_cuts": 5,
}

# Initial gate thresholds (PLAN "Assumptions [minor]"). They are frozen into each
# run's config_json; v1 never accepts client overrides -- a new threshold means
# a new run in the same family.
DEFAULT_THRESHOLDS: dict = {
    "wfe_min": 0.5,
    "holdout_sharpe_decay_max": 0.5,
    "holdout_pnl_positive": True,
    "holdout_min_trades": 20,
    # Owner decision 2026-10-23: the holdout also passes when its Sharpe is within
    # z standard errors of train (gates/holdout.py criterion (b)). Runs frozen
    # without this key keep decay-only judging, so no CONFIG_VERSION bump.
    "holdout_se_z": 1.96,
    # Owner decision 2026-10-23 (pre-registered, CONFIG_VERSION 5): the benchmark
    # gate passes iff train alpha over TX buy-and-hold > 0 and its HAC t >= this;
    # the holdout additionally needs holdout alpha > 0. A run frozen without a key
    # has that criterion off (gates/benchmark.py, gates/holdout.py).
    "benchmark_alpha_t_min": 1.0,
    "holdout_alpha_positive": True,
    "cost_stress_multiplier": 2.0,
    "cost_stress_pnl_positive": True,
    "dsr_min": 0.95,
    "dsr_min_obs": 30,
    "param_neighbourhood_profitable_min": 0.6,
    "param_min_neighbours": 2,
    # Owner decision 2026-10-28 (run #18's rule, applied to the DSL path): a neighbour cut off by a validator hard
    # bound (not the spec's declared range) no longer sinks param_stability on insufficient_neighbours when every
    # in-range neighbour is profitable (gates/param_stability.py). Runs frozen without the key (run #6) keep the old rule.
    "param_edge_at_bound_pass": True,
    "regime_profitable_segments_min": 3,
    "regime_min_segments": 3,
    "regime_min_trades_per_segment": 5,
    "regime_trend_sma_days": 60,
    "regime_atr_days": 20,
    "regime_vol_median_days": 250,
    "paper_min_trading_days": 20,
    "paper_min_trades": 10,
    "paper_replay_match_min": 0.9,
    "paper_replay_tolerance_ticks": 1,
}

# 3: frozen ``gates`` settings (Step 3b); 4: ``shadow_paper`` settings + full plan (Step 7);
# 5: ``benchmark`` stage + ``gates.benchmark`` + benchmark_alpha_t_min / holdout_alpha_positive
CONFIG_VERSION = 5
# 6: spec v3 runs (design 2026-10-24 §8): engine "target_v1", spec_validator "spec_v3",
# registry_version, component fingerprints, strategy_class, brief sha256, refdata content
# sha256s, the class table entry (poseidon.pipeline.v3.config.daily_run_config). A run is
# frozen at 6 only when its spec_json is a spec v3 document; default_config() keeps freezing
# CONFIG_VERSION 5 for DSL v2 runs.
SPEC_V3_CONFIG_VERSION = 6
# 7: idea-only runs (spec_generate v3, step 6). The strategy class is unknown until spec_generate has
# written the spec, so the run freezes, at creation, one complete config per candidate class under
# ``by_class`` (the CONFIG_VERSION 5 DSL config for intraday_fixed, a CONFIG_VERSION 6 daily config per
# daily class) plus the v3 ``spec_generate`` settings. spec_generate records the chosen class in its
# report (``route``) and every later stage runs on that class's frozen config (``effective_config``).
# config_sha256 is the hash of the whole routed config, computed once at creation.
ROUTED_CONFIG_VERSION = 7
SHADOW_PAPER_CONFIG_VERSION = 4
BENCHMARK_CONFIG_VERSION = 5
GATES_CONFIG_VERSION = 3
DEFAULT_SPEC_VALIDATOR = "dsl_v2"


class StagePlanError(ValueError):
    """The requested stage plan is not executable."""


def default_config(*, stage_plan: list[str] | None, as_of: date) -> dict:
    """Return the server-side configuration frozen into a new run."""

    return {
        "version": CONFIG_VERSION,
        "spec_generate": copy.deepcopy(DEFAULT_SPEC_GENERATE),
        "stage_plan": list(stage_plan) if stage_plan is not None else list(DEFAULT_STAGE_PLAN),
        "thresholds": copy.deepcopy(DEFAULT_THRESHOLDS),
        "data": data_config(as_of=as_of, holdout_fraction=HOLDOUT_FRACTION),
        "holdout": {"fraction": HOLDOUT_FRACTION, "locked": True},
        "seed": 0,
        "backtest": copy.deepcopy(DEFAULT_BACKTEST),
        "optimize": copy.deepcopy(DEFAULT_OPTIMIZE),
        "gates": copy.deepcopy(DEFAULT_GATES),
        "shadow_paper": copy.deepcopy(DEFAULT_SHADOW_PAPER),
        "spec_validator": DEFAULT_SPEC_VALIDATOR,
    }


# --- Step 2: data window, backtest and optimize settings (frozen per run) ---

DATA_START = "2020-03-02"  # TX 5m is continuous from 2020-03 (PLAN "Given")
HOLDOUT_FRACTION = 0.2

# Owner decisions 2026-10-22: pipeline cost model, 1 contract, gap-aware stops.
DEFAULT_BACKTEST: dict = {
    "market": "tw_futures",
    "symbol": "TX",
    "cost_model": "tw_futures_pipeline",
    "sizing": {"mode": "fixed_contracts", "contracts": 1},
    "gap_aware_stops": True,
    "initial_capital_twd": 1_000_000.0,
    "bar_label": "open",
    "sharpe": {"period": "trading_day", "annualisation_factor": 252},
    # Configs frozen from 2026-10-28 on: a night session belongs to the next day session in the bars, as TAIFEX
    # books it. Runs frozen without the key (run #6) roll it to the next weekday (backtesting.trading_days).
    "night_session": "next_day_session",
}

DEFAULT_OPTIMIZE: dict = {
    "sampler": "tpe",
    "objective": "sharpe_daily",
    # Measured 2026-10-23 in the runner image on stormtrooper: one backtest of
    # the ORB example on the ~5.3-year train window (294,520 5m bars) takes
    # 53 s alone and 55 s with both worker processes busy. 45 search trials
    # plus one full neighbour round at the 8-parameter maximum (16 trials) is
    # 61 x 55 s = 56 min, inside the 60-minute budget per optimize attempt.
    "max_trials": 45,
    "budget_seconds": 3600,  # safety stop only; see optimization.tpe_search
    "max_consecutive_duplicates": 100,
    "neighbour_steps": 1,
}


# --- Step 3b: gate-stage settings (frozen per run; thresholds stay in ``thresholds``) ---
#
# wfe budget (<= 60 min per attempt on the worker). Measured 2026-10-23: one
# train-window backtest of the ORB example (294,520 5m bars) takes ~55 s, i.e.
# ~1.9e-4 s per bar, so one 730-day IS trial (~116k bars) is ~22 s and one
# warm-up + 182-day OOS run (~32k bars) ~6 s. 4 windows x 16 IS trials =
# 64 x 22 s + 4 x 6 s = ~24 min nominal, ~48 min when both worker processes
# are busy and the run is 2x slower -- inside the budget with headroom for
# larger specs. ``budget_seconds`` is the per-attempt safety stop, split evenly
# over the windows (tpe_search stops a window early only if it would overrun).
DEFAULT_GATES: dict = {
    "cost_stress": {
        # The stressed run only: stop fills get this many ticks of slippage on
        # top of commission/tax/slippage x cost_stress_multiplier. Base stop
        # fills have none, so this is stricter than a pure x2.
        "stop_slippage_ticks": 1.0,
    },
    "regime_split": {
        "daily_bars": "train-part 5m bars resampled per TX trading day (night session -> next day session)",
    },
    "wfe": {
        "windows": 4,
        "is_days": 730,
        "oos_days": 182,
        "max_trials_per_window": 16,
        "budget_seconds": 3600,
        "objective": "sharpe_daily",
    },
    # Bars before the evaluated span that warm features up; entries are refused
    # during the warm-up, so it carries no trades and no PnL.
    "warmup_days": 14,
    # Alpha over TX buy-and-hold (CONFIG_VERSION 5): daily strategy PnL regressed on
    # daily buy-and-hold PnL (1 contract, point value of the cost model), alpha
    # t-stat Newey-West HAC with this many lags; the holdout gate reports the same
    # regression on the holdout days.
    "benchmark": {"hac_lags": 5},
    "holdout": {"family_reuse": "report_only"},
}


# --- Step 7: shadow paper settings (frozen per run; min days / trades / replay match stay in ``thresholds``) ---
#
# One evaluation per day at 09:30 Asia/Taipei. Thalassa's TX 5m beat runs only
# 09:00-13:55 Asia/Taipei, and on 2026-10-23 at 14:30 it held nothing after
# that morning's 05:00 although every fetch of the day had run: a day session
# arrives, at the earliest, with the next morning's fetches (together with the
# night session after it). A wake-up 15 min after 13:45 or 05:00 would see no
# new bars; 09:30 (after the 09:00 fetch) sees every completed trading day.
# What is evaluated is decided by the bars themselves (only trading days whose
# closing bar has arrived), so the wake time only sets latency. Weekend wake-ups
# are cheap no-ops that catch up a failed Saturday fetch.
DEFAULT_SHADOW_PAPER: dict = {
    "wake_local_time": "09:30",
    "wake_weekdays": [0, 1, 2, 3, 4, 5, 6],
    "source_retry_minutes": 60,  # Thalassa unreachable: try again this soon (never fails the stage)
    "stale_after_weekdays": 3,  # report (never fail) when no new bars for more weekdays than this
    "max_trading_days": 60,  # still < paper_min_trades by then -> fail insufficient_paper_trades
    "protection": {
        # Daily loss beyond multiple x the train backtest's 95th-percentile daily loss,
        # or a trade-level max drawdown beyond multiple x the train max drawdown -> fail at once.
        "daily_loss_quantile": 0.95,
        "daily_loss_multiple": 3.0,
        "max_drawdown_multiple": 1.5,
    },
}


# --- Strategy Pipeline v2 (design 2026-10-24 §7): thresholds and gate settings per strategy class ---
#
# A CONFIG_VERSION 6 run freezes the entry of its ``strategy_class`` (v3.spec.strategy_class) into
# ``thresholds`` and ``gates`` (``class_settings``). Runs frozen at <= 5 never read this table.
# The daily numbers are §7 / §12 #5, #8, #9 as approved by the owner on 2026-10-24, registered before
# the first daily run. As before, a key missing from a frozen run turns its criterion off.
# Daily metrics (gates/daily.py): ``daily_equity`` judges daily returns (pnl / previous equity),
# ``daily_fixed`` daily TWD PnL; a day-count threshold counts trading days of the engine's daily_pnl.
DAILY_THRESHOLDS: dict = {
    "wfe_min": 0.5,
    "holdout_sharpe_decay_max": 0.5,
    "holdout_se_z": 1.96,  # SE from the holdout's trading-day count
    "holdout_min_days": 250,
    "holdout_pnl_positive": True,  # holdout total return (equity) / PnL (fixed) > 0
    "holdout_alpha_positive": True,  # holdout alpha over the 1x underlying > 0
    # Owner decision 2026-10-24 afternoon #3 (set on the controls, before any real result): train alpha
    # over the 1x underlying > 0 and HAC t >= 1.65 (one-sided 5 %). Static 2-4x leverage of the underlying
    # (synthetic control, seeds 1-20) reached t >= 1.0 in 3/20 seeds, t >= 1.65 in none. Intraday keeps 1.0.
    "benchmark_alpha_t_min": 1.65,
    # Owner decision 2026-10-28: a second pass route for leveraged-index timing whose value is a smaller drawdown:
    # Calmar of the strategy vs the 1x underlying at constant leverage = its beta (gates/benchmark.py). Checked on
    # nulls before any real run was judged (tests/pipeline/benchmark_risk_calibration.py, TMF on TX 2001-2021):
    # constant leverage 0.5-3x 0/8; 200-day-MA-like random switching (2x/0.5x, 2x/0x, 3x/0.5x Markov, and the
    # real filter path shifted), 200 each: risk route alone <= 3.0 %, alpha or risk <= 4.5 %. A run frozen
    # without benchmark_rule judges alpha alone.
    "benchmark_rule": "alpha_or_beta_matched_risk",
    "benchmark_risk_block_days": 252,
    "benchmark_risk_resamples": 2000,
    "benchmark_risk_confidence": 0.95,
    "benchmark_risk_min_beta": 0.25,
    # 2026-10-29, before any run froze the rule: at train lengths (1260-5000 days, random windows 2001-2026) the
    # route passed 0-23 % of constant-leverage nulls on contract rounding alone (tracking <= 0.10 in every such
    # pass; constant nulls <= 0.17, switching nulls >= 0.09, p1 >= 0.15). Below this tracking the route does not
    # apply: constant nulls 0 %, switching nulls unchanged (tests/pipeline/holdout_risk_calibration.py). A config
    # frozen with benchmark_rule but without this key keeps the route without the floor.
    "benchmark_risk_min_tracking": 0.2,
    # Owner decision 2026-10-29: the holdout's alpha > 0 condition also passes on the risk route evaluated on the
    # holdout (gates/holdout.py), for runs frozen from today on. Nulls at holdout lengths 252-1260 days
    # (tests/pipeline/holdout_risk_calibration.py): with 21-day blocks at 99% and the tracking floor above, the risk
    # route alone passed <= 1.5% and never passed a null whose alpha failed. The 4.5% alpha-or-risk bar does not
    # apply here: holdout alpha > 0 alone passes 33-44% of the switching nulls. daily_fixed is judged in TWD, where
    # the route does not apply and alpha alone decides.
    "holdout_benchmark_rule": "alpha_or_beta_matched_risk",
    "holdout_risk_block_days": 21,
    "holdout_risk_confidence": 0.99,
    # 2026-10-29: the floor above holds while one contract is a small part of the matched run's exposure (TMF at
    # 1M, MTX at 5M, TX at 20M TWD: constant nulls 0 %). With one contract (MTX at 1-2.5M, TX at 1-5M TWD over
    # much of 2001-2026) rounding alone moves the matched run's exposure by half or all of it (tracking up to 1.9):
    # constant nulls passed up to 5 % (train) and 18 % of the spec_validate-feasible ones (holdout). A futures
    # matched run holding fewer contracts at the window's first or highest close is not applicable
    # (daily_stages.beta_matched_run): constant nulls <= 2.0 % on MTX / TX at 1-20M, switching nulls not higher,
    # TMF verdicts unchanged (tests/pipeline/holdout_risk_calibration.py). Missing: no such rule.
    "benchmark_risk_min_contracts": 2,
    "cost_stress_multiplier": 2.0,
    "cost_stress_pnl_positive": True,  # final equity > initial with every cost x2
    "dsr_min": 0.95,
    "dsr_min_obs": 250,  # daily returns
    "param_neighbourhood_profitable_min": 0.6,  # profitable = final equity > initial
    "param_min_neighbours": 2,
    # Owner decision 2026-10-28 (run #18): a neighbour cut off by a component's own hard bound (not the spec's
    # declared range) no longer sinks param_stability on insufficient_neighbours when every in-range neighbour
    # is profitable (gates/param_stability.py). A run frozen without this key keeps the old rule.
    "param_edge_at_bound_pass": True,
    "regime_profitable_segments_min": 3,
    "regime_min_segments": 3,
    "regime_min_days_per_segment": 60,
    "regime_trend_sma_days": 60,
    "regime_atr_days": 20,
    "regime_vol_median_days": 250,
    # §12 #5: forced liquidation or ruin in the train backtest fails; a margin call is only reported.
    "liquidation_fails": True,
    # Owner decision 2026-10-24 afternoon #2 (pre-registered before any real result): the same params on
    # train under each timing scenario (gates.execution_timing: the other fill, signal one day late)
    # must keep total PnL > 0 and daily Sharpe >= this share of the base run's.
    "execution_timing_pnl_positive": True,
    "execution_timing_sharpe_ratio_min": 0.5,
    # Daily shadow paper (§7; read by the daily paper mode, not by the gates): no trade count.
    "paper_min_trading_days": 20,
    "paper_target_match_min": 0.95,  # share of days whose paper target units equal the replay's
    "paper_pnl_tolerance_ticks": 1,  # |daily pnl diff| <= ticks x units x point value
}

DAILY_GATES: dict = {
    "cost_stress": {
        "scope": "commission, tax and slippage of every fill (rebalance, roll, margin_call, forced_liquidation) x multiplier",
    },
    "regime_split": {
        "daily_bars": "the 1x underlying's daily bars (benchmark.underlying)",
        "day_label": "state after the latest underlying bar dated strictly before the day",
    },
    "wfe": {
        # The first layout that fits in the train part (after warm-up) is used: the long history
        # first, the Thalassa-length layout as the fallback (§7).
        "layouts": [
            {"windows": 5, "is_days": 1460, "oos_days": 365},
            {"windows": 4, "is_days": 730, "oos_days": 182},
        ],
        "days": "calendar",
        "max_trials_per_window": 16,
        "budget_seconds": 3600,
        "objective": "sharpe_daily",
    },
    # Daily strategy metric regressed on the 1x underlying's daily return (§12 #8): static leverage
    # of the underlying has alpha ~ 0 - costs and fails.
    "benchmark": {
        "hac_lags": 5,
        "underlying": {
            "tw_futures": "TX back-adjusted continuous daily close (refdata TAIFEX daily, roll before settlement)",
            "tw_stock": "0050 total return",
            "multi": "0050 total return",
        },
    },
    "holdout": {"family_reuse": "report_only"},
    # Blocking (thresholds execution_timing_*): the selected params re-run over the train window per scenario.
    "execution_timing": {
        "scenarios": {
            "next_open": ["signal_close", "signal_lag_1d"],
            "signal_close": ["next_open", "signal_lag_1d"],
        },
        "signal_lag_1d": "the spec's own execution, every signal series one row late",
    },
}

# Daily target-position shadow paper (§7; shadow_paper_daily). Thresholds (paper_min_trading_days,
# paper_target_match_min, paper_pnl_tolerance_ticks) are in DAILY_THRESHOLDS above: one source.
DAILY_SHADOW_PAPER: dict = {
    "mode": "daily_target",
    "wake_local_time": "09:30",  # after the 17:40 refdata job of the previous day (refdata.daily)
    "wake_weekdays": [0, 1, 2, 3, 4, 5, 6],
    "source_retry_minutes": 60,  # refdata unreadable: try again this soon (never fails the stage)
    "stale_after_trading_days": 3,  # report (never fail) when the data lags more scheduled trading days than this
    "max_calendar_days": 90,  # fewer than paper_min_trading_days paper days this long after the start -> fail
    "calendar_horizon_days": 120,  # official schedule appended after the cutoff
    "protection": {
        "daily_loss_quantile": 0.95,
        "daily_loss_multiple": 3.0,
        "max_drawdown_multiple": 1.5,
        "fail_on_forced_liquidation": True,
    },
}

DEFAULT_THRESHOLDS_BY_CLASS: dict[str, dict] = {
    "intraday_fixed": DEFAULT_THRESHOLDS,  # the v1 / DSL class: today's thresholds, unchanged
    "daily_equity": DAILY_THRESHOLDS,
    "daily_fixed": DAILY_THRESHOLDS,
}
DEFAULT_GATES_BY_CLASS: dict[str, dict] = {
    "intraday_fixed": DEFAULT_GATES,
    "daily_equity": DAILY_GATES,
    "daily_fixed": DAILY_GATES,
}
# The intraday class keeps the v1 paper (DEFAULT_SHADOW_PAPER, frozen by default_config).
SHADOW_PAPER_BY_CLASS: dict[str, dict] = {
    "daily_equity": DAILY_SHADOW_PAPER,
    "daily_fixed": DAILY_SHADOW_PAPER,
}
# Classes a spec v3 run can be created for today (the daily engine); intraday v3 waits for DSL parity (step 7).
V3_RUN_CLASSES: tuple[str, ...] = ("daily_equity", "daily_fixed")
# A run frozen without ``strategy_class`` (every CONFIG_VERSION <= 5 run) is the v1 / DSL class.
LEGACY_STRATEGY_CLASS = "intraday_fixed"


def strategy_class_of(config) -> str:
    return (config or {}).get("strategy_class") or LEGACY_STRATEGY_CLASS


def class_settings(strategy_class: str) -> dict:
    """``{"thresholds", "gates"}`` (plus ``shadow_paper`` for the daily classes) frozen into a
    CONFIG_VERSION 6 run of ``strategy_class``.

    ``intraday_equity`` is deferred (§7) and has no entry: such a run cannot be frozen.
    """

    if strategy_class not in DEFAULT_THRESHOLDS_BY_CLASS:
        raise KeyError(f"no pre-registered thresholds for strategy class {strategy_class!r}")
    settings = {
        "thresholds": copy.deepcopy(DEFAULT_THRESHOLDS_BY_CLASS[strategy_class]),
        "gates": copy.deepcopy(DEFAULT_GATES_BY_CLASS[strategy_class]),
    }
    if strategy_class in SHADOW_PAPER_BY_CLASS:
        settings["shadow_paper"] = copy.deepcopy(SHADOW_PAPER_BY_CLASS[strategy_class])
    return settings


def data_config(*, as_of: date, holdout_fraction: float, start: str = DATA_START) -> dict:
    """Frozen data window: ``[start, end)`` in Asia/Taipei, plus the locked holdout split."""

    from poseidon.pipeline.window import holdout_split

    end = as_of.isoformat()
    return {
        "source": "thalassa",
        "interval": "5m",
        "timezone": "Asia/Taipei",
        "start": start,
        "end": end,
        "end_exclusive": True,
        "holdout_start": holdout_split(start, end, holdout_fraction).isoformat(),
    }


def stage_requires(stage, version: int) -> tuple[str, ...]:
    """A registry entry's ``requires`` at config ``version`` (an engine-routed stage answers per version)."""

    requires_at = getattr(stage, "requires_at", None)
    return requires_at(version) if requires_at is not None else requires_for(stage.requires, version)


def validate_stage_plan(
    stage_plan: list[str], registry, *, version: int = CONFIG_VERSION, routed: bool = False
) -> list[str]:
    """Reject unknown, repeated, out-of-order, or dependency-starved stages (as of config ``version``).

    ``routed``: the daily plan of an idea-only run (CONFIG_VERSION 7), which starts with spec_generate.
    """

    if not isinstance(stage_plan, list) or not stage_plan:
        raise StagePlanError("stage_plan must be a non-empty list")
    if len(set(stage_plan)) != len(stage_plan):
        raise StagePlanError("stage_plan repeats a stage")
    order = {name: index for index, name in enumerate(CANONICAL_ORDER)}
    for name in stage_plan:
        if name not in order or name not in registry:
            raise StagePlanError(f"unknown stage: {name}")
    if [order[name] for name in stage_plan] != sorted(order[name] for name in stage_plan):
        raise StagePlanError("stage_plan must follow the v1 stage order")
    if version >= SPEC_V3_CONFIG_VERSION:
        allowed = V3_ROUTED_DAILY_STAGE_PLAN if routed else V3_DAILY_STAGE_PLAN
        outside = [name for name in stage_plan if name not in allowed]
        if outside:
            raise StagePlanError(f"stage {outside[0]} is not part of a spec v3 daily run")
    seen: set[str] = set()
    for name in stage_plan:
        if STAGE_INTRODUCED_AT.get(name, 0) > version:
            raise StagePlanError(f"stage {name} does not exist at config version {version}")
        missing = [required for required in stage_requires(registry[name], version) if required not in seen]
        if missing:
            raise StagePlanError(f"stage {name} requires {', '.join(missing)} earlier in the plan")
        seen.add(name)
    return list(stage_plan)


# --- Step 6b: JEV spec faithfulness (frozen into spec v3 runs by v3.config.daily_run_config) ---
# Key, pin and identity check are spec_generate's (poseidon.pipeline.jev). ``blocking`` stays False --
# the verdict is pass with a warning summary -- until the owner's pipeline control set qualifies JEV
# (``qualification``, §4 / §12 #12); the flip is frozen into new runs only. ``precheck`` "strict": only a
# multiple or sign mismatch in a clause decides ``contradicts`` without JEV (clause_precheck).
DEFAULT_SPEC_FAITHFULNESS: dict = {
    "provider": "openrouter",
    "endpoint": "/systemone",
    "model": "typesafe/jev-1.13",
    "blocking": False,
    "qualification": {
        "status": "not_qualified",
        "control_set": "local_dev/design/pv2-jev-control-set.jsonl (owner-labelled)",
        "agreement_min": 0.95,
        "contradicts_recall_min": 1.0,
    },
    "precheck": "strict",
    "max_clauses": 40,
    # 3 questions per clause. Measured by research: ~$0.00002-0.00004 and ~470 ms per call.
    "max_calls_per_attempt": 120,
    "max_cost_usd_per_attempt": 0.05,
    "estimate_usd_per_call": 0.0002,  # booked when OpenRouter reports no cost (5-10x the measured price)
    "request_timeout_seconds": 30,
}


# --- Step 4: LLM spec generation (frozen per run) ---
# Key: STRATEGY_PIPELINE_OPENROUTER_KEY only (see poseidon.pipeline.spec_llm).
DEFAULT_SPEC_GENERATE: dict = {
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4.1-flash-1017",
    "max_repair_rounds": 2,
    "max_tokens": 6000,
    "temperature": 0.0,
    "reasoning_enabled": False,
    "max_cost_usd_per_attempt": 0.05,
    # Only used when OpenRouter returns no cost: conservative (2x the 1017 list price).
    "estimate_usd_per_mtok": {"prompt": 0.30, "completion": 1.20},
    "request_timeout_seconds": 120,
    "max_idea_chars": 4000,
}


# --- Step 6: spec_generate v3 (idea-only runs, CONFIG_VERSION 7) ---
# Same key, label pin and identity check as DSL generation (poseidon.pipeline.spec_llm). The registry
# catalog makes the prompt larger and a long idea needs one clause per sentence, so the per-attempt cap is
# 0.10 (design §3.8) and max_tokens is larger. An intraday idea is then generated by the DSL v2 path with
# the intraday_fixed config's own (CONFIG_VERSION 5) spec_generate section, cap 0.05, unchanged.
DEFAULT_SPEC_GENERATE_V3: dict = {
    **DEFAULT_SPEC_GENERATE,
    "prompt": "spec_v3",
    "max_tokens": 16000,
    "max_cost_usd_per_attempt": 0.10,
    "request_timeout_seconds": 240,
}
ROUTED_CLASSES: tuple[str, ...] = ("intraday_fixed", "daily_equity", "daily_fixed")


class RouteError(ValueError):
    """A routed run's spec_generate route does not name a class frozen into the run."""


def is_routed(config) -> bool:
    return config_version(config) == ROUTED_CONFIG_VERSION


def effective_config(config: dict, route: dict | None) -> dict:
    """The config a stage of the run sees.

    A run frozen at <= 6 is its own config. A routed run (7) is its own config until spec_generate passed
    with a ``route``; from then on it is the frozen ``by_class`` entry of the routed class. A daily entry
    froze every registry component's fingerprint at creation; only the components the generated spec uses
    stay, so the frozen-component check is the same as for a hand-written run of that spec. A route with
    ``data`` (a series the generated spec reads starts after the default window start; owner decision
    2026-10-25) replaces the entry's data window with the one spec_generate froze; the end and the entry's
    ``trading_calendar`` rule stay.
    """

    if not is_routed(config) or route is None:
        return config
    strategy_class = route.get("strategy_class") if isinstance(route, dict) else None
    variants = config.get("by_class") or {}
    if strategy_class not in variants:
        raise RouteError(f"spec_generate routed to {strategy_class!r}, not a class frozen into this run")
    variant = copy.deepcopy(variants[strategy_class])
    if "components" in variant:
        used = route.get("components")
        if not isinstance(used, list) or not used:
            raise RouteError("spec_generate route of a daily run lists no components")
        frozen = {key.split("@", 1)[0]: key for key in variant["components"]}
        unknown = sorted(set(used) - set(frozen))
        if unknown:
            raise RouteError(f"components {', '.join(unknown)} were not registered when the run was frozen")
        variant["components"] = {frozen[name]: variant["components"][frozen[name]] for name in sorted(set(used))}
    data = route.get("data")
    if data is not None:
        if not isinstance(data, dict) or data.get("end") != (variant.get("data") or {}).get("end"):
            raise RouteError("spec_generate route data must keep the frozen window end")
        frozen = variant.get("data") or {}
        variant["data"] = copy.deepcopy(data)
        for key in ("trading_calendar", "option_chain"):  # frozen rules spec_generate's window does not carry
            if key in frozen:
                variant["data"][key] = frozen[key]
    return variant
