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
STAGE_INTRODUCED_AT: dict[str, int] = {"benchmark": 5}
# The gates every run with a holdout has had (completion_reason).
CORE_GATE_STAGES: tuple[str, ...] = tuple(name for name in GATE_STAGES if name not in STAGE_INTRODUCED_AT)
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


def validate_stage_plan(stage_plan: list[str], registry, *, version: int = CONFIG_VERSION) -> list[str]:
    """Reject unknown, repeated, out-of-order, or dependency-starved stages (as of config ``version``)."""

    if not isinstance(stage_plan, list) or not stage_plan:
        raise StagePlanError("stage_plan must be a non-empty list")
    if len(set(stage_plan)) != len(stage_plan):
        raise StagePlanError("stage_plan repeats a stage")
    order = {name: index for index, name in enumerate(V1_STAGE_PLAN)}
    for name in stage_plan:
        if name not in order or name not in registry:
            raise StagePlanError(f"unknown stage: {name}")
    if [order[name] for name in stage_plan] != sorted(order[name] for name in stage_plan):
        raise StagePlanError("stage_plan must follow the v1 stage order")
    seen: set[str] = set()
    for name in stage_plan:
        if STAGE_INTRODUCED_AT.get(name, 0) > version:
            raise StagePlanError(f"stage {name} does not exist at config version {version}")
        missing = [required for required in requires_for(registry[name].requires, version) if required not in seen]
        if missing:
            raise StagePlanError(f"stage {name} requires {', '.join(missing)} earlier in the plan")
        seen.add(name)
    return list(stage_plan)


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
