"""The v1 stage plan and the frozen per-run configuration."""

from __future__ import annotations

import copy
from datetime import date

V1_STAGE_PLAN: tuple[str, ...] = (
    "spec_generate",
    "spec_validate",
    "backtest",
    "optimize",
    "wfe",
    "holdout",
    "cost_stress",
    "dsr",
    "param_stability",
    "regime_split",
    "shadow_paper",
    "awaiting_release",
)

# Initial gate thresholds (PLAN "Assumptions [minor]"). They are frozen into each
# run's config_json; v1 never accepts client overrides -- a new threshold means
# a new run in the same family.
DEFAULT_THRESHOLDS: dict = {
    "wfe_min": 0.5,
    "holdout_sharpe_decay_max": 0.5,
    "holdout_pnl_positive": True,
    "holdout_min_trades": 20,
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

CONFIG_VERSION = 2
DEFAULT_SPEC_VALIDATOR = "dsl_v2"


class StagePlanError(ValueError):
    """The requested stage plan is not executable."""


def default_config(*, stage_plan: list[str] | None, as_of: date) -> dict:
    """Return the server-side configuration frozen into a new run."""

    return {
        "version": CONFIG_VERSION,
        "spec_generate": copy.deepcopy(DEFAULT_SPEC_GENERATE),
        "stage_plan": list(stage_plan) if stage_plan is not None else list(V1_STAGE_PLAN),
        "thresholds": copy.deepcopy(DEFAULT_THRESHOLDS),
        "data": data_config(as_of=as_of, holdout_fraction=HOLDOUT_FRACTION),
        "holdout": {"fraction": HOLDOUT_FRACTION, "locked": True},
        "seed": 0,
        "backtest": copy.deepcopy(DEFAULT_BACKTEST),
        "optimize": copy.deepcopy(DEFAULT_OPTIMIZE),
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


def validate_stage_plan(stage_plan: list[str], registry) -> list[str]:
    """Reject unknown, repeated, out-of-order, or dependency-starved stages."""

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
        missing = [required for required in registry[name].requires if required not in seen]
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
