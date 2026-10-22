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

CONFIG_VERSION = 1
DEFAULT_SPEC_VALIDATOR = "dsl_v1"


class StagePlanError(ValueError):
    """The requested stage plan is not executable."""


def default_config(*, stage_plan: list[str] | None, as_of: date) -> dict:
    """Return the server-side configuration frozen into a new run."""

    return {
        "version": CONFIG_VERSION,
        "stage_plan": list(stage_plan) if stage_plan is not None else list(V1_STAGE_PLAN),
        "thresholds": copy.deepcopy(DEFAULT_THRESHOLDS),
        "data": {"interval": "5m", "start": "2020-03-02", "end": as_of.isoformat()},
        "holdout": {"fraction": 0.2, "locked": True},
        "seed": 0,
        "max_trials": 200,
        "spec_validator": DEFAULT_SPEC_VALIDATOR,
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
