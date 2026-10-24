"""The CONFIG_VERSION 6 configuration a spec v3 run freezes (design 2026-10-24 §8).

``daily_run_config`` assembles everything a daily v3 run is judged by, at creation, from the
spec document alone plus the refdata content hashes: the spec/registry identity
(``v3_config_fields``), the class table entry of the spec's ``strategy_class``
(``plan.class_settings``: thresholds, gate settings, daily shadow paper), the daily data window
with its locked holdout, and the optimizer/backtest settings. Runs frozen at <= 5 have no
``engine`` key and keep the old runner, gates and paper byte for byte (``is_v3_run``).
"""

from __future__ import annotations

import copy
import hashlib
from datetime import date

from poseidon.pipeline.plan import (
    DAILY_BACKTEST,
    DAILY_DATA_START,
    DEFAULT_OPTIMIZE,
    DEFAULT_SPEC_FAITHFULNESS,
    HOLDOUT_FRACTION,
    SPEC_V3_CONFIG_VERSION,
    V3_DAILY_STAGE_PLAN,
    class_settings,
    config_version,
)
from poseidon.pipeline.v3.registry import REGISTRY_VERSION, component_fingerprints
from poseidon.pipeline.v3.spec import SPEC_VERSION, component_slots, strategy_class

ENGINE = "target_v1"
SPEC_VALIDATOR = "spec_v3"
# Datasets a daily run reads; their manifest content_sha256 at creation is frozen as provenance.
REFDATA_DATASETS = (
    "taifex_futures_daily",
    "taifex_settlement_calendar",
    "twse_market_calendar",
    "twse_taiex_daily",
    "twse_taiex_total_return",
)


def is_v3_run(config) -> bool:
    """A spec v3 run: frozen at CONFIG_VERSION >= 6 on the target-position engine."""

    return config_version(config) >= SPEC_V3_CONFIG_VERSION and (config or {}).get("engine") == ENGINE


def v3_config_fields(resolved: dict, *, brief_text: str) -> dict:
    """The v3-only keys merged over the base config when a v3 run is created (step 4)."""

    used = sorted({ref["component"] for _path, _slot, ref in component_slots(resolved)})
    return {
        "version": SPEC_V3_CONFIG_VERSION,
        "spec_version": SPEC_VERSION,
        "spec_validator": SPEC_VALIDATOR,
        "engine": ENGINE,
        "registry_version": REGISTRY_VERSION,
        "components": component_fingerprints(used),
        "strategy_class": strategy_class(resolved),
        "capability_brief_sha256": hashlib.sha256(brief_text.encode("utf-8")).hexdigest(),
    }


def daily_data_config(*, as_of: date, start: str = DAILY_DATA_START, fraction: float = HOLDOUT_FRACTION) -> dict:
    """``[start, as_of)`` in trading days (Asia/Taipei dates); the last ``fraction`` is the locked holdout."""

    from poseidon.pipeline.window import MARKET_TZ, holdout_split

    split = holdout_split(start, as_of, fraction).astimezone(MARKET_TZ).date()
    return {
        "source": "refdata",
        "frequency": "1d",
        "timezone": "Asia/Taipei",
        "start": start,
        "end": as_of.isoformat(),
        "end_exclusive": True,
        "holdout_start": split.isoformat(),
        "parts": "train = trading days in [start, holdout_start); holdout = [holdout_start, end)",
    }


def daily_run_config(
    resolved: dict,
    *,
    as_of: date,
    brief_text: str,
    refdata: dict[str, str],
    stage_plan: list[str] | None = None,
) -> dict:
    """Everything a daily v3 run is frozen with (``resolved``: the spec with every ``$param`` at its default)."""

    fields = v3_config_fields(resolved, brief_text=brief_text)
    settings = class_settings(fields["strategy_class"])
    if "shadow_paper" not in settings:
        raise KeyError(f"strategy class {fields['strategy_class']} has no daily shadow paper settings")
    return {
        **fields,
        "stage_plan": list(stage_plan) if stage_plan is not None else list(V3_DAILY_STAGE_PLAN),
        "thresholds": settings["thresholds"],
        "gates": settings["gates"],
        "shadow_paper": settings["shadow_paper"],
        "data": daily_data_config(as_of=as_of),
        "holdout": {"fraction": HOLDOUT_FRACTION, "locked": True},
        "refdata": {
            "datasets": dict(sorted(refdata.items())),
            "note": "manifest content_sha256 at run creation (provenance); the daily job appends rows later, "
            "each stage records the hash of the window it actually read",
        },
        "seed": 0,
        "backtest": copy.deepcopy(DAILY_BACKTEST),
        "optimize": copy.deepcopy(DEFAULT_OPTIMIZE),
        "spec_faithfulness": copy.deepcopy(DEFAULT_SPEC_FAITHFULNESS),
    }
