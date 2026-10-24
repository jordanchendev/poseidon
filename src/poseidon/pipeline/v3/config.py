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
    CANONICAL_ORDER,
    DAILY_BACKTEST,
    DAILY_DATA_START,
    DEFAULT_OPTIMIZE,
    DEFAULT_SPEC_GENERATE_V3,
    HOLDOUT_FRACTION,
    ROUTED_CLASSES,
    ROUTED_CONFIG_VERSION,
    SPEC_V3_CONFIG_VERSION,
    V1_STAGE_PLAN,
    V3_DAILY_STAGE_PLAN,
    V3_ROUTED_DAILY_STAGE_PLAN,
    V3_RUN_CLASSES,
    class_settings,
    config_version,
    default_config,
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
    return _daily_config(
        fields, as_of=as_of, refdata=refdata, stage_plan=stage_plan if stage_plan is not None else V3_DAILY_STAGE_PLAN
    )


def _daily_config(fields: dict, *, as_of: date, refdata: dict[str, str], stage_plan) -> dict:
    settings = class_settings(fields["strategy_class"])
    if "shadow_paper" not in settings:
        raise KeyError(f"strategy class {fields['strategy_class']} has no daily shadow paper settings")
    return {
        **fields,
        "stage_plan": list(stage_plan),
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
    }


# --------------------------------------------------------------------------- idea-only runs (CONFIG_VERSION 7)

SPEC_SOURCE_GENERATED = "spec_generate"


def class_plan(run_class: str) -> tuple[str, ...]:
    """The full stage plan an idea-only run follows once spec_generate routed it to ``run_class``."""

    return V3_ROUTED_DAILY_STAGE_PLAN if run_class in V3_RUN_CLASSES else V1_STAGE_PLAN


def routed_daily_config(daily_class: str, *, as_of: date, brief_text: str, refdata: dict[str, str], stage_plan) -> dict:
    """The CONFIG_VERSION 6 config of one daily class, frozen before the spec exists.

    Spec-dependent fields are frozen for every candidate instead: ``components`` holds every registered
    component's fingerprint (``plan.effective_config`` keeps the ones the generated spec uses), and
    ``spec_source`` tells ``spec_validate`` to read spec_generate's spec instead of ``input_spec_json``.
    """

    fields = {
        "version": SPEC_V3_CONFIG_VERSION,
        "spec_version": SPEC_VERSION,
        "spec_validator": SPEC_VALIDATOR,
        "engine": ENGINE,
        "registry_version": REGISTRY_VERSION,
        "components": component_fingerprints(),
        "strategy_class": daily_class,
        "capability_brief_sha256": hashlib.sha256(brief_text.encode("utf-8")).hexdigest(),
        "spec_source": SPEC_SOURCE_GENERATED,
    }
    return _daily_config(fields, as_of=as_of, refdata=refdata, stage_plan=stage_plan)


def idea_run_config(
    *,
    as_of: date,
    brief_text: str,
    refdata: dict[str, str] | None,
    classes: tuple[str, ...] = ROUTED_CLASSES,
    plans: dict[str, list[str]] | None = None,
    refdata_error: str | None = None,
) -> dict:
    """CONFIG_VERSION 7: the routed config of an idea-only run (see ``plan.ROUTED_CONFIG_VERSION``).

    ``classes``: the candidate classes (a family member: its family's class only). A daily class is frozen
    only when the refdata hashes are known (``refdata``); otherwise it is listed under ``unavailable`` with
    ``refdata_error`` and a daily route ends spec_generate as unsupported. ``plans``: per-class stage plans
    (a client's truncated plan); by default each class's full plan. The top-level ``stage_plan`` lists every
    stage a candidate class can run, in canonical order: what the run shows before it is routed.
    """

    plans = plans or {}
    by_class: dict[str, dict] = {}
    unavailable: dict[str, str] = {}
    for name in classes:
        plan = list(plans.get(name) or class_plan(name))
        if name in V3_RUN_CLASSES:
            if refdata is None:
                unavailable[name] = f"refdata was not available at run creation: {refdata_error}"[:300]
                continue
            by_class[name] = routed_daily_config(
                name, as_of=as_of, brief_text=brief_text, refdata=refdata, stage_plan=plan
            )
        else:
            by_class[name] = default_config(stage_plan=plan, as_of=as_of)
    stages = {name for variant in by_class.values() for name in variant["stage_plan"]} or {"spec_generate"}
    config = {
        "version": ROUTED_CONFIG_VERSION,
        "spec_generate": copy.deepcopy(DEFAULT_SPEC_GENERATE_V3),
        "routing": {
            "by": "spec_generate",
            "classes": sorted(by_class),
            "note": "spec_generate's report names the class (route); later stages run on by_class[class]",
        },
        "stage_plan": [name for name in CANONICAL_ORDER if name in stages],
        "by_class": by_class,
    }
    if unavailable:
        config["unavailable"] = unavailable
    return config
