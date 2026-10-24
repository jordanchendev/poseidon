"""CONFIG_VERSION 6 skeleton: the keys a spec v3 run will freeze (design 2026-10-24 §8).

Not wired into run creation yet (step 4 does that): ``plan.default_config``
still freezes CONFIG_VERSION 5 DSL v2 runs, and runs frozen at <= 5 have no
``engine`` key, so they keep the old runner, gates and paper byte for byte.
"""

from __future__ import annotations

import hashlib

from poseidon.pipeline.plan import SPEC_V3_CONFIG_VERSION
from poseidon.pipeline.v3.registry import REGISTRY_VERSION, component_fingerprints
from poseidon.pipeline.v3.spec import SPEC_VERSION, component_slots, strategy_class

ENGINE = "target_v1"
SPEC_VALIDATOR = "spec_v3"


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
