"""Strategy spec v3: component registry, instrument table, validator, renderers (design 2026-10-24, step 1).

Importing the package registers the components.
"""

from poseidon.pipeline.v3 import components  # noqa: F401  (registers the components)
from poseidon.pipeline.v3.dsl_bridge import NotAFastPathSpec, v2_to_v3, v3_to_v2
from poseidon.pipeline.v3.instruments import INSTRUMENTS, Instrument
from poseidon.pipeline.v3.registry import REGISTRY, REGISTRY_VERSION, component_fingerprints
from poseidon.pipeline.v3.validation import FeasibilityContext, SpecIssue, V3Validation, validate_spec_v3

__all__ = [
    "INSTRUMENTS",
    "REGISTRY",
    "REGISTRY_VERSION",
    "FeasibilityContext",
    "Instrument",
    "NotAFastPathSpec",
    "SpecIssue",
    "V3Validation",
    "component_fingerprints",
    "v2_to_v3",
    "v3_to_v2",
    "validate_spec_v3",
]
