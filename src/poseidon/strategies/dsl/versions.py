"""DSL version dispatch.

``parse_rule_config()`` is the single entry point that turns a raw rule
document into a validated config object. Documents without ``dsl_version``
(or with ``dsl_version: 1``) take the untouched v1 path.
"""

from __future__ import annotations

from typing import Any

from poseidon.strategies.dsl.schema import RuleConfig
from poseidon.strategies.dsl.v2.schema import StrategyV2
from poseidon.strategies.dsl.v2.validate import validate_v2


def dsl_version_of(doc: dict[str, Any]) -> int:
    """Return the declared DSL version of a raw document (default 1)."""
    version = doc.get("dsl_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"dsl_version must be an integer, got {version!r}")
    return version


def parse_rule_config(doc: dict[str, Any] | RuleConfig | StrategyV2) -> RuleConfig | StrategyV2:
    """Validate a raw rule document, dispatching on ``dsl_version``.

    v1 returns :class:`RuleConfig` (unchanged path). v2 returns the resolved
    :class:`StrategyV2` (``$param`` defaults substituted) and raises
    :class:`DslValidationError` / :class:`DslUnsupportedError` with paths.
    """
    if isinstance(doc, RuleConfig | StrategyV2):
        return doc
    version = dsl_version_of(doc)
    if version == 1:
        return RuleConfig(**doc)
    if version == 2:
        return validate_v2(doc).model
    raise ValueError(f"Unsupported dsl_version: {version}")
