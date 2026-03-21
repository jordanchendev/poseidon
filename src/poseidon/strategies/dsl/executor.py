"""Recursive tree evaluator for DSL condition trees.

Handles all/any/none combinator nodes and dispatches leaf nodes
to registered condition evaluators.
"""

import pandas as pd

from poseidon.strategies.dsl.conditions import CONDITION_REGISTRY


def evaluate_condition(
    condition: dict,
    features: pd.DataFrame,
    row_idx: int,
    *,
    max_depth: int = 10,
    _current_depth: int = 0,
) -> bool:
    """Recursively evaluate a DSL condition tree against a feature DataFrame row.

    Args:
        condition: Condition dict — either a combinator (all/any/none)
                   or a leaf (type + parameters).
        features: Wide DataFrame from FeatureEngine.
        row_idx: Which row to evaluate (integer positional index).
        max_depth: Maximum nesting depth.
        _current_depth: Internal depth tracker.

    Returns:
        True if the condition is satisfied.

    Raises:
        ValueError: If condition type is unknown or max depth exceeded.
    """
    if _current_depth > max_depth:
        raise ValueError(f"DSL condition nesting exceeds max depth ({max_depth})")

    next_depth = _current_depth + 1

    # Combinator nodes
    if "all" in condition:
        return all(
            evaluate_condition(c, features, row_idx, max_depth=max_depth, _current_depth=next_depth)
            for c in condition["all"]
        )
    if "any" in condition:
        return any(
            evaluate_condition(c, features, row_idx, max_depth=max_depth, _current_depth=next_depth)
            for c in condition["any"]
        )
    if "none" in condition:
        return not any(
            evaluate_condition(c, features, row_idx, max_depth=max_depth, _current_depth=next_depth)
            for c in condition["none"]
        )

    # Leaf node — dispatch to registered condition evaluator
    cond_type = condition.get("type")
    if cond_type not in CONDITION_REGISTRY:
        raise ValueError(
            f"Unknown condition type: '{cond_type}'. "
            f"Available: {list(CONDITION_REGISTRY.keys())}"
        )
    return CONDITION_REGISTRY[cond_type](condition, features, row_idx)
