"""The released-config condition schema is derived from the DSL condition registry."""

from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from poseidon.decision_loop.strategy_config import ExecutableCondition
from poseidon.strategies.dsl.conditions import (
    CONDITION_REGISTRY,
    CONDITION_SCHEMAS,
    VolumeSpikeCondition,
    register_condition,
)

EXAMPLES = {
    "indicator_above": {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 70},
    "indicator_below": {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
    "price_crosses": {"type": "price_crosses", "indicator": "sma", "params": {"period": 20}},
    "indicator_crosses": {"type": "indicator_crosses", "params": {"fast": 5, "slow": 20}},
    "price_change_pct": {"type": "price_change_pct", "threshold": 0.02},
    "volume_spike": {"type": "volume_spike"},
    "bollinger_width_percentile": {"type": "bollinger_width_percentile"},
    "indicator_comparison": {
        "type": "indicator_comparison",
        "indicator_a": "sma",
        "indicator_b": "sma",
        "params": {"period_a": 5, "period_b": 20},
    },
    "feature_above": {"type": "feature_above", "column": "close", "threshold": 100},
    "feature_below": {"type": "feature_below", "column": "close", "threshold": 100},
    "ml_prediction": {"type": "ml_prediction", "model_version_id": "00000000-0000-4000-8000-0000000000a1"},
}


def test_every_registered_evaluator_has_the_schema_that_names_it():
    assert set(CONDITION_SCHEMAS) == set(CONDITION_REGISTRY)
    named = {name for schema in CONDITION_SCHEMAS.values() for name in get_args(schema.model_fields["type"].annotation)}
    assert named == set(CONDITION_REGISTRY)


def test_executable_condition_leaves_are_the_registered_schemas():
    leaves = get_args(ExecutableCondition)[4:]
    assert leaves == tuple(dict.fromkeys(CONDITION_SCHEMAS.values()))


@pytest.mark.parametrize("name", sorted(CONDITION_REGISTRY))
def test_each_registered_condition_type_validates_as_its_schema(name):
    parsed = TypeAdapter(ExecutableCondition).validate_python(EXAMPLES[name])

    assert type(parsed) is CONDITION_SCHEMAS[name]


def test_unregistered_condition_type_is_rejected():
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutableCondition).validate_python({"type": "moon_phase"})


def test_a_schema_must_name_the_type_it_registers():
    with pytest.raises(ValueError, match="does not accept type"):
        register_condition("moon_phase", schema=VolumeSpikeCondition)
    assert "moon_phase" not in CONDITION_REGISTRY
