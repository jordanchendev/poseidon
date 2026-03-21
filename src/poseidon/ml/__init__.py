"""Model Engine — ML model training, versioning, and lifecycle management."""

from poseidon.ml.base import BaseModel
from poseidon.ml.lifecycle import InvalidTransitionError, validate_transition
from poseidon.ml.registry import get_model, list_models, register_model

# Import implementations to trigger registration
from poseidon.ml import implementations  # noqa: F401

__all__ = [
    "BaseModel",
    "register_model",
    "get_model",
    "list_models",
    "validate_transition",
    "InvalidTransitionError",
]
