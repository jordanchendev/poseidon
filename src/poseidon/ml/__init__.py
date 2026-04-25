"""Model Engine — ML model training, versioning, and lifecycle management."""

# Import implementations to trigger registration
from poseidon.ml import implementations  # noqa: F401
from poseidon.ml.base import BaseModel
from poseidon.ml.lifecycle import InvalidTransitionError, validate_transition
from poseidon.ml.registry import get_model, list_models, register_model

__all__ = [
    "BaseModel",
    "InvalidTransitionError",
    "get_model",
    "list_models",
    "register_model",
    "validate_transition",
]
