"""Tests for the Phase 41 handler/model allowlist module (RESEARCH-API-02).

Pure Python unit tests -- no database, no SQLite shim needed. Validates that
the static allowlist dicts contain the expected entries and that the resolve
functions correctly accept valid names and reject unknowns with descriptive
``ValueError`` messages.
"""

from __future__ import annotations

import pytest

from poseidon.qlib.allowlist import (
    ALLOWED_HANDLER_CLASSES,
    ALLOWED_MODEL_CLASSES,
    resolve_handler,
    resolve_model,
)

# ---------------------------------------------------------------------------
# Dict contents
# ---------------------------------------------------------------------------


def test_allowed_handler_classes_contains_expected_entries():
    """Alpha158Handler and Alpha360Handler must be present with PoseidonDataHandler paths."""
    assert "Alpha158Handler" in ALLOWED_HANDLER_CLASSES
    assert "Alpha360Handler" in ALLOWED_HANDLER_CLASSES
    assert ALLOWED_HANDLER_CLASSES["Alpha158Handler"] == "poseidon.qlib.data_handler.PoseidonDataHandler"
    assert ALLOWED_HANDLER_CLASSES["Alpha360Handler"] == "poseidon.qlib.data_handler.PoseidonDataHandler"


def test_allowed_model_classes_contains_expected_entries():
    """LGBModel, LinearModel, XGBModel must be present with correct import paths."""
    assert "LGBModel" in ALLOWED_MODEL_CLASSES
    assert "LinearModel" in ALLOWED_MODEL_CLASSES
    assert "XGBModel" in ALLOWED_MODEL_CLASSES
    assert ALLOWED_MODEL_CLASSES["LGBModel"] == "qlib.contrib.model.gbdt.LGBModel"
    assert ALLOWED_MODEL_CLASSES["LinearModel"] == "qlib.contrib.model.linear.LinearModel"
    assert ALLOWED_MODEL_CLASSES["XGBModel"] == "qlib.contrib.model.xgboost.XGBModel"


# ---------------------------------------------------------------------------
# resolve_handler
# ---------------------------------------------------------------------------


def test_resolve_handler_valid():
    """resolve_handler with a valid name returns the correct import path."""
    assert resolve_handler("Alpha158Handler") == "poseidon.qlib.data_handler.PoseidonDataHandler"
    assert resolve_handler("Alpha360Handler") == "poseidon.qlib.data_handler.PoseidonDataHandler"


def test_resolve_handler_invalid_raises_valueerror():
    """resolve_handler with an unknown name raises ValueError with 'Allowed:' in message."""
    with pytest.raises(ValueError, match="Allowed:"):
        resolve_handler("EvilHandler")


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_valid():
    """resolve_model with a valid name returns the correct import path."""
    assert resolve_model("LGBModel") == "qlib.contrib.model.gbdt.LGBModel"
    assert resolve_model("LinearModel") == "qlib.contrib.model.linear.LinearModel"
    assert resolve_model("XGBModel") == "qlib.contrib.model.xgboost.XGBModel"


def test_resolve_model_invalid_raises_valueerror():
    """resolve_model with an unknown name raises ValueError with 'Allowed:' in message."""
    with pytest.raises(ValueError, match="Allowed:"):
        resolve_model("EvilModel")


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


def test_allowlist_is_not_empty():
    """Both dicts must have at least 1 entry (guard against accidental clearing)."""
    assert len(ALLOWED_HANDLER_CLASSES) >= 1
    assert len(ALLOWED_MODEL_CLASSES) >= 1


def test_allowlist_values_are_importable_paths():
    """All values must be valid dotted Python import paths (contain at least one '.')."""
    for name, path in ALLOWED_HANDLER_CLASSES.items():
        assert "." in path, f"Handler {name!r} path {path!r} is not a dotted import path"
    for name, path in ALLOWED_MODEL_CLASSES.items():
        assert "." in path, f"Model {name!r} path {path!r} is not a dotted import path"
