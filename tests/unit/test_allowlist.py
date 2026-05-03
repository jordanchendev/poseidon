"""Tests for the Phase 41 handler/model allowlist module (RESEARCH-API-02).

Pure Python unit tests -- no database, no SQLite shim needed. Validates that
the static allowlist dicts contain the expected entries and that the resolve
functions correctly accept valid names and reject unknowns with descriptive
``ValueError`` messages.
"""

from __future__ import annotations

import importlib

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


def test_allowed_model_classes_contains_zoo_entries():
    """Phase 94: LocalformerModel, TRAModel, ALSTM must be present with correct FQN."""
    assert "ALSTM" in ALLOWED_MODEL_CLASSES
    assert "TRAModel" in ALLOWED_MODEL_CLASSES
    assert "LocalformerModel" in ALLOWED_MODEL_CLASSES
    assert ALLOWED_MODEL_CLASSES["ALSTM"] == "qlib.contrib.model.pytorch_alstm.ALSTM"
    assert ALLOWED_MODEL_CLASSES["TRAModel"] == "qlib.contrib.model.pytorch_tra.TRAModel"
    assert ALLOWED_MODEL_CLASSES["LocalformerModel"] == "qlib.contrib.model.pytorch_localformer.LocalformerModel"


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
# Phase 94 — model zoo entries (D-01..D-03 amended; OQ-1 OPTION 1 Localformer)
# ---------------------------------------------------------------------------


def test_resolve_model_lgb():
    """resolve_model('LGBModel') returns the FQN AND the parent module imports."""
    fqn = resolve_model("LGBModel")
    assert fqn == "qlib.contrib.model.gbdt.LGBModel"
    pytest.importorskip("qlib")
    module_path, _ = fqn.rsplit(".", 1)
    importlib.import_module(module_path)


def test_resolve_model_linear():
    """resolve_model('LinearModel') returns the FQN AND the parent module imports."""
    fqn = resolve_model("LinearModel")
    assert fqn == "qlib.contrib.model.linear.LinearModel"
    pytest.importorskip("qlib")
    module_path, _ = fqn.rsplit(".", 1)
    importlib.import_module(module_path)


def test_resolve_model_xgb():
    """resolve_model('XGBModel') returns the FQN AND the parent module imports."""
    fqn = resolve_model("XGBModel")
    assert fqn == "qlib.contrib.model.xgboost.XGBModel"
    pytest.importorskip("qlib")
    module_path, _ = fqn.rsplit(".", 1)
    importlib.import_module(module_path)


def test_resolve_model_localformer():
    """resolve_model('LocalformerModel') returns the FQN AND the parent module imports.

    The importlib check is the Pitfall 1 catch-net: a typo in allowlist.py would
    pass string equality but break the worker at qlib_train time. Skipped when
    qlib is unavailable (Mac dev environment).
    """
    fqn = resolve_model("LocalformerModel")
    assert fqn == "qlib.contrib.model.pytorch_localformer.LocalformerModel"
    pytest.importorskip("qlib")
    module_path, _ = fqn.rsplit(".", 1)
    importlib.import_module(module_path)  # raises ModuleNotFoundError on bad path


def test_resolve_model_tra():
    """resolve_model('TRAModel') returns the FQN AND the parent module imports."""
    fqn = resolve_model("TRAModel")
    assert fqn == "qlib.contrib.model.pytorch_tra.TRAModel"
    pytest.importorskip("qlib")
    module_path, _ = fqn.rsplit(".", 1)
    importlib.import_module(module_path)


def test_resolve_model_alstm():
    """resolve_model('ALSTM') returns the FQN AND the parent module imports."""
    fqn = resolve_model("ALSTM")
    assert fqn == "qlib.contrib.model.pytorch_alstm.ALSTM"
    pytest.importorskip("qlib")
    module_path, _ = fqn.rsplit(".", 1)
    importlib.import_module(module_path)


def test_allowed_model_classes_pinned_keys():
    """Pin the exact 6-key set so accidental partial deletion is caught."""
    expected = {"LGBModel", "LinearModel", "XGBModel", "ALSTM", "TRAModel", "LocalformerModel"}
    assert set(ALLOWED_MODEL_CLASSES) == expected
    assert len(ALLOWED_MODEL_CLASSES) == 6


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
