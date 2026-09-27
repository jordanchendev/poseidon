"""Unit tests for INGEST_CURSOR_MODE config flag (DATA-FOUND-03).

Implemented in plan 38-02.
"""

import importlib

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.phase38


def _fresh_settings():
    """Rebuild Settings so env changes are picked up."""
    import poseidon.core.config as cfg

    importlib.reload(cfg)
    return cfg.Settings()


def test_execution_mode_defaults_preserve_legacy():
    from poseidon.core.config import Settings

    config = Settings(_env_file=None)
    assert config.decision_loop_execution_mode == "legacy"
    assert config.decision_loop_execution_enabled is False
    assert config.decision_loop_approved_account_scope == ""
    assert config.decision_loop_approved_market == ""
    assert config.decision_loop_approved_account_generation == ""
    assert config.decision_loop_legacy_protective_scopes == ()


@pytest.mark.parametrize(
    "values",
    [
        {"decision_loop_execution_mode": "unknown"},
        {"decision_loop_execution_mode": "decision"},
        {"decision_loop_execution_enabled": True},
        {"decision_loop_execution_mode": "shadow", "decision_loop_execution_enabled": True},
        {"decision_loop_execution_mode": "decision", "decision_loop_execution_enabled": True},
        {"decision_loop_approved_account_scope": "paper:pilot"},
        {"decision_loop_legacy_protective_scopes": ["*"]},
    ],
)
def test_execution_configuration_invalid_combinations_fail_closed(values):
    from poseidon.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


@pytest.mark.parametrize("enabled", [False, True])
def test_halted_identity_survives_enabled_toggle(enabled):
    from poseidon.core.config import Settings

    config = Settings(
        _env_file=None,
        decision_loop_execution_mode="halted",
        decision_loop_execution_enabled=enabled,
        decision_loop_approved_account_scope="paper:owner",
        decision_loop_approved_market="tw_stock",
        decision_loop_approved_account_generation="owner-generation",
    )
    assert config.decision_loop_execution_mode == "halted"
    assert config.decision_loop_approved_account_scope == "paper:owner"
