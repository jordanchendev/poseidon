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


def test_ingest_cursor_mode_default(monkeypatch):
    # Default (no env var) → legacy
    monkeypatch.delenv("INGEST_CURSOR_MODE", raising=False)
    monkeypatch.delenv("POSEIDON_INGEST_CURSOR_MODE", raising=False)
    s = _fresh_settings()
    assert s.ingest_cursor_mode == "legacy"

    # Explicit cursor via unprefixed env var
    monkeypatch.setenv("INGEST_CURSOR_MODE", "cursor")
    s = _fresh_settings()
    assert s.ingest_cursor_mode == "cursor"

    # Invalid value → pydantic ValidationError
    monkeypatch.setenv("INGEST_CURSOR_MODE", "bogus")
    with pytest.raises(ValidationError):
        _fresh_settings()

    # Back to legacy
    monkeypatch.setenv("INGEST_CURSOR_MODE", "legacy")
    s = _fresh_settings()
    assert s.ingest_cursor_mode == "legacy"
