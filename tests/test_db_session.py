"""Unit tests for core/database.py session management.

Tests db_session() context manager lifecycle and backward-compatible imports.
Uses SQLite in-memory to avoid requiring PostgreSQL.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker


# ---------------------------------------------------------------------------
# Fixtures: test engine + sessionmaker using SQLite in-memory
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_session_factory():
    """Create an in-memory SQLite sessionmaker for testing."""
    test_engine = create_engine("sqlite:///:memory:")
    return sessionmaker(bind=test_engine)


# ---------------------------------------------------------------------------
# Test 1: db_session() yields a Session and closes on normal exit
# ---------------------------------------------------------------------------

def test_db_session_yields_session_and_closes(test_session_factory):
    """db_session() must yield a Session object and call close() on exit."""
    mock_session = MagicMock(spec=Session)
    patched_factory = MagicMock(return_value=mock_session)

    with patch("poseidon.core.database.SessionLocal", patched_factory):
        from poseidon.core.database import db_session

        with db_session() as session:
            assert session is mock_session

    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: db_session() rolls back and closes on exception
# ---------------------------------------------------------------------------

def test_db_session_rollback_on_exception(test_session_factory):
    """db_session() must call rollback() then close() on exception."""
    mock_session = MagicMock(spec=Session)
    patched_factory = MagicMock(return_value=mock_session)

    with patch("poseidon.core.database.SessionLocal", patched_factory):
        from poseidon.core.database import db_session

        with pytest.raises(ValueError, match="boom"):
            with db_session() as session:
                raise ValueError("boom")

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3: db_session() does NOT auto-commit on normal exit
# ---------------------------------------------------------------------------

def test_db_session_no_auto_commit(test_session_factory):
    """db_session() must NOT call commit() on normal exit."""
    mock_session = MagicMock(spec=Session)
    patched_factory = MagicMock(return_value=mock_session)

    with patch("poseidon.core.database.SessionLocal", patched_factory):
        from poseidon.core.database import db_session

        with db_session() as session:
            pass  # normal exit, no explicit commit

    mock_session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: get_db() yields a Session and closes on exit (FastAPI dependency)
# ---------------------------------------------------------------------------

def test_get_db_yields_and_closes(test_session_factory):
    """get_db() generator must yield a Session and call close()."""
    mock_session = MagicMock(spec=Session)
    patched_factory = MagicMock(return_value=mock_session)

    with patch("poseidon.core.database.SessionLocal", patched_factory):
        from poseidon.core.database import get_db

        gen = get_db()
        session = next(gen)
        assert session is mock_session

        # Simulate FastAPI calling .close() on the generator
        try:
            next(gen)
        except StopIteration:
            pass

    mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test 5: SessionLocal importable from poseidon.core.database
# ---------------------------------------------------------------------------

def test_session_local_importable_from_core_database():
    """SessionLocal must be importable from poseidon.core.database."""
    from poseidon.core.database import SessionLocal

    assert SessionLocal is not None
    assert callable(SessionLocal)


# ---------------------------------------------------------------------------
# Test 6: SessionLocal still importable from poseidon.models (backward compat)
# ---------------------------------------------------------------------------

def test_session_local_backward_compat_from_models():
    """SessionLocal must still be importable from poseidon.models."""
    from poseidon.models import SessionLocal

    assert SessionLocal is not None
    assert callable(SessionLocal)


# ---------------------------------------------------------------------------
# Test 7: engine importable from poseidon.core.database
# ---------------------------------------------------------------------------

def test_engine_importable_from_core_database():
    """engine must be importable from poseidon.core.database."""
    from poseidon.core.database import engine

    assert engine is not None


# ---------------------------------------------------------------------------
# Test 8: db_session importable from poseidon.core.database
# ---------------------------------------------------------------------------

def test_db_session_importable():
    """db_session must be importable from poseidon.core.database."""
    from poseidon.core.database import db_session

    assert callable(db_session)
