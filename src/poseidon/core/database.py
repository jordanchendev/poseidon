"""Centralized database session management.

Provides:
- engine: SQLAlchemy engine bound to settings.database_url
- SessionLocal: sessionmaker factory
- db_session(): Context manager for Celery tasks / standalone code
- get_db(): FastAPI dependency for request-scoped sessions
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from poseidon.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def db_session():
    """Yield a SQLAlchemy session with automatic rollback and close.

    Usage in Celery tasks:
        with db_session() as session:
            result = session.query(Model).filter(...).all()
            session.add(new_obj)
            session.commit()  # explicit commit required

    On exception: rolls back, then re-raises.
    On normal exit: closes session (no auto-commit).
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency for request-scoped DB session.

    Usage:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
