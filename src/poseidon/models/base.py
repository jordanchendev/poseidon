from sqlalchemy.orm import DeclarativeBase

from poseidon.core.database import (  # noqa: F401 -- backward compat re-exports
    SessionLocal,
    engine,
    get_db,
)


class Base(DeclarativeBase):
    pass
