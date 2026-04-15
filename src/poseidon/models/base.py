from sqlalchemy.orm import DeclarativeBase

from poseidon.core.database import engine  # noqa: F401 -- needed for metadata.create_all
from poseidon.core.database import SessionLocal, get_db  # noqa: F401 -- backward compat re-exports


class Base(DeclarativeBase):
    pass
