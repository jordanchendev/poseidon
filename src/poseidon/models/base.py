from sqlalchemy.orm import DeclarativeBase

from poseidon.core.database import engine  # noqa: F401 -- needed for metadata.create_all


class Base(DeclarativeBase):
    pass
