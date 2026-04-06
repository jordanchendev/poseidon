"""ProtectionLockRecord ORM model.

Stores stateful protection locks that persist across container restarts.
Supports both per-symbol and portfolio-level (symbol=NULL) locks.
"""

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text, text

from poseidon.models.base import Base


class ProtectionLockRecord(Base):
    """A persistent protection lock record."""

    __tablename__ = "protection_locks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    protection_type = Column(String(64), nullable=False, index=True)
    symbol = Column(String(32), nullable=True, index=True)  # NULL for portfolio-level
    market = Column(String(32), nullable=False, index=True)
    reason = Column(Text, nullable=False)
    locked_at = Column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )
    expires_at = Column(DateTime(timezone=True), nullable=True)  # NULL for manual-release
    active = Column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
        index=True,
    )
    released_at = Column(DateTime(timezone=True), nullable=True)
    released_by = Column(String(32), nullable=True)  # "auto" / "manual" / "system"

    __table_args__ = (
        Index(
            "ix_protection_locks_active_lookup",
            "active",
            "protection_type",
            "symbol",
            "market",
        ),
    )
