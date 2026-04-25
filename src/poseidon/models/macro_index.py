"""ORM model for macro economic index data (yfinance-sourced)."""

from sqlalchemy import Column, Date, Float, Index, PrimaryKeyConstraint, String

from poseidon.models.base import Base


class MacroIndex(Base):
    """Daily macro economic index values. Ingested periodically from yfinance.

    Stores indicators like VIX, DXY, TNX, TWDUSD as individual rows
    in a long-format table (one row per date+indicator pair).
    """

    __tablename__ = "macro_index"

    date = Column(Date, nullable=False)
    indicator = Column(String(32), nullable=False)  # "macro_vix", "macro_dxy", etc.
    value = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("date", "indicator", name="pk_macro_index"),
        Index("idx_macro_index_indicator_date", "indicator", date.desc()),
    )

    def __repr__(self) -> str:
        return f"<MacroIndex(date={self.date}, indicator={self.indicator}, value={self.value})>"
