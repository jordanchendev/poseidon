"""ORM model for non-price timeseries data (FinLab-sourced)."""

from sqlalchemy import Column, Date, Float, Index, PrimaryKeyConstraint, String

from poseidon.models.base import Base


class NonpriceTimeseries(Base):
    """Long-format non-price timeseries data. Ingested periodically from FinLab.

    Categories: institutional, fundamental, margin, trade_structure.
    Each row stores one indicator value for one symbol on one date.
    """

    __tablename__ = "nonprice_timeseries"

    date = Column(Date, nullable=False)
    symbol = Column(String(32), nullable=False)  # TW stock code e.g. "2330"
    category = Column(String(32), nullable=False)  # "institutional", "fundamental", etc.
    indicator = Column(String(64), nullable=False)  # e.g. "foreign", "pe_ratio"
    value = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "date", "symbol", "category", "indicator", name="pk_nonprice_ts"
        ),
        Index("idx_nonprice_ts_sym_cat_date", "symbol", "category", date.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<NonpriceTimeseries(date={self.date}, symbol={self.symbol}, "
            f"category={self.category}, indicator={self.indicator}, "
            f"value={self.value})>"
        )
