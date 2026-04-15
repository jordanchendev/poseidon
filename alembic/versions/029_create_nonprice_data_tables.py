"""Create macro_index and nonprice_timeseries tables (Phase 57 FEAT-02).

Revision ID: 029
Revises: 028
Create Date: 2026-04-15

Two new tables for ingest-first non-price data pattern:
- macro_index: daily macro indicators (VIX, DXY, TNX, TWDUSD) from yfinance
- nonprice_timeseries: long-format FinLab data (institutional, fundamental,
  margin, trade_structure)
"""

import sqlalchemy as sa
from alembic import op

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade():
    # macro_index table
    op.create_table(
        "macro_index",
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("indicator", sa.String(32), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("date", "indicator", name="pk_macro_index"),
    )
    op.create_index(
        "idx_macro_index_indicator_date",
        "macro_index",
        ["indicator", sa.text("date DESC")],
    )

    # nonprice_timeseries table
    op.create_table(
        "nonprice_timeseries",
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("indicator", sa.String(64), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint(
            "date", "symbol", "category", "indicator", name="pk_nonprice_ts"
        ),
    )
    op.create_index(
        "idx_nonprice_ts_sym_cat_date",
        "nonprice_timeseries",
        ["symbol", "category", sa.text("date DESC")],
    )


def downgrade():
    op.drop_table("nonprice_timeseries")
    op.drop_table("macro_index")
