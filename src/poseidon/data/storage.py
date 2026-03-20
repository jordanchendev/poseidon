"""Database storage layer for OHLCV, fundamentals, and sentiment data."""

from datetime import date, datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from poseidon.models.backfill import BackfillProgress
from poseidon.models.fundamentals import Fundamentals
from poseidon.models.ohlcv import OHLCV
from poseidon.models.sentiment import Sentiment


def upsert_ohlcv(session: Session, df: pd.DataFrame, symbol: str, market: str, instrument: str, interval: str) -> int:
    """Insert or update OHLCV rows. Returns number of rows upserted.

    DataFrame must have columns: time, open, high, low, close, volume.
    The 'time' column must be timezone-aware (UTC).
    """
    if df.empty:
        return 0

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "time": row["time"],
            "symbol": symbol,
            "market": market,
            "instrument": instrument,
            "interval": interval,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        })

    stmt = insert(OHLCV).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="pk_ohlcv",
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "instrument": stmt.excluded.instrument,
        },
    )
    session.execute(stmt)
    session.commit()
    return len(rows)


def read_ohlcv(
    session: Session,
    symbol: str,
    market: str,
    interval: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Read OHLCV data for a symbol. Returns a DataFrame with columns: time, open, high, low, close, volume."""
    query = session.query(OHLCV).filter(
        OHLCV.symbol == symbol,
        OHLCV.market == market,
        OHLCV.interval == interval,
    )
    if start is not None:
        query = query.filter(OHLCV.time >= start)
    if end is not None:
        query = query.filter(OHLCV.time <= end)
    query = query.order_by(OHLCV.time.asc())

    rows = query.all()
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    data = [
        {
            "time": r.time,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": float(r.volume),
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def write_fundamentals(session: Session, symbol: str, market: str, report_date: date, data: dict) -> Fundamentals:
    """Write a fundamentals row. Uses UPSERT on (symbol, market, date)."""
    stmt = insert(Fundamentals).values(
        symbol=symbol,
        market=market,
        date=report_date,
        data=data,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_fundamentals_symbol_market_date",
        set_={"data": stmt.excluded.data},
    )
    session.execute(stmt)
    session.commit()
    return session.query(Fundamentals).filter_by(symbol=symbol, market=market, date=report_date).first()


def read_fundamentals(session: Session, symbol: str, market: str) -> list[Fundamentals]:
    """Read all fundamentals rows for a symbol, ordered by date descending."""
    return (
        session.query(Fundamentals)
        .filter_by(symbol=symbol, market=market)
        .order_by(Fundamentals.date.desc())
        .all()
    )


def write_sentiment(session: Session, symbol: str, market: str, source_type: str, score: float) -> Sentiment:
    """Write a sentiment score. Always inserts a new row (no upsert)."""
    row = Sentiment(symbol=symbol, market=market, source_type=source_type, score=score)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def read_sentiment(
    session: Session, symbol: str, market: str | None = None, limit: int = 100
) -> list[Sentiment]:
    """Read sentiment scores for a symbol, most recent first."""
    query = session.query(Sentiment).filter(Sentiment.symbol == symbol)
    if market is not None:
        query = query.filter(Sentiment.market == market)
    return query.order_by(Sentiment.created_at.desc()).limit(limit).all()


def get_or_create_backfill_progress(
    session: Session, symbol: str, market: str, interval: str, target_start_date: datetime
) -> BackfillProgress:
    """Get existing backfill progress or create a new pending row."""
    row = (
        session.query(BackfillProgress)
        .filter_by(symbol=symbol, market=market, interval=interval)
        .first()
    )
    if row is None:
        row = BackfillProgress(
            symbol=symbol,
            market=market,
            interval=interval,
            target_start_date=target_start_date,
            status="pending",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def update_backfill_progress(
    session: Session, progress: BackfillProgress, status: str, last_fetched_date: datetime | None = None, error_message: str | None = None
) -> None:
    """Update backfill progress status."""
    progress.status = status
    if last_fetched_date is not None:
        progress.last_fetched_date = last_fetched_date
    if error_message is not None:
        progress.error_message = error_message
    session.commit()
