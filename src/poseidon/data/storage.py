"""Database storage layer for OHLCV, fundamentals, and sentiment data."""

from datetime import date, datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from poseidon.models.fundamentals import Fundamentals
from poseidon.models.ohlcv import OHLCV
from poseidon.models.sentiment import Sentiment


def _upsert_ohlcv_core(session: Session, df: pd.DataFrame, symbol: str, market: str, instrument: str, interval: str) -> int:
    """Core upsert logic WITHOUT commit. Used by DataRepository.

    DataFrame must have columns: time, open, high, low, close, volume.
    The 'time' column must be timezone-aware (UTC).

    Returns number of rows upserted.
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
    return len(rows)


def upsert_ohlcv(session: Session, df: pd.DataFrame, symbol: str, market: str, instrument: str, interval: str) -> int:
    """Insert or update OHLCV rows. Commits automatically (legacy behavior).

    Prefer DataRepository.upsert_ohlcv() which does NOT auto-commit.
    """
    count = _upsert_ohlcv_core(session, df, symbol, market, instrument, interval)
    session.commit()
    return count


def read_ohlcv(
    session: Session,
    symbol: str,
    market: str,
    interval: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Read OHLCV data for a symbol.

    Phase 40 D-21..D-22: when ``interval='1d'`` and ``market`` is in
    ``settings.cagg_1d_markets``, the read is transparently dispatched
    to the ``ohlcv_1d_cagg`` continuous aggregate (Phase 40 plan 40-01
    migration 024). All other combinations fall through to the raw
    ``ohlcv`` hypertable. Response schema is IDENTICAL across both
    branches: a DataFrame indexed by ``time`` with float columns
    ``open, high, low, close, volume``. Callers (DatasetBuilder,
    /api/data/ohlcv, feature engine, backtest engine) need ZERO code
    changes.
    """
    # Phase 40 D-21: transparent CAGG dispatch
    from poseidon.core.config import settings

    if interval == "1d" and market in settings.cagg_1d_markets:
        return _read_ohlcv_from_cagg(session, symbol, market, start, end)

    # Raw hypertable read (legacy path, unchanged behavior)
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
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], name="time")
        return df

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
    df = pd.DataFrame(data)
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index)
    return df


def _read_ohlcv_from_cagg(
    session: Session,
    symbol: str,
    market: str,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    """SELECT 1d candles from ``ohlcv_1d_cagg`` (Phase 40 D-17..D-22).

    The CAGG groups by interval_source, producing one row per source
    interval per day. All sources yield identical OHLCV values, so we
    filter to a single source ('1h') to get exactly one row per day
    with correct volume (no SUM inflation).

    Response schema MUST be identical to the raw read_ohlcv path.
    """
    where_clauses = ["market = :market", "symbol = :symbol", "interval_source = '1h'"]
    params: dict = {"market": market, "symbol": symbol}
    if start is not None:
        where_clauses.append("time_bucket_day >= :start")
        params["start"] = start
    if end is not None:
        where_clauses.append("time_bucket_day <= :end")
        params["end"] = end
    where_sql = " AND ".join(where_clauses)

    stmt = text(
        f"""
        SELECT
            time_bucket_day AS time,
            open, high, low, close, volume
        FROM ohlcv_1d_cagg
        WHERE {where_sql}
        ORDER BY time_bucket_day ASC
        """
    )
    rows = session.execute(stmt, params).fetchall()

    if not rows:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], name="time")
        return df

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
    df = pd.DataFrame(data)
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index)
    return df


def _write_fundamentals_core(session: Session, symbol: str, market: str, report_date: date, data: dict) -> Fundamentals:
    """Core write logic WITHOUT commit. Used by DataRepository.

    Returns the upserted Fundamentals row.
    """
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
    return session.query(Fundamentals).filter_by(symbol=symbol, market=market, date=report_date).first()


def write_fundamentals(session: Session, symbol: str, market: str, report_date: date, data: dict) -> Fundamentals:
    """Write a fundamentals row. Commits automatically (legacy behavior).

    Prefer DataRepository.write_fundamentals() which does NOT auto-commit.
    """
    result = _write_fundamentals_core(session, symbol, market, report_date, data)
    session.commit()
    return result


def read_fundamentals(session: Session, symbol: str, market: str) -> list[Fundamentals]:
    """Read all fundamentals rows for a symbol, ordered by date descending."""
    return (
        session.query(Fundamentals)
        .filter_by(symbol=symbol, market=market)
        .order_by(Fundamentals.date.desc())
        .all()
    )


def _write_sentiment_core(session: Session, symbol: str, market: str, source_type: str, score: float) -> Sentiment:
    """Core write logic WITHOUT commit. Used by DataRepository.

    Returns the new Sentiment row (not yet refreshed from DB).
    """
    row = Sentiment(symbol=symbol, market=market, source_type=source_type, score=score)
    session.add(row)
    return row


def write_sentiment(session: Session, symbol: str, market: str, source_type: str, score: float) -> Sentiment:
    """Write a sentiment score. Commits automatically (legacy behavior).

    Prefer DataRepository.write_sentiment() which does NOT auto-commit.
    """
    row = _write_sentiment_core(session, symbol, market, source_type, score)
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


