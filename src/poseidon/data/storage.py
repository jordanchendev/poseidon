"""Database storage layer for OHLCV, fundamentals, and sentiment data."""

from datetime import date, datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

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

    The CAGG can carry multiple rows per (symbol, time_bucket_day) if
    a tuple has more than one sub-daily source interval. We collapse
    per-day with first/max/min/last/sum so the response stays one row
    per day, matching the raw path's contract.

    Response schema MUST be identical to the raw read_ohlcv path.
    """
    where_clauses = ["market = :market", "symbol = :symbol"]
    params: dict = {"market": market, "symbol": symbol}
    if start is not None:
        where_clauses.append("time_bucket_day >= :start")
        params["start"] = start
    if end is not None:
        where_clauses.append("time_bucket_day <= :end")
        params["end"] = end
    where_sql = " AND ".join(where_clauses)

    # GROUP BY collapse: handles the multi-source case (e.g. both
    # 1h and 4h source rows for the same day) by re-running the
    # CAGG aggregations across rows. For the v8.0 default markets
    # (crypto_perp, crypto_spot) this is a no-op because only one
    # sub-daily source exists per tuple, but it future-proofs the
    # contract.
    stmt = text(
        f"""
        SELECT
            time_bucket_day AS time,
            MIN(open)  AS open,
            MAX(high)  AS high,
            MIN(low)   AS low,
            MAX(close) AS close,
            SUM(volume) AS volume
        FROM ohlcv_1d_cagg
        WHERE {where_sql}
        GROUP BY time_bucket_day
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


