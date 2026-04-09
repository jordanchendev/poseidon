"""Tests for the Phase 40 read_ohlcv CAGG dispatch branch (plan 40-04).

Covers the locked Phase 40 decisions:
- D-21: ``if interval == '1d' and market in settings.cagg_1d_markets:`` guard
- D-22: DatasetBuilder requires zero changes (transparency)
- D-20: ``settings.cagg_1d_markets`` controls the dispatch allowlist

The unit suite runs against an in-memory SQLite harness that mirrors the
Postgres ``ohlcv`` table via the ORM and shims the ``ohlcv_1d_cagg``
continuous aggregate as a plain SQLite table (same column shape).

Tests are grouped into:
  - Tests 1-3: dispatch routing (CAGG vs raw based on interval/market)
  - Test 4: schema parity between CAGG and raw paths
  - Test 5: date range filtering on the CAGG branch
  - Test 6: empty cagg_1d_markets rollback safety
  - Test 7: empty CAGG result returns correct empty DataFrame shape
  - Test 8: parity — CAGG output matches pandas resample('1D') reference
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility shims for Postgres-only types ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.models.base import Base  # noqa: E402

# --- Test DB wiring ---

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def _create_cagg_table(engine):
    """Create an SQLite stand-in for the ohlcv_1d_cagg continuous aggregate.

    Mirrors the column shape from migration 024: market, symbol,
    interval_source, time_bucket_day, open, high, low, close, volume.
    """
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ohlcv_1d_cagg (
                    market VARCHAR(32) NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    interval_source VARCHAR(8) NOT NULL,
                    time_bucket_day DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """
            )
        )
        conn.commit()


@pytest.fixture(autouse=True)
def _setup_db():
    """Create all ORM tables + the CAGG stand-in, then tear down after each test."""
    Base.metadata.create_all(bind=_engine)
    _create_cagg_table(_engine)
    yield
    # Drop the CAGG stand-in first (not managed by Base.metadata)
    with _engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ohlcv_1d_cagg"))
        conn.commit()
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# --- Seed helpers ---

def _seed_raw_ohlcv(session, symbol, market, interval, rows):
    """Insert rows into the raw ohlcv table.

    Each row is a dict with keys: time, open, high, low, close, volume.
    """
    from poseidon.models.ohlcv import OHLCV

    for row in rows:
        session.add(
            OHLCV(
                time=row["time"],
                symbol=symbol,
                market=market,
                instrument=f"{market}_instrument",
                interval=interval,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
        )
    session.commit()


def _seed_cagg(session, symbol, market, interval_source, rows):
    """Insert rows into the ohlcv_1d_cagg stand-in table.

    Each row is a dict with keys: time_bucket_day, open, high, low, close, volume.
    """
    for row in rows:
        session.execute(
            text(
                """
                INSERT INTO ohlcv_1d_cagg
                    (market, symbol, interval_source, time_bucket_day, open, high, low, close, volume)
                VALUES
                    (:market, :symbol, :interval_source, :time_bucket_day, :open, :high, :low, :close, :volume)
                """
            ),
            {
                "market": market,
                "symbol": symbol,
                "interval_source": interval_source,
                "time_bucket_day": row["time_bucket_day"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            },
        )
    session.commit()


# =============================================================================
# Test 1: crypto_perp + interval=1d dispatches to CAGG
# =============================================================================


def test_dispatch_crypto_perp_1d_reads_from_cagg(db_session, monkeypatch):
    """read_ohlcv(interval='1d', market='crypto_perp') with cagg_1d_markets=['crypto_perp']
    should return data from ohlcv_1d_cagg, NOT from raw ohlcv."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp", "crypto_spot"])

    day1 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Seed CAGG with distinctive values
    _seed_cagg(db_session, "BTCUSDT", "crypto_perp", "4h", [
        {"time_bucket_day": day1, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
    ])

    # Seed raw ohlcv with DIFFERENT values to detect which path was taken
    _seed_raw_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d", [
        {"time": day1, "open": 999.0, "high": 999.0, "low": 999.0, "close": 999.0, "volume": 999.0},
    ])

    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d")

    # Should get CAGG values (100/110/90/105/1000), not raw values (999)
    assert len(df) == 1
    assert df.iloc[0]["open"] == pytest.approx(100.0)
    assert df.iloc[0]["close"] == pytest.approx(105.0)


# =============================================================================
# Test 2: tw_stock + interval=1d falls through to raw ohlcv
# =============================================================================


def test_dispatch_tw_stock_1d_reads_from_raw(db_session, monkeypatch):
    """read_ohlcv(interval='1d', market='tw_stock') should read from raw ohlcv
    because tw_stock is NOT in cagg_1d_markets."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp", "crypto_spot"])

    day1 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Seed raw ohlcv with distinctive values
    _seed_raw_ohlcv(db_session, "2330", "tw_stock", "1d", [
        {"time": day1, "open": 600.0, "high": 620.0, "low": 590.0, "close": 610.0, "volume": 50000.0},
    ])

    # Seed CAGG with different values (should NOT be read)
    _seed_cagg(db_session, "2330", "tw_stock", "1d", [
        {"time_bucket_day": day1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
    ])

    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db_session, "2330", "tw_stock", "1d")

    # Should get raw values (600/620/590/610/50000), not CAGG values (1)
    assert len(df) == 1
    assert df.iloc[0]["open"] == pytest.approx(600.0)
    assert df.iloc[0]["close"] == pytest.approx(610.0)


# =============================================================================
# Test 3: crypto_perp + interval=4h always reads from raw (not 1d)
# =============================================================================


def test_dispatch_crypto_perp_4h_reads_from_raw(db_session, monkeypatch):
    """read_ohlcv(interval='4h', market='crypto_perp') should read from raw ohlcv
    regardless of cagg_1d_markets — the dispatch only triggers for interval='1d'."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp", "crypto_spot"])

    ts = datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc)

    _seed_raw_ohlcv(db_session, "BTCUSDT", "crypto_perp", "4h", [
        {"time": ts, "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0, "volume": 500.0},
    ])

    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "4h")

    assert len(df) == 1
    assert df.iloc[0]["open"] == pytest.approx(200.0)


# =============================================================================
# Test 4: CAGG path returns DataFrame with identical schema to raw path
# =============================================================================


def test_cagg_path_schema_matches_raw_path(db_session, monkeypatch):
    """The CAGG dispatch path must return a DataFrame with columns
    [open, high, low, close, volume] indexed by 'time' — IDENTICAL
    to the raw ohlcv path."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp"])

    day1 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    _seed_cagg(db_session, "BTCUSDT", "crypto_perp", "4h", [
        {"time_bucket_day": day1, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
    ])

    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "time"
    assert isinstance(df.index, pd.DatetimeIndex)


# =============================================================================
# Test 5: CAGG path applies date range filtering
# =============================================================================


def test_cagg_path_date_range_filtering(db_session, monkeypatch):
    """read_ohlcv(..., interval='1d', start=..., end=...) correctly applies
    the date range to the CAGG query."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp"])

    # Seed 10 days of CAGG data
    days = [
        datetime(2026, 3, d, 0, 0, 0, tzinfo=timezone.utc) for d in range(1, 11)
    ]
    cagg_rows = [
        {
            "time_bucket_day": day,
            "open": 100.0 + i,
            "high": 110.0 + i,
            "low": 90.0 + i,
            "close": 105.0 + i,
            "volume": 1000.0 + i * 100,
        }
        for i, day in enumerate(days)
    ]
    _seed_cagg(db_session, "BTCUSDT", "crypto_perp", "4h", cagg_rows)

    from poseidon.data.storage import read_ohlcv

    # Request a 5-day slice (March 3 to March 7 inclusive)
    start = datetime(2026, 3, 3, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 7, 0, 0, 0, tzinfo=timezone.utc)
    df = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d", start=start, end=end)

    assert len(df) == 5
    # First row should be March 3 (open=102.0)
    assert df.iloc[0]["open"] == pytest.approx(102.0)
    # Last row should be March 7 (open=106.0)
    assert df.iloc[-1]["open"] == pytest.approx(106.0)


# =============================================================================
# Test 6: empty cagg_1d_markets disables dispatch (rollback safety)
# =============================================================================


def test_empty_cagg_1d_markets_disables_dispatch(db_session, monkeypatch):
    """When cagg_1d_markets is empty, the dispatch branch is NEVER taken —
    every call goes to raw ohlcv (rollback safety)."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", [])

    day1 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Seed raw ohlcv (should be read)
    _seed_raw_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d", [
        {"time": day1, "open": 777.0, "high": 780.0, "low": 770.0, "close": 775.0, "volume": 5000.0},
    ])

    # Seed CAGG (should NOT be read when cagg_1d_markets is empty)
    _seed_cagg(db_session, "BTCUSDT", "crypto_perp", "4h", [
        {"time_bucket_day": day1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
    ])

    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d")

    # Should get raw values, not CAGG values
    assert len(df) == 1
    assert df.iloc[0]["open"] == pytest.approx(777.0)


# =============================================================================
# Test 7: empty CAGG result returns correct empty DataFrame shape
# =============================================================================


def test_empty_cagg_returns_correct_empty_shape(db_session, monkeypatch):
    """An empty CAGG query result returns the same empty DataFrame shape
    as the raw path: columns=[open, high, low, close, volume], empty
    DatetimeIndex named 'time'."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp"])

    # No data seeded — CAGG is empty
    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d")

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "time"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 0


# =============================================================================
# Test 8 (parity): CAGG output matches pandas resample('1D') reference
# =============================================================================


def test_cagg_parity_with_pandas_resample(db_session, monkeypatch):
    """Seed 4h bars for 5 days (6 bars per day = 30 total), compute the
    expected 1d values via pandas resample('1D').agg(...), seed the CAGG
    stand-in with the same aggregated values (mimicking what Timescale CAGG
    would produce), and assert read_ohlcv(..., interval='1d') output
    matches the pandas reference exactly."""
    from poseidon.core.config import settings

    monkeypatch.setattr(settings, "cagg_1d_markets", ["crypto_perp"])

    # Generate 5 days of 4h bars (6 bars per day: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
    raw_4h_rows = []
    base_price = 40000.0
    for day_offset in range(5):
        for bar_idx in range(6):
            hour = bar_idx * 4
            ts = datetime(2026, 3, 1 + day_offset, hour, 0, 0, tzinfo=timezone.utc)
            # Deterministic but varied prices per bar
            o = base_price + day_offset * 100 + bar_idx * 10
            h = o + 50
            lo = o - 30
            c = o + 20
            v = 100.0 + day_offset * 10 + bar_idx
            raw_4h_rows.append({
                "time": ts,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": v,
            })

    # Seed raw 4h bars (used only for the pandas reference computation)
    _seed_raw_ohlcv(db_session, "BTCUSDT", "crypto_perp", "4h", raw_4h_rows)

    # Compute the expected 1d values via pandas resample
    raw_df = pd.DataFrame(raw_4h_rows)
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = raw_df.set_index("time")
    expected_1d = raw_df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    # Drop any NaN rows from resample
    expected_1d = expected_1d.dropna()

    # Seed the CAGG stand-in with the same aggregated values
    cagg_rows = []
    for ts, row in expected_1d.iterrows():
        cagg_rows.append({
            "time_bucket_day": ts.to_pydatetime(),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        })
    _seed_cagg(db_session, "BTCUSDT", "crypto_perp", "4h", cagg_rows)

    from poseidon.data.storage import read_ohlcv

    # Read via the CAGG dispatch path
    start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 5, 23, 59, 59, tzinfo=timezone.utc)
    actual_1d = read_ohlcv(db_session, "BTCUSDT", "crypto_perp", "1d", start=start, end=end)

    # Both should have 5 rows
    assert len(actual_1d) == len(expected_1d) == 5

    # Compare values exactly (CAGG aggregations are identical operations)
    for i in range(5):
        assert actual_1d.iloc[i]["open"] == pytest.approx(expected_1d.iloc[i]["open"])
        assert actual_1d.iloc[i]["high"] == pytest.approx(expected_1d.iloc[i]["high"])
        assert actual_1d.iloc[i]["low"] == pytest.approx(expected_1d.iloc[i]["low"])
        assert actual_1d.iloc[i]["close"] == pytest.approx(expected_1d.iloc[i]["close"])
        assert actual_1d.iloc[i]["volume"] == pytest.approx(expected_1d.iloc[i]["volume"])


# =============================================================================
# Test 9: CAGG-03 transparency — DatasetBuilder call site unchanged
# =============================================================================


def test_dataset_builder_call_site_unchanged():
    """CAGG-03 transparency check: DatasetBuilder still calls
    read_ohlcv with the same signature, no source diff required."""
    import pathlib

    src = pathlib.Path("src/poseidon/qlib/dataset_builder.py").read_text()
    assert "read_ohlcv(" in src
    # The exact call site from Phase 37; touching it would mean we
    # broke the CAGG-03 transparency contract.
    assert "self.session, symbol, self.market, self.interval" in src


# =============================================================================
# Test 10: CAGG-02 transparency — /api/data/ohlcv endpoint call site unchanged
# =============================================================================


def test_api_data_ohlcv_endpoint_call_site_unchanged():
    """CAGG-02 transparency check: /api/data/ohlcv endpoint still
    calls read_ohlcv with the same signature, no source diff required."""
    import pathlib

    src = pathlib.Path("src/poseidon/api/data.py").read_text()
    assert "read_ohlcv(db, symbol, market, interval" in src
