import pytest
import fakeredis


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Override settings for tests."""
    monkeypatch.setenv("POSEIDON_API_KEY", "test-key")
    monkeypatch.setenv("POSEIDON_DATABASE_URL", "postgresql://test:test@localhost:5432/test_poseidon")
    monkeypatch.setenv("POSEIDON_REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("POSEIDON_REDIS_CELERY_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("POSEIDON_REDIS_CACHE_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("POSEIDON_REDIS_STREAM_URL", "redis://localhost:6379/2")
    monkeypatch.setenv("POSEIDON_REDIS_RATELIMIT_URL", "redis://localhost:6379/3")


@pytest.fixture
def fake_redis():
    """In-memory Redis client for testing (with Lua support)."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=False)
    yield client
    client.flushall()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "phase38: Phase 38 data-foundation tests")


@pytest.fixture
def ingest_state_seed():
    """Helper to insert IngestState rows for cursor testing.

    Usage:
        def test_x(db_session, ingest_state_seed):
            ingest_state_seed(db_session, symbol="BTCUSDT", market="crypto_perp",
                              interval="4h", last_successful_ts=ts)
    """
    def _seed(db_session, *, symbol, market, interval, last_successful_ts=None,
              last_attempt_ts=None, last_error=None, first_backfill_done=False):
        from poseidon.models.ingest_state import IngestState
        row = IngestState(
            symbol=symbol,
            market=market,
            interval=interval,
            last_successful_ts=last_successful_ts,
            last_attempt_ts=last_attempt_ts,
            last_error=last_error,
            first_backfill_done=first_backfill_done,
        )
        db_session.add(row)
        db_session.commit()
        return row
    return _seed


@pytest.fixture
def hypertable_seed_ohlcv():
    """Helper to insert OHLCV rows into the hypertable for bootstrap testing.

    Usage:
        def test_x(db_session, hypertable_seed_ohlcv):
            hypertable_seed_ohlcv(db_session, symbol="BTCUSDT", market="crypto_perp",
                                  interval="4h", timestamps=[ts1, ts2, ts3])
    """
    def _seed(db_session, *, symbol, market, interval, timestamps,
              instrument="spot",
              open_=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0):
        from sqlalchemy import text
        for ts in timestamps:
            db_session.execute(
                text(
                    "INSERT INTO ohlcv (time, symbol, market, instrument, interval, open, high, low, close, volume) "
                    "VALUES (:ts, :symbol, :market, :instrument, :interval, :o, :h, :l, :c, :v) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "ts": ts, "symbol": symbol, "market": market,
                    "instrument": instrument, "interval": interval,
                    "o": open_, "h": high, "l": low, "c": close, "v": volume,
                },
            )
        db_session.commit()
    return _seed


@pytest.fixture
def db_session():
    """Real SQLAlchemy session bound to the live poseidon DB.

    Tests using this fixture run inside the qlib-research container on
    stormtrooper against the real TimescaleDB. The autouse ``test_settings``
    fixture overrides ``POSEIDON_DATABASE_URL`` for the general test suite, so
    we bypass the cached ``SessionLocal`` and bind directly to the real DSN
    provided by the container environment (or the default poseidon compose
    DSN). Each test is responsible for cleaning up its own rows.
    """
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Prefer an explicit real-DB DSN; fall back to the poseidon compose
    # default wired via host.docker.internal.
    real_dsn = (
        os.environ.get("POSEIDON_REAL_DATABASE_URL")
        or "postgresql://poseidon:poseidon@host.docker.internal:5433/poseidon"
    )
    engine = create_engine(real_dsn, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()
