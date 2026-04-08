import pytest
import fakeredis


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Override settings for tests."""
    monkeypatch.setenv("POSEIDON_API_KEY", "test-key")
    monkeypatch.setenv("POSEIDON_DATABASE_URL", "postgresql://test:test@localhost:5432/test_poseidon")
    monkeypatch.setenv("POSEIDON_REDIS_URL", "redis://localhost:6379/1")


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
              open_=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0):
        from sqlalchemy import text
        for ts in timestamps:
            db_session.execute(
                text(
                    "INSERT INTO ohlcv (symbol, market, interval, ts, open, high, low, close, volume) "
                    "VALUES (:symbol, :market, :interval, :ts, :o, :h, :l, :c, :v) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "symbol": symbol, "market": market, "interval": interval, "ts": ts,
                    "o": open_, "h": high, "l": low, "c": close, "v": volume,
                },
            )
        db_session.commit()
    return _seed
