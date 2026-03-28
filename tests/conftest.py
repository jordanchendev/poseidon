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
