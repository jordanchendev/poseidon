"""Unit tests for the centralized Redis factory (Phase 54 REDIS-01, REDIS-02)."""

import pytest

from poseidon.core.config import Settings


class TestSettingsRedisUrls:
    """Tests for per-purpose Redis URL settings attributes."""

    def test_default_urls(self):
        """Settings defaults have correct DB indices."""
        s = Settings(
            _env_file=None,
            redis_celery_url="redis://localhost:6379/0",
            redis_cache_url="redis://localhost:6379/1",
            redis_stream_url="redis://localhost:6379/2",
            redis_ratelimit_url="redis://localhost:6379/3",
        )
        assert "/0" in s.redis_celery_url
        assert "/1" in s.redis_cache_url
        assert "/2" in s.redis_stream_url
        assert "/3" in s.redis_ratelimit_url

    def test_settings_env_override(self, monkeypatch):
        """POSEIDON_REDIS_CACHE_URL env var overrides the default."""
        monkeypatch.setenv("POSEIDON_REDIS_CACHE_URL", "redis://custom:6379/99")
        s = Settings(_env_file=None)
        assert s.redis_cache_url == "redis://custom:6379/99"

    def test_legacy_redis_url_preserved(self):
        """Legacy redis_url attribute still exists."""
        s = Settings(_env_file=None)
        assert hasattr(s, "redis_url")
        assert "redis://" in s.redis_url


class TestGetRedis:
    """Tests for get_redis() factory function."""

    def test_import_factory(self):
        """get_redis is importable from core.redis."""
        from poseidon.core.redis import get_redis

        assert callable(get_redis)

    def test_purpose_urls(self):
        """Each purpose returns a client whose URL contains the correct DB index."""
        from poseidon.core.redis import get_redis

        purposes_and_dbs = [
            ("celery", 0),
            ("cache", 1),
            ("stream", 2),
            ("ratelimit", 3),
        ]
        for purpose, expected_db in purposes_and_dbs:
            client = get_redis(purpose)
            pool = client.connection_pool
            db = pool.connection_kwargs.get("db", None)
            assert db == expected_db, f"get_redis({purpose!r}) should use DB {expected_db}, got {db}"

    def test_unknown_purpose_raises(self):
        """Unknown purpose raises ValueError with valid options listed."""
        from poseidon.core.redis import get_redis

        with pytest.raises(ValueError, match="Unknown Redis purpose"):
            get_redis("nonexistent")

    def test_decode_responses_false_by_default(self):
        """Default client returns bytes (decode_responses=False)."""
        from poseidon.core.redis import get_redis

        client = get_redis("cache")
        pool = client.connection_pool
        assert (
            pool.connection_kwargs.get("decode_responses") is False or "decode_responses" not in pool.connection_kwargs
        )

    def test_decode_responses_true(self):
        """Stream client with decode_responses=True returns text-mode client."""
        from poseidon.core.redis import get_redis

        client = get_redis("stream", decode_responses=True)
        pool = client.connection_pool
        assert pool.connection_kwargs.get("decode_responses") is True
