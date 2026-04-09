"""Tests for CacheManager - Redis L1 cache with TTL jitter and single-flight locking."""

from unittest.mock import MagicMock

import pandas as pd
import pytest


def _sample_df(rows=5):
    """Create a small test DataFrame mimicking OHLCV data."""
    import numpy as np

    idx = pd.date_range("2026-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.arange(100.0, 100.0 + rows),
            "high": np.arange(101.0, 101.0 + rows),
            "low": np.arange(99.0, 99.0 + rows),
            "close": np.arange(100.5, 100.5 + rows),
            "volume": np.arange(1000.0, 1000.0 + rows),
        },
        index=idx,
    )


class TestCacheKeyFormat:
    def test_cache_key_format(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        key = cm._cache_key("2330", "1d", "2026-01-01", "2026-01-31")
        assert key == "poseidon:cache:2330:1d:2026-01-01:2026-01-31"

    def test_cache_key_prefix(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        key = cm._cache_key("BTCUSDT", "5m", "2026-01-01", "2026-01-02")
        assert key.startswith("poseidon:cache:")


class TestCacheGetSet:
    def test_cache_set_get_roundtrip(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df()
        cm.set("2330", "1d", "2026-01-01", "2026-01-05", df)
        result = cm.get("2330", "1d", "2026-01-01", "2026-01-05")
        assert result is not None
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)
        pd.testing.assert_frame_equal(result, df, check_names=False, check_freq=False)

    def test_cache_miss_returns_none(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        result = cm.get("NONEXIST", "1d", "2026-01-01", "2026-01-31")
        assert result is None


class TestCacheTTL:
    def test_cache_ttl_1m(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df(3)
        cm.set("BTC", "1m", "2026-01-01", "2026-01-01", df)
        key = cm._cache_key("BTC", "1m", "2026-01-01", "2026-01-01")
        ttl = fake_redis.ttl(key)
        assert 60 <= ttl <= 180, f"1m TTL should be 60-180, got {ttl}"

    def test_cache_ttl_5m(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df(3)
        cm.set("BTC", "5m", "2026-01-01", "2026-01-01", df)
        key = cm._cache_key("BTC", "5m", "2026-01-01", "2026-01-01")
        ttl = fake_redis.ttl(key)
        assert 300 <= ttl <= 420, f"5m TTL should be 300-420, got {ttl}"

    def test_cache_ttl_1d(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df(3)
        cm.set("BTC", "1d", "2026-01-01", "2026-01-01", df)
        key = cm._cache_key("BTC", "1d", "2026-01-01", "2026-01-01")
        ttl = fake_redis.ttl(key)
        assert 3600 <= ttl <= 3720, f"1d TTL should be 3600-3720, got {ttl}"

    def test_cache_ttl_jitter_varies(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        values = {cm._ttl_with_jitter("1d") for _ in range(100)}
        assert len(values) >= 2, "Jitter should produce varying TTL values"


class TestFallback:
    def test_fallback_cache_hit(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df()
        cm.set("2330", "1d", "2026-01-01", "2026-01-05", df)
        db_fn = MagicMock()
        api_fn = MagicMock()
        result = cm.get_or_fetch("2330", "1d", "2026-01-01", "2026-01-05", db_fn, api_fn)
        assert result is not None
        db_fn.assert_not_called()
        api_fn.assert_not_called()

    def test_fallback_cache_miss_db_hit(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df()
        db_fn = MagicMock(return_value=df)
        api_fn = MagicMock()
        result = cm.get_or_fetch("2330", "1d", "2026-01-01", "2026-01-05", db_fn, api_fn)
        assert result is not None
        db_fn.assert_called_once()
        api_fn.assert_not_called()
        # Verify it was cached
        cached = cm.get("2330", "1d", "2026-01-01", "2026-01-05")
        assert cached is not None

    def test_fallback_cache_miss_db_miss_api_hit(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df()
        empty_df = pd.DataFrame()
        db_fn = MagicMock(return_value=empty_df)
        api_fn = MagicMock(return_value=df)
        result = cm.get_or_fetch("2330", "1d", "2026-01-01", "2026-01-05", db_fn, api_fn)
        assert result is not None
        db_fn.assert_called_once()
        api_fn.assert_called_once()
        cached = cm.get("2330", "1d", "2026-01-01", "2026-01-05")
        assert cached is not None


class TestSingleFlight:
    def test_single_flight_lock_acquired(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        lock_key = f"{cm.LOCK_PREFIX}:2330:1d:2026-01-01:2026-01-05"
        acquired = fake_redis.set(lock_key, "1", nx=True, ex=30)
        assert acquired is True
        assert fake_redis.exists(lock_key)


class TestSerialization:
    def test_serialize_deserialize(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cm = CacheManager(fake_redis)
        df = _sample_df(10)
        raw = cm._serialize(df)
        assert isinstance(raw, bytes)
        result = cm._deserialize(raw)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)
        # Check float values match
        pd.testing.assert_frame_equal(result, df, check_names=False, check_freq=False)


class TestCacheRoundTripTimeColumn:
    """Verify cache round-trip restores the canonical 'time' column."""

    def test_cache_round_trip_restores_time_column(self, fake_redis):
        from poseidon.data.cache import CacheManager

        cache = CacheManager(fake_redis)
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2026-04-08T05:30:00Z", "2026-04-09T05:30:00Z"], utc=True
                ),
                "open": [100.0, 110.0],
                "high": [105.0, 112.0],
                "low": [99.0, 109.0],
                "close": [104.0, 111.0],
                "volume": [1000, 2000],
            }
        )

        cache.set("2330", "1d", "2026-04-01", "2026-04-09", df)
        restored = cache.get("2330", "1d", "2026-04-01", "2026-04-09")

        assert restored is not None
        assert "time" in restored.columns
        assert restored["time"].tolist() == df["time"].tolist()
        assert restored["close"].tolist() == [104.0, 111.0]
