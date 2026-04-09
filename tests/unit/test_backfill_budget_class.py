"""Phase 39-02 Task 1 — budget_class-aware rate limiting.

Verifies that the distributed rate limiter scopes Redis keys by an explicit
``budget_class`` so that bulk backfill traffic and live ingest traffic do not
share the same sliding window. Live cpu_tasks call sites must continue to use
``live_ingest`` while the new backfill worker path uses ``bulk_backfill``.
"""

from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest

from poseidon.core.config import Settings, settings
from poseidon.data.rate_limiter import DistributedRateLimiter


@pytest.fixture()
def redis_client():
    return fakeredis.FakeStrictRedis()


def test_make_key_includes_budget_class(redis_client):
    rl = DistributedRateLimiter(redis_client)
    live_key = rl._make_key("finmind", 3600, budget_class="live_ingest")
    backfill_key = rl._make_key("finmind", 3600, budget_class="bulk_backfill")
    assert live_key != backfill_key
    assert "live_ingest" in live_key
    assert "bulk_backfill" in backfill_key
    assert live_key == "poseidon:ratelimit:finmind:live_ingest:3600"
    assert backfill_key == "poseidon:ratelimit:finmind:bulk_backfill:3600"


def test_default_budget_class_is_live_ingest(redis_client):
    rl = DistributedRateLimiter(redis_client)
    default_key = rl._make_key("ccxt", 60)
    assert default_key == "poseidon:ratelimit:ccxt:live_ingest:60"


def test_acquire_does_not_increment_other_budget_class(redis_client):
    rl = DistributedRateLimiter(redis_client)
    # Hammer the live_ingest bucket up to limit; backfill bucket must stay clean.
    for _ in range(5):
        assert rl.acquire("finmind", 3600, 5, budget_class="live_ingest") is True
    # Live bucket is now full
    assert rl.acquire("finmind", 3600, 5, budget_class="live_ingest") is False
    # Backfill bucket is independent and still empty
    assert rl.get_usage("finmind", 3600, budget_class="bulk_backfill") == 0
    assert rl.acquire("finmind", 3600, 5, budget_class="bulk_backfill") is True
    # And live bucket count is unaffected by the backfill acquire
    assert rl.get_usage("finmind", 3600, budget_class="live_ingest") == 5


def test_settings_expose_backfill_budgets():
    s = Settings()
    assert s.ratelimit_finmind_backfill_hourly == 100
    assert s.ratelimit_yfinance_backfill_daily == 200
    assert s.ratelimit_ccxt_backfill_per_minute == 300
    # And the live ingest knobs are still present (regression guard)
    assert hasattr(s, "ratelimit_finmind_hourly")
    assert hasattr(s, "ratelimit_yfinance_daily")
    assert hasattr(s, "ratelimit_ccxt_per_minute")


def test_cpu_tasks_pass_live_ingest_budget_class():
    """Live ingest call sites in cpu_tasks must explicitly pass live_ingest."""
    src = Path(__file__).resolve().parents[2] / "src/poseidon/workers/cpu_tasks.py"
    body = src.read_text()
    assert 'budget_class="live_ingest"' in body
    # Both wait_and_acquire sites should have it (legacy + cursor mode)
    assert body.count('budget_class="live_ingest"') >= 2
