"""Tests for DrawdownMonitor with HWM tracking and Redis Streams alerts.

Covers:
- High-water mark tracking and persistence in Redis
- Three-tier progressive drawdown alerts (5%/10%/20%)
- Only highest-severity alert per update (no spam)
- Alert payload correctness
- Alert publishing to Redis Streams poseidon:alerts:risk
- VaR limit breach alert publishing (D-12)
- Consumer group creation
- Edge cases (equity=0, first call)
"""

from __future__ import annotations

import json

import fakeredis
import pytest

from poseidon.risk.var.drawdown import DrawdownMonitor


@pytest.fixture
def redis_client():
    """In-memory Redis with decode_responses=True for string comparisons."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    yield client
    client.flushall()


@pytest.fixture
def monitor(redis_client):
    """DrawdownMonitor with default thresholds."""
    return DrawdownMonitor(redis_client=redis_client)


class TestHWMTracking:
    """High-water mark tracking tests."""

    def test_first_update_sets_hwm(self, monitor, redis_client):
        """First call with equity=100000 sets HWM, no alert."""
        result = monitor.update(current_equity=100_000)
        assert result is None
        hwm = redis_client.get(DrawdownMonitor.HWM_KEY)
        assert hwm is not None
        assert float(hwm) == 100_000

    def test_higher_equity_updates_hwm(self, monitor, redis_client):
        """Equity above HWM updates HWM."""
        monitor.update(current_equity=100_000)
        monitor.update(current_equity=110_000)
        hwm = float(redis_client.get(DrawdownMonitor.HWM_KEY))
        assert hwm == 110_000

    def test_lower_equity_does_not_update_hwm(self, monitor, redis_client):
        """Equity below HWM does not update HWM."""
        monitor.update(current_equity=100_000)
        monitor.update(current_equity=95_000)
        hwm = float(redis_client.get(DrawdownMonitor.HWM_KEY))
        assert hwm == 100_000

    def test_hwm_persists_in_redis(self, redis_client):
        """HWM value persists at Redis key poseidon:risk:hwm:portfolio."""
        mon1 = DrawdownMonitor(redis_client=redis_client)
        mon1.update(current_equity=100_000)

        # Create new monitor instance — should read persisted HWM
        mon2 = DrawdownMonitor(redis_client=redis_client)
        result = mon2.update(current_equity=95_500)
        # 4.5% drawdown — below 5% threshold, no alert
        assert result is None


class TestDrawdownAlerts:
    """Progressive threshold alert tests."""

    def test_5pct_drawdown_fires_warning(self, monitor, redis_client):
        """5% drawdown from HWM fires WARNING alert."""
        monitor.update(current_equity=110_000)
        result = monitor.update(current_equity=104_500)  # exactly 5%
        assert result is not None
        assert result["level"] == "WARNING"
        assert result["event_type"] == "drawdown_breach"

    def test_10pct_drawdown_fires_alert_not_warning(self, monitor, redis_client):
        """10% drawdown fires ALERT (highest matching), not WARNING."""
        monitor.update(current_equity=110_000)
        result = monitor.update(current_equity=99_000)  # exactly 10%
        assert result is not None
        assert result["level"] == "ALERT"
        assert result["event_type"] == "drawdown_breach"

    def test_20pct_drawdown_fires_critical(self, monitor, redis_client):
        """20% drawdown fires CRITICAL."""
        monitor.update(current_equity=110_000)
        result = monitor.update(current_equity=88_000)  # exactly 20%
        assert result is not None
        assert result["level"] == "CRITICAL"

    def test_below_threshold_no_alert(self, monitor, redis_client):
        """Drawdown below 5% produces no alert."""
        monitor.update(current_equity=100_000)
        result = monitor.update(current_equity=96_000)  # 4% drawdown
        assert result is None

    def test_only_highest_severity_published(self, monitor, redis_client):
        """25% drawdown triggers only CRITICAL, not WARNING+ALERT+CRITICAL."""
        monitor.update(current_equity=100_000)
        result = monitor.update(current_equity=75_000)  # 25% drawdown
        assert result["level"] == "CRITICAL"

        # Check only 1 alert in the stream (not 3)
        messages = redis_client.xrange(DrawdownMonitor.STREAM_KEY)
        assert len(messages) == 1

    def test_alert_payload_fields(self, monitor, redis_client):
        """Alert contains all required fields."""
        monitor.update(current_equity=100_000)
        result = monitor.update(current_equity=95_000)  # 5% drawdown
        assert result is not None

        required_keys = {
            "event_type",
            "level",
            "drawdown_pct",
            "threshold_pct",
            "current_equity",
            "hwm",
            "timestamp",
        }
        assert required_keys.issubset(result.keys())
        assert result["event_type"] == "drawdown_breach"
        assert result["drawdown_pct"] == pytest.approx(0.05, abs=1e-6)
        assert result["threshold_pct"] == 0.05
        assert result["current_equity"] == 95_000
        assert result["hwm"] == 100_000


class TestRedisStreamPublishing:
    """Alert publishing to Redis Streams tests."""

    def test_alert_published_to_stream(self, monitor, redis_client):
        """Alert is published to poseidon:alerts:risk via XADD."""
        monitor.update(current_equity=100_000)
        monitor.update(current_equity=90_000)  # 10% drawdown

        messages = redis_client.xrange(DrawdownMonitor.STREAM_KEY)
        assert len(messages) == 1

        _msg_id, data = messages[0]
        payload = json.loads(data["data"])
        assert payload["event_type"] == "drawdown_breach"
        assert payload["level"] == "ALERT"

    def test_consumer_group_created(self, redis_client):
        """Consumer group 'default' is created on stream."""
        DrawdownMonitor(redis_client=redis_client)
        groups = redis_client.xinfo_groups(DrawdownMonitor.STREAM_KEY)
        group_names = [g["name"] for g in groups]
        assert "default" in group_names

    def test_no_alert_no_stream_message(self, monitor, redis_client):
        """No alert means no message in stream."""
        monitor.update(current_equity=100_000)  # first call, sets HWM
        messages = redis_client.xrange(DrawdownMonitor.STREAM_KEY)
        assert len(messages) == 0


class TestVaRBreachAlert:
    """VaR limit breach alert tests (D-12)."""

    def test_var_breach_alert_published(self, monitor, redis_client):
        """publish_var_breach_alert() publishes to same stream."""
        result = monitor.publish_var_breach_alert(current_var=0.12, limit=0.10, method="parametric")
        assert result["event_type"] == "var_limit_breach"
        assert result["level"] == "CRITICAL"
        assert result["current_var"] == pytest.approx(0.12, abs=1e-6)
        assert result["var_limit"] == 0.10
        assert result["var_method"] == "parametric"

        messages = redis_client.xrange(DrawdownMonitor.STREAM_KEY)
        assert len(messages) == 1
        _msg_id, data = messages[0]
        payload = json.loads(data["data"])
        assert payload["event_type"] == "var_limit_breach"

    def test_var_breach_different_from_drawdown(self, monitor, redis_client):
        """VaR breach and drawdown alerts have different event_type."""
        monitor.update(current_equity=100_000)
        monitor.update(current_equity=90_000)  # drawdown alert
        monitor.publish_var_breach_alert(current_var=0.15, limit=0.10, method="historical")

        messages = redis_client.xrange(DrawdownMonitor.STREAM_KEY)
        assert len(messages) == 2
        payloads = [json.loads(m[1]["data"]) for m in messages]
        event_types = {p["event_type"] for p in payloads}
        assert event_types == {"drawdown_breach", "var_limit_breach"}


class TestEdgeCases:
    """Edge case handling."""

    def test_equity_zero(self, monitor, redis_client):
        """Equity=0 handled gracefully."""
        monitor.update(current_equity=100_000)
        result = monitor.update(current_equity=0)
        # 100% drawdown — should fire CRITICAL
        assert result is not None
        assert result["level"] == "CRITICAL"
        assert result["drawdown_pct"] == pytest.approx(1.0, abs=1e-6)

    def test_hwm_zero_no_division_error(self, redis_client):
        """If HWM is 0 (equity starts at 0), no division error."""
        mon = DrawdownMonitor(redis_client=redis_client)
        result = mon.update(current_equity=0)
        # HWM=0, equity=0 → hwm<=0 guard → None
        assert result is None

    def test_custom_thresholds(self, redis_client):
        """Custom thresholds work correctly."""
        mon = DrawdownMonitor(
            redis_client=redis_client,
            warning_pct=0.03,
            alert_pct=0.07,
            critical_pct=0.15,
        )
        mon.update(current_equity=100_000)
        result = mon.update(current_equity=97_000)  # 3% drawdown
        assert result is not None
        assert result["level"] == "WARNING"
        assert result["threshold_pct"] == 0.03
