"""Tests for GET /api/notifications endpoint (API-08)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create TestClient with API key auth bypassed."""
    from poseidon.api.auth import verify_api_key
    from poseidon.main import app

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    yield TestClient(app)
    app.dependency_overrides.clear()


def _make_stream_entry(msg_id: str, event_type: str, **extra) -> tuple:
    """Create a fake Redis stream entry (msg_id_bytes, {b"data": json_bytes})."""
    payload = {"event_type": event_type, **extra}
    return (msg_id.encode(), {b"data": json.dumps(payload).encode()})


# --- Tests ---


def test_get_notifications_returns_feed(client):
    """Seeded risk stream returns notifications at GET /api/notifications."""
    entries = [
        _make_stream_entry("1680000000000-0", "drawdown_warning", symbol="BTCUSDT", drawdown=-0.08),
        _make_stream_entry("1680000001000-0", "var_breach", level="99%"),
    ]

    mock_redis = MagicMock()

    def xrevrange_side_effect(key, count=None):
        if key == "poseidon:alerts:risk":
            return entries
        return []

    mock_redis.xrevrange.side_effect = xrevrange_side_effect

    with patch("poseidon.api.notifications.get_redis", return_value=mock_redis):
        resp = client.get("/api/notifications")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["notifications"]) == 2
    # Verify structure
    n = body["notifications"][0]
    assert "id" in n
    assert n["source"] == "risk"
    assert n["event_type"] in ("drawdown_warning", "var_breach")


def test_get_notifications_empty(client):
    """No stream data returns empty list."""
    mock_redis = MagicMock()
    mock_redis.xrevrange.return_value = []

    with patch("poseidon.api.notifications.get_redis", return_value=mock_redis):
        resp = client.get("/api/notifications")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["notifications"] == []


def test_get_notifications_source_filter(client):
    """source=risk only returns risk alerts, not data_quality."""
    risk_entries = [
        _make_stream_entry("1680000000000-0", "drawdown_warning", symbol="BTCUSDT"),
    ]
    dq_entries = [
        _make_stream_entry("1680000002000-0", "gap_detected", symbol="ETHUSDT"),
    ]

    mock_redis = MagicMock()

    def xrevrange_side_effect(key, count=None):
        if key == "poseidon:alerts:risk":
            return risk_entries
        if key == "poseidon:alerts:data_quality":
            return dq_entries
        return []

    mock_redis.xrevrange.side_effect = xrevrange_side_effect

    with patch("poseidon.api.notifications.get_redis", return_value=mock_redis):
        resp = client.get("/api/notifications", params={"source": "risk"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert all(n["source"] == "risk" for n in body["notifications"])
