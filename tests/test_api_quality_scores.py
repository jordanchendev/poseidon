"""Integration tests for data quality scores API endpoint (RAPI-05)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_score_row(symbol="BTCUSDT", interval="1d", score=0.85, **kwargs):
    """Create a mock QualityScore row."""
    row = MagicMock()
    row.time = kwargs.get("time", datetime(2026, 3, 28, tzinfo=timezone.utc))
    row.symbol = symbol
    row.interval = interval
    row.score = Decimal(str(score))
    row.completeness = Decimal(str(kwargs.get("completeness", 0.90)))
    row.consistency = Decimal(str(kwargs.get("consistency", 0.80)))
    row.anomaly_free = Decimal(str(kwargs.get("anomaly_free", 0.85)))
    row.timeliness = Decimal(str(kwargs.get("timeliness", 0.88)))
    return row


@pytest.fixture()
def client():
    """Create TestClient with API key auth bypassed."""
    from poseidon.main import app
    from poseidon.api.auth import verify_api_key

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def mock_db():
    """Create mock DB session with quality score data."""
    db = MagicMock()
    return db


def _setup_db_query(mock_db, rows):
    """Wire up the chained query mock to return given rows."""
    query = MagicMock()
    query.order_by.return_value = query
    query.filter.return_value = query
    query.limit.return_value = query
    query.all.return_value = rows
    mock_db.query.return_value = query
    return query


# --- RAPI-05: GET /api/data-quality/scores ---


def test_quality_scores_no_filter(client, mock_db):
    """GET /api/data-quality/scores returns quality scores list."""
    rows = [
        _make_score_row("BTCUSDT", "1d"),
        _make_score_row("ETHUSDT", "1h"),
    ]
    _setup_db_query(mock_db, rows)

    from poseidon.models.base import get_db
    from poseidon.main import app

    app.dependency_overrides[get_db] = lambda: mock_db

    resp = client.get("/api/data-quality/scores")
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data
    assert len(data["scores"]) == 2
    assert data["scores"][0]["symbol"] == "BTCUSDT"
    assert data["scores"][1]["symbol"] == "ETHUSDT"

    app.dependency_overrides.pop(get_db, None)


def test_quality_scores_filter_symbol(client, mock_db):
    """GET /api/data-quality/scores?symbol=AAPL returns filtered scores."""
    rows = [_make_score_row("AAPL", "1d")]
    query = _setup_db_query(mock_db, rows)

    from poseidon.models.base import get_db
    from poseidon.main import app

    app.dependency_overrides[get_db] = lambda: mock_db

    resp = client.get("/api/data-quality/scores?symbol=AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["scores"]) == 1
    assert data["scores"][0]["symbol"] == "AAPL"

    app.dependency_overrides.pop(get_db, None)


def test_quality_scores_filter_interval(client, mock_db):
    """GET /api/data-quality/scores?interval=1d returns filtered scores."""
    rows = [_make_score_row("BTCUSDT", "1d")]
    query = _setup_db_query(mock_db, rows)

    from poseidon.models.base import get_db
    from poseidon.main import app

    app.dependency_overrides[get_db] = lambda: mock_db

    resp = client.get("/api/data-quality/scores?interval=1d")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["scores"]) == 1
    assert data["scores"][0]["interval"] == "1d"

    app.dependency_overrides.pop(get_db, None)


def test_quality_scores_limit(client, mock_db):
    """GET /api/data-quality/scores?limit=5 returns at most 5 scores."""
    rows = [_make_score_row(f"SYM{i}", "1d") for i in range(5)]
    query = _setup_db_query(mock_db, rows)

    from poseidon.models.base import get_db
    from poseidon.main import app

    app.dependency_overrides[get_db] = lambda: mock_db

    resp = client.get("/api/data-quality/scores?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["scores"]) == 5

    app.dependency_overrides.pop(get_db, None)


def test_providers_still_works(client):
    """GET /api/data-quality/providers returns 200 (regression check for RAPI-06)."""
    mock_redis = MagicMock()
    mock_health = {"state": "closed", "failure_count": 0}
    mock_circuit = MagicMock()
    mock_circuit.get_health.return_value = mock_health

    mock_limiter = MagicMock()
    mock_limiter.get_usage.return_value = 10

    with (
        patch("poseidon.api.data_quality._get_redis_client", return_value=mock_redis),
        patch("poseidon.api.data_quality.CircuitBreaker", return_value=mock_circuit),
        patch("poseidon.api.data_quality.DistributedRateLimiter", return_value=mock_limiter),
    ):
        resp = client.get("/api/data-quality/providers")

    assert resp.status_code == 200
    assert "providers" in resp.json()
