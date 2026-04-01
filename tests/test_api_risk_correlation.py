"""Tests for GET /api/risk/correlation endpoint (API-07)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create TestClient with API key auth bypassed."""
    from poseidon.main import app
    from poseidon.api.auth import verify_api_key

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    yield TestClient(app)
    app.dependency_overrides.clear()


def _make_covariance_cache(symbols: list[str], cov_matrix: np.ndarray) -> bytes:
    """Serialize a covariance matrix the same way covariance.py does."""
    payload = {
        "symbols": symbols,
        "matrix": cov_matrix.tolist(),
        "as_of": "2026-03-30T00:00:00",
        "computed_at": "2026-03-30T01:00:00",
    }
    return msgpack.packb(payload, use_bin_type=True)


# --- Tests ---


def test_get_correlation_returns_matrix(client):
    """Seeded covariance cache returns 200 with correlation matrix."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    # Covariance matrix: var(BTC)=0.04, var(ETH)=0.09, cov=0.012
    cov = np.array([[0.04, 0.012], [0.012, 0.09]])
    packed = _make_covariance_cache(symbols, cov)

    mock_redis = MagicMock()
    mock_redis.get.return_value = packed

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/correlation")

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert len(body["matrix"]) == 2
    assert len(body["matrix"][0]) == 2
    # Diagonal should be exactly 1.0
    assert body["matrix"][0][0] == pytest.approx(1.0)
    assert body["matrix"][1][1] == pytest.approx(1.0)
    # Off-diagonal: 0.012 / (0.2 * 0.3) = 0.2
    assert body["matrix"][0][1] == pytest.approx(0.2)
    assert body["matrix"][1][0] == pytest.approx(0.2)
    assert body["as_of"] == "2026-03-30T00:00:00"


def test_get_correlation_no_cache_404(client):
    """Empty cache returns 404."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/correlation")

    assert resp.status_code == 404
    assert "No covariance data cached" in resp.json()["detail"]


def test_get_correlation_zero_variance(client):
    """Zero-variance asset doesn't cause NaN in correlation matrix."""
    symbols = ["BTCUSDT", "STABLECOIN"]
    # STABLECOIN has zero variance
    cov = np.array([[0.04, 0.0], [0.0, 0.0]])
    packed = _make_covariance_cache(symbols, cov)

    mock_redis = MagicMock()
    mock_redis.get.return_value = packed

    with patch("poseidon.api.risk_metrics._get_redis_client", return_value=mock_redis):
        resp = client.get("/api/risk/correlation")

    assert resp.status_code == 200
    body = resp.json()
    matrix = body["matrix"]
    # No NaN values
    for row in matrix:
        for val in row:
            assert val == val  # NaN != NaN
    # Diagonal should be 1.0
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)
