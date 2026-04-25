"""Tests for GET /api/data/ohlcv endpoint (API-01)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api.data import router as data_router

_test_app = FastAPI()
_test_app.include_router(data_router, prefix="/api/data", tags=["data"])


@pytest.fixture()
def client():
    return TestClient(_test_app)


def _sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": [40000, 40100, 40200, 40300, 40400],
            "high": [40500, 40600, 40700, 40800, 40900],
            "low": [39500, 39600, 39700, 39800, 39900],
            "close": [40200, 40300, 40400, 40500, 40600],
            "volume": [1000, 1010, 1020, 1030, 1040],
        },
        index=index,
    )


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_get_ohlcv_returns_data(mock_from_settings, client):
    """GET /api/data/ohlcv returns remote OHLCV data."""
    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = _sample_ohlcv()
    mock_from_settings.return_value = mock_repo

    resp = client.get("/api/data/ohlcv", params={"symbol": "BTCUSDT", "market": "crypto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["market"] == "crypto"
    assert body["interval"] == "1d"
    assert body["count"] == 5
    assert len(body["data"]) == 5
    point = body["data"][0]
    assert "time" in point
    assert "open" in point
    assert "close" in point
    assert "volume" in point


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_get_ohlcv_empty(mock_from_settings, client):
    """Non-existent symbol returns 200 with empty data and count=0."""
    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    mock_from_settings.return_value = mock_repo

    resp = client.get("/api/data/ohlcv", params={"symbol": "NONEXIST", "market": "crypto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["data"] == []


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_get_ohlcv_date_range(mock_from_settings, client):
    """start/end filters are forwarded to the remote repository."""
    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = _sample_ohlcv().iloc[1:4]
    mock_from_settings.return_value = mock_repo

    resp = client.get(
        "/api/data/ohlcv",
        params={
            "symbol": "BTCUSDT",
            "market": "crypto",
            "start": "2026-01-02T00:00:00",
            "end": "2026-01-04T00:00:00",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3

    assert mock_repo.read_ohlcv.call_args.args == ("BTCUSDT", "crypto", "1d")
    start = mock_repo.read_ohlcv.call_args.kwargs["start"]
    end = mock_repo.read_ohlcv.call_args.kwargs["end"]
    assert start.isoformat().startswith("2026-01-02T00:00:00")
    assert end.isoformat().startswith("2026-01-04T00:00:00")


def test_get_ohlcv_requires_params(client):
    """Missing required symbol/market returns 422."""
    resp = client.get("/api/data/ohlcv")
    assert resp.status_code == 422
