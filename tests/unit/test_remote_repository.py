"""Unit tests for RemoteDataRepository HTTP client.

All tests mock httpx.Client.request to avoid real network calls.
Tests verify correct HTTP method/path/params, response deserialization,
circuit breaker integration, and retry behavior.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from poseidon.core.circuit_breaker import CircuitBreakerOpenError
from poseidon.data.remote_repository import RemoteDataRepository


@pytest.fixture
def repo():
    """Create a RemoteDataRepository with mocked internals."""
    return RemoteDataRepository(
        base_url="http://test:8001",
        api_key="test-key",
        timeout=5.0,
        cb_threshold=3,
        cb_recovery_timeout=10.0,
    )


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    return resp


class TestReadOHLCV:
    def test_read_ohlcv_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"time": "2024-01-01T00:00:00", "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
                {"time": "2024-01-02T00:00:00", "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 1200.0},
            ],
            "symbol": "BTCUSDT", "market": "crypto_perp", "interval": "1d", "count": 2,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_ohlcv("BTCUSDT", "crypto_perp", "1d")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_read_ohlcv_empty_data(self, repo):
        mock_resp = _mock_response({
            "data": [], "symbol": "BTCUSDT", "market": "crypto_perp", "interval": "1d", "count": 0,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_ohlcv("BTCUSDT", "crypto_perp", "1d")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_read_ohlcv_passes_start_end(self, repo):
        mock_resp = _mock_response({
            "data": [], "symbol": "BTCUSDT", "market": "crypto_perp", "interval": "1d", "count": 0,
        })
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        with patch.object(repo._client, "request", return_value=mock_resp) as mock_req:
            repo.read_ohlcv("BTCUSDT", "crypto_perp", "1d", start=start, end=end)
            call_kwargs = mock_req.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
            assert params["start"] == start.isoformat()
            assert params["end"] == end.isoformat()


class TestUpsertOHLCV:
    def test_upsert_ohlcv_returns_count(self, repo):
        mock_resp = _mock_response({"upserted": 5})
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [100.0, 105.0],
            "high": [110.0, 115.0],
            "low": [90.0, 95.0],
            "close": [105.0, 110.0],
            "volume": [1000.0, 1200.0],
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            count = repo.upsert_ohlcv(df, "BTCUSDT", "crypto_perp", "perp", "1d")
            assert count == 5


class TestFundamentals:
    def test_read_fundamentals_returns_list(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"id": "1", "symbol": "2330", "market": "tw_stock", "date": "2024-01-01", "data": {"pe": 15.0}},
            ],
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            result = repo.read_fundamentals("2330", "tw_stock")
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["data"]["pe"] == 15.0


class TestDataFrameEndpoints:
    def test_read_funding_rates_returns_dataframe_with_date_index(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2024-01-01", "funding_rate": 0.001},
                {"date": "2024-01-02", "funding_rate": 0.002},
            ],
            "columns": ["date", "funding_rate"],
            "count": 2,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_funding_rates("BTCUSDT")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2

    def test_read_macro_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2024-01-01", "cpi": 3.0, "gdp": 2.5},
            ],
            "columns": ["date", "cpi", "gdp"],
            "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_macro()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1


class TestPerpEndpoints:
    def test_read_perp_ohlcv_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"time": "2024-01-01T00:00:00", "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
            ],
            "symbol": "BTCUSDT", "market": "crypto_perp", "interval": "4h", "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_perp_ohlcv("BTCUSDT")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1

    def test_read_latest_funding_rate_returns_float(self, repo):
        mock_resp = _mock_response({"symbol": "BTCUSDT", "value": 0.0005})
        with patch.object(repo._client, "request", return_value=mock_resp):
            result = repo.read_latest_funding_rate("BTCUSDT")
            assert result == 0.0005

    def test_read_latest_price_returns_none(self, repo):
        mock_resp = _mock_response({"symbol": "BTCUSDT", "value": None})
        with patch.object(repo._client, "request", return_value=mock_resp):
            result = repo.read_latest_price("BTCUSDT")
            assert result is None


class TestRetryAndCircuitBreaker:
    def test_request_retries_on_connect_error(self, repo):
        """Retry fires on ConnectError, succeeds on 3rd attempt."""
        mock_resp = _mock_response({"symbol": "BTCUSDT", "value": 42.0})
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            return mock_resp

        with patch.object(repo._client, "request", side_effect=side_effect):
            result = repo.read_latest_funding_rate("BTCUSDT")
            assert result == 42.0
            assert call_count == 3

    def test_request_raises_circuit_breaker_open_error(self, repo):
        """When CB is open, _request raises CircuitBreakerOpenError immediately."""
        # Trip the breaker (threshold=3)
        for _ in range(3):
            repo._cb.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            repo.read_latest_funding_rate("BTCUSDT")

    def test_record_failure_on_server_error(self, repo):
        """500 errors should record CB failure."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp,
        )
        mock_resp.json.return_value = {}

        with patch.object(repo._client, "request", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                repo.read_latest_funding_rate("BTCUSDT")

        assert repo._cb._failure_count == 1


class TestNonpriceExtendedEndpoints:
    """Tests for Phase 66 nonprice reader methods."""

    def test_read_fundamentals_extended_df_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2025-03-31", "gross_margin": 0.35, "operating_margin": 0.20,
                 "debt_ratio": 0.40, "eps": 5.0},
            ],
            "columns": ["date", "gross_margin", "operating_margin", "debt_ratio", "eps"],
            "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_fundamentals_extended_df("2330")
            assert isinstance(df, pd.DataFrame)
            assert "gross_margin" in df.columns

    def test_read_fundamentals_extended_df_passes_as_of_date(self, repo):
        mock_resp = _mock_response({"data": [], "columns": [], "count": 0})
        with patch.object(repo._client, "request", return_value=mock_resp) as mock_req:
            repo.read_fundamentals_extended_df("2330", as_of_date="2025-06-01")
            call_kwargs = mock_req.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
            assert params.get("as_of_date") == "2025-06-01"

    def test_read_monthly_revenue_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2025-01-31", "monthly_rev": 5000.0, "monthly_rev_yoy": 0.12,
                 "cum_rev": 5000.0, "cum_rev_yoy": 0.12},
            ],
            "columns": ["date", "monthly_rev", "monthly_rev_yoy", "cum_rev", "cum_rev_yoy"],
            "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_monthly_revenue("2330")
            assert isinstance(df, pd.DataFrame)
            assert "monthly_rev_yoy" in df.columns

    def test_read_monthly_revenue_passes_as_of_date(self, repo):
        mock_resp = _mock_response({"data": [], "columns": [], "count": 0})
        with patch.object(repo._client, "request", return_value=mock_resp) as mock_req:
            repo.read_monthly_revenue("2330", as_of_date="2025-03-01")
            call_kwargs = mock_req.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
            assert params.get("as_of_date") == "2025-03-01"

    def test_read_foreign_holding_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2025-01-15", "foreign_holding_ratio": 0.75},
            ],
            "columns": ["date", "foreign_holding_ratio"],
            "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_foreign_holding("2330")
            assert isinstance(df, pd.DataFrame)
            assert "foreign_holding_ratio" in df.columns

    def test_read_quality_factor_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2025-03-31", "profitability_z": 0.5, "growth_z": 0.3, "safety_z": -0.1},
            ],
            "columns": ["date", "profitability_z", "growth_z", "safety_z"],
            "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_quality_factor("2330")
            assert isinstance(df, pd.DataFrame)
            assert "profitability_z" in df.columns

    def test_read_pe_pbr_returns_dataframe(self, repo):
        mock_resp = _mock_response({
            "data": [
                {"date": "2025-01-15", "pe_ratio": 15.0, "pb_ratio": 2.0, "dividend_yield": 0.03},
            ],
            "columns": ["date", "pe_ratio", "pb_ratio", "dividend_yield"],
            "count": 1,
        })
        with patch.object(repo._client, "request", return_value=mock_resp):
            df = repo.read_pe_pbr("2330")
            assert isinstance(df, pd.DataFrame)
            assert "dividend_yield" in df.columns

    def test_read_risk_filters_active_returns_list(self, repo):
        mock_resp = _mock_response({"excluded_symbols": ["2330", "2317"]})
        with patch.object(repo._client, "request", return_value=mock_resp):
            result = repo.read_risk_filters_active("2025-01-15")
            assert isinstance(result, list)
            assert "2330" in result
            assert len(result) == 2

    def test_read_risk_filters_active_empty(self, repo):
        mock_resp = _mock_response({"excluded_symbols": []})
        with patch.object(repo._client, "request", return_value=mock_resp):
            result = repo.read_risk_filters_active("2025-01-15")
            assert isinstance(result, list)
            assert len(result) == 0
