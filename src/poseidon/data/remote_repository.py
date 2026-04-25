"""Remote data repository -- HTTP client for Thalassa DataService.

Mirrors the 17-method interface of DataRepository, delegating to Thalassa
REST API endpoints instead of local database queries.

Resilience:
- Circuit breaker (outer): rejects requests when Thalassa is consistently failing
- Tenacity retry (inner): retries transient errors (ConnectError, TimeoutException)
  up to 3 times with exponential backoff + jitter

Design decisions:
- D-01: Same public method signatures as DataRepository
- D-05: CB check -> retry -> HTTP request -> CB record
- D-11: tenacity for retry with exponential backoff
- D-12: CircuitBreaker thresholds configurable via Settings
- D-13: httpx.Client (sync) -- Celery workers are sync
- D-14: time.monotonic() in CircuitBreaker
- D-15: Settings fields for all Thalassa connection params
"""

from __future__ import annotations

import contextlib
import logging
from datetime import date, datetime

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from poseidon.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class RemoteDataRepository:
    """HTTP client mirroring DataRepository's 17-method interface.

    Connects to Thalassa DataService REST API with circuit breaker
    and retry resilience.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        cb_threshold: int = 5,
        cb_recovery_timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )
        self._cb = CircuitBreaker(
            threshold=cb_threshold,
            recovery_timeout=cb_recovery_timeout,
        )

    @classmethod
    def from_settings(cls) -> RemoteDataRepository:
        """Construct from Poseidon Settings -- preferred construction method."""
        from poseidon.core.config import settings

        return cls(
            base_url=settings.thalassa_base_url,
            api_key=settings.thalassa_api_key,
            timeout=settings.thalassa_timeout,
            cb_threshold=settings.thalassa_cb_threshold,
            cb_recovery_timeout=settings.thalassa_cb_recovery_timeout,
        )

    # ------------------------------------------------------------------
    # Core request infrastructure
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Execute an HTTP request with circuit breaker (outer) + retry (inner).

        Flow: CB check -> retry wrapper -> HTTP call -> CB record
        """
        if not self._cb.allow_request():
            raise CircuitBreakerOpenError(self._base_url, self._cb.time_to_recovery)

        try:
            resp = self._request_with_retry(method, path, **kwargs)
            resp.raise_for_status()
            self._cb.record_success()
            return resp
        except (httpx.ConnectError, httpx.TimeoutException):
            self._cb.record_failure()
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                self._cb.record_failure()
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 1),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    def _request_with_retry(self, method: str, path: str, **kwargs) -> httpx.Response:
        """HTTP request with tenacity retry on transient errors."""
        return self._client.request(method, path, **kwargs)

    # ------------------------------------------------------------------
    # Response deserialization helpers
    # ------------------------------------------------------------------

    def _parse_ohlcv_response(self, data: dict) -> pd.DataFrame:
        """Parse OHLCVResponse JSON into a DataFrame with datetime index.

        Includes adj_close column with NULL→close fallback (Phase 74 D-08).
        """
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "adj_close"])
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        # Phase 74 D-08: adj_close with fallback to close
        if "adj_close" in df.columns:
            df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
            df["adj_close"] = df["adj_close"].fillna(df["close"])
        else:
            df["adj_close"] = df["close"]
        return df[["open", "high", "low", "close", "volume", "adj_close"]]

    def _parse_dataframe_response(self, data: dict) -> pd.DataFrame:
        """Parse DataFrameResponse JSON into a DataFrame with date/time index."""
        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Detect and set index column
        for col in ("date", "time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        return df

    def _parse_scalar_response(self, data: dict) -> float | None:
        """Parse ScalarValueResponse JSON, returning value or None."""
        return data.get("value")

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def read_ohlcv(
        self,
        symbol: str,
        market: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read OHLCV data from Thalassa."""
        params: dict = {"symbol": symbol, "market": market, "interval": interval}
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        resp = self._request("GET", "/api/v1/ohlcv", params=params)
        return self._parse_ohlcv_response(resp.json())

    def upsert_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: str,
        instrument: str,
        interval: str,
    ) -> int:
        """Upsert OHLCV rows to Thalassa. Returns count of upserted rows."""
        # Prepare records: ensure time column is string
        work = df.copy()
        if "time" not in work.columns and work.index.name == "time":
            work = work.reset_index()
        records = work.to_dict(orient="records")
        for rec in records:
            if "time" in rec and not isinstance(rec["time"], str):
                rec["time"] = str(rec["time"])
        body = {
            "symbol": symbol,
            "market": market,
            "instrument": instrument,
            "interval": interval,
            "data": records,
        }
        resp = self._request("POST", "/api/v1/ohlcv", json=body)
        return resp.json()["upserted"]

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def read_fundamentals(self, symbol: str, market: str):
        """Read fundamentals rows from Thalassa. Returns list of dicts."""
        resp = self._request("GET", "/api/v1/fundamentals", params={"symbol": symbol, "market": market})
        return resp.json()["data"]

    def write_fundamentals(self, symbol: str, market: str, report_date: date, data: dict):
        """Write a fundamentals row to Thalassa. Returns dict."""
        body = {
            "symbol": symbol,
            "market": market,
            "report_date": report_date.isoformat(),
            "data": data,
        }
        resp = self._request("POST", "/api/v1/fundamentals", json=body)
        return resp.json()

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def read_sentiment(self, symbol: str, market: str | None = None, limit: int = 100):
        """Read sentiment scores from Thalassa. Returns list of dicts."""
        params: dict = {"symbol": symbol, "limit": limit}
        if market is not None:
            params["market"] = market
        resp = self._request("GET", "/api/v1/sentiment", params=params)
        return resp.json()["data"]

    def write_sentiment(self, symbol: str, market: str, source_type: str, score: float):
        """Write a sentiment score to Thalassa. Returns dict."""
        body = {
            "symbol": symbol,
            "market": market,
            "source_type": source_type,
            "score": score,
        }
        resp = self._request("POST", "/api/v1/sentiment", json=body)
        return resp.json()

    # ------------------------------------------------------------------
    # DataFrameResponse endpoints (7 methods)
    # ------------------------------------------------------------------

    def read_funding_rates(self, symbol: str, start: str = "2024-01-01") -> pd.DataFrame:
        """Read daily funding rates from Thalassa."""
        resp = self._request("GET", "/api/v1/funding-rates", params={"symbol": symbol, "start": start})
        return self._parse_dataframe_response(resp.json())

    def read_open_interest(self, symbol: str, interval: str = "1h", start: str = "2024-01-01") -> pd.DataFrame:
        """Read open interest data from Thalassa."""
        resp = self._request(
            "GET",
            "/api/v1/open-interest",
            params={"symbol": symbol, "interval": interval, "start": start},
        )
        return self._parse_dataframe_response(resp.json())

    def read_macro(self, start: str = "2020-01-01") -> pd.DataFrame:
        """Read macro economic data from Thalassa."""
        resp = self._request("GET", "/api/v1/macro", params={"start": start})
        return self._parse_dataframe_response(resp.json())

    def read_institutional_flow(self, symbol: str) -> pd.DataFrame:
        """Read institutional investor flow data from Thalassa."""
        resp = self._request("GET", "/api/v1/nonprice/institutional", params={"symbol": symbol})
        return self._parse_dataframe_response(resp.json())

    def read_margin_transactions(self, symbol: str) -> pd.DataFrame:
        """Read margin transaction data from Thalassa."""
        resp = self._request("GET", "/api/v1/nonprice/margin", params={"symbol": symbol})
        return self._parse_dataframe_response(resp.json())

    def read_fundamentals_df(self, symbol: str) -> pd.DataFrame:
        """Read fundamentals as DataFrame from Thalassa."""
        resp = self._request("GET", "/api/v1/nonprice/fundamental", params={"symbol": symbol})
        return self._parse_dataframe_response(resp.json())

    def read_trade_structure(self, symbol: str) -> pd.DataFrame:
        """Read trade structure data from Thalassa."""
        resp = self._request("GET", "/api/v1/nonprice/trade-structure", params={"symbol": symbol})
        return self._parse_dataframe_response(resp.json())

    # ------------------------------------------------------------------
    # Phase 66: Extended nonprice endpoints (6 methods)
    # ------------------------------------------------------------------

    def read_fundamentals_extended_df(self, symbol: str, as_of_date: str | None = None) -> pd.DataFrame:
        """Read extended fundamentals (gross_margin, eps, etc.) from Thalassa."""
        params: dict = {"symbol": symbol}
        if as_of_date is not None:
            params["as_of_date"] = as_of_date
        resp = self._request("GET", "/api/v1/nonprice/fundamentals-extended", params=params)
        return self._parse_dataframe_response(resp.json())

    def read_monthly_revenue(self, symbol: str, as_of_date: str | None = None) -> pd.DataFrame:
        """Read monthly revenue series from Thalassa."""
        params: dict = {"symbol": symbol}
        if as_of_date is not None:
            params["as_of_date"] = as_of_date
        resp = self._request("GET", "/api/v1/nonprice/monthly-revenue", params=params)
        return self._parse_dataframe_response(resp.json())

    def read_foreign_holding(self, symbol: str, as_of_date: str | None = None) -> pd.DataFrame:
        """Read foreign holding ratio series from Thalassa."""
        params: dict = {"symbol": symbol}
        if as_of_date is not None:
            params["as_of_date"] = as_of_date
        resp = self._request("GET", "/api/v1/nonprice/foreign-holding", params=params)
        return self._parse_dataframe_response(resp.json())

    def read_quality_factor(self, symbol: str, as_of_date: str | None = None) -> pd.DataFrame:
        """Read quality factor Z-scores from Thalassa."""
        params: dict = {"symbol": symbol}
        if as_of_date is not None:
            params["as_of_date"] = as_of_date
        resp = self._request("GET", "/api/v1/nonprice/quality-factor", params=params)
        return self._parse_dataframe_response(resp.json())

    def read_pe_pbr(self, symbol: str, as_of_date: str | None = None) -> pd.DataFrame:
        """Read PE/PBR/dividend yield series from Thalassa."""
        params: dict = {"symbol": symbol}
        if as_of_date is not None:
            params["as_of_date"] = as_of_date
        resp = self._request("GET", "/api/v1/nonprice/pe-pbr", params=params)
        return self._parse_dataframe_response(resp.json())

    def read_market_value(self, symbol: str, as_of_date: str | None = None) -> pd.DataFrame:
        """Read daily market value (float shares * price) from Thalassa (Phase 71 D-06)."""
        params: dict = {"symbol": symbol}
        if as_of_date is not None:
            params["as_of_date"] = as_of_date
        resp = self._request("GET", "/api/v1/nonprice/market-value", params=params)
        return self._parse_dataframe_response(resp.json())

    def read_risk_filters_active(self, ref_date: str) -> list[str]:
        """Return list of symbols under any active risk flag on the given date."""
        resp = self._request("GET", "/api/v1/risk-filters/active", params={"date": ref_date})
        return resp.json().get("excluded_symbols", [])

    # ------------------------------------------------------------------
    # Perpetual contract endpoints
    # ------------------------------------------------------------------

    def read_perp_ohlcv(self, symbol: str, interval: str = "4h", lookback_days: int = 30) -> pd.DataFrame:
        """Read perpetual contract OHLCV from Thalassa."""
        resp = self._request(
            "GET",
            "/api/v1/perp/ohlcv",
            params={"symbol": symbol, "interval": interval, "lookback_days": lookback_days},
        )
        return self._parse_ohlcv_response(resp.json())

    def read_latest_funding_rate(self, symbol: str) -> float | None:
        """Return the most recent funding rate for a symbol."""
        resp = self._request("GET", "/api/v1/perp/latest-funding-rate", params={"symbol": symbol})
        return self._parse_scalar_response(resp.json())

    def read_latest_price(self, symbol: str, interval: str = "4h") -> float | None:
        """Return the latest close price for a perpetual symbol."""
        resp = self._request(
            "GET",
            "/api/v1/perp/latest-price",
            params={"symbol": symbol, "interval": interval},
        )
        return self._parse_scalar_response(resp.json())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
