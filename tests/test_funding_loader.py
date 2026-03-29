"""Tests for FundingRateLoader."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from poseidon.data.loaders.funding_loader import FundingRateLoader


class TestSpotToPerp:
    def test_btcusdt(self):
        loader = FundingRateLoader()
        assert loader._spot_to_perp("BTCUSDT") == "BTC/USDT:USDT"

    def test_ethusdt(self):
        loader = FundingRateLoader()
        assert loader._spot_to_perp("ETHUSDT") == "ETH/USDT:USDT"

    def test_solusdt(self):
        loader = FundingRateLoader()
        assert loader._spot_to_perp("SOLUSDT") == "SOL/USDT:USDT"


class TestGetDailyFundingRate:
    def _make_funding_records(self) -> list[dict]:
        """Create synthetic funding rate data: 3 entries/day for 3 days."""
        records = []
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        hours = [0, 8, 16]  # 3 funding intervals per day

        for day in range(3):
            for h in hours:
                ts = int(
                    datetime(2024, 1, 1 + day, h, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
                )
                records.append(
                    {
                        "timestamp": ts,
                        "fundingRate": 0.0001 * (day + 1),
                        "symbol": "BTC/USDT:USDT",
                    }
                )
        return records

    def test_daily_aggregation(self):
        """Funding rates are aggregated to daily sums."""
        records = self._make_funding_records()

        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = int(
            datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
        )
        # Return all records on first call, empty on second
        mock_exchange.fetch_funding_rate_history.side_effect = [records, []]

        loader = FundingRateLoader()
        loader._exchange = mock_exchange

        result = loader.get_daily_funding_rate("BTCUSDT", start="2024-01-01")

        assert not result.empty
        assert "funding_rate_daily" in result.columns
        assert len(result) == 3  # 3 days
        # Day 1: 3 * 0.0001 = 0.0003
        assert abs(result["funding_rate_daily"].iloc[0] - 0.0003) < 1e-10
        # Day 2: 3 * 0.0002 = 0.0006
        assert abs(result["funding_rate_daily"].iloc[1] - 0.0006) < 1e-10

    def test_empty_response_returns_empty(self):
        """Empty funding rate response returns empty DataFrame."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1704067200000
        mock_exchange.fetch_funding_rate_history.return_value = []

        loader = FundingRateLoader()
        loader._exchange = mock_exchange

        result = loader.get_daily_funding_rate("BTCUSDT")
        assert result.empty

    def test_api_error_returns_empty(self):
        """API error returns empty DataFrame."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1704067200000
        mock_exchange.fetch_funding_rate_history.side_effect = Exception("API error")

        loader = FundingRateLoader()
        loader._exchange = mock_exchange

        result = loader.get_daily_funding_rate("BTCUSDT")
        assert result.empty
