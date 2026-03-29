"""Tests for trade structure feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.trade_structure import AvgTradeSize, TurnoverRatio


@pytest.fixture
def ohlcv():
    """Synthetic OHLCV with known close and volume."""
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0] * 30,
            "high": [105.0] * 30,
            "low": [95.0] * 30,
            "close": [100.0] * 30,
            "volume": [10000.0] * 30,
        },
        index=dates,
    )


@pytest.fixture
def trade_structure_data(ohlcv):
    """Synthetic trade structure data aligned to OHLCV."""
    return pd.DataFrame(
        {
            "trade_value": [5_000_000.0] * len(ohlcv),
            "trade_count": [1000.0] * len(ohlcv),
        },
        index=ohlcv.index,
    )


class TestAvgTradeSize:
    def test_compute_correctly(self, ohlcv, trade_structure_data):
        feat = AvgTradeSize()
        result = feat.compute(ohlcv, trade_structure_data=trade_structure_data)
        assert result.name == "avg_trade_size"
        # 5_000_000 / 1000 = 5000
        expected_val = 5000.0
        assert (result == expected_val).all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = AvgTradeSize()
        result = feat.compute(ohlcv, trade_structure_data=None)
        assert result.name == "avg_trade_size"
        assert result.isna().all()

    def test_empty_ohlcv_returns_empty(self):
        feat = AvgTradeSize()
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = feat.compute(empty)
        assert len(result) == 0


class TestTurnoverRatio:
    def test_compute_correctly(self, ohlcv, trade_structure_data):
        feat = TurnoverRatio()
        result = feat.compute(ohlcv, trade_structure_data=trade_structure_data)
        assert result.name == "turnover_ratio"
        # 5_000_000 / (100 * 10000) = 5.0
        expected_val = 5.0
        assert np.allclose(result.values, expected_val)

    def test_none_data_returns_nan(self, ohlcv):
        feat = TurnoverRatio()
        result = feat.compute(ohlcv, trade_structure_data=None)
        assert result.name == "turnover_ratio"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        """If trade_value column missing, return NaN."""
        feat = TurnoverRatio()
        bad_data = pd.DataFrame(
            {"trade_count": [1000.0] * len(ohlcv)},
            index=ohlcv.index,
        )
        result = feat.compute(ohlcv, trade_structure_data=bad_data)
        assert result.isna().all()
