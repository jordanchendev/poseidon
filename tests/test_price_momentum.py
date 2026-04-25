"""Tests for PriceMomentum feature classes (Phase 71 D-15)."""

import numpy as np
import pandas as pd

from poseidon.data.features.price_momentum import (
    _DAYS_3M,
    _DAYS_6M,
    _DAYS_12M,
    PriceMomentum3M,
    PriceMomentum6M,
    PriceMomentum12M,
)


def _make_ohlcv(n_rows: int = 300, base_price: float = 100.0, growth: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with trending close prices."""
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    close = base_price * (1 + growth) ** np.arange(n_rows)
    return pd.DataFrame(
        {
            "time": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000, 10000, n_rows),
        }
    ).set_index("time")


class TestPriceMomentum:
    """Unit tests for PriceMomentum3M/6M/12M feature classes."""

    def test_3m_return_computation(self):
        ohlcv = _make_ohlcv(n_rows=200, growth=0.002)
        feat = PriceMomentum3M()
        result = feat.compute(ohlcv)
        assert result.name == "momentum_3m"
        # First 63 values should be NaN (no lookback data)
        assert result.iloc[:_DAYS_3M].isna().all()
        # Values after lookback should be non-NaN
        assert result.iloc[_DAYS_3M:].notna().any()
        # Check one specific value: close[i] / close[i-63] - 1
        idx = 100
        expected = ohlcv["close"].iloc[idx] / ohlcv["close"].iloc[idx - _DAYS_3M] - 1.0
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_6m_return_computation(self):
        ohlcv = _make_ohlcv(n_rows=300, growth=0.001)
        feat = PriceMomentum6M()
        result = feat.compute(ohlcv)
        assert result.name == "momentum_6m"
        assert result.iloc[:_DAYS_6M].isna().all()
        assert result.iloc[_DAYS_6M:].notna().any()
        idx = 200
        expected = ohlcv["close"].iloc[idx] / ohlcv["close"].iloc[idx - _DAYS_6M] - 1.0
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_12m_return_computation(self):
        ohlcv = _make_ohlcv(n_rows=400, growth=0.001)
        feat = PriceMomentum12M()
        result = feat.compute(ohlcv)
        assert result.name == "momentum_12m"
        assert result.iloc[:_DAYS_12M].isna().all()
        assert result.iloc[_DAYS_12M:].notna().any()
        idx = 300
        expected = ohlcv["close"].iloc[idx] / ohlcv["close"].iloc[idx - _DAYS_12M] - 1.0
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_insufficient_data_returns_empty_series(self):
        short = _make_ohlcv(n_rows=30)
        for feat_cls in [PriceMomentum3M, PriceMomentum6M, PriceMomentum12M]:
            result = feat_cls().compute(short)
            assert len(result) == 0  # empty Series

    def test_empty_ohlcv_returns_empty_series(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        for feat_cls in [PriceMomentum3M, PriceMomentum6M, PriceMomentum12M]:
            result = feat_cls().compute(empty)
            assert len(result) == 0

    def test_column_names_match_registry(self):
        assert PriceMomentum3M.name == "momentum_3m"
        assert PriceMomentum6M.name == "momentum_6m"
        assert PriceMomentum12M.name == "momentum_12m"

    def test_momentum_is_not_nonprice_spec(self):
        """CRITICAL: momentum routes to OHLCV, not nonprice (Pitfall 1)."""
        from poseidon.data.feature_engine.specs import is_nonprice_spec

        assert is_nonprice_spec("momentum_3m") is False
        assert is_nonprice_spec("momentum_6m") is False
        assert is_nonprice_spec("momentum_12m") is False

    # --- Phase 74: adj_close tests ---

    def test_momentum_3m_uses_adj_close(self):
        """Momentum should use adj_close when available (Phase 74 D-10)."""
        n_rows = 200
        dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
        # close has a 4x drop at midpoint (simulating split), adj_close is smooth
        close_vals = np.ones(n_rows) * 400.0
        close_vals[100:] = 100.0  # raw close drops 4x at index 100

        adj_close_vals = np.ones(n_rows) * 400.0  # adjusted: no discontinuity
        adj_close_vals = 400.0 * (1 + 0.001) ** np.arange(n_rows)  # smooth uptrend

        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": close_vals * 0.99,
                "high": close_vals * 1.01,
                "low": close_vals * 0.98,
                "close": close_vals,
                "volume": np.random.randint(1000, 10000, n_rows),
                "adj_close": adj_close_vals,
            }
        ).set_index("time")

        feat = PriceMomentum3M()
        result = feat.compute(ohlcv)

        # Check at index 120 (post-split): should use adj_close, not close
        idx = 120
        expected = adj_close_vals[idx] / adj_close_vals[idx - _DAYS_3M] - 1.0
        assert abs(result.iloc[idx] - expected) < 1e-10

        # If it used close, the result would be 100/400 - 1 = -0.75 (wrong)
        wrong_val = close_vals[idx] / close_vals[idx - _DAYS_3M] - 1.0
        assert abs(result.iloc[idx] - wrong_val) > 0.1  # must NOT match close-based calc

    def test_momentum_3m_fallback_to_close(self):
        """Momentum should fall back to close when adj_close is absent (backward compat)."""
        ohlcv = _make_ohlcv(n_rows=200, growth=0.002)
        # _make_ohlcv does NOT include adj_close column
        assert "adj_close" not in ohlcv.columns

        feat = PriceMomentum3M()
        result = feat.compute(ohlcv)

        # Should compute from close (same as before Phase 74)
        idx = 100
        expected = ohlcv["close"].iloc[idx] / ohlcv["close"].iloc[idx - _DAYS_3M] - 1.0
        assert abs(result.iloc[idx] - expected) < 1e-10
