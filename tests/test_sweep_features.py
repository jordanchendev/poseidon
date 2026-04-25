"""Tests for liquidity sweep feature classes (Phase 43.1-02).

Covers wick, swing, trend, volatility extension, and funding rate extension features.
"""

import numpy as np
import pandas as pd


def make_ohlcv(n=50):
    """Create synthetic OHLCV data with valid constraints.

    Guarantees: high >= max(open, close), low <= min(open, close).
    """
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    # high must be >= max(open, close), low must be <= min(open, close)
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + np.abs(np.random.randn(n)) * 2
    low = bar_min - np.abs(np.random.randn(n)) * 2
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


class TestWickRatio:
    def test_columns(self):
        from poseidon.data.features.wick import WickRatio

        result = WickRatio().compute(make_ohlcv())
        assert set(result.columns) == {
            "wick_ratio_upper",
            "wick_ratio_lower",
            "wick_ratio_total",
        }

    def test_values_between_0_and_1(self):
        from poseidon.data.features.wick import WickRatio

        result = WickRatio().compute(make_ohlcv())
        assert (result.dropna() >= 0).all().all()
        assert (result.dropna() <= 1.01).all().all()  # small float tolerance

    def test_empty_input(self):
        from poseidon.data.features.wick import WickRatio

        result = WickRatio().compute(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestRangeExpansion:
    def test_column_name(self):
        from poseidon.data.features.wick import RangeExpansion

        result = RangeExpansion().compute(make_ohlcv(), period=14)
        assert result.name == "range_expansion_14"

    def test_empty_input(self):
        from poseidon.data.features.wick import RangeExpansion

        result = RangeExpansion().compute(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), period=14)
        assert isinstance(result, pd.Series)


class TestBodyRatio:
    def test_column_name(self):
        from poseidon.data.features.wick import BodyRatio

        result = BodyRatio().compute(make_ohlcv())
        assert result.name == "body_ratio"

    def test_values_between_0_and_1(self):
        from poseidon.data.features.wick import BodyRatio

        result = BodyRatio().compute(make_ohlcv())
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.01).all()


class TestSwingHigh:
    def test_column_name(self):
        from poseidon.data.features.swing import SwingHigh

        result = SwingHigh().compute(make_ohlcv(), period=10)
        assert result.name == "swing_high_10"

    def test_values_gte_high(self):
        from poseidon.data.features.swing import SwingHigh

        ohlcv = make_ohlcv()
        result = SwingHigh().compute(ohlcv, period=10)
        valid = result.dropna()
        assert (valid >= ohlcv["high"].iloc[: len(valid)].min()).all()


class TestSwingLow:
    def test_column_name(self):
        from poseidon.data.features.swing import SwingLow

        result = SwingLow().compute(make_ohlcv(), period=10)
        assert result.name == "swing_low_10"


class TestBreakoutDistance:
    def test_columns(self):
        from poseidon.data.features.swing import BreakoutDistance

        result = BreakoutDistance().compute(make_ohlcv(), period=10, atr_period=14)
        assert "breakout_up_10" in result.columns
        assert "breakout_down_10" in result.columns


class TestFibExtension:
    def test_columns(self):
        from poseidon.data.features.swing import FibExtension

        result = FibExtension().compute(make_ohlcv(), period=10)
        assert "fib_ext_up_0_618" in result.columns
        assert "fib_ext_down_1_618" in result.columns


class TestADX:
    def test_column_name(self):
        from poseidon.data.features.trend import ADX

        result = ADX().compute(make_ohlcv(), period=14)
        assert result.name == "adx_14"

    def test_values_0_to_100(self):
        from poseidon.data.features.trend import ADX

        result = ADX().compute(make_ohlcv(100), period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestTrendStrength:
    def test_column_name(self):
        from poseidon.data.features.trend import TrendStrength

        result = TrendStrength().compute(make_ohlcv(100), long_period=50, atr_period=14)
        assert result.name == "trend_strength_50"


class TestHourOfDay:
    def test_column_name(self):
        from poseidon.data.features.trend import HourOfDay

        ohlcv = make_ohlcv()
        ohlcv.index = pd.date_range("2024-01-01", periods=len(ohlcv), freq="h", tz="UTC")
        result = HourOfDay().compute(ohlcv)
        assert result.name == "hour_of_day"
        assert result.iloc[0] == 0  # midnight UTC


class TestATRPercentile:
    def test_column_name(self):
        from poseidon.data.features.volatility import ATRPercentile

        result = ATRPercentile().compute(make_ohlcv(100), period=14, lookback=50)
        assert result.name == "atr_percentile_50"


class TestVolRegime:
    def test_column_name(self):
        from poseidon.data.features.volatility import VolRegime

        result = VolRegime().compute(make_ohlcv(100), short_period=5, long_period=20)
        assert result.name == "vol_regime"

    def test_values_0_to_3(self):
        from poseidon.data.features.volatility import VolRegime

        result = VolRegime().compute(make_ohlcv(100), short_period=5, long_period=20)
        valid = result.dropna()
        assert set(valid.unique()).issubset({0, 1, 2, 3})


class TestFundingRateExtreme:
    def test_columns(self):
        from poseidon.data.features.funding_rate import FundingRateExtreme

        ohlcv = make_ohlcv()
        funding_data = pd.DataFrame(
            {"funding_rate_daily": np.random.randn(len(ohlcv)) * 0.001},
            index=ohlcv.index,
        )
        result = FundingRateExtreme().compute(ohlcv, funding_data=funding_data, period=20, threshold=2.0)
        assert "funding_zscore" in result.columns
        assert "funding_extreme" in result.columns
        assert "funding_direction" in result.columns

    def test_no_funding_data(self):
        from poseidon.data.features.funding_rate import FundingRateExtreme

        ohlcv = make_ohlcv()
        result = FundingRateExtreme().compute(ohlcv, funding_data=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(ohlcv)
