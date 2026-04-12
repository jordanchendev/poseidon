"""Tests for OI feature computation (OIChange, OIBuildup, and OICostBasis)."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.open_interest import OIChange, OIBuildup, OICostBasis, _align_oi_to_index


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 40000 + rng.standard_normal(n).cumsum() * 100
    return pd.DataFrame(
        {
            "time": dates,
            "open": close - rng.uniform(0, 50, n),
            "high": close + rng.uniform(0, 100, n),
            "low": close - rng.uniform(0, 100, n),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=dates,
    )


def _make_oi_data(n: int = 50, trend: str = "rising") -> pd.DataFrame:
    """Create synthetic OI data for testing.

    Args:
        n: Number of data points.
        trend: "rising" for monotonically increasing OI, "flat" for constant.
    """
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    if trend == "rising":
        oi = 100000 + np.arange(n) * 500 + np.random.default_rng(42).standard_normal(n) * 50
    elif trend == "flat":
        oi = np.full(n, 100000.0)
    else:
        oi = 100000 + np.random.default_rng(42).standard_normal(n).cumsum() * 200
    return pd.DataFrame({"open_interest": oi}, index=dates)


class TestOIChange:
    """Test OIChange feature computation."""

    def test_output_columns(self):
        """OIChange should produce oi_change_pct and oi_change_zscore_{period}."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=oi_data, period=20)
        assert isinstance(result, pd.DataFrame)
        assert "oi_change_pct" in result.columns
        assert "oi_change_zscore_20" in result.columns
        assert len(result) == len(ohlcv)

    def test_custom_period(self):
        """OIChange should respect custom period parameter."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=oi_data, period=10)
        assert "oi_change_zscore_10" in result.columns

    def test_none_oi_data_returns_nan(self):
        """OIChange with no OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=None)
        assert isinstance(result, pd.DataFrame)
        assert result["oi_change_pct"].isna().all()
        assert result["oi_change_zscore_20"].isna().all()

    def test_empty_oi_data_returns_nan(self):
        """OIChange with empty OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=pd.DataFrame())
        assert result["oi_change_pct"].isna().all()

    def test_positive_oi_change(self):
        """Rising OI should produce positive change percentages."""
        ohlcv = _make_ohlcv(n=30)
        oi_data = _make_oi_data(n=30, trend="rising")
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=oi_data, period=10)
        # Skip first row (NaN from shift) and first few from rolling
        valid = result["oi_change_pct"].dropna()
        assert (valid > 0).sum() > len(valid) * 0.8  # Mostly positive for rising OI


class TestOIBuildup:
    """Test OIBuildup feature computation."""

    def test_output_columns(self):
        """OIBuildup should produce oi_buildup_{period} and oi_price_divergence_{period}."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=24)
        assert isinstance(result, pd.DataFrame)
        assert "oi_buildup_24" in result.columns
        assert "oi_price_divergence_24" in result.columns
        assert len(result) == len(ohlcv)

    def test_custom_period(self):
        """OIBuildup should respect custom period parameter."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=12)
        assert "oi_buildup_12" in result.columns
        assert "oi_price_divergence_12" in result.columns

    def test_none_oi_data_returns_nan(self):
        """OIBuildup with no OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=None)
        assert result["oi_buildup_24"].isna().all()
        assert result["oi_price_divergence_24"].isna().all()

    def test_rising_oi_flat_price_positive_divergence(self):
        """Rising OI + flat price should produce positive divergence."""
        n = 50
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")

        # Flat price
        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": np.full(n, 40000.0),
                "high": np.full(n, 40050.0),
                "low": np.full(n, 39950.0),
                "close": np.full(n, 40000.0),
                "volume": np.full(n, 500.0),
            },
            index=dates,
        )

        # Rising OI
        oi = 100000 + np.arange(n) * 1000
        oi_data = pd.DataFrame({"open_interest": oi}, index=dates)

        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=10)

        # After warmup period, divergence should be positive
        # (OI rising but price flat → OI change > abs(price change))
        valid_divergence = result["oi_price_divergence_10"].dropna()
        assert len(valid_divergence) > 0
        assert (valid_divergence > 0).sum() > len(valid_divergence) * 0.9

    def test_buildup_values_are_cumulative_pct(self):
        """OI buildup values should be cumulative percentage change over period."""
        n = 30
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = _make_ohlcv(n)
        # Linear increasing OI: 100k, 101k, 102k, ...
        oi_values = 100000 + np.arange(n) * 1000.0
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=dates)

        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=5)
        # At index 5: oi_buildup = (105000 - 100000) / 100000 * 100 = 5.0%
        buildup_at_5 = result["oi_buildup_5"].iloc[5]
        assert abs(buildup_at_5 - 5.0) < 0.01


class TestOITimestampAlignment:
    """Verify OI timestamp alignment prevents look-ahead bias.

    Binance fetchOpenInterestHistory returns timestamps representing the
    START of the measurement period (D-01). The _align_oi_to_index(method="ffill")
    ensures that at any bar time T, the OI value used is the most recent
    snapshot with timestamp <= T.

    References: CONTEXT.md D-01, D-02.
    """

    def test_no_lookahead_with_gap(self):
        """Feature at time T must use OI from T or earlier, never T+1.

        Creates OI data with a gap at T=5h. After ffill, the aligned value
        at T=5h must be T=4h's OI (1400), NOT T=6h's (1600).
        """
        n = 10
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": [100.0] * n,
                "high": [105.0] * n,
                "low": [95.0] * n,
                "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )

        # OI data: 1000, 1100, 1200, 1300, 1400, [gap], 1600, 1700, 1800, 1900
        all_oi = [1000 + i * 100 for i in range(n)]
        # Remove index 5 to create a gap
        oi_dates = [d for i, d in enumerate(dates) if i != 5]
        oi_values = [v for i, v in enumerate(all_oi) if i != 5]
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=oi_dates)

        aligned = _align_oi_to_index(oi_data["open_interest"], ohlcv.index)

        # At T=5 (index 5), ffill should carry forward T=4's value (1400)
        assert aligned.iloc[5] == 1400.0, (
            f"Look-ahead bias detected: T=5 got {aligned.iloc[5]}, expected 1400.0 "
            f"(ffill from T=4). If 1600.0 appeared, future data leaked."
        )

    def test_no_lookahead_oi_arrives_later(self):
        """OI data that starts AFTER OHLCV should produce NaN, not backfill.

        If OI only exists from T=5 onward, bars T=0..4 must be NaN (no data
        available yet), not filled from the future.
        """
        n = 10
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")

        # OI only available from index 5 onward
        oi_dates = dates[5:]
        oi_values = [5000.0 + i * 100 for i in range(5)]
        oi_series = pd.Series(oi_values, index=oi_dates, name="open_interest")

        aligned = _align_oi_to_index(oi_series, dates)

        # First 5 bars should be NaN -- no OI data available yet
        assert aligned.iloc[:5].isna().all(), (
            "Look-ahead bias: bars before first OI snapshot must be NaN, "
            f"got {aligned.iloc[:5].tolist()}"
        )
        # Bars from index 5 onward should have values
        assert aligned.iloc[5:].notna().all()

    def test_ffill_carries_last_known_value(self):
        """Forward-fill must carry the last known OI value into subsequent bars."""
        dates = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
        # OI available only at T=0 and T=3
        oi_series = pd.Series(
            [1000.0, 2000.0],
            index=[dates[0], dates[3]],
            name="open_interest",
        )

        aligned = _align_oi_to_index(oi_series, dates)

        assert aligned.iloc[0] == 1000.0  # T=0: direct match
        assert aligned.iloc[1] == 1000.0  # T=1: ffill from T=0
        assert aligned.iloc[2] == 1000.0  # T=2: ffill from T=0
        assert aligned.iloc[3] == 2000.0  # T=3: direct match
        assert aligned.iloc[4] == 2000.0  # T=4: ffill from T=3
        assert aligned.iloc[5] == 2000.0  # T=5: ffill from T=3


class TestOICostBasis:
    """Test OICostBasis (OIWAP) feature computation.

    References: CONTEXT.md D-04, D-05, D-06, D-07, D-08, D-11.
    """

    def test_output_columns(self):
        """OICostBasis should produce oiwap_{period} and oiwap_distance_{period}."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=oi_data)
        assert isinstance(result, pd.DataFrame)
        assert "oiwap_168" in result.columns
        assert "oiwap_distance_168" in result.columns
        assert len(result) == len(ohlcv)

    def test_custom_period(self):
        """OICostBasis should respect custom period parameter."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=oi_data, period=24)
        assert "oiwap_24" in result.columns
        assert "oiwap_distance_24" in result.columns

    def test_oiwap_basic_computation(self):
        """With all-increasing OI, OIWAP should be a weighted average of close prices.

        Construct simple scenario: constant close=100, linearly increasing OI.
        All delta_oi > 0, all close=100 -> OIWAP should be exactly 100.
        """
        n = 20
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": [100.0] * n,
                "high": [105.0] * n,
                "low": [95.0] * n,
                "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        # Linearly increasing OI: every bar has delta_oi > 0
        oi_values = [10000.0 + i * 500 for i in range(n)]
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=dates)

        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=oi_data, period=50)

        # All bars have close=100 and all delta_oi>0, so OIWAP must be 100
        valid_oiwap = result["oiwap_50"].dropna()
        assert len(valid_oiwap) > 0
        # Skip first bar (delta_oi is NaN from diff)
        for val in valid_oiwap.iloc[1:]:
            assert abs(val - 100.0) < 0.01, f"Expected OIWAP ~100.0, got {val}"

    def test_oi_decrease_ignored(self):
        """OI decreases must NOT affect OIWAP computation (D-06).

        Scenario: OI increases at close=100, then decreases at close=200.
        OIWAP should stay at ~100 because decreases are filtered out.
        """
        n = 10
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        # First 5 bars: close=100, OI rising
        # Last 5 bars: close=200, OI falling
        close_prices = [100.0] * 5 + [200.0] * 5
        oi_values = [10000.0, 11000.0, 12000.0, 13000.0, 14000.0,
                     13000.0, 12000.0, 11000.0, 10000.0, 9000.0]
        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": close_prices,
                "high": [c + 5 for c in close_prices],
                "low": [c - 5 for c in close_prices],
                "close": close_prices,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=dates)

        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=oi_data, period=50)

        # OIWAP in the last bars should still be ~100, not ~200
        # because OI decreases (bars 5-9) are ignored
        last_oiwap = result["oiwap_50"].iloc[-1]
        assert abs(last_oiwap - 100.0) < 0.01, (
            f"OI decrease contaminated OIWAP: got {last_oiwap}, expected ~100.0"
        )

    def test_none_oi_returns_nan(self):
        """OICostBasis with no OI data should return NaN columns (D-11)."""
        ohlcv = _make_ohlcv()
        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=None)
        assert isinstance(result, pd.DataFrame)
        assert "oiwap_168" in result.columns
        assert "oiwap_distance_168" in result.columns
        assert result["oiwap_168"].isna().all()
        assert result["oiwap_distance_168"].isna().all()

    def test_empty_oi_returns_nan(self):
        """OICostBasis with empty OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=pd.DataFrame())
        assert result["oiwap_168"].isna().all()

    def test_missing_column_returns_nan(self):
        """OICostBasis with oi_data missing 'open_interest' column returns NaN."""
        ohlcv = _make_ohlcv()
        feature = OICostBasis()
        bad_data = pd.DataFrame({"wrong_column": [1, 2, 3]})
        result = feature.compute(ohlcv, oi_data=bad_data)
        assert result["oiwap_168"].isna().all()

    def test_oiwap_distance_sign(self):
        """oiwap_distance should be positive when close > OIWAP, negative otherwise (D-08).

        Scenario: OI increases at close=100 (OIWAP anchors at 100).
        Then close jumps to 120 -> distance should be positive (~20%).
        """
        n = 10
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        # First 5 bars: close=100, OI rising -> OIWAP ~ 100
        # Bar 6+: close=120, OI still rising -> distance should be positive
        close_prices = [100.0] * 5 + [120.0] * 5
        oi_values = [10000.0 + i * 500 for i in range(n)]
        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": close_prices,
                "high": [c + 5 for c in close_prices],
                "low": [c - 5 for c in close_prices],
                "close": close_prices,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=dates)

        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=oi_data, period=50)

        # Last bar: close=120, OIWAP is weighted avg biased toward 100
        # -> distance should be positive
        last_distance = result["oiwap_distance_50"].iloc[-1]
        assert last_distance > 0, f"Expected positive distance, got {last_distance}"

    def test_all_oi_decrease_produces_nan(self):
        """When OI only decreases (no new positions), OIWAP should be NaN.

        rolling_oi_sum == 0 -> division produces NaN naturally.
        """
        n = 10
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        # Monotonically decreasing OI
        oi_values = [20000.0 - i * 1000 for i in range(n)]
        ohlcv = _make_ohlcv(n)
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=dates)

        feature = OICostBasis()
        result = feature.compute(ohlcv, oi_data=oi_data, period=50)

        # After first bar (which has NaN delta from diff), all delta_oi <= 0
        # So rolling_oi_sum is 0 -> OIWAP is NaN for bars 1+
        assert result["oiwap_50"].iloc[2:].isna().all(), (
            "OIWAP should be NaN when no OI increases exist in window"
        )
