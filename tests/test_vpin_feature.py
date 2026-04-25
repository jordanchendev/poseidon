"""Tests for VPIN (Volume-synchronized Probability of Informed Trading) feature.

RED phase: Tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_ohlcv(n: int, *, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data with realistic volume variance."""
    rng = np.random.default_rng(seed)
    close = 100 + rng.standard_normal(n).cumsum() * 0.5
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = low + rng.uniform(0.0, 1.0, n) * (high - low)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
    )


def _make_balanced_ohlcv(n: int) -> pd.DataFrame:
    """Create OHLCV where close == open for every bar -> 50/50 BVC split.

    With close == open, BVC z = 0, norm.cdf(0) = 0.5 -> buy_vol == sell_vol.
    This means every volume bucket has |buy-sell| == 0, so VPIN ~ 0.
    """
    mid = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "open": mid,
            "high": mid + 1.0,
            "low": mid - 1.0,
            "close": mid,  # close == open -> z = 0 -> buy_pct = 0.5
            "volume": np.full(n, 500.0),
        },
    )


def _make_one_sided_ohlcv(n: int) -> pd.DataFrame:
    """Create OHLCV where close == high consistently -> ~100% buy volume.

    When close == high, z = (high - open) / (high - low).  With open == low,
    z = 1.0, norm.cdf(1.0) ~ 0.84, so buy_pct ~ 0.84.  Every bucket has
    a strong buy imbalance -> VPIN should be high.
    """
    base = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "open": base,  # open == low
            "high": base + 2.0,  # close == high
            "low": base,
            "close": base + 2.0,
            "volume": np.full(n, 500.0),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVPINNaming:
    """Test 1: VPIN returns Series named vpin_{n_buckets}."""

    def test_default_name(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = _make_ohlcv(300)
        result = VPIN().compute(ohlcv)
        assert isinstance(result, pd.Series)
        assert result.name == "vpin_50"

    def test_custom_n_buckets_name(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = _make_ohlcv(300)
        result = VPIN().compute(ohlcv, n_buckets=30)
        assert result.name == "vpin_30"


class TestVPINRange:
    """Test 2: All non-NaN VPIN values are in [0, 1] range."""

    def test_values_in_unit_range(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = _make_ohlcv(300)
        result = VPIN().compute(ohlcv)
        valid = result.dropna()
        assert len(valid) > 0, "Expected some non-NaN VPIN values"
        assert (valid >= 0).all(), f"VPIN has values below 0: {valid.min()}"
        assert (valid <= 1).all(), f"VPIN has values above 1: {valid.max()}"


class TestVPINBalancedVolume:
    """Test 3: Perfectly balanced volume produces VPIN near 0."""

    def test_balanced_near_zero(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = _make_balanced_ohlcv(300)
        result = VPIN().compute(ohlcv, lookback=50, n_buckets=50)
        valid = result.dropna()
        assert len(valid) > 0, "Expected some non-NaN values for balanced OHLCV"
        # Balanced volume: buy_vol == sell_vol, so bucket imbalance ~ 0
        assert valid.mean() < 0.15, f"Balanced VPIN should be near 0, got mean={valid.mean():.4f}"


class TestVPINOneSided:
    """Test 4: Perfectly one-sided volume produces VPIN near 1.

    With open=low and close=high, z=1.0 -> buy_pct ~ 0.84.
    Bucket imbalance = |0.84 - 0.16| / 1.0 = 0.68 per bucket.
    VPIN should be substantially above 0.5.
    """

    def test_one_sided_high_vpin(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = _make_one_sided_ohlcv(300)
        result = VPIN().compute(ohlcv, lookback=50, n_buckets=50)
        valid = result.dropna()
        assert len(valid) > 0, "Expected some non-NaN values for one-sided OHLCV"
        # One-sided: strong buy imbalance -> VPIN should be high
        assert valid.mean() > 0.5, f"One-sided VPIN should be > 0.5, got mean={valid.mean():.4f}"


class TestVPINEmpty:
    """Test 5: Empty DataFrame returns empty Series with correct name."""

    def test_empty_input(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = VPIN().compute(ohlcv)
        assert isinstance(result, pd.Series)
        assert result.name == "vpin_50"
        assert len(result) == 0


class TestVPINShortData:
    """Test 6: Short DataFrame returns all NaN."""

    def test_short_input_all_nan(self):
        from poseidon.data.features.vpin import VPIN

        # lookback=50 + n_buckets=50 = 100 minimum
        ohlcv = _make_ohlcv(50)
        result = VPIN().compute(ohlcv, lookback=50, n_buckets=50)
        assert isinstance(result, pd.Series)
        assert result.name == "vpin_50"
        # Too short: all NaN (or empty)
        if len(result) > 0:
            assert result.isna().all(), "Short input should produce all NaN"


class TestVPINLongData:
    """Test 7: VPIN produces values for sufficiently long DataFrames."""

    def test_produces_values_for_long_data(self):
        from poseidon.data.features.vpin import VPIN

        ohlcv = _make_ohlcv(400)
        result = VPIN().compute(ohlcv, lookback=50, n_buckets=50)
        valid = result.dropna()
        assert len(valid) > 0, "Expected non-NaN VPIN values for 400-row data"


class TestVPINAdaptiveBucketSize:
    """Test 8: Adaptive bucket size changes with varying volume levels."""

    def test_different_volume_levels_different_buckets(self):
        from poseidon.data.features.vpin import VPIN

        # Low volume data
        ohlcv_low = _make_ohlcv(300, seed=10)
        ohlcv_low["volume"] = ohlcv_low["volume"] * 0.1  # 10-100 range

        # High volume data
        ohlcv_high = _make_ohlcv(300, seed=10)
        ohlcv_high["volume"] = ohlcv_high["volume"] * 10.0  # 1000-10000 range

        vpin_inst = VPIN()
        result_low = vpin_inst.compute(ohlcv_low, lookback=50, n_buckets=50)
        result_high = vpin_inst.compute(ohlcv_high, lookback=50, n_buckets=50)

        # Both should produce non-NaN values (adaptive sizing handles different levels)
        valid_low = result_low.dropna()
        valid_high = result_high.dropna()
        assert len(valid_low) > 0, "Low-volume data should produce VPIN values"
        assert len(valid_high) > 0, "High-volume data should produce VPIN values"
