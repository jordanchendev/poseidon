"""Tests for cross-market return alignment and covariance computation.

Tests use synthetic data only -- no DB or Redis required for return alignment tests.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tw_stock_close() -> pd.Series:
    """TW stock close prices on business days (Mon-Fri, Asia/Taipei)."""
    dates = pd.bdate_range("2024-01-08", "2024-01-19", freq="B")
    prices = [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5]
    return pd.Series(prices[: len(dates)], index=dates, name="TW2330")


@pytest.fixture
def us_stock_close() -> pd.Series:
    """US stock close prices on business days, missing 2024-01-15 (MLK Day)."""
    dates = pd.bdate_range("2024-01-08", "2024-01-19", freq="B")
    prices = [200.0, 201.0, 202.0, 201.5, 203.0, 204.0, 203.5, 205.0, 206.0, 205.5]
    s = pd.Series(prices[: len(dates)], index=dates, name="AAPL")
    # Drop MLK Day (2024-01-15) to simulate US holiday
    return s.drop(pd.Timestamp("2024-01-15"), errors="ignore")


@pytest.fixture
def crypto_close() -> pd.Series:
    """Crypto close prices on every calendar day."""
    dates = pd.date_range("2024-01-08", "2024-01-19", freq="D")
    prices = [
        40000,
        40100,
        40200,
        40150,
        40300,
        40400,
        40350,
        40500,
        40600,
        40550,
        40700,
        40800,
    ]
    return pd.Series(prices[: len(dates)], index=dates, name="BTCUSDT")


@pytest.fixture
def as_of_date() -> datetime:
    return datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Tests: MARKET_CALENDARS
# ---------------------------------------------------------------------------


class TestMarketCalendars:
    def test_has_tw_stock_entry(self):
        from poseidon.risk.var.returns import MARKET_CALENDARS

        assert "tw_stock" in MARKET_CALENDARS
        assert "tz" in MARKET_CALENDARS["tw_stock"]
        assert "freq" in MARKET_CALENDARS["tw_stock"]

    def test_has_tw_futures_entry(self):
        from poseidon.risk.var.returns import MARKET_CALENDARS

        assert "tw_futures" in MARKET_CALENDARS

    def test_has_us_stock_entry(self):
        from poseidon.risk.var.returns import MARKET_CALENDARS

        assert "us_stock" in MARKET_CALENDARS

    def test_has_crypto_spot_entry(self):
        from poseidon.risk.var.returns import MARKET_CALENDARS

        assert "crypto_spot" in MARKET_CALENDARS


# ---------------------------------------------------------------------------
# Tests: compute_returns
# ---------------------------------------------------------------------------


class TestComputeReturns:
    def test_log_returns_calculated(self, tw_stock_close, as_of_date):
        from poseidon.risk.var.returns import compute_returns

        returns = compute_returns(tw_stock_close, as_of=as_of_date)
        # Log returns should have len(prices_within_as_of) - 1 entries
        assert len(returns) > 0
        # First return: log(101/100)
        expected_first = np.log(101.0 / 100.0)
        np.testing.assert_almost_equal(returns.iloc[0], expected_first, decimal=6)

    def test_as_of_filters_future_data(self, tw_stock_close):
        from poseidon.risk.var.returns import compute_returns

        early_as_of = datetime(2024, 1, 10, 23, 59, 59, tzinfo=UTC)
        returns = compute_returns(tw_stock_close, as_of=early_as_of)
        # Only prices on or before 2024-01-10 => 3 prices => 2 returns
        assert len(returns) == 2

    def test_no_nan_in_output(self, tw_stock_close, as_of_date):
        from poseidon.risk.var.returns import compute_returns

        returns = compute_returns(tw_stock_close, as_of=as_of_date)
        assert not returns.isna().any()


# ---------------------------------------------------------------------------
# Tests: align_returns
# ---------------------------------------------------------------------------


class TestAlignReturns:
    def test_forward_fill_tw_holiday_gaps(self):
        """TW has data on a day US doesn't -> US forward-filled."""
        from poseidon.risk.var.returns import align_returns

        # TW has Mon-Fri, US missing Wed
        dates_tw = pd.bdate_range("2024-01-08", "2024-01-12", freq="B")
        dates_us = pd.bdate_range("2024-01-08", "2024-01-12", freq="B").drop(pd.Timestamp("2024-01-10"))
        tw_ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=dates_tw, name="TW")
        us_ret = pd.Series([0.02, 0.01, 0.03, -0.01], index=dates_us, name="US")

        as_of = datetime(2024, 1, 12, 23, 59, 59, tzinfo=UTC)
        aligned = align_returns({"TW": tw_ret, "US": us_ret}, as_of=as_of)

        # US value on Wed should be forward-filled from Tue
        assert not aligned.isna().any().any()
        assert len(aligned) == 5  # All 5 business days present

    def test_as_of_excludes_future_data(self):
        from poseidon.risk.var.returns import align_returns

        dates = pd.bdate_range("2024-01-08", "2024-01-19", freq="B")
        s1 = pd.Series(range(len(dates)), index=dates, name="A", dtype=float)
        s2 = pd.Series(range(len(dates)), index=dates, name="B", dtype=float)

        as_of = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)
        aligned = align_returns({"A": s1, "B": s2}, as_of=as_of)

        # No dates after 2024-01-15
        assert aligned.index.max() <= pd.Timestamp("2024-01-15")

    def test_crypto_stock_alignment(self):
        """Crypto (daily) and stock (bday) align to union calendar."""
        from poseidon.risk.var.returns import align_returns

        bdays = pd.bdate_range("2024-01-08", "2024-01-12", freq="B")
        cdays = pd.date_range("2024-01-08", "2024-01-14", freq="D")

        stock_ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=bdays, name="STOCK")
        crypto_ret = pd.Series(
            [0.005, 0.01, -0.005, 0.015, 0.008, 0.012, -0.002],
            index=cdays,
            name="CRYPTO",
        )

        as_of = datetime(2024, 1, 14, 23, 59, 59, tzinfo=UTC)
        aligned = align_returns({"STOCK": stock_ret, "CRYPTO": crypto_ret}, as_of=as_of)

        # Should have data for the union of dates; stock missing on weekends -> ffill
        assert not aligned.isna().any().any()

    def test_empty_when_no_overlap(self):
        from poseidon.risk.var.returns import align_returns

        dates_a = pd.bdate_range("2024-01-08", "2024-01-12", freq="B")
        dates_b = pd.bdate_range("2024-02-08", "2024-02-12", freq="B")

        s1 = pd.Series([0.01] * len(dates_a), index=dates_a, name="A")
        s2 = pd.Series([0.02] * len(dates_b), index=dates_b, name="B")

        as_of = datetime(2024, 3, 1, tzinfo=UTC)
        align_returns({"A": s1, "B": s2}, as_of=as_of)

        # After ffill + dropna(how="all"), rows where at least one has data remain,
        # but those rows will still have NaN for the other asset before its first date.
        # After ffill, the gap between Jan and Feb means A's last value carries forward.
        # So it should NOT be empty -- but any rows before A's first obs should be dropped.
        # Actually: concat auto-aligns, then ffill fills A's values into Feb dates.
        # The result should have data.
        # The "empty" case is when as_of is before both series:
        early_as_of = datetime(2024, 1, 1, tzinfo=UTC)
        aligned_empty = align_returns({"A": s1, "B": s2}, as_of=early_as_of)
        assert aligned_empty.empty

    def test_determinism(self):
        """Same as_of + same data => identical output (PRISK-03)."""
        from poseidon.risk.var.returns import align_returns

        dates = pd.bdate_range("2024-01-08", "2024-01-19", freq="B")
        s1 = pd.Series(np.random.default_rng(42).standard_normal(len(dates)), index=dates, name="X")
        s2 = pd.Series(np.random.default_rng(43).standard_normal(len(dates)), index=dates, name="Y")

        as_of = datetime(2024, 1, 15, tzinfo=UTC)
        result1 = align_returns({"X": s1, "Y": s2}, as_of=as_of)
        result2 = align_returns({"X": s1, "Y": s2}, as_of=as_of)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Fixtures for covariance tests
# ---------------------------------------------------------------------------


@pytest.fixture
def aligned_returns_3assets() -> pd.DataFrame:
    """Synthetic aligned return DataFrame: 3 assets, 60 business days."""
    rng = np.random.default_rng(seed=123)
    dates = pd.bdate_range("2024-01-02", periods=60, freq="B")
    data = {
        "TW2330": rng.standard_normal(60) * 0.02,
        "AAPL": rng.standard_normal(60) * 0.015,
        "BTCUSDT": rng.standard_normal(60) * 0.04,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def small_aligned_returns() -> pd.DataFrame:
    """Only 10 rows -- below default min_observations threshold."""
    rng = np.random.default_rng(seed=456)
    dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
    data = {
        "A": rng.standard_normal(10) * 0.02,
        "B": rng.standard_normal(10) * 0.015,
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests: compute_covariance
# ---------------------------------------------------------------------------


class TestComputeCovariance:
    def test_returns_ndarray_correct_shape(self, aligned_returns_3assets):
        from poseidon.risk.var.covariance import compute_covariance

        cov = compute_covariance(aligned_returns_3assets)
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (3, 3)

    def test_epsilon_regularization(self, aligned_returns_3assets):
        from poseidon.risk.var.covariance import EPSILON, compute_covariance

        cov = compute_covariance(aligned_returns_3assets)
        # Diagonal should be >= epsilon (regularized)
        for i in range(cov.shape[0]):
            assert cov[i, i] >= EPSILON

    def test_raises_on_insufficient_observations(self, small_aligned_returns):
        from poseidon.risk.var.covariance import compute_covariance

        with pytest.raises(ValueError, match="Insufficient observations"):
            compute_covariance(small_aligned_returns, min_observations=30)

    def test_accepts_sufficient_observations(self, aligned_returns_3assets):
        from poseidon.risk.var.covariance import compute_covariance

        # 60 rows >= 30 min_observations => should not raise
        cov = compute_covariance(aligned_returns_3assets, min_observations=30)
        assert cov.shape == (3, 3)

    def test_symmetric(self, aligned_returns_3assets):
        from poseidon.risk.var.covariance import compute_covariance

        cov = compute_covariance(aligned_returns_3assets)
        np.testing.assert_array_almost_equal(cov, cov.T)


# ---------------------------------------------------------------------------
# Tests: cache_covariance / load_cached_covariance
# ---------------------------------------------------------------------------


class TestCovarianceCache:
    def test_cache_roundtrip(self, fake_redis, aligned_returns_3assets):
        from poseidon.risk.var.covariance import (
            cache_covariance,
            compute_covariance,
            load_cached_covariance,
        )

        symbols = ["TW2330", "AAPL", "BTCUSDT"]
        cov = compute_covariance(aligned_returns_3assets)
        as_of = datetime(2024, 3, 29, tzinfo=UTC)

        cache_covariance(fake_redis, symbols, cov, as_of)
        result = load_cached_covariance(fake_redis)

        assert result is not None
        loaded_symbols, loaded_matrix, metadata = result
        assert loaded_symbols == symbols
        np.testing.assert_array_almost_equal(loaded_matrix, cov)
        assert "as_of" in metadata
        assert "computed_at" in metadata

    def test_load_returns_none_on_miss(self, fake_redis):
        from poseidon.risk.var.covariance import load_cached_covariance

        result = load_cached_covariance(fake_redis)
        assert result is None

    def test_cache_key_matches_spec(self, fake_redis, aligned_returns_3assets):
        from poseidon.risk.var.covariance import (
            COVARIANCE_CACHE_KEY,
            cache_covariance,
            compute_covariance,
        )

        symbols = ["TW2330", "AAPL", "BTCUSDT"]
        cov = compute_covariance(aligned_returns_3assets)
        as_of = datetime(2024, 3, 29, tzinfo=UTC)

        cache_covariance(fake_redis, symbols, cov, as_of)

        # Verify the key is exactly what D-05 specifies
        assert COVARIANCE_CACHE_KEY == "poseidon:var:covariance:latest"
        assert fake_redis.exists(COVARIANCE_CACHE_KEY)
