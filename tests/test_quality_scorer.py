"""Tests for DataQualityScorer (DVAL-05).

Covers 4 weighted dimensions: completeness, consistency, anomaly-free, timeliness.
All external dependencies (validate_ohlcv, read_ohlcv) are mocked.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from poseidon.data.quality_scorer import DataQualityScorer, QualityDimensions
from poseidon.data.validation import ValidationResult
from poseidon.data.validation_rules import CheckResult, Severity


def _make_df(n: int = 10, freq: str = "D") -> pd.DataFrame:
    """Create a simple OHLCV DataFrame with *n* rows."""
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "time": idx,
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [102.0] * n,
            "volume": [1000] * n,
        },
        index=range(n),
    )


def _all_pass_result() -> ValidationResult:
    """ValidationResult where all checks pass."""
    return ValidationResult(
        checks=[
            CheckResult(code="OHLC_CONSISTENCY", passed=True, severity=Severity.INFO, message="OK"),
            CheckResult(code="DUPLICATE_TIMESTAMP", passed=True, severity=Severity.INFO, message="OK"),
            CheckResult(code="FUTURE_TIMESTAMP", passed=True, severity=Severity.INFO, message="OK"),
            CheckResult(code="PRICE_CHANGE_EXCESSIVE", passed=True, severity=Severity.INFO, message="OK"),
        ],
        passed=True,
        critical_count=0,
        warning_count=0,
        info_count=0,
    )


class TestDataQualityScorer:
    """Tests for DataQualityScorer.compute()."""

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_perfect_data_returns_score_near_one(self, mock_validate):
        """Test 1: Perfect data -> score ~1.0."""
        mock_validate.return_value = _all_pass_result()
        df = _make_df(10)
        scorer = DataQualityScorer()
        result = scorer.compute(
            df,
            market="crypto_spot",
            interval="1d",
            expected_rows=10,
            latest_expected=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        assert isinstance(result, QualityDimensions)
        assert result.completeness == pytest.approx(1.0)
        assert result.consistency == pytest.approx(1.0)
        assert result.anomaly_free == pytest.approx(1.0)
        assert result.timeliness == pytest.approx(1.0)
        assert result.composite == pytest.approx(1.0)

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_half_missing_rows_reduces_completeness(self, mock_validate):
        """Test 2: 50% missing rows -> completeness=0.5, total reduced by 0.30*0.5."""
        mock_validate.return_value = _all_pass_result()
        df = _make_df(5)  # 5 of expected 10
        scorer = DataQualityScorer()
        result = scorer.compute(
            df,
            market="crypto_spot",
            interval="1d",
            expected_rows=10,
            latest_expected=datetime(2026, 1, 5, tzinfo=timezone.utc),
        )
        assert result.completeness == pytest.approx(0.5)
        # Other dimensions still 1.0, so composite = 0.30*0.5 + 0.25*1 + 0.25*1 + 0.20*1 = 0.85
        assert result.composite == pytest.approx(0.85)

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_critical_failures_reduce_consistency(self, mock_validate):
        """Test 3: 2 of 10 rows with CRITICAL failures -> consistency=0.8."""
        mock_validate.return_value = ValidationResult(
            checks=[
                CheckResult(code="OHLC_CONSISTENCY", passed=False, severity=Severity.CRITICAL, message="bad", affected_rows=[0]),
                CheckResult(code="OHLC_CONSISTENCY", passed=False, severity=Severity.CRITICAL, message="bad", affected_rows=[1]),
                CheckResult(code="DUPLICATE_TIMESTAMP", passed=True, severity=Severity.INFO, message="OK"),
            ],
            passed=False,
            critical_count=2,
            warning_count=0,
            info_count=0,
        )
        df = _make_df(10)
        scorer = DataQualityScorer()
        result = scorer.compute(
            df,
            market="crypto_spot",
            interval="1d",
            expected_rows=10,
            latest_expected=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        assert result.consistency == pytest.approx(0.8)

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_warnings_reduce_anomaly_free(self, mock_validate):
        """Test 4: 3 of 10 rows with WARNING flags -> anomaly_free=0.7."""
        mock_validate.return_value = ValidationResult(
            checks=[
                CheckResult(code="PRICE_CHANGE_EXCESSIVE", passed=False, severity=Severity.WARNING, message="big change", affected_rows=[0]),
                CheckResult(code="PRICE_CHANGE_EXCESSIVE", passed=False, severity=Severity.WARNING, message="big change", affected_rows=[1]),
                CheckResult(code="PRICE_CHANGE_EXCESSIVE", passed=False, severity=Severity.WARNING, message="big change", affected_rows=[2]),
                CheckResult(code="OHLC_CONSISTENCY", passed=True, severity=Severity.INFO, message="OK"),
            ],
            passed=True,
            critical_count=0,
            warning_count=3,
            info_count=0,
        )
        df = _make_df(10)
        scorer = DataQualityScorer()
        result = scorer.compute(
            df,
            market="crypto_spot",
            interval="1d",
            expected_rows=10,
            latest_expected=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        assert result.anomaly_free == pytest.approx(0.7)

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_stale_data_reduces_timeliness(self, mock_validate):
        """Test 5: Data 2 hours stale vs 1-hour expected -> timeliness < 1.0."""
        mock_validate.return_value = _all_pass_result()
        # Data ends at hour 22, but expected at hour 24 (2 hours stale)
        df = _make_df(10)
        scorer = DataQualityScorer()
        # latest data is 2026-01-10 00:00 UTC, expected is 2026-01-10 02:00 UTC
        result = scorer.compute(
            df,
            market="crypto_spot",
            interval="1d",
            expected_rows=10,
            latest_expected=datetime(2026, 1, 10, 2, 0, tzinfo=timezone.utc),
        )
        # 2 hours stale -> timeliness = 1.0 - 2/24 = ~0.9167
        assert result.timeliness < 1.0
        assert result.timeliness == pytest.approx(1.0 - 2.0 / 24.0, abs=0.01)

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_composite_weights_sum_correctly(self, mock_validate):
        """Test 6: Composite = 0.30*comp + 0.25*cons + 0.25*anom + 0.20*time."""
        # Set up: completeness=0.5, consistency=0.8, anomaly_free=0.7, timeliness=0.9
        mock_validate.side_effect = [
            # First call (consistency)
            ValidationResult(
                checks=[
                    CheckResult(code="C", passed=False, severity=Severity.CRITICAL, message="", affected_rows=[0]),
                    CheckResult(code="C", passed=False, severity=Severity.CRITICAL, message="", affected_rows=[1]),
                ],
                passed=False, critical_count=2, warning_count=0, info_count=0,
            ),
            # Second call (anomaly_free)
            ValidationResult(
                checks=[
                    CheckResult(code="W", passed=False, severity=Severity.WARNING, message="", affected_rows=[0]),
                    CheckResult(code="W", passed=False, severity=Severity.WARNING, message="", affected_rows=[1]),
                    CheckResult(code="W", passed=False, severity=Severity.WARNING, message="", affected_rows=[2]),
                ],
                passed=True, critical_count=0, warning_count=3, info_count=0,
            ),
        ]
        df = _make_df(10)
        scorer = DataQualityScorer()
        # 5 of 10 expected -> completeness=0.5
        # 2 CRITICAL / 10 rows -> consistency=0.8
        # 3 WARNING / 10 rows -> anomaly_free=0.7
        # stale by ~2.4 hours -> timeliness ~0.9
        result = scorer.compute(
            df,
            market="crypto_spot",
            interval="1d",
            expected_rows=20,  # 10/20 = 0.5 completeness
            latest_expected=datetime(2026, 1, 10, 2, 24, tzinfo=timezone.utc),  # ~2.4h stale
        )
        expected_composite = (
            0.30 * result.completeness
            + 0.25 * result.consistency
            + 0.25 * result.anomaly_free
            + 0.20 * result.timeliness
        )
        assert result.composite == pytest.approx(expected_composite, abs=0.01)

    @patch("poseidon.data.quality_scorer.validate_ohlcv")
    def test_crypto_vs_tw_market_calendar_expected_rows(self, mock_validate):
        """Test 7: Crypto uses 365 calendar days; TW uses business days.

        This test validates the scorer accepts different expected_rows
        (the Celery task computes expected_rows using MARKET_CALENDARS).
        """
        mock_validate.return_value = _all_pass_result()
        df = _make_df(200)
        scorer = DataQualityScorer()

        # Crypto: 365 expected in a year -> 200/365 completeness
        result_crypto = scorer.compute(
            df, market="crypto_spot", interval="1d",
            expected_rows=365,
            latest_expected=datetime(2026, 7, 19, tzinfo=timezone.utc),
        )
        assert result_crypto.completeness == pytest.approx(200 / 365, abs=0.01)

        # TW stock: ~252 business days expected -> 200/252 completeness
        result_tw = scorer.compute(
            df, market="tw_stock", interval="1d",
            expected_rows=252,
            latest_expected=datetime(2026, 7, 19, tzinfo=timezone.utc),
        )
        assert result_tw.completeness == pytest.approx(200 / 252, abs=0.01)

        # Crypto completeness should be lower than TW (same 200 rows, more expected)
        assert result_crypto.completeness < result_tw.completeness
