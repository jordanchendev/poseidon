"""Unit tests for OHLCV data validation pipeline (DVAL-01..04)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from poseidon.data.validation import ValidationResult, validate_ohlcv
from poseidon.data.validation_rules import (
    MARKET_THRESHOLDS,
    BaseValidationRule,
    CheckResult,
    DuplicateTimestampRule,
    FutureTimestampRule,
    OHLCConsistencyRule,
    PriceChangeRule,
    Severity,
    get_rules_for_market,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a test DataFrame with standard OHLCV columns."""
    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        now = datetime.now(timezone.utc)
        df["time"] = [now - timedelta(days=len(rows) - i) for i in range(len(rows))]
    return df


def _clean_df(n: int = 5) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame with no anomalies."""
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        rows.append({
            "time": now - timedelta(days=n - i),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "volume": 1000.0 + i * 10,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ValidationResult on clean data
# ---------------------------------------------------------------------------

class TestValidationResultClean:
    def test_validation_result_clean(self):
        """validate_ohlcv() on clean data returns passed=True, critical_count=0."""
        df = _clean_df()
        result = validate_ohlcv(df, "crypto_spot")
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.critical_count == 0


# ---------------------------------------------------------------------------
# OHLCConsistencyRule
# ---------------------------------------------------------------------------

class TestOHLCConsistency:
    def test_ohlc_consistency_high_lt_low(self):
        """high < low -> CRITICAL."""
        df = _make_df([
            {"open": 100, "high": 90, "low": 95, "close": 98, "volume": 100},
        ])
        rule = OHLCConsistencyRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.code == "OHLC_CONSISTENCY"

    def test_ohlc_consistency_negative_volume(self):
        """volume < 0 -> CRITICAL."""
        df = _make_df([
            {"open": 100, "high": 105, "low": 95, "close": 102, "volume": -10},
        ])
        rule = OHLCConsistencyRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.code == "OHLC_CONSISTENCY"

    def test_ohlc_consistency_high_lt_close(self):
        """high < close -> CRITICAL."""
        df = _make_df([
            {"open": 100, "high": 101, "low": 95, "close": 105, "volume": 100},
        ])
        rule = OHLCConsistencyRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL

    def test_ohlc_consistency_low_gt_open(self):
        """low > open -> CRITICAL."""
        df = _make_df([
            {"open": 100, "high": 110, "low": 102, "close": 105, "volume": 100},
        ])
        rule = OHLCConsistencyRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# DuplicateTimestampRule
# ---------------------------------------------------------------------------

class TestDuplicateTimestamp:
    def test_duplicate_timestamp(self):
        """Two identical timestamps -> CRITICAL."""
        now = datetime.now(timezone.utc)
        df = _make_df([
            {"time": now, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 100},
            {"time": now, "open": 101, "high": 106, "low": 96, "close": 103, "volume": 200},
        ])
        rule = DuplicateTimestampRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.code == "DUPLICATE_TIMESTAMP"


# ---------------------------------------------------------------------------
# FutureTimestampRule
# ---------------------------------------------------------------------------

class TestFutureTimestamp:
    def test_future_timestamp(self):
        """Timestamp > now -> CRITICAL."""
        future = datetime.now(timezone.utc) + timedelta(days=5)
        df = _make_df([
            {"time": future, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 100},
        ])
        rule = FutureTimestampRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert result.code == "FUTURE_TIMESTAMP"


# ---------------------------------------------------------------------------
# PriceChangeRule -- market-specific thresholds
# ---------------------------------------------------------------------------

class TestPriceChange:
    def test_tw_price_change_within_limit(self):
        """TW stock +9% daily change -> passes."""
        now = datetime.now(timezone.utc)
        df = _make_df([
            {"time": now - timedelta(days=1), "open": 100, "high": 105, "low": 95, "close": 100, "volume": 100},
            {"time": now, "open": 109, "high": 112, "low": 107, "close": 109, "volume": 200},
        ])
        rule = PriceChangeRule()
        result = rule.check(df, "tw_stock")
        assert result.passed is True

    def test_tw_price_change_at_limit(self):
        """TW stock +10% (limit-up) -> WARNING not CRITICAL per D-04."""
        now = datetime.now(timezone.utc)
        df = _make_df([
            {"time": now - timedelta(days=1), "open": 100, "high": 105, "low": 95, "close": 100, "volume": 100},
            {"time": now, "open": 110, "high": 115, "low": 108, "close": 110, "volume": 200},
        ])
        rule = PriceChangeRule()
        result = rule.check(df, "tw_stock")
        assert result.passed is False
        assert result.severity == Severity.WARNING
        assert result.code == "PRICE_CHANGE_EXCESSIVE"

    def test_us_price_change_excessive(self):
        """US stock +200% daily change -> WARNING."""
        now = datetime.now(timezone.utc)
        df = _make_df([
            {"time": now - timedelta(days=1), "open": 100, "high": 105, "low": 95, "close": 100, "volume": 100},
            {"time": now, "open": 300, "high": 310, "low": 290, "close": 300, "volume": 200},
        ])
        rule = PriceChangeRule()
        result = rule.check(df, "us_stock")
        assert result.passed is False
        assert result.severity == Severity.WARNING
        assert result.code == "PRICE_CHANGE_EXCESSIVE"

    def test_crypto_price_change_within_limit(self):
        """Crypto +45% daily change -> passes (threshold is 50%)."""
        now = datetime.now(timezone.utc)
        df = _make_df([
            {"time": now - timedelta(days=1), "open": 100, "high": 105, "low": 95, "close": 100, "volume": 100},
            {"time": now, "open": 145, "high": 150, "low": 140, "close": 145, "volume": 200},
        ])
        rule = PriceChangeRule()
        result = rule.check(df, "crypto_spot")
        assert result.passed is True


# ---------------------------------------------------------------------------
# MARKET_THRESHOLDS
# ---------------------------------------------------------------------------

class TestMarketThresholds:
    def test_market_thresholds(self):
        """MARKET_THRESHOLDS has correct max_daily_change_pct per market."""
        assert MARKET_THRESHOLDS["tw_stock"]["max_daily_change_pct"] == 10.0
        assert MARKET_THRESHOLDS["tw_futures"]["max_daily_change_pct"] == 10.0
        assert MARKET_THRESHOLDS["us_stock"]["max_daily_change_pct"] == 25.0
        assert MARKET_THRESHOLDS["crypto_spot"]["max_daily_change_pct"] == 50.0


# ---------------------------------------------------------------------------
# Integration: validate_ohlcv critical vs warning behavior
# ---------------------------------------------------------------------------

class TestValidateOHLCVIntegration:
    def test_critical_blocks_upsert(self):
        """validate_ohlcv() with CRITICAL failure -> has_critical=True, passed=False."""
        df = _make_df([
            {"open": 100, "high": 90, "low": 95, "close": 98, "volume": 100},
        ])
        result = validate_ohlcv(df, "crypto_spot")
        assert result.has_critical is True
        assert result.passed is False

    def test_warning_allows_upsert(self):
        """validate_ohlcv() with only WARNING -> has_critical=False, passed=True."""
        now = datetime.now(timezone.utc)
        # Build a DF where price change exceeds threshold but OHLC is consistent
        df = _make_df([
            {"time": now - timedelta(days=1), "open": 100, "high": 105, "low": 95, "close": 100, "volume": 100},
            {"time": now, "open": 160, "high": 165, "low": 155, "close": 160, "volume": 200},
        ])
        result = validate_ohlcv(df, "tw_stock")
        # 60% change on TW stock (threshold 10%) -> WARNING
        assert result.has_critical is False
        assert result.passed is True
        assert result.warning_count > 0


# ---------------------------------------------------------------------------
# get_rules_for_market
# ---------------------------------------------------------------------------

class TestGetRulesForMarket:
    def test_returns_all_rules(self):
        """get_rules_for_market returns instances of all 4 rule classes."""
        rules = get_rules_for_market("crypto_spot")
        assert len(rules) == 4
        rule_types = {type(r) for r in rules}
        assert OHLCConsistencyRule in rule_types
        assert DuplicateTimestampRule in rule_types
        assert FutureTimestampRule in rule_types
        assert PriceChangeRule in rule_types
        for r in rules:
            assert isinstance(r, BaseValidationRule)
