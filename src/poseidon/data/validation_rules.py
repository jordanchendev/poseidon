"""OHLCV validation rules with market-specific thresholds (DVAL-01..04).

Each rule implements BaseValidationRule.check(df, market) -> CheckResult.
Rules are composed by validate_ohlcv() in validation.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd


# ---------------------------------------------------------------------------
# Severity levels (D-03)
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Validation severity: CRITICAL blocks upsert, WARNING allows it."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single validation rule check."""

    code: str
    passed: bool
    severity: Severity
    message: str
    affected_rows: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market-specific thresholds (D-04)
# ---------------------------------------------------------------------------

MARKET_THRESHOLDS: dict[str, dict[str, float]] = {
    "tw_stock": {"max_daily_change_pct": 10.0},
    "tw_futures": {"max_daily_change_pct": 10.0},
    "us_stock": {"max_daily_change_pct": 25.0},
    "crypto_spot": {"max_daily_change_pct": 50.0},
}


# ---------------------------------------------------------------------------
# Base rule ABC
# ---------------------------------------------------------------------------

class BaseValidationRule(ABC):
    """Abstract base class for OHLCV validation rules."""

    @abstractmethod
    def check(self, df: pd.DataFrame, market: str) -> CheckResult:
        """Run the validation check on *df* for the given *market*.

        Returns a CheckResult with pass/fail, severity, and affected rows.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete rules (D-05)
# ---------------------------------------------------------------------------

class OHLCConsistencyRule(BaseValidationRule):
    """Check OHLC bar consistency: high >= low, high >= max(open,close),
    low <= min(open,close), volume >= 0.  Severity: CRITICAL."""

    def check(self, df: pd.DataFrame, market: str) -> CheckResult:
        bad = (
            (df["high"] < df["low"])
            | (df["high"] < df["close"])
            | (df["high"] < df["open"])
            | (df["low"] > df["close"])
            | (df["low"] > df["open"])
            | (df["volume"] < 0)
        )
        affected = df.index[bad].tolist()
        if affected:
            return CheckResult(
                code="OHLC_CONSISTENCY",
                passed=False,
                severity=Severity.CRITICAL,
                message=f"OHLC consistency violated on {len(affected)} row(s)",
                affected_rows=affected,
            )
        return CheckResult(
            code="OHLC_CONSISTENCY",
            passed=True,
            severity=Severity.INFO,
            message="OHLC consistency OK",
        )


class DuplicateTimestampRule(BaseValidationRule):
    """Check for duplicate timestamps. Severity: CRITICAL."""

    def check(self, df: pd.DataFrame, market: str) -> CheckResult:
        dups = df["time"].duplicated(keep=False)
        affected = df.index[dups].tolist()
        if affected:
            return CheckResult(
                code="DUPLICATE_TIMESTAMP",
                passed=False,
                severity=Severity.CRITICAL,
                message=f"Duplicate timestamps found on {len(affected)} row(s)",
                affected_rows=affected,
            )
        return CheckResult(
            code="DUPLICATE_TIMESTAMP",
            passed=True,
            severity=Severity.INFO,
            message="No duplicate timestamps",
        )


class FutureTimestampRule(BaseValidationRule):
    """Check for timestamps in the future. Severity: CRITICAL."""

    def check(self, df: pd.DataFrame, market: str) -> CheckResult:
        now = datetime.now(timezone.utc)
        # Ensure comparison works: convert to pandas Timestamps if needed
        times = pd.to_datetime(df["time"], utc=True)
        future = times > now
        affected = df.index[future].tolist()
        if affected:
            return CheckResult(
                code="FUTURE_TIMESTAMP",
                passed=False,
                severity=Severity.CRITICAL,
                message=f"Future timestamps found on {len(affected)} row(s)",
                affected_rows=affected,
            )
        return CheckResult(
            code="FUTURE_TIMESTAMP",
            passed=True,
            severity=Severity.INFO,
            message="No future timestamps",
        )


class PriceChangeRule(BaseValidationRule):
    """Check for excessive daily price changes per market threshold (D-04).

    Severity: WARNING (not CRITICAL) to avoid blocking legitimate TW limit-up data.
    """

    def check(self, df: pd.DataFrame, market: str) -> CheckResult:
        if len(df) < 2:
            return CheckResult(
                code="PRICE_CHANGE_EXCESSIVE",
                passed=True,
                severity=Severity.INFO,
                message="Not enough rows to compute price change",
            )

        threshold = MARKET_THRESHOLDS.get(market, {}).get("max_daily_change_pct", 50.0)
        pct_change = df["close"].pct_change().abs() * 100
        excessive = pct_change > threshold
        # First row is NaN from pct_change — never flag it
        excessive = excessive.fillna(False)
        affected = df.index[excessive].tolist()

        if affected:
            return CheckResult(
                code="PRICE_CHANGE_EXCESSIVE",
                passed=False,
                severity=Severity.WARNING,
                message=f"Price change exceeds {threshold}% on {len(affected)} row(s)",
                affected_rows=affected,
            )
        return CheckResult(
            code="PRICE_CHANGE_EXCESSIVE",
            passed=True,
            severity=Severity.INFO,
            message="Price changes within limits",
        )


# ---------------------------------------------------------------------------
# Rule factory
# ---------------------------------------------------------------------------

def get_rules_for_market(market: str) -> list[BaseValidationRule]:
    """Return all validation rules applicable to *market*.

    All rules apply to all markets. Market-specific behavior is parameterized
    inside PriceChangeRule via MARKET_THRESHOLDS lookup.
    """
    return [
        OHLCConsistencyRule(),
        DuplicateTimestampRule(),
        FutureTimestampRule(),
        PriceChangeRule(),
    ]
